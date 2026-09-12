"""运行 S09 两个商业场站的十三方法最小外部验证。"""
from __future__ import annotations

import gc
import json
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


S09_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = Path(__file__).resolve().parents[2]
AUTHORITATIVE_CODE = REVISION_ROOT / "_权威代码" / "code"
sys.path.insert(0, str(AUTHORITATIVE_CODE))
sys.path.insert(0, str(S09_ROOT / "scripts"))

from clara_event_contract import load_frozen_contracts  # noqa: E402
from clara_errf import ErrfTheta  # noqa: E402
import s09_minimum_external_core as core  # noqa: E402
from s09_external_summary import summarize_method_complete  # noqa: E402
from s09_linucb_batch_accelerator import run_linucb_batch_equivalent  # noqa: E402


CONFIG_PATH = S09_ROOT / "configs" / "s09_minimum_external_evaluation_v1.json"
GATE_PATH = S09_ROOT / "configs" / "s09_minimum_external_full_run_gate_v1.json"
EXTERNAL_PROTOCOL_PATH = S09_ROOT / "configs" / "s09_external_validation_v1.json"
LOCAL_SELECTION_ROOT = S09_ROOT / "results_raw" / "local_conformal_selection_v1"
LOCAL_SELECTION_QA = S09_ROOT / "qa" / "local_conformal_selection_v1" / "qa_summary.json"
LINUCB_EQUIVALENCE = S09_ROOT / "results_raw" / "linucb_batch_equivalence_v1" / "manifest.json"
LINUCB_EQUIVALENCE_QA = (
    S09_ROOT / "qa" / "linucb_batch_equivalence_v1" / "qa_summary.json"
)
CORE_PATH = S09_ROOT / "scripts" / "s09_minimum_external_core.py"
ACCELERATOR_PATH = S09_ROOT / "scripts" / "s09_linucb_batch_accelerator.py"
SUMMARY_PATH = S09_ROOT / "scripts" / "s09_external_summary.py"
RUNNER_PATH = Path(__file__).resolve()
V4_CONFIG_PATH = (
    REVISION_ROOT / "测试CLARA" / "configs" / "test_clara_four_versions_v1.json"
)
RESULT_ROOT = S09_ROOT / "results_raw" / "minimum_external_evaluation_v1"
UNIT_ROOT = RESULT_ROOT / "units"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def evaluation_unit_id(farm: str, seed: int) -> str:
    return f"{farm}-S{int(seed)}"


def check_entry_conditions(config: dict[str, Any]) -> dict[str, str]:
    gate = core.load_json(GATE_PATH)
    local_root = core.load_json(LOCAL_SELECTION_ROOT / "root_manifest.json")
    local_qa = core.load_json(LOCAL_SELECTION_QA)
    equivalence = core.load_json(LINUCB_EQUIVALENCE)
    equivalence_qa = core.load_json(LINUCB_EQUIVALENCE_QA)
    if gate.get("status") != "AUTHORIZED_FOR_FULL_TECHNICAL_EXECUTION":
        raise RuntimeError("S09 正式评估安全门未授权")
    if gate["minimum_protocol"]["sha256"] != core.sha256_file(CONFIG_PATH):
        raise RuntimeError("S09 正式评估协议身份与安全门失配")
    if local_root.get("status") != "COMPLETE_PENDING_INDEPENDENT_QA":
        raise RuntimeError("S09 本地共形选择尚未封存")
    if local_qa.get("status") != "PASS" or int(local_qa.get("failed_count", -1)) != 0:
        raise RuntimeError("S09 本地共形选择独立 QA 未通过")
    if equivalence.get("status") != "PASS":
        raise RuntimeError("S09 LinUCB 批量加速器等价回归未通过")
    if (
        equivalence_qa.get("status") != "PASS"
        or int(equivalence_qa.get("failed_count", -1)) != 0
    ):
        raise RuntimeError("S09 LinUCB 批量加速器独立 QA 未通过")
    if bool(config.get("performance_read_allowed")):
        raise RuntimeError("S09 正式评估期间禁止开放汇总性能读取")
    return {
        "config": core.sha256_file(CONFIG_PATH),
        "gate": core.sha256_file(GATE_PATH),
        "external_protocol": core.sha256_file(EXTERNAL_PROTOCOL_PATH),
        "local_selection_root": core.sha256_file(
            LOCAL_SELECTION_ROOT / "root_manifest.json"
        ),
        "local_selection_qa": core.sha256_file(LOCAL_SELECTION_QA),
        "linucb_equivalence": core.sha256_file(LINUCB_EQUIVALENCE),
        "linucb_equivalence_qa": core.sha256_file(LINUCB_EQUIVALENCE_QA),
        "core": core.sha256_file(CORE_PATH),
        "accelerator": core.sha256_file(ACCELERATOR_PATH),
        "summary": core.sha256_file(SUMMARY_PATH),
        "runner": core.sha256_file(RUNNER_PATH),
        "v4_config": core.sha256_file(V4_CONFIG_PATH),
    }


def completed_unit(path: Path, identities: dict[str, str]) -> bool:
    manifest_path = path / "manifest.json"
    required = (
        path / "cell_metrics.parquet",
        path / "action_counts.parquet",
        path / "paired_block_statistics.parquet",
        path / "method_audits.json",
    )
    if not manifest_path.exists() or any(not item.exists() for item in required):
        return False
    manifest = core.load_json(manifest_path)
    if manifest.get("status") != "PASS":
        return False
    if manifest.get("identities") != identities:
        return False
    for key, filename in (
        ("cell_metrics_sha256", "cell_metrics.parquet"),
        ("action_counts_sha256", "action_counts.parquet"),
        ("paired_block_statistics_sha256", "paired_block_statistics.parquet"),
        ("method_audits_sha256", "method_audits.json"),
    ):
        if manifest.get(key) != core.sha256_file(path / filename):
            return False
    return True


def interval_arrays_for_conformal(
    *,
    facts: pd.DataFrame,
    base_predictions: pd.DataFrame,
    contracts: Any,
    theta: ErrfTheta,
    dataset_id: str,
    farm: str,
    predictor: str,
    seed: int,
    family: str,
    configuration: dict[str, Any],
    target_coverages: Sequence[float],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    names = ("lower", "upper", "width", "covered", "reserve", "miss", "errf")
    arrays: dict[str, np.ndarray] = {
        name: np.empty(
            len(facts),
            dtype=bool if name == "covered" else np.float64,
        )
        for name in names
    }
    audit_rows: list[dict[str, Any]] = []
    assigned = np.zeros(len(facts), dtype=bool)
    for target_coverage in target_coverages:
        mask = facts["target_coverage"].astype(float).eq(float(target_coverage)).to_numpy()
        positions = np.flatnonzero(mask)
        coverage_facts = facts.iloc[positions].copy().reset_index(drop=True)
        decisions, audit = core.run_conformal_method(
            base_predictions=base_predictions,
            contracts=contracts,
            dataset_id=dataset_id,
            farm=farm,
            predictor=predictor,
            seed=int(seed),
            target_coverage=float(target_coverage),
            family=family,
            configuration=configuration,
        )
        current = core.endpoint_metrics_for_direct_interval(
            coverage_facts,
            decisions,
            theta=theta,
        )
        for name in names:
            arrays[name][positions] = np.asarray(current[name])
        assigned[positions] = True
        audit_rows.append(
            {
                "target_coverage": float(target_coverage),
                "test_event_count": int(audit["test_event_count"]),
                "feedback_count": int(audit["feedback_count"]),
                "pending_feedback_count": int(audit["pending_feedback_count"]),
                "future_feedback_violation_count": int(
                    audit["future_feedback_violation_count"]
                ),
                "within_issue_feedback_use_count": int(
                    audit["within_issue_feedback_use_count"]
                ),
                "invalid_interval_count": int(audit["invalid_interval_count"]),
                "out_of_bounds_count": int(audit["out_of_bounds_count"]),
                "final_feedback_drain": bool(audit["drain_final_feedback"]),
            }
        )
    if not assigned.all():
        raise RuntimeError("共形方法未覆盖全部最终事件")
    if any(
        row["future_feedback_violation_count"] != 0
        or row["within_issue_feedback_use_count"] != 0
        or row["invalid_interval_count"] != 0
        or row["out_of_bounds_count"] != 0
        or row["final_feedback_drain"]
        for row in audit_rows
    ):
        raise RuntimeError("共形方法完整运行的因果或区间审计失败")
    return arrays, {
        "family": family,
        "configuration": configuration,
        "stream_count": len(audit_rows),
        "audits": audit_rows,
    }


def append_summaries(
    *,
    facts: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    method: str,
    selected_actions: Sequence[str] | None,
    metric_parts: list[pd.DataFrame],
    action_parts: list[pd.DataFrame],
    block_parts: list[pd.DataFrame],
) -> dict[str, int]:
    metrics, actions, blocks = summarize_method_complete(
        facts,
        arrays,
        method=method,
        selected_actions=selected_actions,
    )
    metric_parts.append(metrics)
    action_parts.append(actions)
    block_parts.append(blocks)
    return {
        "cell_metric_rows": len(metrics),
        "action_count_rows": len(actions),
        "paired_block_rows": len(blocks),
    }


def run_unit(task: tuple[str, int, dict[str, str]]) -> dict[str, Any]:
    farm, seed, identities = task
    identifier = evaluation_unit_id(farm, seed)
    output = UNIT_ROOT / identifier
    if completed_unit(output, identities):
        return {"unit_id": identifier, "status": "REUSED"}
    if output.exists():
        raise RuntimeError(f"发现未封存的 S09 正式评估单元，拒绝覆盖: {output}")
    output.mkdir(parents=True, exist_ok=False)
    started_at = utc_now()
    try:
        config = core.load_json(CONFIG_PATH)
        external_protocol = core.load_json(EXTERNAL_PROTOCOL_PATH)
        selected_local = core.load_json(
            LOCAL_SELECTION_ROOT / "selected_configurations.json"
        )[farm]
        contracts = load_frozen_contracts()
        theta = ErrfTheta()
        v4_protocol = core.load_json(V4_CONFIG_PATH)
        predictors = [str(value) for value in config["scope"]["predictors"]]
        horizons = [int(value) for value in config["scope"]["horizon_steps"]]
        target_coverages = [
            float(value) for value in config["scope"]["target_coverages"]
        ]
        bundle_keys = [(predictor, horizon) for predictor in predictors for horizon in horizons]

        adaptation_frames: dict[tuple[str, int], pd.DataFrame] = {}
        final_frames: dict[tuple[str, int], pd.DataFrame] = {}
        bundle_audits: list[dict[str, Any]] = []
        for predictor, horizon in bundle_keys:
            adaptation_raw, adaptation_audit = core.load_fact_bundle(
                root=core.ADAPTATION_ROOT,
                farm=farm,
                predictor=predictor,
                horizon=horizon,
                seed=int(seed),
                contracts=contracts,
                theta=theta,
            )
            thresholds = core.fit_local_width_thresholds(adaptation_raw, farm)
            adaptation = core.attach_local_width_states(adaptation_raw, thresholds)
            final_raw, final_audit = core.load_fact_bundle(
                root=core.FULL_REBUILD_ROOT,
                farm=farm,
                predictor=predictor,
                horizon=horizon,
                seed=int(seed),
                contracts=contracts,
                theta=theta,
            )
            final = core.attach_local_width_states(final_raw, thresholds)
            adaptation_frames[(predictor, horizon)] = adaptation
            final_frames[(predictor, horizon)] = final
            bundle_audits.append(
                {
                    "predictor": predictor,
                    "horizon_steps": int(horizon),
                    "adaptation_fact_content_sha256": adaptation_audit[
                        "fact_content_sha256"
                    ],
                    "final_fact_content_sha256": final_audit["fact_content_sha256"],
                    "adaptation_event_count": len(adaptation),
                    "final_event_count": len(final),
                    "adaptation_future_violation_count": int(
                        adaptation_audit["future_information_violation_count"]
                    ),
                    "final_future_violation_count": int(
                        final_audit["future_information_violation_count"]
                    ),
                }
            )
        adaptation_all = pd.concat(
            [adaptation_frames[key] for key in bundle_keys],
            ignore_index=True,
            sort=False,
        )
        final_all = pd.concat(
            [final_frames[key] for key in bundle_keys],
            ignore_index=True,
            sort=False,
        )
        if adaptation_all["event_id"].duplicated().any() or final_all[
            "event_id"
        ].duplicated().any():
            raise RuntimeError(f"S09 正式评估跨流事件标识重复: {identifier}")

        local_clara_decisions, local_clara_audit = core.fit_local_clara(
            adaptation_all,
            contracts=contracts,
            protocol=config["local_clara_estimator"],
            v4_protocol=v4_protocol,
        )
        local_cart_model, local_cart_statistics = core.fit_cart_local(
            adaptation_all,
            farm=farm,
            contracts=contracts,
            config=config["cart"]["local_configuration"],
        )
        fold_thresholds = pd.read_parquet(core.FOLD_WIDTH_THRESHOLDS)
        source_decisions, source_clara_audit = core.load_source_v4_decisions(int(seed))
        source_cart_models, source_cart_audit = core.load_source_cart_models(contracts)

        linucb_config = config["linucb"]["configuration"]
        linucb_direct_all, linucb_direct_audit = run_linucb_batch_equivalent(
            final_facts=final_all,
            contracts=contracts,
            exploration_alpha=float(linucb_config["exploration_alpha"]),
            l2_regularization=float(linucb_config["l2_regularization"]),
        )
        linucb_local_all, linucb_local_audit = run_linucb_batch_equivalent(
            final_facts=final_all,
            adaptation_facts=adaptation_all,
            contracts=contracts,
            exploration_alpha=float(linucb_config["exploration_alpha"]),
            l2_regularization=float(linucb_config["l2_regularization"]),
        )

        slices: dict[tuple[str, int], slice] = {}
        cursor = 0
        for key in bundle_keys:
            count = len(final_frames[key])
            slices[key] = slice(cursor, cursor + count)
            cursor += count
        if cursor != len(final_all):
            raise RuntimeError("S09 正式评估 LinUCB 分流切片守恒失败")

        metric_parts: list[pd.DataFrame] = []
        action_parts: list[pd.DataFrame] = []
        block_parts: list[pd.DataFrame] = []
        bundle_method_audits: list[dict[str, Any]] = []
        direct_conformal = config["tuned_single_conformal"]
        dataset_id = str(external_protocol["data"]["farms"][farm]["dataset_id"])
        for bundle_number, (predictor, horizon) in enumerate(bundle_keys, start=1):
            facts = final_frames[(predictor, horizon)].reset_index(drop=True)
            current_slice = slices[(predictor, horizon)]
            method_actions: dict[str, np.ndarray] = {}
            clara_direct, clara_direct_audit = core.predict_clara_direct(
                facts,
                seed=int(seed),
                thresholds=fold_thresholds,
                source_decisions=source_decisions,
            )
            method_actions["CLARA_V4_Direct"] = clara_direct
            method_actions["CLARA_V4_Local"] = core.predict_local_clara(
                facts, local_clara_decisions
            )
            cart_direct, cart_direct_audit = core.predict_cart_direct(
                facts,
                seed=int(seed),
                thresholds=fold_thresholds,
                models=source_cart_models,
            )
            method_actions["CART_Direct"] = cart_direct
            cart_local, cart_local_scores = core.predict_cart_local(
                facts, local_cart_model
            )
            method_actions["CART_Local"] = cart_local
            method_actions["LinUCB_Direct"] = linucb_direct_all[current_slice]
            method_actions["LinUCB_Local"] = linucb_local_all[current_slice]
            for action in core.ACTIONS:
                method_actions[action] = np.full(len(facts), action, dtype=object)

            method_row_counts: dict[str, Any] = {}
            for method, actions in method_actions.items():
                arrays = core.endpoint_metrics_for_actions(facts, actions, theta=theta)
                method_row_counts[method] = append_summaries(
                    facts=facts,
                    arrays=arrays,
                    method=method,
                    selected_actions=actions,
                    metric_parts=metric_parts,
                    action_parts=action_parts,
                    block_parts=block_parts,
                )
            ensemble_arrays = core.endpoint_metrics_for_ensemble(facts, theta=theta)
            method_row_counts["EqualEndpointEnsemble"] = append_summaries(
                facts=facts,
                arrays=ensemble_arrays,
                method="EqualEndpointEnsemble",
                selected_actions=None,
                metric_parts=metric_parts,
                action_parts=action_parts,
                block_parts=block_parts,
            )

            paths = core.bundle_paths(
                root=core.FULL_REBUILD_ROOT,
                farm=farm,
                predictor=predictor,
                horizon=horizon,
                seed=int(seed),
            )
            base_predictions = pd.read_parquet(paths["base_predictions"])
            conformal_direct_arrays, conformal_direct_audit = (
                interval_arrays_for_conformal(
                    facts=facts,
                    base_predictions=base_predictions,
                    contracts=contracts,
                    theta=theta,
                    dataset_id=dataset_id,
                    farm=farm,
                    predictor=predictor,
                    seed=int(seed),
                    family=str(direct_conformal["direct_family"]),
                    configuration=dict(direct_conformal["direct_configuration"]),
                    target_coverages=target_coverages,
                )
            )
            method_row_counts["TunedSingleConformal_Direct"] = append_summaries(
                facts=facts,
                arrays=conformal_direct_arrays,
                method="TunedSingleConformal_Direct",
                selected_actions=None,
                metric_parts=metric_parts,
                action_parts=action_parts,
                block_parts=block_parts,
            )
            conformal_local_arrays, conformal_local_audit = interval_arrays_for_conformal(
                facts=facts,
                base_predictions=base_predictions,
                contracts=contracts,
                theta=theta,
                dataset_id=dataset_id,
                farm=farm,
                predictor=predictor,
                seed=int(seed),
                family=str(selected_local["family"]),
                configuration=dict(selected_local["configuration"]),
                target_coverages=target_coverages,
            )
            method_row_counts["TunedSingleConformal_Local"] = append_summaries(
                facts=facts,
                arrays=conformal_local_arrays,
                method="TunedSingleConformal_Local",
                selected_actions=None,
                metric_parts=metric_parts,
                action_parts=action_parts,
                block_parts=block_parts,
            )
            if set(method_row_counts) != set(config["scope"]["methods"]):
                raise RuntimeError(f"S09 正式评估十三方法集合失配: {identifier}")
            bundle_method_audits.append(
                {
                    "predictor": predictor,
                    "horizon_steps": int(horizon),
                    "final_event_count": len(facts),
                    "method_row_counts": method_row_counts,
                    "clara_direct_source_count": len(clara_direct_audit),
                    "cart_direct_source_count": len(cart_direct_audit),
                    "cart_local_score_finite_count": int(
                        np.isfinite(cart_local_scores).sum()
                    ),
                    "conformal_direct": conformal_direct_audit,
                    "conformal_local": conformal_local_audit,
                }
            )
            print(
                json.dumps(
                    {
                        "unit_id": identifier,
                        "bundle_progress": f"{bundle_number}/{len(bundle_keys)}",
                        "predictor": predictor,
                        "horizon_steps": horizon,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            del base_predictions, conformal_direct_arrays, conformal_local_arrays
            gc.collect()

        cell_metrics = pd.concat(metric_parts, ignore_index=True)
        action_counts = pd.concat(action_parts, ignore_index=True)
        paired_blocks = pd.concat(block_parts, ignore_index=True)
        expected_cell_rows = len(bundle_keys) * 13 * len(target_coverages) * 3
        if len(cell_metrics) != expected_cell_rows:
            raise RuntimeError(
                f"S09 正式评估单元指标行数失配: {identifier}: "
                f"{len(cell_metrics)} != {expected_cell_rows}"
            )
        if cell_metrics.duplicated(
            [
                "zone_or_farm",
                "predictor",
                "seed",
                "horizon_steps",
                "target_coverage",
                "method",
                "regime",
            ]
        ).any():
            raise RuntimeError(f"S09 正式评估单元指标键重复: {identifier}")
        core.write_parquet_atomic(cell_metrics, output / "cell_metrics.parquet")
        core.write_parquet_atomic(action_counts, output / "action_counts.parquet")
        core.write_parquet_atomic(
            paired_blocks, output / "paired_block_statistics.parquet"
        )
        method_audits = {
            "schema": "S09_MINIMUM_EXTERNAL_METHOD_AUDITS_V1",
            "unit_id": identifier,
            "bundle_audits": bundle_audits,
            "local_clara": local_clara_audit,
            "local_cart_training_statistics_row_count": len(local_cart_statistics),
            "source_clara": source_clara_audit,
            "source_cart": source_cart_audit.to_dict(orient="records"),
            "linucb_direct": linucb_direct_audit,
            "linucb_local": linucb_local_audit,
            "selected_local_conformal": selected_local,
            "bundle_method_audits": bundle_method_audits,
            "event_level_method_table_written": False,
            "aggregate_performance_read_by_runner": False,
        }
        core.write_json_atomic(method_audits, output / "method_audits.json")
        manifest = {
            "schema": "S09_MINIMUM_EXTERNAL_EVALUATION_UNIT_V1",
            "status": "PASS",
            "unit_id": identifier,
            "farm": farm,
            "seed": int(seed),
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "identities": identities,
            "bundle_count": len(bundle_keys),
            "method_count": 13,
            "adaptation_event_count": len(adaptation_all),
            "final_event_count": len(final_all),
            "cell_metric_row_count": len(cell_metrics),
            "action_count_row_count": len(action_counts),
            "paired_block_row_count": len(paired_blocks),
            "cell_metrics_sha256": core.sha256_file(output / "cell_metrics.parquet"),
            "action_counts_sha256": core.sha256_file(output / "action_counts.parquet"),
            "paired_block_statistics_sha256": core.sha256_file(
                output / "paired_block_statistics.parquet"
            ),
            "method_audits_sha256": core.sha256_file(output / "method_audits.json"),
            "event_level_method_table_written": False,
            "aggregate_performance_read_by_runner": False,
        }
        core.write_json_atomic(manifest, output / "manifest.json")
        return {"unit_id": identifier, "status": "COMPLETED"}
    except Exception as exc:
        core.write_json_atomic(
            {
                "schema": "S09_MINIMUM_EXTERNAL_EVALUATION_EXCEPTION_V1",
                "status": "FAILED_PRESERVED",
                "unit_id": identifier,
                "failed_at_utc": utc_now(),
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "aggregate_performance_read_by_runner": False,
            },
            output / "exception.json",
        )
        raise


def main() -> int:
    config = core.load_json(CONFIG_PATH)
    identities = check_entry_conditions(config)
    if shutil.disk_usage(S09_ROOT.drive + "\\").free < 30 * 1024**3:
        raise RuntimeError("S09 正式评估磁盘安全门失败")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    UNIT_ROOT.mkdir(parents=True, exist_ok=True)
    if (RESULT_ROOT / "root_manifest.json").exists():
        raise RuntimeError("S09 正式评估根清单已存在，拒绝重复运行")
    tasks = [
        (str(farm), int(seed), identities)
        for farm in config["scope"]["farms"]
        for seed in config["scope"]["seeds"]
    ]
    completed = 0
    with ProcessPoolExecutor(max_workers=2) as executor:
        future_map = {executor.submit(run_unit, task): task for task in tasks}
        for future in as_completed(future_map):
            result = future.result()
            completed += 1
            core.write_json_atomic(
                {
                    "schema": "S09_MINIMUM_EXTERNAL_EVALUATION_CHECKPOINT_V1",
                    "status": "RUNNING",
                    "completed_unit_count": completed,
                    "expected_unit_count": len(tasks),
                    "last_completed_unit": result["unit_id"],
                    "updated_at_utc": utc_now(),
                    "aggregate_performance_read": False,
                },
                RESULT_ROOT / "checkpoint.json",
            )
            print(
                json.dumps(
                    {"progress": f"{completed}/{len(tasks)}", **result},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    metric_parts: list[pd.DataFrame] = []
    action_parts: list[pd.DataFrame] = []
    block_parts: list[pd.DataFrame] = []
    unit_records: list[dict[str, Any]] = []
    for farm, seed, _ in tasks:
        output = UNIT_ROOT / evaluation_unit_id(farm, seed)
        if not completed_unit(output, identities):
            raise RuntimeError(f"S09 正式评估单元封存失败: {output.name}")
        manifest = core.load_json(output / "manifest.json")
        metric_parts.append(pd.read_parquet(output / "cell_metrics.parquet"))
        action_parts.append(pd.read_parquet(output / "action_counts.parquet"))
        block_parts.append(pd.read_parquet(output / "paired_block_statistics.parquet"))
        unit_records.append(
            {
                "unit_id": output.name,
                "manifest_sha256": core.sha256_file(output / "manifest.json"),
                "cell_metrics_sha256": manifest["cell_metrics_sha256"],
                "action_counts_sha256": manifest["action_counts_sha256"],
                "paired_block_statistics_sha256": manifest[
                    "paired_block_statistics_sha256"
                ],
                "method_audits_sha256": manifest["method_audits_sha256"],
            }
        )
    cell_metrics = pd.concat(metric_parts, ignore_index=True)
    action_counts = pd.concat(action_parts, ignore_index=True)
    paired_blocks = pd.concat(block_parts, ignore_index=True)
    expected_cell_rows = 6 * 16 * 13 * 11 * 3
    if len(cell_metrics) != expected_cell_rows:
        raise RuntimeError("S09 正式评估根指标行数失配")
    core.write_parquet_atomic(cell_metrics, RESULT_ROOT / "cell_metrics.parquet")
    core.write_parquet_atomic(action_counts, RESULT_ROOT / "action_counts.parquet")
    core.write_parquet_atomic(
        paired_blocks, RESULT_ROOT / "paired_block_statistics.parquet"
    )
    root_manifest = {
        "schema": "S09_MINIMUM_EXTERNAL_EVALUATION_ROOT_MANIFEST_V1",
        "status": "COMPLETE_PENDING_INDEPENDENT_QA",
        "completed_at_utc": utc_now(),
        "identities": identities,
        "unit_count": len(tasks),
        "bundle_count": 96,
        "method_count": 13,
        "cell_metric_row_count": len(cell_metrics),
        "action_count_row_count": len(action_counts),
        "paired_block_row_count": len(paired_blocks),
        "cell_metrics_sha256": core.sha256_file(RESULT_ROOT / "cell_metrics.parquet"),
        "action_counts_sha256": core.sha256_file(RESULT_ROOT / "action_counts.parquet"),
        "paired_block_statistics_sha256": core.sha256_file(
            RESULT_ROOT / "paired_block_statistics.parquet"
        ),
        "units": unit_records,
        "event_level_method_table_written": False,
        "aggregate_performance_read_by_orchestrator": False,
    }
    core.write_json_atomic(root_manifest, RESULT_ROOT / "root_manifest.json")
    core.write_json_atomic(
        {
            "schema": "S09_MINIMUM_EXTERNAL_EVALUATION_CHECKPOINT_V1",
            "status": "COMPLETE_PENDING_INDEPENDENT_QA",
            "completed_unit_count": len(tasks),
            "expected_unit_count": len(tasks),
            "updated_at_utc": utc_now(),
            "aggregate_performance_read": False,
        },
        RESULT_ROOT / "checkpoint.json",
    )
    print(json.dumps(root_manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
