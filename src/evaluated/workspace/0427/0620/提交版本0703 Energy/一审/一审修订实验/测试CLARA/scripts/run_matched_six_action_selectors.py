"""运行六动作 CART 与六动作 LinUCB 的匹配外部评价。"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TEST_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = TEST_ROOT.parent
S09_ROOT = REVISION_ROOT / "09_商业场站外部验证"
AUTHORITATIVE_CODE = REVISION_ROOT / "_权威代码" / "code"
sys.path.insert(0, str(AUTHORITATIVE_CODE))
sys.path.insert(0, str(S09_ROOT / "scripts"))
sys.path.insert(0, str(TEST_ROOT / "scripts"))

from clara_event_contract import load_frozen_contracts  # noqa: E402
from clara_errf import ErrfTheta  # noqa: E402
from extended_selector_adapters import (  # noqa: E402
    fit_cart_local_actions,
    predict_cart_local_actions,
    run_linucb_local_actions,
)
import run_extended_action_external as extended  # noqa: E402
import run_s09_minimum_external_evaluation as s09_evaluation  # noqa: E402
import s09_minimum_external_core as core  # noqa: E402


CONFIG_PATH = (
    TEST_ROOT / "configs" / "test_clara_matched_six_action_selectors_v1.json"
)
EXTENDED_CONFIG_PATH = (
    TEST_ROOT / "configs" / "test_clara_extended_action_external_v1.json"
)
S09_CONFIG_PATH = S09_ROOT / "configs" / "s09_minimum_external_evaluation_v1.json"
S09_EXTERNAL_PROTOCOL = S09_ROOT / "configs" / "s09_external_validation_v1.json"
S09_SELECTED_LOCAL = (
    S09_ROOT
    / "results_raw"
    / "local_conformal_selection_v1"
    / "selected_configurations.json"
)
REGRESSION_ROOT = (
    TEST_ROOT
    / "results_raw"
    / "matched_six_action_selector_four_action_regression_v1"
)
RESULT_ROOT = TEST_ROOT / "results_raw" / "matched_six_action_selectors_v1"
UNIT_ROOT = RESULT_ROOT / "units"
RUNNER_PATH = Path(__file__).resolve()
ADAPTER_PATH = TEST_ROOT / "scripts" / "extended_selector_adapters.py"
EXTENDED_RUNNER_PATH = TEST_ROOT / "scripts" / "run_extended_action_external.py"
METHOD_CART = "CART_6A_Local"
METHOD_LINUCB = "LinUCB_6A_Local"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def unit_id(farm: str, seed: int) -> str:
    return f"{farm}-S{int(seed)}"


def _unit_complete(path: Path, identities: dict[str, str]) -> bool:
    manifest_path = path / "manifest.json"
    required = (
        path / "cell_metrics.parquet",
        path / "action_counts.parquet",
        path / "paired_block_statistics.parquet",
        path / "cart_state_statistics.parquet",
        path / "method_audits.json",
    )
    if not manifest_path.exists() or any(not item.exists() for item in required):
        return False
    manifest = load_json(manifest_path)
    if manifest.get("status") != "PASS" or manifest.get("identities") != identities:
        return False
    checks = (
        ("cell_metrics_sha256", "cell_metrics.parquet"),
        ("action_counts_sha256", "action_counts.parquet"),
        ("paired_block_statistics_sha256", "paired_block_statistics.parquet"),
        ("cart_state_statistics_sha256", "cart_state_statistics.parquet"),
        ("method_audits_sha256", "method_audits.json"),
    )
    return all(
        manifest.get(key) == sha256_file(path / filename)
        for key, filename in checks
    )


def _identities() -> dict[str, str]:
    regression_manifest = REGRESSION_ROOT / "manifest.json"
    regression = load_json(regression_manifest)
    if regression.get("status") != "PASS":
        raise RuntimeError("匹配选择器四动作等价回归尚未通过")
    if int(regression["cart"]["action_mismatch_count"]) != 0:
        raise RuntimeError("匹配选择器 CART 四动作回归仍有失配")
    if int(regression["cart"]["metric_regression"]["mismatch_count"]) != 0:
        raise RuntimeError("匹配选择器 CART 四动作指标回归仍有失配")
    if int(regression["linucb"]["metric_regression"]["mismatch_count"]) != 0:
        raise RuntimeError("匹配选择器 LinUCB 四动作指标回归仍有失配")
    paths = {
        "config": CONFIG_PATH,
        "extended_config": EXTENDED_CONFIG_PATH,
        "runner": RUNNER_PATH,
        "adapter": ADAPTER_PATH,
        "extended_runner": EXTENDED_RUNNER_PATH,
        "s09_core": S09_ROOT / "scripts" / "s09_minimum_external_core.py",
        "s09_evaluation_runner": S09_ROOT
        / "scripts"
        / "run_s09_minimum_external_evaluation.py",
        "s09_linucb_accelerator": S09_ROOT
        / "scripts"
        / "s09_linucb_batch_accelerator.py",
        "s09_config": S09_CONFIG_PATH,
        "s09_external_protocol": S09_EXTERNAL_PROTOCOL,
        "s09_selected_local": S09_SELECTED_LOCAL,
        "adaptation_root_manifest": core.ADAPTATION_ROOT / "root_manifest.json",
        "full_root_manifest": core.FULL_REBUILD_ROOT / "root_manifest.json",
        "four_action_regression_manifest": regression_manifest,
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def run_unit(task: tuple[str, int, dict[str, str]]) -> dict[str, str]:
    farm, seed, identities = task
    identifier = unit_id(farm, seed)
    output = UNIT_ROOT / identifier
    if _unit_complete(output, identities):
        return {"unit_id": identifier, "status": "REUSED"}
    if output.exists():
        raise RuntimeError(f"发现未封存的六动作匹配选择器单元，拒绝覆盖: {output}")
    output.mkdir(parents=True, exist_ok=False)
    started_at = utc_now()
    try:
        config = load_json(CONFIG_PATH)
        s09_config = load_json(S09_CONFIG_PATH)
        selected_local = load_json(S09_SELECTED_LOCAL)[farm]
        actions = tuple(str(value) for value in config["action_library"])
        if actions != extended.SIX_ACTIONS:
            raise RuntimeError("匹配选择器动作列表与 CLARA 六动作库失配")
        contracts = load_frozen_contracts()
        theta = ErrfTheta()
        predictors = [str(value) for value in config["scope"]["predictors"]]
        horizons = [int(value) for value in config["scope"]["horizon_steps"]]
        target_coverages = [
            float(value) for value in config["scope"]["target_coverages"]
        ]
        bundle_keys = [
            (predictor, horizon)
            for predictor in predictors
            for horizon in horizons
        ]
        adaptation_frames: dict[tuple[str, int], pd.DataFrame] = {}
        final_frames: dict[tuple[str, int], pd.DataFrame] = {}
        extended_action_audits: list[dict[str, Any]] = []
        input_audits: list[dict[str, Any]] = []
        for bundle_number, (predictor, horizon) in enumerate(bundle_keys, start=1):
            adaptation_raw, adaptation_audit = core.load_fact_bundle(
                root=core.ADAPTATION_ROOT,
                farm=farm,
                predictor=predictor,
                horizon=horizon,
                seed=seed,
                contracts=contracts,
                theta=theta,
            )
            thresholds = core.fit_local_width_thresholds(adaptation_raw, farm)
            adaptation = core.attach_local_width_states(adaptation_raw, thresholds)
            adaptation, adaptation_tsc_audit = extended._generate_tuned_conformal_action(
                root=core.ADAPTATION_ROOT,
                facts=adaptation,
                farm=farm,
                predictor=predictor,
                horizon=horizon,
                seed=seed,
                selected_local=selected_local,
                target_coverages=target_coverages,
                contracts=contracts,
                theta=theta,
                include_training_reliability=True,
            )
            adaptation, adaptation_ensemble_audit = (
                extended._attach_equal_endpoint_action(
                    adaptation,
                    theta=theta,
                    include_training_reliability=True,
                )
            )
            final_raw, final_audit = core.load_fact_bundle(
                root=core.FULL_REBUILD_ROOT,
                farm=farm,
                predictor=predictor,
                horizon=horizon,
                seed=seed,
                contracts=contracts,
                theta=theta,
            )
            final = core.attach_local_width_states(final_raw, thresholds)
            final, final_tsc_audit = extended._generate_tuned_conformal_action(
                root=core.FULL_REBUILD_ROOT,
                facts=final,
                farm=farm,
                predictor=predictor,
                horizon=horizon,
                seed=seed,
                selected_local=selected_local,
                target_coverages=target_coverages,
                contracts=contracts,
                theta=theta,
                include_training_reliability=False,
            )
            final, final_ensemble_audit = extended._attach_equal_endpoint_action(
                final,
                theta=theta,
                include_training_reliability=False,
            )
            adaptation_frames[(predictor, horizon)] = adaptation
            final_frames[(predictor, horizon)] = final
            input_audits.append(
                {
                    "predictor": predictor,
                    "horizon_steps": horizon,
                    "adaptation_fact_content_sha256": adaptation_audit[
                        "fact_content_sha256"
                    ],
                    "final_fact_content_sha256": final_audit["fact_content_sha256"],
                    "adaptation_event_count": len(adaptation),
                    "final_event_count": len(final),
                }
            )
            extended_action_audits.append(
                {
                    "predictor": predictor,
                    "horizon_steps": horizon,
                    "tuned_conformal": {
                        "adaptation": adaptation_tsc_audit,
                        "final": final_tsc_audit,
                    },
                    "equal_endpoint_ensemble": {
                        "adaptation": adaptation_ensemble_audit,
                        "final": final_ensemble_audit,
                    },
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
            del adaptation_raw, final_raw
            gc.collect()

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
            raise RuntimeError("六动作匹配选择器跨流事件标识重复")

        cart_selector, cart_statistics, cart_fit_audit = fit_cart_local_actions(
            adaptation_all,
            farm=farm,
            contracts=contracts,
            config=s09_config["cart"]["local_configuration"],
            actions=actions,
        )
        linucb_config = s09_config["linucb"]["configuration"]
        linucb_all, linucb_audit = run_linucb_local_actions(
            final_facts=final_all,
            adaptation_facts=adaptation_all,
            contracts=contracts,
            exploration_alpha=float(linucb_config["exploration_alpha"]),
            l2_regularization=float(linucb_config["l2_regularization"]),
            actions=actions,
        )

        slices: dict[tuple[str, int], slice] = {}
        cursor = 0
        for key in bundle_keys:
            count = len(final_frames[key])
            slices[key] = slice(cursor, cursor + count)
            cursor += count
        if cursor != len(final_all):
            raise RuntimeError("六动作匹配选择器事件切片不闭合")

        metric_parts: list[pd.DataFrame] = []
        action_parts: list[pd.DataFrame] = []
        block_parts: list[pd.DataFrame] = []
        score_audits: list[dict[str, Any]] = []
        for predictor, horizon in bundle_keys:
            facts = final_frames[(predictor, horizon)].reset_index(drop=True)
            cart_actions, cart_scores = predict_cart_local_actions(
                facts, cart_selector
            )
            linucb_actions = linucb_all[slices[(predictor, horizon)]]
            cart_arrays = extended._selected_endpoint_metrics(
                facts,
                cart_actions,
                actions=actions,
                theta=theta,
            )
            s09_evaluation.append_summaries(
                facts=facts,
                arrays=cart_arrays,
                method=METHOD_CART,
                selected_actions=cart_actions,
                metric_parts=metric_parts,
                action_parts=action_parts,
                block_parts=block_parts,
            )
            linucb_arrays = extended._selected_endpoint_metrics(
                facts,
                linucb_actions,
                actions=actions,
                theta=theta,
            )
            s09_evaluation.append_summaries(
                facts=facts,
                arrays=linucb_arrays,
                method=METHOD_LINUCB,
                selected_actions=linucb_actions,
                metric_parts=metric_parts,
                action_parts=action_parts,
                block_parts=block_parts,
            )
            score_audits.append(
                {
                    "predictor": predictor,
                    "horizon_steps": horizon,
                    "event_count": len(facts),
                    "cart_score_finite_count": int(np.isfinite(cart_scores).sum()),
                    "cart_score_minimum": float(np.min(cart_scores)),
                    "cart_score_maximum": float(np.max(cart_scores)),
                }
            )

        cell_metrics = pd.concat(metric_parts, ignore_index=True)
        action_counts = pd.concat(action_parts, ignore_index=True)
        paired_blocks = pd.concat(block_parts, ignore_index=True)
        expected_metric_rows = len(bundle_keys) * len(target_coverages) * 3 * 2
        if len(cell_metrics) != expected_metric_rows:
            raise RuntimeError(
                f"六动作匹配选择器指标行数失配: {len(cell_metrics)}"
            )
        if set(cell_metrics["method"].astype(str)) != {METHOD_CART, METHOD_LINUCB}:
            raise RuntimeError("六动作匹配选择器方法集合失配")
        overall_counts = action_counts[
            action_counts["regime"].astype(str).eq("overall")
        ]
        for method in (METHOD_CART, METHOD_LINUCB):
            method_total = int(
                overall_counts.loc[
                    overall_counts["method"].astype(str).eq(method), "event_count"
                ].sum()
            )
            if method_total != len(final_all):
                raise RuntimeError(f"{method} 动作事件数量不守恒")

        core.write_parquet_atomic(cell_metrics, output / "cell_metrics.parquet")
        core.write_parquet_atomic(action_counts, output / "action_counts.parquet")
        core.write_parquet_atomic(
            paired_blocks, output / "paired_block_statistics.parquet"
        )
        core.write_parquet_atomic(
            cart_statistics, output / "cart_state_statistics.parquet"
        )
        method_audits = {
            "schema": "TEST_CLARA_MATCHED_SIX_ACTION_SELECTOR_AUDIT_V1",
            "unit_id": identifier,
            "actions": list(actions),
            "cart": {
                "fit": cart_fit_audit,
                "scores": score_audits,
            },
            "linucb": linucb_audit,
            "inputs": input_audits,
            "extended_action_generation": extended_action_audits,
            "event_level_method_table_written": False,
            "aggregate_performance_read_by_runner": False,
        }
        core.write_json_atomic(method_audits, output / "method_audits.json")
        manifest = {
            "schema": "TEST_CLARA_MATCHED_SIX_ACTION_SELECTOR_UNIT_V1",
            "status": "PASS",
            "unit_id": identifier,
            "farm": farm,
            "seed": seed,
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "identities": identities,
            "actions": list(actions),
            "method_count": 2,
            "adaptation_event_count": len(adaptation_all),
            "final_event_count": len(final_all),
            "metric_row_count": len(cell_metrics),
            "cell_metrics_sha256": sha256_file(output / "cell_metrics.parquet"),
            "action_counts_sha256": sha256_file(output / "action_counts.parquet"),
            "paired_block_statistics_sha256": sha256_file(
                output / "paired_block_statistics.parquet"
            ),
            "cart_state_statistics_sha256": sha256_file(
                output / "cart_state_statistics.parquet"
            ),
            "method_audits_sha256": sha256_file(output / "method_audits.json"),
            "aggregate_performance_read_by_runner": False,
        }
        core.write_json_atomic(manifest, output / "manifest.json")
        return {"unit_id": identifier, "status": "COMPLETED"}
    except Exception as exc:
        core.write_json_atomic(
            {
                "schema": "TEST_CLARA_MATCHED_SIX_ACTION_SELECTOR_EXCEPTION_V1",
                "status": "FAILED_PRESERVED",
                "unit_id": identifier,
                "failed_at_utc": utc_now(),
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            output / "exception.json",
        )
        raise


def _finalize_root(config: dict[str, Any], identities: dict[str, str]) -> None:
    expected_units = [
        unit_id(farm, seed)
        for farm in config["scope"]["farms"]
        for seed in config["scope"]["seeds"]
    ]
    if not all(_unit_complete(UNIT_ROOT / item, identities) for item in expected_units):
        raise RuntimeError("六动作匹配选择器仍有未完成单元")
    cells = pd.concat(
        [pd.read_parquet(UNIT_ROOT / item / "cell_metrics.parquet") for item in expected_units],
        ignore_index=True,
    )
    counts = pd.concat(
        [pd.read_parquet(UNIT_ROOT / item / "action_counts.parquet") for item in expected_units],
        ignore_index=True,
    )
    blocks = pd.concat(
        [
            pd.read_parquet(UNIT_ROOT / item / "paired_block_statistics.parquet")
            for item in expected_units
        ],
        ignore_index=True,
    )
    core.write_parquet_atomic(cells, RESULT_ROOT / "cell_metrics.parquet")
    core.write_parquet_atomic(counts, RESULT_ROOT / "action_counts.parquet")
    core.write_parquet_atomic(
        blocks, RESULT_ROOT / "paired_block_statistics.parquet"
    )
    root_manifest = {
        "schema": "TEST_CLARA_MATCHED_SIX_ACTION_SELECTOR_ROOT_V1",
        "status": "TECHNICAL_RUN_COMPLETE_AWAITING_INDEPENDENT_QA",
        "completed_at_utc": utc_now(),
        "identities": identities,
        "unit_count": len(expected_units),
        "units": expected_units,
        "method_count": 2,
        "cell_metric_row_count": len(cells),
        "cell_metrics_sha256": sha256_file(RESULT_ROOT / "cell_metrics.parquet"),
        "action_counts_sha256": sha256_file(RESULT_ROOT / "action_counts.parquet"),
        "paired_block_statistics_sha256": sha256_file(
            RESULT_ROOT / "paired_block_statistics.parquet"
        ),
        "aggregate_performance_read_by_runner": False,
    }
    core.write_json_atomic(root_manifest, RESULT_ROOT / "root_manifest.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=None)
    arguments = parser.parse_args()
    config = load_json(CONFIG_PATH)
    identities = _identities()
    minimum_free = float(config["execution"]["minimum_free_disk_gib"])
    free_gib = shutil.disk_usage(TEST_ROOT).free / (1024**3)
    if free_gib < minimum_free:
        raise RuntimeError(
            f"六动作匹配选择器磁盘安全门失败: {free_gib:.2f} GiB"
        )
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    UNIT_ROOT.mkdir(parents=True, exist_ok=True)
    tasks = [
        (str(farm), int(seed), identities)
        for farm in config["scope"]["farms"]
        for seed in config["scope"]["seeds"]
    ]
    pending = [
        task
        for task in tasks
        if not _unit_complete(UNIT_ROOT / unit_id(task[0], task[1]), identities)
    ]
    workers = arguments.max_workers or int(config["execution"]["max_workers"])
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_unit, task): task for task in pending}
            for future in as_completed(futures):
                result = future.result()
                print(json.dumps(result, ensure_ascii=False), flush=True)
    _finalize_root(config, identities)
    print(
        json.dumps(
            {
                "status": "TECHNICAL_RUN_COMPLETE_AWAITING_INDEPENDENT_QA",
                "result_root": str(RESULT_ROOT),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
