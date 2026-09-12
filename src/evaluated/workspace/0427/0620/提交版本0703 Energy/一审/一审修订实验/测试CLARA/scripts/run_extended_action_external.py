"""运行五动作与六动作 CLARA 的商业场站外部消融。

五动作版本加入已封存的本地 TunedSingleConformal 区间。六动作版本再加入
原四个传统端点的事件级等权平均。状态、支持回退、自适应 n_min 与 nu、
经验均值护栏和 ERRF 定义均继承 S09。
"""
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
from typing import Any, Sequence

import numpy as np
import pandas as pd


TEST_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = TEST_ROOT.parent
S09_ROOT = REVISION_ROOT / "09_商业场站外部验证"
AUTHORITATIVE_CODE = REVISION_ROOT / "_权威代码" / "code"
sys.path.insert(0, str(AUTHORITATIVE_CODE))
sys.path.insert(0, str(S09_ROOT / "scripts"))

from clara_event_contract import load_frozen_contracts  # noqa: E402
from clara_errf import ErrfTheta  # noqa: E402
from source_tuning_facts import (  # noqa: E402
    STREAM_FIELDS,
    _CompactRollingSummaries,
    _datetime_ns,
)
import run_s09_external_analysis as external_analysis  # noqa: E402
import run_s09_minimum_external_evaluation as s09_evaluation  # noqa: E402
import s09_minimum_external_core as core  # noqa: E402


CONFIG_PATH = TEST_ROOT / "configs" / "test_clara_extended_action_external_v1.json"
V4_CONFIG_PATH = TEST_ROOT / "configs" / "test_clara_four_versions_v1.json"
S09_EVALUATION_CONFIG = S09_ROOT / "configs" / "s09_minimum_external_evaluation_v1.json"
S09_EXTERNAL_PROTOCOL = S09_ROOT / "configs" / "s09_external_validation_v1.json"
S09_SELECTED_LOCAL = (
    S09_ROOT
    / "results_raw"
    / "local_conformal_selection_v1"
    / "selected_configurations.json"
)
S09_RAW_EVALUATION = S09_ROOT / "results_raw" / "minimum_external_evaluation_v1"
S09_RAW_QA = S09_ROOT / "qa" / "minimum_external_evaluation_v1" / "qa_summary.json"
RESULT_ROOT = TEST_ROOT / "results_raw" / "extended_action_external_v1"
UNIT_ROOT = RESULT_ROOT / "units"
RUNNER_PATH = Path(__file__).resolve()

ORIGINAL_ACTIONS = ("Static", "ACI", "AgACI", "EnbPI_RH")
FIFTH_ACTION = "TunedSingleConformal_Local"
FIVE_ACTIONS = (*ORIGINAL_ACTIONS, FIFTH_ACTION)
SIXTH_ACTION = "EqualEndpointEnsemble"
SIX_ACTIONS = (*FIVE_ACTIONS, SIXTH_ACTION)
METHOD_5A = "CLARA_5A_Local"
METHOD_6A = "CLARA_6A_Local"
METRIC_COLUMNS = (
    "mean_errf",
    "mean_reserve",
    "mean_miss",
    "empirical_coverage",
    "coverage_gap",
    "average_width",
    "TOWR",
    "TUWR",
    "ARD",
)


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


def _validate_conformal_audit(audit: dict[str, Any], label: str) -> None:
    for row in audit["audits"]:
        if (
            int(row["future_feedback_violation_count"]) != 0
            or int(row["within_issue_feedback_use_count"]) != 0
            or int(row["invalid_interval_count"]) != 0
            or int(row["out_of_bounds_count"]) != 0
            or bool(row["final_feedback_drain"])
        ):
            raise RuntimeError(f"扩展动作 CLARA 的 {label} 共形因果审计失败")


def _attach_candidate_action(
    facts: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    *,
    action: str,
    include_training_reliability: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """把候选动作端点和训练期可靠性事实加入冻结事实表。"""

    output = facts.copy().reset_index(drop=True)
    expected = {"lower", "upper", "errf", "covered"}
    if not expected.issubset(arrays):
        raise RuntimeError(f"候选动作数组字段不完整: {action}")
    for key in expected:
        if len(np.asarray(arrays[key])) != len(output):
            raise RuntimeError(f"候选动作数组长度失配: {action}, {key}")
    lower = np.asarray(arrays["lower"], dtype=np.float64)
    upper = np.asarray(arrays["upper"], dtype=np.float64)
    errf = np.asarray(arrays["errf"], dtype=np.float64)
    covered = np.asarray(arrays["covered"], dtype=bool)
    if (
        not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or not np.isfinite(errf).all()
        or np.any(lower > upper)
        or np.any(lower < 0.0)
        or np.any(upper > 1.0)
    ):
        raise RuntimeError(f"候选动作端点或 ERRF 违反冻结合同: {action}")
    output[f"{action}__candidate_lower"] = lower
    output[f"{action}__candidate_upper"] = upper
    output[f"{action}__errf"] = errf
    output[f"{action}__covered"] = covered
    output[f"{action}__tuwr_indicator"] = np.nan
    output[f"{action}__ard_value"] = np.nan
    if not include_training_reliability:
        return output, {
            "event_count": len(output),
            "reliability_event_count": 0,
            "cold_state_count": int(
                output["rolling_state"].astype(str).eq("cold_start").sum()
            ),
        }

    reliability_count = 0
    cold_count = 0
    grouped = output.groupby(list(STREAM_FIELDS), sort=True, dropna=False)
    for _, group in grouped:
        ordered = group.sort_values(["issue_timestamp", "event_id"], kind="mergesort")
        positions = ordered.index.to_numpy(dtype=np.int64)
        label_ns = _datetime_ns(ordered["label_timestamp"])
        availability_ns = _datetime_ns(ordered["label_available_timestamp"])
        if len(label_ns) > 1 and np.any(np.diff(label_ns) <= 0):
            raise RuntimeError(f"候选动作可靠性流的标签时刻未严格递增: {action}")
        if len(availability_ns) > 1 and np.any(np.diff(availability_ns) <= 0):
            raise RuntimeError(f"候选动作可靠性流的反馈时刻未严格递增: {action}")
        cadence = ordered["nominal_cadence_minutes"].to_numpy(dtype=np.float64)
        if len(np.unique(cadence)) != 1:
            raise RuntimeError(f"候选动作可靠性流的时间步长不唯一: {action}")
        target_coverage = float(ordered["target_coverage"].iloc[0])
        summaries = _CompactRollingSummaries(
            label_ns=label_ns,
            covered=covered[positions],
            target_coverage=target_coverage,
            cadence_minutes=float(cadence[0]),
        )
        release_full = ~ordered["rolling_state"].astype(str).eq("cold_start").to_numpy()
        action_tuwr = np.full(len(ordered), np.nan, dtype=np.float64)
        action_ard = np.full(len(ordered), np.nan, dtype=np.float64)
        for row_index, feedback_time_ns in enumerate(availability_ns):
            if not release_full[row_index]:
                cold_count += 1
                continue
            _, _, _, tuwr, ard, state = summaries.summarize(
                end_exclusive=row_index + 1,
                query_ns=int(feedback_time_ns),
            )
            if state == "cold_start" or not np.isfinite(tuwr) or not np.isfinite(ard):
                raise RuntimeError(f"候选动作完整历史状态缺少可靠性事实: {action}")
            action_tuwr[row_index] = tuwr
            action_ard[row_index] = ard
            reliability_count += 1
        output.loc[positions, f"{action}__tuwr_indicator"] = action_tuwr
        output.loc[positions, f"{action}__ard_value"] = action_ard
    expected_reliability = int(
        (~output["rolling_state"].astype(str).eq("cold_start")).sum()
    )
    if reliability_count != expected_reliability:
        raise RuntimeError(f"候选动作训练可靠性事件数量不守恒: {action}")
    return output, {
        "event_count": len(output),
        "reliability_event_count": reliability_count,
        "cold_state_count": cold_count,
    }


def _selected_endpoint_metrics(
    facts: pd.DataFrame,
    selected_actions: Sequence[str],
    *,
    actions: Sequence[str],
    theta: ErrfTheta,
) -> dict[str, np.ndarray]:
    selected = np.asarray(selected_actions, dtype=str)
    if len(selected) != len(facts) or set(selected) - set(actions):
        raise RuntimeError("五动作 CLARA 所选动作集合失配")
    lower = np.empty(len(facts), dtype=np.float64)
    upper = np.empty(len(facts), dtype=np.float64)
    for action in actions:
        mask = selected == action
        lower[mask] = facts.loc[
            mask, f"{action}__candidate_lower"
        ].to_numpy(dtype=np.float64)
        upper[mask] = facts.loc[
            mask, f"{action}__candidate_upper"
        ].to_numpy(dtype=np.float64)
    return core._metric_arrays_from_endpoints(facts, lower, upper, theta=theta)


def _generate_tuned_conformal_action(
    *,
    root: Path,
    facts: pd.DataFrame,
    farm: str,
    predictor: str,
    horizon: int,
    seed: int,
    selected_local: dict[str, Any],
    target_coverages: Sequence[float],
    contracts: Any,
    theta: ErrfTheta,
    include_training_reliability: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = core.bundle_paths(
        root=root,
        farm=farm,
        predictor=predictor,
        horizon=horizon,
        seed=seed,
    )
    manifest = load_json(paths["manifest"])
    base_predictions = pd.read_parquet(paths["base_predictions"])
    arrays, conformal_audit = s09_evaluation.interval_arrays_for_conformal(
        facts=facts,
        base_predictions=base_predictions,
        contracts=contracts,
        theta=theta,
        dataset_id=str(manifest["dataset_id"]),
        farm=farm,
        predictor=predictor,
        seed=seed,
        family=str(selected_local["family"]),
        configuration=dict(selected_local["configuration"]),
        target_coverages=target_coverages,
    )
    _validate_conformal_audit(conformal_audit, str(root.name))
    output, reliability_audit = _attach_candidate_action(
        facts,
        arrays,
        action=FIFTH_ACTION,
        include_training_reliability=include_training_reliability,
    )
    audit = {
        "root": str(root),
        "family": str(selected_local["family"]),
        "configuration": dict(selected_local["configuration"]),
        "base_predictions_sha256": sha256_file(paths["base_predictions"]),
        "stream_count": int(conformal_audit["stream_count"]),
        "causal_audits": conformal_audit["audits"],
        "reliability": reliability_audit,
    }
    del base_predictions, arrays
    gc.collect()
    return output, audit


def _attach_equal_endpoint_action(
    facts: pd.DataFrame,
    *,
    theta: ErrfTheta,
    include_training_reliability: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """加入原四个传统动作端点的事件级等权平均。"""

    lower = np.mean(
        np.column_stack(
            [
                facts[f"{action}__candidate_lower"].to_numpy(dtype=np.float64)
                for action in ORIGINAL_ACTIONS
            ]
        ),
        axis=1,
    )
    upper = np.mean(
        np.column_stack(
            [
                facts[f"{action}__candidate_upper"].to_numpy(dtype=np.float64)
                for action in ORIGINAL_ACTIONS
            ]
        ),
        axis=1,
    )
    arrays = core._metric_arrays_from_endpoints(facts, lower, upper, theta=theta)
    output, reliability_audit = _attach_candidate_action(
        facts,
        arrays,
        action=SIXTH_ACTION,
        include_training_reliability=include_training_reliability,
    )
    original_global_actions = core.ACTIONS
    try:
        core.ACTIONS = ORIGINAL_ACTIONS
        direct = core.endpoint_metrics_for_ensemble(facts, theta=theta)
    finally:
        core.ACTIONS = original_global_actions
    maximum_difference = 0.0
    for key in ("lower", "upper", "errf", "covered"):
        left = np.asarray(arrays[key])
        right = np.asarray(direct[key])
        if left.dtype == bool:
            if not np.array_equal(left, right):
                raise RuntimeError("等权端点动作布尔结果复算失配")
        else:
            difference = float(np.max(np.abs(left.astype(float) - right.astype(float))))
            maximum_difference = max(maximum_difference, difference)
    if maximum_difference != 0.0:
        raise RuntimeError("等权端点动作逐值复算失配")
    return output, {
        "definition": "mean_of_original_four_lower_and_upper_endpoints",
        "source_actions": list(ORIGINAL_ACTIONS),
        "maximum_recalculation_difference": maximum_difference,
        "reliability": reliability_audit,
    }


def _compare_four_action_replay(
    replay: pd.DataFrame,
    *,
    identifier: str,
) -> dict[str, Any]:
    reference_path = S09_RAW_EVALUATION / "units" / identifier / "cell_metrics.parquet"
    reference = pd.read_parquet(reference_path)
    reference = reference[
        reference["method"].astype(str).eq("CLARA_V4_Local")
    ].copy()
    candidate = replay.copy()
    candidate["method"] = "CLARA_V4_Local"
    keys = [
        "zone_or_farm",
        "predictor",
        "seed",
        "horizon_steps",
        "target_coverage",
        "method",
        "regime",
    ]
    reference = reference.sort_values(keys, kind="mergesort").reset_index(drop=True)
    candidate = candidate.sort_values(keys, kind="mergesort").reset_index(drop=True)
    if len(reference) != len(candidate) or not reference[keys].equals(candidate[keys]):
        raise RuntimeError("四动作回放键与 S09 封存结果失配")
    count_mismatch = int(
        reference["event_count"].astype(int).ne(candidate["event_count"].astype(int)).sum()
    ) + int(
        reference["reliability_event_count"]
        .astype(int)
        .ne(candidate["reliability_event_count"].astype(int))
        .sum()
    )
    max_abs = 0.0
    mismatch = count_mismatch
    for column in METRIC_COLUMNS:
        left = reference[column].to_numpy(dtype=np.float64)
        right = candidate[column].to_numpy(dtype=np.float64)
        both_nan = np.isnan(left) & np.isnan(right)
        delta = np.abs(left - right)
        delta[both_nan] = 0.0
        finite_delta = delta[np.isfinite(delta)]
        if len(finite_delta):
            max_abs = max(max_abs, float(finite_delta.max()))
        mismatch += int((~both_nan & ~np.isclose(left, right, rtol=0.0, atol=1e-12)).sum())
    if mismatch != 0:
        raise RuntimeError(
            f"四动作回放未精确复现 S09: mismatch={mismatch}, max_abs={max_abs}"
        )
    return {
        "reference_path": str(reference_path),
        "row_count": len(reference),
        "mismatch_count": mismatch,
        "maximum_absolute_difference": max_abs,
    }


def _unit_complete(path: Path, identities: dict[str, str]) -> bool:
    manifest_path = path / "manifest.json"
    required = (
        path / "cell_metrics.parquet",
        path / "action_counts.parquet",
        path / "paired_block_statistics.parquet",
        path / "state_decisions.parquet",
        path / "method_audits.json",
    )
    if not manifest_path.exists() or any(not item.exists() for item in required):
        return False
    manifest = load_json(manifest_path)
    if manifest.get("status") != "PASS" or manifest.get("identities") != identities:
        return False
    for key, filename in (
        ("cell_metrics_sha256", "cell_metrics.parquet"),
        ("action_counts_sha256", "action_counts.parquet"),
        ("paired_block_statistics_sha256", "paired_block_statistics.parquet"),
        ("state_decisions_sha256", "state_decisions.parquet"),
        ("method_audits_sha256", "method_audits.json"),
    ):
        if manifest.get(key) != sha256_file(path / filename):
            return False
    return True


def run_unit(task: tuple[str, int, dict[str, str]]) -> dict[str, str]:
    farm, seed, identities = task
    identifier = unit_id(farm, seed)
    output = UNIT_ROOT / identifier
    if _unit_complete(output, identities):
        return {"unit_id": identifier, "status": "REUSED"}
    if output.exists():
        raise RuntimeError(f"发现未封存的五动作单元，拒绝覆盖: {output}")
    output.mkdir(parents=True, exist_ok=False)
    started_at = utc_now()
    try:
        config = load_json(CONFIG_PATH)
        s09_config = load_json(S09_EVALUATION_CONFIG)
        selected_local = load_json(S09_SELECTED_LOCAL)[farm]
        contracts = load_frozen_contracts()
        theta = ErrfTheta()
        v4_protocol = load_json(V4_CONFIG_PATH)
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
            adaptation, adaptation_tsc_audit = _generate_tuned_conformal_action(
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
            adaptation, adaptation_ensemble_audit = _attach_equal_endpoint_action(
                adaptation,
                theta=theta,
                include_training_reliability=True,
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
            final, final_tsc_audit = _generate_tuned_conformal_action(
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
            final, final_ensemble_audit = _attach_equal_endpoint_action(
                final,
                theta=theta,
                include_training_reliability=False,
            )
            adaptation_frames[(predictor, horizon)] = adaptation
            final_frames[(predictor, horizon)] = final
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
            raise RuntimeError("扩展动作 CLARA 跨流事件标识重复")

        core.ACTIONS = ORIGINAL_ACTIONS
        four_decisions, four_audit = core.fit_local_clara(
            adaptation_all,
            contracts=contracts,
            protocol=s09_config["local_clara_estimator"],
            v4_protocol=v4_protocol,
        )
        core.ACTIONS = FIVE_ACTIONS
        five_decisions, five_audit = core.fit_local_clara(
            adaptation_all,
            contracts=contracts,
            protocol=s09_config["local_clara_estimator"],
            v4_protocol=v4_protocol,
        )
        core.ACTIONS = SIX_ACTIONS
        six_decisions, six_audit = core.fit_local_clara(
            adaptation_all,
            contracts=contracts,
            protocol=s09_config["local_clara_estimator"],
            v4_protocol=v4_protocol,
        )
        core.ACTIONS = ORIGINAL_ACTIONS

        four_metric_parts: list[pd.DataFrame] = []
        five_metric_parts: list[pd.DataFrame] = []
        five_action_parts: list[pd.DataFrame] = []
        five_block_parts: list[pd.DataFrame] = []
        six_metric_parts: list[pd.DataFrame] = []
        six_action_parts: list[pd.DataFrame] = []
        six_block_parts: list[pd.DataFrame] = []
        dummy_action_parts: list[pd.DataFrame] = []
        dummy_block_parts: list[pd.DataFrame] = []
        for predictor, horizon in bundle_keys:
            facts = final_frames[(predictor, horizon)].reset_index(drop=True)
            four_actions = core.predict_local_clara(facts, four_decisions)
            four_arrays = _selected_endpoint_metrics(
                facts,
                four_actions,
                actions=ORIGINAL_ACTIONS,
                theta=theta,
            )
            s09_evaluation.append_summaries(
                facts=facts,
                arrays=four_arrays,
                method="CLARA_V4_Local_Replay",
                selected_actions=four_actions,
                metric_parts=four_metric_parts,
                action_parts=dummy_action_parts,
                block_parts=dummy_block_parts,
            )
            five_actions = core.predict_local_clara(facts, five_decisions)
            five_arrays = _selected_endpoint_metrics(
                facts,
                five_actions,
                actions=FIVE_ACTIONS,
                theta=theta,
            )
            s09_evaluation.append_summaries(
                facts=facts,
                arrays=five_arrays,
                method=METHOD_5A,
                selected_actions=five_actions,
                metric_parts=five_metric_parts,
                action_parts=five_action_parts,
                block_parts=five_block_parts,
            )
            six_actions = core.predict_local_clara(facts, six_decisions)
            six_arrays = _selected_endpoint_metrics(
                facts,
                six_actions,
                actions=SIX_ACTIONS,
                theta=theta,
            )
            s09_evaluation.append_summaries(
                facts=facts,
                arrays=six_arrays,
                method=METHOD_6A,
                selected_actions=six_actions,
                metric_parts=six_metric_parts,
                action_parts=six_action_parts,
                block_parts=six_block_parts,
            )
        four_metrics = pd.concat(four_metric_parts, ignore_index=True)
        five_metrics = pd.concat(five_metric_parts, ignore_index=True)
        five_counts = pd.concat(five_action_parts, ignore_index=True)
        five_blocks = pd.concat(five_block_parts, ignore_index=True)
        six_metrics = pd.concat(six_metric_parts, ignore_index=True)
        six_counts = pd.concat(six_action_parts, ignore_index=True)
        six_blocks = pd.concat(six_block_parts, ignore_index=True)
        expected_metric_rows = len(bundle_keys) * len(target_coverages) * 3
        if len(five_metrics) != expected_metric_rows:
            raise RuntimeError("五动作 CLARA 单元指标行数失配")
        if len(six_metrics) != expected_metric_rows:
            raise RuntimeError("六动作 CLARA 单元指标行数失配")
        four_regression = _compare_four_action_replay(
            four_metrics,
            identifier=identifier,
        )
        fifth_selected = int(
            five_counts.loc[
                five_counts["selected_action"].astype(str).eq(FIFTH_ACTION)
                & five_counts["regime"].astype(str).eq("overall"),
                "event_count",
            ].sum()
        )
        total_selected = int(
            five_counts.loc[
                five_counts["regime"].astype(str).eq("overall"), "event_count"
            ].sum()
        )
        if total_selected != len(final_all):
            raise RuntimeError("五动作 CLARA 最终动作事件数量不守恒")
        sixth_selected = int(
            six_counts.loc[
                six_counts["selected_action"].astype(str).eq(SIXTH_ACTION)
                & six_counts["regime"].astype(str).eq("overall"),
                "event_count",
            ].sum()
        )
        fifth_selected_in_six = int(
            six_counts.loc[
                six_counts["selected_action"].astype(str).eq(FIFTH_ACTION)
                & six_counts["regime"].astype(str).eq("overall"),
                "event_count",
            ].sum()
        )
        total_selected_six = int(
            six_counts.loc[
                six_counts["regime"].astype(str).eq("overall"), "event_count"
            ].sum()
        )
        if total_selected_six != len(final_all):
            raise RuntimeError("六动作 CLARA 最终动作事件数量不守恒")

        cell_metrics = pd.concat([five_metrics, six_metrics], ignore_index=True)
        action_counts = pd.concat([five_counts, six_counts], ignore_index=True)
        paired_blocks = pd.concat([five_blocks, six_blocks], ignore_index=True)
        state_decisions = pd.concat(
            [
                five_decisions.assign(method=METHOD_5A),
                six_decisions.assign(method=METHOD_6A),
            ],
            ignore_index=True,
            sort=False,
        )
        core.write_parquet_atomic(cell_metrics, output / "cell_metrics.parquet")
        core.write_parquet_atomic(action_counts, output / "action_counts.parquet")
        core.write_parquet_atomic(
            paired_blocks, output / "paired_block_statistics.parquet"
        )
        core.write_parquet_atomic(state_decisions, output / "state_decisions.parquet")
        method_audits = {
            "schema": "TEST_CLARA_EXTENDED_ACTION_METHOD_AUDIT_V1",
            "unit_id": identifier,
            "five_actions": list(FIVE_ACTIONS),
            "six_actions": list(SIX_ACTIONS),
            "fifth_action": FIFTH_ACTION,
            "sixth_action": SIXTH_ACTION,
            "five_action_model": {
                "fifth_action_selected_event_count": fifth_selected,
                "fifth_action_selection_share": fifth_selected / total_selected,
            },
            "six_action_model": {
                "fifth_action_selected_event_count": fifth_selected_in_six,
                "fifth_action_selection_share": fifth_selected_in_six
                / total_selected_six,
                "sixth_action_selected_event_count": sixth_selected,
                "sixth_action_selection_share": sixth_selected / total_selected_six,
            },
            "four_action_replay": four_regression,
            "four_action_fit": four_audit,
            "five_action_fit": five_audit,
            "six_action_fit": six_audit,
            "inputs": input_audits,
            "extended_action_generation": extended_action_audits,
        }
        core.write_json_atomic(method_audits, output / "method_audits.json")
        manifest = {
            "schema": "TEST_CLARA_EXTENDED_ACTION_UNIT_MANIFEST_V1",
            "status": "PASS",
            "unit_id": identifier,
            "farm": farm,
            "seed": seed,
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "identities": identities,
            "adaptation_event_count": len(adaptation_all),
            "final_event_count": len(final_all),
            "metric_row_count": len(cell_metrics),
            "fifth_action_selected_event_count": fifth_selected,
            "fifth_action_selection_share": fifth_selected / total_selected,
            "six_action_fifth_selected_event_count": fifth_selected_in_six,
            "sixth_action_selected_event_count": sixth_selected,
            "sixth_action_selection_share": sixth_selected / total_selected_six,
            "four_action_regression_mismatch_count": four_regression[
                "mismatch_count"
            ],
            "cell_metrics_sha256": sha256_file(output / "cell_metrics.parquet"),
            "action_counts_sha256": sha256_file(output / "action_counts.parquet"),
            "paired_block_statistics_sha256": sha256_file(
                output / "paired_block_statistics.parquet"
            ),
            "state_decisions_sha256": sha256_file(output / "state_decisions.parquet"),
            "method_audits_sha256": sha256_file(output / "method_audits.json"),
        }
        core.write_json_atomic(manifest, output / "manifest.json")
        return {"unit_id": identifier, "status": "COMPLETED"}
    except Exception as exc:
        core.ACTIONS = ORIGINAL_ACTIONS
        core.write_json_atomic(
            {
                "schema": "TEST_CLARA_EXTENDED_ACTION_EXCEPTION_V1",
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


def _identities() -> dict[str, str]:
    paths = {
        "config": CONFIG_PATH,
        "runner": RUNNER_PATH,
        "s09_core": S09_ROOT / "scripts" / "s09_minimum_external_core.py",
        "s09_evaluation_runner": S09_ROOT
        / "scripts"
        / "run_s09_minimum_external_evaluation.py",
        "s09_analysis": S09_ROOT / "scripts" / "run_s09_external_analysis.py",
        "s09_evaluation_config": S09_EVALUATION_CONFIG,
        "s09_external_protocol": S09_EXTERNAL_PROTOCOL,
        "s09_selected_local": S09_SELECTED_LOCAL,
        "v4_protocol": V4_CONFIG_PATH,
        "adaptation_root_manifest": core.ADAPTATION_ROOT / "root_manifest.json",
        "full_root_manifest": core.FULL_REBUILD_ROOT / "root_manifest.json",
        "s09_qa": S09_RAW_QA,
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _finalize_root(config: dict[str, Any], identities: dict[str, str]) -> None:
    tasks = [
        (farm, int(seed))
        for farm in config["scope"]["farms"]
        for seed in config["scope"]["seeds"]
    ]
    metric_parts: list[pd.DataFrame] = []
    count_parts: list[pd.DataFrame] = []
    block_parts: list[pd.DataFrame] = []
    unit_records: list[dict[str, Any]] = []
    for farm, seed in tasks:
        path = UNIT_ROOT / unit_id(farm, seed)
        if not _unit_complete(path, identities):
            raise RuntimeError(f"扩展动作 CLARA 单元尚未封存: {path.name}")
        manifest = load_json(path / "manifest.json")
        metric_parts.append(pd.read_parquet(path / "cell_metrics.parquet"))
        count_parts.append(pd.read_parquet(path / "action_counts.parquet"))
        block_parts.append(pd.read_parquet(path / "paired_block_statistics.parquet"))
        unit_records.append(
            {
                "unit_id": path.name,
                "manifest_sha256": sha256_file(path / "manifest.json"),
                "final_event_count": int(manifest["final_event_count"]),
                "fifth_action_selection_share": float(
                    manifest["fifth_action_selection_share"]
                ),
                "sixth_action_selection_share": float(
                    manifest["sixth_action_selection_share"]
                ),
            }
        )
    cell_metrics = pd.concat(metric_parts, ignore_index=True)
    action_counts = pd.concat(count_parts, ignore_index=True)
    paired_blocks = pd.concat(block_parts, ignore_index=True)
    core.write_parquet_atomic(cell_metrics, RESULT_ROOT / "cell_metrics.parquet")
    core.write_parquet_atomic(action_counts, RESULT_ROOT / "action_counts.parquet")
    core.write_parquet_atomic(
        paired_blocks, RESULT_ROOT / "paired_block_statistics.parquet"
    )

    existing = pd.read_parquet(S09_RAW_EVALUATION / "cell_metrics.parquet")
    combined = pd.concat([existing, cell_metrics], ignore_index=True, sort=False)
    farm_summary = external_analysis.build_farm_summary(combined)
    external_protocol = load_json(S09_EXTERNAL_PROTOCOL)
    capacities = {
        farm: float(values["capacity_mw"])
        for farm, values in external_protocol["data"]["farms"].items()
    }
    weighted = external_analysis.build_weighted_summary(farm_summary, capacities)
    rankings = external_analysis.build_rankings(weighted)
    core.write_parquet_atomic(farm_summary, RESULT_ROOT / "farm_method_summary.parquet")
    core.write_parquet_atomic(weighted, RESULT_ROOT / "weighted_method_summary.parquet")
    core.write_parquet_atomic(rankings, RESULT_ROOT / "method_rankings.parquet")
    primary = rankings[
        rankings["weighting"].astype(str).eq("equal_farm_weighted")
        & rankings["regime"].astype(str).eq("overall")
    ].sort_values("rank", kind="mergesort")
    own_five = primary[primary["method"].astype(str).eq(METHOD_5A)].iloc[0]
    own_six = primary[primary["method"].astype(str).eq(METHOD_6A)].iloc[0]
    current_best = float(config["evaluation"]["current_best_value"])
    five_success = float(own_five["mean_errf"]) < current_best
    six_success = float(own_six["mean_errf"]) < current_best
    lines = [
        "# 扩展动作 CLARA 外部测试初步结果",
        "",
        f"主指标：场站等权总体 mean ERRF。",
        f"CLARA 5A：{float(own_five['mean_errf']):.9f}，排名 {int(own_five['rank'])}/{len(primary)}。",
        f"CLARA 6A：{float(own_six['mean_errf']):.9f}，排名 {int(own_six['rank'])}/{len(primary)}。",
        f"既有第一名：{current_best:.9f}。",
        f"五动作是否超过既有第一名：{'YES' if five_success else 'NO'}。",
        f"六动作是否超过既有第一名：{'YES' if six_success else 'NO'}。",
        "",
        "## 总体排名",
        "",
        "| 排名 | 方法 | mean ERRF | TUWR | ARD |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in primary.itertuples():
        lines.append(
            f"| {int(row.rank)} | {row.method} | {float(row.mean_errf):.9f} | "
            f"{float(row.TUWR):.6f} | {float(row.ARD):.6f} |"
        )
    (RESULT_ROOT / "RESULTS_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    root_manifest = {
        "schema": "TEST_CLARA_EXTENDED_ACTION_ROOT_MANIFEST_V1",
        "status": "COMPLETE_PENDING_INDEPENDENT_QA",
        "completed_at_utc": utc_now(),
        "identities": identities,
        "unit_count": len(tasks),
        "metric_row_count": len(cell_metrics),
        "action_count_row_count": len(action_counts),
        "paired_block_row_count": len(paired_blocks),
        "five_action_equal_farm_overall_mean_errf": float(own_five["mean_errf"]),
        "five_action_rank": int(own_five["rank"]),
        "six_action_equal_farm_overall_mean_errf": float(own_six["mean_errf"]),
        "six_action_rank": int(own_six["rank"]),
        "comparison_method_count": len(primary),
        "five_action_beats_current_best": five_success,
        "six_action_beats_current_best": six_success,
        "cell_metrics_sha256": sha256_file(RESULT_ROOT / "cell_metrics.parquet"),
        "action_counts_sha256": sha256_file(RESULT_ROOT / "action_counts.parquet"),
        "paired_block_statistics_sha256": sha256_file(
            RESULT_ROOT / "paired_block_statistics.parquet"
        ),
        "farm_method_summary_sha256": sha256_file(
            RESULT_ROOT / "farm_method_summary.parquet"
        ),
        "weighted_method_summary_sha256": sha256_file(
            RESULT_ROOT / "weighted_method_summary.parquet"
        ),
        "method_rankings_sha256": sha256_file(RESULT_ROOT / "method_rankings.parquet"),
        "results_summary_sha256": sha256_file(RESULT_ROOT / "RESULTS_SUMMARY.md"),
        "units": unit_records,
    }
    core.write_json_atomic(root_manifest, RESULT_ROOT / "root_manifest.json")
    print(json.dumps(root_manifest, ensure_ascii=False, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", help="只运行一个场站种子单元，例如 FarmA-S0")
    args = parser.parse_args()
    config = load_json(CONFIG_PATH)
    if config.get("status") != "FROZEN_EXPLORATORY_ABLATION_AFTER_S09_READ":
        raise RuntimeError("五动作 CLARA 配置状态异常")
    libraries = config["action_libraries"]
    if tuple(libraries[METHOD_5A]) != FIVE_ACTIONS:
        raise RuntimeError("五动作 CLARA 动作库与代码失配")
    if tuple(libraries[METHOD_6A]) != SIX_ACTIONS:
        raise RuntimeError("六动作 CLARA 动作库与代码失配")
    qa = load_json(S09_RAW_QA)
    if qa.get("status") != "PASS" or int(qa.get("failed_count", -1)) != 0:
        raise RuntimeError("S09 正式评价独立 QA 未通过")
    free = shutil.disk_usage(TEST_ROOT.drive + "\\").free
    minimum = int(config["execution"]["minimum_free_disk_gib"]) * 1024**3
    if free < minimum:
        raise RuntimeError("扩展动作 CLARA 磁盘安全门失败")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    UNIT_ROOT.mkdir(parents=True, exist_ok=True)
    identities = _identities()
    tasks = [
        (str(farm), int(seed), identities)
        for farm in config["scope"]["farms"]
        for seed in config["scope"]["seeds"]
    ]
    if args.unit:
        selected = [task for task in tasks if unit_id(task[0], task[1]) == args.unit]
        if len(selected) != 1:
            raise RuntimeError(f"未知五动作测试单元: {args.unit}")
        result = run_unit(selected[0])
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    if (RESULT_ROOT / "root_manifest.json").exists():
        raise RuntimeError("扩展动作 CLARA 根清单已经存在，拒绝重复运行")
    completed = 0
    with ProcessPoolExecutor(
        max_workers=int(config["execution"]["max_workers"])
    ) as executor:
        futures = {executor.submit(run_unit, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            print(
                json.dumps(
                    {"progress": f"{completed}/{len(tasks)}", **result},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    _finalize_root(config, identities)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
