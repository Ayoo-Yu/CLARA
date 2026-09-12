from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from baseline_supplement_common import (
    AUTH_CODE,
    CART_CACHE_ROOT,
    CONFIG_PATH,
    PILOT_ROOT,
    RESULT_ROOT,
    REVISION_ROOT,
    S03_ROOT,
    S06_FACT_ROOT,
    S06_ROOT,
    STATE_FIELDS,
    atomic_json,
    compute_event_metrics,
    free_disk_gib,
    load_json,
    sha256_file,
    unit_id,
    validate_frozen_inputs,
    write_parquet,
)


sys.path.insert(0, str(AUTH_CODE))

from baseline_conformal import run_conformal_configuration  # noqa: E402
from baseline_ensemble import select_equal_endpoint_ensemble  # noqa: E402
from baseline_sequential import run_linucb  # noqa: E402
from clara_event_contract import load_frozen_contracts  # noqa: E402


ACTION_CODES = {"Static": 0, "ACI": 1, "AgACI": 2, "EnbPI_RH": 3}
METHOD_CODES = {
    "CARTBestAction": "cart",
    "LinUCB": "linucb",
    "TunedSingleConformal": "tuned",
    "EqualEndpointEnsemble": "ensemble",
}


def decisions_frame(decisions: Any) -> pd.DataFrame:
    return pd.DataFrame([item.__dict__ for item in decisions])


def verify_candidate_bundle(bundle_dir: Path) -> dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    manifest = load_json(manifest_path)
    if (
        manifest.get("manifest_schema") != "S03_NORMALIZED_CANDIDATE_BUNDLE_V1"
        or manifest.get("status") != "COMPLETE_VALIDATED"
    ):
        raise RuntimeError(f"候选包清单身份失配: {bundle_dir}")
    observed = {
        "event_core": sha256_file(bundle_dir / "event_core.parquet"),
        "candidate_intervals": sha256_file(bundle_dir / "candidate_intervals.parquet"),
        "feedback_trace": sha256_file(bundle_dir / "feedback_trace.parquet"),
    }
    expected = {
        "event_core": str(manifest["event_core_sha256"]),
        "candidate_intervals": str(manifest["candidate_intervals_sha256"]),
        "feedback_trace": str(manifest["feedback_trace_sha256"]),
    }
    if observed != expected:
        raise RuntimeError(f"候选包文件哈希失配: {bundle_dir}")
    return {
        "stream_id": bundle_dir.name,
        "manifest_sha256": sha256_file(manifest_path),
        **{f"{key}_sha256": value for key, value in observed.items()},
    }


def load_existing_event_states(
    *,
    zone: str,
    seed: int,
    horizon: int,
    predictors: list[str],
    target_coverages: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """直接读取已验证的 S03 与 S06 工件，不运行 CLARA。"""
    thresholds_path = S06_FACT_ROOT / "fold_width_thresholds.parquet"
    thresholds = pd.read_parquet(thresholds_path)
    thresholds = thresholds[
        thresholds["outer_heldout_zone"].astype(str).eq(zone)
        & thresholds["seed"].astype(int).eq(int(seed))
        & thresholds["target_coverage"].astype(float).isin(target_coverages)
    ].copy()
    threshold_keys = ["predictor", "horizon_group", "target_coverage", "seed"]
    if thresholds.loc[:, threshold_keys].duplicated().any():
        raise RuntimeError(f"{zone} 宽度阈值键重复")

    parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    fact_columns = [
        "event_id",
        "zone_or_farm",
        "predictor",
        "horizon_steps",
        "horizon_hours",
        "seed",
        "target_coverage",
        "issue_timestamp",
        "label_timestamp",
        "label_available_timestamp",
        "horizon_group",
        "ramp_state",
        "raw_width_value",
        "rolling_n",
        "rolling_gap",
        "rolling_towr",
        "rolling_tuwr",
        "rolling_ard",
        "rolling_state",
    ]
    for predictor in predictors:
        stream_id = f"{predictor}-H{int(horizon):02d}-{zone}-S{int(seed)}"
        fact_dir = (
            S06_FACT_ROOT
            / "source_facts"
            / f"predictor={predictor}"
            / f"zone={zone}"
            / f"horizon={int(horizon):02d}"
            / f"seed={int(seed)}"
        )
        fact_path = fact_dir / "facts.parquet"
        fact_manifest_path = fact_dir / "manifest.json"
        fact_manifest = load_json(fact_manifest_path)
        if (
            fact_manifest.get("status") != "COMPLETE_VALIDATED"
            or sha256_file(fact_path) != str(fact_manifest["facts_sha256"])
        ):
            raise RuntimeError(f"S06 发布时状态身份失配: {stream_id}")

        bundle_dir = S03_ROOT / "bundles" / predictor / stream_id
        bundle_identity = verify_candidate_bundle(bundle_dir)
        facts = pd.read_parquet(fact_path, columns=fact_columns)
        facts = facts[
            facts["target_coverage"].astype(float).isin(target_coverages)
        ].copy()
        feedback = pd.read_parquet(
            bundle_dir / "feedback_trace.parquet",
            columns=["event_id", "action", "eligible_by_strict_time_rule"],
        )
        feedback = feedback[
            feedback["event_id"].astype(str).isin(set(facts["event_id"].astype(str)))
        ].copy()
        mature_counts = (
            feedback[feedback["eligible_by_strict_time_rule"].astype(bool)]
            .groupby("event_id", sort=False)["action"]
            .nunique()
        )
        mature_ids = set(mature_counts[mature_counts.eq(len(ACTION_CODES))].index.astype(str))
        facts = facts[facts["event_id"].astype(str).isin(mature_ids)].copy()
        if len(feedback[feedback["event_id"].astype(str).isin(mature_ids)]) != len(facts) * 4:
            raise RuntimeError(f"S03 成熟反馈四动作守恒失败: {stream_id}")

        event_core = pd.read_parquet(
            bundle_dir / "event_core.parquet",
            columns=["event_id", "target", "base_center"],
        )
        facts = facts.merge(
            event_core.rename(
                columns={
                    "target": "target_after_maturity",
                    "base_center": "schedule_proxy",
                }
            ),
            on="event_id",
            how="left",
            validate="one_to_one",
        )
        facts = facts.merge(
            thresholds.loc[
                thresholds["predictor"].astype(str).eq(predictor),
                [*threshold_keys, "raw_width_q33", "raw_width_q67", "heldout_zone_in_threshold"],
            ],
            on=threshold_keys,
            how="left",
            validate="many_to_one",
        )
        if facts[["target_after_maturity", "schedule_proxy", "raw_width_q33", "raw_width_q67"]].isna().any().any():
            raise RuntimeError(f"S03/S06 事件字段关联失败: {stream_id}")
        if facts["heldout_zone_in_threshold"].astype(bool).any():
            raise RuntimeError(f"宽度阈值读取了持出区: {stream_id}")
        width = facts["raw_width_value"].to_numpy(dtype=np.float64)
        q33 = facts["raw_width_q33"].to_numpy(dtype=np.float64)
        q67 = facts["raw_width_q67"].to_numpy(dtype=np.float64)
        facts["raw_width_state"] = np.where(
            width <= q33,
            "narrow",
            np.where(width <= q67, "medium", "wide"),
        )
        if facts["event_id"].duplicated().any():
            raise RuntimeError(f"S06 发布时事件编号重复: {stream_id}")
        parts.append(facts)
        audit_rows.append(
            {
                "stream_id": stream_id,
                "zone": zone,
                "seed": int(seed),
                "horizon": int(horizon),
                "predictor": predictor,
                "event_count": int(len(facts)),
                "source_fact_manifest_sha256": sha256_file(fact_manifest_path),
                "source_facts_sha256": sha256_file(fact_path),
                "width_thresholds_sha256": sha256_file(thresholds_path),
                **bundle_identity,
                "status": "PASS",
            }
        )
    events = pd.concat(parts, ignore_index=True)
    if events["event_id"].duplicated().any():
        raise RuntimeError("新增基线单元事件编号跨预测器重复")
    return events, pd.DataFrame(audit_rows)


def load_candidate_evidence(
    *,
    events: pd.DataFrame,
    zone: str,
    seed: int,
    horizon: int,
    predictors: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    event_ids_all = set(events["event_id"].astype(str))
    for predictor in predictors:
        stream_id = f"{predictor}-H{int(horizon):02d}-{zone}-S{int(seed)}"
        bundle_dir = S03_ROOT / "bundles" / predictor / stream_id
        identity = verify_candidate_bundle(bundle_dir)
        predictor_events = events[events["predictor"].astype(str).eq(predictor)].copy()
        event_ids = set(predictor_events["event_id"].astype(str))
        candidates_all = pd.read_parquet(
            bundle_dir / "candidate_intervals.parquet",
            columns=["event_id", "action", "candidate_lower", "candidate_upper"],
        )
        candidates = candidates_all[candidates_all["event_id"].astype(str).isin(event_ids)].copy()
        feedback_all = pd.read_parquet(
            bundle_dir / "feedback_trace.parquet",
            columns=["event_id", "action", "eligible_by_strict_time_rule"],
        )
        feedback = feedback_all[feedback_all["event_id"].astype(str).isin(event_ids)].copy()
        expected_rows = len(event_ids) * len(ACTION_CODES)
        if (
            len(candidates) != expected_rows
            or len(feedback) != expected_rows
            or candidates[["event_id", "action"]].duplicated().any()
            or feedback[["event_id", "action"]].duplicated().any()
            or not feedback["eligible_by_strict_time_rule"].astype(bool).all()
        ):
            raise RuntimeError(f"{stream_id} 完整事件四动作候选或严格反馈失配")
        parts.append(candidates)
        audit_rows.append(
            {
                **identity,
                "zone": zone,
                "seed": int(seed),
                "horizon": int(horizon),
                "predictor": predictor,
                "event_count": int(len(event_ids)),
                "candidate_row_count": int(len(candidates)),
                "feedback_row_count": int(len(feedback)),
                "status": "PASS",
            }
        )
        del candidates_all, feedback_all, feedback
    candidates = pd.concat(parts, ignore_index=True)
    if (
        set(candidates["event_id"].astype(str)) != event_ids_all
        or len(candidates) != len(events) * len(ACTION_CODES)
    ):
        raise RuntimeError("单元候选事件集合或四动作守恒失败")
    axes = events[
        ["event_id", "target_after_maturity", "schedule_proxy"]
    ].copy()
    candidates = candidates.merge(axes, on="event_id", how="left", validate="many_to_one")
    metrics = compute_event_metrics(
        target=candidates["target_after_maturity"].to_numpy(dtype=np.float64),
        schedule=candidates["schedule_proxy"].to_numpy(dtype=np.float64),
        lower=candidates["candidate_lower"].to_numpy(dtype=np.float64),
        upper=candidates["candidate_upper"].to_numpy(dtype=np.float64),
    )
    candidates["covered_after_maturity"] = metrics["covered"]
    candidates["ERRF_after_maturity"] = metrics["errf"]
    candidates["reserve_after_maturity"] = metrics["reserve_up"] + metrics["reserve_down"]
    candidates["miss_after_maturity"] = metrics["miss_upper"] + metrics["miss_lower"]
    return candidates, pd.DataFrame(audit_rows)


def cart_decisions(
    *,
    events: pd.DataFrame,
    candidates: pd.DataFrame,
    zone: str,
) -> pd.DataFrame:
    cache_path = CART_CACHE_ROOT / f"outer_fold={zone}" / "cart_state_cache.parquet"
    manifest_path = CART_CACHE_ROOT / f"outer_fold={zone}" / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("status") != "PASS" or sha256_file(cache_path) != manifest.get("cache_sha256"):
        raise RuntimeError(f"{zone} CART 状态缓存身份失配")
    cache = pd.read_parquet(cache_path)
    lookup = cache.loc[
        :, [*STATE_FIELDS, "selected_action", "decision_score", "configuration_id"]
    ]
    state_rows = events.loc[:, ["event_id", *STATE_FIELDS]].merge(
        lookup,
        on=list(STATE_FIELDS),
        how="left",
        validate="many_to_one",
    )
    if state_rows[["selected_action", "configuration_id"]].isna().any().any():
        raise RuntimeError(f"{zone} CART 存在未覆盖状态")
    selected = candidates.merge(
        state_rows[["event_id", "selected_action", "decision_score"]],
        left_on=["event_id", "action"],
        right_on=["event_id", "selected_action"],
        how="inner",
        validate="one_to_one",
    )
    if len(selected) != len(events):
        raise RuntimeError(f"{zone} CART 决策数量失配")
    return selected.loc[
        :, ["event_id", "selected_action", "candidate_lower", "candidate_upper", "decision_score"]
    ].rename(
        columns={
            "candidate_lower": "selected_lower",
            "candidate_upper": "selected_upper",
        }
    )


def ensemble_decisions(
    *,
    contracts: Any,
    events: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    baseline_events = events.loc[
        :, ["event_id", "zone_or_farm", "issue_timestamp", *STATE_FIELDS]
    ].copy()
    baseline_candidates = candidates.loc[
        :, ["event_id", "action", "candidate_lower", "candidate_upper"]
    ].copy()
    result = decisions_frame(
        select_equal_endpoint_ensemble(
            contracts=contracts,
            events=baseline_events,
            candidates=baseline_candidates,
        )
    )
    return result.loc[
        :, ["event_id", "selected_action", "selected_lower", "selected_upper", "decision_score"]
    ]


def build_sequential_observations(
    *,
    events: pd.DataFrame,
    candidates: pd.DataFrame,
    theta_id: str,
) -> pd.DataFrame:
    event_fields = [
        "event_id",
        "zone_or_farm",
        "seed",
        "issue_timestamp",
        "label_timestamp",
        "label_available_timestamp",
        "predictor",
        "horizon_group",
        "target_coverage",
        "ramp_state",
        "rolling_state",
        "raw_width_state",
    ]
    joined = candidates.loc[
        :, ["event_id", "action", "ERRF_after_maturity", "covered_after_maturity"]
    ].merge(events.loc[:, event_fields], on="event_id", how="inner", validate="many_to_one")
    joined = joined.rename(
        columns={"ERRF_after_maturity": "errf", "covered_after_maturity": "covered"}
    )
    joined["origin"] = "target"
    joined["theta_id"] = theta_id
    if joined[["errf", "covered"]].isna().any().any():
        raise RuntimeError("LinUCB 顺序观测含未成熟反馈")
    return joined


def linucb_decisions(
    *,
    contracts: Any,
    events: pd.DataFrame,
    candidates: pd.DataFrame,
    registry_row: pd.Series,
    zone: str,
    source_zones: tuple[str, ...],
    theta_id: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    configuration = json.loads(str(registry_row["selected_configuration_json"]))
    decisions: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    observations = build_sequential_observations(
        events=events,
        candidates=candidates,
        theta_id=theta_id,
    )
    for coverage in sorted(events["target_coverage"].astype(float).unique()):
        event_part = events[events["target_coverage"].astype(float).eq(float(coverage))].copy()
        event_ids = set(event_part["event_id"].astype(str))
        candidate_part = candidates[candidates["event_id"].astype(str).isin(event_ids)].copy()
        observation_part = observations[observations["event_id"].astype(str).isin(event_ids)].copy()
        result = run_linucb(
            contracts=contracts,
            evaluation_observations=observation_part,
            candidates=candidate_part.loc[
                :, ["event_id", "action", "candidate_lower", "candidate_upper"]
            ],
            evaluation_zone=zone,
            allowed_origins=("target",),
            forbidden_zones=source_zones,
            exploration_alpha=float(configuration["exploration_alpha"]),
            l2_regularization=float(configuration["l2_regularization"]),
            drain_final_feedback=True,
        )
        frame = decisions_frame(result.decisions)
        feedback = pd.DataFrame(result.feedback)
        if (
            len(frame) != len(event_part)
            or len(feedback) != len(event_part)
            or int(result.audit["future_feedback_violation_count"]) != 0
            or int(result.audit["within_issue_feedback_use_count"]) != 0
        ):
            raise RuntimeError(f"{zone} LinUCB 计数或因果审计失败: coverage={coverage}")
        decisions.append(frame)
        audit_rows.append(
            {
                "target_coverage": float(coverage),
                "event_count": int(len(frame)),
                "feedback_count": int(len(feedback)),
                "future_feedback_violation_count": int(result.audit["future_feedback_violation_count"]),
                "within_issue_feedback_use_count": int(result.audit["within_issue_feedback_use_count"]),
                "configuration_id": str(registry_row["selected_configuration_id"]),
                "status": "PASS",
            }
        )
    combined = pd.concat(decisions, ignore_index=True)
    return combined.loc[
        :, ["event_id", "selected_action", "selected_lower", "selected_upper", "decision_score"]
    ], audit_rows


def tuned_conformal_decisions(
    *,
    contracts: Any,
    events: pd.DataFrame,
    registry_row: pd.Series,
    zone: str,
    seed: int,
    horizon: int,
    predictors: list[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    family = str(registry_row["selected_family"])
    configuration = json.loads(str(registry_row["selected_configuration_json"]))
    expected_configuration_id = str(registry_row["selected_configuration_id"])
    decision_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for predictor in predictors:
        stream_id = f"{predictor}-H{int(horizon):02d}-{zone}-S{int(seed)}"
        base_dir = S03_ROOT / "base" / predictor / stream_id
        manifest_path = base_dir / "manifest.json"
        base_path = base_dir / "base_predictions.parquet"
        manifest = load_json(manifest_path)
        observed_hash = sha256_file(base_path)
        if (
            manifest.get("status") != "COMPLETE_VALIDATED"
            or observed_hash != str(manifest["output_sha256"])
            or str(manifest["predictor"]) != predictor
            or int(manifest["horizon"]) != int(horizon)
            or int(manifest["seed"]) != int(seed)
            or str(manifest["zone"]) != zone
        ):
            raise RuntimeError(f"共形基础预测身份失配: {stream_id}")
        base_predictions = pd.read_parquet(base_path)
        calibration = base_predictions[base_predictions["split"].astype(str).eq("calibration")].copy()
        test_all = base_predictions[base_predictions["split"].astype(str).eq("test")].copy()
        for coverage in sorted(events["target_coverage"].astype(float).unique()):
            event_subset = events[
                events["predictor"].astype(str).eq(predictor)
                & events["target_coverage"].astype(float).eq(float(coverage))
            ].copy()
            issue_set = set(pd.to_datetime(event_subset["issue_timestamp"]))
            selected_test = test_all[
                pd.to_datetime(test_all["issue_timestamp"]).isin(issue_set)
            ].copy()
            base_slice = pd.concat([calibration, selected_test], ignore_index=True)
            if len(selected_test) != len(event_subset) or selected_test["label_available_timestamp"].isna().any():
                raise RuntimeError(f"共形测试切片失配: {stream_id} coverage={coverage}")
            result = run_conformal_configuration(
                contracts=contracts,
                base_predictions=base_slice,
                dataset_id="gefcom2014",
                zone_or_farm=zone,
                predictor=predictor,
                seed=int(seed),
                target_coverage=float(coverage),
                family=family,
                configuration=configuration,
                drain_final_feedback=True,
            )
            frame = decisions_frame(result.decisions)
            if (
                result.configuration.configuration_id != expected_configuration_id
                or len(frame) != len(event_subset)
                or len(result.feedback) != len(event_subset)
                or int(result.audit["future_feedback_violation_count"]) != 0
                or int(result.audit["within_issue_feedback_use_count"]) != 0
                or set(frame["event_id"].astype(str)) != set(event_subset["event_id"].astype(str))
            ):
                raise RuntimeError(f"共形配置、计数或因果审计失败: {stream_id} coverage={coverage}")
            decision_parts.append(frame)
            audit_rows.append(
                {
                    "stream_id": stream_id,
                    "predictor": predictor,
                    "target_coverage": float(coverage),
                    "calibration_rows": int(len(calibration)),
                    "test_rows": int(len(selected_test)),
                    "configuration_id": expected_configuration_id,
                    "future_feedback_violation_count": int(result.audit["future_feedback_violation_count"]),
                    "within_issue_feedback_use_count": int(result.audit["within_issue_feedback_use_count"]),
                    "base_manifest_sha256": sha256_file(manifest_path),
                    "base_predictions_sha256": observed_hash,
                    "status": "PASS",
                }
            )
        del base_predictions, calibration, test_all
    combined = pd.concat(decision_parts, ignore_index=True)
    return combined.loc[
        :, ["event_id", "selected_action", "selected_lower", "selected_upper", "decision_score"]
    ], audit_rows


def build_event_results(
    *,
    events: pd.DataFrame,
    method_decisions: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common = events.loc[
        :,
        [
            "event_id",
            "zone_or_farm",
            "predictor",
            "seed",
            "horizon_steps",
            "target_coverage",
            "issue_timestamp",
            "label_available_timestamp",
            "target_after_maturity",
            "schedule_proxy",
        ],
    ].copy()
    common = common.sort_values("event_id", kind="mergesort").reset_index(drop=True)
    metric_rows: list[dict[str, Any]] = []
    action_parts: list[pd.DataFrame] = []
    for method, decisions in method_decisions.items():
        prefix = METHOD_CODES[method]
        decisions = decisions.copy()
        decisions["event_id"] = decisions["event_id"].astype(str)
        if len(decisions) != len(common) or decisions["event_id"].duplicated().any():
            raise RuntimeError(f"{method} 决策数量或事件唯一性失配")
        selected = common[["event_id"]].merge(
            decisions,
            on="event_id",
            how="left",
            validate="one_to_one",
        )
        if selected[["selected_lower", "selected_upper"]].isna().any().any():
            raise RuntimeError(f"{method} 端点存在缺失")
        metrics = compute_event_metrics(
            target=common["target_after_maturity"].to_numpy(dtype=np.float64),
            schedule=common["schedule_proxy"].to_numpy(dtype=np.float64),
            lower=selected["selected_lower"].to_numpy(dtype=np.float64),
            upper=selected["selected_upper"].to_numpy(dtype=np.float64),
        )
        actions = selected["selected_action"].astype("string")
        common[f"{prefix}_action"] = actions.map(ACTION_CODES).fillna(-1).astype(np.int8)
        common[f"{prefix}_lower"] = metrics["lower"]
        common[f"{prefix}_upper"] = metrics["upper"]
        common[f"{prefix}_covered"] = metrics["covered"]
        common[f"{prefix}_errf"] = metrics["errf"]

        metric_frame = common[
            ["zone_or_farm", "predictor", "seed", "horizon_steps", "target_coverage"]
        ].copy()
        metric_frame["errf"] = metrics["errf"]
        metric_frame["covered"] = metrics["covered"].astype(np.int64)
        metric_frame["width"] = metrics["width"]
        metric_frame["reserve"] = metrics["reserve_up"] + metrics["reserve_down"]
        metric_frame["miss"] = metrics["miss_upper"] + metrics["miss_lower"]
        for keys, part in metric_frame.groupby(
            ["zone_or_farm", "predictor", "seed", "horizon_steps", "target_coverage"],
            sort=True,
        ):
            zone, predictor, seed, horizon, coverage = keys
            event_count = int(len(part))
            covered_count = int(part["covered"].sum())
            metric_rows.append(
                {
                    "zone": str(zone),
                    "predictor": str(predictor),
                    "seed": int(seed),
                    "horizon": int(horizon),
                    "target_coverage": float(coverage),
                    "baseline": method,
                    "event_count": event_count,
                    "sum_errf": float(part["errf"].sum()),
                    "mean_errf": float(part["errf"].mean()),
                    "covered_count": covered_count,
                    "coverage": float(covered_count / event_count),
                    "coverage_gap": float(covered_count / event_count - float(coverage)),
                    "width_total": float(part["width"].sum()),
                    "avg_width": float(part["width"].mean()),
                    "reserve_total": float(part["reserve"].sum()),
                    "miss_total": float(part["miss"].sum()),
                }
            )
        if method in ("CARTBestAction", "LinUCB"):
            action_frame = common[
                ["zone_or_farm", "predictor", "seed", "horizon_steps", "target_coverage"]
            ].copy()
            action_frame["baseline"] = method
            action_frame["action_code"] = common[f"{prefix}_action"].to_numpy(dtype=np.int8)
            action_parts.append(
                action_frame.groupby(
                    [
                        "zone_or_farm",
                        "predictor",
                        "seed",
                        "horizon_steps",
                        "target_coverage",
                        "baseline",
                        "action_code",
                    ],
                    sort=True,
                )
                .size()
                .rename("event_count")
                .reset_index()
            )
    metrics_frame = pd.DataFrame(metric_rows)
    action_frame = pd.concat(action_parts, ignore_index=True)
    return common, metrics_frame, action_frame


def run_unit(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    process = psutil.Process()
    config = load_json(CONFIG_PATH)
    zone = str(task["zone"])
    seed = int(task["seed"])
    horizon = int(task["horizon"])
    mode = str(task["mode"])
    output_root = Path(task["output_root"])
    current_unit_id = unit_id(zone, seed, horizon)
    final_dir = output_root / "units" / current_unit_id
    manifest_path = final_dir / "unit_manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        event_path = final_dir / "event_results.parquet"
        metric_path = final_dir / "unit_metrics.parquet"
        if (
            manifest.get("status") == "PASS"
            and event_path.exists()
            and metric_path.exists()
            and sha256_file(event_path) == manifest.get("event_results_sha256")
            and sha256_file(metric_path) == manifest.get("unit_metrics_sha256")
        ):
            return {**manifest, "resumed": True}
        raise RuntimeError(f"既有单元目录状态不完整: {final_dir}")
    temporary_dir = output_root / "units" / f".{current_unit_id}.tmp.{os.getpid()}"
    if temporary_dir.exists():
        raise RuntimeError(f"临时单元目录已存在: {temporary_dir}")
    temporary_dir.mkdir(parents=True)

    expected_coverages = set(float(value) for value in config["scope"]["target_coverages"])
    events, event_input_audit = load_existing_event_states(
        zone=zone,
        seed=seed,
        horizon=horizon,
        predictors=list(config["scope"]["predictors"]),
        target_coverages=sorted(expected_coverages),
    )
    if (
        events["event_id"].duplicated().any()
        or set(events["predictor"].astype(str)) != set(config["scope"]["predictors"])
        or set(events["target_coverage"].astype(float)) != expected_coverages
        or events["zone_or_farm"].astype(str).ne(zone).any()
        or events["seed"].astype(int).ne(seed).any()
        or events["horizon_steps"].astype(int).ne(horizon).any()
    ):
        raise RuntimeError(f"S03/S06 单元事件范围失配: {current_unit_id}")

    candidates, bundle_audit = load_candidate_evidence(
        events=events,
        zone=zone,
        seed=seed,
        horizon=horizon,
        predictors=list(config["scope"]["predictors"]),
    )
    contracts = load_frozen_contracts()
    registry_all = pd.read_parquet(
        S06_ROOT
        / "results_verified"
        / "s07_execution_registry_v2"
        / "s07_baseline_execution_registry.parquet"
    )
    registry = registry_all[registry_all["outer_heldout_zone"].astype(str).eq(zone)].set_index(
        "baseline_id"
    )
    source_zones = tuple(value for value in config["scope"]["zones"] if value != zone)

    cart = cart_decisions(events=events, candidates=candidates, zone=zone)
    ensemble = ensemble_decisions(contracts=contracts, events=events, candidates=candidates)
    linucb, linucb_audit = linucb_decisions(
        contracts=contracts,
        events=events,
        candidates=candidates,
        registry_row=registry.loc["LinUCB"],
        zone=zone,
        source_zones=source_zones,
        theta_id=str(config["protocol"]["theta_id"]),
    )
    tuned, tuned_audit = tuned_conformal_decisions(
        contracts=contracts,
        events=events,
        registry_row=registry.loc["TunedSingleConformal"],
        zone=zone,
        seed=seed,
        horizon=horizon,
        predictors=list(config["scope"]["predictors"]),
    )
    method_decisions = {
        "CARTBestAction": cart,
        "LinUCB": linucb,
        "TunedSingleConformal": tuned,
        "EqualEndpointEnsemble": ensemble,
    }
    event_results, unit_metrics, action_distribution = build_event_results(
        events=events,
        method_decisions=method_decisions,
    )
    expected_baselines = set(config["scope"]["baselines"])
    if (
        set(unit_metrics["baseline"].astype(str)) != expected_baselines
        or len(event_results) != len(events)
        or len(unit_metrics) != len(config["scope"]["predictors"]) * len(expected_coverages) * 4
    ):
        raise RuntimeError(f"新增基线单元输出范围失配: {current_unit_id}")

    write_parquet(event_results, temporary_dir / "event_results.parquet")
    write_parquet(unit_metrics, temporary_dir / "unit_metrics.parquet")
    action_distribution.to_csv(
        temporary_dir / "action_distribution.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    bundle_audit.to_csv(
        temporary_dir / "bundle_input_audit.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    event_input_audit.to_csv(
        temporary_dir / "event_input_audit.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    pd.DataFrame(linucb_audit).to_csv(
        temporary_dir / "linucb_audit.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    pd.DataFrame(tuned_audit).to_csv(
        temporary_dir / "tuned_conformal_audit.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    peak_rss_gib = float(process.memory_info().rss / (1024**3))
    manifest = {
        "schema": "BASELINE_SUPPLEMENT_UNIT_V1",
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "unit_id": current_unit_id,
        "zone": zone,
        "seed": seed,
        "horizon": horizon,
        "event_count": int(len(events)),
        "candidate_row_count": int(len(candidates)),
        "baseline_count": 4,
        "event_baseline_count": int(len(events) * 4),
        "metric_row_count": int(len(unit_metrics)),
        "linucb_future_feedback_violation_count": int(
            sum(row["future_feedback_violation_count"] for row in linucb_audit)
        ),
        "linucb_within_issue_feedback_use_count": int(
            sum(row["within_issue_feedback_use_count"] for row in linucb_audit)
        ),
        "conformal_future_feedback_violation_count": int(
            sum(row["future_feedback_violation_count"] for row in tuned_audit)
        ),
        "conformal_within_issue_feedback_use_count": int(
            sum(row["within_issue_feedback_use_count"] for row in tuned_audit)
        ),
        "event_source": "S03_CANDIDATE_BUNDLES_PLUS_S06_VALIDATED_SOURCE_FACTS",
        "event_results_sha256": sha256_file(temporary_dir / "event_results.parquet"),
        "unit_metrics_sha256": sha256_file(temporary_dir / "unit_metrics.parquet"),
        "action_distribution_sha256": sha256_file(temporary_dir / "action_distribution.csv"),
        "bundle_input_audit_sha256": sha256_file(temporary_dir / "bundle_input_audit.csv"),
        "event_input_audit_sha256": sha256_file(temporary_dir / "event_input_audit.csv"),
        "linucb_audit_sha256": sha256_file(temporary_dir / "linucb_audit.csv"),
        "tuned_conformal_audit_sha256": sha256_file(temporary_dir / "tuned_conformal_audit.csv"),
        "elapsed_seconds": float(time.perf_counter() - started),
        "peak_rss_gib": peak_rss_gib,
        "aggregate_performance_computed": True,
        "cross_method_ranking_executed": False,
    }
    atomic_json(temporary_dir / "unit_manifest.json", manifest)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir.rename(final_dir)
    del events, candidates, cart, ensemble, linucb, tuned, event_results, unit_metrics
    gc.collect()
    return manifest


def weighted_summary(metrics: pd.DataFrame, group_fields: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, part in metrics.groupby(group_fields, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        event_count = int(part["event_count"].sum())
        covered_count = int(part["covered_count"].sum())
        target_total = float((part["target_coverage"] * part["event_count"]).sum())
        row = {field: value for field, value in zip(group_fields, keys)}
        row.update(
            {
                "event_count": event_count,
                "sum_errf": float(part["sum_errf"].sum()),
                "mean_errf": float(part["sum_errf"].sum() / event_count),
                "covered_count": covered_count,
                "coverage": float(covered_count / event_count),
                "coverage_gap": float((covered_count - target_total) / event_count),
                "avg_width": float(part["width_total"].sum() / event_count),
                "reserve_total": float(part["reserve_total"].sum()),
                "miss_total": float(part["miss_total"].sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_root(output_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    manifest_paths = sorted((output_root / "units").glob("*/unit_manifest.json"))
    manifests = [load_json(path) for path in manifest_paths]
    expected_units = int(config["scope"]["expected_unit_count"])
    if len(manifests) != expected_units or any(item.get("status") != "PASS" for item in manifests):
        raise RuntimeError("新增基线根汇总单元数量或状态失配")
    observed_events = int(sum(item["event_count"] for item in manifests))
    if observed_events != int(config["scope"]["expected_event_count"]):
        raise RuntimeError("新增基线根汇总事件数失配")
    metrics = pd.concat(
        [pd.read_parquet(path.parent / "unit_metrics.parquet") for path in manifest_paths],
        ignore_index=True,
    )
    actions = pd.concat(
        [pd.read_csv(path.parent / "action_distribution.csv") for path in manifest_paths],
        ignore_index=True,
    )
    write_parquet(metrics, output_root / "all_unit_metrics.parquet")
    overall = weighted_summary(metrics, ["baseline"])
    zone_metrics = weighted_summary(metrics, ["baseline", "zone"])
    by_horizon = weighted_summary(metrics, ["baseline", "horizon"])
    by_coverage = weighted_summary(metrics, ["baseline", "target_coverage"])
    by_predictor = weighted_summary(metrics, ["baseline", "predictor"])
    zone_equal = (
        zone_metrics.groupby("baseline", sort=True)
        .agg(
            zone_count=("zone", "nunique"),
            zone_equal_mean_errf=("mean_errf", "mean"),
            zone_equal_mean_coverage=("coverage", "mean"),
            zone_equal_mean_coverage_gap=("coverage_gap", "mean"),
            zone_equal_mean_width=("avg_width", "mean"),
        )
        .reset_index()
    )
    for frame, name in [
        (overall, "overall_event_weighted.csv"),
        (zone_metrics, "zone_metrics.csv"),
        (zone_equal, "zone_equal_summary.csv"),
        (by_horizon, "by_horizon.csv"),
        (by_coverage, "by_coverage.csv"),
        (by_predictor, "by_predictor.csv"),
        (actions, "action_distribution_all_units.csv"),
    ]:
        frame.to_csv(
            output_root / name,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )

    summary = {
        "schema": "BASELINE_SUPPLEMENT_ROOT_V1",
        "status": "PASS_READY_FOR_INDEPENDENT_QA",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "unit_count": len(manifests),
        "event_count": observed_events,
        "event_baseline_count": int(observed_events * 4),
        "baseline_count": 4,
        "causal_violation_count": int(
            sum(
                item["linucb_future_feedback_violation_count"]
                + item["linucb_within_issue_feedback_use_count"]
                + item["conformal_future_feedback_violation_count"]
                + item["conformal_within_issue_feedback_use_count"]
                for item in manifests
            )
        ),
        "corrected_clara_comparison_executed": False,
        "adverse_result_gate_triggered": False,
        "adverse_baselines": [],
        "next_stage": "INDEPENDENT_FULL_QA",
        "artifacts": {
            "all_unit_metrics": sha256_file(output_root / "all_unit_metrics.parquet"),
            "overall_event_weighted": sha256_file(output_root / "overall_event_weighted.csv"),
            "zone_metrics": sha256_file(output_root / "zone_metrics.csv"),
            "zone_equal_summary": sha256_file(output_root / "zone_equal_summary.csv"),
        },
    }
    atomic_json(output_root / "root_manifest.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    config = load_json(CONFIG_PATH)
    output_root = PILOT_ROOT if args.mode == "pilot" else RESULT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    input_audit = validate_frozen_inputs(config)
    input_audit.to_csv(
        output_root / "input_hash_audit.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    cart_root_manifest = CART_CACHE_ROOT / "root_manifest.json"
    if not cart_root_manifest.exists() or load_json(cart_root_manifest).get("status") != "PASS":
        raise RuntimeError("CART 十折状态缓存尚未通过")
    if free_disk_gib() < float(config["execution"]["minimum_free_disk_gib"]):
        raise RuntimeError("新增基线运行前磁盘安全门失败")

    if args.mode == "pilot":
        tasks = [
            {
                "zone": "zone1",
                "seed": 0,
                "horizon": 24,
                "mode": "pilot",
                "output_root": str(output_root),
            }
        ]
        worker_count = 1
    else:
        tasks = [
            {
                "zone": str(zone),
                "seed": int(seed),
                "horizon": int(horizon),
                "mode": "full",
                "output_root": str(output_root),
            }
            for zone in config["scope"]["zones"]
            for seed in config["scope"]["seeds"]
            for horizon in config["scope"]["horizons"]
        ]
        worker_count = int(args.workers or config["execution"]["maximum_workers"])

    started = time.perf_counter()
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if worker_count == 1:
        for task in tasks:
            try:
                completed.append(run_unit(task))
            except Exception as error:
                failures.append(
                    {
                        "task": task,
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
                break
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(run_unit, task): task for task in tasks}
            for future in as_completed(future_map):
                task = future_map[future]
                try:
                    completed.append(future.result())
                except Exception as error:
                    failures.append(
                        {
                            "task": task,
                            "error_type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    for pending in future_map:
                        pending.cancel()
                    break
                atomic_json(
                    output_root / "checkpoint.json",
                    {
                        "status": "RUNNING",
                        "mode": args.mode,
                        "worker_count": worker_count,
                        "unit_count": len(tasks),
                        "completed_unit_count": len(completed),
                        "failed_unit_count": len(failures),
                        "elapsed_seconds": float(time.perf_counter() - started),
                        "completed_units": sorted(item["unit_id"] for item in completed),
                    },
                )

    if failures:
        with (output_root / "exception_ledger.jsonl").open("a", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
        atomic_json(
            output_root / "checkpoint.json",
            {
                "status": "FAILED_STOP_REQUIRED",
                "mode": args.mode,
                "completed_unit_count": len(completed),
                "failed_unit_count": len(failures),
                "elapsed_seconds": float(time.perf_counter() - started),
                "failures": failures,
            },
        )
        raise RuntimeError(f"新增基线运行失败: {failures[0]['message']}")

    if args.mode == "pilot":
        summary = {
            "schema": "BASELINE_SUPPLEMENT_PILOT_V1",
            "status": "PASS_READY_FOR_REFERENCE_QA",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "unit_count": 1,
            "event_count": int(completed[0]["event_count"]),
            "event_baseline_count": int(completed[0]["event_baseline_count"]),
            "causal_violation_count": int(
                completed[0]["linucb_future_feedback_violation_count"]
                + completed[0]["linucb_within_issue_feedback_use_count"]
                + completed[0]["conformal_future_feedback_violation_count"]
                + completed[0]["conformal_within_issue_feedback_use_count"]
            ),
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        atomic_json(output_root / "root_manifest.json", summary)
    else:
        summary = aggregate_root(output_root, config)
    atomic_json(
        output_root / "checkpoint.json",
        {
            "status": summary["status"],
            "mode": args.mode,
            "completed_unit_count": len(completed),
            "failed_unit_count": 0,
            "elapsed_seconds": float(time.perf_counter() - started),
            "root_manifest_sha256": sha256_file(output_root / "root_manifest.json"),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
