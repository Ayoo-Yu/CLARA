from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from price_sensitivity_common import (
    REVISION_ROOT,
    SENSITIVITY_ROOT,
    append_jsonl,
    atomic_json,
    authoritative_git_identity,
    ensure_disk_gate,
    load_config,
    sha256_file,
    utc_now,
    validate_frozen_inputs,
)


ACTIONS = ("Static", "ACI", "AgACI", "EnbPI_RH")
STATE_FIELDS = [
    "predictor",
    "horizon_group",
    "target_coverage",
    "ramp_state",
    "rolling_state",
    "raw_width_state",
]
AUTH_CODE = REVISION_ROOT / "_权威代码" / "code"
S07_ROOT = REVISION_ROOT / "07_GEFCom完整主实验"
S06_ROOT = REVISION_ROOT / "06_基线实现与训练区调参"
INDEX_CONTRACT_PATH = S07_ROOT / "configs" / "s07_landscape_index_contract_v1.json"
OLD_INDEX_ROOT = S07_ROOT / "results_raw" / "full_landscape_index_v1"
COMPONENT_ROOT = SENSITIVITY_ROOT / "results_raw" / "price_component_index_v1"
WIDTH_THRESHOLDS_PATH = (
    S06_ROOT
    / "results_raw"
    / "nested_source_selection_v1"
    / "fold_width_thresholds.parquet"
)
CORRECTED_CACHE_ROOT = (
    REVISION_ROOT
    / "07B_CLARA经验阈值修正重算与新旧对比"
    / "results_raw"
    / "corrected_decision_cache_v1"
)
OUTPUT_ROOT = SENSITIVITY_ROOT / "results_raw" / "price_decision_cache_v1"

sys.path.insert(0, str(AUTH_CODE))
sys.path.insert(0, str(S07_ROOT / "scripts"))

from clara_event_contract import load_frozen_contracts  # noqa: E402
import s07_decision_cache_accelerator as accelerator  # noqa: E402


def unit_root(heldout_zone: str, seed: int) -> Path:
    return OUTPUT_ROOT / f"outer_fold={heldout_zone}" / f"seed={int(seed)}"


def source_zones(config: dict[str, Any], heldout_zone: str) -> list[str]:
    return [
        str(zone)
        for zone in config["scope"]["zones"]
        if str(zone) != str(heldout_zone)
    ]


def component_unit_valid(zone: str, seed: int, config_sha256: str) -> tuple[Path, Path]:
    root = COMPONENT_ROOT / f"zone={zone}" / f"seed={int(seed)}"
    manifest_path = root / "unit_manifest.json"
    component_path = root / "price_component_index.parquet"
    if not manifest_path.is_file() or not component_path.is_file():
        raise RuntimeError(f"成本分量索引缺失: {zone}/seed{seed}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "PASS"
        or manifest.get("config_sha256") != config_sha256
        or str(manifest.get("component_index_sha256")) != sha256_file(component_path)
    ):
        raise RuntimeError(f"成本分量索引身份失配: {zone}/seed{seed}")
    return component_path, manifest_path


def aggregate_fold_components(
    *,
    zones: Sequence[str],
    seed: int,
    q33: np.ndarray,
    q67: np.ndarray,
    full_state_count: int,
    config_sha256: str,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    ordered_zones = tuple(sorted(str(zone) for zone in zones))
    zone_count = len(ordered_zones)
    action_count = len(ACTIONS)
    maximum_time = np.iinfo(np.int64).max
    minimum_time = np.iinfo(np.int64).min
    arrays = {
        "event_count": np.zeros((zone_count, full_state_count), dtype=np.int64),
        "issue_min_us": np.full((zone_count, full_state_count), maximum_time, dtype=np.int64),
        "issue_max_us": np.full((zone_count, full_state_count), minimum_time, dtype=np.int64),
        "guardrail_count": np.zeros((zone_count, full_state_count), dtype=np.int64),
        "guardrail_issue_min_us": np.full(
            (zone_count, full_state_count), maximum_time, dtype=np.int64
        ),
        "guardrail_issue_max_us": np.full(
            (zone_count, full_state_count), minimum_time, dtype=np.int64
        ),
        "capacity_sum": np.zeros(
            (action_count, zone_count, full_state_count), dtype=np.float64
        ),
        "miss_sum": np.zeros(
            (action_count, zone_count, full_state_count), dtype=np.float64
        ),
        "covered_sum": np.zeros(
            (action_count, zone_count, full_state_count), dtype=np.float64
        ),
        "guardrail_covered_sum": np.zeros(
            (action_count, zone_count, full_state_count), dtype=np.float64
        ),
        "tuwr_sum": np.zeros(
            (action_count, zone_count, full_state_count), dtype=np.float64
        ),
        "ard_sum": np.zeros(
            (action_count, zone_count, full_state_count), dtype=np.float64
        ),
    }
    audit_rows: list[dict[str, Any]] = []
    old_columns = [
        "base_state_code",
        "issue_timestamp",
        "raw_width_value",
        *(f"{action}__covered" for action in ACTIONS),
        *(f"{action}__tuwr_indicator" for action in ACTIONS),
        *(f"{action}__ard_value" for action in ACTIONS),
    ]
    component_columns = [
        "base_state_code",
        "issue_timestamp",
        *(f"{action}__capacity_exposure" for action in ACTIONS),
        *(f"{action}__miss_exposure" for action in ACTIONS),
    ]
    for zone_index, zone in enumerate(ordered_zones):
        component_path, component_manifest_path = component_unit_valid(
            zone, int(seed), config_sha256
        )
        old_root = OLD_INDEX_ROOT / f"zone={zone}" / f"seed={int(seed)}"
        old_path = old_root / "source_landscape_index.parquet"
        old_manifest_path = old_root / "unit_manifest.json"
        old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
        if sha256_file(old_path) != str(old_manifest["index_sha256"]):
            raise RuntimeError(f"S07 景观索引哈希失配: {zone}/seed{seed}")
        old_batches = pq.ParquetFile(old_path).iter_batches(
            columns=old_columns, batch_size=262144
        )
        component_batches = pq.ParquetFile(component_path).iter_batches(
            columns=component_columns, batch_size=262144
        )
        input_rows = 0
        batch_count = 0
        while True:
            try:
                old_batch = next(old_batches)
            except StopIteration:
                old_batch = None
            try:
                component_batch = next(component_batches)
            except StopIteration:
                component_batch = None
            if old_batch is None or component_batch is None:
                if old_batch is not None or component_batch is not None:
                    raise RuntimeError(f"索引批次终止位置失配: {zone}/seed{seed}")
                break
            if old_batch.num_rows != component_batch.num_rows:
                raise RuntimeError(f"索引批次行数失配: {zone}/seed{seed}")
            old_names = old_batch.schema.names
            component_names = component_batch.schema.names
            base = old_batch.column(old_names.index("base_state_code")).to_numpy(
                zero_copy_only=False
            ).astype(np.int64)
            component_base = component_batch.column(
                component_names.index("base_state_code")
            ).to_numpy(zero_copy_only=False).astype(np.int64)
            old_times = (
                old_batch.column(old_names.index("issue_timestamp"))
                .cast(pa.int64())
                .to_numpy(zero_copy_only=False)
                .astype(np.int64)
            )
            component_times = (
                component_batch.column(component_names.index("issue_timestamp"))
                .cast(pa.int64())
                .to_numpy(zero_copy_only=False)
                .astype(np.int64)
            )
            if not np.array_equal(base, component_base) or not np.array_equal(
                old_times, component_times
            ):
                raise RuntimeError(f"成本分量索引与景观索引逐批对齐失败: {zone}/seed{seed}")
            widths = old_batch.column(old_names.index("raw_width_value")).to_numpy(
                zero_copy_only=False
            ).astype(np.float64)
            width_code = np.where(
                widths <= q33[base],
                0,
                np.where(widths <= q67[base], 1, 2),
            ).astype(np.int64)
            full_code = base * 3 + width_code
            arrays["event_count"][zone_index] += np.bincount(
                full_code, minlength=full_state_count
            ).astype(np.int64)
            np.minimum.at(arrays["issue_min_us"][zone_index], full_code, old_times)
            np.maximum.at(arrays["issue_max_us"][zone_index], full_code, old_times)
            reference_mask: np.ndarray | None = None
            for action_index, action in enumerate(ACTIONS):
                capacity = component_batch.column(
                    component_names.index(f"{action}__capacity_exposure")
                ).to_numpy(zero_copy_only=False).astype(np.float64)
                miss = component_batch.column(
                    component_names.index(f"{action}__miss_exposure")
                ).to_numpy(zero_copy_only=False).astype(np.float64)
                covered = old_batch.column(
                    old_names.index(f"{action}__covered")
                ).to_numpy(zero_copy_only=False).astype(np.float64)
                tuwr = old_batch.column(
                    old_names.index(f"{action}__tuwr_indicator")
                ).to_numpy(zero_copy_only=False).astype(np.float64)
                ard = old_batch.column(
                    old_names.index(f"{action}__ard_value")
                ).to_numpy(zero_copy_only=False).astype(np.float64)
                guardrail_mask = np.isfinite(tuwr) & np.isfinite(ard)
                if reference_mask is None:
                    reference_mask = guardrail_mask
                    if guardrail_mask.any():
                        guardrail_codes = full_code[guardrail_mask]
                        guardrail_times = old_times[guardrail_mask]
                        arrays["guardrail_count"][zone_index] += np.bincount(
                            guardrail_codes, minlength=full_state_count
                        ).astype(np.int64)
                        np.minimum.at(
                            arrays["guardrail_issue_min_us"][zone_index],
                            guardrail_codes,
                            guardrail_times,
                        )
                        np.maximum.at(
                            arrays["guardrail_issue_max_us"][zone_index],
                            guardrail_codes,
                            guardrail_times,
                        )
                elif not np.array_equal(reference_mask, guardrail_mask):
                    raise RuntimeError(f"四动作可靠性缺失模式失配: {zone}/seed{seed}")
                arrays["capacity_sum"][action_index, zone_index] += np.bincount(
                    full_code, weights=capacity, minlength=full_state_count
                )
                arrays["miss_sum"][action_index, zone_index] += np.bincount(
                    full_code, weights=miss, minlength=full_state_count
                )
                arrays["covered_sum"][action_index, zone_index] += np.bincount(
                    full_code, weights=covered, minlength=full_state_count
                )
                if guardrail_mask.any():
                    guardrail_codes = full_code[guardrail_mask]
                    arrays["guardrail_covered_sum"][action_index, zone_index] += np.bincount(
                        guardrail_codes,
                        weights=covered[guardrail_mask],
                        minlength=full_state_count,
                    )
                    arrays["tuwr_sum"][action_index, zone_index] += np.bincount(
                        guardrail_codes,
                        weights=tuwr[guardrail_mask],
                        minlength=full_state_count,
                    )
                    arrays["ard_sum"][action_index, zone_index] += np.bincount(
                        guardrail_codes,
                        weights=ard[guardrail_mask],
                        minlength=full_state_count,
                    )
            input_rows += len(base)
            batch_count += 1
        aggregated_rows = int(arrays["event_count"][zone_index].sum())
        audit_rows.append(
            {
                "source_zone": zone,
                "seed": int(seed),
                "input_event_count": input_rows,
                "aggregated_event_count": aggregated_rows,
                "batch_count": batch_count,
                "old_index_sha256": sha256_file(old_path),
                "component_index_sha256": sha256_file(component_path),
                "component_unit_manifest_sha256": sha256_file(component_manifest_path),
                "status": "PASS" if input_rows == aggregated_rows else "FAIL",
            }
        )
    return arrays, pd.DataFrame(audit_rows)


def arrays_for_price(
    base_arrays: dict[str, np.ndarray], capacity_weight: float, miss_weight: float
) -> dict[str, np.ndarray]:
    result = {
        key: value
        for key, value in base_arrays.items()
        if key not in {"capacity_sum", "miss_sum"}
    }
    result["errf_sum"] = (
        float(capacity_weight) * base_arrays["capacity_sum"]
        + float(miss_weight) * base_arrays["miss_sum"]
    )
    return result


def finalize_empirical(cache: pd.DataFrame, contracts: Any) -> pd.DataFrame:
    thresholds = contracts.protocol["guardrail_contract"]["default_thresholds"]
    tolerance = float(contracts.protocol["selection_contract"]["score_tie_tolerance"])
    output = cache.copy()
    reasons: list[str] = []
    passes: list[bool] = []
    for row in output.itertuples(index=False):
        current: list[str] = []
        if float(row.coverage_point) < float(row.target_coverage) - float(
            thresholds["coverage_shortfall_epsilon"]
        ):
            current.append("coverage_point")
        if not bool(row.cold_start_guardrail):
            if pd.isna(row.tuwr_point) or float(row.tuwr_point) > float(
                thresholds["tuwr_upper"]
            ):
                current.append("tuwr_point")
            if pd.isna(row.ard_point) or float(row.ard_point) > float(
                thresholds["ard_upper"]
            ):
                current.append("ard_point")
        reasons.append(";".join(current))
        passes.append(not current)
    output["guardrail_failure_reasons"] = reasons
    output["guardrail_pass"] = passes
    decisions: list[dict[str, Any]] = []
    for full_state_code, rows in output.groupby("full_state_code", sort=True):
        ordered = rows.sort_values("action_order", kind="mergesort")
        safe = ordered[ordered["guardrail_pass"]]
        empty = safe.empty
        candidates = ordered.copy() if empty else safe.copy()
        minimum_risk = float(candidates["risk_score"].min())
        candidates = candidates[candidates["risk_score"] <= minimum_risk + tolerance]
        maximum_coverage = float(candidates["coverage_point"].max())
        candidates = candidates[
            candidates["coverage_point"] >= maximum_coverage - tolerance
        ]
        if candidates["tuwr_point"].notna().all():
            minimum_tuwr = float(candidates["tuwr_point"].min())
            candidates = candidates[
                candidates["tuwr_point"] <= minimum_tuwr + tolerance
            ]
        elif candidates["tuwr_point"].notna().any():
            raise RuntimeError("并列候选 TUWR 可用性不一致")
        if candidates["ard_point"].notna().all():
            minimum_ard = float(candidates["ard_point"].min())
            candidates = candidates[
                candidates["ard_point"] <= minimum_ard + tolerance
            ]
        elif candidates["ard_point"].notna().any():
            raise RuntimeError("并列候选 ARD 可用性不一致")
        selected = candidates.sort_values("action_order", kind="mergesort").iloc[0]
        cold = bool(ordered["cold_start_guardrail"].iloc[0])
        decisions.append(
            {
                "full_state_code": int(full_state_code),
                "guardrail_empty_fallback": bool(empty),
                "selected_action": str(selected["action"]),
                "decision_evidence_level": (
                    "COLD_START_COVERAGE_ONLY" if cold else "EMPIRICAL_FALLBACK"
                ),
                "decision_bound_method": "EMPIRICAL_FALLBACK",
            }
        )
    output = output.merge(
        pd.DataFrame(decisions), on="full_state_code", how="left", validate="many_to_one"
    )
    output["coverage_lcb"] = output["coverage_point"]
    output["tuwr_ucb"] = output["tuwr_point"]
    output["ard_ucb"] = output["ard_point"]
    output["action_evidence_level"] = np.where(
        output["cold_start_guardrail"],
        "COLD_START_COVERAGE_ONLY",
        "EMPIRICAL_FALLBACK",
    )
    output["action_bound_method"] = "EMPIRICAL_FALLBACK"
    return output


def base_cache_regression(
    candidate: pd.DataFrame,
    *,
    heldout_zone: str,
    seed: int,
    tolerance: float,
) -> dict[str, Any]:
    reference_path = (
        CORRECTED_CACHE_ROOT
        / f"outer_fold={heldout_zone}"
        / f"seed={int(seed)}"
        / "corrected_landscape_decision_cache.parquet"
    )
    reference_manifest_path = reference_path.parent / "unit_manifest.json"
    reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    if sha256_file(reference_path) != str(reference_manifest["output_cache_sha256"]):
        raise RuntimeError(f"07B 修正缓存哈希失配: {heldout_zone}/seed{seed}")
    reference = pd.read_parquet(reference_path)
    left = candidate.sort_values(STATE_FIELDS + ["action"], kind="mergesort").reset_index(
        drop=True
    )
    right = reference.sort_values(STATE_FIELDS + ["action"], kind="mergesort").reset_index(
        drop=True
    )
    if len(left) != len(right) or len(left) != 26400:
        raise RuntimeError(f"基准价格缓存回归行数失配: {heldout_zone}/seed{seed}")
    identity_mismatch = 0
    for field in STATE_FIELDS + ["action"]:
        if pd.api.types.is_numeric_dtype(right[field]):
            identity_mismatch += int(
                (~np.isclose(
                    left[field].to_numpy(dtype=np.float64),
                    right[field].to_numpy(dtype=np.float64),
                    atol=1e-12,
                    rtol=0.0,
                    equal_nan=True,
                )).sum()
            )
        else:
            identity_mismatch += int(
                left[field].astype(str).ne(right[field].astype(str)).sum()
            )
    categorical_fields = [
        "support_backoff_level",
        "support_backoff_name",
        "support_n",
        "support_source_zones",
        "guardrail_pass",
        "guardrail_empty_fallback",
        "selected_action",
    ]
    categorical_mismatch: dict[str, int] = {}
    for field in categorical_fields:
        categorical_mismatch[field] = int(
            left[field].astype(str).ne(right[field].astype(str)).sum()
        )
    numeric_fields = [
        "risk_mean",
        "risk_parent_mean",
        "risk_shrunken_mean",
        "risk_zone_cluster_se",
        "risk_score",
        "coverage_point",
        "tuwr_point",
        "ard_point",
    ]
    maximum_numeric_error: dict[str, float] = {}
    numeric_mismatch: dict[str, int] = {}
    for field in numeric_fields:
        left_values = left[field].to_numpy(dtype=np.float64)
        right_values = right[field].to_numpy(dtype=np.float64)
        both_nan = np.isnan(left_values) & np.isnan(right_values)
        difference = np.abs(left_values - right_values)
        difference[both_nan] = 0.0
        invalid = np.isnan(difference) & ~both_nan
        maximum_numeric_error[field] = (
            math.inf if invalid.any() else float(np.nanmax(difference))
        )
        numeric_mismatch[field] = int((difference > tolerance).sum() + invalid.sum())
    failure_count = (
        identity_mismatch
        + sum(categorical_mismatch.values())
        + sum(numeric_mismatch.values())
    )
    if failure_count:
        raise RuntimeError(
            f"基准价格缓存回归失败: identity={identity_mismatch}, "
            f"categorical={categorical_mismatch}, numeric={numeric_mismatch}"
        )
    return {
        "status": "PASS",
        "reference_cache_sha256": sha256_file(reference_path),
        "reference_manifest_sha256": sha256_file(reference_manifest_path),
        "row_count": len(left),
        "state_count": int(left.groupby(STATE_FIELDS, dropna=False).ngroups),
        "identity_mismatch_count": identity_mismatch,
        "categorical_mismatch": categorical_mismatch,
        "numeric_mismatch": numeric_mismatch,
        "maximum_numeric_error": maximum_numeric_error,
    }


def completed_unit(
    heldout_zone: str, seed: int, config_sha256: str, price_ids: Sequence[str]
) -> dict[str, Any] | None:
    root = unit_root(heldout_zone, seed)
    manifest_path = root / "unit_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "PASS"
        or manifest.get("config_sha256") != config_sha256
        or set(manifest.get("price_ids", [])) != set(price_ids)
    ):
        return None
    for price_id in price_ids:
        path = root / f"price_id={price_id}" / "price_decision_cache.parquet"
        expected = manifest["artifacts"][price_id]["sha256"]
        if not path.is_file() or sha256_file(path) != expected:
            return None
    return {**manifest, "resume_status": "REUSED"}


def build_unit(task: dict[str, Any]) -> dict[str, Any]:
    heldout_zone = str(task["heldout_zone"])
    seed = int(task["seed"])
    config = task["config"]
    config_sha256 = str(task["config_sha256"])
    price_grid = config["price_grid"]
    price_ids = [str(item["price_id"]) for item in price_grid]
    existing = completed_unit(heldout_zone, seed, config_sha256, price_ids)
    if existing is not None:
        return existing
    root = unit_root(heldout_zone, seed)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"价格缓存单元存在未封存输出: {heldout_zone}/seed{seed}")
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "checkpoint.json"
    exception_path = root / "exception_ledger.jsonl"
    started = time.perf_counter()
    try:
        ensure_disk_gate(config, OUTPUT_ROOT)
        contracts = load_frozen_contracts()
        expected_protocol = config["causal_event_protocol"]
        if (
            contracts.protocol_id != expected_protocol["protocol_id"]
            or contracts.protocol_version != expected_protocol["protocol_version"]
            or contracts.protocol_sha256 != expected_protocol["sha256"]
        ):
            raise RuntimeError("因果事件协议身份失配")
        index_contract = json.loads(INDEX_CONTRACT_PATH.read_text(encoding="utf-8"))
        base_states, states = accelerator.state_frames(index_contract, contracts.protocol)
        thresholds = pd.read_parquet(WIDTH_THRESHOLDS_PATH)
        q33, q67 = accelerator.threshold_vectors(
            thresholds,
            base_states,
            heldout_zone=heldout_zone,
            seed=seed,
        )
        zones = source_zones(config, heldout_zone)
        arrays, source_audit = aggregate_fold_components(
            zones=zones,
            seed=seed,
            q33=q33,
            q67=q67,
            full_state_count=len(states),
            config_sha256=config_sha256,
        )
        if not source_audit["status"].eq("PASS").all():
            raise RuntimeError(f"源区事件聚合失败: {heldout_zone}/seed{seed}")
        mappings, group_frames = accelerator.factor_level_mappings(states, contracts)
        base_arrays = arrays_for_price(arrays, 3.56, 20.0)
        base_levels = accelerator.build_level_aggregates(
            base_arrays, mappings, group_frames
        )
        profile = accelerator.select_support_levels(
            states, contracts, mappings, base_levels
        )
        if len(profile) != int(config["scope"]["expected_query_state_count_per_fold_seed"]):
            raise RuntimeError(f"查询状态数量失配: {heldout_zone}/seed{seed}")
        source_audit_path = root / "source_unit_audit.csv"
        source_audit.to_csv(
            source_audit_path, index=False, encoding="utf-8", lineterminator="\n"
        )
        atomic_json(
            checkpoint_path,
            {
                "stage": "SOURCE_AGGREGATION_COMPLETE",
                "heldout_zone": heldout_zone,
                "seed": seed,
                "source_event_count": int(arrays["event_count"].sum()),
                "config_sha256": config_sha256,
                "timestamp_utc": utc_now(),
            },
        )

        artifacts: dict[str, Any] = {}
        base_regression: dict[str, Any] | None = None
        action_change_rows: list[dict[str, Any]] = []
        base_state_actions: pd.DataFrame | None = None
        for price in price_grid:
            price_id = str(price["price_id"])
            theta = [float(value) for value in price["theta"]]
            if theta[0] != theta[1] or theta[2] != theta[3]:
                raise RuntimeError(f"当前分量索引要求方向对称价格: {price_id}")
            price_arrays = arrays_for_price(arrays, theta[0], theta[2])
            levels = accelerator.build_level_aggregates(
                price_arrays, mappings, group_frames
            )
            evidence = accelerator.compute_cache_evidence(
                profile=profile,
                contracts=contracts,
                mappings=mappings,
                levels=levels,
            )
            cache = finalize_empirical(evidence, contracts)
            if len(cache) != int(config["scope"]["expected_action_row_count_per_fold_seed"]):
                raise RuntimeError(f"价格缓存动作行数失配: {heldout_zone}/seed{seed}/{price_id}")
            state_actions = cache.drop_duplicates(STATE_FIELDS)[
                STATE_FIELDS + ["selected_action"]
            ].copy()
            if len(state_actions) != 6600:
                raise RuntimeError(f"价格缓存状态动作数量失配: {heldout_zone}/seed{seed}/{price_id}")
            if bool(price.get("base_submission_price", False)):
                base_regression = base_cache_regression(
                    cache,
                    heldout_zone=heldout_zone,
                    seed=seed,
                    tolerance=float(config["execution"]["numeric_tolerance"]),
                )
                base_state_actions = state_actions.rename(
                    columns={"selected_action": "base_selected_action"}
                )
            price_root = root / f"price_id={price_id}"
            price_root.mkdir(parents=True, exist_ok=False)
            cache.insert(0, "miss_to_capacity_ratio", float(price["miss_to_capacity_ratio"]))
            cache.insert(0, "miss_weight", theta[2])
            cache.insert(0, "capacity_weight", theta[0])
            cache.insert(0, "price_id", price_id)
            cache_path = price_root / "price_decision_cache.parquet"
            cache.to_parquet(cache_path, index=False, compression="zstd")
            artifacts[price_id] = {
                "relative_path": cache_path.relative_to(root).as_posix(),
                "sha256": sha256_file(cache_path),
                "bytes": cache_path.stat().st_size,
                "rows": len(cache),
                "state_count": len(state_actions),
                "guardrail_empty_fallback_state_count": int(
                    state_actions.merge(
                        cache.drop_duplicates(STATE_FIELDS)[
                            STATE_FIELDS + ["guardrail_empty_fallback"]
                        ],
                        on=STATE_FIELDS,
                        how="left",
                        validate="one_to_one",
                    )["guardrail_empty_fallback"].sum()
                ),
            }
            atomic_json(
                checkpoint_path,
                {
                    "stage": "PRICE_CACHE_COMPLETE",
                    "heldout_zone": heldout_zone,
                    "seed": seed,
                    "price_id": price_id,
                    "completed_price_count": len(artifacts),
                    "config_sha256": config_sha256,
                    "timestamp_utc": utc_now(),
                },
            )
        if base_regression is None or base_state_actions is None:
            raise RuntimeError("基准价格配置缺失")
        for price in price_grid:
            price_id = str(price["price_id"])
            path = root / f"price_id={price_id}" / "price_decision_cache.parquet"
            current = pd.read_parquet(path).drop_duplicates(STATE_FIELDS)[
                STATE_FIELDS + ["selected_action"]
            ]
            compared = current.merge(
                base_state_actions,
                on=STATE_FIELDS,
                how="left",
                validate="one_to_one",
            )
            action_change_rows.append(
                {
                    "price_id": price_id,
                    "miss_to_capacity_ratio": float(price["miss_to_capacity_ratio"]),
                    "state_count": len(compared),
                    "changed_from_base_state_count": int(
                        compared["selected_action"].astype(str).ne(
                            compared["base_selected_action"].astype(str)
                        ).sum()
                    ),
                }
            )
        action_change_path = root / "action_change_from_base.csv"
        pd.DataFrame(action_change_rows).to_csv(
            action_change_path, index=False, encoding="utf-8", lineterminator="\n"
        )
        base_regression_path = root / "base_price_regression.json"
        atomic_json(base_regression_path, base_regression)
        append_jsonl(
            exception_path,
            {
                "timestamp_utc": utc_now(),
                "record_type": "EXCEPTION_SUMMARY",
                "exception_count": 0,
                "heldout_zone": heldout_zone,
                "seed": seed,
            },
        )
        manifest = {
            "schema": "S08_PRICE_DECISION_CACHE_UNIT_V1",
            "status": "PASS",
            "generated_at_utc": utc_now(),
            "config_sha256": config_sha256,
            "heldout_zone": heldout_zone,
            "seed": seed,
            "source_zones": zones,
            "source_event_count": int(arrays["event_count"].sum()),
            "query_state_count": len(profile),
            "price_ids": price_ids,
            "artifacts": artifacts,
            "source_unit_audit_sha256": sha256_file(source_audit_path),
            "action_change_from_base_sha256": sha256_file(action_change_path),
            "base_price_regression_sha256": sha256_file(base_regression_path),
            "base_price_regression": base_regression,
            "elapsed_seconds": time.perf_counter() - started,
            "heldout_target_performance_read": False,
            "performance_comparison_executed": False,
        }
        atomic_json(root / "unit_manifest.json", manifest)
        atomic_json(
            checkpoint_path,
            {
                "stage": "COMPLETE",
                "heldout_zone": heldout_zone,
                "seed": seed,
                "config_sha256": config_sha256,
                "unit_manifest_sha256": sha256_file(root / "unit_manifest.json"),
                "timestamp_utc": utc_now(),
            },
        )
        return {**manifest, "resume_status": "BUILT"}
    except Exception as error:
        append_jsonl(
            exception_path,
            {
                "timestamp_utc": utc_now(),
                "record_type": "EXCEPTION",
                "heldout_zone": heldout_zone,
                "seed": seed,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        atomic_json(
            checkpoint_path,
            {
                "stage": "FAILED",
                "heldout_zone": heldout_zone,
                "seed": seed,
                "config_sha256": config_sha256,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "timestamp_utc": utc_now(),
            },
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zones", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--workers", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_sha256 = load_config()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    input_audit = validate_frozen_inputs(config)
    git_identity = authoritative_git_identity()
    if not git_identity["clean"] or git_identity["commit"] != "d02e477184991d521a7eb0999d0e619c5e68c9f4":
        raise RuntimeError(f"权威代码身份失配: {git_identity}")
    free_gib = ensure_disk_gate(config, OUTPUT_ROOT)
    zones = [str(value) for value in (args.zones or config["scope"]["zones"])]
    seeds = [int(value) for value in (args.seeds or config["scope"]["seeds"])]
    tasks = [
        {
            "heldout_zone": zone,
            "seed": seed,
            "config": config,
            "config_sha256": config_sha256,
        }
        for zone in zones
        for seed in seeds
    ]
    workers = min(
        int(args.workers or config["execution"]["maximum_workers"]), len(tasks)
    )
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(build_unit, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "stage": "UNIT_COMPLETE",
                        "heldout_zone": result["heldout_zone"],
                        "seed": result["seed"],
                        "source_event_count": result["source_event_count"],
                        "resume_status": result["resume_status"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    results = sorted(
        results, key=lambda item: (str(item["heldout_zone"]), int(item["seed"]))
    )
    root_manifest = {
        "schema": "S08_PRICE_DECISION_CACHE_ROOT_V1",
        "status": "PASS" if len(results) == len(tasks) else "FAIL",
        "generated_at_utc": utc_now(),
        "config_sha256": config_sha256,
        "builder_sha256": sha256_file(Path(__file__)),
        "common_module_sha256": sha256_file(
            Path(__file__).with_name("price_sensitivity_common.py")
        ),
        "accelerator_sha256": sha256_file(
            S07_ROOT / "scripts" / "s07_decision_cache_accelerator.py"
        ),
        "authoritative_git_identity": git_identity,
        "input_audit": input_audit,
        "requested_zones": zones,
        "requested_seeds": seeds,
        "unit_count": len(results),
        "price_count": len(config["price_grid"]),
        "source_event_count_all_units": int(
            sum(int(item["source_event_count"]) for item in results)
        ),
        "base_price_regression_all_pass": all(
            item["base_price_regression"]["status"] == "PASS" for item in results
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "free_disk_gib_at_start": free_gib,
        "free_disk_gib_at_completion": ensure_disk_gate(config, OUTPUT_ROOT),
        "units": [
            {
                "heldout_zone": item["heldout_zone"],
                "seed": item["seed"],
                "unit_manifest_sha256": sha256_file(
                    unit_root(str(item["heldout_zone"]), int(item["seed"]))
                    / "unit_manifest.json"
                ),
                "base_price_regression_sha256": item[
                    "base_price_regression_sha256"
                ],
            }
            for item in results
        ],
    }
    full = set(zones) == set(config["scope"]["zones"]) and set(seeds) == set(
        config["scope"]["seeds"]
    )
    filename = "root_manifest.json" if full else "technical_regression_manifest.json"
    atomic_json(OUTPUT_ROOT / filename, root_manifest)
    print(json.dumps(root_manifest, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
