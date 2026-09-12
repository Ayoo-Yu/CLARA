from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numba as nb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


S07_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = S07_ROOT.parent
AUTH_CODE = REVISION_ROOT / "_权威代码" / "code"
FULL_INDEX_ROOT = S07_ROOT / "results_raw" / "full_landscape_index_v1"
sys.path.insert(0, str(AUTH_CODE))

from clara_event_contract import FrozenContracts


ACTIONS = ("Static", "ACI", "AgACI", "EnbPI_RH")
CACHE_COLUMNS = (
    "predictor",
    "horizon_group",
    "target_coverage",
    "ramp_state",
    "rolling_state",
    "raw_width_state",
    "support_backoff_level",
    "support_backoff_name",
    "decision_evidence_level",
    "decision_bound_method",
    "guardrail_empty_fallback",
    "selected_action",
    "action",
    "support_n",
    "support_source_zones",
    "risk_mean",
    "risk_parent_mean",
    "risk_shrunken_mean",
    "risk_zone_cluster_se",
    "risk_score",
    "coverage_point",
    "coverage_lcb",
    "tuwr_point",
    "tuwr_ucb",
    "ard_point",
    "ard_ucb",
    "guardrail_pass",
    "guardrail_failure_reasons",
    "action_evidence_level",
    "action_bound_method",
)

_PCG_TEMPLATE = np.random.PCG64()
_NEXT_UINT32 = _PCG_TEMPLATE.ctypes.next_uint32


@nb.njit(cache=False)
def bounded_uint32(high: int, state_address: int) -> int:
    threshold = np.uint64(4294967296) % np.uint64(high)
    while True:
        value = np.uint64(_NEXT_UINT32(state_address))
        product = value * np.uint64(high)
        low = product & np.uint64(0xFFFFFFFF)
        if low >= threshold:
            return int(product >> np.uint64(32))


@nb.njit(cache=False)
def draw_bounded_sequence(high: int, count: int, state_address: int) -> np.ndarray:
    output = np.empty(count, dtype=np.int64)
    for index in range(count):
        output[index] = bounded_uint32(high, state_address)
    return output


@nb.njit(cache=False)
def bootstrap_samples_compiled(
    offsets: np.ndarray,
    time_counts: np.ndarray,
    block_counts: np.ndarray,
    full_event_count: np.ndarray,
    partial_event_count: np.ndarray,
    full_covered_sum: np.ndarray,
    partial_covered_sum: np.ndarray,
    full_tuwr_sum: np.ndarray,
    partial_tuwr_sum: np.ndarray,
    full_ard_sum: np.ndarray,
    partial_ard_sum: np.ndarray,
    replicates: int,
    cold_start: bool,
    state_address: int,
) -> np.ndarray:
    zone_count = len(time_counts)
    column_count = 1 if cold_start else 3
    samples = np.empty((replicates, column_count), dtype=np.float64)
    selected_zones = np.empty(zone_count, dtype=np.int64)
    for replicate in range(replicates):
        for position in range(zone_count):
            selected_zones[position] = bounded_uint32(zone_count, state_address)
        coverage_zone_sum = 0.0
        tuwr_zone_sum = 0.0
        ard_zone_sum = 0.0
        for position in range(zone_count):
            zone_index = selected_zones[position]
            start_offset = offsets[zone_index]
            n_times = time_counts[zone_index]
            n_blocks = block_counts[zone_index]
            event_count = 0.0
            covered_sum = 0.0
            tuwr_sum = 0.0
            ard_sum = 0.0
            for block_index in range(n_blocks):
                start = bounded_uint32(n_times, state_address)
                array_index = start_offset + start
                if block_index == n_blocks - 1:
                    event_count += partial_event_count[array_index]
                    covered_sum += partial_covered_sum[array_index]
                    if not cold_start:
                        tuwr_sum += partial_tuwr_sum[array_index]
                        ard_sum += partial_ard_sum[array_index]
                else:
                    event_count += full_event_count[array_index]
                    covered_sum += full_covered_sum[array_index]
                    if not cold_start:
                        tuwr_sum += full_tuwr_sum[array_index]
                        ard_sum += full_ard_sum[array_index]
            coverage_zone_sum += covered_sum / event_count
            if not cold_start:
                tuwr_zone_sum += tuwr_sum / event_count
                ard_zone_sum += ard_sum / event_count
        samples[replicate, 0] = coverage_zone_sum / zone_count
        if not cold_start:
            samples[replicate, 1] = tuwr_zone_sum / zone_count
            samples[replicate, 2] = ard_zone_sum / zone_count
    return samples


@dataclass(frozen=True)
class BootstrapPrepared:
    zones: tuple[str, ...]
    offsets: np.ndarray
    time_counts: np.ndarray
    block_counts: np.ndarray
    full_event_count: np.ndarray
    partial_event_count: np.ndarray
    full_covered_sum: np.ndarray
    partial_covered_sum: np.ndarray
    full_tuwr_sum: np.ndarray
    partial_tuwr_sum: np.ndarray
    full_ard_sum: np.ndarray
    partial_ard_sum: np.ndarray


def circular_window_sums(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    size = len(values)
    if size == 0 or length <= 0:
        raise ValueError("循环窗口要求正长度非空数组")
    cycles, remainder = divmod(int(length), size)
    base = float(cycles) * float(np.sum(values, dtype=np.float64))
    if remainder == 0:
        return np.full(size, base, dtype=np.float64)
    doubled = np.concatenate([values, values[: remainder - 1]])
    cumulative = np.concatenate([[0.0], np.cumsum(doubled, dtype=np.float64)])
    return base + cumulative[remainder : remainder + size] - cumulative[:size]


def prepare_bootstrap_group(
    frame: pd.DataFrame,
    *,
    action: str,
    cold_start: bool,
    block_hours: float,
) -> BootstrapPrepared:
    required = [
        "source_zone",
        "issue_timestamp",
        f"{action}__covered",
        f"{action}__tuwr_indicator",
        f"{action}__ard_value",
    ]
    working = frame[required].copy()
    if not cold_start:
        working = working[
            working[f"{action}__tuwr_indicator"].notna()
            & working[f"{action}__ard_value"].notna()
        ].copy()
    zones = tuple(sorted(working["source_zone"].astype(str).unique()))
    offsets = [0]
    time_counts: list[int] = []
    block_counts: list[int] = []
    full_event_parts: list[np.ndarray] = []
    partial_event_parts: list[np.ndarray] = []
    full_covered_parts: list[np.ndarray] = []
    partial_covered_parts: list[np.ndarray] = []
    full_tuwr_parts: list[np.ndarray] = []
    partial_tuwr_parts: list[np.ndarray] = []
    full_ard_parts: list[np.ndarray] = []
    partial_ard_parts: list[np.ndarray] = []

    for zone in zones:
        zone_frame = working[working["source_zone"].astype(str).eq(zone)].copy()
        zone_frame["issue_timestamp"] = pd.to_datetime(zone_frame["issue_timestamp"])
        aggregation: dict[str, tuple[str, str]] = {
            "event_count": (f"{action}__covered", "size"),
            "covered_sum": (f"{action}__covered", "sum"),
        }
        if not cold_start:
            aggregation["tuwr_sum"] = (f"{action}__tuwr_indicator", "sum")
            aggregation["ard_sum"] = (f"{action}__ard_value", "sum")
        by_time = (
            zone_frame.groupby("issue_timestamp", sort=True, as_index=False)
            .agg(**aggregation)
            .sort_values("issue_timestamp", kind="mergesort")
            .reset_index(drop=True)
        )
        times = by_time["issue_timestamp"].to_numpy(dtype="datetime64[us]").astype(np.int64)
        if len(times) < 2:
            raise ValueError("bootstrap支持组的时间点少于2")
        differences = np.maximum(np.diff(times).astype(np.float64) / 3_600_000_000.0, 1e-9)
        cadence = float(np.median(differences))
        block_length = max(1, int(math.ceil(float(block_hours) / cadence)))
        n_blocks = int(math.ceil(len(times) / block_length))
        partial_length = len(times) - (n_blocks - 1) * block_length
        event_values = by_time["event_count"].to_numpy(dtype=np.float64)
        covered_values = by_time["covered_sum"].to_numpy(dtype=np.float64)
        full_event_parts.append(circular_window_sums(event_values, block_length))
        partial_event_parts.append(circular_window_sums(event_values, partial_length))
        full_covered_parts.append(circular_window_sums(covered_values, block_length))
        partial_covered_parts.append(circular_window_sums(covered_values, partial_length))
        if cold_start:
            zeros = np.zeros(len(times), dtype=np.float64)
            full_tuwr_parts.append(zeros)
            partial_tuwr_parts.append(zeros)
            full_ard_parts.append(zeros)
            partial_ard_parts.append(zeros)
        else:
            tuwr_values = by_time["tuwr_sum"].to_numpy(dtype=np.float64)
            ard_values = by_time["ard_sum"].to_numpy(dtype=np.float64)
            full_tuwr_parts.append(circular_window_sums(tuwr_values, block_length))
            partial_tuwr_parts.append(circular_window_sums(tuwr_values, partial_length))
            full_ard_parts.append(circular_window_sums(ard_values, block_length))
            partial_ard_parts.append(circular_window_sums(ard_values, partial_length))
        time_counts.append(len(times))
        block_counts.append(n_blocks)
        offsets.append(offsets[-1] + len(times))

    return BootstrapPrepared(
        zones=zones,
        offsets=np.asarray(offsets, dtype=np.int64),
        time_counts=np.asarray(time_counts, dtype=np.int64),
        block_counts=np.asarray(block_counts, dtype=np.int64),
        full_event_count=np.concatenate(full_event_parts),
        partial_event_count=np.concatenate(partial_event_parts),
        full_covered_sum=np.concatenate(full_covered_parts),
        partial_covered_sum=np.concatenate(partial_covered_parts),
        full_tuwr_sum=np.concatenate(full_tuwr_parts),
        partial_tuwr_sum=np.concatenate(partial_tuwr_parts),
        full_ard_sum=np.concatenate(full_ard_parts),
        partial_ard_sum=np.concatenate(partial_ard_parts),
    )


def run_bootstrap_prepared(
    prepared: BootstrapPrepared,
    *,
    derived_seed: int,
    replicates: int,
    cold_start: bool,
) -> tuple[float, float | None, float | None, np.ndarray]:
    bit_generator = np.random.PCG64(int(derived_seed))
    samples = bootstrap_samples_compiled(
        prepared.offsets,
        prepared.time_counts,
        prepared.block_counts,
        prepared.full_event_count,
        prepared.partial_event_count,
        prepared.full_covered_sum,
        prepared.partial_covered_sum,
        prepared.full_tuwr_sum,
        prepared.partial_tuwr_sum,
        prepared.full_ard_sum,
        prepared.partial_ard_sum,
        int(replicates),
        bool(cold_start),
        int(bit_generator.ctypes.state.value),
    )
    coverage_lcb = float(np.quantile(samples[:, 0], 0.05, method="linear"))
    if cold_start:
        return coverage_lcb, None, None, samples
    return (
        coverage_lcb,
        float(np.quantile(samples[:, 1], 0.95, method="linear")),
        float(np.quantile(samples[:, 2], 0.95, method="linear")),
        samples,
    )


def state_frames(
    index_contract: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    vocabularies = index_contract["vocabularies"]
    base_records: list[dict[str, Any]] = []
    for predictor_index, predictor in enumerate(vocabularies["predictor"]):
        for horizon_group_index, horizon_group in enumerate(vocabularies["horizon_group"]):
            for coverage_index, target_coverage in enumerate(vocabularies["target_coverage"]):
                for ramp_index, ramp_state in enumerate(vocabularies["ramp_state"]):
                    for rolling_index, rolling_state in enumerate(vocabularies["rolling_state"]):
                        code = (
                            (((predictor_index * 5 + horizon_group_index) * 11 + coverage_index) * 2 + ramp_index)
                            * 5
                            + rolling_index
                        )
                        base_records.append(
                            {
                                "base_state_code": code,
                                "predictor": predictor,
                                "horizon_group": horizon_group,
                                "target_coverage": float(target_coverage),
                                "ramp_state": ramp_state,
                                "rolling_state": rolling_state,
                            }
                        )
    base = pd.DataFrame(base_records).sort_values("base_state_code", kind="mergesort").reset_index(drop=True)
    width_states = [str(value) for value in protocol["state_contract"]["raw_width"]["states"]]
    full_parts: list[pd.DataFrame] = []
    for width_index, width_state in enumerate(width_states):
        part = base.copy()
        part["raw_width_state"] = width_state
        part["full_state_code"] = part["base_state_code"].astype(np.int64) * len(width_states) + width_index
        full_parts.append(part)
    full = pd.concat(full_parts, ignore_index=True).sort_values("full_state_code", kind="mergesort").reset_index(drop=True)
    return base, full


def threshold_vectors(
    threshold_frame: pd.DataFrame,
    base_states: pd.DataFrame,
    *,
    heldout_zone: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    keys = ["predictor", "horizon_group", "target_coverage"]
    selected = threshold_frame[
        threshold_frame["outer_heldout_zone"].astype(str).eq(str(heldout_zone))
        & threshold_frame["seed"].astype(int).eq(int(seed))
    ][keys + ["raw_width_q33", "raw_width_q67"]].drop_duplicates(keys)
    merged = (
        base_states.merge(selected, on=keys, how="left", validate="many_to_one")
        .sort_values("base_state_code", kind="mergesort")
        .reset_index(drop=True)
    )
    if merged[["raw_width_q33", "raw_width_q67"]].isna().any().any():
        raise RuntimeError("折级宽度阈值映射不完整")
    return merged["raw_width_q33"].to_numpy(dtype=np.float64), merged["raw_width_q67"].to_numpy(dtype=np.float64)


def aggregate_fold_seed_metrics(
    *,
    source_zones: Sequence[str],
    seed: int,
    q33: np.ndarray,
    q67: np.ndarray,
    full_state_count: int,
    actions: Sequence[str] = ACTIONS,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    zones = tuple(sorted(str(zone) for zone in source_zones))
    zone_count = len(zones)
    action_count = len(actions)
    maximum_time = np.iinfo(np.int64).max
    minimum_time = np.iinfo(np.int64).min
    arrays = {
        "event_count": np.zeros((zone_count, full_state_count), dtype=np.int64),
        "issue_min_us": np.full((zone_count, full_state_count), maximum_time, dtype=np.int64),
        "issue_max_us": np.full((zone_count, full_state_count), minimum_time, dtype=np.int64),
        "guardrail_count": np.zeros((zone_count, full_state_count), dtype=np.int64),
        "guardrail_issue_min_us": np.full((zone_count, full_state_count), maximum_time, dtype=np.int64),
        "guardrail_issue_max_us": np.full((zone_count, full_state_count), minimum_time, dtype=np.int64),
        "errf_sum": np.zeros((action_count, zone_count, full_state_count), dtype=np.float64),
        "covered_sum": np.zeros((action_count, zone_count, full_state_count), dtype=np.float64),
        "guardrail_covered_sum": np.zeros((action_count, zone_count, full_state_count), dtype=np.float64),
        "tuwr_sum": np.zeros((action_count, zone_count, full_state_count), dtype=np.float64),
        "ard_sum": np.zeros((action_count, zone_count, full_state_count), dtype=np.float64),
    }
    audit_records: list[dict[str, Any]] = []
    metric_columns = [
        "base_state_code",
        "issue_timestamp",
        "raw_width_value",
        *(f"{action}__{field}" for action in actions for field in ("errf", "covered", "tuwr_indicator", "ard_value")),
    ]
    for zone_index, zone in enumerate(zones):
        path = FULL_INDEX_ROOT / f"zone={zone}" / f"seed={int(seed)}" / "source_landscape_index.parquet"
        parquet_file = pq.ParquetFile(path)
        input_rows = 0
        for batch in parquet_file.iter_batches(columns=metric_columns, batch_size=262144):
            names = batch.schema.names
            base = batch.column(names.index("base_state_code")).to_numpy(zero_copy_only=False).astype(np.int64)
            times = batch.column(names.index("issue_timestamp")).cast(pa.int64()).to_numpy(zero_copy_only=False).astype(np.int64)
            widths = batch.column(names.index("raw_width_value")).to_numpy(zero_copy_only=False).astype(np.float64)
            width_code = np.where(widths <= q33[base], 0, np.where(widths <= q67[base], 1, 2)).astype(np.int64)
            full_code = base * 3 + width_code
            arrays["event_count"][zone_index] += np.bincount(full_code, minlength=full_state_count).astype(np.int64)
            np.minimum.at(arrays["issue_min_us"][zone_index], full_code, times)
            np.maximum.at(arrays["issue_max_us"][zone_index], full_code, times)
            reference_guardrail_mask: np.ndarray | None = None
            for action_index, action in enumerate(actions):
                errf = batch.column(names.index(f"{action}__errf")).to_numpy(zero_copy_only=False).astype(np.float64)
                covered = batch.column(names.index(f"{action}__covered")).to_numpy(zero_copy_only=False).astype(np.float64)
                tuwr = batch.column(names.index(f"{action}__tuwr_indicator")).to_numpy(zero_copy_only=False).astype(np.float64)
                ard = batch.column(names.index(f"{action}__ard_value")).to_numpy(zero_copy_only=False).astype(np.float64)
                guardrail_mask = np.isfinite(tuwr) & np.isfinite(ard)
                if reference_guardrail_mask is None:
                    reference_guardrail_mask = guardrail_mask
                    if guardrail_mask.any():
                        guardrail_codes = full_code[guardrail_mask]
                        guardrail_times = times[guardrail_mask]
                        arrays["guardrail_count"][zone_index] += np.bincount(guardrail_codes, minlength=full_state_count).astype(np.int64)
                        np.minimum.at(arrays["guardrail_issue_min_us"][zone_index], guardrail_codes, guardrail_times)
                        np.maximum.at(arrays["guardrail_issue_max_us"][zone_index], guardrail_codes, guardrail_times)
                elif not np.array_equal(reference_guardrail_mask, guardrail_mask):
                    raise RuntimeError(f"四动作Guardrail缺失模式不一致: {zone}")
                arrays["errf_sum"][action_index, zone_index] += np.bincount(full_code, weights=errf, minlength=full_state_count)
                arrays["covered_sum"][action_index, zone_index] += np.bincount(full_code, weights=covered, minlength=full_state_count)
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
        audit_records.append(
            {
                "source_zone": zone,
                "seed": int(seed),
                "index_sha256": sha256_path_cached(path),
                "input_event_count": input_rows,
                "aggregated_event_count": int(arrays["event_count"][zone_index].sum()),
                "guardrail_event_count": int(arrays["guardrail_count"][zone_index].sum()),
                "status": "PASS" if input_rows == int(arrays["event_count"][zone_index].sum()) else "FAIL",
            }
        )
    return arrays, pd.DataFrame(audit_records)


_PATH_HASH_CACHE: dict[str, str] = {}


def sha256_path_cached(path: Path) -> str:
    key = str(path)
    if key not in _PATH_HASH_CACHE:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        _PATH_HASH_CACHE[key] = digest.hexdigest()
    return _PATH_HASH_CACHE[key]


def factor_level_mappings(
    states: pd.DataFrame,
    contracts: FrozenContracts,
) -> tuple[list[np.ndarray], list[pd.DataFrame]]:
    mappings: list[np.ndarray] = []
    groups: list[pd.DataFrame] = []
    for level in contracts.support_levels:
        fields = [str(field) for field in level["fields"]]
        if fields:
            keys = pd.MultiIndex.from_frame(states[fields])
            codes, unique = pd.factorize(keys, sort=True)
            group_frame = unique.to_frame(index=False)
            group_frame.columns = fields
        else:
            codes = np.zeros(len(states), dtype=np.int64)
            group_frame = pd.DataFrame(index=[0])
        group_frame.insert(0, "group_id", np.arange(len(group_frame), dtype=np.int64))
        mappings.append(np.asarray(codes, dtype=np.int64))
        groups.append(group_frame)
    return mappings, groups


def aggregate_by_mapping(
    values: np.ndarray,
    mapping: np.ndarray,
    group_count: int,
) -> np.ndarray:
    leading_shape = values.shape[:-1]
    flattened = values.reshape((-1, values.shape[-1]))
    output = np.zeros((len(flattened), group_count), dtype=values.dtype)
    for row_index in range(len(flattened)):
        output[row_index] = np.bincount(
            mapping,
            weights=flattened[row_index],
            minlength=group_count,
        ).astype(values.dtype)
    return output.reshape((*leading_shape, group_count))


def minimum_by_mapping(
    values: np.ndarray,
    present: np.ndarray,
    mapping: np.ndarray,
    group_count: int,
) -> np.ndarray:
    maximum = np.iinfo(np.int64).max
    output = np.full((values.shape[0], group_count), maximum, dtype=np.int64)
    for zone_index in range(values.shape[0]):
        mask = present[zone_index]
        if mask.any():
            np.minimum.at(output[zone_index], mapping[mask], values[zone_index, mask])
    return output


def maximum_by_mapping(
    values: np.ndarray,
    present: np.ndarray,
    mapping: np.ndarray,
    group_count: int,
) -> np.ndarray:
    minimum = np.iinfo(np.int64).min
    output = np.full((values.shape[0], group_count), minimum, dtype=np.int64)
    for zone_index in range(values.shape[0]):
        mask = present[zone_index]
        if mask.any():
            np.maximum.at(output[zone_index], mapping[mask], values[zone_index, mask])
    return output


def build_level_aggregates(
    arrays: dict[str, np.ndarray],
    mappings: list[np.ndarray],
    groups: list[pd.DataFrame],
) -> list[dict[str, np.ndarray]]:
    results: list[dict[str, np.ndarray]] = []
    for mapping, group_frame in zip(mappings, groups):
        group_count = len(group_frame)
        event_count = aggregate_by_mapping(arrays["event_count"], mapping, group_count)
        guardrail_count = aggregate_by_mapping(arrays["guardrail_count"], mapping, group_count)
        results.append(
            {
                "event_count": event_count,
                "guardrail_count": guardrail_count,
                "errf_sum": aggregate_by_mapping(arrays["errf_sum"], mapping, group_count),
                "covered_sum": aggregate_by_mapping(arrays["covered_sum"], mapping, group_count),
                "guardrail_covered_sum": aggregate_by_mapping(arrays["guardrail_covered_sum"], mapping, group_count),
                "tuwr_sum": aggregate_by_mapping(arrays["tuwr_sum"], mapping, group_count),
                "ard_sum": aggregate_by_mapping(arrays["ard_sum"], mapping, group_count),
                "issue_min_us": minimum_by_mapping(
                    arrays["issue_min_us"],
                    arrays["event_count"] > 0,
                    mapping,
                    group_count,
                ),
                "issue_max_us": maximum_by_mapping(
                    arrays["issue_max_us"],
                    arrays["event_count"] > 0,
                    mapping,
                    group_count,
                ),
                "guardrail_issue_min_us": minimum_by_mapping(
                    arrays["guardrail_issue_min_us"],
                    arrays["guardrail_count"] > 0,
                    mapping,
                    group_count,
                ),
                "guardrail_issue_max_us": maximum_by_mapping(
                    arrays["guardrail_issue_max_us"],
                    arrays["guardrail_count"] > 0,
                    mapping,
                    group_count,
                ),
            }
        )
    return results


def zone_balanced_metric(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    means = np.full_like(sums, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=means, where=counts > 0)
    valid = np.isfinite(means)
    valid_count = np.sum(valid, axis=-2)
    total = np.sum(np.where(valid, means, 0.0), axis=-2)
    result = np.full_like(total, np.nan, dtype=np.float64)
    np.divide(total, valid_count, out=result, where=valid_count > 0)
    return result


def zone_cluster_se(errf_sum: np.ndarray, counts: np.ndarray) -> np.ndarray:
    means = np.full_like(errf_sum, np.nan, dtype=np.float64)
    np.divide(errf_sum, counts, out=means, where=counts > 0)
    valid = np.isfinite(means)
    valid_count = np.sum(valid, axis=-2)
    total = np.sum(np.where(valid, means, 0.0), axis=-2)
    average = np.zeros_like(total, dtype=np.float64)
    np.divide(total, valid_count, out=average, where=valid_count > 0)
    centered = np.where(valid, means - np.expand_dims(average, axis=-2), 0.0)
    variance = np.zeros_like(average, dtype=np.float64)
    mask = valid_count > 1
    if np.any(mask):
        variance[mask] = np.sum(centered * centered, axis=-2)[mask] / (valid_count[mask] - 1)
    standard_error = np.zeros_like(average, dtype=np.float64)
    standard_error[mask] = np.sqrt(variance[mask]) / np.sqrt(valid_count[mask])
    return standard_error


def select_support_levels(
    states: pd.DataFrame,
    contracts: FrozenContracts,
    mappings: list[np.ndarray],
    levels: list[dict[str, np.ndarray]],
) -> pd.DataFrame:
    n_min = int(contracts.protocol["support_and_backoff_contract"]["default_n_min"])
    minimum_zones = int(contracts.protocol["support_and_backoff_contract"]["minimum_source_zones"])
    selected = np.full(len(states), -1, dtype=np.int64)
    support_n = np.zeros(len(states), dtype=np.int64)
    support_zones = np.zeros(len(states), dtype=np.int64)
    bootstrap_supported = np.zeros(len(states), dtype=bool)
    selected_group_id = np.full(len(states), -1, dtype=np.int64)
    minimum_span_hours = (
        float(contracts.protocol["guardrail_contract"]["dependence_robust_bounds"]["primary_block_hours"])
        * float(contracts.protocol["guardrail_contract"]["bootstrap_support_requirements"]["minimum_time_span_in_primary_blocks"])
    )
    for level_index, (mapping, aggregate) in enumerate(zip(mappings, levels)):
        grouped_count = aggregate["event_count"][:, mapping]
        total = grouped_count.sum(axis=0)
        zones = (grouped_count > 0).sum(axis=0)
        eligible = (selected < 0) & (total >= n_min) & (zones >= minimum_zones)
        indices = np.flatnonzero(eligible)
        if not len(indices):
            continue
        selected[indices] = level_index
        selected_group_id[indices] = mapping[indices]
        support_n[indices] = total[indices]
        support_zones[indices] = zones[indices]
        cold = states.loc[indices, "rolling_state"].astype(str).eq("cold_start").to_numpy()
        for local_index, state_index in enumerate(indices):
            group_id = int(mapping[state_index])
            if cold[local_index]:
                counts = aggregate["event_count"][:, group_id]
                minimums = aggregate["issue_min_us"][:, group_id]
                maximums = aggregate["issue_max_us"][:, group_id]
            else:
                counts = aggregate["guardrail_count"][:, group_id]
                minimums = aggregate["guardrail_issue_min_us"][:, group_id]
                maximums = aggregate["guardrail_issue_max_us"][:, group_id]
            present = counts > 0
            if int(present.sum()) >= minimum_zones:
                spans = (maximums[present].astype(np.float64) - minimums[present].astype(np.float64)) / 3_600_000_000.0
                bootstrap_supported[state_index] = bool(float(spans.min()) >= minimum_span_hours)
    if np.any(selected < 0):
        raise RuntimeError("七级支持回退仍存在未支持状态")
    profile = states.copy()
    profile["support_backoff_level"] = selected
    name_map = {int(level["level"]): str(level["name"]) for level in contracts.support_levels}
    profile["support_backoff_name"] = pd.Series(selected).map(name_map)
    profile["selected_group_id"] = selected_group_id
    profile["support_n"] = support_n
    profile["support_source_zones"] = support_zones
    profile["bootstrap_supported"] = bootstrap_supported
    profile["cold_start_guardrail"] = profile["rolling_state"].astype(str).eq("cold_start")
    return profile


def compute_cache_evidence(
    *,
    profile: pd.DataFrame,
    contracts: FrozenContracts,
    mappings: list[np.ndarray],
    levels: list[dict[str, np.ndarray]],
) -> pd.DataFrame:
    action_count = len(contracts.actions)
    state_count = len(profile)
    level_count = len(levels)
    level_risk_mean: list[np.ndarray] = []
    level_risk_se: list[np.ndarray] = []
    level_coverage_all: list[np.ndarray] = []
    level_coverage_guardrail: list[np.ndarray] = []
    level_tuwr: list[np.ndarray] = []
    level_ard: list[np.ndarray] = []
    for aggregate in levels:
        event_counts = aggregate["event_count"][None, :, :]
        guardrail_counts = aggregate["guardrail_count"][None, :, :]
        level_risk_mean.append(zone_balanced_metric(aggregate["errf_sum"], event_counts))
        level_risk_se.append(zone_cluster_se(aggregate["errf_sum"], event_counts))
        level_coverage_all.append(zone_balanced_metric(aggregate["covered_sum"], event_counts))
        level_coverage_guardrail.append(zone_balanced_metric(aggregate["guardrail_covered_sum"], guardrail_counts))
        level_tuwr.append(zone_balanced_metric(aggregate["tuwr_sum"], guardrail_counts))
        level_ard.append(zone_balanced_metric(aggregate["ard_sum"], guardrail_counts))

    selected_levels = profile["support_backoff_level"].to_numpy(dtype=np.int64)
    cold = profile["cold_start_guardrail"].to_numpy(dtype=bool)
    nu = float(contracts.protocol["support_and_backoff_contract"]["default_nu"])
    beta = float(contracts.protocol["landscape_contract"]["beta"])
    records: list[pd.DataFrame] = []
    for action_index, action in enumerate(contracts.actions):
        risk_mean = np.empty(state_count, dtype=np.float64)
        risk_parent = np.empty(state_count, dtype=np.float64)
        risk_shrunken = np.empty(state_count, dtype=np.float64)
        risk_se = np.empty(state_count, dtype=np.float64)
        coverage_point = np.empty(state_count, dtype=np.float64)
        tuwr_point = np.empty(state_count, dtype=np.float64)
        ard_point = np.empty(state_count, dtype=np.float64)
        for selected_level in range(level_count):
            indices = np.flatnonzero(selected_levels == selected_level)
            if not len(indices):
                continue
            global_values = level_risk_mean[-1][action_index, mappings[-1][indices]]
            if selected_level == level_count - 1:
                risk_mean[indices] = global_values
                risk_parent[indices] = global_values
                risk_shrunken[indices] = global_values
            else:
                parent = global_values.copy()
                for current_level in range(level_count - 2, selected_level - 1, -1):
                    group_ids = mappings[current_level][indices]
                    current_mean = level_risk_mean[current_level][action_index, group_ids]
                    current_n = levels[current_level]["event_count"][:, group_ids].sum(axis=0).astype(np.float64)
                    current_shrunken = (current_n * current_mean + nu * parent) / (current_n + nu)
                    if current_level == selected_level:
                        risk_mean[indices] = current_mean
                        risk_parent[indices] = parent
                        risk_shrunken[indices] = current_shrunken
                    else:
                        parent = current_shrunken
            selected_groups = mappings[selected_level][indices]
            risk_se[indices] = level_risk_se[selected_level][action_index, selected_groups]
            coverage_all = level_coverage_all[selected_level][action_index, selected_groups]
            coverage_guardrail = level_coverage_guardrail[selected_level][action_index, selected_groups]
            coverage_point[indices] = np.where(cold[indices], coverage_all, coverage_guardrail)
            tuwr_point[indices] = level_tuwr[selected_level][action_index, selected_groups]
            ard_point[indices] = level_ard[selected_level][action_index, selected_groups]
        tuwr_point[cold] = np.nan
        ard_point[cold] = np.nan
        action_frame = profile[
            [
                "full_state_code",
                "predictor",
                "horizon_group",
                "target_coverage",
                "ramp_state",
                "rolling_state",
                "raw_width_state",
                "support_backoff_level",
                "support_backoff_name",
                "support_n",
                "support_source_zones",
                "bootstrap_supported",
                "cold_start_guardrail",
                "selected_group_id",
            ]
        ].copy()
        action_frame["action"] = str(action)
        action_frame["action_order"] = action_index
        action_frame["risk_mean"] = risk_mean
        action_frame["risk_parent_mean"] = risk_parent
        action_frame["risk_shrunken_mean"] = risk_shrunken
        action_frame["risk_zone_cluster_se"] = risk_se
        action_frame["risk_score"] = risk_shrunken + beta * risk_se
        action_frame["coverage_point"] = coverage_point
        action_frame["coverage_lcb"] = coverage_point
        action_frame["tuwr_point"] = tuwr_point
        action_frame["tuwr_ucb"] = tuwr_point
        action_frame["ard_point"] = ard_point
        action_frame["ard_ucb"] = ard_point
        action_frame["action_evidence_level"] = np.where(
            cold,
            "COLD_START_COVERAGE_ONLY",
            "EMPIRICAL_FALLBACK",
        )
        action_frame["action_bound_method"] = "EMPIRICAL_FALLBACK"
        records.append(action_frame)
    return (
        pd.concat(records, ignore_index=True)
        .sort_values(["full_state_code", "action_order"], kind="mergesort")
        .reset_index(drop=True)
    )


def signature_seed(
    *,
    theta_id: str,
    seed: int,
    level: int,
    action: str,
    fields: Sequence[str],
    state: dict[str, Any],
    base_seed: int,
) -> tuple[str, int]:
    signature = "|".join(
        [str(theta_id), str(int(seed)), str(int(level)), str(action)]
        + [f"{field}={state[field]}" for field in fields]
        + ["0"]
    )
    derived = int(hashlib.sha256(signature.encode("utf-8")).hexdigest()[:8], 16)
    return signature, (int(base_seed) + derived) % (2**32)


def finalize_cache_decisions(cache: pd.DataFrame, contracts: FrozenContracts) -> pd.DataFrame:
    thresholds = contracts.protocol["guardrail_contract"]["default_thresholds"]
    tolerance = float(contracts.protocol["selection_contract"]["score_tie_tolerance"])
    cache = cache.copy()
    reasons: list[str] = []
    passes: list[bool] = []
    for row in cache.itertuples(index=False):
        current: list[str] = []
        if float(row.coverage_lcb) < float(row.target_coverage) - float(thresholds["coverage_shortfall_epsilon"]):
            current.append("coverage_lcb")
        if not bool(row.cold_start_guardrail):
            if float(row.tuwr_ucb) > float(thresholds["tuwr_upper"]):
                current.append("tuwr_ucb")
            if float(row.ard_ucb) > float(thresholds["ard_upper"]):
                current.append("ard_ucb")
        reasons.append(";".join(current))
        passes.append(not current)
    cache["guardrail_failure_reasons"] = reasons
    cache["guardrail_pass"] = passes
    decision_records: list[dict[str, Any]] = []
    for full_state_code, rows in cache.groupby("full_state_code", sort=True):
        ordered = rows.sort_values("action_order", kind="mergesort")
        safe = ordered[ordered["guardrail_pass"]]
        empty = safe.empty
        candidates = ordered.copy() if empty else safe.copy()
        minimum_risk = float(candidates["risk_score"].min())
        candidates = candidates[candidates["risk_score"] <= minimum_risk + tolerance]
        maximum_coverage = float(candidates["coverage_lcb"].max())
        candidates = candidates[candidates["coverage_lcb"] >= maximum_coverage - tolerance]
        if candidates["tuwr_ucb"].notna().all():
            minimum_tuwr = float(candidates["tuwr_ucb"].min())
            candidates = candidates[candidates["tuwr_ucb"] <= minimum_tuwr + tolerance]
        elif candidates["tuwr_ucb"].notna().any():
            raise RuntimeError("并列候选TUWR可用性不一致")
        if candidates["ard_ucb"].notna().all():
            minimum_ard = float(candidates["ard_ucb"].min())
            candidates = candidates[candidates["ard_ucb"] <= minimum_ard + tolerance]
        elif candidates["ard_ucb"].notna().any():
            raise RuntimeError("并列候选ARD可用性不一致")
        selected = candidates.sort_values("action_order", kind="mergesort").iloc[0]
        evidence_levels = set(ordered["action_evidence_level"].astype(str))
        if len(evidence_levels) != 1:
            raise RuntimeError("同一决策的Guardrail证据等级不一致")
        decision_records.append(
            {
                "full_state_code": int(full_state_code),
                "decision_evidence_level": next(iter(evidence_levels)),
                "decision_bound_method": (
                    "EMPIRICAL_FALLBACK"
                    if ordered["action_bound_method"].astype(str).eq("EMPIRICAL_FALLBACK").any()
                    else "BLOCK_BOOTSTRAP_BOUND"
                ),
                "guardrail_empty_fallback": bool(empty),
                "selected_action": str(selected["action"]),
            }
        )
    decisions = pd.DataFrame(decision_records)
    cache = cache.merge(decisions, on="full_state_code", how="left", validate="many_to_one")
    output = cache.rename(columns={"support_backoff_level": "support_backoff_level"})
    return output[list(CACHE_COLUMNS)].copy()


def cache_content_sha256(frame: pd.DataFrame) -> str:
    records = json.loads(frame.to_json(orient="records", date_format="iso"))
    raw = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
