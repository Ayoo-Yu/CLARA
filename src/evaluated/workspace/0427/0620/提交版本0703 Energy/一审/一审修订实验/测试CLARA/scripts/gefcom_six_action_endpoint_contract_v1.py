"""Stable endpoint algorithms for the GEFCom six-action experiment.

This module is deliberately isolated from orchestration. Endpoint unit manifests bind
this file, the frozen ``endpoint_contract`` config subtree, and authoritative input
dependencies; later selector/replay implementation changes therefore do not invalidate
already sealed endpoint units.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


TEST_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = TEST_ROOT.parent
AUTH_CODE = REVISION_ROOT / "_权威代码" / "code"
if str(AUTH_CODE) not in sys.path:
    sys.path.insert(0, str(AUTH_CODE))

from source_tuning_facts import _CompactRollingSummaries, _datetime_ns  # noqa: E402


CONTRACT_SCHEMA = "TEST_CLARA_GEFCOM_6A_ENDPOINT_ALGORITHM_V1"
RELIABILITY_STREAM_FIELDS = (
    "zone_or_farm",
    "predictor",
    "horizon_steps",
    "seed",
    "target_coverage",
)


def reliability_arrays(
    events: pd.DataFrame,
    covered: np.ndarray,
    *,
    cadence_minutes: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Reproduce the frozen 168 h action-specific TUWR/ARD release contract."""

    frame = events.reset_index(drop=True).copy()
    covered = np.asarray(covered, dtype=bool)
    if len(frame) != len(covered):
        raise RuntimeError("新增动作可靠性覆盖数组长度失配")
    tuwr = np.full(len(frame), np.nan, dtype=np.float64)
    ard = np.full(len(frame), np.nan, dtype=np.float64)
    reliable_count = 0
    cold_count = 0
    for _, group in frame.groupby(list(RELIABILITY_STREAM_FIELDS), sort=True, dropna=False):
        ordered = group.sort_values(["issue_timestamp", "event_id"], kind="mergesort")
        positions = ordered.index.to_numpy(dtype=np.int64)
        label_ns = _datetime_ns(ordered["label_timestamp"])
        availability_ns = _datetime_ns(ordered["label_available_timestamp"])
        if len(label_ns) > 1 and np.any(np.diff(label_ns) <= 0):
            raise RuntimeError("新增动作可靠性流标签时刻未严格递增")
        if len(availability_ns) > 1 and np.any(np.diff(availability_ns) <= 0):
            raise RuntimeError("新增动作可靠性流反馈时刻未严格递增")
        summary = _CompactRollingSummaries(
            label_ns=label_ns,
            covered=covered[positions],
            target_coverage=float(ordered["target_coverage"].iloc[0]),
            cadence_minutes=float(cadence_minutes),
        )
        release_full = ~ordered["rolling_state"].astype(str).eq("cold_start").to_numpy()
        for row_index, feedback_time_ns in enumerate(availability_ns):
            position = int(positions[row_index])
            if not release_full[row_index]:
                cold_count += 1
                continue
            _, _, _, value_tuwr, value_ard, state = summary.summarize(
                end_exclusive=row_index + 1,
                query_ns=int(feedback_time_ns),
            )
            if state == "cold_start" or not np.isfinite(value_tuwr) or not np.isfinite(value_ard):
                raise RuntimeError("新增动作完整历史状态缺少对应可靠性事实")
            tuwr[position] = value_tuwr
            ard[position] = value_ard
            reliable_count += 1
    expected_reliable = int((~frame["rolling_state"].astype(str).eq("cold_start")).sum())
    if reliable_count != expected_reliable or cold_count + reliable_count != len(frame):
        raise RuntimeError("新增动作可靠性事件数量不守恒")
    return tuwr, ard, {
        "event_count": len(frame),
        "reliability_event_count": reliable_count,
        "cold_state_count": cold_count,
    }


def endpoint_errf(
    *,
    target: np.ndarray,
    schedule: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    theta: Sequence[float],
    cadence_minutes: float = 60.0,
) -> np.ndarray:
    """Score fixed endpoints under a supplied price vector without changing endpoints."""

    target = np.asarray(target, dtype=np.float64)
    schedule = np.asarray(schedule, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if not (len(target) == len(schedule) == len(lower) == len(upper)):
        raise ValueError("ERRF 数组长度失配")
    if len(theta) != 4 or np.any(lower > upper):
        raise ValueError("ERRF theta 或区间端点无效")
    scale = float(cadence_minutes) / 60.0
    pi_plus, pi_minus, kappa_plus, kappa_minus = (float(value) for value in theta)
    return scale * (
        np.maximum(upper - schedule, 0.0) * pi_plus
        + np.maximum(schedule - lower, 0.0) * pi_minus
        + np.maximum(target - upper, 0.0) * kappa_plus
        + np.maximum(lower - target, 0.0) * kappa_minus
    )


def aligned_endpoints(
    events: pd.DataFrame,
    decisions: pd.DataFrame,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Map direct-interval decisions to the immutable event order and validate bounds."""

    mapped = events[["event_id"]].copy()
    mapped["_row_order"] = np.arange(len(mapped), dtype=np.int64)
    mapped = mapped.merge(
        decisions[["event_id", "selected_lower", "selected_upper"]],
        on="event_id",
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values("_row_order", kind="mergesort")
    if mapped[["selected_lower", "selected_upper"]].isna().any().any():
        raise RuntimeError(f"{label} 端点未覆盖完整事件")
    lower = mapped["selected_lower"].to_numpy(dtype=np.float64)
    upper = mapped["selected_upper"].to_numpy(dtype=np.float64)
    if not np.isfinite(np.column_stack([lower, upper])).all() or np.any(lower > upper):
        raise RuntimeError(f"{label} 端点含非有限值或上下界倒置")
    if np.any(lower < 0.0) or np.any(upper > 1.0):
        raise RuntimeError(f"{label} 端点超出冻结归一化边界")
    return lower, upper
