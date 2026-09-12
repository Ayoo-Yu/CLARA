"""S09 外部验证的低存储汇总器。"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd


CELL_FIELDS = (
    "zone_or_farm",
    "predictor",
    "seed",
    "horizon_steps",
    "target_coverage",
    "method",
)


def _attach_physical_reliability(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["towr_indicator"] = np.nan
    output["tuwr_indicator"] = np.nan
    output["ard_value"] = np.nan
    stream_fields = [
        "zone_or_farm",
        "predictor",
        "seed",
        "horizon_steps",
        "target_coverage",
        "method",
    ]
    for _, group in output.groupby(stream_fields, sort=True, dropna=False):
        ordered = group.sort_values(["issue_timestamp", "event_id"], kind="mergesort")
        cadence_values = ordered["nominal_cadence_minutes"].astype(float).unique()
        if len(cadence_values) != 1:
            raise RuntimeError("外部汇总单流名义时间步长不唯一")
        cadence = float(cadence_values[0])
        window = int(math.ceil(168.0 * 60.0 / cadence))
        if len(ordered) < window:
            continue
        covered = ordered["covered"].to_numpy(dtype=np.float64)
        rolling = np.convolve(covered, np.ones(window), mode="valid") / float(window)
        target = float(ordered["target_coverage"].iloc[0])
        gap = rolling - target
        tolerance = 1.96 * math.sqrt(target * (1.0 - target) / float(window))
        positions = ordered.index.to_numpy(dtype=np.int64)[window - 1 :]
        output.loc[positions, "towr_indicator"] = (gap > tolerance).astype(float)
        output.loc[positions, "tuwr_indicator"] = (gap < -tolerance).astype(float)
        output.loc[positions, "ard_value"] = np.abs(gap)
    return output


def summarize_method_complete(
    facts: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    *,
    method: str,
    selected_actions: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = facts[
        [
            "event_id",
            "zone_or_farm",
            "predictor",
            "seed",
            "horizon_steps",
            "target_coverage",
            "issue_timestamp",
            "ramp_state",
            "nominal_cadence_minutes",
        ]
    ].copy()
    for name in ("errf", "reserve", "miss", "covered", "width"):
        values = np.asarray(arrays[name])
        if len(values) != len(frame):
            raise RuntimeError(f"外部汇总数组长度失配: {method}: {name}")
        frame[name] = values
    if selected_actions is None:
        frame["selected_action"] = pd.NA
    else:
        actions = np.asarray(selected_actions, dtype=str)
        if len(actions) != len(frame):
            raise RuntimeError(f"外部汇总动作长度失配: {method}")
        frame["selected_action"] = actions
    frame["method"] = str(method)
    frame["issue_timestamp"] = pd.to_datetime(frame["issue_timestamp"], errors="raise")
    frame = frame.reset_index(drop=True)
    frame = _attach_physical_reliability(frame)

    metric_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    for regime in ("overall", "ordinary", "ramp"):
        part = (
            frame
            if regime == "overall"
            else frame[frame["ramp_state"].astype(str).eq(regime)]
        )
        for key, group in part.groupby(list(CELL_FIELDS), sort=True, dropna=False):
            reliability = group["tuwr_indicator"].notna()
            metric_rows.append(
                {
                    **dict(zip(CELL_FIELDS, key)),
                    "regime": regime,
                    "event_count": len(group),
                    "mean_errf": float(group["errf"].mean()),
                    "mean_reserve": float(group["reserve"].mean()),
                    "mean_miss": float(group["miss"].mean()),
                    "empirical_coverage": float(group["covered"].mean()),
                    "coverage_gap": float(group["covered"].mean()) - float(key[4]),
                    "average_width": float(group["width"].mean()),
                    "TOWR": float(group.loc[reliability, "towr_indicator"].mean())
                    if reliability.any()
                    else math.nan,
                    "TUWR": float(group.loc[reliability, "tuwr_indicator"].mean())
                    if reliability.any()
                    else math.nan,
                    "ARD": float(group.loc[reliability, "ard_value"].mean())
                    if reliability.any()
                    else math.nan,
                    "reliability_event_count": int(reliability.sum()),
                }
            )
            counts = group["selected_action"].value_counts(dropna=False)
            for action, count in counts.items():
                action_rows.append(
                    {
                        **dict(zip(CELL_FIELDS, key)),
                        "regime": regime,
                        "selected_action": None if pd.isna(action) else str(action),
                        "event_count": int(count),
                    }
                )

        utc_issue = part["issue_timestamp"]
        if utc_issue.dt.tz is None:
            utc_issue = utc_issue.dt.tz_localize(
                "Asia/Shanghai", ambiguous="raise", nonexistent="raise"
            )
        utc_issue = utc_issue.dt.tz_convert("UTC")
        for block_hours in (24, 168):
            working = part.copy()
            working["block_hours"] = int(block_hours)
            working["block_start_utc"] = utc_issue.dt.floor(f"{int(block_hours)}h")
            reliability = working["tuwr_indicator"].notna()
            working["reliability_count"] = reliability.astype(np.int64)
            working["towr_sum"] = working["towr_indicator"].fillna(0.0)
            working["tuwr_sum"] = working["tuwr_indicator"].fillna(0.0)
            working["ard_sum"] = working["ard_value"].fillna(0.0)
            grouping = [*CELL_FIELDS, "block_hours", "block_start_utc"]
            aggregated = (
                working.groupby(grouping, sort=True, dropna=False)
                .agg(
                    event_count=("event_id", "size"),
                    errf_sum=("errf", "sum"),
                    reserve_sum=("reserve", "sum"),
                    miss_sum=("miss", "sum"),
                    covered_sum=("covered", "sum"),
                    width_sum=("width", "sum"),
                    reliability_count=("reliability_count", "sum"),
                    towr_sum=("towr_sum", "sum"),
                    tuwr_sum=("tuwr_sum", "sum"),
                    ard_sum=("ard_sum", "sum"),
                )
                .reset_index()
            )
            aggregated["regime"] = regime
            block_rows.extend(aggregated.to_dict(orient="records"))
    return pd.DataFrame(metric_rows), pd.DataFrame(action_rows), pd.DataFrame(block_rows)
