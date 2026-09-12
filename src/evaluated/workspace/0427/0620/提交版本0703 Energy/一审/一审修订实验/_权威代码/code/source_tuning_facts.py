from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from clara_errf import ErrfTheta
from clara_event_contract import FrozenContracts, sha256_file


SOURCE_FACT_SCHEMA_VERSION = "S06_SOURCE_TUNING_FACTS_V1"

EVENT_COLUMNS = (
    "event_id",
    "issue_timestamp",
    "label_timestamp",
    "label_available_timestamp",
    "horizon_steps",
    "nominal_cadence_minutes",
    "lead_time_hours",
    "target_lag1_timestamp",
    "feature_target_lag1",
    "target",
    "base_lower",
    "base_center",
    "base_upper",
    "dataset_id",
    "zone_or_farm",
    "predictor",
    "seed",
    "target_coverage",
)

CANDIDATE_COLUMNS = (
    "event_id",
    "action",
    "candidate_lower",
    "candidate_upper",
)

STREAM_FIELDS = (
    "dataset_id",
    "zone_or_farm",
    "predictor",
    "horizon_steps",
    "seed",
    "target_coverage",
)

WIDTH_THRESHOLD_FIELDS = (
    "predictor",
    "horizon_group",
    "target_coverage",
    "seed",
)


@dataclass(frozen=True)
class SourceFactBuildResult:
    facts: pd.DataFrame
    audit: dict[str, Any]


@dataclass(frozen=True)
class LoadedSourceFactResult:
    facts: pd.DataFrame
    audit: dict[str, Any]
    manifest: dict[str, Any]


def _canonical_frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    ordered = frame.loc[:, list(columns)].sort_values("event_id", kind="mergesort").reset_index(drop=True)
    row_hashes = pd.util.hash_pandas_object(ordered, index=False, categorize=True).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update("|".join(columns).encode("utf-8"))
    digest.update(row_hashes.astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{table_name}缺少字段: {missing}")


def _datetime_ns(values: pd.Series) -> np.ndarray:
    """统一不同Parquet时间物理单位，返回纳秒整数。"""

    return values.to_numpy(dtype="datetime64[ns]").astype(np.int64)


def _rolling_state(
    *,
    gap: float,
    towr: float,
    tuwr: float,
    ard: float,
) -> str:
    if tuwr > 0.35 or gap < -0.05:
        return "undercoverage_pressure"
    if towr > 0.50 or gap > 0.05:
        return "overcoverage_pressure"
    if ard > 0.08:
        return "volatile"
    return "stable"


class _CompactRollingSummaries:
    """用前缀和精确复现冻结的168小时与24小时可靠性摘要。"""

    def __init__(
        self,
        *,
        label_ns: np.ndarray,
        covered: np.ndarray,
        target_coverage: float,
        cadence_minutes: float,
        history_hours: float = 168.0,
        subwindow_hours: float = 24.0,
    ) -> None:
        self.label_ns = np.asarray(label_ns, dtype=np.int64)
        self.covered = np.asarray(covered, dtype=np.float64)
        self.target_coverage = float(target_coverage)
        self.cadence_minutes = float(cadence_minutes)
        self.expected_n = int(math.ceil(float(history_hours) * 60.0 / self.cadence_minutes))
        self.subwindow_n = max(
            1,
            int(math.ceil(float(subwindow_hours) * 60.0 / self.cadence_minutes)),
        )
        self.history_ns = int(round(float(history_hours) * 3600.0 * 1e9))
        self.cadence_ns = int(round(self.cadence_minutes * 60.0 * 1e9))
        if self.expected_n <= 0 or self.subwindow_n <= 0:
            raise ValueError("可靠性窗口长度必须为正")
        if self.subwindow_n > self.expected_n:
            raise ValueError("可靠性子窗口不得长于历史窗口")
        if len(self.label_ns) != len(self.covered):
            raise ValueError("可靠性标签时刻与覆盖指示长度失配")
        if len(self.label_ns) > 1 and np.any(np.diff(self.label_ns) <= 0):
            raise ValueError("同一事件流的标签时刻必须严格递增")
        if not np.isin(self.covered, (0.0, 1.0)).all():
            raise ValueError("可靠性覆盖指示必须为零或一")

        self.prefix = np.concatenate(([0.0], np.cumsum(self.covered, dtype=np.float64)))
        if len(self.covered) >= self.subwindow_n:
            subwindow_sum = self.prefix[self.subwindow_n :] - self.prefix[: -self.subwindow_n]
            rolling_gap = subwindow_sum / float(self.subwindow_n) - self.target_coverage
            tau = 1.96 * math.sqrt(
                self.target_coverage * (1.0 - self.target_coverage) / float(self.subwindow_n)
            )
            self.rolling_gap = rolling_gap
            self.tau = tau
        else:
            self.rolling_gap = np.zeros(0, dtype=np.float64)
            self.tau = 0.0

    def summarize(self, *, end_exclusive: int, query_ns: int) -> tuple[int, float, float, float, float, str]:
        end = int(end_exclusive)
        if end < 0 or end > len(self.label_ns):
            raise ValueError("可靠性历史终点越界")
        cutoff_ns = int(query_ns) - self.history_ns
        eligible_start = int(np.searchsorted(self.label_ns, cutoff_ns, side="left"))
        eligible_start = min(eligible_start, end)
        eligible_n = end - eligible_start
        if eligible_n < self.expected_n:
            return eligible_n, math.nan, math.nan, math.nan, math.nan, "cold_start"

        start = end - self.expected_n
        if int(self.label_ns[start]) > cutoff_ns + self.cadence_ns:
            return self.expected_n, math.nan, math.nan, math.nan, math.nan, "cold_start"

        values = self.covered[start:end]
        gap = float(np.mean(values) - self.target_coverage)
        subwindow_start = start
        subwindow_end = end - self.subwindow_n + 1
        subwindow_count = subwindow_end - subwindow_start
        if subwindow_count <= 0:
            raise RuntimeError("可靠性滚动子窗口数量异常")
        rolling_gap = self.rolling_gap[subwindow_start:subwindow_end]
        towr = float(np.mean(rolling_gap > self.tau))
        tuwr = float(np.mean(rolling_gap < -self.tau))
        ard = float(np.mean(np.abs(rolling_gap)))
        return self.expected_n, gap, towr, tuwr, ard, _rolling_state(
            gap=gap,
            towr=towr,
            tuwr=tuwr,
            ard=ard,
        )


def _align_candidates(
    *,
    event_ids: pd.Series,
    candidates: pd.DataFrame,
    actions: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    event_id_values = event_ids.astype(str).to_numpy()
    event_position = pd.Series(np.arange(len(event_id_values), dtype=np.int64), index=event_id_values)
    action_rank = {action: index for index, action in enumerate(actions)}
    aligned = candidates.loc[:, list(CANDIDATE_COLUMNS)].copy()
    aligned["event_id"] = aligned["event_id"].astype(str)
    aligned["_event_position"] = aligned["event_id"].map(event_position)
    aligned["_action_rank"] = aligned["action"].astype(str).map(action_rank)
    if aligned["_event_position"].isna().any():
        unknown = aligned.loc[aligned["_event_position"].isna(), "event_id"].drop_duplicates().head(10).tolist()
        raise ValueError(f"候选表含完整案例事件集合之外的事件: {unknown}")
    if aligned["_action_rank"].isna().any():
        unknown_actions = sorted(set(aligned.loc[aligned["_action_rank"].isna(), "action"].astype(str)))
        raise ValueError(f"候选表含冻结动作集合之外的动作: {unknown_actions}")
    if aligned.loc[:, ["event_id", "action"]].duplicated().any():
        raise ValueError("候选表存在重复event_id与action")
    if len(aligned) != len(event_id_values) * len(actions):
        raise ValueError("完整案例候选行数未精确等于事件数乘动作数")
    counts = aligned.groupby("event_id", sort=False)["action"].nunique()
    if len(counts) != len(event_id_values) or not counts.eq(len(actions)).all():
        raise ValueError("完整案例事件的四动作候选不完整")
    aligned = aligned.sort_values(["_event_position", "_action_rank"], kind="mergesort")
    lower = pd.to_numeric(aligned["candidate_lower"], errors="coerce").to_numpy(dtype=np.float64)
    upper = pd.to_numeric(aligned["candidate_upper"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("候选端点含非有限值")
    if np.any(lower > upper):
        raise ValueError("候选区间下界高于上界")
    if np.any(lower < 0.0) or np.any(upper > 1.0):
        raise ValueError("候选区间违反零到一共同裁剪规则")
    return lower.reshape(len(event_id_values), len(actions)), upper.reshape(len(event_id_values), len(actions))


def _compute_action_metrics(
    *,
    target: np.ndarray,
    schedule: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    cadence_minutes: np.ndarray,
    theta: ErrfTheta,
) -> tuple[np.ndarray, np.ndarray]:
    scale = cadence_minutes[:, None] / 60.0
    reserve_up = scale * np.maximum(upper - schedule[:, None], 0.0) * float(theta.pi_plus)
    reserve_down = scale * np.maximum(schedule[:, None] - lower, 0.0) * float(theta.pi_minus)
    miss_upper = scale * np.maximum(target[:, None] - upper, 0.0) * float(theta.kappa_plus)
    miss_lower = scale * np.maximum(lower - target[:, None], 0.0) * float(theta.kappa_minus)
    errf = reserve_up + reserve_down + miss_upper + miss_lower
    covered = (lower <= target[:, None]) & (target[:, None] <= upper)
    if not np.isfinite(errf).all() or np.any(errf < 0.0):
        raise ValueError("高效事实构建器产生了非法ERRF")
    return errf, covered


def build_source_tuning_facts(
    *,
    event_core: pd.DataFrame,
    candidates: pd.DataFrame,
    contracts: FrozenContracts,
    theta: ErrfTheta | None = None,
    ramp_threshold: float = 0.12,
) -> SourceFactBuildResult:
    """从S03封存表构建不含外层宽度分箱的紧凑源域事实。"""

    _require_columns(event_core, EVENT_COLUMNS, "S03事件表")
    _require_columns(candidates, CANDIDATE_COLUMNS, "S03候选表")
    resolved_theta = theta or ErrfTheta()
    events = event_core.loc[:, list(EVENT_COLUMNS)].copy()
    events["event_id"] = events["event_id"].astype(str)
    if events["event_id"].isna().any() or events["event_id"].duplicated().any():
        raise ValueError("S03事件表event_id必须非空且唯一")
    for column in (
        "issue_timestamp",
        "label_timestamp",
        "label_available_timestamp",
        "target_lag1_timestamp",
    ):
        events[column] = pd.to_datetime(events[column], errors="coerce")
    terminal_mask = events["label_available_timestamp"].isna()
    terminal_event_ids = set(events.loc[terminal_mask, "event_id"])
    complete = events.loc[~terminal_mask].copy().reset_index(drop=True)
    if complete.empty:
        raise ValueError("完整案例门排除后没有可用事件")
    if complete.loc[:, ["issue_timestamp", "label_timestamp", "target_lag1_timestamp"]].isna().any().any():
        raise ValueError("完整案例事件仍含必要时刻缺失")
    if not (complete["issue_timestamp"] < complete["label_timestamp"]).all():
        raise ValueError("完整案例事件label_timestamp未晚于issue_timestamp")
    if not (complete["label_timestamp"] < complete["label_available_timestamp"]).all():
        raise ValueError("完整案例事件label_available_timestamp未严格晚于label_timestamp")

    candidate_frame = candidates.loc[:, list(CANDIDATE_COLUMNS)].copy()
    candidate_frame["event_id"] = candidate_frame["event_id"].astype(str)
    terminal_candidate_count = int(candidate_frame["event_id"].isin(terminal_event_ids).sum())
    candidate_frame = candidate_frame.loc[~candidate_frame["event_id"].isin(terminal_event_ids)].copy()
    lower, upper = _align_candidates(
        event_ids=complete["event_id"],
        candidates=candidate_frame,
        actions=contracts.actions,
    )

    numeric_columns = (
        "horizon_steps",
        "nominal_cadence_minutes",
        "lead_time_hours",
        "feature_target_lag1",
        "target",
        "base_lower",
        "base_center",
        "base_upper",
        "seed",
        "target_coverage",
    )
    for column in numeric_columns:
        complete[column] = pd.to_numeric(complete[column], errors="coerce")
    if not np.isfinite(complete.loc[:, list(numeric_columns)].to_numpy(dtype=np.float64)).all():
        raise ValueError("完整案例事件含非有限数值")
    if (complete["nominal_cadence_minutes"] <= 0.0).any():
        raise ValueError("名义时间步长必须为正")

    target = complete["target"].to_numpy(dtype=np.float64)
    schedule = complete["base_center"].to_numpy(dtype=np.float64)
    cadence = complete["nominal_cadence_minutes"].to_numpy(dtype=np.float64)
    errf, covered = _compute_action_metrics(
        target=target,
        schedule=schedule,
        lower=lower,
        upper=upper,
        cadence_minutes=cadence,
        theta=resolved_theta,
    )

    fact = pd.DataFrame(
        {
            "event_id": complete["event_id"].astype(str).to_numpy(),
            "origin": "source",
            "dataset_id": complete["dataset_id"].astype(str).to_numpy(),
            "zone_or_farm": complete["zone_or_farm"].astype(str).to_numpy(),
            "predictor": complete["predictor"].astype(str).to_numpy(),
            "horizon_steps": complete["horizon_steps"].astype(np.int64).to_numpy(),
            "horizon_hours": complete["lead_time_hours"].astype(float).to_numpy(),
            "seed": complete["seed"].astype(np.int64).to_numpy(),
            "target_coverage": complete["target_coverage"].astype(float).to_numpy(),
            "issue_timestamp": complete["issue_timestamp"].to_numpy(),
            "label_timestamp": complete["label_timestamp"].to_numpy(),
            "label_available_timestamp": complete["label_available_timestamp"].to_numpy(),
            "theta_id": resolved_theta.theta_id,
            "ramp_proxy": np.abs(
                complete["base_center"].to_numpy(dtype=np.float64)
                - complete["feature_target_lag1"].to_numpy(dtype=np.float64)
            ),
            "raw_width_value": (
                complete["base_upper"].to_numpy(dtype=np.float64)
                - complete["base_lower"].to_numpy(dtype=np.float64)
            ),
        }
    )
    if np.any(fact["raw_width_value"].to_numpy(dtype=np.float64) < 0.0):
        raise ValueError("基础区间原始宽度为负")
    fact["ramp_state"] = np.where(
        fact["ramp_proxy"].to_numpy(dtype=np.float64) >= float(ramp_threshold),
        "ramp",
        "ordinary",
    )
    fact["horizon_group"] = [
        contracts.horizon_group(float(value)) for value in fact["horizon_hours"].to_numpy(dtype=float)
    ]

    for action_index, action in enumerate(contracts.actions):
        fact[f"{action}__candidate_lower"] = lower[:, action_index]
        fact[f"{action}__candidate_upper"] = upper[:, action_index]
        fact[f"{action}__errf"] = errf[:, action_index]
        fact[f"{action}__covered"] = covered[:, action_index]
        fact[f"{action}__tuwr_indicator"] = np.nan
        fact[f"{action}__ard_value"] = np.nan

    fact["rolling_n"] = 0
    fact["rolling_gap"] = np.nan
    fact["rolling_towr"] = np.nan
    fact["rolling_tuwr"] = np.nan
    fact["rolling_ard"] = np.nan
    fact["rolling_state"] = "cold_start"

    complete["_fact_position"] = np.arange(len(complete), dtype=np.int64)
    stream_issue_position_count = 0
    stream_count = 0
    full_history_count = 0
    action_summary_cold_failure_count = 0
    grouped = complete.groupby(list(STREAM_FIELDS), sort=True, dropna=False)
    static_index = list(contracts.actions).index("Static")
    for _, group in grouped:
        stream_count += 1
        ordered = group.sort_values(["issue_timestamp", "event_id"], kind="mergesort")
        positions = ordered["_fact_position"].to_numpy(dtype=np.int64)
        issue_ns = _datetime_ns(ordered["issue_timestamp"])
        label_ns = _datetime_ns(ordered["label_timestamp"])
        availability_ns = _datetime_ns(ordered["label_available_timestamp"])
        if len(issue_ns) > 1 and np.any(np.diff(issue_ns) <= 0):
            raise ValueError("同一事件流的签发时刻必须严格递增")
        if len(label_ns) > 1 and np.any(np.diff(label_ns) <= 0):
            raise ValueError("同一事件流的标签时刻必须严格递增")
        if len(availability_ns) > 1 and np.any(np.diff(availability_ns) <= 0):
            raise ValueError("同一事件流的标签可用时刻必须严格递增")
        cadence_values = ordered["nominal_cadence_minutes"].to_numpy(dtype=np.float64)
        if not np.all(cadence_values == cadence_values[0]):
            raise ValueError("同一事件流的名义时间步长不一致")
        coverage_value = float(ordered["target_coverage"].iloc[0])
        stream_issue_position_count += len(issue_ns)

        static_summaries = _CompactRollingSummaries(
            label_ns=label_ns,
            covered=covered[positions, static_index],
            target_coverage=coverage_value,
            cadence_minutes=float(cadence_values[0]),
        )
        release_full = np.zeros(len(ordered), dtype=bool)
        release_state_values = np.empty(len(ordered), dtype=object)
        release_n = np.zeros(len(ordered), dtype=np.int64)
        release_gap = np.full(len(ordered), np.nan, dtype=np.float64)
        release_towr = np.full(len(ordered), np.nan, dtype=np.float64)
        release_tuwr = np.full(len(ordered), np.nan, dtype=np.float64)
        release_ard = np.full(len(ordered), np.nan, dtype=np.float64)
        for row_index, current_issue_ns in enumerate(issue_ns):
            end_by_availability = int(np.searchsorted(availability_ns, current_issue_ns, side="right"))
            end_by_label = int(np.searchsorted(label_ns, current_issue_ns, side="left"))
            end = min(end_by_availability, end_by_label)
            n_value, gap_value, towr_value, tuwr_value, ard_value, state_value = static_summaries.summarize(
                end_exclusive=end,
                query_ns=int(current_issue_ns),
            )
            release_n[row_index] = n_value
            release_gap[row_index] = gap_value
            release_towr[row_index] = towr_value
            release_tuwr[row_index] = tuwr_value
            release_ard[row_index] = ard_value
            release_state_values[row_index] = state_value
            release_full[row_index] = state_value != "cold_start"

        fact.loc[positions, "rolling_n"] = release_n
        fact.loc[positions, "rolling_gap"] = release_gap
        fact.loc[positions, "rolling_towr"] = release_towr
        fact.loc[positions, "rolling_tuwr"] = release_tuwr
        fact.loc[positions, "rolling_ard"] = release_ard
        fact.loc[positions, "rolling_state"] = release_state_values
        full_history_count += int(release_full.sum())

        for action_index, action in enumerate(contracts.actions):
            action_summaries = _CompactRollingSummaries(
                label_ns=label_ns,
                covered=covered[positions, action_index],
                target_coverage=coverage_value,
                cadence_minutes=float(cadence_values[0]),
            )
            action_tuwr = np.full(len(ordered), np.nan, dtype=np.float64)
            action_ard = np.full(len(ordered), np.nan, dtype=np.float64)
            for row_index, feedback_time_ns in enumerate(availability_ns):
                if not release_full[row_index]:
                    continue
                _, _, _, tuwr_value, ard_value, state_value = action_summaries.summarize(
                    end_exclusive=row_index + 1,
                    query_ns=int(feedback_time_ns),
                )
                if state_value == "cold_start":
                    action_summary_cold_failure_count += 1
                    raise RuntimeError("完整历史发布状态缺少对应动作可靠性历史")
                action_tuwr[row_index] = tuwr_value
                action_ard[row_index] = ard_value
            fact.loc[positions, f"{action}__tuwr_indicator"] = action_tuwr
            fact.loc[positions, f"{action}__ard_value"] = action_ard

    fact = fact.sort_values("event_id", kind="mergesort").reset_index(drop=True)
    action_columns = []
    for action in contracts.actions:
        action_columns.extend(
            [
                f"{action}__candidate_lower",
                f"{action}__candidate_upper",
                f"{action}__errf",
                f"{action}__covered",
                f"{action}__tuwr_indicator",
                f"{action}__ard_value",
            ]
        )
    content_columns = [
        "event_id",
        "zone_or_farm",
        "predictor",
        "horizon_steps",
        "seed",
        "target_coverage",
        "issue_timestamp",
        "label_timestamp",
        "label_available_timestamp",
        "ramp_state",
        "raw_width_value",
        "rolling_n",
        "rolling_gap",
        "rolling_towr",
        "rolling_tuwr",
        "rolling_ard",
        "rolling_state",
        *action_columns,
    ]
    audit = {
        "fact_schema_version": SOURCE_FACT_SCHEMA_VERSION,
        "input_event_count": len(events),
        "terminal_censored_event_count": int(terminal_mask.sum()),
        "complete_case_event_count": len(fact),
        "input_candidate_count": len(candidates),
        "terminal_censored_candidate_count": terminal_candidate_count,
        "complete_case_candidate_count": len(candidate_frame),
        "stream_count": stream_count,
        "issue_batch_count": int(fact["issue_timestamp"].nunique()),
        "stream_issue_position_count": stream_issue_position_count,
        "full_history_event_count": full_history_count,
        "cold_start_event_count": len(fact) - full_history_count,
        "action_observation_count": len(fact) * len(contracts.actions),
        "action_summary_cold_failure_count": action_summary_cold_failure_count,
        "future_information_violation_count": 0,
        "within_issue_feedback_use_count": 0,
        "pending_feedback_count": 0,
        "raw_width_state_attached": False,
        "fact_content_sha256": _canonical_frame_sha256(fact, content_columns),
    }
    return SourceFactBuildResult(facts=fact, audit=audit)


def fit_raw_width_thresholds(
    facts: pd.DataFrame,
    *,
    source_zones: Iterable[str],
) -> pd.DataFrame:
    _require_columns(
        facts,
        ("zone_or_farm", *WIDTH_THRESHOLD_FIELDS, "raw_width_value"),
        "源域事实表",
    )
    zones = tuple(sorted(set(str(zone) for zone in source_zones)))
    if not zones:
        raise ValueError("原始宽度阈值源区集合为空")
    frame = facts[facts["zone_or_farm"].astype(str).isin(zones)].copy()
    observed = set(frame["zone_or_farm"].astype(str))
    if observed != set(zones):
        raise ValueError(f"原始宽度阈值源区不完整: observed={sorted(observed)}, expected={list(zones)}")
    if not np.isfinite(pd.to_numeric(frame["raw_width_value"], errors="coerce").to_numpy(dtype=float)).all():
        raise ValueError("原始宽度含非有限值")
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(list(WIDTH_THRESHOLD_FIELDS), sort=True, dropna=False):
        values = group["raw_width_value"].to_numpy(dtype=np.float64)
        rows.append(
            {
                **dict(zip(WIDTH_THRESHOLD_FIELDS, key)),
                "raw_width_q33": float(np.quantile(values, 0.33, method="linear")),
                "raw_width_q67": float(np.quantile(values, 0.67, method="linear")),
                "threshold_event_count": len(values),
                "threshold_source_zone_count": group["zone_or_farm"].nunique(),
            }
        )
    thresholds = pd.DataFrame(rows)
    if thresholds.empty:
        raise ValueError("没有可用的原始宽度阈值")
    return thresholds


def attach_raw_width_states(
    facts: pd.DataFrame,
    *,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(facts, (*WIDTH_THRESHOLD_FIELDS, "raw_width_value"), "源域事实表")
    _require_columns(
        thresholds,
        (*WIDTH_THRESHOLD_FIELDS, "raw_width_q33", "raw_width_q67"),
        "原始宽度阈值表",
    )
    if thresholds.loc[:, list(WIDTH_THRESHOLD_FIELDS)].duplicated().any():
        raise ValueError("原始宽度阈值键重复")
    merged = facts.merge(
        thresholds.loc[:, [*WIDTH_THRESHOLD_FIELDS, "raw_width_q33", "raw_width_q67"]],
        on=list(WIDTH_THRESHOLD_FIELDS),
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if merged.loc[:, ["raw_width_q33", "raw_width_q67"]].isna().any().any():
        raise ValueError("存在无法关联原始宽度阈值的事件")
    value = merged["raw_width_value"].to_numpy(dtype=np.float64)
    q33 = merged["raw_width_q33"].to_numpy(dtype=np.float64)
    q67 = merged["raw_width_q67"].to_numpy(dtype=np.float64)
    merged["raw_width_state"] = np.where(
        value <= q33,
        "narrow",
        np.where(value <= q67, "medium", "wide"),
    )
    return merged


def facts_to_long_observations(
    facts: pd.DataFrame,
    *,
    contracts: FrozenContracts,
) -> pd.DataFrame:
    required_event_columns = (
        "event_id",
        "origin",
        "zone_or_farm",
        "seed",
        "theta_id",
        "issue_timestamp",
        "label_timestamp",
        "label_available_timestamp",
        "predictor",
        "horizon_group",
        "target_coverage",
        "ramp_state",
        "rolling_state",
        "raw_width_state",
    )
    _require_columns(facts, required_event_columns, "带宽度状态的源域事实表")
    rows: list[pd.DataFrame] = []
    for action in contracts.actions:
        action_columns = (
            f"{action}__candidate_lower",
            f"{action}__candidate_upper",
            f"{action}__errf",
            f"{action}__covered",
            f"{action}__tuwr_indicator",
            f"{action}__ard_value",
        )
        _require_columns(facts, action_columns, "带宽度状态的源域事实表")
        part = facts.loc[:, list(required_event_columns)].copy()
        part["action"] = action
        part["candidate_lower"] = facts[action_columns[0]].to_numpy(dtype=float)
        part["candidate_upper"] = facts[action_columns[1]].to_numpy(dtype=float)
        part["errf"] = facts[action_columns[2]].to_numpy(dtype=float)
        part["covered"] = facts[action_columns[3]].to_numpy(dtype=bool)
        part["tuwr_indicator"] = facts[action_columns[4]].to_numpy(dtype=float)
        part["ard_value"] = facts[action_columns[5]].to_numpy(dtype=float)
        rows.append(part)
    result = pd.concat(rows, ignore_index=True)
    action_rank = {action: index for index, action in enumerate(contracts.actions)}
    result["_action_rank"] = result["action"].map(action_rank)
    return (
        result.sort_values(["event_id", "_action_rank"], kind="mergesort")
        .drop(columns="_action_rank")
        .reset_index(drop=True)
    )


def load_s03_source_tuning_facts(
    bundle_dir: str | Path,
    *,
    contracts: FrozenContracts,
    expected_manifest_sha256: str | None = None,
    verify_input_file_hashes: bool = True,
    theta: ErrfTheta | None = None,
) -> LoadedSourceFactResult:
    directory = Path(bundle_dir)
    manifest_path = directory / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    if expected_manifest_sha256 is not None and manifest_sha256 != str(expected_manifest_sha256):
        raise RuntimeError("S03候选包清单SHA256失配")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_schema") != "S03_NORMALIZED_CANDIDATE_BUNDLE_V1":
        raise RuntimeError("S03候选包清单模式失配")
    if manifest.get("status") != "COMPLETE_VALIDATED":
        raise RuntimeError("S03候选包未处于完整验证状态")
    observed_identity = {
        "protocol_id": str(manifest.get("protocol_id")),
        "protocol_version": str(manifest.get("protocol_version")),
        "protocol_sha256": str(manifest.get("protocol_sha256")),
        "baseline_registry_sha256": str(manifest.get("baseline_registry_sha256")),
        "output_contract_sha256": str(manifest.get("output_contract_sha256")),
    }
    if observed_identity not in contracts.accepted_candidate_bundle_identities:
        raise RuntimeError("S03候选包身份不在冻结只读兼容集合中")

    files = {
        "event_core": directory / "event_core.parquet",
        "candidate_intervals": directory / "candidate_intervals.parquet",
        "feedback_trace": directory / "feedback_trace.parquet",
    }
    expected_hashes = {
        "event_core": str(manifest["event_core_sha256"]),
        "candidate_intervals": str(manifest["candidate_intervals_sha256"]),
        "feedback_trace": str(manifest["feedback_trace_sha256"]),
    }
    observed_hashes: dict[str, str] = {}
    if verify_input_file_hashes:
        for name, path in files.items():
            observed_hashes[name] = sha256_file(path)
            if observed_hashes[name] != expected_hashes[name]:
                raise RuntimeError(f"S03候选包文件SHA256失配: {name}")

    event_core = pd.read_parquet(files["event_core"], columns=list(EVENT_COLUMNS))
    candidates = pd.read_parquet(files["candidate_intervals"], columns=list(CANDIDATE_COLUMNS))
    result = build_source_tuning_facts(
        event_core=event_core,
        candidates=candidates,
        contracts=contracts,
        theta=theta,
    )
    if len(event_core) != int(manifest["event_rows"]):
        raise RuntimeError("S03事件行数与清单失配")
    if len(candidates) != int(manifest["candidate_rows"]):
        raise RuntimeError("S03候选行数与清单失配")
    audit = {
        **result.audit,
        "bundle_id": str(manifest["bundle_id"]),
        "manifest_sha256": manifest_sha256,
        "expected_file_sha256": expected_hashes,
        "observed_file_sha256": observed_hashes,
        "input_file_hashes_verified": bool(verify_input_file_hashes),
        "bundle_protocol_version": observed_identity["protocol_version"],
        "bundle_protocol_sha256": observed_identity["protocol_sha256"],
        "effective_protocol_version": contracts.protocol_version,
        "effective_protocol_sha256": contracts.protocol_sha256,
        "compatibility_bridge_used": observed_identity["protocol_sha256"] != contracts.protocol_sha256,
    }
    return LoadedSourceFactResult(
        facts=result.facts,
        audit=audit,
        manifest=manifest,
    )
