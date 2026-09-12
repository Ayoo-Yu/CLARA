from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from baseline_common import BaselineDecision
from clara_event_contract import FrozenContracts, as_datetime, canonical_json_sha256


BASE_REQUIRED_FIELDS = (
    "split",
    "issue_timestamp",
    "label_timestamp",
    "label_available_timestamp",
    "target",
    "horizon_steps",
    "nominal_cadence_minutes",
)


@dataclass(frozen=True)
class ConformalConfiguration:
    family: str
    configuration: dict[str, Any]
    configuration_id: str
    complexity_rank: int

    def as_record(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "configuration": self.configuration,
            "configuration_id": self.configuration_id,
            "complexity_rank": self.complexity_rank,
        }


@dataclass(frozen=True)
class ConformalRunResult:
    configuration: ConformalConfiguration
    decisions: tuple[BaselineDecision, ...]
    candidates: pd.DataFrame
    feedback: pd.DataFrame
    audit: dict[str, Any]


@dataclass(frozen=True)
class _PendingConformalFeedback:
    event_id: str
    label_timestamp: datetime
    label_available_timestamp: datetime
    payload: Any
    payload_summary: float
    payload_dimension: int


def _canonical_config(value: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): (
            [float(item) if isinstance(item, (int, float)) else item for item in item_value]
            if isinstance(item_value, list)
            else item_value
        )
        for key, item_value in value.items()
    }


def conformal_grid(contracts: FrozenContracts) -> tuple[ConformalConfiguration, ...]:
    entries = contracts.baseline_registry["mandatory_conformal_and_ensemble_baselines"]
    matches = [entry for entry in entries if entry["id"] == "TunedSingleConformal"]
    if len(matches) != 1:
        raise RuntimeError("冻结基线注册表中的TunedSingleConformal定义数量异常")
    grid = matches[0]["candidate_grid"]
    configurations: list[ConformalConfiguration] = []
    complexity_rank = 0
    for family, family_configs in grid.items():
        for raw_config in family_configs:
            config = _canonical_config(dict(raw_config))
            configuration_id = canonical_json_sha256(
                {
                    "baseline_id": "TunedSingleConformal",
                    "family": str(family),
                    "configuration": config,
                }
            )
            configurations.append(
                ConformalConfiguration(
                    family=str(family),
                    configuration=config,
                    configuration_id=configuration_id,
                    complexity_rank=complexity_rank,
                )
            )
            complexity_rank += 1
    if len(configurations) != 19 or len({item.configuration_id for item in configurations}) != 19:
        raise RuntimeError("TunedSingleConformal冻结候选应恰有19项唯一配置")
    return tuple(configurations)


def validate_conformal_configuration(
    *,
    contracts: FrozenContracts,
    family: str,
    configuration: dict[str, Any],
) -> ConformalConfiguration:
    normalized = _canonical_config(dict(configuration))
    matches = [
        item
        for item in conformal_grid(contracts)
        if item.family == str(family) and item.configuration == normalized
    ]
    if len(matches) != 1:
        raise ValueError(f"TunedSingleConformal配置不在冻结19项网格中: {family} {normalized}")
    return matches[0]


def _conformity_score(target: float, lower: float, upper: float) -> float:
    return max(float(lower) - float(target), float(target) - float(upper), 0.0)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if len(values) == 0:
        return 0.0
    sorter = np.argsort(values, kind="mergesort")
    ordered_values = np.asarray(values, dtype=float)[sorter]
    ordered_weights = np.asarray(weights, dtype=float)[sorter]
    if not np.isfinite(ordered_weights).all() or float(ordered_weights.sum()) <= 0.0:
        ordered_weights = np.ones_like(ordered_values, dtype=float)
    cdf = np.cumsum(ordered_weights) / float(ordered_weights.sum())
    return float(np.interp(float(quantile), cdf, ordered_values))


def _clip_interval(lower: float, upper: float) -> tuple[float, float]:
    clipped_lower = float(np.clip(float(lower), 0.0, 1.0))
    clipped_upper = float(np.clip(float(upper), 0.0, 1.0))
    if clipped_lower > clipped_upper:
        raise RuntimeError("TunedSingleConformal裁剪后区间反转")
    return clipped_lower, clipped_upper


def _prune_physical_history(
    history: deque[tuple[datetime, Any]],
    *,
    current_time: datetime,
    duration_hours: float,
) -> None:
    cutoff = current_time - timedelta(hours=float(duration_hours))
    while history and history[0][0] < cutoff:
        history.popleft()


def _prepare_base_stream(
    *,
    contracts: FrozenContracts,
    base_predictions: pd.DataFrame,
    target_coverage: float,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    missing = [field for field in BASE_REQUIRED_FIELDS if field not in base_predictions.columns]
    if missing:
        raise ValueError(f"TunedSingleConformal基础预测缺少字段: {missing}")
    coverage = float(target_coverage)
    if coverage not in contracts.coverages:
        raise ValueError(f"TunedSingleConformal覆盖率不在冻结十一档: {coverage}")
    alpha = 1.0 - coverage
    lower_name = f"q_{alpha / 2.0:.4f}"
    upper_name = f"q_{1.0 - alpha / 2.0:.4f}"
    center_name = "q_0.5000"
    quantile_fields = (lower_name, upper_name, center_name)
    missing_quantiles = [field for field in quantile_fields if field not in base_predictions.columns]
    if missing_quantiles:
        raise ValueError(f"基础预测缺少目标覆盖率分位数: {missing_quantiles}")
    frame = base_predictions.copy()
    if set(frame["split"].astype(str)) != {"calibration", "test"}:
        raise ValueError("TunedSingleConformal基础预测必须精确包含calibration与test")
    for field in ("issue_timestamp", "label_timestamp", "label_available_timestamp"):
        frame[field] = pd.to_datetime(frame[field], errors="raise")
    if not (frame["issue_timestamp"] < frame["label_timestamp"]).all():
        raise ValueError("基础预测label_timestamp必须晚于issue_timestamp")
    if not (frame["label_timestamp"] < frame["label_available_timestamp"]).all():
        raise ValueError("基础预测label_available_timestamp必须严格晚于label_timestamp")
    numeric_fields = ["target", "horizon_steps", "nominal_cadence_minutes", *quantile_fields]
    numeric = frame.loc[:, numeric_fields].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("TunedSingleConformal基础预测含非有限数值")
    frame["base_lower"] = numeric[lower_name].astype(float)
    frame["base_upper"] = numeric[upper_name].astype(float)
    frame["base_center"] = numeric[center_name].astype(float)
    if (frame["base_lower"] > frame["base_upper"]).any():
        raise ValueError("基础预测目标分位数仍存在交叉")
    calibration = (
        frame[frame["split"] == "calibration"]
        .sort_values(["issue_timestamp", "label_timestamp"], kind="mergesort")
        .reset_index(drop=True)
    )
    test = (
        frame[frame["split"] == "test"]
        .sort_values(["issue_timestamp", "label_timestamp"], kind="mergesort")
        .reset_index(drop=True)
    )
    if calibration.empty or test.empty:
        raise ValueError("TunedSingleConformal校准或测试划分为空")
    if calibration["issue_timestamp"].duplicated().any() or test["issue_timestamp"].duplicated().any():
        raise ValueError("单一conformal流的问题时刻必须唯一")
    first_test_issue = as_datetime(test["issue_timestamp"].iloc[0])
    if as_datetime(calibration["label_available_timestamp"].max()) > first_test_issue:
        raise ValueError("校准标签在测试起点尚未全部可用")
    return calibration, test, lower_name, upper_name


def run_conformal_configuration(
    *,
    contracts: FrozenContracts,
    base_predictions: pd.DataFrame,
    dataset_id: str,
    zone_or_farm: str,
    predictor: str,
    seed: int,
    target_coverage: float,
    family: str,
    configuration: dict[str, Any],
    drain_final_feedback: bool = True,
) -> ConformalRunResult:
    selected_config = validate_conformal_configuration(
        contracts=contracts,
        family=family,
        configuration=configuration,
    )
    calibration, test, _, _ = _prepare_base_stream(
        contracts=contracts,
        base_predictions=base_predictions,
        target_coverage=target_coverage,
    )
    alpha = 1.0 - float(target_coverage)
    calibration_scores = np.asarray(
        [
            _conformity_score(row.target, row.base_lower, row.base_upper)
            for row in calibration.itertuples(index=False)
        ],
        dtype=float,
    )
    calibration_residuals = (
        calibration["target"].to_numpy(dtype=float)
        - calibration["base_center"].to_numpy(dtype=float)
    )
    calibration_labels = [as_datetime(value) for value in calibration["label_timestamp"]]
    static_margin = max(0.0, float(np.quantile(calibration_scores, 1.0 - alpha, method="linear")))
    cadence_minutes_values = set(float(value) for value in test["nominal_cadence_minutes"])
    if len(cadence_minutes_values) != 1:
        raise ValueError("单一conformal流的名义时间间隔必须唯一")
    cadence_hours = next(iter(cadence_minutes_values)) / 60.0
    if cadence_hours <= 0.0:
        raise ValueError("名义时间间隔必须为正")

    config = selected_config.configuration
    pending: list[_PendingConformalFeedback] = []
    feedback_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    decision_records: list[BaselineDecision] = []
    revision = 0
    future_feedback_violation_count = 0
    within_issue_feedback_use_count = 0

    offset = static_margin
    offsets: list[float] = []
    expert_losses: np.ndarray | None = None
    recent_misses: deque[tuple[datetime, float]] = deque()
    score_history: deque[tuple[datetime, float]] = deque(
        (label, float(score)) for label, score in zip(calibration_labels, calibration_scores)
    )
    residual_history: deque[tuple[datetime, float]] = deque(
        (label, float(residual)) for label, residual in zip(calibration_labels, calibration_residuals)
    )
    coverage_histories: list[deque[tuple[datetime, float]]] = []
    reference_width = float(
        np.quantile(
            calibration["base_upper"].to_numpy(dtype=float)
            - calibration["base_lower"].to_numpy(dtype=float),
            0.5,
            method="linear",
        )
    )

    if family == "AgACI":
        rates = [float(value) for value in config["learning_rates"]]
        offsets = [static_margin for _ in rates]
        expert_losses = np.zeros(len(rates), dtype=float)
    elif family == "SAOCP":
        windows = [float(value) for value in config["window_hours"]]
        offsets = [static_margin for _ in windows]
        coverage_histories = [deque() for _ in windows]

    def apply_payload(item: _PendingConformalFeedback, application_time: datetime) -> None:
        nonlocal offset, revision, future_feedback_violation_count
        if not item.label_timestamp < application_time:
            future_feedback_violation_count += 1
            raise RuntimeError("TunedSingleConformal检测到非严格成熟反馈")
        before = revision
        updated = family != "SplitCF"
        if family == "ACI":
            offset = max(0.0, offset + float(config["learning_rate"]) * (float(item.payload) - alpha))
        elif family == "AgACI":
            assert expert_losses is not None
            rates = [float(value) for value in config["learning_rates"]]
            misses = np.asarray(item.payload, dtype=float)
            for index, rate in enumerate(rates):
                offsets[index] = max(0.0, offsets[index] + rate * (float(misses[index]) - alpha))
                expert_losses[index] += float(misses[index])
        elif family == "FACI":
            recent_misses.append((item.label_timestamp, float(item.payload)))
            _prune_physical_history(
                recent_misses,
                current_time=application_time,
                duration_hours=float(config["adapt_window_hours"]),
            )
            miss_rate = float(np.mean([value for _, value in recent_misses]))
            eta_min = float(config["eta_min"])
            eta_max = float(config["eta_max"])
            eta = eta_min + (eta_max - eta_min) * min(
                1.0,
                abs(miss_rate - alpha) / max(alpha, 1e-12),
            )
            offset = max(0.0, offset + eta * (float(item.payload) - alpha))
        elif family in {"EnbPI_RH", "SPCI", "NEX"}:
            score_history.append((item.label_timestamp, float(item.payload)))
        elif family == "SAOCP":
            misses = np.asarray(item.payload, dtype=float)
            windows = [float(value) for value in config["window_hours"]]
            minimum_window = min(windows)
            for index, window in enumerate(windows):
                coverage_histories[index].append((item.label_timestamp, 1.0 - float(misses[index])))
                _prune_physical_history(
                    coverage_histories[index],
                    current_time=application_time,
                    duration_hours=window,
                )
                scale = float(np.sqrt(minimum_window / window))
                offsets[index] = max(
                    0.0,
                    offsets[index]
                    + float(config["learning_rate"]) * scale * (float(misses[index]) - alpha),
                )
        elif family == "KOWCPI":
            residual_history.append((item.label_timestamp, float(item.payload)))
        elif family == "WACI":
            miss_value, adapt_multiplier = item.payload
            offset = max(
                0.0,
                offset
                + float(config["learning_rate"])
                * float(adapt_multiplier)
                * (float(miss_value) - alpha),
            )
        elif family != "SplitCF":
            raise RuntimeError(f"未实现的conformal反馈家族: {family}")
        if updated:
            revision += 1
        feedback_records.append(
            {
                "event_id": item.event_id,
                "baseline_id": "TunedSingleConformal",
                "family": family,
                "configuration_id": selected_config.configuration_id,
                "label_timestamp": item.label_timestamp,
                "label_available_timestamp": item.label_available_timestamp,
                "feedback_application_timestamp": application_time,
                "eligible_by_strict_time_rule": item.label_timestamp < application_time,
                "calibrator_updated": updated,
                "feedback_revision_before": before,
                "feedback_revision_after": revision,
                "payload_summary": item.payload_summary,
                "payload_dimension": item.payload_dimension,
            }
        )

    def apply_matured(issue_timestamp: datetime) -> int:
        nonlocal pending
        matured = [
            item
            for item in pending
            if item.label_available_timestamp <= issue_timestamp
            and item.label_timestamp < issue_timestamp
        ]
        matured.sort(
            key=lambda item: (
                item.label_available_timestamp,
                item.label_timestamp,
                item.event_id,
            )
        )
        for item in matured:
            apply_payload(item, issue_timestamp)
        matured_ids = {item.event_id for item in matured}
        pending = [item for item in pending if item.event_id not in matured_ids]
        return len(matured)

    for row in test.itertuples(index=False):
        issue = as_datetime(row.issue_timestamp)
        applied = apply_matured(issue)
        feedback_count_before_decision = len(feedback_records)
        base_lower = float(row.base_lower)
        base_upper = float(row.base_upper)
        base_center = float(row.base_center)
        target = float(row.target)
        state_value = 0.0
        payload: Any
        payload_summary: float
        payload_dimension = 1

        if family == "SplitCF":
            raw_lower = base_lower - static_margin
            raw_upper = base_upper + static_margin
            lower, upper = _clip_interval(raw_lower, raw_upper)
            payload = _conformity_score(target, base_lower, base_upper)
            payload_summary = float(payload)
            state_value = static_margin
        elif family == "ACI":
            raw_lower = base_lower - offset
            raw_upper = base_upper + offset
            lower, upper = _clip_interval(raw_lower, raw_upper)
            payload = 0.0 if lower <= target <= upper else 1.0
            payload_summary = float(payload)
            state_value = offset
        elif family == "AgACI":
            assert expert_losses is not None
            shifted = expert_losses - float(expert_losses.min())
            weights = np.exp(-shifted)
            weights = weights / float(weights.sum())
            raw_lowers = np.asarray([base_lower - value for value in offsets], dtype=float)
            raw_uppers = np.asarray([base_upper + value for value in offsets], dtype=float)
            raw_lower = float(weights @ raw_lowers)
            raw_upper = float(weights @ raw_uppers)
            lower, upper = _clip_interval(raw_lower, raw_upper)
            expert_bounds = [_clip_interval(lo, up) for lo, up in zip(raw_lowers, raw_uppers)]
            payload = np.asarray(
                [0.0 if lo <= target <= up else 1.0 for lo, up in expert_bounds],
                dtype=float,
            )
            payload_summary = float(payload.mean())
            payload_dimension = len(payload)
            state_value = float(weights @ np.asarray(offsets, dtype=float))
        elif family == "FACI":
            raw_lower = base_lower - offset
            raw_upper = base_upper + offset
            lower, upper = _clip_interval(raw_lower, raw_upper)
            payload = 0.0 if lower <= target <= upper else 1.0
            payload_summary = float(payload)
            state_value = offset
        elif family == "EnbPI_RH":
            duration = float(config["history_duration_hours"])
            _prune_physical_history(score_history, current_time=issue, duration_hours=duration)
            values = np.asarray([value for _, value in score_history], dtype=float)
            margin = max(0.0, float(np.quantile(values, 1.0 - alpha, method="linear"))) if len(values) else 0.0
            raw_lower = base_lower - margin
            raw_upper = base_upper + margin
            lower, upper = _clip_interval(raw_lower, raw_upper)
            payload = _conformity_score(target, base_lower, base_upper)
            payload_summary = float(payload)
            state_value = margin
        elif family == "SAOCP":
            windows = [float(value) for value in config["window_hours"]]
            coverage_errors = []
            for index, window in enumerate(windows):
                _prune_physical_history(
                    coverage_histories[index],
                    current_time=issue,
                    duration_hours=window,
                )
                observed_coverage = (
                    float(np.mean([value for _, value in coverage_histories[index]]))
                    if coverage_histories[index]
                    else float(target_coverage)
                )
                coverage_errors.append(abs(observed_coverage - float(target_coverage)))
            shifted = np.asarray(coverage_errors, dtype=float) - min(coverage_errors)
            weights = np.exp(-shifted / max(float(config["temperature"]), 1e-12))
            weights = weights / float(weights.sum())
            raw_lowers = np.asarray([base_lower - value for value in offsets], dtype=float)
            raw_uppers = np.asarray([base_upper + value for value in offsets], dtype=float)
            raw_lower = float(weights @ raw_lowers)
            raw_upper = float(weights @ raw_uppers)
            lower, upper = _clip_interval(raw_lower, raw_upper)
            expert_bounds = [_clip_interval(lo, up) for lo, up in zip(raw_lowers, raw_uppers)]
            payload = np.asarray(
                [0.0 if lo <= target <= up else 1.0 for lo, up in expert_bounds],
                dtype=float,
            )
            payload_summary = float(payload.mean())
            payload_dimension = len(payload)
            state_value = float(weights @ np.asarray(offsets, dtype=float))
        elif family in {"SPCI", "NEX"}:
            duration = float(config["history_duration_hours"])
            _prune_physical_history(score_history, current_time=issue, duration_hours=duration)
            values = np.asarray([value for _, value in score_history], dtype=float)
            labels = [label for label, _ in score_history]
            if len(values):
                age_steps = np.asarray(
                    [max((issue - label).total_seconds() / 3600.0 / cadence_hours, 0.0) for label in labels],
                    dtype=float,
                )
                weights = float(config["decay"]) ** age_steps
                margin = max(0.0, _weighted_quantile(values, weights, 1.0 - alpha))
            else:
                margin = 0.0
            raw_lower = base_lower - margin
            raw_upper = base_upper + margin
            lower, upper = _clip_interval(raw_lower, raw_upper)
            payload = _conformity_score(target, base_lower, base_upper)
            payload_summary = float(payload)
            state_value = margin
        elif family == "KOWCPI":
            duration = float(config["history_duration_hours"])
            lag_duration = float(config["lag_duration_hours"])
            _prune_physical_history(residual_history, current_time=issue, duration_hours=duration)
            labels = np.asarray([np.datetime64(label, "ns") for label, _ in residual_history])
            residuals = np.asarray([value for _, value in residual_history], dtype=float)
            if len(residuals) == 0:
                q_low = q_high = 0.0
            else:
                issue_ns = np.datetime64(issue, "ns")
                lag_ns = np.timedelta64(int(round(lag_duration * 3600.0 * 1e9)), "ns")
                current_start = np.searchsorted(labels, issue_ns - lag_ns, side="left")
                current_context = float(np.mean(residuals[current_start:])) if current_start < len(residuals) else 0.0
                prefix = np.concatenate(([0.0], np.cumsum(residuals)))
                starts = np.searchsorted(labels, labels - lag_ns, side="left")
                indices = np.arange(len(residuals))
                counts = indices - starts
                valid = counts > 0
                if valid.any():
                    historical_context = (prefix[indices[valid]] - prefix[starts[valid]]) / counts[valid]
                    candidate_residuals = residuals[valid]
                    distance = np.abs(historical_context - current_context)
                    bandwidth = max(float(config["bandwidth"]), 1e-12)
                    u = distance / bandwidth
                    weights = np.where(np.abs(u) <= 1.0, np.maximum(0.75 * (1.0 - u ** 2), 0.0), 0.0)
                else:
                    candidate_residuals = residuals
                    weights = np.ones_like(residuals, dtype=float)
                best_width = float("inf")
                q_low = q_high = 0.0
                for beta in np.linspace(0.0, alpha, 21):
                    low_value = _weighted_quantile(candidate_residuals, weights, float(beta))
                    high_value = _weighted_quantile(candidate_residuals, weights, float(1.0 - alpha + beta))
                    width = high_value - low_value
                    if high_value >= low_value and width < best_width:
                        best_width = float(width)
                        q_low = float(low_value)
                        q_high = float(high_value)
            raw_lower = base_center + q_low
            raw_upper = base_center + q_high
            lower, upper = _clip_interval(raw_lower, raw_upper)
            payload = target - base_center
            payload_summary = float(payload)
            state_value = q_high - q_low
        elif family == "WACI":
            width = max(base_upper - base_lower, 1e-12)
            adapt_multiplier = float(
                np.clip(
                    (max(reference_width, 1e-12) / width) ** float(config["width_power"]),
                    float(config["adapt_floor"]),
                    float(config["adapt_cap"]),
                )
            )
            raw_lower = base_lower - offset
            raw_upper = base_upper + offset
            lower, upper = _clip_interval(raw_lower, raw_upper)
            miss = 0.0 if lower <= target <= upper else 1.0
            payload = (miss, adapt_multiplier)
            payload_summary = float(miss)
            payload_dimension = 2
            state_value = offset
        else:
            raise RuntimeError(f"未实现的TunedSingleConformal家族: {family}")

        if len(feedback_records) != feedback_count_before_decision:
            within_issue_feedback_use_count += 1
            raise RuntimeError("TunedSingleConformal决策阶段使用了当前事件反馈")
        event_id = contracts.make_event_id(
            dataset_id=dataset_id,
            zone_or_farm=zone_or_farm,
            predictor=predictor,
            horizon_steps=int(row.horizon_steps),
            seed=int(seed),
            target_coverage=float(target_coverage),
            issue_timestamp=issue,
        )
        candidate_records.append(
            {
                "event_id": event_id,
                "baseline_id": "TunedSingleConformal",
                "family": family,
                "configuration_id": selected_config.configuration_id,
                "zone_or_farm": str(zone_or_farm),
                "predictor": str(predictor),
                "seed": int(seed),
                "target_coverage": float(target_coverage),
                "issue_timestamp": issue,
                "label_timestamp": as_datetime(row.label_timestamp),
                "label_available_timestamp": as_datetime(row.label_available_timestamp),
                "base_center": base_center,
                "target": target,
                "candidate_lower": lower,
                "candidate_upper": upper,
                "preclip_lower": raw_lower,
                "preclip_upper": raw_upper,
                "candidate_clip_applied": lower != raw_lower or upper != raw_upper,
                "feedback_revision_before": revision,
                "feedback_applied_at_issue": applied,
                "calibrator_state_value_before": state_value,
            }
        )
        decision_records.append(
            BaselineDecision(
                event_id=event_id,
                baseline_id="TunedSingleConformal",
                selected_action=None,
                selected_lower=lower,
                selected_upper=upper,
                decision_score=None,
                decision_reason=f"{family}:{selected_config.configuration_id}",
            )
        )
        pending.append(
            _PendingConformalFeedback(
                event_id=event_id,
                label_timestamp=as_datetime(row.label_timestamp),
                label_available_timestamp=as_datetime(row.label_available_timestamp),
                payload=payload,
                payload_summary=payload_summary,
                payload_dimension=payload_dimension,
            )
        )

    if drain_final_feedback:
        pending.sort(
            key=lambda item: (
                item.label_available_timestamp,
                item.label_timestamp,
                item.event_id,
            )
        )
        for item in pending:
            apply_payload(item, item.label_available_timestamp)
        pending = []

    candidates = pd.DataFrame(candidate_records).sort_values("event_id", kind="mergesort").reset_index(drop=True)
    feedback = pd.DataFrame(feedback_records)
    if not feedback.empty:
        feedback = feedback.sort_values(
            ["feedback_application_timestamp", "event_id"],
            kind="mergesort",
        ).reset_index(drop=True)
    decisions = tuple(sorted(decision_records, key=lambda item: item.event_id))
    audit = {
        "baseline_id": "TunedSingleConformal",
        "family": family,
        "configuration": config,
        "configuration_id": selected_config.configuration_id,
        "complexity_rank": selected_config.complexity_rank,
        "target_coverage": float(target_coverage),
        "calibration_row_count": len(calibration),
        "test_event_count": len(test),
        "feedback_count": len(feedback),
        "final_revision": revision,
        "pending_feedback_count": len(pending),
        "future_feedback_violation_count": future_feedback_violation_count,
        "within_issue_feedback_use_count": within_issue_feedback_use_count,
        "strict_delayed_feedback": True,
        "calibration_available_before_test": True,
        "drain_final_feedback": bool(drain_final_feedback),
        "invalid_interval_count": int((candidates["candidate_lower"] > candidates["candidate_upper"]).sum()),
        "out_of_bounds_count": int(
            ((candidates["candidate_lower"] < 0.0) | (candidates["candidate_upper"] > 1.0)).sum()
        ),
    }
    return ConformalRunResult(
        configuration=selected_config,
        decisions=decisions,
        candidates=candidates,
        feedback=feedback,
        audit=audit,
    )
