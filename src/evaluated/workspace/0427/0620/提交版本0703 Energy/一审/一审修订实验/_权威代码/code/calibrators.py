"""Calibration methods for 0427.

All methods extracted from 0407 run_calibration.py with a unified interface.
Each method receives (pred_df, alpha, config) and returns a DataFrame with
columns: timestamp, target, lower, upper, base_lower, base_upper, base_center.
"""
from __future__ import annotations

from collections import deque
from typing import Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def conformity_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.maximum.reduce([lower - y, y - upper, np.zeros_like(y)])


def lower_nonconformity(y: np.ndarray, lower: np.ndarray) -> np.ndarray:
    return np.maximum(lower - y, 0.0)


def upper_nonconformity(y: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.maximum(y - upper, 0.0)


def fit_static_margin(cal_df: pd.DataFrame, alpha: float) -> float:
    scores = conformity_score(
        cal_df["target"].to_numpy(),
        cal_df["base_lower"].to_numpy(),
        cal_df["base_upper"].to_numpy(),
    )
    return max(0.0, float(np.quantile(scores, 1.0 - alpha)))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if len(values) == 0:
        return 0.0
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cdf = np.cumsum(weights) / np.sum(weights)
    return float(np.interp(quantile, cdf, values))


def boundary_proximity(center: np.ndarray) -> np.ndarray:
    dist = np.minimum(center, 1.0 - center)
    return np.clip(1.0 - dist / 0.5, 0.0, 1.0)


def detect_regime(center: float, lag1: float, ramp_threshold: float = 0.12) -> str:
    if center < 0.2:
        return "low_power"
    if abs(center - lag1) >= ramp_threshold:
        return "ramping"
    if center >= 0.9:
        return "near_rated"
    if center >= 0.7:
        return "platform"
    return "default"


REGIME_FACTORS = {
    "low_power": {"lower": 1.35, "upper": 0.90, "shrink": 1.10},
    "ramping":   {"lower": 1.10, "upper": 1.20, "shrink": 0.80},
    "near_rated": {"lower": 0.90, "upper": 1.35, "shrink": 1.05},
    "platform":  {"lower": 1.00, "upper": 1.15, "shrink": 0.95},
    "default":   {"lower": 1.00, "upper": 1.00, "shrink": 1.00},
}


def add_state_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lag1 = out.get("feature_target_lag1", out["base_center"]).astype(float)
    out["state_width"] = out["base_upper"] - out["base_lower"]
    out["state_lag1"] = lag1
    out["state_ramp_abs"] = np.abs(out["base_center"] - lag1)
    out["state_boundary_proximity"] = boundary_proximity(out["base_center"].to_numpy())
    out["regime"] = [
        detect_regime(float(c), float(l))
        for c, l in zip(out["base_center"].to_numpy(), lag1.to_numpy())
    ]
    for rn in ["low_power", "ramping", "near_rated", "platform", "default"]:
        out[f"regime_{rn}"] = (out["regime"] == rn).astype(float)
    return out


def directional_regime_quantiles(
    cal_df: pd.DataFrame, alpha: float,
) -> tuple[dict, dict, float, float]:
    lower_score = lower_nonconformity(cal_df["target"].to_numpy(), cal_df["base_lower"].to_numpy())
    upper_score = upper_nonconformity(cal_df["target"].to_numpy(), cal_df["base_upper"].to_numpy())
    cal_df = cal_df.copy()
    cal_df["lower_score"] = lower_score
    cal_df["upper_score"] = upper_score
    global_lower = max(0.0, float(np.quantile(lower_score, 1.0 - alpha / 2.0)))
    global_upper = max(0.0, float(np.quantile(upper_score, 1.0 - alpha / 2.0)))
    reg_lower, reg_upper = {}, {}
    for regime, group in cal_df.groupby("regime"):
        reg_lower[regime] = max(0.0, float(np.quantile(group["lower_score"].to_numpy(), 1.0 - alpha / 2.0)))
        reg_upper[regime] = max(0.0, float(np.quantile(group["upper_score"].to_numpy(), 1.0 - alpha / 2.0)))
    return reg_lower, reg_upper, global_lower, global_upper


def _clip(lower: float, upper: float, clip_final: bool):
    if clip_final:
        l, u = np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0)
        return float(l), float(u)
    return lower, upper


def label_delay_steps(cfg: dict) -> int:
    if cfg.get("update_policy") != "causal_issue_time":
        return 0
    return max(0, int(cfg.get("label_delay_steps", 0)))


def release_due(pending: deque, row_idx: int, update_fn) -> int:
    released = 0
    while pending and pending[0][0] <= row_idx:
        _, payload = pending.popleft()
        update_fn(payload)
        released += 1
    return released


def uses_strict_timestamp_feedback(cfg: dict) -> bool:
    return (
        cfg.get("feedback_rule")
        == "label_timestamp_strictly_before_issue_timestamp"
        or cfg.get("update_policy") == "causal_issue_time"
    )


def strict_event_times(
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    required = {"timestamp", "label_timestamp"}
    missing = sorted(required.difference(test_df.columns))
    if missing:
        raise ValueError(f"严格反馈缺少时间字段: {missing}")
    issue = pd.to_datetime(test_df["timestamp"], errors="raise").to_numpy()
    label = pd.to_datetime(test_df["label_timestamp"], errors="raise").to_numpy()
    issue_ns = issue.astype("datetime64[ns]")
    label_ns = label.astype("datetime64[ns]")
    if pd.isna(issue_ns).any() or pd.isna(label_ns).any():
        raise ValueError("严格反馈时间字段不能包含缺失值")
    if len(issue_ns) > 1 and np.any(issue_ns[1:] <= issue_ns[:-1]):
        raise ValueError("单一候选流的问题时刻必须严格递增且唯一")
    if np.any(label_ns <= issue_ns):
        raise ValueError("标签时刻必须严格晚于对应问题时刻")
    return issue_ns, label_ns


def release_strictly_matured(
    pending: deque,
    issue_timestamp: np.datetime64,
    update_fn,
) -> int:
    """只释放标签时刻严格早于当前问题时刻的反馈。"""
    released = 0
    while pending and pending[0][0] < issue_timestamp:
        _, payload = pending.popleft()
        update_fn(payload)
        released += 1
    return released


def _result_frame(
    test_df: pd.DataFrame,
    lower: np.ndarray,
    upper: np.ndarray,
    preclip_lower: np.ndarray | None = None,
    preclip_upper: np.ndarray | None = None,
    diagnostics: dict[str, np.ndarray | list | float | int | str] | None = None,
) -> pd.DataFrame:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if len(lower) != len(test_df) or len(upper) != len(test_df):
        raise ValueError("候选区间与测试事件行数不一致")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("候选区间包含非有限值")
    if np.any(lower > upper):
        raise ValueError("候选区间存在下界高于上界")

    metadata_columns = [
        "timestamp",
        "issue_timestamp",
        "label_timestamp",
        "label_available_timestamp",
        "issue_row_index",
        "label_row_index",
        "horizon_steps",
        "nominal_cadence_minutes",
        "lead_time_minutes",
        "lead_time_hours",
        "target_lag1_timestamp",
        "feature_target_lag1",
        "split",
        "target",
        "base_lower",
        "base_upper",
        "base_center",
        "quantile_crossing_before",
        "quantile_rearrangement_mean_abs_adjustment",
        "quantile_rearrangement_max_abs_adjustment",
        "quantile_rearrangement_method",
    ]
    available_columns = [
        column for column in metadata_columns if column in test_df.columns
    ]
    result = test_df.loc[:, available_columns].copy().reset_index(drop=True)
    if "issue_timestamp" not in result.columns:
        result["issue_timestamp"] = pd.to_datetime(result["timestamp"])
    result["lower"] = lower
    result["upper"] = upper

    raw_lower = lower if preclip_lower is None else np.asarray(preclip_lower, dtype=float)
    raw_upper = upper if preclip_upper is None else np.asarray(preclip_upper, dtype=float)
    if len(raw_lower) != len(result) or len(raw_upper) != len(result):
        raise ValueError("裁剪前区间与测试事件行数不一致")
    result["preclip_lower"] = raw_lower
    result["preclip_upper"] = raw_upper
    result["candidate_clip_applied"] = (
        (np.abs(raw_lower - lower) > 0.0)
        | (np.abs(raw_upper - upper) > 0.0)
    )

    if diagnostics:
        for name, values in diagnostics.items():
            if np.isscalar(values) or isinstance(values, str):
                result[name] = values
                continue
            array = np.asarray(values)
            if len(array) != len(result):
                raise ValueError(f"诊断字段{name}与测试事件行数不一致")
            result[name] = array
    return result


# ---------------------------------------------------------------------------
# Static methods
# ---------------------------------------------------------------------------

def run_split_conformal(pred_df: pd.DataFrame, alpha: float, cfg: dict) -> pd.DataFrame:
    cal_df = pred_df[pred_df["split"] == "calibration"]
    test_df = pred_df[pred_df["split"] == "test"].copy().sort_values("timestamp")
    clip_final = cfg.get("clip_final", True)
    margin = fit_static_margin(cal_df, alpha)
    preclip_lower = test_df["base_lower"].to_numpy(dtype=float) - margin
    preclip_upper = test_df["base_upper"].to_numpy(dtype=float) + margin
    lower = preclip_lower.copy()
    upper = preclip_upper.copy()
    if clip_final:
        lower = np.clip(lower, 0.0, 1.0)
        upper = np.clip(upper, 0.0, 1.0)
    return _result_frame(
        test_df,
        lower,
        upper,
        preclip_lower,
        preclip_upper,
        diagnostics={
            "feedback_revision_before": np.zeros(len(test_df), dtype=np.int64),
            "feedback_applied_at_issue": np.zeros(len(test_df), dtype=np.int64),
            "calibrator_state_value_before": np.full(len(test_df), margin),
        },
    )


def run_cqr(pred_df: pd.DataFrame, alpha: float, cfg: dict) -> pd.DataFrame:
    return run_split_conformal(pred_df, alpha, {**cfg, "clip_final": True})


def run_lcf(pred_df: pd.DataFrame, alpha: float, cfg: dict) -> pd.DataFrame:
    """LCF: regime-conditional static conformal (l12_style_wind_approx)."""
    cal_df = add_state_columns(pred_df[pred_df["split"] == "calibration"].copy())
    test_df = add_state_columns(pred_df[pred_df["split"] == "test"].copy())
    clip_final = cfg.get("clip_final", True)

    scores = conformity_score(
        cal_df["target"].to_numpy(), cal_df["base_lower"].to_numpy(), cal_df["base_upper"].to_numpy(),
    )
    cal_df["score"] = scores
    global_margin = max(0.0, float(np.quantile(scores, 1.0 - alpha)))
    regime_margin = {
        regime: max(0.0, float(np.quantile(group["score"].to_numpy(), 1.0 - alpha)))
        for regime, group in cal_df.groupby("regime")
    }

    margins = test_df["regime"].map(regime_margin).fillna(global_margin).to_numpy(dtype=float)
    lower = test_df["base_lower"].to_numpy(dtype=float) - margins
    upper = test_df["base_upper"].to_numpy(dtype=float) + margins
    if clip_final:
        lower = np.clip(lower, 0.0, 1.0)
        upper = np.clip(upper, 0.0, 1.0)
    return _result_frame(test_df, lower, upper)


# ---------------------------------------------------------------------------
# Dynamic methods
# ---------------------------------------------------------------------------

def run_aci(pred_df: pd.DataFrame, alpha: float, cfg: dict) -> pd.DataFrame:
    lr = cfg.get("learning_rate", 0.05)
    clip_final = cfg.get("clip_final", True)
    delay = label_delay_steps(cfg)
    cal_df = pred_df[pred_df["split"] == "calibration"]
    test_df = pred_df[pred_df["split"] == "test"].copy().sort_values("timestamp")
    offset = fit_static_margin(cal_df, alpha)
    n = len(test_df)
    lower = np.empty(n, dtype=float)
    upper = np.empty(n, dtype=float)
    preclip_lower = np.empty(n, dtype=float)
    preclip_upper = np.empty(n, dtype=float)
    feedback_revision_before = np.zeros(n, dtype=np.int64)
    feedback_applied_at_issue = np.zeros(n, dtype=np.int64)
    state_before = np.empty(n, dtype=float)
    base_lower = test_df["base_lower"].to_numpy(dtype=float)
    base_upper = test_df["base_upper"].to_numpy(dtype=float)
    target = test_df["target"].to_numpy(dtype=float)
    pending: deque = deque()
    strict = uses_strict_timestamp_feedback(cfg)
    if strict:
        issue_timestamps, label_timestamps = strict_event_times(test_df)
    else:
        issue_timestamps = label_timestamps = None
    revision = 0

    def apply_update(miss_value: float) -> None:
        nonlocal offset
        offset = max(0.0, offset + lr * (float(miss_value) - alpha))

    for i in range(n):
        if strict:
            released = release_strictly_matured(
                pending,
                issue_timestamps[i],
                apply_update,
            )
        else:
            released = release_due(pending, i, apply_update)
        revision += released
        feedback_revision_before[i] = revision
        feedback_applied_at_issue[i] = released
        state_before[i] = offset
        raw_l = base_lower[i] - offset
        raw_u = base_upper[i] + offset
        l = raw_l
        u = raw_u
        if clip_final:
            l = float(np.clip(l, 0.0, 1.0))
            u = float(np.clip(u, 0.0, 1.0))
        preclip_lower[i] = raw_l
        preclip_upper[i] = raw_u
        lower[i] = l
        upper[i] = u
        miss = 0.0 if l <= target[i] <= u else 1.0
        if strict:
            pending.append((label_timestamps[i], miss))
        elif delay > 0:
            pending.append((i + delay, miss))
        else:
            apply_update(miss)
            revision += 1
    return _result_frame(
        test_df,
        lower,
        upper,
        preclip_lower,
        preclip_upper,
        diagnostics={
            "feedback_revision_before": feedback_revision_before,
            "feedback_applied_at_issue": feedback_applied_at_issue,
            "calibrator_state_value_before": state_before,
        },
    )


def run_agaci(pred_df: pd.DataFrame, alpha: float, cfg: dict) -> pd.DataFrame:
    learning_rates = cfg.get("learning_rates", [0.01, 0.02, 0.05, 0.1])
    clip_final = cfg.get("clip_final", True)
    delay = label_delay_steps(cfg)
    cal_df = pred_df[pred_df["split"] == "calibration"]
    test_df = pred_df[pred_df["split"] == "test"].copy().sort_values("timestamp")
    offsets = [fit_static_margin(cal_df, alpha) for _ in learning_rates]
    losses = np.zeros(len(learning_rates), dtype=float)
    n = len(test_df)
    lower = np.empty(n, dtype=float)
    upper = np.empty(n, dtype=float)
    preclip_lower = np.empty(n, dtype=float)
    preclip_upper = np.empty(n, dtype=float)
    feedback_revision_before = np.zeros(n, dtype=np.int64)
    feedback_applied_at_issue = np.zeros(n, dtype=np.int64)
    state_before = np.empty(n, dtype=float)
    base_lower = test_df["base_lower"].to_numpy(dtype=float)
    base_upper = test_df["base_upper"].to_numpy(dtype=float)
    target = test_df["target"].to_numpy(dtype=float)
    pending: deque = deque()
    strict = uses_strict_timestamp_feedback(cfg)
    if strict:
        issue_timestamps, label_timestamps = strict_event_times(test_df)
    else:
        issue_timestamps = label_timestamps = None
    revision = 0

    def apply_update(miss_flags: np.ndarray) -> None:
        for j, lr_val in enumerate(learning_rates):
            offsets[j] = max(0.0, offsets[j] + lr_val * (float(miss_flags[j]) - alpha))
            losses[j] += float(miss_flags[j])

    for row_idx in range(n):
        if strict:
            released = release_strictly_matured(
                pending,
                issue_timestamps[row_idx],
                apply_update,
            )
        else:
            released = release_due(pending, row_idx, apply_update)
        revision += released
        feedback_revision_before[row_idx] = revision
        feedback_applied_at_issue[row_idx] = released
        raw_lowers, raw_uppers = [], []
        for offset in offsets:
            l = base_lower[row_idx] - offset
            u = base_upper[row_idx] + offset
            raw_lowers.append(l)
            raw_uppers.append(u)
        shifted = losses - losses.min()
        weights = np.exp(-shifted)
        ws = float(weights.sum())
        weights = weights / ws if np.isfinite(ws) and ws > 0 else np.full(len(learning_rates), 1.0 / len(learning_rates))
        raw_l_mix = float(np.dot(weights, raw_lowers))
        raw_u_mix = float(np.dot(weights, raw_uppers))
        l_mix = raw_l_mix
        u_mix = raw_u_mix
        if clip_final:
            l_mix = float(np.clip(l_mix, 0.0, 1.0))
            u_mix = float(np.clip(u_mix, 0.0, 1.0))
        preclip_lower[row_idx] = raw_l_mix
        preclip_upper[row_idx] = raw_u_mix
        lower[row_idx] = l_mix
        upper[row_idx] = u_mix
        state_before[row_idx] = float(np.dot(weights, offsets))
        expert_lowers = np.asarray(raw_lowers, dtype=float)
        expert_uppers = np.asarray(raw_uppers, dtype=float)
        if clip_final:
            expert_lowers = np.clip(expert_lowers, 0.0, 1.0)
            expert_uppers = np.clip(expert_uppers, 0.0, 1.0)
        miss_flags = np.array([
            0.0 if l <= target[row_idx] <= u else 1.0
            for l, u in zip(expert_lowers, expert_uppers)
        ])
        if strict:
            pending.append((label_timestamps[row_idx], miss_flags))
        elif delay > 0:
            pending.append((row_idx + delay, miss_flags))
        else:
            apply_update(miss_flags)
            revision += 1
    return _result_frame(
        test_df,
        lower,
        upper,
        preclip_lower,
        preclip_upper,
        diagnostics={
            "feedback_revision_before": feedback_revision_before,
            "feedback_applied_at_issue": feedback_applied_at_issue,
            "calibrator_state_value_before": state_before,
        },
    )


def run_faci(pred_df: pd.DataFrame, alpha: float, cfg: dict) -> pd.DataFrame:
    eta_min = cfg.get("eta_min", 0.01)
    eta_max = cfg.get("eta_max", 0.1)
    adapt_window = cfg.get("adapt_window", 24)
    clip_final = cfg.get("clip_final", True)
    delay = label_delay_steps(cfg)
    cal_df = pred_df[pred_df["split"] == "calibration"]
    test_df = pred_df[pred_df["split"] == "test"].copy()
    offset = fit_static_margin(cal_df, alpha)
    recent = deque(maxlen=adapt_window)
    n = len(test_df)
    lower = np.empty(n, dtype=float)
    upper = np.empty(n, dtype=float)
    base_lower = test_df["base_lower"].to_numpy(dtype=float)
    base_upper = test_df["base_upper"].to_numpy(dtype=float)
    target = test_df["target"].to_numpy(dtype=float)
    pending: deque = deque()

    def apply_update(miss_value: float) -> None:
        nonlocal offset
        recent.append(float(miss_value))
        miss_rate = np.mean(recent)
        eta = eta_min + (eta_max - eta_min) * min(1.0, abs(miss_rate - alpha) / max(alpha, 1e-6))
        offset = max(0.0, offset + eta * (float(miss_value) - alpha))

    for i in range(n):
        release_due(pending, i, apply_update)
        l = base_lower[i] - offset
        u = base_upper[i] + offset
        if clip_final:
            l = float(np.clip(l, 0.0, 1.0))
            u = float(np.clip(u, 0.0, 1.0))
        lower[i] = l
        upper[i] = u
        miss = 0.0 if l <= target[i] <= u else 1.0
        if delay > 0:
            pending.append((i + delay, miss))
        else:
            apply_update(miss)
    return _result_frame(test_df, lower, upper)


def run_enbpi(pred_df: pd.DataFrame, alpha: float, cfg: dict) -> pd.DataFrame:
    history_window = int(cfg.get("history_window", 336))
    history_duration_hours = float(cfg.get("history_duration_hours", 336.0))
    if history_duration_hours <= 0.0:
        raise ValueError("history_duration_hours必须为正数")
    clip_final = cfg.get("clip_final", True)
    delay = label_delay_steps(cfg)
    cal_df = pred_df[pred_df["split"] == "calibration"].copy().sort_values("timestamp")
    test_df = pred_df[pred_df["split"] == "test"].copy().sort_values("timestamp")
    cal_scores = conformity_score(
        cal_df["target"].to_numpy(), cal_df["base_lower"].to_numpy(), cal_df["base_upper"].to_numpy(),
    )
    pending: deque = deque()
    n = len(test_df)
    lower = np.empty(n, dtype=float)
    upper = np.empty(n, dtype=float)
    preclip_lower = np.empty(n, dtype=float)
    preclip_upper = np.empty(n, dtype=float)
    feedback_revision_before = np.zeros(n, dtype=np.int64)
    feedback_applied_at_issue = np.zeros(n, dtype=np.int64)
    state_before = np.empty(n, dtype=float)
    history_n_before = np.zeros(n, dtype=np.int64)
    history_start_before = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    history_end_before = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    base_lower = test_df["base_lower"].to_numpy(dtype=float)
    base_upper = test_df["base_upper"].to_numpy(dtype=float)
    target = test_df["target"].to_numpy(dtype=float)
    strict = uses_strict_timestamp_feedback(cfg)
    revision = 0

    if strict:
        issue_timestamps, label_timestamps = strict_event_times(test_df)
        if "label_timestamp" not in cal_df.columns:
            raise ValueError("EnbPI严格物理窗口需要校准标签时刻")
        calibration_labels = pd.to_datetime(
            cal_df["label_timestamp"], errors="raise"
        ).to_numpy(dtype="datetime64[ns]")
        if len(issue_timestamps) and np.any(calibration_labels >= issue_timestamps[0]):
            raise ValueError("校准标签没有在测试起点前严格成熟")
        history: deque = deque(
            (label_timestamp, float(score))
            for label_timestamp, score in zip(calibration_labels, cal_scores)
        )

        def append_score(payload: tuple[np.datetime64, float]) -> None:
            label_timestamp, score_value = payload
            history.append((label_timestamp, float(score_value)))

    else:
        issue_timestamps = label_timestamps = None
        history = deque(float(score) for score in cal_scores)

        def append_score(score_value: float) -> None:
            history.append(float(score_value))

    for i in range(n):
        if strict:
            released = release_strictly_matured(
                pending,
                issue_timestamps[i],
                append_score,
            )
            cutoff = issue_timestamps[i] - np.timedelta64(
                int(round(history_duration_hours * 3600.0 * 1_000_000_000)),
                "ns",
            )
            while history and history[0][0] < cutoff:
                history.popleft()
            active_values = np.asarray(
                [score for _, score in history],
                dtype=float,
            )
            history_n_before[i] = len(history)
            if history:
                history_start_before[i] = history[0][0]
                history_end_before[i] = history[-1][0]
        else:
            released = release_due(pending, i, append_score)
            active_values = np.asarray(
                list(history)[-history_window:] if history_window > 0 else list(history),
                dtype=float,
            )
            history_n_before[i] = len(active_values)
        revision += released
        feedback_revision_before[i] = revision
        feedback_applied_at_issue[i] = released
        margin = max(0.0, float(np.quantile(active_values, 1.0 - alpha))) if len(active_values) else 0.0
        state_before[i] = margin
        raw_l = base_lower[i] - margin
        raw_u = base_upper[i] + margin
        l = raw_l
        u = raw_u
        if clip_final:
            l = float(np.clip(l, 0.0, 1.0))
            u = float(np.clip(u, 0.0, 1.0))
        preclip_lower[i] = raw_l
        preclip_upper[i] = raw_u
        lower[i] = l
        upper[i] = u
        score = conformity_score(np.array([target[i]]), np.array([base_lower[i]]), np.array([base_upper[i]]))[0]
        if strict:
            pending.append(
                (
                    label_timestamps[i],
                    (label_timestamps[i], float(score)),
                )
            )
        elif delay > 0:
            pending.append((i + delay, float(score)))
        else:
            append_score(float(score))
            revision += 1
    return _result_frame(
        test_df,
        lower,
        upper,
        preclip_lower,
        preclip_upper,
        diagnostics={
            "feedback_revision_before": feedback_revision_before,
            "feedback_applied_at_issue": feedback_applied_at_issue,
            "calibrator_state_value_before": state_before,
            "enbpi_history_n_before": history_n_before,
            "enbpi_history_start_before": history_start_before,
            "enbpi_history_end_before": history_end_before,
            "enbpi_history_duration_hours": history_duration_hours,
        },
    )


# ---------------------------------------------------------------------------
# Dynamic extension methods
# ---------------------------------------------------------------------------

def run_saocp(pred_df: pd.DataFrame, alpha: float, cfg: dict) -> pd.DataFrame:
    lr = cfg.get("learning_rate", 0.05)
    windows = cfg.get("windows", [24, 48, 96, 168])
    temperature = cfg.get("temperature", 0.1)
    clip_final = cfg.get("clip_final", True)
    cal_df = pred_df[pred_df["split"] == "calibration"]
    test_df = pred_df[pred_df["split"] == "test"].copy()
    base_offset = fit_static_margin(cal_df, alpha)
    target_cov = 1.0 - alpha
    offsets = [base_offset for _ in windows]
    recent_cov = [deque(maxlen=max(1, int(w))) for w in windows]
    min_w = max(1, min(int(w) for w in windows))
    rows = []
    for _, row in test_df.iterrows():
        e_lowers, e_uppers, e_cov_err, misses = [], [], [], []
        for idx, w in enumerate(windows):
            l, u = _clip(float(row["base_lower"]) - offsets[idx], float(row["base_upper"]) + offsets[idx], clip_final)
            e_lowers.append(l)
            e_uppers.append(u)
            cov = np.mean(recent_cov[idx]) if recent_cov[idx] else target_cov
            e_cov_err.append(abs(cov - target_cov))
            misses.append(0.0 if l <= row["target"] <= u else 1.0)
        cov_err = np.asarray(e_cov_err)
        shifted = cov_err - cov_err.min()
        weights = np.exp(-shifted / max(temperature, 1e-6))
        ws = float(weights.sum())
        weights = weights / ws if np.isfinite(ws) and ws > 0 else np.full(len(windows), 1.0 / len(windows))
        lower = float(np.dot(weights, np.asarray(e_lowers)))
        upper = float(np.dot(weights, np.asarray(e_uppers)))
        for idx, w in enumerate(windows):
            recent_cov[idx].append(1.0 - misses[idx])
            scale = float(np.sqrt(min_w / max(1, int(w))))
            offsets[idx] = max(0.0, offsets[idx] + lr * scale * (misses[idx] - alpha))
        rows.append({
            "timestamp": row["timestamp"], "target": row["target"],
            "lower": lower, "upper": upper,
            "base_lower": float(row["base_lower"]), "base_upper": float(row["base_upper"]),
            "base_center": float(row["base_center"]),
        })
    return pd.DataFrame(rows)


def run_pid(pred_df: pd.DataFrame, alpha: float, cfg: dict) -> pd.DataFrame:
    kp = cfg.get("kp", 0.05)
    ki = cfg.get("ki", 0.01)
    kd = cfg.get("kd", 0.02)
    clip_final = cfg.get("clip_final", True)
    cal_df = pred_df[pred_df["split"] == "calibration"]
    test_df = pred_df[pred_df["split"] == "test"].copy()
    offset = fit_static_margin(cal_df, alpha)
    target_cov = 1.0 - alpha
    integral, prev_error = 0.0, 0.0
    covered_recent = deque(maxlen=24)
    rows = []
    for _, row in test_df.iterrows():
        lower, upper = _clip(float(row["base_lower"]) - offset, float(row["base_upper"]) + offset, clip_final)
        covered = 1.0 if lower <= row["target"] <= upper else 0.0
        covered_recent.append(covered)
        error = target_cov - (np.mean(covered_recent) if covered_recent else target_cov)
        integral += error
        derivative = error - prev_error
        prev_error = error
        offset = max(0.0, offset + kp * error + ki * integral + kd * derivative)
        rows.append({
            "timestamp": row["timestamp"], "target": row["target"],
            "lower": lower, "upper": upper,
            "base_lower": float(row["base_lower"]), "base_upper": float(row["base_upper"]),
            "base_center": float(row["base_center"]),
        })
    return pd.DataFrame(rows)


def run_spci(pred_df: pd.DataFrame, alpha: float, cfg: dict) -> pd.DataFrame:
    history_window = cfg.get("history_window", 336)
    decay = cfg.get("decay", 0.995)
    clip_final = cfg.get("clip_final", True)
    delay = label_delay_steps(cfg)
    cal_df = pred_df[pred_df["split"] == "calibration"]
    test_df = pred_df[pred_df["split"] == "test"].copy()
    residual_history = list(conformity_score(
        cal_df["target"].to_numpy(), cal_df["base_lower"].to_numpy(), cal_df["base_upper"].to_numpy(),
    ))
    rows = []
    pending: deque = deque()

    def append_score(score_value: float) -> None:
        residual_history.append(float(score_value))

    for row_idx, row in enumerate(test_df.itertuples(index=False)):
        release_due(pending, row_idx, append_score)
        active = residual_history[-history_window:] if history_window > 0 else residual_history
        values = np.asarray(active, dtype=float)
        weights = decay ** np.arange(len(values) - 1, -1, -1, dtype=float)
        margin = max(0.0, weighted_quantile(values, weights, 1.0 - alpha))
        lower, upper = _clip(float(row.base_lower) - margin, float(row.base_upper) + margin, clip_final)
        rows.append({
            "timestamp": row.timestamp, "target": row.target,
            "lower": lower, "upper": upper,
            "base_lower": float(row.base_lower), "base_upper": float(row.base_upper),
            "base_center": float(row.base_center),
        })
        score = float(conformity_score(
            np.array([row.target]), np.array([row.base_lower]), np.array([row.base_upper]),
        )[0])
        if delay > 0:
            pending.append((row_idx + delay, score))
        else:
            append_score(score)
    return pd.DataFrame(rows)


def run_nex(pred_df: pd.DataFrame, alpha: float, cfg: dict) -> pd.DataFrame:
    """Non-exchangeable weighted CP (alias for weighted_cp_family)."""
    return run_weighted_cp(pred_df, alpha, cfg)


def run_weighted_cp(pred_df: pd.DataFrame, alpha: float, cfg: dict) -> pd.DataFrame:
    decay = cfg.get("decay", 0.99)
    history_window = cfg.get("history_window", 336)
    clip_final = cfg.get("clip_final", True)
    delay = label_delay_steps(cfg)
    cal_df = pred_df[pred_df["split"] == "calibration"]
    test_df = pred_df[pred_df["split"] == "test"].copy()
    history = list(conformity_score(
        cal_df["target"].to_numpy(), cal_df["base_lower"].to_numpy(), cal_df["base_upper"].to_numpy(),
    ))
    rows = []
    pending: deque = deque()

    def append_score(score_value: float) -> None:
        history.append(float(score_value))

    for row_idx, row in enumerate(test_df.itertuples(index=False)):
        release_due(pending, row_idx, append_score)
        active = history[-history_window:] if history_window > 0 else history
        values = np.asarray(active, dtype=float)
        weights = decay ** np.arange(len(values) - 1, -1, -1, dtype=float)
        margin = max(0.0, weighted_quantile(values, weights, 1.0 - alpha))
        lower, upper = _clip(float(row.base_lower) - margin, float(row.base_upper) + margin, clip_final)
        rows.append({
            "timestamp": row.timestamp, "target": row.target,
            "lower": lower, "upper": upper,
            "base_lower": float(row.base_lower), "base_upper": float(row.base_upper),
            "base_center": float(row.base_center),
        })
        score = float(conformity_score(
            np.array([row.target]), np.array([row.base_lower]), np.array([row.base_upper]),
        )[0])
        if delay > 0:
            pending.append((row_idx + delay, score))
        else:
            append_score(score)
    return pd.DataFrame(rows)


def _epanechnikov_weights(distance: np.ndarray, bandwidth: float) -> np.ndarray:
    bandwidth = max(float(bandwidth), 1e-8)
    u = distance / bandwidth
    weights = 0.75 * (1.0 - u ** 2)
    return np.where(np.abs(u) <= 1.0, np.maximum(weights, 0.0), 0.0)


def _weighted_residual_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return 0.0
    if not np.isfinite(weights).all() or float(np.sum(weights)) <= 0.0:
        weights = np.ones_like(values, dtype=float)
    return weighted_quantile(np.asarray(values, dtype=float), np.asarray(weights, dtype=float), q)


def run_kowcpi(pred_df: pd.DataFrame, alpha: float, cfg: dict) -> pd.DataFrame:
    """Kernel-weighted online conformal interval for time-series residuals.

    This revision baseline adapts the KOWCPI idea to the project's saved
    base forecasts: signed residuals around the base median define the
    calibration distribution, recent residual lags define the local time-series
    context, and Epanechnikov kernel weights select residual quantiles online.
    """
    history_window = int(cfg.get("history_window", 336))
    lag_window = int(cfg.get("lag_window", 24))
    bandwidth = float(cfg.get("bandwidth", 0.15))
    beta_grid_size = int(cfg.get("beta_grid_size", 21))
    clip_final = cfg.get("clip_final", True)

    cal_df = pred_df[pred_df["split"] == "calibration"].copy()
    test_df = pred_df[pred_df["split"] == "test"].copy()
    residual_history = list(
        (cal_df["target"].to_numpy(dtype=float) - cal_df["base_center"].to_numpy(dtype=float))
    )

    rows = []
    for _, row in test_df.iterrows():
        active = residual_history[-history_window:] if history_window > 0 else residual_history
        residuals = np.asarray(active, dtype=float)
        if len(residuals) < max(2, lag_window + 2):
            weights = np.ones(len(residuals), dtype=float)
        else:
            # Scalar residual context keeps the revision baseline tractable on
            # the 0427 grid while preserving online kernel weighting on recent
            # non-exchangeable residual structure.
            context = float(np.mean(residuals[-lag_window:]))
            values = residuals[lag_window:]
            past = residuals[:-1]
            if len(values) == 0:
                weights = np.ones(len(residuals), dtype=float)
            else:
                csum = np.cumsum(np.r_[0.0, past])
                starts = np.arange(0, len(values))
                stops = starts + lag_window
                means = (csum[stops] - csum[starts]) / lag_window
                distances = np.abs(means - context)
                weights = _epanechnikov_weights(distances, bandwidth)
                residuals = values

        if len(residuals) == 0:
            q_low, q_high = 0.0, 0.0
        else:
            beta_grid = np.linspace(0.0, alpha, max(2, beta_grid_size))
            best_width = float("inf")
            q_low, q_high = 0.0, 0.0
            for beta in beta_grid:
                lo = _weighted_residual_quantile(residuals, weights, beta)
                hi = _weighted_residual_quantile(residuals, weights, 1.0 - alpha + beta)
                width = hi - lo
                if np.isfinite(width) and width < best_width and hi >= lo:
                    best_width = float(width)
                    q_low, q_high = float(lo), float(hi)

        lower = float(row["base_center"]) + q_low
        upper = float(row["base_center"]) + q_high
        lower, upper = _clip(lower, upper, clip_final)
        rows.append({
            "timestamp": row["timestamp"], "target": row["target"],
            "lower": lower, "upper": upper,
            "base_lower": float(row["base_lower"]), "base_upper": float(row["base_upper"]),
            "base_center": float(row["base_center"]),
        })
        residual_history.append(float(row["target"]) - float(row["base_center"]))
    return pd.DataFrame(rows)


def run_waci(pred_df: pd.DataFrame, alpha: float, cfg: dict) -> pd.DataFrame:
    lr = cfg.get("learning_rate", 0.05)
    width_power = cfg.get("width_power", 0.5)
    ref_q = cfg.get("ref_width_quantile", 0.5)
    adapt_floor = cfg.get("adapt_floor", 0.2)
    adapt_cap = cfg.get("adapt_cap", 5.0)
    clip_final = cfg.get("clip_final", True)
    cal_df = pred_df[pred_df["split"] == "calibration"]
    test_df = pred_df[pred_df["split"] == "test"].copy()
    offset = fit_static_margin(cal_df, alpha)
    cal_width = (cal_df["base_upper"] - cal_df["base_lower"]).to_numpy()
    ref_width = max(1e-6, float(np.quantile(cal_width, ref_q)))
    rows = []
    for _, row in test_df.iterrows():
        bw = max(1e-6, float(row["base_upper"] - row["base_lower"]))
        ratio = ref_width / bw
        adapt_mult = float(np.clip(ratio ** width_power, adapt_floor, adapt_cap))
        lower, upper = _clip(float(row["base_lower"]) - offset, float(row["base_upper"]) + offset, clip_final)
        miss = 0.0 if lower <= row["target"] <= upper else 1.0
        offset = max(0.0, offset + lr * adapt_mult * (miss - alpha))
        rows.append({
            "timestamp": row["timestamp"], "target": row["target"],
            "lower": lower, "upper": upper,
            "base_lower": float(row["base_lower"]), "base_upper": float(row["base_upper"]),
            "base_center": float(row["base_center"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Wind-task custom methods
# ---------------------------------------------------------------------------

def run_clip(pred_df: pd.DataFrame, alpha: float, cfg: dict) -> pd.DataFrame:
    """Clipping-only: static conformal with clipping."""
    return run_split_conformal(pred_df, alpha, {**cfg, "clip_final": True})


def run_pwshrink(pred_df: pd.DataFrame, alpha: float, cfg: dict) -> pd.DataFrame:
    """Physics with shrinkage: regime-conditional directional margins + rolling shrinkage."""
    shrink_lr = cfg.get("shrink_lr", 0.5)
    shrink_tol = cfg.get("shrink_tol", 0.02)
    clip_final = cfg.get("clip_final", True)
    cal_df = add_state_columns(pred_df[pred_df["split"] == "calibration"].copy())
    test_df = add_state_columns(pred_df[pred_df["split"] == "test"].copy())
    reg_lower, reg_upper, global_lower, global_upper = directional_regime_quantiles(cal_df, alpha)
    recent = deque(maxlen=72)
    target_cov = 1.0 - alpha
    rows = []
    for _, row in test_df.iterrows():
        factors = REGIME_FACTORS[row["regime"]]
        l_margin = reg_lower.get(row["regime"], global_lower) * factors["lower"]
        u_margin = reg_upper.get(row["regime"], global_upper) * factors["upper"]
        if len(recent) >= 24:
            rc = sum(recent) / len(recent)
            excess = rc - (target_cov + shrink_tol)
            if excess > 0:
                sf = max(0.0, 1.0 - shrink_lr * factors["shrink"] * excess)
                l_margin *= sf
                u_margin *= sf
        lower = float(row["base_lower"]) - l_margin
        upper = float(row["base_upper"]) + u_margin
        lower, upper = _clip(lower, upper, clip_final)
        covered = 1.0 if lower <= row["target"] <= upper else 0.0
        recent.append(covered)
        rows.append({
            "timestamp": row["timestamp"], "target": row["target"],
            "lower": lower, "upper": upper,
            "base_lower": float(row["base_lower"]), "base_upper": float(row["base_upper"]),
            "base_center": float(row["base_center"]),
        })
    return pd.DataFrame(rows)


def run_hybrid(pred_df: pd.DataFrame, alpha: float, cfg: dict) -> pd.DataFrame:
    """Soft blend of PWShrink (physics) and ACI (dynamic)."""
    blend_alpha_val = cfg.get("blend_alpha", 0.3)
    ramp_boost = cfg.get("ramp_boost", 0.2)
    a3s_df = run_pwshrink(pred_df, alpha, cfg)
    d1c_df = run_aci(pred_df, alpha, cfg)
    a3s_df = add_state_columns(a3s_df)
    rows = []
    for i in range(len(a3s_df)):
        a3s_row = a3s_df.iloc[i]
        d1c_row = d1c_df.iloc[i]
        ramp = float(a3s_row.get("state_ramp_abs", 0.0))
        bp = float(a3s_row.get("state_boundary_proximity", 0.0))
        risk = min(1.0, ramp / 0.2 + bp)
        w_d1c = blend_alpha_val + ramp_boost * risk
        w_d1c = min(w_d1c, 0.8)
        w_a3s = 1.0 - w_d1c
        lower = w_a3s * float(a3s_row["lower"]) + w_d1c * float(d1c_row["lower"])
        upper = w_a3s * float(a3s_row["upper"]) + w_d1c * float(d1c_row["upper"])
        lower, upper = _clip(lower, upper, True)
        rows.append({
            "timestamp": a3s_row["timestamp"], "target": a3s_row["target"],
            "lower": lower, "upper": upper,
            "base_lower": float(a3s_row["base_lower"]), "base_upper": float(a3s_row["base_upper"]),
            "base_center": float(a3s_row["base_center"]),
        })
    return pd.DataFrame(rows)


def run_riskscale(pred_df: pd.DataFrame, alpha: float, cfg: dict) -> pd.DataFrame:
    """Risk-scaled PWShrink: scales width by coverage surplus."""
    shrink_lr = cfg.get("shrink_lr", 0.5)
    shrink_tol = cfg.get("shrink_tol", 0.02)
    scale_power = cfg.get("scale_power", 0.5)
    clip_final = cfg.get("clip_final", True)
    cal_df = add_state_columns(pred_df[pred_df["split"] == "calibration"].copy())
    test_df = add_state_columns(pred_df[pred_df["split"] == "test"].copy())
    reg_lower, reg_upper, global_lower, global_upper = directional_regime_quantiles(cal_df, alpha)
    recent = deque(maxlen=72)
    target_cov = 1.0 - alpha
    cal_width = (cal_df["base_upper"] - cal_df["base_lower"]).to_numpy()
    ref_width = max(1e-6, float(np.median(cal_width)))
    rows = []
    for _, row in test_df.iterrows():
        factors = REGIME_FACTORS[row["regime"]]
        l_margin = reg_lower.get(row["regime"], global_lower) * factors["lower"]
        u_margin = reg_upper.get(row["regime"], global_upper) * factors["upper"]
        if len(recent) >= 24:
            rc = sum(recent) / len(recent)
            excess = rc - (target_cov + shrink_tol)
            if excess > 0:
                sf = max(0.0, 1.0 - shrink_lr * factors["shrink"] * excess)
                l_margin *= sf
                u_margin *= sf
            deficit = target_cov - rc - shrink_tol
            if deficit > 0:
                bw = max(1e-6, float(row["base_upper"] - float(row["base_lower"])))
                ratio = ref_width / bw
                scale = float(np.clip(ratio ** scale_power, 0.5, 2.0))
                l_margin *= scale
                u_margin *= scale
        lower = float(row["base_lower"]) - l_margin
        upper = float(row["base_upper"]) + u_margin
        lower, upper = _clip(lower, upper, clip_final)
        covered = 1.0 if lower <= row["target"] <= upper else 0.0
        recent.append(covered)
        rows.append({
            "timestamp": row["timestamp"], "target": row["target"],
            "lower": lower, "upper": upper,
            "base_lower": float(row["base_lower"]), "base_upper": float(row["base_upper"]),
            "base_center": float(row["base_center"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY: dict[str, callable] = {
    # Static
    "SplitCF": run_split_conformal,
    "CQR": run_cqr,
    "LCF": run_lcf,
    # Dynamic
    "ACI": run_aci,
    "AgACI": run_agaci,
    "FACI": run_faci,
    "EnbPI": run_enbpi,
    # Dynamic extension
    "SAOCP": run_saocp,
    "PID": run_pid,
    "SPCI": run_spci,
    "NEX": run_nex,
    "KOWCPI": run_kowcpi,
    "WACI": run_waci,
    # Custom
    "Clip": run_clip,
    "PWShrink": run_pwshrink,
    "Hybrid": run_hybrid,
    "RiskScale": run_riskscale,
}


# Default hyperparameters per method
DEFAULT_METHOD_CONFIG: Dict[str, dict] = {
    "SplitCF": {"clip_final": True},
    "CQR": {"clip_final": True},
    "LCF": {"clip_final": True},
    "ACI": {
        "learning_rate": 0.05,
        "clip_final": True,
        "feedback_rule": "label_timestamp_strictly_before_issue_timestamp",
    },
    "AgACI": {
        "learning_rates": [0.01, 0.02, 0.05, 0.1],
        "clip_final": True,
        "feedback_rule": "label_timestamp_strictly_before_issue_timestamp",
    },
    "FACI": {"eta_min": 0.01, "eta_max": 0.1, "adapt_window": 24, "clip_final": True},
    "EnbPI": {
        "history_duration_hours": 336,
        "clip_final": True,
        "feedback_rule": "label_timestamp_strictly_before_issue_timestamp",
    },
    "SAOCP": {"learning_rate": 0.05, "windows": [24, 48, 96, 168], "temperature": 0.1, "clip_final": True},
    "PID": {"kp": 0.05, "ki": 0.01, "kd": 0.02, "clip_final": True},
    "SPCI": {"history_window": 336, "decay": 0.995, "clip_final": True},
    "NEX": {"decay": 0.99, "history_window": 336, "clip_final": True},
    "KOWCPI": {"history_window": 336, "lag_window": 24, "bandwidth": 0.15, "kernel": "epanechnikov", "beta_grid_size": 9, "clip_final": True},
    "WACI": {"learning_rate": 0.05, "width_power": 0.5, "ref_width_quantile": 0.5, "adapt_floor": 0.2, "adapt_cap": 5.0, "clip_final": True},
    "Clip": {"clip_final": True},
    "PWShrink": {"shrink_lr": 0.5, "shrink_tol": 0.02, "clip_final": True},
    "Hybrid": {"blend_alpha": 0.3, "ramp_boost": 0.2, "clip_final": True},
    "RiskScale": {"shrink_lr": 0.5, "shrink_tol": 0.02, "scale_power": 0.5, "clip_final": True},
}


def get_method(name: str) -> callable:
    func = METHOD_REGISTRY.get(name)
    if func is None:
        raise ValueError(f"Unknown calibration method: {name}")
    return func


def get_method_config(name: str) -> dict:
    return DEFAULT_METHOD_CONFIG.get(name, {}).copy()
