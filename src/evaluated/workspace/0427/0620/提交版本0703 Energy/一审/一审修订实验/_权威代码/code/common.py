from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_json(data: Dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)


def canonical_json_sha256(data: Any) -> str:
    """计算可复算的规范JSON摘要。"""
    payload = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        cls=_NumpyEncoder,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """以分块方式计算文件SHA256，避免将大文件一次读入内存。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_commit_for_path(path: str | Path) -> str:
    """返回路径所属Git仓库的当前提交。"""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(path),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def git_status_porcelain(path: str | Path) -> str:
    """返回Git工作树状态，空字符串表示工作树干净。"""
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=Path(path),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("feature_")]


def apply_split(
    df: pd.DataFrame,
    train_ratio: float,
    calibration_ratio: float,
) -> pd.DataFrame:
    """对已经独立构造好标签的数据执行普通时间顺序切分。"""
    n = len(df)
    train_end = int(n * train_ratio)
    calib_end = int(n * (train_ratio + calibration_ratio))
    split = np.full(n, "test", dtype=object)
    split[:train_end] = "train"
    split[train_end:calib_end] = "calibration"
    out = df.copy()
    out["split"] = split
    return out


def _validate_split_ratios(train_ratio: float, calibration_ratio: float) -> None:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio必须位于0和1之间")
    if not 0.0 < calibration_ratio < 1.0:
        raise ValueError("calibration_ratio必须位于0和1之间")
    if train_ratio + calibration_ratio >= 1.0:
        raise ValueError("train_ratio与calibration_ratio之和必须小于1")


def _infer_nominal_cadence(timestamps: pd.Series) -> pd.Timedelta:
    """从唯一且有序的时间戳中推断最常见的正时间间隔。"""
    deltas = timestamps.diff().dropna()
    if deltas.empty or (deltas <= pd.Timedelta(0)).any():
        raise ValueError("至少需要两个严格递增的时间戳")

    counts = deltas.value_counts()
    cadence = min(counts[counts.eq(counts.max())].index)
    ratios = deltas / cadence
    if not np.allclose(ratios, np.round(ratios), rtol=0.0, atol=1e-9):
        raise ValueError("时间轴包含无法表示为名义采样间隔整数倍的间隔")
    return pd.Timedelta(cadence)


def prepare_multihorizon(
    df: pd.DataFrame,
    horizon: int,
    train_ratio: float,
    calibration_ratio: float,
) -> pd.DataFrame:
    """构造精确物理时长标签，并执行带标签禁运的因果切分。

    切分边界由完整起报时间轴固定。训练与校准样本的标签时间必须
    严格早于下一阶段起点。所有预测时长因此共享同一个测试起报边界。
    """
    required_columns = {"timestamp", "target"}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"缺少必需列: {missing_columns}")
    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("horizon必须为正整数")
    _validate_split_ratios(train_ratio, calibration_ratio)

    source = df.copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"], errors="raise")
    source = source.sort_values("timestamp").reset_index(drop=True)
    if source["timestamp"].duplicated().any():
        duplicated = int(source["timestamp"].duplicated(keep=False).sum())
        raise ValueError(f"时间戳必须唯一，当前有{duplicated}行属于重复时间戳")
    if len(source) < 3:
        raise ValueError("数据行数过少，无法建立训练、校准和测试切分")

    cadence = _infer_nominal_cadence(source["timestamp"])
    physical_lead = cadence * int(horizon)
    source_rows = len(source)
    train_boundary_index = int(source_rows * train_ratio)
    test_boundary_index = int(source_rows * (train_ratio + calibration_ratio))
    if not 0 < train_boundary_index < test_boundary_index < source_rows:
        raise ValueError("切分比例产生了空阶段")

    issue_row_index = np.arange(source_rows, dtype=np.int64)
    expected_label_timestamp = source["timestamp"] + physical_lead
    timestamp_index = pd.Index(source["timestamp"])
    label_row_index = timestamp_index.get_indexer(expected_label_timestamp)
    exact_label_available = label_row_index >= 0

    source_target = pd.to_numeric(source["target"], errors="raise")
    aligned_target = np.full(source_rows, np.nan, dtype=np.float64)
    aligned_target[exact_label_available] = source_target.to_numpy(dtype=np.float64)[
        label_row_index[exact_label_available]
    ]

    out = source.copy()
    out["issue_row_index"] = issue_row_index
    out["label_row_index"] = label_row_index
    out["label_timestamp"] = expected_label_timestamp
    out["horizon_steps"] = int(horizon)
    out["nominal_cadence_minutes"] = cadence.total_seconds() / 60.0
    out["lead_time_minutes"] = physical_lead.total_seconds() / 60.0
    out["lead_time_hours"] = physical_lead.total_seconds() / 3600.0
    out["target"] = aligned_target

    if "feature_target_lag1" in out.columns:
        out["feature_target_lag1"] = source_target.shift(1)
        out["target_lag1_timestamp"] = source["timestamp"].shift(1)

    feature_columns = select_feature_columns(out)
    required_value_columns = ["target", *feature_columns]
    required_values_available = out[required_value_columns].notna().all(axis=1).to_numpy()

    train_mask = (
        exact_label_available
        & required_values_available
        & (issue_row_index < train_boundary_index)
        & (label_row_index < train_boundary_index)
    )
    calibration_mask = (
        exact_label_available
        & required_values_available
        & (issue_row_index >= train_boundary_index)
        & (issue_row_index < test_boundary_index)
        & (label_row_index < test_boundary_index)
    )
    test_mask = (
        exact_label_available
        & required_values_available
        & (issue_row_index >= test_boundary_index)
    )

    split = np.full(source_rows, "excluded", dtype=object)
    split[train_mask] = "train"
    split[calibration_mask] = "calibration"
    split[test_mask] = "test"
    out["split"] = split

    valid_candidate = exact_label_available & required_values_available
    train_embargo_mask = (
        valid_candidate
        & (issue_row_index < train_boundary_index)
        & (label_row_index >= train_boundary_index)
    )
    calibration_embargo_mask = (
        valid_candidate
        & (issue_row_index >= train_boundary_index)
        & (issue_row_index < test_boundary_index)
        & (label_row_index >= test_boundary_index)
    )
    within_source_range = expected_label_timestamp.le(source["timestamp"].iloc[-1]).to_numpy()
    missing_exact_target_mask = (~exact_label_available) & within_source_range
    tail_target_unavailable_mask = (~exact_label_available) & (~within_source_range)
    missing_required_value_mask = exact_label_available & (~required_values_available)

    keep_mask = train_mask | calibration_mask | test_mask
    out = out.loc[keep_mask].reset_index(drop=True)
    split_counts = out["split"].value_counts().to_dict()
    test_rows = out.loc[out["split"].eq("test"), "timestamp"]

    out.attrs["causal_split"] = {
        "policy": "fixed_issue_boundaries_exact_physical_label_v1",
        "horizon_steps": int(horizon),
        "nominal_cadence_minutes": cadence.total_seconds() / 60.0,
        "lead_time_minutes": physical_lead.total_seconds() / 60.0,
        "lead_time_hours": physical_lead.total_seconds() / 3600.0,
        "source_rows": int(source_rows),
        "train_ratio": float(train_ratio),
        "calibration_ratio": float(calibration_ratio),
        "train_boundary_index": int(train_boundary_index),
        "train_boundary_timestamp": source["timestamp"].iloc[train_boundary_index].isoformat(),
        "test_boundary_index": int(test_boundary_index),
        "test_boundary_timestamp": source["timestamp"].iloc[test_boundary_index].isoformat(),
        "missing_exact_target_within_range_rows": int(missing_exact_target_mask.sum()),
        "tail_target_unavailable_rows": int(tail_target_unavailable_mask.sum()),
        "missing_required_value_rows": int(missing_required_value_mask.sum()),
        "train_boundary_embargo_rows": int(train_embargo_mask.sum()),
        "calibration_boundary_embargo_rows": int(calibration_embargo_mask.sum()),
        "n_train": int(split_counts.get("train", 0)),
        "n_calibration": int(split_counts.get("calibration", 0)),
        "n_test": int(split_counts.get("test", 0)),
        "n_total": int(len(out)),
        "first_test_issue_timestamp": test_rows.iloc[0].isoformat() if not test_rows.empty else None,
        "target_lag_policy": "previous_observed_target_y_i_minus_1"
        if "feature_target_lag1" in out.columns
        else "not_present_in_source_features",
    }
    return out


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def empirical_coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y >= lower) & (y <= upper)))


def avg_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(upper - lower))


def out_of_bound_rate(
    lower: np.ndarray, upper: np.ndarray,
    low: float = 0.0, high: float = 1.0,
) -> float:
    return float(np.mean((lower < low) | (upper > high)))


def interval_score(
    y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float,
) -> float:
    """Winkler interval score (lower is better)."""
    width = upper - lower
    below = np.maximum(lower - y, 0.0)
    above = np.maximum(y - upper, 0.0)
    score = width + (2.0 / alpha) * (below + above)
    return float(np.mean(score))


def pinball_loss(y: np.ndarray, quantile_pred: np.ndarray, q: float) -> float:
    """Pinball / quantile loss for a single quantile."""
    diff = y - quantile_pred
    return float(np.mean(np.where(diff >= 0, q * diff, (q - 1) * diff)))


def pinball_loss_both(y: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                      alpha: float) -> float:
    """Average pinball loss for lower (q=alpha/2) and upper (q=1-alpha/2)."""
    q_low = alpha / 2.0
    q_high = 1.0 - alpha / 2.0
    return (pinball_loss(y, lower, q_low) + pinball_loss(y, upper, q_high)) / 2.0


def crps_from_interval(
    y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float,
) -> float:
    """Approximate CRPS from a single prediction interval.

    Uses the quantile-score decomposition: average pinball loss at
    q=alpha/2 and q=1-alpha/2, plus the median (q=0.5 using midpoint).
    """
    q_low = alpha / 2.0
    q_high = 1.0 - alpha / 2.0
    center = (lower + upper) / 2.0
    # Average pinball losses at lower, center, upper quantiles
    pl_low = pinball_loss(y, lower, q_low)
    pl_high = pinball_loss(y, upper, q_high)
    pl_mid = pinball_loss(y, center, 0.5)
    return (pl_low + pl_high + pl_mid) / 3.0


def crps_from_multi_alpha(
    y: np.ndarray,
    intervals: list[Tuple[np.ndarray, np.ndarray, float]],
) -> float:
    """Approximate CRPS by averaging pinball losses across multiple quantiles.

    intervals: list of (lower, upper, alpha) tuples.
    """
    total = 0.0
    count = 0
    for lower, upper, alpha in intervals:
        q_low = alpha / 2.0
        q_high = 1.0 - alpha / 2.0
        total += pinball_loss(y, lower, q_low)
        total += pinball_loss(y, upper, q_high)
        count += 2
    return total / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Regime & boundary classification
# ---------------------------------------------------------------------------

def classify_ramp(
    center: np.ndarray, lag1: np.ndarray, threshold: float = 0.12,
) -> np.ndarray:
    """Boolean mask: True where |center[t] - center[t-1]| > threshold."""
    diff = np.abs(center - lag1)
    return diff > threshold


def classify_boundary_region(
    center: np.ndarray, low: float = 0.05, high: float = 0.95,
) -> np.ndarray:
    """Boolean mask: True where center is near 0 or near capacity."""
    return (center < low) | (center > high)



# ---------------------------------------------------------------------------
# Rolling metrics
# ---------------------------------------------------------------------------

def rolling_metrics(df: pd.DataFrame, target_alpha: float, window: int) -> pd.DataFrame:
    rows = []
    target_coverage = 1.0 - target_alpha
    for start in range(0, len(df) - window + 1):
        chunk = df.iloc[start: start + window]
        cov = empirical_coverage(
            chunk["target"].to_numpy(),
            chunk["lower"].to_numpy(),
            chunk["upper"].to_numpy(),
        )
        rows.append({
            "window_start": str(chunk["timestamp"].iloc[0]),
            "window_end": str(chunk["timestamp"].iloc[-1]),
            "coverage": cov,
            "coverage_gap": cov - target_coverage,
            "avg_width": avg_width(chunk["lower"].to_numpy(), chunk["upper"].to_numpy()),
            "out_of_bound_rate": out_of_bound_rate(
                chunk["lower"].to_numpy(), chunk["upper"].to_numpy(),
            ),
        })
    return pd.DataFrame(rows)


def rolling_summary_arrays(
    covered: np.ndarray,
    widths: np.ndarray,
    target_coverage: float,
    window: int,
) -> Dict:
    if window <= 0 or len(covered) < window:
        return {}
    kernel = np.ones(window, dtype=float)
    rolling_cov = np.convolve(covered.astype(float), kernel, mode="valid") / float(window)
    rolling_width = np.convolve(widths.astype(float), kernel, mode="valid") / float(window)
    gaps = rolling_cov - target_coverage
    n_non = len(covered) // window
    out = {
        "worst_rolling_coverage_gap": float(np.min(gaps)),
        "median_rolling_gap": float(np.median(gaps)),
        "max_rolling_gap": float(np.max(gaps)),
        "overcoverage_window_rate": float(np.mean(gaps > 0.0)),
        "severe_overcoverage_window_rate": float(np.mean(gaps > 0.10)),
        "undercoverage_window_rate": float(np.mean(gaps < 0.0)),
        "severe_undercoverage_window_rate": float(np.mean(gaps < -0.10)),
        "abs_rolling_coverage_deviation": float(np.mean(np.abs(gaps))),
        "median_rolling_width": float(np.median(rolling_width)),
    }
    if n_non > 0:
        non_cov = covered[: n_non * window].reshape(n_non, window).mean(axis=1)
        non_width = widths[: n_non * window].reshape(n_non, window).mean(axis=1)
        non_gaps = non_cov - target_coverage
        out.update(
            {
                "n_nonoverlap_windows": int(n_non),
                "nonoverlap_mean_gap": float(np.mean(non_gaps)),
                "nonoverlap_max_gap": float(np.max(non_gaps)),
                "nonoverlap_min_gap": float(np.min(non_gaps)),
                "nonoverlap_owr": float(np.mean(non_gaps > 0.0)),
                "nonoverlap_sowr": float(np.mean(non_gaps > 0.10)),
                "nonoverlap_uwr": float(np.mean(non_gaps < 0.0)),
                "nonoverlap_suwr": float(np.mean(non_gaps < -0.10)),
                "nonoverlap_abs_mean_gap": float(np.mean(np.abs(non_gaps))),
                "nonoverlap_median_width": float(np.median(non_width)),
            }
        )
    return out


def summarize_metrics(
    df: pd.DataFrame, alpha: float, rolling_window: int, return_rolling: bool = True,
) -> Tuple[Dict, pd.DataFrame]:
    y = df["target"].to_numpy()
    lower = df["lower"].to_numpy()
    upper = df["upper"].to_numpy()
    coverage = empirical_coverage(y, lower, upper)
    widths = upper - lower
    covered = ((y >= lower) & (y <= upper)).astype(float)
    target_coverage = 1.0 - alpha
    metrics = {
        "coverage": coverage,
        "target_coverage": target_coverage,
        "coverage_gap": coverage - target_coverage,
        "avg_width": float(np.mean(widths)),
        "interval_score": interval_score(y, lower, upper, alpha),
        "pinball_loss": pinball_loss_both(y, lower, upper, alpha),
        "out_of_bound_rate": out_of_bound_rate(lower, upper),
        "n_test": int(len(df)),
    }
    # CRPS approximation for single alpha
    metrics["crps"] = crps_from_interval(y, lower, upper, alpha)

    metrics.update(rolling_summary_arrays(covered, widths, target_coverage, rolling_window))
    roll_df = rolling_metrics(df, alpha, rolling_window) if return_rolling else pd.DataFrame()
    return metrics, roll_df


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------

def clip_bounds(
    lower: np.ndarray, upper: np.ndarray,
    low: float = 0.0, high: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    return np.clip(lower, low, high), np.clip(upper, low, high)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def zone_name_from_path(csv_path: str | Path) -> str:
    stem = Path(csv_path).stem
    for token in stem.split("_"):
        if token.startswith("zone"):
            return token
    if stem.endswith("_processed"):
        return stem[: -len("_processed")]
    return stem


def aggregate_zone_metrics(rows: Iterable[Dict]) -> Dict:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return {"n_zones": 0}
    summary = {"n_zones": int(len(df))}
    metric_cols = [
        "coverage", "coverage_gap", "avg_width", "interval_score",
        "pinball_loss", "crps", "out_of_bound_rate",
        "worst_rolling_coverage_gap",
    ]
    for col in metric_cols:
        if col in df.columns:
            summary[f"{col}_mean"] = float(df[col].mean())
            summary[f"{col}_median"] = float(df[col].median())
    target_cov = float(df["target_coverage"].iloc[0]) if "target_coverage" in df.columns else 0.9
    if "coverage" in df.columns:
        summary["zones_meeting_target_coverage"] = int((df["coverage"] >= target_cov).sum())
    if "out_of_bound_rate" in df.columns:
        summary["zones_with_zero_oob"] = int((df["out_of_bound_rate"] == 0.0).sum())
    return summary
