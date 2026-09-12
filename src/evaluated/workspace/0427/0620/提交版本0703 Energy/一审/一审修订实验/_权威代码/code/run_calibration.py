"""Run one causally auditable candidate interval stream."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from calibrators import get_method, get_method_config
from common import (
    canonical_json_sha256,
    ensure_dir,
    git_commit_for_path,
    git_status_porcelain,
    load_json,
    save_json,
    sha256_file,
    summarize_metrics,
)
from predictors import ALL_QUANTILES


ACTION_MAP = {
    "SplitCF": "Static",
    "ACI": "ACI",
    "AgACI": "AgACI",
    "EnbPI": "EnbPI_RH",
}
REQUIRED_PROTOCOL_FIELDS = (
    "protocol_id",
    "protocol_version",
    "protocol_sha256",
    "baseline_registry_sha256",
    "output_contract_sha256",
)


def _read_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"不支持的基础预测格式: {path.suffix}")


def _validate_protocol_fields(cfg: dict) -> None:
    missing = [field for field in REQUIRED_PROTOCOL_FIELDS if not cfg.get(field)]
    if missing:
        raise ValueError(f"候选区间配置缺少协议字段: {missing}")


def _validate_frozen_method_config(method_name: str, method_cfg: dict) -> None:
    if not bool(method_cfg.get("clip_final")):
        raise ValueError("四个冻结动作必须统一执行最终裁剪")
    if method_name == "ACI" and not np.isclose(
        float(method_cfg.get("learning_rate")), 0.05
    ):
        raise ValueError("ACI学习率必须保持冻结值0.05")
    if method_name == "AgACI":
        rates = [float(value) for value in method_cfg.get("learning_rates", [])]
        if rates != [0.01, 0.02, 0.05, 0.1]:
            raise ValueError("AgACI学习率集合与冻结协议不一致")
    if method_name == "EnbPI" and not np.isclose(
        float(method_cfg.get("history_duration_hours")), 336.0
    ):
        raise ValueError("EnbPI历史窗口必须保持336物理小时")


def _validate_base_cache(
    base_path: Path,
    cfg: dict,
    expected_commit: str | None = None,
) -> dict:
    manifest_path = base_path.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"基础预测缺少manifest: {manifest_path}")
    manifest = load_json(manifest_path)
    repository_root = Path(__file__).resolve().parents[1]
    expected_commit = expected_commit or git_commit_for_path(repository_root)
    checks = {
        "status": "COMPLETE_VALIDATED",
        "protocol_id": cfg["protocol_id"],
        "protocol_version": cfg["protocol_version"],
        "protocol_sha256": cfg["protocol_sha256"],
        "code_commit": expected_commit,
        "output_sha256": sha256_file(base_path),
    }
    mismatches = [
        key for key, value in checks.items() if manifest.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(f"基础预测缓存身份核验失败: {sorted(mismatches)}")
    return manifest


def prepare_candidate_base(
    run_cfg: dict,
    *,
    code_status: str | None = None,
    code_commit: str | None = None,
) -> dict:
    """一次性核验并载入候选区间所需的基础预测。"""
    _validate_protocol_fields(run_cfg)
    repository_root = Path(__file__).resolve().parents[1]
    resolved_status = (
        git_status_porcelain(repository_root)
        if code_status is None
        else code_status
    )
    if resolved_status and run_cfg.get("require_clean_code", True):
        raise RuntimeError("权威代码工作树不干净，禁止生成可批准候选区间")
    resolved_commit = code_commit or git_commit_for_path(repository_root)
    base_path = Path(
        run_cfg.get("base_predictions_path")
        or run_cfg.get("base_predictions_csv")
    )
    base_manifest = _validate_base_cache(
        base_path,
        run_cfg,
        expected_commit=resolved_commit,
    )

    pred_df = _read_frame(base_path)
    for column in ("timestamp", "issue_timestamp", "label_timestamp"):
        if column not in pred_df.columns:
            raise ValueError(f"基础预测缺少时间字段: {column}")
        pred_df[column] = pd.to_datetime(pred_df[column], errors="raise")
    if "label_available_timestamp" in pred_df.columns:
        pred_df["label_available_timestamp"] = pd.to_datetime(
            pred_df["label_available_timestamp"], errors="coerce"
        )
    pred_df = pred_df.sort_values(["split", "timestamp"]).reset_index(drop=True)

    quantile_columns = [f"q_{quantile:.4f}" for quantile in ALL_QUANTILES]
    missing_quantiles = [
        column for column in quantile_columns if column not in pred_df.columns
    ]
    if missing_quantiles:
        raise ValueError(f"基础预测缺少分位数列: {missing_quantiles}")
    quantile_matrix = pred_df[quantile_columns].to_numpy(dtype=float)
    if not np.isfinite(quantile_matrix).all():
        raise ValueError("基础分位数包含非有限值")
    if (np.diff(quantile_matrix, axis=1) < 0.0).any():
        raise ValueError("基础分位数在重排后仍存在交叉")

    return {
        "predictions": pred_df,
        "base_path": base_path,
        "base_manifest": base_manifest,
        "code_status": resolved_status,
        "code_commit": resolved_commit,
    }


def _event_ids(df: pd.DataFrame, cfg: dict, target_coverage: float) -> list[str]:
    event_ids = []
    for row in df.itertuples(index=False):
        payload = {
            "dataset_id": cfg.get("dataset_id", "gefcom2014"),
            "zone_or_farm": cfg["zone"],
            "predictor": cfg["predictor"],
            "horizon_steps": int(getattr(row, "horizon_steps", cfg["horizon"])),
            "seed": int(cfg["seed"]),
            "target_coverage": float(target_coverage),
            "issue_timestamp": pd.Timestamp(row.issue_timestamp).isoformat(),
        }
        event_ids.append(canonical_json_sha256(payload))
    return event_ids


def _build_feedback_trace(
    candidate_df: pd.DataFrame,
    action: str,
) -> pd.DataFrame:
    issue_times = pd.DatetimeIndex(candidate_df["issue_timestamp"])
    label_times = pd.DatetimeIndex(candidate_df["label_timestamp"])
    positions = issue_times.searchsorted(label_times, side="right")
    valid = positions < len(issue_times)
    feedback = candidate_df.loc[
        valid,
        ["event_id", "label_timestamp"],
    ].copy().reset_index(drop=True)
    feedback["action"] = action
    feedback["feedback_application_timestamp"] = issue_times.to_numpy()[positions[valid]]
    feedback["eligible_by_strict_time_rule"] = (
        pd.to_datetime(feedback["label_timestamp"])
        < pd.to_datetime(feedback["feedback_application_timestamp"])
    )
    feedback["candidate_calibrator_updated"] = action != "Static"
    feedback["rolling_diagnostic_updated"] = False
    feedback["upper_landscape_updated"] = False
    if action == "Static":
        feedback["calibrator_revision_before"] = 0
        feedback["calibrator_revision_after"] = 0
    else:
        feedback["calibrator_revision_before"] = np.arange(len(feedback), dtype=np.int64)
        feedback["calibrator_revision_after"] = np.arange(1, len(feedback) + 1, dtype=np.int64)
    return feedback


def build_candidate_frames(
    run_cfg: dict,
    prepared_base: dict,
    event_identity: tuple[np.ndarray, list[str]] | None = None,
) -> dict:
    """在内存中生成单一候选流，供独立运行和规范化候选包共用。"""
    _validate_protocol_fields(run_cfg)
    method_name = run_cfg["method"]
    if method_name not in ACTION_MAP:
        raise ValueError(f"S03只允许四个冻结动作，收到: {method_name}")
    action = ACTION_MAP[method_name]
    alpha = float(run_cfg["alpha"])
    target_coverage = round(1.0 - alpha, 10)

    pred_df = prepared_base["predictions"].copy()
    q_low = alpha / 2.0
    q_high = 1.0 - alpha / 2.0
    pred_df["base_lower"] = pred_df[f"q_{q_low:.4f}"]
    pred_df["base_upper"] = pred_df[f"q_{q_high:.4f}"]
    pred_df["base_center"] = pred_df["q_0.5000"]

    method_cfg = get_method_config(method_name)
    method_cfg.update(run_cfg.get("method_parameters", {}))
    method_cfg["clip_final"] = True
    if method_name in {"ACI", "AgACI", "EnbPI"}:
        method_cfg["feedback_rule"] = (
            "label_timestamp_strictly_before_issue_timestamp"
        )
    if method_name == "EnbPI":
        method_cfg["history_duration_hours"] = 336.0
    _validate_frozen_method_config(method_name, method_cfg)

    result_df = get_method(method_name)(pred_df, alpha, method_cfg)
    result_df = result_df.dropna(subset=["lower", "upper"]).reset_index(drop=True)
    if result_df.empty:
        raise ValueError("候选区间输出为空")
    if not result_df["issue_timestamp"].is_unique:
        raise ValueError("单一候选流存在重复问题时刻")

    lower = result_df["lower"].to_numpy(dtype=float)
    upper = result_df["upper"].to_numpy(dtype=float)
    if np.any(lower > upper):
        raise AssertionError("候选区间存在下界高于上界")
    if np.any(lower < 0.0) or np.any(upper > 1.0):
        raise AssertionError("候选区间超出0到1物理边界")

    run_id = run_cfg.get(
        "run_id",
        (
            f"{run_cfg['predictor']}-H{int(run_cfg['horizon']):02d}-"
            f"{run_cfg['zone']}-S{int(run_cfg['seed'])}-"
            f"C{target_coverage:.2f}-{action}"
        ),
    )
    result_df["protocol_id"] = run_cfg["protocol_id"]
    result_df["protocol_sha256"] = run_cfg["protocol_sha256"]
    result_df["run_id"] = run_id
    result_df["dataset_id"] = run_cfg.get("dataset_id", "gefcom2014")
    result_df["zone_or_farm"] = run_cfg["zone"]
    result_df["predictor"] = run_cfg["predictor"]
    result_df["seed"] = int(run_cfg["seed"])
    result_df["alpha"] = alpha
    result_df["target_coverage"] = target_coverage
    result_df["action"] = action
    result_df["candidate_lower"] = lower
    result_df["candidate_upper"] = upper

    issue_identity = result_df["issue_timestamp"].to_numpy(dtype="datetime64[ns]")
    if event_identity is None:
        event_ids = _event_ids(result_df, run_cfg, target_coverage)
        resolved_identity = (issue_identity.copy(), event_ids)
    else:
        reference_issue_identity, event_ids = event_identity
        if not np.array_equal(reference_issue_identity, issue_identity):
            raise AssertionError("同一覆盖率的候选动作事件时间轴不一致")
        if len(event_ids) != len(result_df):
            raise AssertionError("复用的候选事件标识数量不一致")
        resolved_identity = event_identity
    result_df["event_id"] = event_ids
    if not result_df["event_id"].is_unique:
        raise AssertionError("候选区间event_id不唯一")

    feedback_df = _build_feedback_trace(result_df, action)
    if not feedback_df["eligible_by_strict_time_rule"].all():
        raise AssertionError("反馈轨迹违反严格时间规则")
    if action != "Static":
        applied_count = int(result_df["feedback_applied_at_issue"].sum())
        if applied_count != len(feedback_df):
            raise AssertionError(
                f"反馈应用计数不一致: 轨迹{len(feedback_df)}，候选状态{applied_count}"
            )

    validation_summary = {
        "status": "PASS",
        "candidate_rows": int(len(result_df)),
        "feedback_rows": int(len(feedback_df)),
        "duplicate_event_ids": int(result_df["event_id"].duplicated().sum()),
        "nonfinite_candidate_rows": int(
            (~np.isfinite(lower) | ~np.isfinite(upper)).sum()
        ),
        "reversed_interval_rows": int((lower > upper).sum()),
        "out_of_bounds_rows": int(((lower < 0.0) | (upper > 1.0)).sum()),
        "strict_feedback_violations": int(
            (~feedback_df["eligible_by_strict_time_rule"]).sum()
        ),
        "feedback_applied_count": int(
            result_df["feedback_applied_at_issue"].sum()
        ),
        "performance_comparison_performed": False,
    }
    return {
        "candidate": result_df,
        "feedback": feedback_df,
        "validation": validation_summary,
        "method_config": method_cfg,
        "method": method_name,
        "action": action,
        "alpha": alpha,
        "target_coverage": target_coverage,
        "run_id": run_id,
        "event_identity": resolved_identity,
    }


def run_calibration(run_cfg: dict) -> dict:
    prepared_base = prepare_candidate_base(run_cfg)
    built = build_candidate_frames(run_cfg, prepared_base)
    code_status = prepared_base["code_status"]
    base_path = prepared_base["base_path"]
    base_manifest = prepared_base["base_manifest"]
    result_df = built["candidate"]
    feedback_df = built["feedback"]
    validation_summary = built["validation"]
    method_cfg = built["method_config"]
    method_name = built["method"]
    action = built["action"]
    alpha = built["alpha"]
    run_id = built["run_id"]

    run_dir = ensure_dir(run_cfg["results_dir"])
    candidate_path = run_dir / "candidate_intervals.parquet"
    feedback_path = run_dir / "feedback_trace.parquet"
    result_df.to_parquet(candidate_path, index=False, compression="zstd")
    feedback_df.to_parquet(feedback_path, index=False, compression="zstd")

    save_json(validation_summary, run_dir / "validation_summary.json")

    if run_cfg.get("compute_performance_metrics", False):
        rolling_window = int(run_cfg.get("rolling_window", 168))
        metrics, rolling_df = summarize_metrics(
            result_df,
            alpha,
            rolling_window,
        )
        metrics.update(
            {
                "method": method_name,
                "action": action,
                "predictor": run_cfg["predictor"],
                "zone": run_cfg["zone"],
                "seed": int(run_cfg["seed"]),
                "horizon": int(run_cfg["horizon"]),
                "alpha": alpha,
            }
        )
        save_json(metrics, run_dir / "metrics.json")
        if not rolling_df.empty:
            rolling_df.to_parquet(
                run_dir / "rolling_metrics.parquet",
                index=False,
                compression="zstd",
            )

    manifest = {
        "manifest_schema": "S03_CANDIDATE_INTERVAL_MANIFEST_V1",
        "status": "COMPLETE_VALIDATED",
        "protocol_id": run_cfg["protocol_id"],
        "protocol_version": run_cfg["protocol_version"],
        "protocol_sha256": run_cfg["protocol_sha256"],
        "baseline_registry_sha256": run_cfg["baseline_registry_sha256"],
        "output_contract_sha256": run_cfg["output_contract_sha256"],
        "code_commit": prepared_base["code_commit"],
        "code_worktree_clean": not bool(code_status),
        "config_sha256": canonical_json_sha256(run_cfg),
        "base_manifest_sha256": sha256_file(base_path.parent / "manifest.json"),
        "base_predictions_sha256": sha256_file(base_path),
        "candidate_intervals_sha256": sha256_file(candidate_path),
        "feedback_trace_sha256": sha256_file(feedback_path),
        "run_id": run_id,
        "method": method_name,
        "action": action,
        "method_config": method_cfg,
        "candidate_file": str(candidate_path),
        "feedback_file": str(feedback_path),
        "candidate_rows": int(len(result_df)),
        "feedback_rows": int(len(feedback_df)),
        "base_code_commit": base_manifest["code_commit"],
        "validations": validation_summary,
    }
    save_json(run_cfg, run_dir / "config_snapshot.json")
    save_json(manifest, run_dir / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    manifest = run_calibration(load_json(args.config))
    print(
        f"Validated {manifest['action']} candidate stream: "
        f"{manifest['candidate_rows']} events, {manifest['feedback_rows']} feedback rows"
    )


if __name__ == "__main__":
    main()
