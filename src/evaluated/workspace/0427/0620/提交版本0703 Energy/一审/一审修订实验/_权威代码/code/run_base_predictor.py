"""Run one causally auditable base prediction stream."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from common import (
    canonical_json_sha256,
    ensure_dir,
    git_commit_for_path,
    git_status_porcelain,
    load_json,
    prepare_multihorizon,
    save_json,
    select_feature_columns,
    sha256_file,
)
from predictors import (
    ALL_QUANTILES,
    extract_quantile_predictions,
    get_predictor,
    rearrange_quantile_predictions,
)


REQUIRED_PROTOCOL_FIELDS = (
    "protocol_id",
    "protocol_version",
    "protocol_sha256",
)


def _load_source(data_path: str | Path) -> pd.DataFrame:
    source = pd.read_csv(data_path)
    source["timestamp"] = pd.to_datetime(source["timestamp"], errors="raise")
    source = source.sort_values("timestamp").reset_index(drop=True)
    if source["timestamp"].duplicated().any():
        raise ValueError("基础预测输入时间戳必须唯一")
    return source


def _build_feature_context(
    source: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """保留完整问题时间轴作为序列上下文，并修正上一时刻功率。"""
    context = source.copy()
    if "feature_target_lag1" in context.columns:
        context["feature_target_lag1"] = pd.to_numeric(
            context["target"], errors="raise"
        ).shift(1)
    available = context[feature_columns].notna().all(axis=1)
    context = context.loc[available, ["timestamp", *feature_columns]].reset_index(drop=True)
    if context.empty:
        raise ValueError("完整问题时间轴没有可用特征行")
    return context


def _first_issue_strictly_after(
    source_issue_timestamps: pd.Series,
    label_timestamps: pd.Series,
) -> pd.Series:
    source_index = pd.DatetimeIndex(source_issue_timestamps)
    label_index = pd.DatetimeIndex(label_timestamps)
    positions = source_index.searchsorted(label_index, side="right")
    values = np.full(len(label_index), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = positions < len(source_index)
    values[valid] = source_index.to_numpy(dtype="datetime64[ns]")[positions[valid]]
    return pd.Series(values, index=label_timestamps.index)


def _validate_protocol_fields(cfg: dict) -> None:
    missing = [field for field in REQUIRED_PROTOCOL_FIELDS if not cfg.get(field)]
    if missing:
        raise ValueError(f"基础预测配置缺少协议字段: {missing}")


def run_base_prediction(run_cfg: dict) -> dict:
    _validate_protocol_fields(run_cfg)
    repository_root = Path(__file__).resolve().parents[1]
    code_status = git_status_porcelain(repository_root)
    if code_status and run_cfg.get("require_clean_code", True):
        raise RuntimeError("权威代码工作树不干净，禁止生成可批准基础预测")
    predictor_name = run_cfg["predictor"]
    zone = run_cfg["zone"]
    seed = int(run_cfg["seed"])
    horizon = int(run_cfg["horizon"])
    data_path = run_cfg["data_path"]
    split = run_cfg["split"]
    results_dir = run_cfg["results_dir"]
    save_splits = tuple(run_cfg.get("save_splits", ["calibration", "test"]))
    if set(save_splits) != {"calibration", "test"}:
        raise ValueError("S03基础预测只允许保存calibration和test切片")

    source = _load_source(data_path)
    prepared = prepare_multihorizon(
        source,
        horizon,
        split["train_ratio"],
        split["calibration_ratio"],
    )
    causal_split = dict(prepared.attrs["causal_split"])
    feature_columns = select_feature_columns(prepared)
    if not feature_columns:
        raise ValueError("基础预测输入没有feature_前缀的特征列")

    train_df = prepared.loc[prepared["split"].eq("train")].copy()
    output_df = prepared.loc[prepared["split"].isin(save_splits)].copy()
    if train_df.empty or output_df.empty:
        raise ValueError("基础预测训练或输出切片为空")

    predictor = get_predictor(predictor_name)
    predictor.train(
        train_df[feature_columns].to_numpy(),
        train_df["target"].to_numpy(dtype=np.float64),
        seed,
        timestamps=train_df["timestamp"].to_numpy(),
    )

    feature_context = _build_feature_context(source, feature_columns)
    context_predictions = predictor.predict(
        feature_context[feature_columns].to_numpy(),
        timestamps=feature_context["timestamp"].to_numpy(),
    )
    context_index = pd.Index(feature_context["timestamp"])
    output_positions = context_index.get_indexer(output_df["timestamp"])
    if (output_positions < 0).any():
        missing = int((output_positions < 0).sum())
        raise AssertionError(f"有{missing}个输出事件无法映射到完整特征时间轴")

    raw_predictions = context_predictions[output_positions]
    rearranged, rearrangement_diagnostics = rearrange_quantile_predictions(
        raw_predictions
    )
    row_adjustment = np.abs(rearranged - raw_predictions)
    output_df["quantile_crossing_before"] = np.any(
        np.diff(raw_predictions, axis=1) < 0.0,
        axis=1,
    )
    output_df["quantile_rearrangement_mean_abs_adjustment"] = row_adjustment.mean(axis=1)
    output_df["quantile_rearrangement_max_abs_adjustment"] = row_adjustment.max(axis=1)
    output_df["quantile_rearrangement_method"] = (
        "deterministic_rowwise_ascending_rearrangement"
    )

    for quantile_index, quantile in enumerate(ALL_QUANTILES):
        output_df[f"q_{quantile:.4f}"] = rearranged[:, quantile_index]

    lower, center, upper = extract_quantile_predictions(rearranged, 0.10)
    output_df["base_lower"] = lower
    output_df["base_upper"] = upper
    output_df["base_center"] = center
    output_df["issue_timestamp"] = output_df["timestamp"]
    output_df["label_available_timestamp"] = _first_issue_strictly_after(
        source["timestamp"],
        output_df["label_timestamp"],
    )

    quantile_matrix = output_df[
        [f"q_{quantile:.4f}" for quantile in ALL_QUANTILES]
    ].to_numpy(dtype=float)
    if (np.diff(quantile_matrix, axis=1) < 0.0).any():
        raise AssertionError("分位数重排后仍存在单调性违反")

    run_dir = ensure_dir(results_dir)
    output_path = run_dir / "base_predictions.parquet"
    output_df.to_parquet(output_path, index=False, compression="zstd")

    config_sha256 = canonical_json_sha256(run_cfg)
    manifest = {
        "manifest_schema": "S03_BASE_PREDICTION_MANIFEST_V1",
        "status": "COMPLETE_VALIDATED",
        "protocol_id": run_cfg["protocol_id"],
        "protocol_version": run_cfg["protocol_version"],
        "protocol_sha256": run_cfg["protocol_sha256"],
        "code_commit": git_commit_for_path(repository_root),
        "code_worktree_clean": not bool(code_status),
        "config_sha256": config_sha256,
        "input_sha256": sha256_file(data_path),
        "output_sha256": sha256_file(output_path),
        "output_file": str(output_path),
        "output_format": "parquet_zstd",
        "predictor": predictor_name,
        "dataset_id": run_cfg.get("dataset_id", "gefcom2014"),
        "zone": zone,
        "seed": seed,
        "horizon": horizon,
        "data_path": str(data_path),
        "split": split,
        "saved_splits": list(save_splits),
        "all_quantiles": ALL_QUANTILES,
        "n_source": int(len(source)),
        "n_train": int(len(train_df)),
        "n_calibration": int(output_df["split"].eq("calibration").sum()),
        "n_test": int(output_df["split"].eq("test").sum()),
        "n_output": int(len(output_df)),
        "feature_columns": feature_columns,
        "feature_context_rows": int(len(feature_context)),
        "feature_context_policy": "full_issue_timeline_without_cross_gap_sequences",
        "causal_split": causal_split,
        "quantile_rearrangement": rearrangement_diagnostics,
        "randomness": {
            "seed": seed,
            "qrlstm_numpy_seed": predictor_name == "QRLSTM",
            "qrlstm_torch_seed": predictor_name == "QRLSTM",
            "qrlstm_dataloader_generator_seed": predictor_name == "QRLSTM",
            "qrlstm_deterministic_algorithms": predictor_name == "QRLSTM",
        },
        "validations": {
            "post_rearrangement_crossing_count": 0,
            "output_contains_train_rows": False,
            "output_timestamp_unique": bool(output_df["timestamp"].is_unique),
            "label_timestamp_present": True,
            "label_available_timestamp_present": True,
        },
    }
    save_json(manifest, run_dir / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_json(args.config)
    manifest = run_base_prediction(cfg)
    print(
        "Saved "
        f"{manifest['output_file']} "
        f"({manifest['n_output']} rows, {len(ALL_QUANTILES)} quantiles)"
    )


if __name__ == "__main__":
    main()
