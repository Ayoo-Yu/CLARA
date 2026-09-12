"""按基础预测流生成规范化候选区间包。"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import (
    canonical_json_sha256,
    ensure_dir,
    git_commit_for_path,
    git_status_porcelain,
    load_json,
    save_json,
    sha256_file,
)
from run_calibration import (
    ACTION_MAP,
    build_candidate_frames,
    prepare_candidate_base,
)


NORMALIZED_BUNDLE_SCHEMA = "S03_NORMALIZED_CANDIDATE_BUNDLE_V1"
FROZEN_METHOD_ORDER = ("SplitCF", "ACI", "AgACI", "EnbPI")
EXPECTED_ACTIONS = tuple(ACTION_MAP[method] for method in FROZEN_METHOD_ORDER)
REQUIRED_BUNDLE_FIELDS = (
    "protocol_id",
    "protocol_version",
    "protocol_sha256",
    "baseline_registry_sha256",
    "output_contract_sha256",
    "dataset_id",
    "predictor",
    "zone",
    "seed",
    "horizon",
    "base_predictions_path",
    "target_coverages",
    "methods",
    "results_dir",
)

EVENT_CORE_COLUMNS = (
    "event_id",
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
    "base_center",
    "base_upper",
    "quantile_crossing_before",
    "quantile_rearrangement_mean_abs_adjustment",
    "quantile_rearrangement_max_abs_adjustment",
    "quantile_rearrangement_method",
    "dataset_id",
    "zone_or_farm",
    "predictor",
    "seed",
    "alpha",
    "target_coverage",
)

CANDIDATE_COLUMNS = (
    "event_id",
    "target_coverage",
    "action",
    "candidate_lower",
    "candidate_upper",
    "preclip_lower",
    "preclip_upper",
    "candidate_clip_applied",
    "feedback_revision_before",
    "feedback_applied_at_issue",
    "calibrator_state_value_before",
    "enbpi_history_n_before",
    "enbpi_history_start_before",
    "enbpi_history_end_before",
    "enbpi_history_duration_hours",
)

FEEDBACK_COLUMNS = (
    "event_id",
    "action",
    "feedback_application_timestamp",
    "eligible_by_strict_time_rule",
    "candidate_calibrator_updated",
    "rolling_diagnostic_updated",
    "upper_landscape_updated",
    "calibrator_revision_before",
    "calibrator_revision_after",
)


def _validate_bundle_config(cfg: dict[str, Any]) -> tuple[float, ...]:
    missing = []
    for field in REQUIRED_BUNDLE_FIELDS:
        if field not in cfg or cfg[field] is None:
            missing.append(field)
            continue
        if isinstance(cfg[field], (str, list, tuple, dict)) and not cfg[field]:
            missing.append(field)
    if missing:
        raise ValueError(f"规范化候选包配置缺少字段: {missing}")
    methods = tuple(cfg["methods"])
    if len(methods) != len(set(methods)) or set(methods) != set(FROZEN_METHOD_ORDER):
        raise ValueError(f"规范化候选包必须包含四个冻结方法: {FROZEN_METHOD_ORDER}")
    if cfg.get("compute_performance_metrics", False):
        raise ValueError("S03规范化候选包不得计算性能比较")
    coverages = tuple(sorted(round(float(value), 10) for value in cfg["target_coverages"]))
    if not coverages or len(coverages) != len(set(coverages)):
        raise ValueError("目标覆盖率必须非空且唯一")
    if any(value <= 0.0 or value >= 1.0 for value in coverages):
        raise ValueError("目标覆盖率必须严格位于0到1之间")
    return coverages


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], table: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{table}缺少规范化字段: {missing}")


def _event_core(candidate: pd.DataFrame) -> pd.DataFrame:
    _require_columns(candidate, EVENT_CORE_COLUMNS, "候选事件")
    return candidate.loc[:, EVENT_CORE_COLUMNS].copy()


def _normalized_candidates(candidate: pd.DataFrame) -> pd.DataFrame:
    normalized = candidate.copy()
    if "enbpi_history_n_before" not in normalized.columns:
        normalized["enbpi_history_n_before"] = pd.array(
            [pd.NA] * len(normalized), dtype="Int64"
        )
    if "enbpi_history_start_before" not in normalized.columns:
        normalized["enbpi_history_start_before"] = pd.Series(
            pd.NaT, index=normalized.index, dtype="datetime64[ns]"
        )
    if "enbpi_history_end_before" not in normalized.columns:
        normalized["enbpi_history_end_before"] = pd.Series(
            pd.NaT, index=normalized.index, dtype="datetime64[ns]"
        )
    if "enbpi_history_duration_hours" not in normalized.columns:
        normalized["enbpi_history_duration_hours"] = np.nan
    _require_columns(normalized, CANDIDATE_COLUMNS, "候选区间")
    return normalized.loc[:, CANDIDATE_COLUMNS].copy()


def _normalized_feedback(feedback: pd.DataFrame) -> pd.DataFrame:
    _require_columns(feedback, FEEDBACK_COLUMNS, "反馈轨迹")
    return feedback.loc[:, FEEDBACK_COLUMNS].copy()


def _assert_exact_event_core(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_coverage: float,
    action: str,
) -> None:
    try:
        pd.testing.assert_frame_equal(reference, current, check_exact=True)
    except AssertionError as exc:
        raise AssertionError(
            f"覆盖率{target_coverage:.4f}的{action}事件核心与其他动作不一致"
        ) from exc


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    frame.to_parquet(
        temporary_path,
        index=False,
        compression="zstd",
        engine="pyarrow",
    )
    temporary_path.replace(path)


def _logical_child_config(
    bundle_cfg: dict[str, Any],
    target_coverage: float,
    method: str,
) -> dict[str, Any]:
    return {
        "protocol_id": bundle_cfg["protocol_id"],
        "protocol_version": bundle_cfg["protocol_version"],
        "protocol_sha256": bundle_cfg["protocol_sha256"],
        "baseline_registry_sha256": bundle_cfg["baseline_registry_sha256"],
        "output_contract_sha256": bundle_cfg["output_contract_sha256"],
        "dataset_id": bundle_cfg["dataset_id"],
        "method": method,
        "alpha": round(1.0 - target_coverage, 10),
        "horizon": int(bundle_cfg["horizon"]),
        "seed": int(bundle_cfg["seed"]),
        "zone": bundle_cfg["zone"],
        "predictor": bundle_cfg["predictor"],
        "base_predictions_path": bundle_cfg["base_predictions_path"],
        "compute_performance_metrics": False,
        "require_clean_code": bool(bundle_cfg.get("require_clean_code", True)),
    }


def validate_candidate_bundle_cache(
    manifest_path: str | Path,
    cfg: dict[str, Any],
) -> list[str]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return ["manifest_missing"]
    manifest = load_json(manifest_path)
    repository_root = Path(__file__).resolve().parents[1]
    expected = {
        "manifest_schema": NORMALIZED_BUNDLE_SCHEMA,
        "status": "COMPLETE_VALIDATED",
        "protocol_id": cfg["protocol_id"],
        "protocol_version": cfg["protocol_version"],
        "protocol_sha256": cfg["protocol_sha256"],
        "baseline_registry_sha256": cfg["baseline_registry_sha256"],
        "output_contract_sha256": cfg["output_contract_sha256"],
        "code_commit": git_commit_for_path(repository_root),
        "config_sha256": canonical_json_sha256(cfg),
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    for hash_field, path_field in (
        ("event_core_sha256", "event_core_file"),
        ("candidate_intervals_sha256", "candidate_file"),
        ("feedback_trace_sha256", "feedback_file"),
    ):
        output_path = Path(manifest.get(path_field, ""))
        if not output_path.exists():
            mismatches.append(f"{path_field}_missing")
        elif manifest.get(hash_field) != sha256_file(output_path):
            mismatches.append(hash_field)
    return sorted(set(mismatches))


def run_candidate_bundle(bundle_cfg: dict[str, Any]) -> dict[str, Any]:
    coverages = _validate_bundle_config(bundle_cfg)
    repository_root = Path(__file__).resolve().parents[1]
    code_status = git_status_porcelain(repository_root)
    if code_status and bundle_cfg.get("require_clean_code", True):
        raise RuntimeError("权威代码工作树不干净，禁止生成可批准规范化候选包")

    run_dir = ensure_dir(bundle_cfg["results_dir"])
    output_paths = (
        run_dir / "event_core.parquet",
        run_dir / "candidate_intervals.parquet",
        run_dir / "feedback_trace.parquet",
        run_dir / "run_config.json",
        run_dir / "manifest.json",
    )
    if any(path.exists() for path in output_paths):
        raise FileExistsError(f"规范化候选包目录已有输出，禁止覆盖: {run_dir}")

    base_path = Path(bundle_cfg["base_predictions_path"])
    base_manifest_path = base_path.parent / "manifest.json"
    if not base_path.exists() or not base_manifest_path.exists():
        raise FileNotFoundError("规范化候选包缺少基础预测或其运行清单")
    code_commit = git_commit_for_path(repository_root)
    prepared_base = prepare_candidate_base(
        bundle_cfg,
        code_status=code_status,
        code_commit=code_commit,
    )
    base_manifest = prepared_base["base_manifest"]

    event_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    feedback_frames: list[pd.DataFrame] = []
    stream_records: list[dict[str, Any]] = []

    for target_coverage in coverages:
        reference_event_core: pd.DataFrame | None = None
        event_identity: tuple[np.ndarray, list[str]] | None = None
        for method in FROZEN_METHOD_ORDER:
            action = ACTION_MAP[method]
            logical_cfg = _logical_child_config(
                bundle_cfg,
                target_coverage,
                method,
            )
            built = build_candidate_frames(
                logical_cfg,
                prepared_base,
                event_identity=event_identity,
            )
            if event_identity is None:
                event_identity = built["event_identity"]
            candidate = built["candidate"]
            feedback = built["feedback"]

            current_event_core = _event_core(candidate)
            if reference_event_core is None:
                reference_event_core = current_event_core
                event_frames.append(current_event_core)
            else:
                _assert_exact_event_core(
                    reference_event_core,
                    current_event_core,
                    target_coverage,
                    action,
                )
            candidate_frames.append(_normalized_candidates(candidate))
            feedback_frames.append(_normalized_feedback(feedback))
            stream_records.append(
                {
                    "target_coverage": target_coverage,
                    "alpha": round(1.0 - target_coverage, 10),
                    "method": method,
                    "action": action,
                    "run_id": built["run_id"],
                    "logical_config_sha256": canonical_json_sha256(logical_cfg),
                    "method_config": built["method_config"],
                    "candidate_rows": int(len(candidate)),
                    "feedback_rows": int(len(feedback)),
                }
            )

    event_core = pd.concat(event_frames, ignore_index=True)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    feedback = pd.concat(feedback_frames, ignore_index=True)
    candidates["enbpi_history_n_before"] = candidates[
        "enbpi_history_n_before"
    ].astype("Int64")

    duplicate_events = int(event_core["event_id"].duplicated().sum())
    duplicate_candidates = int(
        candidates.duplicated(subset=["event_id", "action"]).sum()
    )
    duplicate_feedback = int(
        feedback.duplicated(
            subset=["event_id", "action", "feedback_application_timestamp"]
        ).sum()
    )
    event_action_counts = candidates.groupby("event_id", sort=False)["action"].nunique()
    incomplete_action_events = int((event_action_counts != len(EXPECTED_ACTIONS)).sum())
    unexpected_actions = sorted(set(candidates["action"]) - set(EXPECTED_ACTIONS))
    event_ids = set(event_core["event_id"].astype(str))
    candidate_event_ids = set(candidates["event_id"].astype(str))
    feedback_event_ids = set(feedback["event_id"].astype(str))
    missing_candidate_events = len(event_ids - candidate_event_ids)
    orphan_candidate_events = len(candidate_event_ids - event_ids)
    orphan_feedback_events = len(feedback_event_ids - event_ids)

    lower = candidates["candidate_lower"].to_numpy(dtype=float)
    upper = candidates["candidate_upper"].to_numpy(dtype=float)
    nonfinite_candidates = int((~np.isfinite(lower) | ~np.isfinite(upper)).sum())
    reversed_candidates = int((lower > upper).sum())
    out_of_bounds_candidates = int(((lower < 0.0) | (upper > 1.0)).sum())

    feedback_with_labels = feedback.merge(
        event_core.loc[:, ["event_id", "label_timestamp"]],
        on="event_id",
        how="left",
        validate="many_to_one",
    )
    missing_feedback_labels = int(feedback_with_labels["label_timestamp"].isna().sum())
    strict_feedback_violations = int(
        (
            pd.to_datetime(feedback_with_labels["label_timestamp"])
            >= pd.to_datetime(feedback_with_labels["feedback_application_timestamp"])
        ).sum()
    )

    validation = {
        "duplicate_event_ids": duplicate_events,
        "duplicate_event_action_rows": duplicate_candidates,
        "duplicate_feedback_rows": duplicate_feedback,
        "incomplete_action_events": incomplete_action_events,
        "unexpected_actions": unexpected_actions,
        "missing_candidate_events": missing_candidate_events,
        "orphan_candidate_events": orphan_candidate_events,
        "orphan_feedback_events": orphan_feedback_events,
        "missing_feedback_labels": missing_feedback_labels,
        "nonfinite_candidate_rows": nonfinite_candidates,
        "reversed_interval_rows": reversed_candidates,
        "out_of_bounds_rows": out_of_bounds_candidates,
        "strict_feedback_violations": strict_feedback_violations,
        "candidate_rows_equal_events_times_actions": bool(
            len(candidates) == len(event_core) * len(EXPECTED_ACTIONS)
        ),
        "performance_comparison_performed": False,
    }
    failed_checks = [
        key
        for key, value in validation.items()
        if key != "performance_comparison_performed"
        and (
            (isinstance(value, bool) and not value)
            or (
                isinstance(value, int)
                and not isinstance(value, bool)
                and value != 0
            )
            or (isinstance(value, list) and bool(value))
        )
    ]
    if failed_checks:
        raise AssertionError(
            "规范化候选包质量检查失败: "
            f"{failed_checks}; event_rows={len(event_core)}; "
            f"candidate_rows={len(candidates)}; "
            f"expected_actions={len(EXPECTED_ACTIONS)}"
        )

    event_core_path = run_dir / "event_core.parquet"
    candidate_path = run_dir / "candidate_intervals.parquet"
    feedback_path = run_dir / "feedback_trace.parquet"
    _write_parquet_atomic(event_core, event_core_path)
    _write_parquet_atomic(candidates, candidate_path)
    _write_parquet_atomic(feedback, feedback_path)
    save_json(bundle_cfg, run_dir / "run_config.json")

    bundle_id = bundle_cfg.get(
        "bundle_id",
        (
            f"{bundle_cfg['predictor']}-H{int(bundle_cfg['horizon']):02d}-"
            f"{bundle_cfg['zone']}-S{int(bundle_cfg['seed'])}"
        ),
    )
    manifest = {
        "manifest_schema": NORMALIZED_BUNDLE_SCHEMA,
        "status": "COMPLETE_VALIDATED",
        "protocol_id": bundle_cfg["protocol_id"],
        "protocol_version": bundle_cfg["protocol_version"],
        "protocol_sha256": bundle_cfg["protocol_sha256"],
        "baseline_registry_sha256": bundle_cfg["baseline_registry_sha256"],
        "output_contract_sha256": bundle_cfg["output_contract_sha256"],
        "code_commit": code_commit,
        "code_worktree_clean": not bool(code_status),
        "config_sha256": canonical_json_sha256(bundle_cfg),
        "bundle_id": bundle_id,
        "dataset_id": bundle_cfg["dataset_id"],
        "predictor": bundle_cfg["predictor"],
        "zone": bundle_cfg["zone"],
        "seed": int(bundle_cfg["seed"]),
        "horizon": int(bundle_cfg["horizon"]),
        "target_coverages": list(coverages),
        "actions": list(EXPECTED_ACTIONS),
        "base_manifest_schema": base_manifest.get("manifest_schema"),
        "base_code_commit": base_manifest.get("code_commit"),
        "base_manifest_sha256": sha256_file(base_manifest_path),
        "base_predictions_sha256": sha256_file(base_path),
        "event_core_file": str(event_core_path),
        "candidate_file": str(candidate_path),
        "feedback_file": str(feedback_path),
        "event_core_sha256": sha256_file(event_core_path),
        "candidate_intervals_sha256": sha256_file(candidate_path),
        "feedback_trace_sha256": sha256_file(feedback_path),
        "event_rows": int(len(event_core)),
        "candidate_rows": int(len(candidates)),
        "feedback_rows": int(len(feedback)),
        "event_core_columns": list(EVENT_CORE_COLUMNS),
        "candidate_columns": list(CANDIDATE_COLUMNS),
        "feedback_columns": list(FEEDBACK_COLUMNS),
        "stream_records": stream_records,
        "validations": validation,
        "storage_layout": {
            "event_time_and_target_fields_stored_once_per_event": True,
            "candidate_alias_columns_removed": True,
            "feedback_event_time_join_key": "event_id",
            "one_config_and_manifest_per_bundle": True,
            "compression": "zstd",
        },
    }
    save_json(manifest, run_dir / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    manifest = run_candidate_bundle(load_json(args.config))
    print(
        f"Saved normalized candidate bundle {manifest['bundle_id']}: "
        f"{manifest['event_rows']} events, "
        f"{manifest['candidate_rows']} candidates, "
        f"{manifest['feedback_rows']} feedback rows"
    )


if __name__ == "__main__":
    main()
