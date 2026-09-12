from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil


S06_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = S06_ROOT.parent
AUTH_ROOT = REVISION_ROOT / "_权威代码"
CODE_ROOT = AUTH_ROOT / "code"
RUN_ROOT = S06_ROOT / "results_raw" / "nested_source_selection_v1"
QA_ROOT = S06_ROOT / "qa" / "state_statistics_preflight_v1"
sys.path.insert(0, str(CODE_ROOT))

from clara_event_contract import load_frozen_contracts, sha256_file  # noqa: E402


EXPECTED_CODE_COMMIT = "980fad5174840883318b15a225957f70df8928f6"
SOURCE_TUNING_CONTRACT_SHA256 = "9a352cba41d7e68911626da950540d1d8506f2f4f63afb72bd7b28cd473c7e14"
SOURCE_FACT_ROOT_MANIFEST_SHA256 = "30a0f86d6f775c7c32a3654117807dfa9acabc8187c2bdefeb8440e1466ef9e7"
WIDTH_THRESHOLD_FILE_SHA256 = "ef3ebd2ed7dc4105212ea27fa0d62729c5101416eff8aba2a37c166ceced96ba"
EXPECTED_STREAM_COUNT = 2880
EXPECTED_COMPLETE_EVENT_COUNT = 110705760
EXPECTED_ALL_FOLD_SOURCE_EVENT_COUNT = EXPECTED_COMPLETE_EVENT_COUNT * 9
EXPECTED_THRESHOLD_ROW_COUNT = 6600
EXPECTED_PREFLIGHT_STREAM_COUNT = 24
MINIMUM_FREE_DISK_GIB = 60.0
CHECKPOINT_INTERVAL = 240
FLOAT_SUM_EQUIVALENCE_TOLERANCE = 5e-10
DERIVED_MEAN_EQUIVALENCE_TOLERANCE = 5e-13


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=AUTH_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def zone_number(zone: str) -> int:
    text = str(zone)
    if text.startswith("zone") and text[4:].isdigit():
        return int(text[4:])
    raise ValueError(f"区域编号无法解析: {text}")


def prefix_digest(stream_ids: list[str]) -> str:
    payload = "\n".join(stream_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_identity() -> tuple[pd.DataFrame, dict[str, Any]]:
    if git_output("rev-parse", "HEAD") != EXPECTED_CODE_COMMIT:
        raise RuntimeError("状态充分统计提交身份失配")
    if git_output("status", "--porcelain"):
        raise RuntimeError("状态充分统计要求权威代码工作树干净")
    contract_path = S06_ROOT / "configs" / "source_tuning_contract_v1.json"
    if sha256_file(contract_path) != SOURCE_TUNING_CONTRACT_SHA256:
        raise RuntimeError("源区嵌套选择合同SHA256失配")
    root_manifest_path = RUN_ROOT / "source_fact_root_manifest.json"
    if sha256_file(root_manifest_path) != SOURCE_FACT_ROOT_MANIFEST_SHA256:
        raise RuntimeError("全量源域事实根清单SHA256失配")
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    ledger_path = RUN_ROOT / "checkpoints" / "source_fact_ledger.csv"
    if sha256_file(ledger_path) != str(root_manifest["source_fact_ledger_sha256"]):
        raise RuntimeError("源域事实账本SHA256与根清单失配")
    threshold_path = RUN_ROOT / "fold_width_thresholds.parquet"
    if sha256_file(threshold_path) != WIDTH_THRESHOLD_FILE_SHA256:
        raise RuntimeError("外层折宽度阈值文件SHA256失配")
    if shutil.disk_usage(REVISION_ROOT).free / (1024.0**3) < MINIMUM_FREE_DISK_GIB:
        raise RuntimeError("状态充分统计磁盘安全门失败")
    ledger = pd.read_csv(ledger_path, keep_default_na=False)
    completed = ledger[ledger["status"].astype(str) == "COMPLETED"].copy()
    if len(completed) != EXPECTED_STREAM_COUNT:
        raise RuntimeError("状态充分统计输入流数量失配")
    completed["_zone_number"] = completed["zone"].map(zone_number)
    completed = (
        completed.sort_values(
            ["predictor", "_zone_number", "horizon", "seed"],
            kind="mergesort",
        )
        .drop(columns="_zone_number")
        .reset_index(drop=True)
    )
    if int(completed["complete_case_event_count"].sum()) != EXPECTED_COMPLETE_EVENT_COUNT:
        raise RuntimeError("状态充分统计完整案例事件总数失配")
    return completed, root_manifest


def build_vocabularies() -> dict[str, tuple[Any, ...]]:
    contracts = load_frozen_contracts()
    protocol = contracts.protocol
    vocabularies: dict[str, tuple[Any, ...]] = {
        "zones": tuple(str(value) for value in protocol["datasets"]["gefcom2014"]["zones"]),
        "predictors": tuple(str(value) for value in protocol["predictor_contract"]["predictors"]),
        "horizon_groups": tuple(str(value) for value in protocol["horizon_group_contract"]["groups"].keys()),
        "coverages": tuple(float(value) for value in contracts.coverages),
        "ramp_states": tuple(str(value) for value in protocol["state_contract"]["ramp"]["regimes"]),
        "rolling_states": tuple(
            str(value) for value in protocol["state_contract"]["rolling_reliability"]["states"]
        ),
        "width_states": tuple(str(value) for value in protocol["state_contract"]["raw_width"]["states"]),
        "actions": tuple(str(value) for value in contracts.actions),
        "seeds": (0, 1, 2),
    }
    expected_sizes = {
        "zones": 10,
        "predictors": 4,
        "horizon_groups": 5,
        "coverages": 11,
        "ramp_states": 2,
        "rolling_states": 5,
        "width_states": 3,
        "actions": 4,
        "seeds": 3,
    }
    observed_sizes = {key: len(values) for key, values in vocabularies.items()}
    if observed_sizes != expected_sizes:
        raise RuntimeError(f"状态充分统计冻结词表尺寸失配: {observed_sizes}")
    return vocabularies


def load_threshold_cube(vocabularies: dict[str, tuple[Any, ...]]) -> np.ndarray:
    thresholds = pd.read_parquet(RUN_ROOT / "fold_width_thresholds.parquet")
    if len(thresholds) != EXPECTED_THRESHOLD_ROW_COUNT:
        raise RuntimeError("外层折宽度阈值行数失配")
    shape = (
        len(vocabularies["zones"]),
        len(vocabularies["predictors"]),
        len(vocabularies["horizon_groups"]),
        len(vocabularies["coverages"]),
        len(vocabularies["seeds"]),
        2,
    )
    cube = np.full(shape, np.nan, dtype=np.float64)
    maps = {
        key: {value: index for index, value in enumerate(vocabularies[key])}
        for key in ("zones", "predictors", "horizon_groups", "coverages", "seeds")
    }
    seen: set[tuple[int, int, int, int, int]] = set()
    for row in thresholds.itertuples(index=False):
        key = (
            maps["zones"][str(row.outer_heldout_zone)],
            maps["predictors"][str(row.predictor)],
            maps["horizon_groups"][str(row.horizon_group)],
            maps["coverages"][float(row.target_coverage)],
            maps["seeds"][int(row.seed)],
        )
        if key in seen:
            raise RuntimeError("外层折宽度阈值键重复")
        seen.add(key)
        cube[key + (0,)] = float(row.raw_width_q33)
        cube[key + (1,)] = float(row.raw_width_q67)
    if len(seen) != EXPECTED_THRESHOLD_ROW_COUNT or not np.isfinite(cube).all():
        raise RuntimeError("外层折宽度阈值立方体不完整")
    if np.any(cube[..., 0] > cube[..., 1]) or np.any(cube < 0.0):
        raise RuntimeError("外层折宽度阈值含非法值")
    return cube


def state_count(vocabularies: dict[str, tuple[Any, ...]]) -> int:
    result = 1
    for key in (
        "predictors",
        "horizon_groups",
        "coverages",
        "ramp_states",
        "rolling_states",
        "width_states",
    ):
        result *= len(vocabularies[key])
    return result


def encode_base_state(
    *,
    frame: pd.DataFrame,
    predictor_index: int,
    horizon_group_index: int,
    vocabularies: dict[str, tuple[Any, ...]],
) -> tuple[np.ndarray, np.ndarray]:
    coverages = np.asarray(vocabularies["coverages"], dtype=np.float64)
    observed_coverages = frame["target_coverage"].to_numpy(dtype=np.float64)
    coverage_code = np.searchsorted(coverages, observed_coverages)
    if np.any(coverage_code < 0) or np.any(coverage_code >= len(coverages)):
        raise RuntimeError("状态充分统计覆盖率编码越界")
    if not np.array_equal(coverages[coverage_code], observed_coverages):
        raise RuntimeError("状态充分统计出现冻结集合外覆盖率")
    ramp_map = {value: index for index, value in enumerate(vocabularies["ramp_states"])}
    rolling_map = {value: index for index, value in enumerate(vocabularies["rolling_states"])}
    ramp_code = frame["ramp_state"].astype(str).map(ramp_map)
    rolling_code = frame["rolling_state"].astype(str).map(rolling_map)
    if ramp_code.isna().any() or rolling_code.isna().any():
        raise RuntimeError("状态充分统计出现冻结集合外状态")
    base = np.full(len(frame), predictor_index, dtype=np.int64)
    base = base * len(vocabularies["horizon_groups"]) + horizon_group_index
    base = base * len(vocabularies["coverages"]) + coverage_code
    base = base * len(vocabularies["ramp_states"]) + ramp_code.to_numpy(dtype=np.int64)
    base = base * len(vocabularies["rolling_states"]) + rolling_code.to_numpy(dtype=np.int64)
    return base, coverage_code


def extract_metrics(
    *,
    frame: pd.DataFrame,
    actions: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    errf = np.column_stack([frame[f"{action}__errf"].to_numpy(dtype=np.float64) for action in actions])
    covered = np.column_stack(
        [frame[f"{action}__covered"].to_numpy(dtype=np.float64) for action in actions]
    )
    tuwr = np.column_stack(
        [frame[f"{action}__tuwr_indicator"].to_numpy(dtype=np.float64) for action in actions]
    )
    if not np.isfinite(errf).all() or np.any(errf < 0.0):
        raise RuntimeError("状态充分统计输入ERRF非法")
    if not np.isfinite(covered).all() or not np.isin(covered, (0.0, 1.0)).all():
        raise RuntimeError("状态充分统计输入covered非法")
    tuwr_finite = np.isfinite(tuwr)
    if np.any(tuwr[tuwr_finite] < 0.0) or np.any(tuwr[tuwr_finite] > 1.0):
        raise RuntimeError("状态充分统计输入TUWR非法")
    tuwr_zero_filled = np.where(tuwr_finite, tuwr, 0.0)
    best_action = np.argmin(errf, axis=1).astype(np.int64)
    return errf, covered, tuwr_zero_filled, tuwr_finite.astype(np.float64), best_action


def aggregate_one_width_state(
    *,
    state_code: np.ndarray,
    errf: np.ndarray,
    covered: np.ndarray,
    tuwr_zero_filled: np.ndarray,
    tuwr_finite: np.ndarray,
    best_action: np.ndarray,
    n_states: int,
    n_actions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    event_count = np.bincount(state_code, minlength=n_states).astype(np.int64)
    action_code = (state_code[:, None] * n_actions + np.arange(n_actions, dtype=np.int64)[None, :]).ravel()
    action_size = n_states * n_actions
    errf_sum = np.bincount(action_code, weights=errf.ravel(), minlength=action_size).reshape(n_states, n_actions)
    covered_sum = np.bincount(
        action_code,
        weights=covered.ravel(),
        minlength=action_size,
    ).reshape(n_states, n_actions)
    tuwr_sum = np.bincount(
        action_code,
        weights=tuwr_zero_filled.ravel(),
        minlength=action_size,
    ).reshape(n_states, n_actions)
    tuwr_count = np.bincount(
        action_code,
        weights=tuwr_finite.ravel(),
        minlength=action_size,
    ).reshape(n_states, n_actions)
    best_code = state_code * n_actions + best_action
    best_count = np.bincount(best_code, minlength=action_size).reshape(n_states, n_actions).astype(np.int64)
    return event_count, errf_sum, covered_sum, tuwr_sum, tuwr_count, best_count


def read_stream_frame(row: Any, *, verify_hash: bool) -> pd.DataFrame:
    fact_path = Path(str(row.facts_file))
    if verify_hash and sha256_file(fact_path) != str(row.facts_sha256):
        raise RuntimeError(f"源域事实文件SHA256失配: {row.stream_id}")
    columns = [
        "predictor",
        "horizon_group",
        "target_coverage",
        "seed",
        "ramp_state",
        "rolling_state",
        "raw_width_value",
    ]
    for action in load_frozen_contracts().actions:
        columns.extend(
            [
                f"{action}__errf",
                f"{action}__covered",
                f"{action}__tuwr_indicator",
            ]
        )
    frame = pd.read_parquet(fact_path, columns=columns)
    if len(frame) != int(row.complete_case_event_count):
        raise RuntimeError(f"状态充分统计输入行数失配: {row.stream_id}")
    if set(frame["predictor"].astype(str)) != {str(row.predictor)}:
        raise RuntimeError(f"状态充分统计预测器身份失配: {row.stream_id}")
    if set(frame["seed"].astype(int)) != {int(row.seed)}:
        raise RuntimeError(f"状态充分统计种子身份失配: {row.stream_id}")
    if frame["horizon_group"].astype(str).nunique() != 1:
        raise RuntimeError(f"状态充分统计时长组不唯一: {row.stream_id}")
    widths = frame["raw_width_value"].to_numpy(dtype=np.float64)
    if not np.isfinite(widths).all() or np.any(widths < 0.0):
        raise RuntimeError(f"状态充分统计输入宽度非法: {row.stream_id}")
    return frame


def reference_aggregate(
    *,
    state_code: np.ndarray,
    errf: np.ndarray,
    covered: np.ndarray,
    tuwr: np.ndarray,
    best_action: np.ndarray,
    n_states: int,
    n_actions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reference = pd.DataFrame({"state_code": state_code, "best_action": best_action})
    for action_index in range(n_actions):
        reference[f"errf_{action_index}"] = errf[:, action_index]
        reference[f"covered_{action_index}"] = covered[:, action_index]
        reference[f"tuwr_{action_index}"] = tuwr[:, action_index]
    grouped = reference.groupby("state_code", sort=True)
    index = pd.RangeIndex(n_states)
    event_count = grouped.size().reindex(index, fill_value=0).to_numpy(dtype=np.int64)
    errf_sum = np.column_stack(
        [grouped[f"errf_{i}"].sum().reindex(index, fill_value=0.0).to_numpy(dtype=float) for i in range(n_actions)]
    )
    covered_sum = np.column_stack(
        [grouped[f"covered_{i}"].sum().reindex(index, fill_value=0.0).to_numpy(dtype=float) for i in range(n_actions)]
    )
    tuwr_sum = np.column_stack(
        [grouped[f"tuwr_{i}"].sum().reindex(index, fill_value=0.0).to_numpy(dtype=float) for i in range(n_actions)]
    )
    tuwr_count = np.column_stack(
        [grouped[f"tuwr_{i}"].count().reindex(index, fill_value=0).to_numpy(dtype=float) for i in range(n_actions)]
    )
    best_count = np.zeros((n_states, n_actions), dtype=np.int64)
    cross = pd.crosstab(reference["state_code"], reference["best_action"])
    for action_index in range(n_actions):
        if action_index in cross.columns:
            best_count[cross.index.to_numpy(dtype=int), action_index] = cross[action_index].to_numpy(dtype=np.int64)
    return event_count, errf_sum, covered_sum, tuwr_sum, tuwr_count, best_count


def run_preflight(
    *,
    completed: pd.DataFrame,
    vocabularies: dict[str, tuple[Any, ...]],
    threshold_cube: np.ndarray,
) -> None:
    QA_ROOT.mkdir(parents=True, exist_ok=True)
    sample_indices = sorted(set(np.linspace(0, len(completed) - 1, EXPECTED_PREFLIGHT_STREAM_COUNT, dtype=int)))
    if len(sample_indices) != EXPECTED_PREFLIGHT_STREAM_COUNT:
        raise RuntimeError("状态充分统计预检抽样数量失配")
    maps = {
        key: {value: index for index, value in enumerate(vocabularies[key])}
        for key in ("zones", "predictors", "horizon_groups", "seeds")
    }
    n_states = state_count(vocabularies)
    n_actions = len(vocabularies["actions"])
    checks: list[dict[str, Any]] = []
    max_float_difference = 0.0
    max_derived_mean_difference = 0.0
    for sample_order, ledger_index in enumerate(sample_indices, start=1):
        row = next(completed.iloc[[ledger_index]].itertuples(index=False))
        frame = read_stream_frame(row, verify_hash=True)
        predictor_index = maps["predictors"][str(row.predictor)]
        horizon_group = str(frame["horizon_group"].iloc[0])
        horizon_group_index = maps["horizon_groups"][horizon_group]
        zone_index = maps["zones"][str(row.zone)]
        seed_index = maps["seeds"][int(row.seed)]
        base, coverage_code = encode_base_state(
            frame=frame,
            predictor_index=predictor_index,
            horizon_group_index=horizon_group_index,
            vocabularies=vocabularies,
        )
        errf, covered, tuwr_zero, tuwr_finite, best = extract_metrics(
            frame=frame,
            actions=vocabularies["actions"],
        )
        original_tuwr = np.where(tuwr_finite.astype(bool), tuwr_zero, np.nan)
        candidate_outer = [index for index in range(len(vocabularies["zones"])) if index != zone_index][:2]
        for outer_index in candidate_outer:
            q33 = threshold_cube[
                outer_index,
                predictor_index,
                horizon_group_index,
                coverage_code,
                seed_index,
                0,
            ]
            q67 = threshold_cube[
                outer_index,
                predictor_index,
                horizon_group_index,
                coverage_code,
                seed_index,
                1,
            ]
            widths = frame["raw_width_value"].to_numpy(dtype=np.float64)
            width_code = np.where(widths <= q33, 0, np.where(widths <= q67, 1, 2)).astype(np.int64)
            codes = base * len(vocabularies["width_states"]) + width_code
            optimized = aggregate_one_width_state(
                state_code=codes,
                errf=errf,
                covered=covered,
                tuwr_zero_filled=tuwr_zero,
                tuwr_finite=tuwr_finite,
                best_action=best,
                n_states=n_states,
                n_actions=n_actions,
            )
            reference = reference_aggregate(
                state_code=codes,
                errf=errf,
                covered=covered,
                tuwr=original_tuwr,
                best_action=best,
                n_states=n_states,
                n_actions=n_actions,
            )
            integer_exact = bool(
                np.array_equal(optimized[0], reference[0])
                and np.array_equal(optimized[5], reference[5])
                and np.array_equal(optimized[2], reference[2])
                and np.array_equal(optimized[4], reference[4])
            )
            differences = [float(np.max(np.abs(optimized[i] - reference[i]))) for i in (1, 2, 3, 4)]
            local_max = max(differences)
            nonzero_event_count = np.maximum(optimized[0][:, None], 1)
            nonzero_tuwr_count = np.maximum(optimized[4], 1.0)
            local_mean_max = max(
                float(np.max(np.abs(optimized[1] - reference[1]) / nonzero_event_count)),
                float(np.max(np.abs(optimized[2] - reference[2]) / nonzero_event_count)),
                float(np.max(np.abs(optimized[3] - reference[3]) / nonzero_tuwr_count)),
            )
            max_float_difference = max(max_float_difference, local_max)
            max_derived_mean_difference = max(max_derived_mean_difference, local_mean_max)
            passed = (
                integer_exact
                and local_max <= FLOAT_SUM_EQUIVALENCE_TOLERANCE
                and local_mean_max <= DERIVED_MEAN_EQUIVALENCE_TOLERANCE
            )
            checks.append(
                {
                    "sample_order": sample_order,
                    "stream_id": str(row.stream_id),
                    "outer_heldout_zone": vocabularies["zones"][outer_index],
                    "event_count": len(frame),
                    "integer_exact": integer_exact,
                    "max_float_difference": local_max,
                    "max_derived_mean_difference": local_mean_max,
                    "status": "PASS" if passed else "FAIL",
                }
            )
    checks_frame = pd.DataFrame(checks)
    checks_path = QA_ROOT / "preflight_checks.csv"
    checks_frame.to_csv(checks_path, index=False, encoding="utf-8")
    passed = bool(checks_frame["status"].eq("PASS").all())
    summary = {
        "qa_id": "S06_STATE_STATISTICS_PREFLIGHT_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "script_sha256": sha256_file(Path(__file__)),
        "code_commit": EXPECTED_CODE_COMMIT,
        "source_fact_root_manifest_sha256": SOURCE_FACT_ROOT_MANIFEST_SHA256,
        "width_threshold_file_sha256": WIDTH_THRESHOLD_FILE_SHA256,
        "sample_stream_count": len(sample_indices),
        "outer_fold_checks_per_stream": 2,
        "check_count": len(checks_frame),
        "pass_count": int(checks_frame["status"].eq("PASS").sum()),
        "integer_mismatch_count": int((~checks_frame["integer_exact"]).sum()),
        "maximum_float_difference": max_float_difference,
        "float_sum_tolerance": FLOAT_SUM_EQUIVALENCE_TOLERANCE,
        "maximum_derived_mean_difference": max_derived_mean_difference,
        "derived_mean_tolerance": DERIVED_MEAN_EQUIVALENCE_TOLERANCE,
        "checks_file": str(checks_path),
        "checks_sha256": sha256_file(checks_path),
        "heldout_zone_metric_use_count": 0,
        "performance_comparison_executed": False,
        "method_ranking_executed": False,
        "statistical_inference_executed": False,
        "paper_performance_claim_executed": False,
        "new_scientific_adverse_evidence": False,
        "stop_triggered": not passed,
    }
    summary_path = QA_ROOT / "preflight_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not passed:
        raise RuntimeError("状态充分统计预检等价性失败")


def checkpoint_paths() -> tuple[Path, Path]:
    checkpoint_root = RUN_ROOT / "checkpoints"
    return (
        checkpoint_root / "state_statistics_checkpoint.npz",
        checkpoint_root / "state_statistics_checkpoint.json",
    )


def save_checkpoint(
    *,
    next_index: int,
    stream_ids: list[str],
    arrays: tuple[np.ndarray, ...],
    resource_rows: list[dict[str, Any]],
) -> None:
    data_path, metadata_path = checkpoint_paths()
    temporary_data_path = data_path.with_name(data_path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary_data_path,
        event_count=arrays[0],
        errf_sum=arrays[1],
        covered_sum=arrays[2],
        tuwr_sum=arrays[3],
        tuwr_count=arrays[4],
        best_count=arrays[5],
    )
    os.replace(temporary_data_path, data_path)
    metadata = {
        "checkpoint_schema": "S06_STATE_STATISTICS_CHECKPOINT_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__)),
        "code_commit": EXPECTED_CODE_COMMIT,
        "source_fact_root_manifest_sha256": SOURCE_FACT_ROOT_MANIFEST_SHA256,
        "width_threshold_file_sha256": WIDTH_THRESHOLD_FILE_SHA256,
        "next_index": next_index,
        "processed_prefix_sha256": prefix_digest(stream_ids[:next_index]),
        "checkpoint_file_sha256": sha256_file(data_path),
    }
    temporary_metadata_path = metadata_path.with_suffix(".tmp.json")
    temporary_metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_metadata_path, metadata_path)
    resource_path = RUN_ROOT / "logs" / "state_statistics_resource_samples.csv"
    pd.DataFrame(resource_rows).to_csv(resource_path, index=False, encoding="utf-8")


def load_or_initialize(
    *,
    stream_ids: list[str],
    shape: tuple[int, int, int],
    n_actions: int,
) -> tuple[int, tuple[np.ndarray, ...], list[dict[str, Any]]]:
    data_path, metadata_path = checkpoint_paths()
    resource_path = RUN_ROOT / "logs" / "state_statistics_resource_samples.csv"
    if data_path.exists() != metadata_path.exists():
        raise RuntimeError("状态充分统计检查点文件不完整")
    if not data_path.exists():
        arrays = (
            np.zeros(shape, dtype=np.int64),
            np.zeros((*shape, n_actions), dtype=np.float64),
            np.zeros((*shape, n_actions), dtype=np.float64),
            np.zeros((*shape, n_actions), dtype=np.float64),
            np.zeros((*shape, n_actions), dtype=np.float64),
            np.zeros((*shape, n_actions), dtype=np.int64),
        )
        return 0, arrays, []
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "checkpoint_schema": "S06_STATE_STATISTICS_CHECKPOINT_V1",
        "script_sha256": sha256_file(Path(__file__)),
        "code_commit": EXPECTED_CODE_COMMIT,
        "source_fact_root_manifest_sha256": SOURCE_FACT_ROOT_MANIFEST_SHA256,
        "width_threshold_file_sha256": WIDTH_THRESHOLD_FILE_SHA256,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"状态充分统计检查点身份失配: {key}")
    if sha256_file(data_path) != str(metadata["checkpoint_file_sha256"]):
        raise RuntimeError("状态充分统计检查点文件SHA256失配")
    next_index = int(metadata["next_index"])
    if next_index < 0 or next_index > len(stream_ids):
        raise RuntimeError("状态充分统计检查点位置非法")
    if prefix_digest(stream_ids[:next_index]) != str(metadata["processed_prefix_sha256"]):
        raise RuntimeError("状态充分统计检查点输入前缀失配")
    with np.load(data_path) as payload:
        arrays = tuple(
            payload[key]
            for key in ("event_count", "errf_sum", "covered_sum", "tuwr_sum", "tuwr_count", "best_count")
        )
    expected_shapes = (shape, *((*shape, n_actions),) * 5)
    if tuple(array.shape for array in arrays) != expected_shapes:
        raise RuntimeError("状态充分统计检查点数组尺寸失配")
    resources = pd.read_csv(resource_path).to_dict("records") if resource_path.exists() else []
    return next_index, arrays, resources


def decode_state_codes(
    *,
    codes: np.ndarray,
    vocabularies: dict[str, tuple[Any, ...]],
) -> dict[str, np.ndarray]:
    remainder = codes.copy()
    decoded: dict[str, np.ndarray] = {}
    for key, output_name in (
        ("width_states", "raw_width_state"),
        ("rolling_states", "rolling_state"),
        ("ramp_states", "ramp_state"),
        ("coverages", "target_coverage"),
        ("horizon_groups", "horizon_group"),
        ("predictors", "predictor"),
    ):
        size = len(vocabularies[key])
        index = remainder % size
        remainder //= size
        values = np.asarray(vocabularies[key])
        decoded[output_name] = values[index]
    if np.any(remainder != 0):
        raise RuntimeError("状态充分统计状态码解码失败")
    return decoded


def build_output_frame(
    *,
    arrays: tuple[np.ndarray, ...],
    vocabularies: dict[str, tuple[Any, ...]],
) -> pd.DataFrame:
    event_count, errf_sum, covered_sum, tuwr_sum, tuwr_count, best_count = arrays
    rows: list[pd.DataFrame] = []
    for outer_index, heldout_zone in enumerate(vocabularies["zones"]):
        for zone_index, source_zone in enumerate(vocabularies["zones"]):
            if source_zone == heldout_zone:
                if np.any(event_count[outer_index, zone_index] != 0):
                    raise RuntimeError("状态充分统计包含外层持出区事件")
                continue
            codes = np.flatnonzero(event_count[outer_index, zone_index] > 0)
            decoded = decode_state_codes(codes=codes, vocabularies=vocabularies)
            payload: dict[str, Any] = {
                "outer_heldout_zone": np.repeat(heldout_zone, len(codes)),
                "source_zone": np.repeat(source_zone, len(codes)),
                "state_code": codes,
                "predictor": decoded["predictor"],
                "horizon_group": decoded["horizon_group"],
                "target_coverage": decoded["target_coverage"].astype(np.float64),
                "ramp_state": decoded["ramp_state"],
                "rolling_state": decoded["rolling_state"],
                "raw_width_state": decoded["raw_width_state"],
                "event_count": event_count[outer_index, zone_index, codes],
            }
            for action_index, action in enumerate(vocabularies["actions"]):
                payload[f"{action}__errf_sum"] = errf_sum[outer_index, zone_index, codes, action_index]
                payload[f"{action}__covered_sum"] = covered_sum[outer_index, zone_index, codes, action_index].astype(np.int64)
                payload[f"{action}__tuwr_sum"] = tuwr_sum[outer_index, zone_index, codes, action_index]
                payload[f"{action}__tuwr_count"] = tuwr_count[outer_index, zone_index, codes, action_index].astype(np.int64)
                payload[f"{action}__best_count"] = best_count[outer_index, zone_index, codes, action_index]
            rows.append(pd.DataFrame(payload))
    result = pd.concat(rows, ignore_index=True)
    zone_rank = {zone: index for index, zone in enumerate(vocabularies["zones"])}
    result["_outer_rank"] = result["outer_heldout_zone"].map(zone_rank)
    result["_source_rank"] = result["source_zone"].map(zone_rank)
    result = (
        result.sort_values(["_outer_rank", "_source_rank", "state_code"], kind="mergesort")
        .drop(columns=["_outer_rank", "_source_rank"])
        .reset_index(drop=True)
    )
    return result


def run_full(
    *,
    completed: pd.DataFrame,
    vocabularies: dict[str, tuple[Any, ...]],
    threshold_cube: np.ndarray,
) -> None:
    preflight_path = QA_ROOT / "preflight_summary.json"
    if not preflight_path.exists():
        raise RuntimeError("状态充分统计全量运行缺少预检记录")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or preflight.get("script_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("状态充分统计预检身份或状态失配")
    started = time.perf_counter()
    maps = {
        key: {value: index for index, value in enumerate(vocabularies[key])}
        for key in ("zones", "predictors", "horizon_groups", "seeds")
    }
    n_states = state_count(vocabularies)
    n_actions = len(vocabularies["actions"])
    shape = (len(vocabularies["zones"]), len(vocabularies["zones"]), n_states)
    stream_ids = completed["stream_id"].astype(str).tolist()
    next_index, arrays, resources = load_or_initialize(
        stream_ids=stream_ids,
        shape=shape,
        n_actions=n_actions,
    )
    event_count, errf_sum, covered_sum, tuwr_sum, tuwr_count, best_count = arrays
    process = psutil.Process()
    peak_rss_gib = process.memory_info().rss / (1024.0**3)
    for ledger_index in range(next_index, len(completed)):
        row = next(completed.iloc[[ledger_index]].itertuples(index=False))
        frame = read_stream_frame(row, verify_hash=True)
        predictor_index = maps["predictors"][str(row.predictor)]
        horizon_group = str(frame["horizon_group"].iloc[0])
        horizon_group_index = maps["horizon_groups"][horizon_group]
        zone_index = maps["zones"][str(row.zone)]
        seed_index = maps["seeds"][int(row.seed)]
        base, coverage_code = encode_base_state(
            frame=frame,
            predictor_index=predictor_index,
            horizon_group_index=horizon_group_index,
            vocabularies=vocabularies,
        )
        errf, covered, tuwr_zero, tuwr_finite, best = extract_metrics(
            frame=frame,
            actions=vocabularies["actions"],
        )
        widths = frame["raw_width_value"].to_numpy(dtype=np.float64)
        for outer_index in range(len(vocabularies["zones"])):
            if outer_index == zone_index:
                continue
            q33 = threshold_cube[
                outer_index,
                predictor_index,
                horizon_group_index,
                coverage_code,
                seed_index,
                0,
            ]
            q67 = threshold_cube[
                outer_index,
                predictor_index,
                horizon_group_index,
                coverage_code,
                seed_index,
                1,
            ]
            width_code = np.where(widths <= q33, 0, np.where(widths <= q67, 1, 2)).astype(np.int64)
            codes = base * len(vocabularies["width_states"]) + width_code
            local = aggregate_one_width_state(
                state_code=codes,
                errf=errf,
                covered=covered,
                tuwr_zero_filled=tuwr_zero,
                tuwr_finite=tuwr_finite,
                best_action=best,
                n_states=n_states,
                n_actions=n_actions,
            )
            event_count[outer_index, zone_index] += local[0]
            errf_sum[outer_index, zone_index] += local[1]
            covered_sum[outer_index, zone_index] += local[2]
            tuwr_sum[outer_index, zone_index] += local[3]
            tuwr_count[outer_index, zone_index] += local[4]
            best_count[outer_index, zone_index] += local[5]
        completed_count = ledger_index + 1
        rss_gib = process.memory_info().rss / (1024.0**3)
        peak_rss_gib = max(peak_rss_gib, rss_gib)
        if completed_count % CHECKPOINT_INTERVAL == 0 or completed_count == len(completed):
            resources.append(
                {
                    "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
                    "completed_streams": completed_count,
                    "total_streams": len(completed),
                    "rss_gib": rss_gib,
                    "free_disk_gib": shutil.disk_usage(REVISION_ROOT).free / (1024.0**3),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            save_checkpoint(
                next_index=completed_count,
                stream_ids=stream_ids,
                arrays=arrays,
                resource_rows=resources,
            )
        if completed_count % 120 == 0:
            print(
                json.dumps(
                    {
                        "stage": "build_outer_fold_state_statistics",
                        "completed": completed_count,
                        "total": len(completed),
                        "rss_gib": rss_gib,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if int(event_count.sum()) != EXPECTED_ALL_FOLD_SOURCE_EVENT_COUNT:
        raise RuntimeError("状态充分统计全折源事件总数不闭合")
    expected_by_zone = completed.groupby("zone")["complete_case_event_count"].sum().astype(int).to_dict()
    for outer_index, heldout_zone in enumerate(vocabularies["zones"]):
        expected = EXPECTED_COMPLETE_EVENT_COUNT - int(expected_by_zone[heldout_zone])
        if int(event_count[outer_index].sum()) != expected:
            raise RuntimeError(f"外层折源事件总数不闭合: {heldout_zone}")
        if np.any(event_count[outer_index, outer_index] != 0):
            raise RuntimeError(f"外层折持出区进入状态充分统计: {heldout_zone}")
    if np.any(~np.isfinite(errf_sum)) or np.any(errf_sum < 0.0):
        raise RuntimeError("状态充分统计ERRF汇总非法")
    if np.any(covered_sum < 0.0) or np.any(covered_sum > event_count[..., None]):
        raise RuntimeError("状态充分统计covered汇总非法")
    if np.any(tuwr_sum < 0.0) or np.any(tuwr_sum > tuwr_count):
        raise RuntimeError("状态充分统计TUWR汇总非法")
    if np.any(tuwr_count < 0.0) or np.any(tuwr_count > event_count[..., None]):
        raise RuntimeError("状态充分统计TUWR计数非法")
    if not np.array_equal(best_count.sum(axis=-1), event_count):
        raise RuntimeError("状态充分统计最优动作标签计数不闭合")
    output = build_output_frame(arrays=arrays, vocabularies=vocabularies)
    key_columns = ["outer_heldout_zone", "source_zone", "state_code"]
    if output[key_columns].duplicated().any():
        raise RuntimeError("状态充分统计输出键重复")
    if (output["outer_heldout_zone"] == output["source_zone"]).any():
        raise RuntimeError("状态充分统计输出含持出区记录")
    if int(output["event_count"].sum()) != EXPECTED_ALL_FOLD_SOURCE_EVENT_COUNT:
        raise RuntimeError("状态充分统计长表事件总数不闭合")
    output_path = RUN_ROOT / "state_sufficient_statistics.parquet"
    output.to_parquet(output_path, index=False, compression="zstd")
    elapsed = time.perf_counter() - started
    summary = {
        "run_id": "S06_OUTER_FOLD_STATE_SUFFICIENT_STATISTICS_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "script_sha256": sha256_file(Path(__file__)),
        "code_commit": EXPECTED_CODE_COMMIT,
        "source_tuning_contract_sha256": SOURCE_TUNING_CONTRACT_SHA256,
        "source_fact_root_manifest_sha256": SOURCE_FACT_ROOT_MANIFEST_SHA256,
        "width_threshold_file_sha256": WIDTH_THRESHOLD_FILE_SHA256,
        "input_stream_count": len(completed),
        "input_complete_case_event_count": EXPECTED_COMPLETE_EVENT_COUNT,
        "outer_fold_count": len(vocabularies["zones"]),
        "source_zone_count_per_outer_fold": 9,
        "state_space_size": n_states,
        "output_row_count": len(output),
        "all_fold_source_event_count": int(output["event_count"].sum()),
        "heldout_zone_metric_use_count": 0,
        "input_hash_mismatch_count": 0,
        "failed_partition_count": 0,
        "preflight_summary_sha256": sha256_file(preflight_path),
        "output_file": str(output_path),
        "output_file_sha256": sha256_file(output_path),
        "checkpoint_file_sha256": sha256_file(checkpoint_paths()[0]),
        "resource_samples_file": str(RUN_ROOT / "logs" / "state_statistics_resource_samples.csv"),
        "elapsed_seconds": elapsed,
        "peak_rss_gib": peak_rss_gib,
        "free_disk_gib_after": shutil.disk_usage(REVISION_ROOT).free / (1024.0**3),
        "heldout_performance_read": False,
        "performance_comparison_executed": False,
        "method_ranking_executed": False,
        "statistical_inference_executed": False,
        "paper_performance_claim_executed": False,
        "new_scientific_adverse_evidence": False,
        "stop_triggered": False,
    }
    summary_path = RUN_ROOT / "state_sufficient_statistics_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def record_exception(error: Exception) -> None:
    path = RUN_ROOT / "logs" / "state_statistics_exception_ledger.csv"
    row = pd.DataFrame(
        [
            {
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "script_sha256": sha256_file(Path(__file__)),
            }
        ]
    )
    if path.exists():
        existing = pd.read_csv(path, keep_default_na=False)
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    completed, _ = require_identity()
    vocabularies = build_vocabularies()
    threshold_cube = load_threshold_cube(vocabularies)
    if args.preflight_only:
        run_preflight(
            completed=completed,
            vocabularies=vocabularies,
            threshold_cube=threshold_cube,
        )
        return
    run_full(
        completed=completed,
        vocabularies=vocabularies,
        threshold_cube=threshold_cube,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        record_exception(error)
        raise
