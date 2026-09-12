from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pandas as pd
import psutil


S06_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = S06_ROOT.parent
AUTH_ROOT = REVISION_ROOT / "_权威代码"
CODE_ROOT = AUTH_ROOT / "code"
S03_ROOT = (
    REVISION_ROOT
    / "03_基础预测与候选区间重建"
    / "results_raw"
    / "full_rebuild_v1"
)
RUN_ROOT = S06_ROOT / "results_raw" / "nested_source_selection_v1"
SELECTION_ROOT = RUN_ROOT / "nested_selection" / "conformal_v1"
STREAM_ROOT = SELECTION_ROOT / "stream_checkpoints"
LOG_ROOT = RUN_ROOT / "logs"
sys.path.insert(0, str(CODE_ROOT))

from baseline_common import build_nested_source_zone_folds  # noqa: E402
from baseline_conformal import conformal_grid  # noqa: E402
from baseline_conformal_compact import (  # noqa: E402
    build_feedback_schedule,
    evaluate_conformal_grid_compact,
    summarize_conformal_grid_compact,
)
from baseline_selection import select_configuration  # noqa: E402
from clara_errf import ErrfTheta  # noqa: E402
from clara_event_contract import load_frozen_contracts, sha256_file  # noqa: E402


EXPECTED_CODE_COMMIT = "8465112a55d7fa20631bae8e9eefe5608078bd9f"
SOURCE_TUNING_CONTRACT_SHA256 = "9a352cba41d7e68911626da950540d1d8506f2f4f63afb72bd7b28cd473c7e14"
CONFORMAL_CONTRACT_SHA256 = "61e160ebeeb84cf71f1aeb4601d0394675afc354ae74a00d7300261fa4569496"
SOURCE_FACT_ROOT_MANIFEST_SHA256 = "30a0f86d6f775c7c32a3654117807dfa9acabc8187c2bdefeb8440e1466ef9e7"
S03_BASE_LEDGER_SHA256 = "adcfb12975abf487a06dfee93fb3ea597ecfcac3a081ebdfe392552c28dbd4b8"
S03_ROOT_MANIFEST_SHA256 = "5063d53444d4270e208ec95aac94ec414cebc5eae722e44e60674e9c1318b96e"
COMPACT_EQUIVALENCE_QA_SHA256 = "a1591ae530c501199ebfb89afc5fbe178eace032091da91e27a5e86ff6ef5ab7"
COMPACT_ENGINE_SHA256 = "04744ca200d8902b944396e9f481c5d94595a2ddfcbba550e4b7932123c493b4"
FULL_STREAM_BENCHMARK_SHA256 = "7eb6ebf5c1cebd3b9f37f4353599502b75ee6818535a2cc2cb7dec4b5f7378cf"
COVERAGES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
PREDICTORS = ("GBR", "MLP", "QRLSTM", "Ridge")
SEEDS = (0, 1, 2)
HORIZONS = tuple(range(1, 25))
EXPECTED_STREAM_COUNT = 2880
EXPECTED_CONFIGURATION_COUNT = 19
EXPECTED_STREAM_METRIC_ROWS = EXPECTED_CONFIGURATION_COUNT
EXPECTED_ZONE_SCORE_ROWS = 190
EXPECTED_INNER_FOLD_COUNT = 90
EXPECTED_VALIDATION_SCORE_ROWS = 1710
EXPECTED_SELECTION_ROWS = 10
EXPECTED_GLOBAL_CALIBRATION_ROWS = 10064160
EXPECTED_GLOBAL_COMPLETE_TEST_ROWS = 10064160
EXPECTED_GLOBAL_FACT_ROWS = 110705760
EXPECTED_GLOBAL_CONFORMAL_CANDIDATE_ROWS = 2103409440
DEFAULT_WORKER_COUNT = 12
MAXIMUM_WORKER_COUNT = 12
MINIMUM_FREE_DISK_GIB = 60.0


def expected_calibration_rows(horizon: int) -> int:
    return 3507 - int(horizon)


def expected_test_rows(horizon: int) -> int:
    return 3508 - int(horizon)


def expected_complete_test_rows(horizon: int) -> int:
    return 3507 - int(horizon)


def expected_base_rows(horizon: int) -> int:
    return expected_calibration_rows(horizon) + expected_test_rows(horizon)


def expected_fact_rows(horizon: int) -> int:
    return expected_complete_test_rows(horizon) * len(COVERAGES)


class InputHashMismatchError(RuntimeError):
    pass


class CausalContractError(RuntimeError):
    pass


class ProtocolIdentityError(RuntimeError):
    pass


_WORKER_CONTRACTS: Any | None = None
_WORKER_GRID: tuple[Any, ...] | None = None
_WORKER_THETA: ErrfTheta | None = None


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def zone_number(value: str) -> int:
    text = str(value)
    if text.startswith("zone") and text[4:].isdigit():
        return int(text[4:])
    raise ValueError(f"区域编号无法解析: {text}")


def quantile_names(coverage: float) -> tuple[str, str]:
    alpha = 1.0 - float(coverage)
    return f"q_{alpha / 2.0:.4f}", f"q_{1.0 - alpha / 2.0:.4f}"


def datetime_ns(series: pd.Series) -> np.ndarray:
    return (
        pd.to_datetime(series, errors="raise")
        .astype("datetime64[ns]")
        .astype("int64")
        .to_numpy(dtype=np.int64)
    )


def array_digest(arrays: tuple[np.ndarray, ...]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(contiguous.shape).encode("ascii"))
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def canonical_configuration_json(configuration: dict[str, Any]) -> str:
    return json.dumps(
        configuration,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stream_paths(record: dict[str, Any]) -> dict[str, Path]:
    predictor = str(record["predictor"])
    zone = str(record["zone"])
    horizon = int(record["horizon"])
    seed = int(record["seed"])
    stream_id = str(record["stream_id"])
    base_root = S03_ROOT / "base" / predictor / stream_id
    facts_root = (
        RUN_ROOT
        / "source_facts"
        / f"predictor={predictor}"
        / f"zone={zone}"
        / f"horizon={horizon:02d}"
        / f"seed={seed}"
    )
    checkpoint_root = (
        STREAM_ROOT
        / f"predictor={predictor}"
        / f"zone={zone}"
        / f"horizon={horizon:02d}"
        / f"seed={seed}"
    )
    return {
        "base": base_root / "base_predictions.parquet",
        "base_manifest": base_root / "manifest.json",
        "facts": facts_root / "facts.parquet",
        "facts_manifest": facts_root / "manifest.json",
        "checkpoint_metrics": checkpoint_root / "stream_metrics.parquet",
        "checkpoint_manifest": checkpoint_root / "manifest.json",
    }


def validate_input_identity(record: dict[str, Any]) -> dict[str, Any]:
    paths = stream_paths(record)
    horizon = int(record["horizon"])
    calibration_rows = expected_calibration_rows(horizon)
    test_rows = expected_test_rows(horizon)
    base_rows = expected_base_rows(horizon)
    fact_rows = expected_fact_rows(horizon)
    for role in ("base", "base_manifest", "facts", "facts_manifest"):
        if not paths[role].is_file():
            raise InputHashMismatchError(f"{record['stream_id']}缺少输入文件: {role}")
    base_manifest_sha256 = sha256_file(paths["base_manifest"])
    if base_manifest_sha256 != str(record["manifest_sha256"]):
        raise InputHashMismatchError(f"{record['stream_id']}基础清单哈希失配")
    base_manifest = load_json(paths["base_manifest"])
    if (
        base_manifest.get("status") != "COMPLETE_VALIDATED"
        or base_manifest.get("stream_id", record["stream_id"]) != record["stream_id"]
        or str(base_manifest.get("predictor")) != str(record["predictor"])
        or str(base_manifest.get("zone")) != str(record["zone"])
        or int(base_manifest.get("horizon")) != int(record["horizon"])
        or int(base_manifest.get("seed")) != int(record["seed"])
        or int(base_manifest.get("n_output")) != base_rows
        or int(base_manifest.get("n_calibration")) != calibration_rows
        or int(base_manifest.get("n_test")) != test_rows
    ):
        raise ProtocolIdentityError(f"{record['stream_id']}基础清单身份失配")
    base_sha256 = sha256_file(paths["base"])
    if base_sha256 != str(record["output_sha256"]) or base_sha256 != base_manifest.get("output_sha256"):
        raise InputHashMismatchError(f"{record['stream_id']}基础预测哈希失配")

    facts_manifest_sha256 = sha256_file(paths["facts_manifest"])
    facts_manifest = load_json(paths["facts_manifest"])
    if (
        facts_manifest.get("status") != "COMPLETE_VALIDATED"
        or facts_manifest.get("stream_id") != record["stream_id"]
        or facts_manifest.get("source_tuning_contract_sha256") != SOURCE_TUNING_CONTRACT_SHA256
        or int(facts_manifest.get("complete_case_event_count", -1)) != fact_rows
        or int(facts_manifest.get("terminal_censored_event_count", -1)) != len(COVERAGES)
        or int(facts_manifest.get("future_information_violation_count", -1)) != 0
        or int(facts_manifest.get("within_issue_feedback_use_count", -1)) != 0
    ):
        raise ProtocolIdentityError(f"{record['stream_id']}源事实清单身份失配")
    facts_sha256 = sha256_file(paths["facts"])
    if facts_sha256 != facts_manifest.get("facts_sha256"):
        raise InputHashMismatchError(f"{record['stream_id']}源事实文件哈希失配")
    return {
        "paths": paths,
        "base_sha256": base_sha256,
        "base_manifest_sha256": base_manifest_sha256,
        "facts_sha256": facts_sha256,
        "facts_manifest_sha256": facts_manifest_sha256,
        "facts_content_sha256": str(facts_manifest.get("fact_content_sha256")),
    }


def full_history_matrix(facts: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    output = np.zeros((len(COVERAGES), len(test)), dtype=np.uint8)
    test_issues = pd.to_datetime(test["issue_timestamp"], errors="raise").reset_index(drop=True)
    for coverage_index, coverage in enumerate(COVERAGES):
        frame = facts[
            np.isclose(facts["target_coverage"].astype(float), coverage, rtol=0.0, atol=1e-12)
        ].copy()
        frame["issue_timestamp"] = pd.to_datetime(frame["issue_timestamp"], errors="raise")
        frame = frame.sort_values("issue_timestamp", kind="mergesort").reset_index(drop=True)
        if len(frame) != len(test) or not frame["issue_timestamp"].equals(test_issues):
            raise ProtocolIdentityError(f"覆盖率{coverage}的源事实与完整测试事件无法对齐")
        output[coverage_index] = frame["rolling_state"].astype(str).ne("cold_start").to_numpy(dtype=np.uint8)
    return output


def validate_feedback_causality(
    *,
    issue_ns: np.ndarray,
    label_ns: np.ndarray,
    available_ns: np.ndarray,
    feedback_order: np.ndarray,
    matured_ends: np.ndarray,
) -> None:
    pointer = 0
    for event_index, matured_end_value in enumerate(matured_ends):
        matured_end = int(matured_end_value)
        if matured_end < pointer or matured_end > len(feedback_order):
            raise CausalContractError("紧凑conformal成熟反馈指针越界")
        sources = feedback_order[pointer:matured_end]
        if len(sources) and (
            np.any(available_ns[sources] > issue_ns[event_index])
            or np.any(label_ns[sources] >= issue_ns[event_index])
        ):
            raise CausalContractError("紧凑conformal检测到未来信息或同批标签使用")
        pointer = matured_end


def evaluate_stream(record: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if _WORKER_CONTRACTS is None or _WORKER_GRID is None or _WORKER_THETA is None:
        raise RuntimeError("紧凑conformal工作进程尚未初始化")
    started = time.perf_counter()
    process = psutil.Process()
    horizon = int(record["horizon"])
    calibration_rows = expected_calibration_rows(horizon)
    base_test_rows = expected_test_rows(horizon)
    complete_test_rows = expected_complete_test_rows(horizon)
    base_rows = expected_base_rows(horizon)
    fact_rows = expected_fact_rows(horizon)
    identity = validate_input_identity(record)
    paths = identity["paths"]
    base = pd.read_parquet(paths["base"])
    facts = pd.read_parquet(
        paths["facts"],
        columns=["target_coverage", "issue_timestamp", "rolling_state"],
    )
    if len(base) != base_rows or len(facts) != fact_rows:
        raise ProtocolIdentityError(f"{record['stream_id']}输入行数失配")
    calibration = base[base["split"].astype(str).eq("calibration")].sort_values(
        ["issue_timestamp", "label_timestamp"], kind="mergesort"
    ).reset_index(drop=True)
    base_test = base[base["split"].astype(str).eq("test")].copy()
    reference_facts = facts[
        np.isclose(facts["target_coverage"].astype(float), COVERAGES[0], rtol=0.0, atol=1e-12)
    ]
    complete_issues = set(pd.to_datetime(reference_facts["issue_timestamp"], errors="raise"))
    test = base_test[
        pd.to_datetime(base_test["issue_timestamp"], errors="raise").isin(complete_issues)
    ].sort_values(["issue_timestamp", "label_timestamp"], kind="mergesort").reset_index(drop=True)
    if (
        len(calibration) != calibration_rows
        or len(base_test) != base_test_rows
        or len(test) != complete_test_rows
        or int(base_test["label_available_timestamp"].isna().sum()) != 1
        or test["label_available_timestamp"].isna().any()
    ):
        raise ProtocolIdentityError(f"{record['stream_id']}完整案例门规模失配")
    calibration_available = pd.to_datetime(calibration["label_available_timestamp"], errors="raise")
    first_test_issue = pd.to_datetime(test["issue_timestamp"], errors="raise").min()
    if calibration_available.max() > first_test_issue:
        raise CausalContractError(f"{record['stream_id']}校准标签在测试起点后才可用")

    coverage_values = np.asarray(COVERAGES, dtype=np.float64)
    calibration_lower = np.vstack(
        [calibration[quantile_names(coverage)[0]].to_numpy(dtype=float) for coverage in COVERAGES]
    )
    calibration_upper = np.vstack(
        [calibration[quantile_names(coverage)[1]].to_numpy(dtype=float) for coverage in COVERAGES]
    )
    test_lower = np.vstack(
        [test[quantile_names(coverage)[0]].to_numpy(dtype=float) for coverage in COVERAGES]
    )
    test_upper = np.vstack(
        [test[quantile_names(coverage)[1]].to_numpy(dtype=float) for coverage in COVERAGES]
    )
    issue_ns = datetime_ns(test["issue_timestamp"])
    label_ns = datetime_ns(test["label_timestamp"])
    available_ns = datetime_ns(test["label_available_timestamp"])
    if np.any(np.diff(issue_ns) <= 0) or np.any(label_ns <= issue_ns) or np.any(available_ns <= label_ns):
        raise CausalContractError(f"{record['stream_id']}时间顺序失配")
    feedback_order, matured_ends, application_ns = build_feedback_schedule(
        issue_ns,
        label_ns,
        available_ns,
    )
    validate_feedback_causality(
        issue_ns=issue_ns,
        label_ns=label_ns,
        available_ns=available_ns,
        feedback_order=feedback_order,
        matured_ends=matured_ends,
    )
    full_history = full_history_matrix(facts, test)
    cadence_values = test["nominal_cadence_minutes"].astype(float).unique()
    if len(cadence_values) != 1 or not np.isclose(cadence_values[0], 60.0, rtol=0.0, atol=1e-12):
        raise ProtocolIdentityError(f"{record['stream_id']}标称步长失配")
    cadence_minutes = float(cadence_values[0])

    engine_started = time.perf_counter()
    compact = evaluate_conformal_grid_compact(
        contracts=_WORKER_CONTRACTS,
        coverages=coverage_values,
        calibration_target=calibration["target"].to_numpy(dtype=float),
        calibration_lower=calibration_lower,
        calibration_upper=calibration_upper,
        calibration_center=calibration["q_0.5000"].to_numpy(dtype=float),
        calibration_label_ns=datetime_ns(calibration["label_timestamp"]),
        test_target=test["target"].to_numpy(dtype=float),
        test_lower=test_lower,
        test_upper=test_upper,
        test_center=test["q_0.5000"].to_numpy(dtype=float),
        test_issue_ns=issue_ns,
        test_label_ns=label_ns,
        feedback_order=feedback_order,
        matured_ends=matured_ends,
        cadence_hours=cadence_minutes / 60.0,
    )
    engine_seconds = time.perf_counter() - engine_started
    expected_shape = (EXPECTED_CONFIGURATION_COUNT, len(COVERAGES), complete_test_rows)
    if any(array.shape != expected_shape for array in compact):
        raise RuntimeError(f"{record['stream_id']}紧凑候选形状失配")
    metric_started = time.perf_counter()
    metrics = summarize_conformal_grid_compact(
        compact[0],
        compact[1],
        test["target"].to_numpy(dtype=float),
        test["q_0.5000"].to_numpy(dtype=float),
        coverage_values,
        label_ns,
        application_ns,
        feedback_order,
        full_history,
        cadence_minutes,
        _WORKER_THETA.pi_plus,
        _WORKER_THETA.pi_minus,
        _WORKER_THETA.kappa_plus,
        _WORKER_THETA.kappa_minus,
    )
    metric_seconds = time.perf_counter() - metric_started
    if int(np.asarray(metrics[4], dtype=np.int64).sum()) != 0:
        raise CausalContractError(f"{record['stream_id']}可靠性完整历史闭合失败")
    candidate_digest_sha256 = array_digest(compact)
    feedback_digest_sha256 = array_digest(
        (
            feedback_order.astype(np.int32, copy=False),
            matured_ends.astype(np.int32, copy=False),
            application_ns.astype(np.int64, copy=False),
        )
    )
    event_count = complete_test_rows * len(COVERAGES)
    target_coverage_sum = complete_test_rows * float(np.sum(coverage_values))
    rows: list[dict[str, Any]] = []
    for configuration_index, configuration in enumerate(_WORKER_GRID):
        raw_lower = compact[2][configuration_index]
        raw_upper = compact[3][configuration_index]
        lower = compact[0][configuration_index]
        upper = compact[1][configuration_index]
        tuwr_count = int(np.asarray(metrics[3][configuration_index], dtype=np.int64).sum())
        rows.append(
            {
                "stream_id": str(record["stream_id"]),
                "predictor": str(record["predictor"]),
                "source_zone": str(record["zone"]),
                "horizon": int(record["horizon"]),
                "seed": int(record["seed"]),
                "baseline_id": "TunedSingleConformal",
                "configuration_id": configuration.configuration_id,
                "family": configuration.family,
                "configuration_json": canonical_configuration_json(configuration.configuration),
                "complexity_rank": int(configuration.complexity_rank),
                "errf_sum": float(np.asarray(metrics[0][configuration_index], dtype=float).sum()),
                "covered_sum": int(np.asarray(metrics[1][configuration_index], dtype=np.int64).sum()),
                "target_coverage_sum": target_coverage_sum,
                "tuwr_sum": float(np.asarray(metrics[2][configuration_index], dtype=float).sum()),
                "tuwr_count": tuwr_count,
                "event_count": event_count,
                "reliability_failure_count": int(
                    np.asarray(metrics[4][configuration_index], dtype=np.int64).sum()
                ),
                "clipped_lower_count": int(np.count_nonzero(raw_lower != lower)),
                "clipped_upper_count": int(np.count_nonzero(raw_upper != upper)),
                "clipped_candidate_count": int(
                    np.count_nonzero((raw_lower != lower) | (raw_upper != upper))
                ),
                "raw_crossing_count": int(np.count_nonzero(raw_lower > raw_upper)),
                "post_clip_crossing_count": int(np.count_nonzero(lower > upper)),
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["complexity_rank", "configuration_id"], kind="mergesort"
    ).reset_index(drop=True)
    numeric = frame[
        ["errf_sum", "covered_sum", "target_coverage_sum", "tuwr_sum", "tuwr_count", "event_count"]
    ].to_numpy(dtype=float)
    if len(frame) != EXPECTED_STREAM_METRIC_ROWS or not np.isfinite(numeric).all():
        raise RuntimeError(f"{record['stream_id']}逐流指标行数或数值失配")
    if frame["configuration_id"].nunique() != EXPECTED_CONFIGURATION_COUNT:
        raise RuntimeError(f"{record['stream_id']}逐流配置集合失配")
    if frame["post_clip_crossing_count"].sum() != 0 or frame["reliability_failure_count"].sum() != 0:
        raise RuntimeError(f"{record['stream_id']}逐流技术门失败")
    metadata = {
        **identity,
        "candidate_digest_sha256": candidate_digest_sha256,
        "feedback_digest_sha256": feedback_digest_sha256,
        "calibration_event_count": len(calibration),
        "base_test_event_count": len(base_test),
        "complete_test_event_count": len(test),
        "terminal_censored_event_count": len(base_test) - len(test),
        "candidate_record_count": int(compact[0].size),
        "feedback_record_count": len(feedback_order),
        "feedback_applied_by_final_issue_count": int(matured_ends[-1]),
        "feedback_final_group_count": len(feedback_order) - int(matured_ends[-1]),
        "full_history_flag_count": int(full_history.sum()),
        "engine_seconds": engine_seconds,
        "metric_seconds": metric_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "rss_gib": process.memory_info().rss / (1024.0**3),
        "numba_thread_count": numba.get_num_threads(),
    }
    metadata.pop("paths")
    return frame, metadata


def checkpoint_paths(record: dict[str, Any]) -> tuple[Path, Path]:
    paths = stream_paths(record)
    return paths["checkpoint_metrics"], paths["checkpoint_manifest"]


def verified_stream_checkpoint(
    record: dict[str, Any],
    *,
    verify_inputs: bool,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    metrics_path, manifest_path = checkpoint_paths(record)
    if not metrics_path.exists() and not manifest_path.exists():
        return None
    if not metrics_path.exists() or not manifest_path.exists():
        raise RuntimeError(f"{record['stream_id']}逐流检查点不完整")
    manifest = load_json(manifest_path)
    expected = {
        "checkpoint_schema": "S06_CONFORMAL_STREAM_METRICS_V1",
        "status": "PASS",
        "code_commit": EXPECTED_CODE_COMMIT,
        "source_tuning_contract_sha256": SOURCE_TUNING_CONTRACT_SHA256,
        "conformal_contract_sha256": CONFORMAL_CONTRACT_SHA256,
        "compact_equivalence_qa_sha256": COMPACT_EQUIVALENCE_QA_SHA256,
        "compact_engine_sha256": COMPACT_ENGINE_SHA256,
        "stream_id": str(record["stream_id"]),
        "predictor": str(record["predictor"]),
        "source_zone": str(record["zone"]),
        "horizon": int(record["horizon"]),
        "seed": int(record["seed"]),
        "metric_row_count": EXPECTED_STREAM_METRIC_ROWS,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"{record['stream_id']}逐流检查点身份失配")
    if sha256_file(metrics_path) != manifest.get("metrics_sha256"):
        raise RuntimeError(f"{record['stream_id']}逐流检查点指标哈希失配")
    if verify_inputs:
        identity = validate_input_identity(record)
        for key in (
            "base_sha256",
            "base_manifest_sha256",
            "facts_sha256",
            "facts_manifest_sha256",
            "facts_content_sha256",
        ):
            if manifest.get(key) != identity[key]:
                raise InputHashMismatchError(f"{record['stream_id']}恢复检查点输入身份失配: {key}")
    frame = pd.read_parquet(metrics_path)
    if (
        len(frame) != EXPECTED_STREAM_METRIC_ROWS
        or frame["configuration_id"].nunique() != EXPECTED_CONFIGURATION_COUNT
        or frame["post_clip_crossing_count"].sum() != 0
        or frame["reliability_failure_count"].sum() != 0
    ):
        raise RuntimeError(f"{record['stream_id']}逐流检查点内容失配")
    return frame, manifest


def write_stream_checkpoint(
    *,
    record: dict[str, Any],
    metrics: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    metrics_path, manifest_path = checkpoint_paths(record)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_metrics = metrics_path.with_name(f"stream_metrics.{os.getpid()}.tmp.parquet")
    metrics.to_parquet(temporary_metrics, index=False, compression="zstd")
    os.replace(temporary_metrics, metrics_path)
    manifest = {
        "checkpoint_schema": "S06_CONFORMAL_STREAM_METRICS_V1",
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": EXPECTED_CODE_COMMIT,
        "source_tuning_contract_sha256": SOURCE_TUNING_CONTRACT_SHA256,
        "conformal_contract_sha256": CONFORMAL_CONTRACT_SHA256,
        "source_fact_root_manifest_sha256": SOURCE_FACT_ROOT_MANIFEST_SHA256,
        "s03_base_ledger_sha256": S03_BASE_LEDGER_SHA256,
        "compact_equivalence_qa_sha256": COMPACT_EQUIVALENCE_QA_SHA256,
        "compact_engine_sha256": COMPACT_ENGINE_SHA256,
        "stream_id": str(record["stream_id"]),
        "predictor": str(record["predictor"]),
        "source_zone": str(record["zone"]),
        "horizon": int(record["horizon"]),
        "seed": int(record["seed"]),
        "metric_row_count": len(metrics),
        "metrics_sha256": sha256_file(metrics_path),
        **metadata,
        "future_information_violation_count": 0,
        "within_issue_feedback_use_count": 0,
        "heldout_performance_read": False,
        "configuration_selection_executed": False,
        "performance_comparison_executed": False,
        "method_ranking_executed": False,
        "statistical_inference_executed": False,
        "paper_performance_claim_executed": False,
    }
    temporary_manifest = manifest_path.with_name(f"manifest.{os.getpid()}.tmp.json")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return manifest


def initialize_worker() -> None:
    global _WORKER_CONTRACTS, _WORKER_GRID, _WORKER_THETA
    numba.set_num_threads(1)
    _WORKER_CONTRACTS = load_frozen_contracts()
    _WORKER_GRID = conformal_grid(_WORKER_CONTRACTS)
    _WORKER_THETA = ErrfTheta()


def run_stream(record: dict[str, Any]) -> dict[str, Any]:
    metrics, metadata = evaluate_stream(record)
    manifest = write_stream_checkpoint(record=record, metrics=metrics, metadata=metadata)
    return {
        "stream_id": str(record["stream_id"]),
        "status": "COMPUTED",
        "elapsed_seconds_this_invocation": float(metadata["elapsed_seconds"]),
        "engine_seconds": float(metadata["engine_seconds"]),
        "metric_seconds": float(metadata["metric_seconds"]),
        "rss_gib": float(metadata["rss_gib"]),
        "metrics_sha256": str(manifest["metrics_sha256"]),
    }


def record_exception(stage: str, error: BaseException, stream_id: str | None = None) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOG_ROOT / "conformal_selection_exception_ledger.csv"
    row = pd.DataFrame(
        [
            {
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "stage": stage,
                "stream_id": stream_id or "",
                "exception_type": type(error).__name__,
                "message": str(error),
                "protocol_drift": isinstance(error, ProtocolIdentityError),
                "input_hash_mismatch": isinstance(error, InputHashMismatchError),
                "causal_violation": isinstance(error, CausalContractError),
                "scientific_stop_trigger": False,
            }
        ]
    )
    if path.exists():
        previous = pd.read_csv(path, encoding="utf-8")
        row = pd.concat([previous, row], ignore_index=True)
    temporary = path.with_suffix(".tmp.csv")
    row.to_csv(temporary, index=False, encoding="utf-8")
    os.replace(temporary, path)


def write_progress(
    *,
    resources: list[dict[str, Any]],
    completed: int,
    total: int,
    started: float,
) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    resource_path = LOG_ROOT / "conformal_selection_resource_samples.csv"
    temporary_resource = resource_path.with_suffix(".tmp.csv")
    pd.DataFrame(resources).to_csv(temporary_resource, index=False, encoding="utf-8")
    os.replace(temporary_resource, resource_path)
    progress = {
        "run_id": "S06_CONFORMAL_NESTED_SELECTION_V1",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_stream_count": completed,
        "total_stream_count": total,
        "computed_stream_count": sum(row["status"] == "COMPUTED" for row in resources),
        "reused_stream_count": sum(row["status"] == "REUSED" for row in resources),
        "elapsed_seconds": time.perf_counter() - started,
        "free_disk_gib": shutil.disk_usage(REVISION_ROOT).free / (1024.0**3),
        "heldout_performance_read": False,
        "performance_comparison_executed": False,
    }
    progress_path = LOG_ROOT / "conformal_selection_progress.json"
    temporary_progress = progress_path.with_suffix(".tmp.json")
    temporary_progress.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_progress, progress_path)


def build_zone_scores(stream_metrics: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "source_zone",
        "baseline_id",
        "configuration_id",
        "family",
        "configuration_json",
        "complexity_rank",
    ]
    aggregate = (
        stream_metrics.groupby(group_columns, sort=True, as_index=False)
        .agg(
            errf_sum=("errf_sum", "sum"),
            covered_sum=("covered_sum", "sum"),
            target_coverage_sum=("target_coverage_sum", "sum"),
            tuwr_sum=("tuwr_sum", "sum"),
            tuwr_count=("tuwr_count", "sum"),
            event_count=("event_count", "sum"),
            stream_count=("stream_id", "nunique"),
            reliability_failure_count=("reliability_failure_count", "sum"),
            clipped_lower_count=("clipped_lower_count", "sum"),
            clipped_upper_count=("clipped_upper_count", "sum"),
            clipped_candidate_count=("clipped_candidate_count", "sum"),
            raw_crossing_count=("raw_crossing_count", "sum"),
            post_clip_crossing_count=("post_clip_crossing_count", "sum"),
        )
    )
    aggregate["mean_errf"] = aggregate["errf_sum"] / aggregate["event_count"]
    aggregate["mean_coverage_gap"] = (
        aggregate["covered_sum"] - aggregate["target_coverage_sum"]
    ) / aggregate["event_count"]
    aggregate["mean_tuwr"] = aggregate["tuwr_sum"] / aggregate["tuwr_count"]
    aggregate["zone_order"] = aggregate["source_zone"].map(zone_number)
    aggregate = aggregate.sort_values(
        ["zone_order", "complexity_rank", "configuration_id"], kind="mergesort"
    ).drop(columns="zone_order").reset_index(drop=True)
    numeric = aggregate[["mean_errf", "mean_coverage_gap", "mean_tuwr"]].to_numpy(dtype=float)
    if (
        len(aggregate) != EXPECTED_ZONE_SCORE_ROWS
        or not np.isfinite(numeric).all()
        or set(aggregate["stream_count"].astype(int)) != {288}
        or int(aggregate["reliability_failure_count"].sum()) != 0
        or int(aggregate["post_clip_crossing_count"].sum()) != 0
    ):
        raise RuntimeError("TunedSingleConformal源区聚合技术门失败")
    return aggregate


def inner_checkpoint_paths(outer_zone: str, validation_zone: str) -> tuple[Path, Path]:
    root = (
        SELECTION_ROOT
        / f"outer_heldout={outer_zone}"
        / f"inner_validation={validation_zone}"
    )
    return root / "validation_scores.parquet", root / "manifest.json"


def write_inner_checkpoint(
    *,
    scores: pd.DataFrame,
    outer_zone: str,
    validation_zone: str,
    training_zones: tuple[str, ...],
    zone_scores_sha256: str,
) -> None:
    score_path, manifest_path = inner_checkpoint_paths(outer_zone, validation_zone)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_score = score_path.with_name("validation_scores.tmp.parquet")
    scores.to_parquet(temporary_score, index=False, compression="zstd")
    os.replace(temporary_score, score_path)
    manifest = {
        "checkpoint_schema": "S06_CONFORMAL_INNER_SELECTION_V1",
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": EXPECTED_CODE_COMMIT,
        "source_tuning_contract_sha256": SOURCE_TUNING_CONTRACT_SHA256,
        "conformal_contract_sha256": CONFORMAL_CONTRACT_SHA256,
        "compact_equivalence_qa_sha256": COMPACT_EQUIVALENCE_QA_SHA256,
        "zone_configuration_scores_sha256": zone_scores_sha256,
        "outer_heldout_zone": outer_zone,
        "inner_validation_zone": validation_zone,
        "training_zones": list(training_zones),
        "score_row_count": len(scores),
        "score_file_sha256": sha256_file(score_path),
        "validation_calibration_rows_by_horizon_rule": "3507_minus_horizon",
        "validation_complete_test_rows_by_horizon_rule": "3507_minus_horizon",
        "inner_validation_test_fit_count": 0,
        "outer_heldout_zone_metric_read_count": 0,
        "outer_heldout_zone_fit_count": 0,
        "outer_heldout_zone_selection_count": 0,
        "configuration_selection_executed": False,
        "performance_comparison_executed": False,
        "method_ranking_executed": False,
        "statistical_inference_executed": False,
        "paper_performance_claim_executed": False,
    }
    temporary_manifest = manifest_path.with_suffix(".tmp.json")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)


def select_outer(
    *,
    contracts: Any,
    outer_zone: str,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    if outer_zone in set(scores["inner_validation_zone"].astype(str)):
        raise CausalContractError(f"{outer_zone}配置选择输入含外层持出区")
    selected = select_configuration(contracts=contracts, validation_scores=scores)
    matching = scores[
        scores["configuration_id"].astype(str).eq(selected.selected_configuration_id)
    ][["family", "configuration_json"]].drop_duplicates()
    if len(matching) != 1:
        raise RuntimeError("TunedSingleConformal选中配置正文不唯一")
    return pd.DataFrame(
        [
            {
                "outer_heldout_zone": outer_zone,
                "baseline_id": "TunedSingleConformal",
                **selected.as_record(),
                "selected_family": str(matching.iloc[0]["family"]),
                "selected_configuration_json": str(matching.iloc[0]["configuration_json"]),
                "selected_configuration_canonical_sha256": selected.selected_configuration_id,
                "source_validation_zone_count": 9,
                "outer_heldout_zone_metric_read_count": 0,
                "outer_heldout_zone_fit_count": 0,
                "outer_heldout_zone_selection_count": 0,
            }
        ]
    )


def validate_root_identities() -> None:
    if git_output("rev-parse", "HEAD") != EXPECTED_CODE_COMMIT or git_output("status", "--porcelain"):
        raise ProtocolIdentityError("TunedSingleConformal全量选择要求批准提交且权威代码工作树干净")
    identities = {
        S06_ROOT / "configs" / "source_tuning_contract_v1.json": SOURCE_TUNING_CONTRACT_SHA256,
        S06_ROOT / "configs" / "conformal_baseline_contract_v1.json": CONFORMAL_CONTRACT_SHA256,
        RUN_ROOT / "source_fact_root_manifest.json": SOURCE_FACT_ROOT_MANIFEST_SHA256,
        S03_ROOT / "base_stream_ledger.csv": S03_BASE_LEDGER_SHA256,
        S03_ROOT / "run_root_manifest.json": S03_ROOT_MANIFEST_SHA256,
        S06_ROOT / "qa" / "conformal_compact_equivalence_v1" / "qa_summary.json": COMPACT_EQUIVALENCE_QA_SHA256,
        CODE_ROOT / "baseline_conformal_compact.py": COMPACT_ENGINE_SHA256,
        S06_ROOT / "qa" / "conformal_compact_benchmark_v1" / "summary.json": FULL_STREAM_BENCHMARK_SHA256,
    }
    for path, expected in identities.items():
        observed = sha256_file(path)
        if observed != expected:
            raise InputHashMismatchError(f"TunedSingleConformal根输入哈希失配: {path.name}, {observed}")
    equivalence = load_json(
        S06_ROOT / "qa" / "conformal_compact_equivalence_v1" / "qa_summary.json"
    )
    benchmark = load_json(S06_ROOT / "qa" / "conformal_compact_benchmark_v1" / "summary.json")
    source_root = load_json(RUN_ROOT / "source_fact_root_manifest.json")
    if (
        equivalence.get("status") != "PASS"
        or int(equivalence.get("passed_count", -1)) != 209
        or benchmark.get("status") != "PASS"
        or int(benchmark.get("reliability_failure_count", -1)) != 0
        or source_root.get("status") != "COMPLETE_VALIDATED"
        or int(source_root.get("stream_count", -1)) != EXPECTED_STREAM_COUNT
    ):
        raise ProtocolIdentityError("TunedSingleConformal根输入状态失配")


def load_stream_records() -> list[dict[str, Any]]:
    ledger = pd.read_csv(S03_ROOT / "base_stream_ledger.csv", encoding="utf-8")
    ledger = ledger[
        ledger["status"].astype(str).eq("COMPLETED")
        & ledger["stage"].astype(str).eq("base")
    ].copy()
    ledger["zone_order"] = ledger["zone"].map(zone_number)
    ledger = ledger.sort_values(
        ["predictor", "zone_order", "horizon", "seed"], kind="mergesort"
    ).drop(columns="zone_order").reset_index(drop=True)
    if (
        len(ledger) != EXPECTED_STREAM_COUNT
        or ledger["stream_id"].nunique() != EXPECTED_STREAM_COUNT
        or set(ledger["predictor"].astype(str)) != set(PREDICTORS)
        or set(ledger["horizon"].astype(int)) != set(HORIZONS)
        or set(ledger["seed"].astype(int)) != set(SEEDS)
        or set(ledger["zone"].astype(str)) != {f"zone{index}" for index in range(1, 11)}
        or any(
            int(row.output_rows) != expected_base_rows(int(row.horizon))
            for row in ledger.itertuples(index=False)
        )
    ):
        raise ProtocolIdentityError("TunedSingleConformal基础流总账范围失配")
    records: list[dict[str, Any]] = []
    for row in ledger.to_dict(orient="records"):
        records.append(
            {
                "stream_id": str(row["stream_id"]),
                "predictor": str(row["predictor"]),
                "zone": str(row["zone"]),
                "horizon": int(row["horizon"]),
                "seed": int(row["seed"]),
                "output_sha256": str(row["output_sha256"]),
                "manifest_sha256": str(row["manifest_sha256"]),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKER_COUNT)
    args = parser.parse_args()
    if not 1 <= args.workers <= MAXIMUM_WORKER_COUNT:
        raise ValueError(f"工作进程数必须位于1至{MAXIMUM_WORKER_COUNT}")
    started = time.perf_counter()
    validate_root_identities()
    free_before = shutil.disk_usage(REVISION_ROOT).free / (1024.0**3)
    if free_before < MINIMUM_FREE_DISK_GIB:
        raise RuntimeError(f"TunedSingleConformal磁盘安全门失败: {free_before:.3f} GiB")
    records = load_stream_records()
    SELECTION_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    exception_path = LOG_ROOT / "conformal_selection_exception_ledger.csv"
    if not exception_path.exists():
        pd.DataFrame(
            columns=[
                "recorded_at_utc",
                "stage",
                "stream_id",
                "exception_type",
                "message",
                "protocol_drift",
                "input_hash_mismatch",
                "causal_violation",
                "scientific_stop_trigger",
            ]
        ).to_csv(exception_path, index=False, encoding="utf-8")

    resources: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for record in records:
        checkpoint = verified_stream_checkpoint(record, verify_inputs=True)
        if checkpoint is None:
            pending.append(record)
        else:
            _, manifest = checkpoint
            resources.append(
                {
                    "stream_id": record["stream_id"],
                    "status": "REUSED",
                    "elapsed_seconds_this_invocation": 0.0,
                    "engine_seconds": float(manifest["engine_seconds"]),
                    "metric_seconds": float(manifest["metric_seconds"]),
                    "rss_gib": float(manifest["rss_gib"]),
                    "metrics_sha256": str(manifest["metrics_sha256"]),
                }
            )
    completed = len(resources)
    write_progress(
        resources=resources,
        completed=completed,
        total=EXPECTED_STREAM_COUNT,
        started=started,
    )
    print(
        json.dumps(
            {
                "stage": "conformal_stream_plan",
                "worker_count": args.workers,
                "reused_stream_count": completed,
                "pending_stream_count": len(pending),
                "free_disk_gib": free_before,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if pending:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize_worker,
        ) as executor:
            future_by_stream = {
                executor.submit(run_stream, record): record
                for record in pending
            }
            for future in as_completed(future_by_stream):
                record = future_by_stream[future]
                try:
                    resource = future.result()
                except BaseException as error:
                    record_exception("conformal_stream_evaluation", error, str(record["stream_id"]))
                    for pending_future in future_by_stream:
                        pending_future.cancel()
                    raise
                resources.append(resource)
                completed += 1
                if completed % 25 == 0 or completed == EXPECTED_STREAM_COUNT:
                    free_current = shutil.disk_usage(REVISION_ROOT).free / (1024.0**3)
                    if free_current < MINIMUM_FREE_DISK_GIB:
                        raise RuntimeError(
                            f"TunedSingleConformal运行中磁盘安全门失败: {free_current:.3f} GiB"
                        )
                    write_progress(
                        resources=resources,
                        completed=completed,
                        total=EXPECTED_STREAM_COUNT,
                        started=started,
                    )
                    print(
                        json.dumps(
                            {
                                "stage": "conformal_stream_evaluation",
                                "completed_stream_count": completed,
                                "total_stream_count": EXPECTED_STREAM_COUNT,
                                "elapsed_seconds": time.perf_counter() - started,
                                "free_disk_gib": free_current,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    if completed != EXPECTED_STREAM_COUNT:
        raise RuntimeError(f"TunedSingleConformal逐流完成数失配: {completed}")

    stream_frames: list[pd.DataFrame] = []
    for record in records:
        checkpoint = verified_stream_checkpoint(record, verify_inputs=False)
        if checkpoint is None:
            raise RuntimeError(f"{record['stream_id']}聚合前缺少逐流检查点")
        stream_frames.append(checkpoint[0])
    stream_metrics = pd.concat(stream_frames, ignore_index=True)
    if (
        len(stream_metrics) != EXPECTED_STREAM_COUNT * EXPECTED_STREAM_METRIC_ROWS
        or stream_metrics["stream_id"].nunique() != EXPECTED_STREAM_COUNT
    ):
        raise RuntimeError("TunedSingleConformal根级逐流指标规模失配")
    stream_metrics_path = SELECTION_ROOT / "conformal_stream_metrics.parquet"
    stream_metrics.to_parquet(stream_metrics_path, index=False, compression="zstd")
    zone_scores = build_zone_scores(stream_metrics)
    zone_scores_path = SELECTION_ROOT / "zone_configuration_scores.parquet"
    zone_scores.to_parquet(zone_scores_path, index=False, compression="zstd")
    zone_scores_sha256 = sha256_file(zone_scores_path)

    contracts = load_frozen_contracts()
    zones = sorted(
        (str(zone) for zone in contracts.protocol["datasets"]["gefcom2014"]["zones"]),
        key=zone_number,
    )
    root_scores: list[pd.DataFrame] = []
    root_selections: list[pd.DataFrame] = []
    outer_manifests: list[dict[str, Any]] = []
    inner_fold_count = 0
    for outer_zone in zones:
        outer_scores: list[pd.DataFrame] = []
        folds = build_nested_source_zone_folds(
            contracts=contracts,
            outer_heldout_zone=outer_zone,
        )
        for fold in folds:
            validation_zone = str(fold.inner_validation_zone)
            training_zones = tuple(str(zone) for zone in fold.inner_training_zones)
            scores = zone_scores[
                zone_scores["source_zone"].astype(str).eq(validation_zone)
            ].copy()
            scores["outer_heldout_zone"] = outer_zone
            scores["inner_validation_zone"] = validation_zone
            scores["training_zones_json"] = json.dumps(list(training_zones), ensure_ascii=False)
            scores["source_score_reused_without_refit"] = True
            scores["inner_validation_test_fit"] = False
            scores["outer_heldout_zone_metric_read"] = False
            scores["outer_heldout_zone_fit"] = False
            scores["outer_heldout_zone_selection"] = False
            scores = scores.sort_values(
                ["complexity_rank", "configuration_id"], kind="mergesort"
            ).reset_index(drop=True)
            if len(scores) != EXPECTED_CONFIGURATION_COUNT:
                raise RuntimeError(f"{outer_zone}/{validation_zone}内层配置数失配")
            write_inner_checkpoint(
                scores=scores,
                outer_zone=outer_zone,
                validation_zone=validation_zone,
                training_zones=training_zones,
                zone_scores_sha256=zone_scores_sha256,
            )
            outer_scores.append(scores)
            inner_fold_count += 1
        all_outer_scores = pd.concat(outer_scores, ignore_index=True)
        selections = select_outer(
            contracts=contracts,
            outer_zone=outer_zone,
            scores=all_outer_scores,
        )
        outer_root = SELECTION_ROOT / f"outer_heldout={outer_zone}"
        all_scores_path = outer_root / "all_validation_scores.parquet"
        selections_path = outer_root / "selected_configurations.parquet"
        all_outer_scores.to_parquet(all_scores_path, index=False, compression="zstd")
        selections.to_parquet(selections_path, index=False, compression="zstd")
        outer_manifest = {
            "manifest_schema": "S06_CONFORMAL_OUTER_SELECTION_V1",
            "status": "PASS",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": EXPECTED_CODE_COMMIT,
            "source_tuning_contract_sha256": SOURCE_TUNING_CONTRACT_SHA256,
            "conformal_contract_sha256": CONFORMAL_CONTRACT_SHA256,
            "outer_heldout_zone": outer_zone,
            "inner_fold_count": len(folds),
            "score_row_count": len(all_outer_scores),
            "selection_row_count": len(selections),
            "scores_sha256": sha256_file(all_scores_path),
            "selections_sha256": sha256_file(selections_path),
            "fallback_selection_count": int(selections["no_feasible_fallback_used"].sum()),
            "outer_heldout_zone_metric_read_count": 0,
            "outer_heldout_zone_fit_count": 0,
            "outer_heldout_zone_selection_count": 0,
            "configuration_selection_executed": True,
            "performance_comparison_executed": False,
            "method_ranking_executed": False,
            "statistical_inference_executed": False,
            "paper_performance_claim_executed": False,
        }
        (outer_root / "manifest.json").write_text(
            json.dumps(outer_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        outer_manifests.append(outer_manifest)
        root_scores.append(all_outer_scores)
        root_selections.append(selections)

    all_scores = pd.concat(root_scores, ignore_index=True)
    all_selections = pd.concat(root_selections, ignore_index=True)
    if (
        inner_fold_count != EXPECTED_INNER_FOLD_COUNT
        or len(all_scores) != EXPECTED_VALIDATION_SCORE_ROWS
        or len(all_selections) != EXPECTED_SELECTION_ROWS
        or all_scores["outer_heldout_zone_metric_read"].any()
        or all_scores["outer_heldout_zone_fit"].any()
        or all_scores["outer_heldout_zone_selection"].any()
    ):
        raise RuntimeError("TunedSingleConformal根级嵌套选择技术门失败")
    validation_scores_path = SELECTION_ROOT / "conformal_validation_scores.parquet"
    selections_path = SELECTION_ROOT / "conformal_selected_configurations.parquet"
    all_scores.to_parquet(validation_scores_path, index=False, compression="zstd")
    all_selections.to_parquet(selections_path, index=False, compression="zstd")
    resources.sort(key=lambda row: str(row["stream_id"]))
    write_progress(
        resources=resources,
        completed=completed,
        total=EXPECTED_STREAM_COUNT,
        started=started,
    )
    resource_path = LOG_ROOT / "conformal_selection_resource_samples.csv"
    free_after = shutil.disk_usage(REVISION_ROOT).free / (1024.0**3)
    if free_after < MINIMUM_FREE_DISK_GIB:
        raise RuntimeError(f"TunedSingleConformal结束磁盘安全门失败: {free_after:.3f} GiB")
    exceptions = pd.read_csv(exception_path, encoding="utf-8")
    fallback_count = int(all_selections["no_feasible_fallback_used"].sum())
    summary = {
        "run_id": "S06_CONFORMAL_NESTED_SELECTION_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "code_commit": EXPECTED_CODE_COMMIT,
        "source_tuning_contract_sha256": SOURCE_TUNING_CONTRACT_SHA256,
        "conformal_contract_sha256": CONFORMAL_CONTRACT_SHA256,
        "source_fact_root_manifest_sha256": SOURCE_FACT_ROOT_MANIFEST_SHA256,
        "s03_base_ledger_sha256": S03_BASE_LEDGER_SHA256,
        "s03_root_manifest_sha256": S03_ROOT_MANIFEST_SHA256,
        "compact_equivalence_qa_sha256": COMPACT_EQUIVALENCE_QA_SHA256,
        "compact_engine_sha256": COMPACT_ENGINE_SHA256,
        "full_stream_benchmark_sha256": FULL_STREAM_BENCHMARK_SHA256,
        "worker_count": args.workers,
        "numba_threads_per_worker": 1,
        "stream_count": EXPECTED_STREAM_COUNT,
        "computed_stream_count": sum(row["status"] == "COMPUTED" for row in resources),
        "reused_stream_count": sum(row["status"] == "REUSED" for row in resources),
        "configuration_count": EXPECTED_CONFIGURATION_COUNT,
        "coverage_count": len(COVERAGES),
        "calibration_rows_by_horizon_rule": "3507_minus_horizon",
        "complete_test_rows_by_horizon_rule": "3507_minus_horizon",
        "global_calibration_event_count": EXPECTED_GLOBAL_CALIBRATION_ROWS,
        "global_complete_test_event_count": EXPECTED_GLOBAL_COMPLETE_TEST_ROWS,
        "global_complete_case_coverage_event_count": EXPECTED_GLOBAL_FACT_ROWS,
        "global_conformal_candidate_record_count": EXPECTED_GLOBAL_CONFORMAL_CANDIDATE_ROWS,
        "stream_metric_row_count": len(stream_metrics),
        "zone_configuration_score_row_count": len(zone_scores),
        "outer_fold_count": len(zones),
        "inner_fold_count": inner_fold_count,
        "validation_score_row_count": len(all_scores),
        "selected_configuration_row_count": len(all_selections),
        "no_feasible_fallback_count": fallback_count,
        "reliability_failure_count": int(stream_metrics["reliability_failure_count"].sum()),
        "raw_crossing_count": int(stream_metrics["raw_crossing_count"].sum()),
        "post_clip_crossing_count": int(stream_metrics["post_clip_crossing_count"].sum()),
        "clipped_candidate_count": int(stream_metrics["clipped_candidate_count"].sum()),
        "exception_ledger_row_count": len(exceptions),
        "outer_heldout_zone_metric_read_count": 0,
        "outer_heldout_zone_fit_count": 0,
        "outer_heldout_zone_selection_count": 0,
        "stream_metrics_sha256": sha256_file(stream_metrics_path),
        "zone_configuration_scores_sha256": zone_scores_sha256,
        "validation_scores_sha256": sha256_file(validation_scores_path),
        "selected_configurations_sha256": sha256_file(selections_path),
        "resource_samples_sha256": sha256_file(resource_path),
        "exception_ledger_sha256": sha256_file(exception_path),
        "elapsed_seconds": time.perf_counter() - started,
        "free_disk_gib_before": free_before,
        "free_disk_gib_after": free_after,
        "configuration_selection_executed": True,
        "heldout_performance_read": False,
        "performance_comparison_executed": False,
        "method_ranking_executed": False,
        "statistical_inference_executed": False,
        "paper_performance_claim_executed": False,
        "new_scientific_adverse_evidence": False,
        "stop_triggered": False,
    }
    summary_path = SELECTION_ROOT / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        record_exception("conformal_nested_selection", error)
        raise
