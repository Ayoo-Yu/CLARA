"""从既有外部基础预测中重放适配期的四个冻结候选动作。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# 每个工作进程限制为单线程，避免并行运行时发生线程过度订阅。
for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import numpy as np
import pandas as pd


S09_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = Path(__file__).resolve().parents[2]
AUTHORITATIVE_CODE = REVISION_ROOT / "_权威代码" / "code"
sys.path.insert(0, str(AUTHORITATIVE_CODE))

from common import canonical_json_sha256, git_commit_for_path, git_status_porcelain  # noqa: E402
from run_candidate_bundle import run_candidate_bundle  # noqa: E402


MASTER_CONFIG = S09_ROOT / "configs" / "s09_external_validation_v1.json"
METHOD_CONFIG = S09_ROOT / "configs" / "s09_method_transfer_v1.json"
SOURCE_ROOT = S09_ROOT / "results_raw" / "full_external_rebuild_v1"
HISTORICAL_ROOT = S09_ROOT / "results_raw" / "historical"
MODE_ROOTS = {
    "pilot": S09_ROOT
    / "results_raw"
    / "pilot_adaptation_candidate_farma_ridge_h01_s0_v1",
    "full": S09_ROOT / "results_raw" / "adaptation_candidate_rebuild_v1",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_atomic(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def stream_id(task: dict[str, Any]) -> str:
    return (
        f"{task['predictor']}-H{int(task['horizon']):02d}-"
        f"{task['farm']}-S{int(task['seed'])}"
    )


def build_tasks(master: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for farm in master["scope"]["farms"]:
        farm_info = master["data"]["farms"][farm]
        for predictor in master["scope"]["predictors"]:
            for seed in master["scope"]["seeds"]:
                for horizon in master["scope"]["horizon_steps"]:
                    task = {
                        "farm": str(farm),
                        "dataset_id": str(farm_info["dataset_id"]),
                        "predictor": str(predictor),
                        "seed": int(seed),
                        "horizon": int(horizon),
                    }
                    if mode == "pilot" and task != {
                        "farm": "FarmA",
                        "dataset_id": "commercial_farma",
                        "predictor": "Ridge",
                        "seed": 0,
                        "horizon": 4,
                    }:
                        continue
                    tasks.append(task)
    tasks.sort(
        key=lambda row: (
            row["farm"],
            row["predictor"],
            row["seed"],
            row["horizon"],
        )
    )
    return tasks


def source_base_dir(task: dict[str, Any]) -> Path:
    return SOURCE_ROOT / "base" / task["predictor"] / stream_id(task)


def derived_base_dir(run_root: Path, task: dict[str, Any]) -> Path:
    return run_root / "base" / task["predictor"] / stream_id(task)


def bundle_dir(run_root: Path, task: dict[str, Any]) -> Path:
    return run_root / "bundles" / task["predictor"] / stream_id(task)


def quarantine_partial(
    path: Path,
    run_root: Path,
    mode: str,
    stage: str,
    identifier: str,
) -> str | None:
    if not path.exists() or not any(path.iterdir()):
        return None
    resolved = path.resolve()
    root_resolved = run_root.resolve()
    if root_resolved not in resolved.parents:
        raise RuntimeError(f"拒绝移动运行范围外目录: {resolved}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = HISTORICAL_ROOT / (
        f"s09_{mode}_adaptation_candidate_{stage}_{identifier}_{stamp}"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    return str(target)


def validate_source_base(
    task: dict[str, Any],
    master: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    source_dir = source_base_dir(task)
    base_path = source_dir / "base_predictions.parquet"
    manifest_path = source_dir / "manifest.json"
    if not base_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"适配期重放缺少源基础预测: {stream_id(task)}")
    manifest = load_json(manifest_path)
    protocol = master["protocol"]
    expected = {
        "status": "COMPLETE_VALIDATED",
        "protocol_sha256": protocol["protocol_sha256"],
        "code_commit": protocol["authoritative_code_commit"],
        "predictor": task["predictor"],
        "dataset_id": task["dataset_id"],
        "zone": task["farm"],
        "seed": int(task["seed"]),
        "horizon": int(task["horizon"]),
        "output_sha256": sha256_file(base_path),
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise RuntimeError(
            f"源基础预测身份核验失败 {stream_id(task)}: {sorted(mismatches)}"
        )
    return base_path, manifest


def derivation_config(
    task: dict[str, Any],
    source_manifest: dict[str, Any],
    method_config_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "S09_ADAPTATION_BASE_DERIVATION_CONFIG_V1",
        "stream_id": stream_id(task),
        "farm": task["farm"],
        "dataset_id": task["dataset_id"],
        "predictor": task["predictor"],
        "seed": int(task["seed"]),
        "horizon": int(task["horizon"]),
        "source_base_predictions_sha256": source_manifest["output_sha256"],
        "method_transfer_config_sha256": method_config_sha256,
        "source_split": "calibration",
        "midpoint_rule": "floor_half_by_issue_timestamp_order",
        "warmup_retention_rule": (
            "label_available_timestamp_strictly_before_first_selector_fit_issue"
        ),
        "selector_fit_retention_rule": (
            "label_available_timestamp_strictly_before_first_final_test_issue"
        ),
        "base_predictor_retraining": False,
        "candidate_definition_refitting": False,
    }


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def derive_adaptation_base(
    task: dict[str, Any],
    master: dict[str, Any],
    method_config_sha256: str,
    run_root: Path,
) -> dict[str, Any]:
    base_path, source_manifest = validate_source_base(task, master)
    output_dir = derived_base_dir(run_root, task)
    output_path = output_dir / "base_predictions.parquet"
    manifest_path = output_dir / "manifest.json"
    cfg = derivation_config(task, source_manifest, method_config_sha256)

    if output_path.exists() and manifest_path.exists():
        cached = load_json(manifest_path)
        expected = {
            "status": "COMPLETE_VALIDATED",
            "config_sha256": canonical_json_sha256(cfg),
            "source_base_predictions_sha256": source_manifest["output_sha256"],
            "method_transfer_config_sha256": method_config_sha256,
            "output_sha256": sha256_file(output_path),
        }
        mismatches = [
            key for key, value in expected.items() if cached.get(key) != value
        ]
        if not mismatches:
            return cached
        quarantine_partial(
            output_dir,
            run_root,
            "resume",
            "base",
            stream_id(task),
        )
    elif output_dir.exists() and any(output_dir.iterdir()):
        quarantine_partial(
            output_dir,
            run_root,
            "resume",
            "base",
            stream_id(task),
        )

    source = pd.read_parquet(base_path)
    for column in (
        "timestamp",
        "issue_timestamp",
        "label_timestamp",
        "label_available_timestamp",
    ):
        source[column] = pd.to_datetime(source[column], errors="coerce")
    calibration = (
        source[source["split"].astype(str) == "calibration"]
        .copy()
        .sort_values("issue_timestamp")
        .reset_index(drop=True)
    )
    final_test = (
        source[source["split"].astype(str) == "test"]
        .copy()
        .sort_values("issue_timestamp")
        .reset_index(drop=True)
    )
    if len(calibration) < 4 or final_test.empty:
        raise RuntimeError(f"适配期或最终测试期为空: {stream_id(task)}")

    midpoint = len(calibration) // 2
    selector_start = pd.Timestamp(calibration.iloc[midpoint]["issue_timestamp"])
    final_test_start = pd.Timestamp(final_test.iloc[0]["issue_timestamp"])
    warmup_source = calibration.iloc[:midpoint].copy()
    selector_source = calibration.iloc[midpoint:].copy()
    warmup = warmup_source[
        warmup_source["label_available_timestamp"].notna()
        & (warmup_source["label_available_timestamp"] < selector_start)
    ].copy()
    selector_fit = selector_source[
        selector_source["label_available_timestamp"].notna()
        & (selector_source["label_available_timestamp"] < final_test_start)
    ].copy()
    if warmup.empty or selector_fit.empty:
        raise RuntimeError(f"适配期内部因果切分为空: {stream_id(task)}")
    warmup["split"] = "calibration"
    selector_fit["split"] = "test"
    derived = (
        pd.concat([warmup, selector_fit], ignore_index=True)
        .sort_values(["split", "issue_timestamp"])
        .reset_index(drop=True)
    )

    quantile_columns = [column for column in derived.columns if column.startswith("q_")]
    quantile_matrix = derived[quantile_columns].to_numpy(dtype=float)
    validations = {
        "source_rows_are_original_calibration_only": True,
        "output_issue_timestamp_unique": bool(derived["issue_timestamp"].is_unique),
        "warmup_labels_mature_before_selector_fit": bool(
            (warmup["label_available_timestamp"] < selector_start).all()
        ),
        "selector_labels_mature_before_final_test": bool(
            (selector_fit["label_available_timestamp"] < final_test_start).all()
        ),
        "final_test_rows_used": 0,
        "quantiles_finite": bool(np.isfinite(quantile_matrix).all()),
        "quantiles_ordered": bool((np.diff(quantile_matrix, axis=1) >= 0.0).all()),
        "performance_comparison_performed": False,
    }
    if not all(
        bool(value)
        for key, value in validations.items()
        if key not in {"final_test_rows_used", "performance_comparison_performed"}
    ):
        raise AssertionError(f"适配期基础预测质量检查失败: {stream_id(task)}")

    write_parquet_atomic(derived, output_path)
    protocol = master["protocol"]
    manifest = {
        "manifest_schema": "S09_DERIVED_ADAPTATION_BASE_V1",
        "status": "COMPLETE_VALIDATED",
        "protocol_id": protocol["protocol_id"],
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol["protocol_sha256"],
        "code_commit": protocol["authoritative_code_commit"],
        "code_worktree_clean": True,
        "config_sha256": canonical_json_sha256(cfg),
        "input_sha256": source_manifest["output_sha256"],
        "output_sha256": sha256_file(output_path),
        "output_file": str(output_path),
        "output_format": "parquet_zstd",
        "predictor": task["predictor"],
        "dataset_id": task["dataset_id"],
        "zone": task["farm"],
        "seed": int(task["seed"]),
        "horizon": int(task["horizon"]),
        "n_source_calibration": int(len(calibration)),
        "n_warmup_source": int(len(warmup_source)),
        "n_warmup_retained": int(len(warmup)),
        "n_selector_fit_source": int(len(selector_source)),
        "n_selector_fit_retained": int(len(selector_fit)),
        "n_output": int(len(derived)),
        "saved_splits": ["calibration", "test"],
        "source_base_manifest_sha256": sha256_file(
            source_base_dir(task) / "manifest.json"
        ),
        "source_base_predictions_sha256": source_manifest["output_sha256"],
        "method_transfer_config_sha256": method_config_sha256,
        "selector_fit_first_issue_timestamp": selector_start.isoformat(),
        "final_test_first_issue_timestamp": final_test_start.isoformat(),
        "validations": validations,
    }
    save_json_atomic(manifest, manifest_path)
    return manifest


def candidate_config(
    task: dict[str, Any],
    master: dict[str, Any],
    run_root: Path,
) -> dict[str, Any]:
    protocol = master["protocol"]
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol["protocol_sha256"],
        "baseline_registry_sha256": protocol["baseline_registry_sha256"],
        "output_contract_sha256": protocol["output_contract_sha256"],
        "dataset_id": task["dataset_id"],
        "predictor": task["predictor"],
        "zone": task["farm"],
        "seed": int(task["seed"]),
        "horizon": int(task["horizon"]),
        "base_predictions_path": str(
            derived_base_dir(run_root, task) / "base_predictions.parquet"
        ),
        "target_coverages": list(master["scope"]["target_coverages"]),
        "methods": ["SplitCF", "ACI", "AgACI", "EnbPI"],
        "compute_performance_metrics": False,
        "require_clean_code": True,
        "bundle_id": f"adaptation-{stream_id(task)}",
        "results_dir": str(bundle_dir(run_root, task)),
    }


def validate_bundle_cache(
    task: dict[str, Any],
    master: dict[str, Any],
    run_root: Path,
    derived_manifest: dict[str, Any],
) -> tuple[bool, str]:
    output_dir = bundle_dir(run_root, task)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return False, "cache_missing"
    required = (
        output_dir / "event_core.parquet",
        output_dir / "candidate_intervals.parquet",
        output_dir / "feedback_trace.parquet",
    )
    if not all(path.exists() for path in required):
        return False, "cache_missing_artifact"
    manifest = load_json(manifest_path)
    cfg = candidate_config(task, master, run_root)
    expected = {
        "status": "COMPLETE_VALIDATED",
        "config_sha256": canonical_json_sha256(cfg),
        "base_predictions_sha256": derived_manifest["output_sha256"],
        "base_manifest_sha256": sha256_file(
            derived_base_dir(run_root, task) / "manifest.json"
        ),
        "event_core_sha256": sha256_file(required[0]),
        "candidate_intervals_sha256": sha256_file(required[1]),
        "feedback_trace_sha256": sha256_file(required[2]),
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        return False, "cache_identity_mismatch:" + ",".join(mismatches)
    return True, "cache_valid"


def run_task(payload: dict[str, Any]) -> dict[str, Any]:
    task = payload["task"]
    master = payload["master"]
    method_config_sha256 = payload["method_config_sha256"]
    run_root = Path(payload["run_root"])
    mode = payload["mode"]
    started = time.perf_counter()
    derived_manifest = derive_adaptation_base(
        task,
        master,
        method_config_sha256,
        run_root,
    )
    valid, reason = validate_bundle_cache(
        task,
        master,
        run_root,
        derived_manifest,
    )
    if valid:
        bundle_manifest = load_json(bundle_dir(run_root, task) / "manifest.json")
        status = "CACHED"
        quarantined = None
    else:
        quarantined = quarantine_partial(
            bundle_dir(run_root, task),
            run_root,
            mode,
            "bundle",
            stream_id(task),
        )
        output_dir = bundle_dir(run_root, task)
        output_dir.mkdir(parents=True, exist_ok=True)
        bundle_manifest = run_candidate_bundle(
            candidate_config(task, master, run_root)
        )
        status = "COMPLETE"
    return {
        "status": status,
        "stream_id": stream_id(task),
        "farm": task["farm"],
        "predictor": task["predictor"],
        "seed": int(task["seed"]),
        "horizon": int(task["horizon"]),
        "base_rows": int(derived_manifest["n_output"]),
        "warmup_rows": int(derived_manifest["n_warmup_retained"]),
        "selector_fit_rows": int(derived_manifest["n_selector_fit_retained"]),
        "event_rows": int(bundle_manifest["event_rows"]),
        "candidate_rows": int(bundle_manifest["candidate_rows"]),
        "feedback_rows": int(bundle_manifest["feedback_rows"]),
        "elapsed_seconds": time.perf_counter() - started,
        "cache_reason": reason,
        "quarantined": quarantined,
    }


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODE_ROOTS), required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    mode = str(args.mode)
    workers = max(1, int(args.workers))
    run_root = MODE_ROOTS[mode]
    run_root.mkdir(parents=True, exist_ok=True)
    master = load_json(MASTER_CONFIG)
    method_config = load_json(METHOD_CONFIG)
    method_config_sha256 = sha256_file(METHOD_CONFIG)
    tasks = build_tasks(master, mode)
    protocol = master["protocol"]

    if sha256_file(MASTER_CONFIG) != method_config["parent_external_config_sha256"]:
        raise RuntimeError("迁移协议绑定的 S09 主配置哈希失配")
    if method_config.get("performance_read_allowed") is not False:
        raise RuntimeError("迁移协议未保持性能读取关闭")
    repository_root = REVISION_ROOT / "_权威代码"
    if git_commit_for_path(repository_root) != protocol["authoritative_code_commit"]:
        raise RuntimeError("权威代码提交失配")
    if git_status_porcelain(repository_root):
        raise RuntimeError("权威代码工作树不干净")
    free_gib = shutil.disk_usage(S09_ROOT).free / (1024**3)
    if free_gib < float(master["execution"]["minimum_free_disk_gib"]):
        raise RuntimeError(f"磁盘安全门失败，当前可用 {free_gib:.3f} GiB")

    checkpoint_path = run_root / "checkpoint.json"
    result_ledger = run_root / "task_ledger.jsonl"
    exception_ledger = run_root / "exception_ledger.jsonl"
    started_utc = utc_now()
    started = time.perf_counter()
    checkpoint = {
        "schema": "S09_ADAPTATION_CANDIDATE_REPLAY_CHECKPOINT_V1",
        "status": "RUNNING",
        "mode": mode,
        "started_at_utc": started_utc,
        "updated_at_utc": started_utc,
        "method_transfer_config_sha256": method_config_sha256,
        "total_tasks": len(tasks),
        "completed_tasks": 0,
        "failed_tasks": 0,
        "performance_comparison_performed": False,
    }
    save_json_atomic(checkpoint, checkpoint_path)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    payloads = [
        {
            "task": task,
            "master": master,
            "method_config_sha256": method_config_sha256,
            "run_root": str(run_root),
            "mode": mode,
        }
        for task in tasks
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(run_task, payload): payload["task"] for payload in payloads
        }
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                result = future.result()
                results.append(result)
                append_jsonl(result, result_ledger)
            except Exception as exc:
                failure = {
                    "created_at_utc": utc_now(),
                    "stream_id": stream_id(task),
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                append_jsonl(failure, exception_ledger)
            checkpoint.update(
                {
                    "updated_at_utc": utc_now(),
                    "completed_tasks": len(results),
                    "failed_tasks": len(failures),
                }
            )
            save_json_atomic(checkpoint, checkpoint_path)

    if failures:
        checkpoint["status"] = "FAILED"
        checkpoint["updated_at_utc"] = utc_now()
        save_json_atomic(checkpoint, checkpoint_path)
        raise RuntimeError(
            f"适配期候选重放失败 {len(failures)} 条流，详见异常账本"
        )

    results.sort(key=lambda row: row["stream_id"])
    root_manifest = {
        "schema": "S09_ADAPTATION_CANDIDATE_REPLAY_ROOT_V1",
        "status": "COMPUTE_COMPLETE_PENDING_QA",
        "mode": mode,
        "started_at_utc": started_utc,
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "protocol_sha256": protocol["protocol_sha256"],
        "code_commit": protocol["authoritative_code_commit"],
        "master_config_sha256": sha256_file(MASTER_CONFIG),
        "method_transfer_config_sha256": method_config_sha256,
        "runner_sha256": sha256_file(Path(__file__)),
        "task_count": len(results),
        "base_rows": int(sum(row["base_rows"] for row in results)),
        "warmup_rows": int(sum(row["warmup_rows"] for row in results)),
        "selector_fit_rows": int(sum(row["selector_fit_rows"] for row in results)),
        "event_rows": int(sum(row["event_rows"] for row in results)),
        "candidate_rows": int(sum(row["candidate_rows"] for row in results)),
        "feedback_rows": int(sum(row["feedback_rows"] for row in results)),
        "output_bytes": directory_bytes(run_root),
        "performance_comparison_performed": False,
        "tasks": results,
    }
    save_json_atomic(root_manifest, run_root / "root_manifest.json")
    checkpoint["status"] = "COMPUTE_COMPLETE_PENDING_QA"
    checkpoint["updated_at_utc"] = utc_now()
    save_json_atomic(checkpoint, checkpoint_path)
    print(
        f"S09 adaptation candidate replay complete: {len(results)} tasks, "
        f"{root_manifest['event_rows']} events, pending independent QA"
    )


if __name__ == "__main__":
    main()
