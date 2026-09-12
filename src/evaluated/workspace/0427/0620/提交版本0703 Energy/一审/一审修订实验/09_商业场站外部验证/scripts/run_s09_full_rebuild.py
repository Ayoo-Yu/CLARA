"""重建 S09 的 96 条外部基础预测流与规范化候选包。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# 每个工作进程限制为单线程，避免四进程运行时发生线程过度订阅。
for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(variable, "1")


S09_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = Path(__file__).resolve().parents[2]
AUTHORITATIVE_ROOT = REVISION_ROOT / "_权威代码"
AUTHORITATIVE_CODE = AUTHORITATIVE_ROOT / "code"
sys.path.insert(0, str(AUTHORITATIVE_CODE))

from common import (  # noqa: E402
    canonical_json_sha256,
    git_commit_for_path,
    git_status_porcelain,
)
from run_base_predictor import run_base_prediction  # noqa: E402
from run_candidate_bundle import (  # noqa: E402
    run_candidate_bundle,
    validate_candidate_bundle_cache,
)


MASTER_CONFIG = S09_ROOT / "configs" / "s09_external_validation_v1.json"
RUN_ROOT = S09_ROOT / "results_raw" / "full_external_rebuild_v1"
HISTORICAL_ROOT = S09_ROOT / "results_raw" / "historical"
QA_SCRIPT = S09_ROOT / "qa" / "qa_s09_full_rebuild.py"


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


def build_tasks(master: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for farm in master["scope"]["farms"]:
        farm_info = master["data"]["farms"][farm]
        for predictor in master["scope"]["predictors"]:
            for seed in master["scope"]["seeds"]:
                for horizon in master["scope"]["horizon_steps"]:
                    tasks.append(
                        {
                            "farm": str(farm),
                            "dataset_id": str(farm_info["dataset_id"]),
                            "data_path": str(farm_info["processed_path"]),
                            "data_sha256": str(farm_info["processed_sha256"]),
                            "predictor": str(predictor),
                            "seed": int(seed),
                            "horizon": int(horizon),
                        }
                    )
    tasks.sort(
        key=lambda row: (
            row["farm"],
            row["predictor"],
            row["seed"],
            row["horizon"],
        )
    )
    return tasks


def base_run_dir(task: dict[str, Any]) -> Path:
    return RUN_ROOT / "base" / task["predictor"] / stream_id(task)


def bundle_run_dir(task: dict[str, Any]) -> Path:
    return RUN_ROOT / "bundles" / task["predictor"] / stream_id(task)


def base_config(task: dict[str, Any], master: dict[str, Any]) -> dict[str, Any]:
    protocol = master["protocol"]
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol["protocol_sha256"],
        "dataset_id": task["dataset_id"],
        "predictor": task["predictor"],
        "zone": task["farm"],
        "seed": int(task["seed"]),
        "horizon": int(task["horizon"]),
        "data_path": task["data_path"],
        "split": {"train_ratio": 0.6, "calibration_ratio": 0.2},
        "save_splits": ["calibration", "test"],
        "require_clean_code": True,
        "results_dir": str(base_run_dir(task)),
    }


def bundle_config(task: dict[str, Any], master: dict[str, Any]) -> dict[str, Any]:
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
        "base_predictions_path": str(base_run_dir(task) / "base_predictions.parquet"),
        "target_coverages": list(master["scope"]["target_coverages"]),
        "methods": ["SplitCF", "ACI", "AgACI", "EnbPI"],
        "compute_performance_metrics": False,
        "require_clean_code": True,
        "bundle_id": stream_id(task),
        "results_dir": str(bundle_run_dir(task)),
    }


def validate_base_cache(
    task: dict[str, Any], master: dict[str, Any]
) -> tuple[bool, str]:
    run_dir = base_run_dir(task)
    manifest_path = run_dir / "manifest.json"
    output_path = run_dir / "base_predictions.parquet"
    if not manifest_path.exists() or not output_path.exists():
        return False, "cache_missing"
    try:
        manifest = load_json(manifest_path)
        cfg = base_config(task, master)
        expected = {
            "status": "COMPLETE_VALIDATED",
            "protocol_sha256": master["protocol"]["protocol_sha256"],
            "code_commit": master["protocol"]["authoritative_code_commit"],
            "config_sha256": canonical_json_sha256(cfg),
            "input_sha256": task["data_sha256"],
            "output_sha256": sha256_file(output_path),
        }
        mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
        if mismatches:
            return False, "cache_identity_mismatch:" + ",".join(mismatches)
        return True, "cache_valid"
    except Exception as exc:
        return False, f"cache_validation_error:{type(exc).__name__}:{exc}"


def quarantine_partial(run_dir: Path, stage: str, identifier: str) -> str | None:
    if not run_dir.exists() or not any(run_dir.iterdir()):
        return None
    run_resolved = run_dir.resolve()
    root_resolved = RUN_ROOT.resolve()
    if root_resolved not in run_resolved.parents:
        raise RuntimeError(f"拒绝移动范围外目录: {run_resolved}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = HISTORICAL_ROOT / f"full_external_rebuild_v1_{stage}_{identifier}_{stamp}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(target))
    return str(target)


def run_base_task(payload: dict[str, Any]) -> dict[str, Any]:
    task = payload["task"]
    master = payload["master"]
    started = time.perf_counter()
    valid, reason = validate_base_cache(task, master)
    if valid:
        manifest = load_json(base_run_dir(task) / "manifest.json")
        return {
            "stage": "base",
            "status": "CACHED",
            "stream_id": stream_id(task),
            "farm": task["farm"],
            "predictor": task["predictor"],
            "seed": task["seed"],
            "horizon": task["horizon"],
            "elapsed_seconds": time.perf_counter() - started,
            "output_rows": int(manifest["n_output"]),
            "output_bytes": int((base_run_dir(task) / "base_predictions.parquet").stat().st_size),
            "output_sha256": manifest["output_sha256"],
            "cache_reason": reason,
        }
    if reason.startswith("cache_identity_mismatch"):
        raise RuntimeError(f"{stream_id(task)} 基础缓存身份失配: {reason}")
    quarantined = quarantine_partial(base_run_dir(task), "base", stream_id(task))
    cfg = base_config(task, master)
    base_run_dir(task).mkdir(parents=True, exist_ok=True)
    save_json_atomic(cfg, base_run_dir(task) / "run_config.json")
    manifest = run_base_prediction(cfg)
    return {
        "stage": "base",
        "status": "COMPLETED",
        "stream_id": stream_id(task),
        "farm": task["farm"],
        "predictor": task["predictor"],
        "seed": task["seed"],
        "horizon": task["horizon"],
        "elapsed_seconds": time.perf_counter() - started,
        "output_rows": int(manifest["n_output"]),
        "output_bytes": int((base_run_dir(task) / "base_predictions.parquet").stat().st_size),
        "output_sha256": manifest["output_sha256"],
        "quarantined_partial_dir": quarantined,
    }


def run_bundle_task(payload: dict[str, Any]) -> dict[str, Any]:
    task = payload["task"]
    master = payload["master"]
    started = time.perf_counter()
    cfg = bundle_config(task, master)
    mismatches = validate_candidate_bundle_cache(
        bundle_run_dir(task) / "manifest.json", cfg
    )
    if not mismatches:
        manifest = load_json(bundle_run_dir(task) / "manifest.json")
        return {
            "stage": "bundle",
            "status": "CACHED",
            "stream_id": stream_id(task),
            "farm": task["farm"],
            "predictor": task["predictor"],
            "seed": task["seed"],
            "horizon": task["horizon"],
            "elapsed_seconds": time.perf_counter() - started,
            "event_rows": int(manifest["event_rows"]),
            "candidate_rows": int(manifest["candidate_rows"]),
            "feedback_rows": int(manifest["feedback_rows"]),
            "output_bytes": sum(
                Path(manifest[key]).stat().st_size
                for key in ("event_core_file", "candidate_file", "feedback_file")
            ),
            "cache_reason": "cache_valid",
        }
    if (bundle_run_dir(task) / "manifest.json").exists():
        raise RuntimeError(
            f"{stream_id(task)} 候选缓存身份失配: {','.join(mismatches)}"
        )
    quarantined = quarantine_partial(bundle_run_dir(task), "bundle", stream_id(task))
    manifest = run_candidate_bundle(cfg)
    return {
        "stage": "bundle",
        "status": "COMPLETED",
        "stream_id": stream_id(task),
        "farm": task["farm"],
        "predictor": task["predictor"],
        "seed": task["seed"],
        "horizon": task["horizon"],
        "elapsed_seconds": time.perf_counter() - started,
        "event_rows": int(manifest["event_rows"]),
        "candidate_rows": int(manifest["candidate_rows"]),
        "feedback_rows": int(manifest["feedback_rows"]),
        "output_bytes": sum(
            Path(manifest[key]).stat().st_size
            for key in ("event_core_file", "candidate_file", "feedback_file")
        ),
        "quarantined_partial_dir": quarantined,
    }


def validate_preflight(master: dict[str, Any], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    if len(tasks) != int(master["scope"]["expected_base_streams"]):
        raise RuntimeError("S09 全量流数量失配")
    if git_status_porcelain(AUTHORITATIVE_ROOT):
        raise RuntimeError("权威代码工作树不干净")
    code_commit = git_commit_for_path(AUTHORITATIVE_ROOT)
    if code_commit != master["protocol"]["authoritative_code_commit"]:
        raise RuntimeError("权威代码提交失配")
    for task in tasks:
        path = Path(task["data_path"])
        if not path.exists() or sha256_file(path) != task["data_sha256"]:
            raise RuntimeError(f"外部输入缺失或哈希失配: {path}")
    free_gib = shutil.disk_usage(S09_ROOT).free / (1024**3)
    minimum = float(master["execution"]["minimum_free_disk_gib"])
    projection = float(master["execution"]["projected_full_rebuild_gib"])
    if free_gib - projection < minimum:
        raise RuntimeError(
            f"S09 磁盘资源门失败: free={free_gib:.3f}, "
            f"projection={projection:.3f}, minimum={minimum:.3f} GiB"
        )
    return {
        "status": "PASS",
        "checked_at_utc": utc_now(),
        "stream_count": len(tasks),
        "code_commit": code_commit,
        "free_disk_gib": free_gib,
        "projected_full_rebuild_gib": projection,
        "free_after_projection_gib": free_gib - projection,
    }


def write_provenance_sidecar(master: dict[str, Any]) -> None:
    weather = master["data"]["weather_provenance_sidecar"]
    payload = {
        "schema": "S09_EXTERNAL_FULL_REBUILD_PROVENANCE_V1",
        "status": "BOUND_TO_INPUTS",
        "created_at_utc": utc_now(),
        "farms": master["data"]["farms"],
        "weather_provenance_sidecar": weather,
        "nwp_provider": "ECMWF",
        "forecast_cycle": "18:00 UTC daily",
        "timestamp_timezone": "Asia/Shanghai",
        "availability_pairing_author_confirmed": True,
        "fifteen_minute_interpolation": "linear",
        "row_level_forecast_reference_time": None,
        "row_level_actual_availability_time": None,
        "unknown_row_level_fields_are_not_reconstructed": True,
    }
    save_json_atomic(payload, RUN_ROOT / "external_provenance_sidecar.json")


def write_checkpoint(
    phase: str,
    total: int,
    base_completed: int,
    bundle_completed: int,
    status: str = "RUNNING",
) -> None:
    save_json_atomic(
        {
            "schema": "S09_FULL_REBUILD_CHECKPOINT_V1",
            "status": status,
            "phase": phase,
            "total_streams": total,
            "base_completed": base_completed,
            "bundle_completed": bundle_completed,
            "updated_at_utc": utc_now(),
        },
        RUN_ROOT / "checkpoint.json",
    )


def run_phase(
    phase: str,
    tasks: list[dict[str, Any]],
    master: dict[str, Any],
    workers: int,
    base_completed: int,
    bundle_completed: int,
) -> list[dict[str, Any]]:
    worker = run_base_task if phase == "base" else run_bundle_task
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(worker, {"task": task, "master": master}): task
            for task in tasks
        }
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                record = future.result()
            except Exception as exc:
                for pending in future_map:
                    pending.cancel()
                append_jsonl(
                    {
                        "created_at_utc": utc_now(),
                        "stage": phase,
                        "stream_id": stream_id(task),
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    RUN_ROOT / "exception_ledger.jsonl",
                )
                raise
            records.append(record)
            if phase == "base":
                base_completed += 1
            else:
                bundle_completed += 1
            write_checkpoint(
                phase,
                len(tasks),
                base_completed,
                bundle_completed,
            )
            print(
                f"[{phase}] {len(records)}/{len(tasks)} {record['stream_id']} "
                f"{record['status']} {record['elapsed_seconds']:.3f}s",
                flush=True,
            )
    records.sort(key=lambda row: row["stream_id"])
    return records


def run_qa() -> None:
    completed = subprocess.run(
        [sys.executable, str(QA_SCRIPT)],
        cwd=str(S09_ROOT),
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    log = S09_ROOT / "runtime_logs" / "full_external_rebuild_v1_qa.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"S09 全量 QA 失败，详见 {log}")
    print(completed.stdout.strip(), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    master = load_json(MASTER_CONFIG)
    tasks = build_tasks(master)
    preflight = validate_preflight(master, tasks)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    save_json_atomic(preflight, RUN_ROOT / "preflight.json")
    write_provenance_sidecar(master)
    save_json_atomic(master, RUN_ROOT / "effective_run_config.json")
    if args.dry_run:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return

    workers = int(args.workers or master["execution"]["maximum_workers"])
    if workers < 1 or workers > int(master["execution"]["maximum_workers"]):
        raise ValueError("工作进程数超出冻结上限")
    started = time.perf_counter()
    write_checkpoint("base", len(tasks), 0, 0)
    base_records = run_phase("base", tasks, master, workers, 0, 0)
    pd.DataFrame(base_records).to_csv(
        RUN_ROOT / "base_stream_ledger.csv", index=False, encoding="utf-8-sig"
    )
    write_checkpoint("bundle", len(tasks), len(tasks), 0)
    bundle_records = run_phase(
        "bundle", tasks, master, workers, len(tasks), 0
    )
    pd.DataFrame(bundle_records).to_csv(
        RUN_ROOT / "bundle_stream_ledger.csv", index=False, encoding="utf-8-sig"
    )

    manifest = {
        "schema": "S09_FULL_EXTERNAL_REBUILD_ROOT_MANIFEST_V1",
        "status": "COMPUTE_COMPLETE_PENDING_QA",
        "completed_at_utc": utc_now(),
        "config_sha256": sha256_file(MASTER_CONFIG),
        "protocol_sha256": master["protocol"]["protocol_sha256"],
        "code_commit": master["protocol"]["authoritative_code_commit"],
        "base_streams": len(base_records),
        "candidate_bundles": len(bundle_records),
        "base_output_rows": int(sum(row["output_rows"] for row in base_records)),
        "event_rows": int(sum(row["event_rows"] for row in bundle_records)),
        "candidate_rows": int(sum(row["candidate_rows"] for row in bundle_records)),
        "feedback_rows": int(sum(row["feedback_rows"] for row in bundle_records)),
        "output_bytes": int(
            sum(row["output_bytes"] for row in base_records)
            + sum(row["output_bytes"] for row in bundle_records)
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "performance_comparison_performed": False,
    }
    save_json_atomic(manifest, RUN_ROOT / "root_manifest.json")
    run_qa()
    manifest["status"] = "COMPLETE_VALIDATED"
    manifest["qa_summary_sha256"] = sha256_file(
        S09_ROOT / "qa" / "full_external_rebuild_v1" / "qa_summary.json"
    )
    save_json_atomic(manifest, RUN_ROOT / "root_manifest.json")
    write_checkpoint("complete", len(tasks), len(tasks), len(tasks), "COMPLETE_VALIDATED")
    print("S09 外部基础预测与候选包全量重建完成", flush=True)


if __name__ == "__main__":
    main()

