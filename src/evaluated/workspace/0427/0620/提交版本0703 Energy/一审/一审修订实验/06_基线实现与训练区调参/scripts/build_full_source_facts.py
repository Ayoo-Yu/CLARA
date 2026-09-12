from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import psutil


S06_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = S06_ROOT.parent
AUTH_ROOT = REVISION_ROOT / "_权威代码"
CODE_ROOT = AUTH_ROOT / "code"
S03_ROOT = REVISION_ROOT / "03_基础预测与候选区间重建" / "results_raw" / "full_rebuild_v1"
DEFAULT_OUTPUT_ROOT = S06_ROOT / "results_raw" / "nested_source_selection_v1"
sys.path.insert(0, str(CODE_ROOT))

from clara_event_contract import load_frozen_contracts, sha256_file  # noqa: E402
from source_tuning_facts import load_s03_source_tuning_facts  # noqa: E402


EXPECTED_CODE_COMMIT = "980fad5174840883318b15a225957f70df8928f6"
SOURCE_TUNING_CONTRACT_SHA256 = "9a352cba41d7e68911626da950540d1d8506f2f4f63afb72bd7b28cd473c7e14"
S03_LEDGER_SHA256 = "6a88612ac993bfb5a0bf83c910b81a7c4d05de67392389da76411e47ab44a427"
EXPECTED_STREAM_COUNT = 2880
EXPECTED_EVENT_COUNT = 110737440
EXPECTED_TERMINAL_EVENT_COUNT = 31680
EXPECTED_COMPLETE_EVENT_COUNT = 110705760
EXPECTED_ACTION_OBSERVATION_COUNT = 442823040
EXPECTED_TERMINAL_PER_STREAM = 11
EXPECTED_TERMINAL_CANDIDATE_PER_STREAM = 44
MINIMUM_FREE_DISK_GIB = 60.0
PREFLIGHT_FACT_BYTES = 7601137


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def natural_zone_number(value: str) -> int:
    text = str(value)
    if text.startswith("zone") and text[4:].isdigit():
        return int(text[4:])
    raise ValueError(f"无法解析区域编号: {text}")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


class PeakMemorySampler:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak_rss = self.process.memory_info().rss
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.wait(0.1):
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)

    def __enter__(self) -> "PeakMemorySampler":
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback_value) -> None:
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        self.stop_event.set()
        self.thread.join(timeout=2.0)


def output_paths(output_root: Path, task: dict[str, Any]) -> tuple[Path, Path]:
    directory = (
        output_root
        / "source_facts"
        / f"predictor={task['predictor']}"
        / f"zone={task['zone']}"
        / f"horizon={int(task['horizon']):02d}"
        / f"seed={int(task['seed'])}"
    )
    return directory / "facts.parquet", directory / "manifest.json"


def build_one(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        output_root = Path(task["output_root"])
        bundle_root = Path(task["bundle_root"])
        facts_path, output_manifest_path = output_paths(output_root, task)
        facts_path.parent.mkdir(parents=True, exist_ok=True)
        contracts = load_frozen_contracts()
        with PeakMemorySampler() as sampler:
            loaded = load_s03_source_tuning_facts(
                bundle_root,
                contracts=contracts,
                expected_manifest_sha256=str(task["input_manifest_sha256"]),
                verify_input_file_hashes=True,
            )
            expected_complete = int(loaded.manifest["event_rows"]) - EXPECTED_TERMINAL_PER_STREAM
            if loaded.audit["terminal_censored_event_count"] != EXPECTED_TERMINAL_PER_STREAM:
                raise RuntimeError("逐流末端右删失事件数失配")
            if loaded.audit["terminal_censored_candidate_count"] != EXPECTED_TERMINAL_CANDIDATE_PER_STREAM:
                raise RuntimeError("逐流末端右删失候选数失配")
            if len(loaded.facts) != expected_complete:
                raise RuntimeError("逐流完整案例事件数失配")
            temporary_facts = facts_path.with_name(facts_path.name + f".tmp.{os.getpid()}")
            loaded.facts.to_parquet(
                temporary_facts,
                index=False,
                compression="zstd",
            )
            temporary_facts.replace(facts_path)
            facts_sha256 = sha256_file(facts_path)
            output_manifest = {
                "manifest_schema": "S06_SOURCE_TUNING_FACT_PARTITION_V1",
                "status": "COMPLETE_VALIDATED",
                "created_at_utc": utc_now(),
                "code_commit": EXPECTED_CODE_COMMIT,
                "source_tuning_contract_sha256": SOURCE_TUNING_CONTRACT_SHA256,
                "stream_id": str(task["stream_id"]),
                "predictor": str(task["predictor"]),
                "zone": str(task["zone"]),
                "horizon": int(task["horizon"]),
                "seed": int(task["seed"]),
                "input_bundle_manifest_sha256": str(task["input_manifest_sha256"]),
                "input_file_sha256": loaded.audit["expected_file_sha256"],
                "input_event_count": int(loaded.audit["input_event_count"]),
                "terminal_censored_event_count": int(loaded.audit["terminal_censored_event_count"]),
                "complete_case_event_count": len(loaded.facts),
                "terminal_censored_candidate_count": int(loaded.audit["terminal_censored_candidate_count"]),
                "complete_case_candidate_count": int(loaded.audit["complete_case_candidate_count"]),
                "action_observation_count": int(loaded.audit["action_observation_count"]),
                "stream_count": int(loaded.audit["stream_count"]),
                "issue_batch_count": int(loaded.audit["issue_batch_count"]),
                "stream_issue_position_count": int(loaded.audit["stream_issue_position_count"]),
                "full_history_event_count": int(loaded.audit["full_history_event_count"]),
                "cold_start_event_count": int(loaded.audit["cold_start_event_count"]),
                "fact_content_sha256": str(loaded.audit["fact_content_sha256"]),
                "facts_file": str(facts_path),
                "facts_sha256": facts_sha256,
                "facts_bytes": facts_path.stat().st_size,
                "future_information_violation_count": int(loaded.audit["future_information_violation_count"]),
                "within_issue_feedback_use_count": int(loaded.audit["within_issue_feedback_use_count"]),
                "pending_feedback_count": int(loaded.audit["pending_feedback_count"]),
                "raw_width_state_attached": False,
                "heldout_performance_read": False,
                "performance_comparison_executed": False,
                "method_ranking_executed": False,
                "statistical_inference_executed": False,
                "paper_performance_claim_executed": False,
            }
            atomic_write_json(output_manifest_path, output_manifest)
            output_manifest_sha256 = sha256_file(output_manifest_path)
        elapsed = time.perf_counter() - started
        return {
            "status": "COMPLETED",
            "stream_id": str(task["stream_id"]),
            "completed_utc": utc_now(),
            "elapsed_seconds": elapsed,
            "peak_rss_gib": sampler.peak_rss / (1024.0**3),
            "input_event_count": int(loaded.audit["input_event_count"]),
            "terminal_censored_event_count": int(loaded.audit["terminal_censored_event_count"]),
            "complete_case_event_count": len(loaded.facts),
            "action_observation_count": int(loaded.audit["action_observation_count"]),
            "fact_content_sha256": str(loaded.audit["fact_content_sha256"]),
            "facts_file": str(facts_path),
            "facts_sha256": facts_sha256,
            "facts_bytes": facts_path.stat().st_size,
            "output_manifest_file": str(output_manifest_path),
            "output_manifest_sha256": output_manifest_sha256,
            "error_type": "",
            "error_message": "",
            "traceback": "",
        }
    except Exception as exc:
        return {
            "status": "FAILED",
            "stream_id": str(task["stream_id"]),
            "completed_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_gib": 0.0,
            "input_event_count": 0,
            "terminal_censored_event_count": 0,
            "complete_case_event_count": 0,
            "action_observation_count": 0,
            "fact_content_sha256": "",
            "facts_file": "",
            "facts_sha256": "",
            "facts_bytes": 0,
            "output_manifest_file": "",
            "output_manifest_sha256": "",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }


def validate_entry_identity() -> None:
    if git_output("rev-parse", "HEAD") != EXPECTED_CODE_COMMIT:
        raise RuntimeError("全量源域事实构建提交身份失配")
    if git_output("status", "--porcelain"):
        raise RuntimeError("全量源域事实构建要求权威代码工作树干净")
    contract_path = S06_ROOT / "configs" / "source_tuning_contract_v1.json"
    if sha256_file(contract_path) != SOURCE_TUNING_CONTRACT_SHA256:
        raise RuntimeError("源区嵌套选择合同SHA256失配")
    ledger_path = S03_ROOT / "bundle_stream_ledger.csv"
    if sha256_file(ledger_path) != S03_LEDGER_SHA256:
        raise RuntimeError("S03流账本SHA256失配")


def prepare_tasks(output_root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    ledger_path = S03_ROOT / "bundle_stream_ledger.csv"
    source = pd.read_csv(ledger_path)
    source = source[source["status"].astype(str) == "COMPLETED"].copy()
    if len(source) != EXPECTED_STREAM_COUNT:
        raise RuntimeError("S03完成流数量失配")
    if int(source["event_rows"].sum()) != EXPECTED_EVENT_COUNT:
        raise RuntimeError("S03事件总数失配")
    source["_zone_number"] = source["zone"].map(natural_zone_number)
    source = source.sort_values(
        ["predictor", "_zone_number", "horizon", "seed"],
        kind="mergesort",
    ).drop(columns="_zone_number").reset_index(drop=True)
    if source["stream_id"].astype(str).duplicated().any():
        raise RuntimeError("S03流账本stream_id重复")

    tasks: list[dict[str, Any]] = []
    for row in source.itertuples(index=False):
        bundle_root = S03_ROOT / "bundles" / str(row.predictor) / str(row.stream_id)
        tasks.append(
            {
                "stream_id": str(row.stream_id),
                "predictor": str(row.predictor),
                "zone": str(row.zone),
                "horizon": int(row.horizon),
                "seed": int(row.seed),
                "input_manifest_sha256": str(row.manifest_sha256),
                "bundle_root": str(bundle_root),
                "output_root": str(output_root),
            }
        )
    return source, tasks


def load_or_initialize_ledger(output_root: Path, source: pd.DataFrame) -> pd.DataFrame:
    path = output_root / "checkpoints" / "source_fact_ledger.csv"
    if path.exists():
        ledger = pd.read_csv(path, keep_default_na=False)
        if len(ledger) != EXPECTED_STREAM_COUNT:
            raise RuntimeError("S06源域事实检查点行数失配")
        if set(ledger["stream_id"].astype(str)) != set(source["stream_id"].astype(str)):
            raise RuntimeError("S06源域事实检查点流集合失配")
        return ledger
    ledger = source.loc[:, ["stream_id", "predictor", "zone", "horizon", "seed", "manifest_sha256"]].copy()
    ledger = ledger.rename(columns={"manifest_sha256": "input_manifest_sha256"})
    ledger["status"] = "PENDING"
    ledger["attempt"] = 0
    ledger["completed_utc"] = ""
    ledger["elapsed_seconds"] = 0.0
    ledger["peak_rss_gib"] = 0.0
    ledger["input_event_count"] = 0
    ledger["terminal_censored_event_count"] = 0
    ledger["complete_case_event_count"] = 0
    ledger["action_observation_count"] = 0
    ledger["fact_content_sha256"] = ""
    ledger["facts_file"] = ""
    ledger["facts_sha256"] = ""
    ledger["facts_bytes"] = 0
    ledger["output_manifest_file"] = ""
    ledger["output_manifest_sha256"] = ""
    ledger["error_type"] = ""
    ledger["error_message"] = ""
    atomic_write_csv(path, ledger)
    return ledger


def verify_completed_checkpoints(output_root: Path, ledger: pd.DataFrame) -> None:
    completed = ledger[ledger["status"].astype(str) == "COMPLETED"]
    for row in completed.itertuples(index=False):
        facts_path = Path(str(row.facts_file))
        manifest_path = Path(str(row.output_manifest_file))
        if not facts_path.exists() or not manifest_path.exists():
            raise RuntimeError(f"已完成检查点文件缺失: {row.stream_id}")
        if sha256_file(facts_path) != str(row.facts_sha256):
            raise RuntimeError(f"已完成事实文件SHA256失配: {row.stream_id}")
        if sha256_file(manifest_path) != str(row.output_manifest_sha256):
            raise RuntimeError(f"已完成事实清单SHA256失配: {row.stream_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "COMPLETE_VALIDATED":
            raise RuntimeError(f"已完成事实清单状态失配: {row.stream_id}")
        if manifest.get("code_commit") != EXPECTED_CODE_COMMIT:
            raise RuntimeError(f"已完成事实清单提交身份失配: {row.stream_id}")


def write_run_config(output_root: Path, workers: int) -> dict[str, Any]:
    path = output_root / "run_config.json"
    payload = {
        "run_id": "S06_FULL_SOURCE_FACTS_V1",
        "status": "FROZEN",
        "created_date": "2026-08-29",
        "code_commit": EXPECTED_CODE_COMMIT,
        "source_tuning_contract_sha256": SOURCE_TUNING_CONTRACT_SHA256,
        "s03_ledger_sha256": S03_LEDGER_SHA256,
        "expected_stream_count": EXPECTED_STREAM_COUNT,
        "expected_event_count": EXPECTED_EVENT_COUNT,
        "expected_terminal_event_count": EXPECTED_TERMINAL_EVENT_COUNT,
        "expected_complete_event_count": EXPECTED_COMPLETE_EVENT_COUNT,
        "expected_action_observation_count": EXPECTED_ACTION_OBSERVATION_COUNT,
        "worker_count": int(workers),
        "minimum_free_disk_gib": MINIMUM_FREE_DISK_GIB,
        "output_compression": "zstd",
        "s03_read_only": True,
        "raw_width_state_attached": False,
        "performance_comparison_executed": False,
    }
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != payload:
            raise RuntimeError("全量源域事实运行配置与既有冻结配置失配")
    else:
        atomic_write_json(path, payload)
    return payload


def resource_record(
    *,
    invocation_id: str,
    completed_count: int,
    failed_count: int,
    elapsed_seconds: float,
    output_bytes: int,
) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(REVISION_ROOT)
    return {
        "timestamp_utc": utc_now(),
        "invocation_id": invocation_id,
        "completed_count": int(completed_count),
        "failed_count": int(failed_count),
        "elapsed_seconds": float(elapsed_seconds),
        "output_bytes": int(output_bytes),
        "available_ram_gib": memory.available / (1024.0**3),
        "free_disk_gib": disk.free / (1024.0**3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-streams", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    workers = int(args.workers)
    if workers < 1 or workers > 8:
        raise ValueError("并发进程数必须位于一到八")
    max_streams = None if args.max_streams is None else int(args.max_streams)
    if max_streams is not None and max_streams <= 0:
        raise ValueError("max-streams必须为正")

    validate_entry_identity()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_config = write_run_config(output_root, workers)
    source, tasks = prepare_tasks(output_root)
    ledger_path = output_root / "checkpoints" / "source_fact_ledger.csv"
    ledger = load_or_initialize_ledger(output_root, source)
    verify_completed_checkpoints(output_root, ledger)
    task_by_id = {task["stream_id"]: task for task in tasks}
    pending_ids = ledger.loc[ledger["status"].astype(str) != "COMPLETED", "stream_id"].astype(str).tolist()
    if max_streams is not None:
        pending_ids = pending_ids[:max_streams]

    disk_before = shutil.disk_usage(REVISION_ROOT)
    free_gib_before = disk_before.free / (1024.0**3)
    remaining_count = int((ledger["status"].astype(str) != "COMPLETED").sum())
    projected_remaining_gib = PREFLIGHT_FACT_BYTES * remaining_count / (1024.0**3)
    if free_gib_before < MINIMUM_FREE_DISK_GIB:
        raise RuntimeError("全量源域事实构建磁盘安全门失败")
    if free_gib_before - projected_remaining_gib < MINIMUM_FREE_DISK_GIB:
        raise RuntimeError("全量源域事实预计输出将触发磁盘安全门")

    invocation_id = datetime.now(timezone.utc).strftime("S06_SOURCE_FACTS_%Y%m%dT%H%M%S%fZ")
    invocation_path = output_root / "logs" / "runner_invocations.jsonl"
    invocation_start = {
        "invocation_id": invocation_id,
        "event": "START",
        "timestamp_utc": utc_now(),
        "workers": workers,
        "max_streams": max_streams,
        "pending_selected": len(pending_ids),
        "completed_before": int((ledger["status"].astype(str) == "COMPLETED").sum()),
        "free_disk_gib_before": free_gib_before,
        "projected_remaining_output_gib": projected_remaining_gib,
        "code_commit": EXPECTED_CODE_COMMIT,
        "source_tuning_contract_sha256": SOURCE_TUNING_CONTRACT_SHA256,
    }
    append_jsonl(invocation_path, invocation_start)
    print(json.dumps(invocation_start, ensure_ascii=False), flush=True)

    if not pending_ids:
        print(json.dumps({"status": "ALREADY_COMPLETE", "invocation_id": invocation_id}, ensure_ascii=False))
        return

    started = time.perf_counter()
    failures: list[dict[str, Any]] = []
    invocation_completed = 0
    resource_rows: list[dict[str, Any]] = []
    last_progress_time = time.monotonic()
    ledger_index = {str(value): index for index, value in ledger["stream_id"].items()}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending_iterator = iter(pending_ids)
        active: dict[Any, str] = {}
        for _ in range(min(workers * 2, len(pending_ids))):
            stream_id = next(pending_iterator, None)
            if stream_id is None:
                break
            active[executor.submit(build_one, task_by_id[stream_id])] = stream_id

        while active:
            done, _ = wait(active, timeout=10.0, return_when=FIRST_COMPLETED)
            if not done:
                now = time.monotonic()
                if now - last_progress_time >= 30.0:
                    completed_total = int((ledger["status"].astype(str) == "COMPLETED").sum())
                    snapshot = resource_record(
                        invocation_id=invocation_id,
                        completed_count=completed_total,
                        failed_count=len(failures),
                        elapsed_seconds=time.perf_counter() - started,
                        output_bytes=int(pd.to_numeric(ledger["facts_bytes"], errors="coerce").fillna(0).sum()),
                    )
                    resource_rows.append(snapshot)
                    print(json.dumps({"event": "PROGRESS", **snapshot}, ensure_ascii=False), flush=True)
                    last_progress_time = now
                continue

            for future in done:
                stream_id = active.pop(future)
                result = future.result()
                row_index = ledger_index[stream_id]
                ledger.at[row_index, "attempt"] = int(ledger.at[row_index, "attempt"]) + 1
                for key, value in result.items():
                    if key in ledger.columns and key != "stream_id":
                        ledger.at[row_index, key] = value
                if result["status"] == "COMPLETED":
                    invocation_completed += 1
                else:
                    failures.append(result)
                    append_jsonl(
                        output_root / "logs" / "exceptions.jsonl",
                        {"invocation_id": invocation_id, **result},
                    )
                atomic_write_csv(ledger_path, ledger)

                disk_now = shutil.disk_usage(REVISION_ROOT)
                if disk_now.free / (1024.0**3) < MINIMUM_FREE_DISK_GIB:
                    failures.append(
                        {
                            "status": "FAILED",
                            "stream_id": "DISK_SAFETY_GATE",
                            "error_type": "DiskSafetyGate",
                            "error_message": "运行中可用磁盘低于冻结安全门",
                        }
                    )

                if failures:
                    for queued in active:
                        queued.cancel()
                    active.clear()
                    break

                next_stream_id = next(pending_iterator, None)
                if next_stream_id is not None:
                    active[executor.submit(build_one, task_by_id[next_stream_id])] = next_stream_id

                now = time.monotonic()
                if invocation_completed % 12 == 0 or now - last_progress_time >= 30.0:
                    completed_total = int((ledger["status"].astype(str) == "COMPLETED").sum())
                    snapshot = resource_record(
                        invocation_id=invocation_id,
                        completed_count=completed_total,
                        failed_count=len(failures),
                        elapsed_seconds=time.perf_counter() - started,
                        output_bytes=int(pd.to_numeric(ledger["facts_bytes"], errors="coerce").fillna(0).sum()),
                    )
                    resource_rows.append(snapshot)
                    print(json.dumps({"event": "PROGRESS", **snapshot}, ensure_ascii=False), flush=True)
                    last_progress_time = now

    elapsed = time.perf_counter() - started
    if resource_rows:
        resource_path = output_root / "logs" / "resource_samples.csv"
        new_resources = pd.DataFrame(resource_rows)
        if resource_path.exists():
            previous = pd.read_csv(resource_path)
            new_resources = pd.concat([previous, new_resources], ignore_index=True)
        atomic_write_csv(resource_path, new_resources)

    ledger = pd.read_csv(ledger_path, keep_default_na=False)
    completed_total = int((ledger["status"].astype(str) == "COMPLETED").sum())
    failed_total = int((ledger["status"].astype(str) == "FAILED").sum())
    pending_total = len(ledger) - completed_total
    output_bytes = int(pd.to_numeric(ledger["facts_bytes"], errors="coerce").fillna(0).sum())
    summary_status = (
        "FAIL"
        if failures or failed_total
        else "COMPLETE"
        if completed_total == EXPECTED_STREAM_COUNT
        else "PARTIAL_LIMIT_REACHED"
    )
    summary = {
        "run_id": run_config["run_id"],
        "invocation_id": invocation_id,
        "generated_at_utc": utc_now(),
        "status": summary_status,
        "code_commit": EXPECTED_CODE_COMMIT,
        "source_tuning_contract_sha256": SOURCE_TUNING_CONTRACT_SHA256,
        "workers": workers,
        "max_streams": max_streams,
        "invocation_completed_count": invocation_completed,
        "completed_stream_count": completed_total,
        "pending_stream_count": pending_total,
        "failed_stream_count": failed_total,
        "elapsed_seconds": elapsed,
        "invocation_streams_per_minute": invocation_completed / elapsed * 60.0 if elapsed > 0 else 0.0,
        "output_bytes": output_bytes,
        "output_gib": output_bytes / (1024.0**3),
        "free_disk_gib_after": shutil.disk_usage(REVISION_ROOT).free / (1024.0**3),
        "failure_count_this_invocation": len(failures),
        "heldout_performance_read": False,
        "performance_comparison_executed": False,
        "method_ranking_executed": False,
        "statistical_inference_executed": False,
        "paper_performance_claim_executed": False,
        "new_scientific_adverse_evidence": False,
        "stop_triggered": bool(failures),
    }
    atomic_write_json(output_root / "source_fact_runner_summary.json", summary)
    append_jsonl(
        invocation_path,
        {"invocation_id": invocation_id, "event": "END", **summary},
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise RuntimeError("全量源域事实构建触发失败安全门")


if __name__ == "__main__":
    main()
