"""Non-scientific subprocess scheduler for the token-gated selector workers.

This file never computes or commits scientific artifacts.  Every child invokes the
frozen selector runner in a fresh ``python -s`` process and must independently pass
the current preflight capability.  Restarting a scheduler run is safe because each
worker deep-validates and resumes its own atomic parent/price children.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


TEST_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = TEST_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import run_gefcom_six_action_selectors_v1 as runner  # noqa: E402


SCHEDULER_PATH = Path(__file__).resolve()
RUNNER_PATH = SCRIPT_ROOT / "run_gefcom_six_action_selectors_v1.py"
RUN_ROOT = runner.CONTROL_ROOT / "scheduler_runs"
FORMAL_ENVIRONMENT = {
    "PYTHONNOUSERSITE": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
STAGE_LIMITS = {"fit": 4, "replay": 2}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def formal_python() -> Path:
    phase, _ = runner.load_configs()
    path = Path(str(phase["environment"]["formal_python_executable"])).resolve()
    if not path.is_file() or Path(sys.executable).resolve() != path:
        raise RuntimeError("scheduler必须由冻结formal python启动")
    if not bool(sys.flags.no_user_site):
        raise RuntimeError("scheduler必须使用python -s")
    return path


def worker_command(
    *,
    stage: str,
    zone: str,
    seed: int,
    token_path: Path,
    token_sha256: str,
) -> list[str]:
    if stage not in STAGE_LIMITS:
        raise ValueError("scheduler stage非法")
    if zone not in runner.core.FORMAL_ZONES or int(seed) not in range(3):
        raise ValueError("scheduler worker轴越界")
    command_name = "fit-one" if stage == "fit" else "replay-one"
    return [
        str(formal_python()),
        "-s",
        str(RUNNER_PATH),
        command_name,
        "--zone",
        str(zone),
        "--seed",
        str(int(seed)),
        "--token",
        str(Path(token_path).resolve()),
        "--token-sha256",
        str(token_sha256),
    ]


def _atomic_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    runner.atomic_json(path, dict(payload))


def run_stage(
    *,
    stage: str,
    workers: int,
    token_path: Path,
    token_sha256: str,
) -> dict[str, Any]:
    if stage not in STAGE_LIMITS:
        raise ValueError("scheduler stage非法")
    worker_count = int(workers)
    if worker_count <= 0 or worker_count > STAGE_LIMITS[stage]:
        raise RuntimeError(
            f"{stage} workers必须在1..{STAGE_LIMITS[stage]}; "
            "replay上限来自真实target峰值基准"
        )
    token = runner.validate_preflight_token(
        Path(token_path), expected_token_sha256=str(token_sha256)
    )
    run_id = (
        f"{stage}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        f"__{uuid.uuid4().hex[:8]}"
    )
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    tasks = [
        (zone, seed)
        for zone in runner.core.FORMAL_ZONES
        for seed in range(3)
    ]
    pending = list(tasks)
    active: dict[subprocess.Popen[str], dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []
    failed = False
    child_environment = os.environ.copy()
    child_environment.update(FORMAL_ENVIRONMENT)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    started = time.perf_counter()
    while pending or active:
        while pending and len(active) < worker_count and not failed:
            zone, seed = pending.pop(0)
            unit_id = f"{zone}__seed{seed}"
            log_path = run_dir / f"{unit_id}.log"
            command = worker_command(
                stage=stage,
                zone=zone,
                seed=seed,
                token_path=Path(token_path),
                token_sha256=str(token_sha256),
            )
            handle = log_path.open("w", encoding="utf-8", newline="\n")
            process = subprocess.Popen(
                command,
                cwd=str(runner.REVISION_ROOT),
                env=child_environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                shell=False,
                creationflags=creationflags,
            )
            active[process] = {
                "unit_id": unit_id,
                "zone": zone,
                "seed": seed,
                "command": command,
                "log_path": str(log_path),
                "log_handle": handle,
                "started_at_utc": utc_now(),
                "started_monotonic": time.perf_counter(),
            }
        if not active:
            break
        progressed = False
        for process, record in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            progressed = True
            record["log_handle"].close()
            finished = {
                key: value
                for key, value in record.items()
                if key not in {"log_handle", "started_monotonic"}
            }
            finished.update(
                {
                    "finished_at_utc": utc_now(),
                    "elapsed_seconds": time.perf_counter()
                    - float(record["started_monotonic"]),
                    "return_code": int(return_code),
                    "status": "PASS" if return_code == 0 else "FAIL",
                    "log_sha256": sha256_file(Path(record["log_path"])),
                }
            )
            completed.append(finished)
            del active[process]
            if return_code != 0:
                failed = True
        if not progressed:
            time.sleep(0.25)
    # Let already-started workers finish, but never dispatch additional units
    # after the first failure.  This preserves atomic children and clear restart.
    while active:
        for process, record in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            record["log_handle"].close()
            finished = {
                key: value
                for key, value in record.items()
                if key not in {"log_handle", "started_monotonic"}
            }
            finished.update(
                {
                    "finished_at_utc": utc_now(),
                    "elapsed_seconds": time.perf_counter()
                    - float(record["started_monotonic"]),
                    "return_code": int(return_code),
                    "status": "PASS" if return_code == 0 else "FAIL",
                    "log_sha256": sha256_file(Path(record["log_path"])),
                }
            )
            completed.append(finished)
            del active[process]
            failed = failed or return_code != 0
        if active:
            time.sleep(0.25)
    runner.validate_preflight_token(
        Path(token_path), expected_token_sha256=str(token_sha256)
    )
    completed.sort(key=lambda row: (row["zone"], int(row["seed"])))
    status = (
        "PASS"
        if not failed and not pending and len(completed) == 30
        else "FAIL"
    )
    manifest = {
        "schema": "TEST_CLARA_GEFCOM_6A_SELECTOR_SCHEDULER_RUN_V1",
        "status": status,
        "stage": stage,
        "worker_count": worker_count,
        "started_at_utc": token.get("issued_at_utc"),
        "finished_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "scheduler_sha256": sha256_file(SCHEDULER_PATH),
        "runner_sha256": sha256_file(RUNNER_PATH),
        "token_path": str(Path(token_path).resolve()),
        "token_manifest_sha256": str(token_sha256),
        "execution_identity_sha256": token["execution_identity_sha256"],
        "planned_unit_count": 30,
        "started_unit_count": len(completed),
        "passed_unit_count": sum(row["status"] == "PASS" for row in completed),
        "failed_unit_count": sum(row["status"] == "FAIL" for row in completed),
        "undispatched_units": [f"{zone}__seed{seed}" for zone, seed in pending],
        "units": completed,
    }
    _atomic_manifest(run_dir / "manifest.json", manifest)
    return {
        "status": status,
        "stage": stage,
        "run_manifest_path": str(run_dir / "manifest.json"),
        "run_manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "passed_unit_count": manifest["passed_unit_count"],
        "failed_unit_count": manifest["failed_unit_count"],
        "undispatched_unit_count": len(pending),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "replay"))
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--token", required=True, type=Path)
    parser.add_argument("--token-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_stage(
        stage=args.stage,
        workers=args.workers,
        token_path=args.token,
        token_sha256=args.token_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
