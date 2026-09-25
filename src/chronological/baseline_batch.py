"""Bounded process orchestration; no baseline, tuning or metric logic lives here."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from _paths import DATA_ROOT as RUN, CODE_ROOT


def run_job(mode, zone, threads):
    if (RUN / "STOP_REQUESTED.txt").exists():
        raise RuntimeError("New-run STOP_REQUESTED is present")
    started = time.perf_counter()
    phases = {
        "cart-outer": [("cart_forward.py", "--outer", zone, "--mode", "tune"),
                       ("cart_forward.py", "--outer", zone, "--mode", "final")],
        "cart-final": [("cart_forward.py", "--outer", zone, "--mode", "final")],
        "linucb-outer": [("linucb_forward.py", "--outer", zone, "--mode", "outer", "--threads", str(threads))],
        "warm": [("linucb_warmstart.py", "--zone", zone, "--seed", str(seed), "--threads", str(threads))
                 for seed in range(3)],
    }[mode]
    receipts = []
    for i, command in enumerate(phases):
        if (RUN / "STOP_REQUESTED.txt").exists():
            raise RuntimeError("New-run STOP_REQUESTED is present")
        path = RUN / "logs" / f"{mode}_{zone}_phase{i}.log"
        with path.open("w", encoding="utf-8") as log:
            process = subprocess.run([sys.executable, "-X", "utf8", "-u", str(CODE_ROOT/command[0]), *command[1:]],
                                     cwd=RUN, stdout=log, stderr=subprocess.STDOUT)
        receipts.append(dict(command=list(command), returncode=process.returncode, log=str(path)))
        if process.returncode:
            raise RuntimeError(f"{mode} {zone} phase {i} failed; inspect {path}")
    return dict(mode=mode, zone=zone, status="complete", seconds=time.perf_counter() - started,
                phases=receipts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("cart-outer", "cart-final", "linucb-outer", "warm"), required=True)
    p.add_argument("--zones", nargs="+", required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()
    if not 1 <= a.workers <= 8 or not 1 <= a.threads <= 4:
        raise ValueError("Unexpected worker/thread count")
    (RUN / "logs").mkdir(exist_ok=True)
    report = dict(mode=a.mode, zones=a.zones, workers=a.workers, threads=a.threads,
                  started_utc=datetime.now(timezone.utc).isoformat(), completed=[], errors=[])
    path = RUN / f"batch_{a.mode}_{a.zones[0]}_{a.zones[-1]}.json"
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs = {pool.submit(run_job, a.mode, z, a.threads): z for z in a.zones}
        for future in as_completed(jobs):
            try:
                row = future.result()
                report["completed"].append(row)
                print(json.dumps({k: row[k] for k in ("mode", "zone", "status", "seconds")}), flush=True)
            except Exception as exc:
                error = dict(zone=jobs[future], error=str(exc))
                report["errors"].append(error)
                print(json.dumps(error), flush=True)
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["status"] = "PASS" if not report["errors"] and len(report["completed"]) == len(a.zones) else "FAIL"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["status"] != "PASS":
        raise RuntimeError(f"Incomplete baseline batch: {path}")


if __name__ == "__main__":
    main()
