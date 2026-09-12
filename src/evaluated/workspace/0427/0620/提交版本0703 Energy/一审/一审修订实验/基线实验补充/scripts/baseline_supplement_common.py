from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUPPLEMENT_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = SUPPLEMENT_ROOT.parent
AUTH_ROOT = REVISION_ROOT / "_权威代码"
AUTH_CODE = AUTH_ROOT / "code"
S03_ROOT = (
    REVISION_ROOT
    / "03_基础预测与候选区间重建"
    / "results_raw"
    / "full_rebuild_v1"
)
S06_ROOT = REVISION_ROOT / "06_基线实现与训练区调参"
S07_ROOT = REVISION_ROOT / "07_GEFCom完整主实验"
S07B_ROOT = REVISION_ROOT / "07B_CLARA经验阈值修正重算与新旧对比"
CONFIG_PATH = SUPPLEMENT_ROOT / "configs" / "baseline_supplement_v1.json"
S06_FACT_ROOT = S06_ROOT / "results_raw" / "nested_source_selection_v1"
RESULT_ROOT = SUPPLEMENT_ROOT / "results_raw" / "baseline_supplement_v2"
CART_CACHE_ROOT = SUPPLEMENT_ROOT / "results_raw" / "cart_state_caches_v1"
PILOT_ROOT = SUPPLEMENT_ROOT / "results_raw" / "pilot_decoupled_v2"

STATE_FIELDS = (
    "predictor",
    "horizon_group",
    "target_coverage",
    "ramp_state",
    "rolling_state",
    "raw_width_state",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(AUTH_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def validate_frozen_inputs(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for input_id, spec in config["inputs"].items():
        path = REVISION_ROOT / str(spec["relative_path"])
        if not path.exists():
            raise FileNotFoundError(f"冻结输入不存在: {path}")
        observed = sha256_file(path)
        passed = observed == str(spec["sha256"])
        rows.append(
            {
                "input_id": input_id,
                "relative_path": str(spec["relative_path"]),
                "expected_sha256": str(spec["sha256"]),
                "observed_sha256": observed,
                "bytes": int(path.stat().st_size),
                "status": "PASS" if passed else "FAIL",
            }
        )
    for module_name, expected in config["authoritative_modules"].items():
        path = AUTH_CODE / module_name
        observed = sha256_file(path)
        passed = observed == str(expected)
        rows.append(
            {
                "input_id": f"module:{module_name}",
                "relative_path": str(path.relative_to(REVISION_ROOT).as_posix()),
                "expected_sha256": str(expected),
                "observed_sha256": observed,
                "bytes": int(path.stat().st_size),
                "status": "PASS" if passed else "FAIL",
            }
        )
    audit = pd.DataFrame(rows)
    if audit["status"].ne("PASS").any():
        failures = audit[audit["status"].ne("PASS")].to_dict("records")
        raise RuntimeError(f"冻结输入或模块哈希失配: {failures}")
    expected_commit = str(config["protocol"]["authoritative_code_commit"])
    observed_commit = git_output("rev-parse", "HEAD")
    if observed_commit != expected_commit or git_output("status", "--short"):
        raise RuntimeError("权威代码提交或清洁状态失配")
    return audit


def free_disk_gib() -> float:
    return float(shutil.disk_usage(SUPPLEMENT_ROOT).free / (1024**3))


def compute_event_metrics(
    *,
    target: np.ndarray,
    schedule: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    cadence_minutes: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    target = np.asarray(target, dtype=np.float64)
    schedule = np.asarray(schedule, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if cadence_minutes is None:
        cadence_minutes = np.full(len(target), 60.0, dtype=np.float64)
    cadence_minutes = np.asarray(cadence_minutes, dtype=np.float64)
    if not (
        len(target)
        == len(schedule)
        == len(lower)
        == len(upper)
        == len(cadence_minutes)
    ):
        raise ValueError("事件指标输入长度失配")
    if np.any(~np.isfinite(np.column_stack([target, schedule, lower, upper, cadence_minutes]))):
        raise ValueError("事件指标输入含非有限值")
    if np.any(lower > upper) or np.any(cadence_minutes <= 0.0):
        raise ValueError("事件指标区间或时间步长无效")
    scale = cadence_minutes / 60.0
    reserve_up = scale * np.maximum(upper - schedule, 0.0) * 3.56
    reserve_down = scale * np.maximum(schedule - lower, 0.0) * 3.56
    miss_upper = scale * np.maximum(target - upper, 0.0) * 20.0
    miss_lower = scale * np.maximum(lower - target, 0.0) * 20.0
    errf = reserve_up + reserve_down + miss_upper + miss_lower
    covered = (lower <= target) & (target <= upper)
    return {
        "lower": lower,
        "upper": upper,
        "width": upper - lower,
        "covered": covered,
        "reserve_up": reserve_up,
        "reserve_down": reserve_down,
        "miss_upper": miss_upper,
        "miss_lower": miss_lower,
        "errf": errf,
    }


def unit_id(zone: str, seed: int, horizon: int) -> str:
    return f"{zone}__seed{int(seed)}__H{int(horizon):02d}"
