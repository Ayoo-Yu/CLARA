from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SENSITIVITY_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = SENSITIVITY_ROOT.parents[1]
CONFIG_PATH = SENSITIVITY_ROOT / "configs" / "price_sensitivity_protocol_v1.json"
AUTH_ROOT = REVISION_ROOT / "_权威代码"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def load_config() -> tuple[dict[str, Any], str]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return config, sha256_file(CONFIG_PATH)


def validate_frozen_inputs(config: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name, item in config["inputs"].items():
        path = REVISION_ROOT / str(item["relative_path"])
        observed = sha256_file(path) if path.is_file() else None
        expected = str(item["sha256"])
        records.append(
            {
                "input_name": name,
                "path": str(path),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "status": "PASS" if observed == expected else "FAIL",
            }
        )
    failures = [record for record in records if record["status"] != "PASS"]
    if failures:
        raise RuntimeError(f"冻结输入哈希失配: {failures}")
    return records


def ensure_disk_gate(config: dict[str, Any], root: Path) -> float:
    free_gib = shutil.disk_usage(root).free / (1024.0**3)
    minimum = float(config["execution"]["minimum_free_disk_gib"])
    if free_gib < minimum:
        raise RuntimeError(f"磁盘安全门失败: free={free_gib:.3f} GiB, minimum={minimum:.3f} GiB")
    return free_gib


def authoritative_git_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(AUTH_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(AUTH_ROOT), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    return {"commit": commit, "status": status, "clean": status == ""}


def horizon_group(horizon: int) -> str:
    if int(horizon) == 1:
        return "H01"
    if int(horizon) <= 3:
        return "H02_H03"
    if int(horizon) <= 6:
        return "H04_H06"
    if int(horizon) <= 12:
        return "H07_H12"
    return "H13_H24"


def price_record(config: dict[str, Any], price_id: str) -> dict[str, Any]:
    matches = [item for item in config["price_grid"] if str(item["price_id"]) == str(price_id)]
    if len(matches) != 1:
        raise KeyError(f"价格配置不存在或不唯一: {price_id}")
    return matches[0]
