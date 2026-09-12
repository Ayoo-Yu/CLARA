from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


EXPECTED_BASE_PROTOCOL_SHA256 = "d5109ccc1492ebb674bbbd8c0c05ec96cb4549072acec9a8c5e0888f507c4757"
EXPECTED_PROTOCOL_SHA256 = "194f9c7d52efe7f59c91e4c4e8bba0a17bb3712240dfb69832ce5cd875ecc22b"
EXPECTED_BASELINE_SHA256 = "ea0d018cf46c94993396dc3ab2d58adcfb5f869e99731f8c4a04befff3659350"
EXPECTED_BASE_OUTPUT_SHA256 = "12fd166c19570073be5f5acde3712e079f7c5a44efa673cfaf753d40ab3d0b31"
EXPECTED_OUTPUT_SHA256 = "e075ad3f80ed18de0c9331b5136cf160cfd6ac243ffd9dd9e3d16c68109425be"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo is not None else value
    if hasattr(value, "to_pydatetime"):
        result = value.to_pydatetime()
        return result.replace(tzinfo=None) if result.tzinfo is not None else result
    text = str(value).replace("Z", "+00:00")
    result = datetime.fromisoformat(text)
    return result.replace(tzinfo=None) if result.tzinfo is not None else result


def canonical_timestamp(value: Any) -> str:
    return as_datetime(value).isoformat()


@dataclass(frozen=True)
class FrozenContracts:
    protocol: dict[str, Any]
    base_protocol: dict[str, Any]
    baseline_registry: dict[str, Any]
    output_contract: dict[str, Any]
    base_output_contract: dict[str, Any]
    protocol_path: Path
    base_protocol_path: Path
    baseline_path: Path
    output_path: Path
    base_output_path: Path
    protocol_sha256: str
    base_protocol_sha256: str
    baseline_sha256: str
    output_sha256: str
    base_output_sha256: str

    @property
    def protocol_id(self) -> str:
        return str(self.protocol["protocol_id"])

    @property
    def protocol_version(self) -> str:
        return str(self.protocol["protocol_version"])

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(self.protocol["action_contract"]["actions_in_tie_order"])

    @property
    def state_fields(self) -> tuple[str, ...]:
        return tuple(self.protocol["state_contract"]["state_fields_in_order"])

    @property
    def coverages(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.protocol["coverage_contract"]["target_coverage_levels"])

    @property
    def support_levels(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.protocol["support_and_backoff_contract"]["levels"])

    @property
    def accepted_candidate_bundle_identities(self) -> tuple[dict[str, str], ...]:
        compatibility = self.protocol["protocol_lineage"]["s03_candidate_bundle_compatibility"]
        return (
            {
                "protocol_id": self.protocol_id,
                "protocol_version": self.protocol_version,
                "protocol_sha256": self.protocol_sha256,
                "baseline_registry_sha256": self.baseline_sha256,
                "output_contract_sha256": self.output_sha256,
            },
            {
                "protocol_id": self.protocol_id,
                "protocol_version": str(compatibility["accepted_protocol_version"]),
                "protocol_sha256": str(compatibility["accepted_protocol_sha256"]),
                "baseline_registry_sha256": self.baseline_sha256,
                "output_contract_sha256": str(compatibility["accepted_output_contract_sha256"]),
            },
        )

    @property
    def event_required_fields(self) -> tuple[str, ...]:
        return tuple(self.output_contract["tables"]["events"]["required_fields"])

    @property
    def candidate_required_fields(self) -> tuple[str, ...]:
        return tuple(self.output_contract["tables"]["candidates"]["required_fields"])

    @property
    def feedback_required_fields(self) -> tuple[str, ...]:
        return tuple(self.output_contract["tables"]["feedback"]["required_fields"])

    def horizon_group(self, horizon_hours: float) -> str:
        rounded = int(round(float(horizon_hours)))
        if abs(float(horizon_hours) - rounded) > 1e-9:
            raise ValueError(f"冻结时长组只接受整数物理小时，收到{horizon_hours}")
        for name, hours in self.protocol["horizon_group_contract"]["groups"].items():
            if rounded in hours:
                return str(name)
        raise ValueError(f"物理时长{rounded}小时不在冻结时长组中")

    def make_event_id(
        self,
        *,
        dataset_id: str,
        zone_or_farm: str,
        predictor: str,
        horizon_steps: int,
        seed: int,
        target_coverage: float,
        issue_timestamp: Any,
    ) -> str:
        coverage = float(target_coverage)
        if coverage not in self.coverages:
            raise ValueError(f"覆盖率{coverage}不在冻结十一档中")
        payload = {
            "dataset_id": str(dataset_id),
            "zone_or_farm": str(zone_or_farm),
            "predictor": str(predictor),
            "horizon_steps": int(horizon_steps),
            "seed": int(seed),
            "target_coverage": coverage,
            "issue_timestamp": canonical_timestamp(issue_timestamp),
        }
        return canonical_json_sha256(payload)

    def validate_output_record(self, table: str, record: Mapping[str, Any]) -> None:
        required = tuple(self.output_contract["tables"][table]["required_fields"])
        missing = [field for field in required if field not in record]
        if missing:
            raise ValueError(f"{table}输出缺少字段: {missing}")


def default_protocol_dir() -> Path:
    revision_root = Path(__file__).resolve().parents[2]
    return revision_root / "02_完整实验协议冻结" / "protocol"


def load_frozen_contracts(protocol_dir: str | Path | None = None) -> FrozenContracts:
    directory = Path(protocol_dir) if protocol_dir is not None else default_protocol_dir()
    base_protocol_path = directory / "causal_event_protocol_v1.json"
    protocol_path = directory / "causal_event_protocol_v1_1.json"
    baseline_path = directory / "baseline_registry_v1.json"
    base_output_path = directory / "event_output_contract_v1.json"
    output_path = directory / "event_output_contract_v1_1.json"
    hashes = {
        "base_protocol": sha256_file(base_protocol_path),
        "protocol": sha256_file(protocol_path),
        "baseline": sha256_file(baseline_path),
        "base_output": sha256_file(base_output_path),
        "output": sha256_file(output_path),
    }
    expected = {
        "base_protocol": EXPECTED_BASE_PROTOCOL_SHA256,
        "protocol": EXPECTED_PROTOCOL_SHA256,
        "baseline": EXPECTED_BASELINE_SHA256,
        "base_output": EXPECTED_BASE_OUTPUT_SHA256,
        "output": EXPECTED_OUTPUT_SHA256,
    }
    mismatches = {name: (hashes[name], expected[name]) for name in expected if hashes[name] != expected[name]}
    if mismatches:
        raise RuntimeError(f"冻结合同SHA256失配: {mismatches}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    base_protocol = json.loads(base_protocol_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    output = json.loads(output_path.read_text(encoding="utf-8"))
    base_output = json.loads(base_output_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "CLARA_CAUSAL_EVENT_V1" or protocol.get("protocol_version") != "1.1.0":
        raise RuntimeError("冻结协议身份失配")
    if base_protocol.get("protocol_id") != "CLARA_CAUSAL_EVENT_V1" or base_protocol.get("protocol_version") != "1.0.0":
        raise RuntimeError("基础协议身份失配")
    return FrozenContracts(
        protocol=protocol,
        base_protocol=base_protocol,
        baseline_registry=baseline,
        output_contract=output,
        base_output_contract=base_output,
        protocol_path=protocol_path,
        base_protocol_path=base_protocol_path,
        baseline_path=baseline_path,
        output_path=output_path,
        base_output_path=base_output_path,
        protocol_sha256=hashes["protocol"],
        base_protocol_sha256=hashes["base_protocol"],
        baseline_sha256=hashes["baseline"],
        output_sha256=hashes["output"],
        base_output_sha256=hashes["base_output"],
    )
