"""Verify the packaged evaluated-source hashes and load its numerical contracts."""
import hashlib
import json
from _bootstrap import REPOSITORY
from clara_event_contract import load_frozen_contracts


def main():
    records = json.loads((REPOSITORY / "configs/source_provenance.json").read_text(encoding="utf-8"))
    failures = []
    for record in records:
        path = REPOSITORY / record["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"
        if digest != record["packaged_sha256"]:
            failures.append(record["path"])
    if failures:
        raise RuntimeError("Source hash mismatch: " + ", ".join(failures))
    contracts = load_frozen_contracts()
    print(json.dumps({"status": "PASS", "source_records": len(records),
                      "python_modules": sum(r["path"].endswith(".py") for r in records),
                      "protocol_sha256": contracts.protocol_sha256}, indent=2))


if __name__ == "__main__":
    main()
