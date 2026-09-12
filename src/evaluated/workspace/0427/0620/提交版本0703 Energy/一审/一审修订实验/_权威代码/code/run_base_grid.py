"""Run a hash-validated base predictor grid."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import (
    canonical_json_sha256,
    ensure_dir,
    git_commit_for_path,
    load_json,
    save_json,
    sha256_file,
)
from run_base_predictor import run_base_prediction


def _build_run_config(grid_cfg: dict, zone: str, horizon: int, seed: int) -> dict:
    return {
        "protocol_id": grid_cfg["protocol_id"],
        "protocol_version": grid_cfg["protocol_version"],
        "protocol_sha256": grid_cfg["protocol_sha256"],
        "dataset_id": grid_cfg.get("dataset_id", "gefcom2014"),
        "predictor": grid_cfg["predictor"],
        "zone": zone,
        "seed": int(seed),
        "horizon": int(horizon),
        "data_path": grid_cfg["data_paths"][zone],
        "split": grid_cfg["split"],
        "save_splits": grid_cfg.get("save_splits", ["calibration", "test"]),
        "results_dir": str(
            Path(grid_cfg["results_dir"])
            / f"{grid_cfg['predictor']}-H{horizon:02d}-{zone}-S{seed}"
        ),
    }


def _cache_mismatches(manifest_path: Path, run_cfg: dict) -> list[str]:
    manifest = load_json(manifest_path)
    repository_root = Path(__file__).resolve().parents[1]
    expected = {
        "protocol_id": run_cfg["protocol_id"],
        "protocol_version": run_cfg["protocol_version"],
        "protocol_sha256": run_cfg["protocol_sha256"],
        "code_commit": git_commit_for_path(repository_root),
        "config_sha256": canonical_json_sha256(run_cfg),
        "input_sha256": sha256_file(run_cfg["data_path"]),
    }
    mismatches = [
        key
        for key, value in expected.items()
        if manifest.get(key) != value
    ]
    output_path = Path(
        manifest.get(
            "output_file",
            Path(run_cfg["results_dir"]) / "base_predictions.parquet",
        )
    )
    if not output_path.exists():
        mismatches.append("output_file_missing")
    elif manifest.get("output_sha256") != sha256_file(output_path):
        mismatches.append("output_sha256")
    if manifest.get("status") != "COMPLETE_VALIDATED":
        mismatches.append("manifest_status")
    return sorted(set(mismatches))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_json(args.config)
    for required in ("protocol_id", "protocol_version", "protocol_sha256"):
        if not cfg.get(required):
            raise ValueError(f"网格配置缺少{required}")

    total = 0
    skipped = 0
    for zone in cfg["zones"]:
        for horizon in cfg["horizons"]:
            for seed in cfg["seeds"]:
                total += 1
                run_cfg = _build_run_config(cfg, zone, int(horizon), int(seed))
                run_dir = Path(run_cfg["results_dir"])
                manifest_path = run_dir / "manifest.json"
                output_path = run_dir / "base_predictions.parquet"

                if manifest_path.exists() and not args.force:
                    mismatches = _cache_mismatches(manifest_path, run_cfg)
                    if mismatches:
                        joined = ", ".join(mismatches)
                        raise RuntimeError(
                            f"缓存身份核验失败: {run_dir}; 差异字段: {joined}"
                        )
                    skipped += 1
                    continue
                if output_path.exists() and not manifest_path.exists() and not args.force:
                    raise RuntimeError(f"发现没有manifest的输出缓存: {output_path}")

                ensure_dir(run_dir)
                save_json(run_cfg, run_dir / "run_config.json")
                run_base_prediction(run_cfg)

    predictor = cfg["predictor"]
    print(
        f"Base grid [{predictor}]: "
        f"{total - skipped} ran, {skipped} hash-validated skips, {total} total"
    )


if __name__ == "__main__":
    main()
