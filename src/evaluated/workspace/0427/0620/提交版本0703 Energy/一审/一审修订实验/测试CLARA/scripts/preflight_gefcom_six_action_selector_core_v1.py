"""Read-only one-stream preflight for the GEFCom six-action selector core."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
TEST_ROOT = SCRIPT_ROOT.parent
REVISION_ROOT = TEST_ROOT.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from gefcom_six_action_selector_core_v1 import (  # noqa: E402
    price_specs_from_config,
    readonly_small_sample_preflight,
)


DEFAULT_CONFIG = TEST_ROOT / "configs" / "test_clara_gefcom_six_action_full_v1.json"
S03_ROOT = REVISION_ROOT / "03_基础预测与候选区间重建" / "results_raw" / "full_rebuild_v1"
S06_ROOT = (
    REVISION_ROOT
    / "06_基线实现与训练区调参"
    / "results_raw"
    / "nested_source_selection_v1"
)
ENDPOINT_ROOT = (
    TEST_ROOT
    / "results_raw"
    / "gefcom_six_action_price_full_v1"
    / "endpoint_units"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-zone", default="zone1")
    parser.add_argument("--heldout-zone", default="zone10")
    parser.add_argument("--predictor", default="Ridge")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-rows", type=int, default=512)
    parser.add_argument("--source-fact", type=Path)
    parser.add_argument("--endpoint-addon", type=Path)
    parser.add_argument("--event-core", type=Path)
    parser.add_argument("--width-thresholds", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected_packages = dict(config.get("environment", {}).get("packages", {}))
    observed_packages: dict[str, str] = {}
    for package, expected_version in expected_packages.items():
        observed_version = importlib.metadata.version(str(package))
        observed_packages[str(package)] = observed_version
        if observed_version != str(expected_version):
            raise RuntimeError(
                f"冻结Python包版本失配: {package}="
                f"{observed_version}, expected={expected_version}"
            )
    for required_package, required_version in {
        "numba": "0.63.1",
        "llvmlite": "0.46.0",
    }.items():
        if observed_packages.get(required_package) != required_version:
            raise RuntimeError(
                f"正式环境未冻结{required_package}={required_version}"
            )
    prices = price_specs_from_config(config)
    stream_id = (
        f"{args.predictor}-H{int(args.horizon):02d}-{args.source_zone}-S{int(args.seed)}"
    )
    fact = args.source_fact or (
        S06_ROOT
        / "source_facts"
        / f"predictor={args.predictor}"
        / f"zone={args.source_zone}"
        / f"horizon={int(args.horizon):02d}"
        / f"seed={int(args.seed)}"
        / "facts.parquet"
    )
    addon = args.endpoint_addon or (
        ENDPOINT_ROOT
        / f"{args.source_zone}__seed{int(args.seed)}__H{int(args.horizon):02d}"
        / "endpoint_addons.parquet"
    )
    core = args.event_core or (
        S03_ROOT
        / "bundles"
        / args.predictor
        / stream_id
        / "event_core.parquet"
    )
    thresholds = args.width_thresholds or (S06_ROOT / "fold_width_thresholds.parquet")
    result = readonly_small_sample_preflight(
        source_fact_path=fact,
        endpoint_addon_path=addon,
        event_core_path=core,
        width_thresholds_path=thresholds,
        heldout_zone=args.heldout_zone,
        prices=prices,
        sample_rows=args.sample_rows,
    )
    result["environment_packages_verified"] = True
    result["observed_package_versions"] = observed_packages
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    raise SystemExit(0 if result.get("status") == "PASS" else 2)


if __name__ == "__main__":
    main()
