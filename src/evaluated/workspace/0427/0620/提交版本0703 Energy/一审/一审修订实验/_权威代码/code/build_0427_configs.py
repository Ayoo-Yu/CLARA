"""0427 configuration factory.

Generates all JSON configs for the five-dimensional experiment grid:
    lead_time(24) × confidence_level(11) × wind_farm(4) × predictor(4) × method(14+)
"""
from __future__ import annotations

import json
from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

WORKSPACE_ID = "0427"
WORKSPACE_ROOT = Path(os.environ.get("CLARA_WORKDIR", str(Path.cwd())))
CONFIGS_DIR = WORKSPACE_ROOT / "configs"
GENERATED_DIR = CONFIGS_DIR / "generated"
RESULTS_DIR = WORKSPACE_ROOT / "results"

# ---------------------------------------------------------------------------
# Five-dimensional grid constants
# ---------------------------------------------------------------------------

LEAD_TIMES = list(range(1, 25))  # 1h .. 24h

LEAD_TIME_GROUPS = {
    "1h": [1],
    "2h": [2],
    "3-4h": [3, 4],
    "5-6h": [5, 6],
    "7-8h": [7, 8],
    "9-12h": [9, 10, 11, 12],
    "13-24h": list(range(13, 25)),
}

# Alpha = nominal miscoverage rate; confidence = 1 - alpha
ALPHA_LEVELS = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.01]

CONFIDENCE_LABELS = {
    0.90: "10%", 0.80: "20%", 0.70: "30%", 0.60: "40%", 0.50: "50%",
    0.40: "60%", 0.30: "70%", 0.20: "80%", 0.10: "90%", 0.05: "95%", 0.01: "99%",
}

# All quantiles needed across all alpha levels
ALL_QUANTILES = sorted(set(
    [a / 2.0 for a in ALPHA_LEVELS]
    + [1.0 - a / 2.0 for a in ALPHA_LEVELS]
    + [0.5]
))

# ---------------------------------------------------------------------------
# Wind farms
# ---------------------------------------------------------------------------

GEFCOM_DATA_PATHS = {
    f"zone{i}": rf"data/gefcom/gefcom2014_zone{i}_processed.csv"
    for i in range(1, 11)
}

EXTERNAL_DATA_PATHS = {
    "farma": r"data/commercial/FarmA_processed.csv",
    "farmb": r"data/commercial/FarmB_processed.csv",
}

DATA_PATHS = {**GEFCOM_DATA_PATHS, **EXTERNAL_DATA_PATHS}
GEFCOM_ZONE_NAMES = list(GEFCOM_DATA_PATHS.keys())
EXTERNAL_ZONE_NAMES = list(EXTERNAL_DATA_PATHS.keys())
LEGACY_ZONE_NAMES = ["zone1", "zone10", "farma", "farmb"]

# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------

PREDICTORS = {
    "Ridge": {"type": "ridge"},
    "GBR": {"type": "gbr"},
    "MLP": {"type": "mlp"},
    "QRLSTM": {"type": "qrlstm"},
}

# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

STATIC_METHODS = ["SplitCF", "CQR", "LCF"]
DYNAMIC_METHODS = ["ACI", "AgACI", "FACI", "EnbPI"]
EXTENDED_DYNAMIC_METHODS = ["SAOCP", "PID", "SPCI", "NEX", "KOWCPI", "WACI"]
CUSTOM_METHODS = ["Clip", "PWShrink", "Hybrid", "RiskScale"]

METHOD_FAMILY_MAP = {}
for m in STATIC_METHODS:
    METHOD_FAMILY_MAP[m] = "static"
for m in DYNAMIC_METHODS:
    METHOD_FAMILY_MAP[m] = "dynamic"
for m in EXTENDED_DYNAMIC_METHODS:
    METHOD_FAMILY_MAP[m] = "dynamic_extension"
for m in CUSTOM_METHODS:
    METHOD_FAMILY_MAP[m] = "custom"

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

SEEDS = [0, 1, 2]
SPLIT = {"train_ratio": 0.6, "calibration_ratio": 0.2}
ROLLING_WINDOW = 168
MIN_TEST_LENGTH = 336
RAMP_THRESHOLD = 0.12
BOUNDARY_LOW = 0.05
BOUNDARY_HIGH = 0.95


# ---------------------------------------------------------------------------
# Phase method rosters
# ---------------------------------------------------------------------------

PHASE_METHODS = {
    "phase1_core_family": STATIC_METHODS + DYNAMIC_METHODS,
    "phase1_core_family_10zone": STATIC_METHODS + DYNAMIC_METHODS,
    "phase2_dynamic_diagnostics": STATIC_METHODS + DYNAMIC_METHODS,
    "phase3_boundary_conditions": STATIC_METHODS + DYNAMIC_METHODS,
    "phase4_custom_methods": STATIC_METHODS + DYNAMIC_METHODS + CUSTOM_METHODS,
    "phase4b_dynamic_extension": EXTENDED_DYNAMIC_METHODS,
    "phase5_predictor_transfer": STATIC_METHODS + DYNAMIC_METHODS,
}

PHASE_PREDICTORS = {
    "phase1_core_family": list(PREDICTORS.keys()),
    "phase1_core_family_10zone": list(PREDICTORS.keys()),
    "phase2_dynamic_diagnostics": list(PREDICTORS.keys()),
    "phase3_boundary_conditions": ["GBR", "QRLSTM"],
    "phase4_custom_methods": ["GBR", "QRLSTM"],
    "phase4b_dynamic_extension": ["GBR", "QRLSTM"],
    "phase5_predictor_transfer": ["QRLSTM"],
}

PHASE_ZONES = {
    "phase1_core_family": LEGACY_ZONE_NAMES,
    "phase1_core_family_10zone": GEFCOM_ZONE_NAMES,
    "phase2_dynamic_diagnostics": LEGACY_ZONE_NAMES,
    "phase3_boundary_conditions": LEGACY_ZONE_NAMES,
    "phase4_custom_methods": LEGACY_ZONE_NAMES,
    "phase4b_dynamic_extension": LEGACY_ZONE_NAMES,
    "phase5_predictor_transfer": LEGACY_ZONE_NAMES,
}


# ---------------------------------------------------------------------------
# Clipping policy
# ---------------------------------------------------------------------------

CLIPPING_POLICY = {
    "status": "FROZEN_FOR_MAINLINE",
    "order": "construct method interval first; apply final clipping only when clip_final=true",
    "method_rules": {},
}
for m in STATIC_METHODS + DYNAMIC_METHODS + EXTENDED_DYNAMIC_METHODS + CUSTOM_METHODS:
    if m == "SplitCF":
        CLIPPING_POLICY["method_rules"][m] = {"clip_final": False, "reporting": "native"}
    else:
        CLIPPING_POLICY["method_rules"][m] = {"clip_final": True, "reporting": "clipped"}


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def build_phase0_protocol() -> dict:
    return {
        "workspace_id": WORKSPACE_ID,
        "workspace_root": str(WORKSPACE_ROOT),
        "alpha_levels": ALPHA_LEVELS,
        "all_quantiles": ALL_QUANTILES,
        "seeds": SEEDS,
        "split": SPLIT,
        "rolling_window": ROLLING_WINDOW,
        "min_test_length": MIN_TEST_LENGTH,
        "lead_times": LEAD_TIMES,
        "lead_time_groups": LEAD_TIME_GROUPS,
        "ramp_threshold": RAMP_THRESHOLD,
        "boundary_low": BOUNDARY_LOW,
        "boundary_high": BOUNDARY_HIGH,
        "family_groups": {
            "static": STATIC_METHODS,
            "dynamic": DYNAMIC_METHODS,
            "dynamic_extension": EXTENDED_DYNAMIC_METHODS,
            "custom": CUSTOM_METHODS,
        },
        "method_family_map": METHOD_FAMILY_MAP,
        "predictor_groups": PREDICTORS,
        "data_paths": DATA_PATHS,
        "primary_gefcom_zones": GEFCOM_ZONE_NAMES,
        "supplementary_external_zones": EXTERNAL_ZONE_NAMES,
        "clipping_policy": CLIPPING_POLICY,
    }


def _base_run_id(predictor: str, zone: str, seed: int, horizon: int) -> str:
    return f"{predictor}-H{horizon:02d}-{zone}-S{seed}"


def _method_run_id(method: str, predictor: str, zone: str, seed: int,
                   horizon: int, alpha: float) -> str:
    return f"{method}-{predictor}-H{horizon:02d}-A{alpha:.2f}-{zone}-S{seed}"


def build_base_grid_config(phase: str, predictor: str) -> dict:
    zones = PHASE_ZONES.get(phase, LEGACY_ZONE_NAMES)
    return {
        "phase": phase,
        "predictor": predictor,
        "predictor_spec": PREDICTORS[predictor],
        "horizons": LEAD_TIMES,
        "seeds": SEEDS,
        "zones": zones,
        "data_paths": {zone: DATA_PATHS[zone] for zone in zones},
        "split": SPLIT,
        "all_quantiles": ALL_QUANTILES,
        "results_dir": str(RESULTS_DIR / "base_predictions" / phase / predictor),
    }


def build_methods_config(phase: str, predictor: str) -> dict:
    methods = PHASE_METHODS.get(phase, STATIC_METHODS + DYNAMIC_METHODS)
    zones = PHASE_ZONES.get(phase, LEGACY_ZONE_NAMES)
    return {
        "phase": phase,
        "predictor": predictor,
        "methods": methods,
        "horizons": LEAD_TIMES,
        "alpha_levels": ALPHA_LEVELS,
        "seeds": SEEDS,
        "zones": zones,
        "data_paths": {zone: DATA_PATHS[zone] for zone in zones},
        "split": SPLIT,
        "rolling_window": ROLLING_WINDOW,
        "clipping_policy": CLIPPING_POLICY,
        "base_predictions_dir": str(RESULTS_DIR / "base_predictions" / phase / predictor),
        "results_dir": str(RESULTS_DIR / "methods" / phase / predictor),
    }


def build_aggregate_config(phase: str, predictor: str) -> dict:
    zones = PHASE_ZONES.get(phase, LEGACY_ZONE_NAMES)
    return {
        "phase": phase,
        "predictor": predictor,
        "methods": PHASE_METHODS.get(phase, STATIC_METHODS + DYNAMIC_METHODS),
        "horizons": LEAD_TIMES,
        "alpha_levels": ALPHA_LEVELS,
        "seeds": SEEDS,
        "zones": zones,
        "rolling_window": ROLLING_WINDOW,
        "methods_dir": str(RESULTS_DIR / "methods" / phase / predictor),
        "reports_dir": str(RESULTS_DIR / "reports" / phase / predictor),
    }


def build_multi_predictor_aggregate_config(phase: str) -> dict:
    predictors = PHASE_PREDICTORS.get(phase, list(PREDICTORS.keys()))
    zones = PHASE_ZONES.get(phase, LEGACY_ZONE_NAMES)
    return {
        "phase": phase,
        "predictors": predictors,
        "horizons": LEAD_TIMES,
        "alpha_levels": ALPHA_LEVELS,
        "seeds": SEEDS,
        "zones": zones,
        "rolling_window": ROLLING_WINDOW,
        "per_predictor_dirs": {
            p: str(RESULTS_DIR / "reports" / phase / p) for p in predictors
        },
        "reports_dir": str(RESULTS_DIR / "reports" / phase / "combined"),
    }


# ---------------------------------------------------------------------------
# Main: generate all configs
# ---------------------------------------------------------------------------

def build_all() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 0 protocol
    protocol = build_phase0_protocol()
    _save(protocol, CONFIGS_DIR / "phase0_protocol.json")

    phases = list(PHASE_METHODS.keys())

    for phase in phases:
        predictors = PHASE_PREDICTORS.get(phase, list(PREDICTORS.keys()))
        for pred in predictors:
            tag = f"{phase}_{pred}"
            _save(build_base_grid_config(phase, pred),
                  GENERATED_DIR / f"{tag}_base_grid.json")
            _save(build_methods_config(phase, pred),
                  GENERATED_DIR / f"{tag}_methods.json")
            _save(build_aggregate_config(phase, pred),
                  GENERATED_DIR / f"{tag}_aggregate.json")

        _save(build_multi_predictor_aggregate_config(phase),
              GENERATED_DIR / f"{phase}_multi_predictor.json")

    print(f"Configs written to {GENERATED_DIR}")
    n = len(list(GENERATED_DIR.glob("*.json")))
    print(f"Total generated configs: {n}")


def _save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    build_all()
