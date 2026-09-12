"""Independent orchestration for the formal GEFCom six-action selector stages.

This runner is deliberately separate from the frozen endpoint producer.  Formal
fit/replay writes require (1) the complete endpoint root seal, (2) an exact runtime
and scientific identity preflight token, and (3) an independently reviewed release
manifest whose file SHA is pinned in the selector-only phase config.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil


TEST_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = TEST_ROOT.parent
PACKAGE_ROOT = REVISION_ROOT.parents[5]
SCRIPT_ROOT = TEST_ROOT / "scripts"
AUTH_CODE = REVISION_ROOT / "_权威代码" / "code"
for directory in (SCRIPT_ROOT, AUTH_CODE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import gefcom_six_action_selector_core_v1 as core  # noqa: E402
import gefcom_six_action_selector_target_loader_v1 as target_loader  # noqa: E402


SCHEMA = "TEST_CLARA_GEFCOM_6A_SELECTOR_RUNNER_V1"
PHASE_CONFIG_PATH = TEST_ROOT / "configs" / "test_clara_gefcom_six_action_selectors_v1.json"
MAIN_CONFIG_PATH = TEST_ROOT / "configs" / "test_clara_gefcom_six_action_full_v1.json"
RUNNER_PATH = Path(__file__).resolve()
CORE_PATH = SCRIPT_ROOT / "gefcom_six_action_selector_core_v1.py"
ARTIFACTS_PATH = SCRIPT_ROOT / "gefcom_six_action_selector_artifacts_v1.py"
TARGET_LOADER_PATH = SCRIPT_ROOT / "gefcom_six_action_selector_target_loader_v1.py"
INDEPENDENT_QA_PATH = SCRIPT_ROOT / "gefcom_six_action_selector_qa_v1.py"
SELECTOR_ROOT = TEST_ROOT / "results_raw" / "gefcom_six_action_price_full_v1" / "selector_v1"
CONTROL_ROOT = SELECTOR_ROOT / "control"
FIT_ROOT = SELECTOR_ROOT / "fit_parents"
REPLAY_ROOT = SELECTOR_ROOT / "replay_parents"
AGGREGATE_ROOT = SELECTOR_ROOT / "aggregate_v1"
QA_ROOT = SELECTOR_ROOT / "qa_v1"
TOKEN_ROOT = CONTROL_ROOT / "preflight_tokens"
PREFLIGHT_ROOT = CONTROL_ROOT / "preflight_runs"
AGGREGATE_SCHEMA = "TEST_CLARA_GEFCOM_6A_COMPACT_AGGREGATE_V1"
QA_SCHEMA = "TEST_CLARA_GEFCOM_6A_INDEPENDENT_QA_V1"
POLICY_MODES = (
    "RESELECT_EACH_RATIO",
    "FIXED_MAIN_RATIO_R05P6179775281",
)
MAIN_PRICE_ID = "R05P6179775281"
RESULT_SCOPE_FIELDS = (
    "policy_mode",
    "evaluation_price_id",
    "policy_price_id",
)
AGGREGATE_OUTPUT_NAMES = (
    "compact_artifact_index",
    "cell_metrics",
    "action_counts",
    "diagnostic_metrics",
    "clara_state_counts",
    "event_conservation",
    "overall_ranking",
    "event_weighted_overall_descriptive",
    "zone_metric_means",
    "zone_design_equal_metrics",
    "stratified_metrics",
    "action_share",
    "diagnostic_zone_metrics",
    "secondary_zone_balanced",
    "secondary_event_weighted_descriptive",
    "clara_support_guardrail_summary",
    "paired_zone_differences",
    "paired_inference",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp__{uuid.uuid4().hex}"
    temporary.write_bytes(canonical_json_bytes(dict(payload)))
    os.replace(temporary, target)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp__{uuid.uuid4().hex}"
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary, target)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp__{uuid.uuid4().hex}"
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, target)


def atomic_text(path: Path, value: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp__{uuid.uuid4().hex}"
    temporary.write_text(str(value), encoding="utf-8", newline="\n")
    os.replace(temporary, target)


def resolved_package_path(relative_path: str) -> Path:
    path = (REVISION_ROOT / str(relative_path)).resolve()
    try:
        path.relative_to(REVISION_ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(f"selector路径逃逸revision root: {relative_path}") from error
    return path


def load_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    if not PHASE_CONFIG_PATH.is_file() or not MAIN_CONFIG_PATH.is_file():
        raise FileNotFoundError("selector phase/main config缺失")
    phase = load_json(PHASE_CONFIG_PATH)
    main = load_json(MAIN_CONFIG_PATH)
    validate_scope_contract(phase, main)
    return phase, main


def validate_scope_contract(phase: Mapping[str, Any], main: Mapping[str, Any]) -> None:
    author = phase.get("author_scope", {})
    scope = phase.get("formal_scope", {})
    implementation = phase.get("implementation", {})
    paths = phase.get("paths", {})
    output_contract = phase.get("output_contract", {})
    expected_scope = {
        "zones": list(core.FORMAL_ZONES),
        "seeds": [0, 1, 2],
        "horizons": list(core.FORMAL_HORIZONS),
        "predictors": list(core.FORMAL_PREDICTORS),
        "coverages": list(core.FORMAL_COVERAGES),
        "actions": list(core.SIX_ACTIONS),
        "methods": list(core.FORMAL_METHODS),
        "price_ids": [row[0] for row in core.FORMAL_PRICE_ROWS],
        "policy_modes": list(POLICY_MODES),
    }
    failures: list[str] = []
    if phase.get("schema") != "TEST_CLARA_GEFCOM_6A_SELECTOR_EXECUTION_PROTOCOL_V1":
        failures.append("phase_schema")
    if (
        author.get("status") != "AUTHOR_APPROVED_B_FROZEN"
        or author.get("training_support_identity")
        != "B_UNIFIED_FIVE_HORIZON_NEW_IDENTITY"
        or not bool(author.get("implementation_authorized"))
        or not str(author.get("scientific_scope_statement") or "")
    ):
        failures.append("author_scope")
    for key, expected in expected_scope.items():
        observed = scope.get(key)
        if key == "coverages":
            observed = [float(value) for value in observed or []]
        elif key in {"seeds", "horizons"}:
            observed = [int(value) for value in observed or []]
        else:
            observed = [str(value) for value in observed or []]
        if observed != expected:
            failures.append(f"scope_{key}")
    expected_counts = {
        "fit_parent_count": 30,
        "replay_parent_count": 30,
        "fit_price_child_count": 150,
        "replay_price_child_count": 300,
        "endpoint_unit_count": 150,
        "endpoint_total_event_count": 23_024_760,
        "event_count_per_method_policy_mode_price": 23_024_760,
        "method_policy_mode_price_row_count": 90,
        "method_event_policy_mode_price_score_count": 2_072_228_400,
    }
    for field, expected in expected_counts.items():
        if int(scope.get(field, -1)) != expected:
            failures.append(f"count_{field}")
    if set(implementation) != {
        "selector_fit",
        "causal_target_loader",
        "global_five_horizon_linucb_replay",
        "compact_streaming_metrics",
        "fixed_main_ratio_deployment_scan",
        "aggregation",
        "independent_qa",
        "full_event_materialization_allowed",
        "nonformal_cli_escape_hatch_allowed",
    }:
        failures.append("implementation_fields")
    if (
        not all(
            bool(implementation.get(field))
            for field in (
                "selector_fit",
                "causal_target_loader",
                "global_five_horizon_linucb_replay",
                "compact_streaming_metrics",
                "fixed_main_ratio_deployment_scan",
                "aggregation",
                "independent_qa",
            )
        )
        or bool(implementation.get("full_event_materialization_allowed"))
        or bool(implementation.get("nonformal_cli_escape_hatch_allowed"))
    ):
        failures.append("implementation_gate")
    expected_streaming_memory_contract = {
        "production_aggregate_pass": "ONE_PARENT_AT_A_TIME",
        "independent_qa_pass": "SECOND_INDEPENDENT_ONE_PARENT_AT_A_TIME_SCAN",
        "maximum_retained_raw_parent_count": 1,
        "maximum_retained_raw_child_count": 10,
        "paired_blocks_rule": (
            "REDUCE_EACH_CHILD_IMMEDIATELY_TO_ZONE_METHOD_SCOPE_TOTALS; "
            "NEVER_CONCAT_RAW_PAIRED_BLOCKS_ACROSS_CHILDREN"
        ),
        "rss_audit_required": True,
        "full_300_child_retention_allowed": False,
    }
    if (
        output_contract.get("mode") != "COMPACT_ONLY"
        or output_contract.get("streaming_memory_contract")
        != expected_streaming_memory_contract
    ):
        failures.append("streaming_memory_contract")
    main_training = main.get("training_support_identity", {})
    if (
        main.get("schema") != "TEST_CLARA_GEFCOM_SIX_ACTION_PRICE_FULL_PROTOCOL_V1"
        or main_training.get("status") != "AUTHOR_APPROVED_B_FROZEN"
        or main_training.get("chosen_identity")
        != "B_UNIFIED_FIVE_HORIZON_NEW_IDENTITY"
        or not bool(main_training.get("author_approved"))
        or not bool(main_training.get("scientific_execution_allowed"))
    ):
        failures.append("main_training_identity")
    main_scope = main.get("scope", {})
    comparisons = {
        "zones": "zones",
        "seeds": "seeds",
        "horizons": "horizons",
        "predictors": "predictors",
        "coverages": "target_coverages",
        "actions": "actions",
        "methods": "methods",
    }
    for phase_key, main_key in comparisons.items():
        if list(scope.get(phase_key, [])) != list(main_scope.get(main_key, [])):
            failures.append(f"main_scope_{phase_key}")
    if [row["price_id"] for row in main.get("price_grid", [])] != expected_scope[
        "price_ids"
    ]:
        failures.append("main_price_ids")
    path_expectations = {
        "main_protocol_relative_path": MAIN_CONFIG_PATH,
        "selector_core_relative_path": CORE_PATH,
        "selector_artifacts_relative_path": ARTIFACTS_PATH,
        "target_loader_relative_path": TARGET_LOADER_PATH,
        "independent_qa_relative_path": INDEPENDENT_QA_PATH,
        "runner_relative_path": RUNNER_PATH,
    }
    for field, expected in path_expectations.items():
        if resolved_package_path(str(paths.get(field, ""))) != expected.resolve():
            failures.append(f"path_{field}")
    if str(paths.get("main_protocol_sha256")) != sha256_file(MAIN_CONFIG_PATH):
        failures.append("main_protocol_sha")
    if failures:
        raise RuntimeError(f"selector冻结范围合同失配: {sorted(set(failures))}")


def statistics_contract_audit() -> dict[str, Any]:
    """Freeze the inherited equal-zone inference contract for this 8-way family."""

    contracts = core.load_frozen_contracts()
    observed = dict(contracts.protocol.get("statistics_contract", {}))
    required = {
        "gefcom_primary_unit": "heldout_zone_paired_difference",
        "zone_aggregation": (
            "average_design_axes_within_zone_then_equal_weight_ten_zones"
        ),
        "confidence_interval": (
            "five_thousand_replicate_paired_zone_bootstrap_percentile_95"
        ),
        "zone_bootstrap_seed": 2026082802,
        "paired_test": "exact_sign_flip_over_all_1024_sign_configurations",
        "multiple_comparison": (
            "holm_familywise_alpha_0.05_for_predeclared_primary_comparators"
        ),
        "minimum_reported_p_value": (
            "one_over_number_of_exact_or_resampled_configurations_plus_one"
        ),
    }
    if any(observed.get(key) != value for key, value in required.items()):
        raise RuntimeError("冻结GEFCom统计合同漂移")
    if str(contracts.protocol_sha256) != (
        "194f9c7d52efe7f59c91e4c4e8bba0a17bb3712240dfb69832ce5cd875ecc22b"
    ):
        raise RuntimeError("冻结因果事件协议SHA漂移")
    payload = {
        "schema": "TEST_CLARA_GEFCOM_6A_STATISTICS_CONTRACT_V1",
        "status": "PASS",
        "causal_event_protocol_path": str(Path(contracts.protocol_path).resolve()),
        "causal_event_protocol_sha256": str(contracts.protocol_sha256),
        "source_statistics_contract": observed,
        "bootstrap_replicates": 5000,
        "bootstrap_seed": 2026082802,
        "exact_sign_configuration_count": 1024,
        "familywise_alpha": 0.05,
        "target_method": "CLARA_6A",
        "comparators_in_frozen_order": [
            method for method in core.FORMAL_METHODS if method != "CLARA_6A"
        ],
        "coverage_scope": list(core.FORMAL_COVERAGES),
        "horizon_scope": list(core.FORMAL_HORIZONS),
    }
    return {**payload, "statistics_contract_sha256": canonical_sha256(payload)}


def phase_scientific_contract(phase: Mapping[str, Any]) -> dict[str, Any]:
    release = dict(phase["reviewer_release"])
    release.pop("manifest_sha256", None)
    release.pop("status", None)
    payload = {
        "schema": "TEST_CLARA_GEFCOM_6A_SELECTOR_SCIENTIFIC_CONTRACT_V1",
        "author_scope": phase["author_scope"],
        "formal_scope": phase["formal_scope"],
        "implementation": phase["implementation"],
        "checkpoint_contract": phase["checkpoint_contract"],
        "output_contract": phase["output_contract"],
        "environment": phase["environment"],
        "paths_without_release_sha": {
            key: value
            for key, value in phase["paths"].items()
            if key not in {"output_relative_path"}
        },
        "statistics_contract": statistics_contract_audit(),
        "reviewer_release_contract": release,
    }
    return {**payload, "scientific_contract_sha256": canonical_sha256(payload)}


def index_contract_audit(
    phase: Mapping[str, Any], *, contract_override: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], pd.DataFrame]:
    paths = phase["paths"]
    path = resolved_package_path(paths["landscape_index_contract_relative_path"])
    if not path.is_file() or sha256_file(path) != str(
        paths["landscape_index_contract_sha256"]
    ):
        raise RuntimeError("landscape index contract路径/SHA失配")
    contract = dict(contract_override) if contract_override is not None else load_json(path)
    expected_scalar = {
        "contract_id": paths["landscape_index_contract_id"],
        "contract_version": paths["landscape_index_contract_version"],
        "status": paths["landscape_index_contract_status"],
        "parent_execution_contract_sha256": paths[
            "landscape_index_parent_execution_contract_sha256"
        ],
        "causal_event_protocol_sha256": paths[
            "landscape_index_causal_event_protocol_sha256"
        ],
        "source_fact_root_manifest_sha256": paths[
            "landscape_index_source_fact_root_manifest_sha256"
        ],
        "source_fact_ledger_sha256": paths[
            "landscape_index_source_fact_ledger_sha256"
        ],
        "base_state_space_size": int(paths["base_state_space_size"]),
    }
    if any(contract.get(key) != value for key, value in expected_scalar.items()):
        raise RuntimeError("landscape index schema/status/parent身份失配")
    expected_vocabulary = {
        "predictor": list(core.FORMAL_PREDICTORS),
        "horizon_group": ["H01", "H02_H03", "H04_H06", "H07_H12", "H13_H24"],
        "target_coverage": list(core.FORMAL_COVERAGES),
        "ramp_state": ["ordinary", "ramp"],
        "rolling_state": [
            "undercoverage_pressure",
            "overcoverage_pressure",
            "volatile",
            "stable",
            "cold_start",
        ],
    }
    if contract.get("vocabularies") != expected_vocabulary:
        raise RuntimeError("landscape index vocabulary漂移")
    if contract.get("base_state_fields_in_order") != [
        "predictor",
        "horizon_group",
        "target_coverage",
        "ramp_state",
        "rolling_state",
    ]:
        raise RuntimeError("landscape index state field顺序漂移")
    contracts = core.six_action_contracts(core.load_frozen_contracts())
    states = core.state_universe_from_contract(contracts, contract)
    state_sha = core._state_universe_content_sha256(states)
    if (
        len(states) != int(paths["full_six_axis_state_space_size"])
        or state_sha != str(paths["full_six_axis_state_universe_sha256"])
        or states["full_state_code"].tolist() != list(range(len(states)))
    ):
        raise RuntimeError("正式6600状态全集计数/内容SHA漂移")
    audit = {
        "schema": "TEST_CLARA_GEFCOM_6A_INDEX_CONTRACT_AUDIT_V1",
        "status": "PASS",
        "path": str(path),
        "file_sha256": sha256_file(path),
        "contract_content_sha256": canonical_sha256(contract),
        "base_state_space_size": int(contract["base_state_space_size"]),
        "full_state_space_size": len(states),
        "state_universe_sha256": state_sha,
        "parent_hashes": {
            key: contract[key]
            for key in (
                "parent_execution_contract_sha256",
                "causal_event_protocol_sha256",
                "source_fact_root_manifest_sha256",
                "source_fact_ledger_sha256",
            )
        },
    }
    return audit, states


def environment_audit(phase: Mapping[str, Any]) -> dict[str, Any]:
    expected = phase["environment"]
    executable = Path(sys.executable).resolve()
    expected_executable = Path(expected["formal_python_executable"]).resolve()
    package_map = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "scikit-learn": "scikit-learn",
        "pyarrow": "pyarrow",
        "psutil": "psutil",
        "numba": "numba",
        "llvmlite": "llvmlite",
    }
    observed_packages = {
        label: importlib.metadata.version(distribution)
        for label, distribution in package_map.items()
    }
    expected_environment = {
        str(key): str(value)
        for key, value in expected["required_process_environment"].items()
    }
    observed_environment = {key: os.environ.get(key) for key in expected_environment}
    failures = []
    if executable != expected_executable:
        failures.append("python_executable")
    if platform.python_version() != str(expected["python"]):
        failures.append("python_version")
    if list(expected.get("formal_python_flags", [])) != ["-s"] or not bool(
        sys.flags.no_user_site
    ):
        failures.append("python_no_user_site")
    if observed_environment != expected_environment:
        failures.append("process_environment")
    if observed_packages != {
        str(key): str(value) for key, value in expected["packages"].items()
    }:
        failures.append("package_versions")
    return {
        "schema": "TEST_CLARA_GEFCOM_6A_SELECTOR_ENVIRONMENT_AUDIT_V1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "python_executable": str(executable),
        "python_version": platform.python_version(),
        "no_user_site": bool(sys.flags.no_user_site),
        "process_environment": observed_environment,
        "packages": observed_packages,
    }


def _artifacts_module():
    module = importlib.import_module("gefcom_six_action_selector_artifacts_v1")
    if Path(module.__file__).resolve() != ARTIFACTS_PATH.resolve():
        raise RuntimeError("selector artifacts导入路径漂移")
    return module


def _independent_qa_module():
    module = importlib.import_module("gefcom_six_action_selector_qa_v1")
    if Path(module.__file__).resolve() != INDEPENDENT_QA_PATH.resolve():
        raise RuntimeError("independent QA导入路径漂移")
    if str(module.module_sha256()) != sha256_file(INDEPENDENT_QA_PATH):
        raise RuntimeError("independent QA导入代码与磁盘SHA失配")
    return module


def code_identity() -> dict[str, Any]:
    paths = {
        "runner": RUNNER_PATH,
        "selector_core": CORE_PATH,
        "selector_artifacts": ARTIFACTS_PATH,
        "target_loader": TARGET_LOADER_PATH,
        "independent_qa": INDEPENDENT_QA_PATH,
    }
    missing = {name: str(path) for name, path in paths.items() if not path.is_file()}
    if missing:
        raise FileNotFoundError(f"selector代码闭包缺失: {missing}")
    module_files = {name: sha256_file(path) for name, path in paths.items()}
    dependency_closure = core.algorithm_dependency_closure()
    payload = {
        "schema": "TEST_CLARA_GEFCOM_6A_SELECTOR_CODE_IDENTITY_V1",
        "module_file_sha256": module_files,
        "core_algorithm_dependency_closure": dependency_closure,
        "action_library_sha256": core.action_library_sha256(),
        "global_replay_scope_sha256": core.global_replay_scope_sha256(),
    }
    return {**payload, "code_identity_sha256": canonical_sha256(payload)}


def endpoint_root_audit(phase: Mapping[str, Any]) -> dict[str, Any]:
    path = resolved_package_path(phase["paths"]["endpoint_root_manifest_relative_path"])
    if not path.is_file():
        return {
            "schema": "TEST_CLARA_GEFCOM_6A_SELECTOR_ENDPOINT_ROOT_AUDIT_V1",
            "status": "BLOCKED",
            "reason": "ENDPOINT_ROOT_SEAL_MISSING",
            "path": str(path),
        }
    root_sha = sha256_file(path)
    manifest, units = core.validate_endpoint_root_seal(
        path, expected_endpoint_root_sha256=root_sha
    )
    if len(units) != 150 or int(manifest["total_event_count"]) != 23_024_760:
        raise RuntimeError("selector endpoint root未闭合150/23,024,760")
    return {
        "schema": "TEST_CLARA_GEFCOM_6A_SELECTOR_ENDPOINT_ROOT_AUDIT_V1",
        "status": "PASS",
        "path": str(path),
        "sha256": root_sha,
        "unit_count": len(units),
        "total_event_count": int(manifest["total_event_count"]),
        "root_identity_sha256": canonical_sha256(manifest["identities"]),
    }


def reviewer_release_audit(
    phase: Mapping[str, Any],
    *,
    scientific_contract: Mapping[str, Any],
    code: Mapping[str, Any],
    endpoint_root: Mapping[str, Any],
) -> dict[str, Any]:
    contract = phase["reviewer_release"]
    path = resolved_package_path(contract["relative_path"])
    expected_sha = contract.get("manifest_sha256")
    if contract.get("status") != "PASS" or not is_sha256(expected_sha):
        return {
            "schema": "TEST_CLARA_GEFCOM_6A_SELECTOR_REVIEW_RELEASE_AUDIT_V1",
            "status": "BLOCKED",
            "reason": "REVIEWER_RELEASE_NOT_PINNED_IN_PHASE_CONFIG",
            "path": str(path),
        }
    if not path.is_file() or sha256_file(path) != str(expected_sha):
        raise RuntimeError("reviewer release路径/配置pin SHA失配")
    release = load_json(path)
    required = {
        "schema": "TEST_CLARA_GEFCOM_6A_SELECTOR_REVIEW_RELEASE_V1",
        "status": "PASS",
        "verdict": str(contract["required_verdict"]),
        "reviewer_role": str(contract["required_reviewer_role"]),
        "scientific_contract_sha256": scientific_contract[
            "scientific_contract_sha256"
        ],
        "code_identity_sha256": code["code_identity_sha256"],
        "endpoint_root_sha256": endpoint_root.get("sha256"),
    }
    if any(release.get(key) != value for key, value in required.items()):
        raise RuntimeError("reviewer release内容与当前科学身份失配")
    if not str(release.get("reviewed_at_utc") or "") or not str(
        release.get("review_summary") or ""
    ):
        raise RuntimeError("reviewer release缺时间/审查摘要")
    return {
        "schema": "TEST_CLARA_GEFCOM_6A_SELECTOR_REVIEW_RELEASE_AUDIT_V1",
        "status": "PASS",
        "path": str(path),
        "sha256": str(expected_sha),
        "release": release,
    }


def execution_identity(
    phase: Mapping[str, Any],
    main: Mapping[str, Any],
    *,
    require_release: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scientific = phase_scientific_contract(phase)
    index_audit, _ = index_contract_audit(phase)
    code = code_identity()
    environment = environment_audit(phase)
    endpoint = endpoint_root_audit(phase)
    release = reviewer_release_audit(
        phase,
        scientific_contract=scientific,
        code=code,
        endpoint_root=endpoint,
    )
    audits = {
        "environment": environment,
        "endpoint_root": endpoint,
        "reviewer_release": release,
        "index_contract": index_audit,
    }
    failures = [
        name
        for name, audit in audits.items()
        if audit.get("status") != "PASS"
        and (name != "reviewer_release" or require_release)
    ]
    if failures:
        raise RuntimeError(f"selector execution identity未就绪: {failures}")
    payload = {
        "schema": "TEST_CLARA_GEFCOM_6A_SELECTOR_EXECUTION_IDENTITY_V1",
        "phase_config_sha256": sha256_file(PHASE_CONFIG_PATH),
        "main_config_sha256": sha256_file(MAIN_CONFIG_PATH),
        "scientific_contract_sha256": scientific["scientific_contract_sha256"],
        "code_identity": code,
        "index_contract_audit": index_audit,
        "environment_audit": environment,
        "endpoint_root_audit": endpoint,
        "reviewer_release_audit": release,
        "v4_adaptive_contract": core.clara_v4_adaptive_contract(),
        "action_library_sha256": core.action_library_sha256(),
        "global_replay_scope_sha256": core.global_replay_scope_sha256(),
        "price_grid_sha256": canonical_sha256(main["price_grid"]),
    }
    return {**payload, "execution_identity_sha256": canonical_sha256(payload)}, audits


def execution_identity_payload(identity: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        str(key): value
        for key, value in identity.items()
        if str(key) != "execution_identity_sha256"
    }
    expected = str(identity.get("execution_identity_sha256", ""))
    if not is_sha256(expected) or canonical_sha256(payload) != expected:
        raise RuntimeError("selector execution identity payload/SHA失配")
    return payload


def fit_parent_id(zone: str, seed: int) -> str:
    return f"{zone}__seed{int(seed)}"


def fit_parent_dir(zone: str, seed: int) -> Path:
    return FIT_ROOT / fit_parent_id(zone, seed)


def replay_parent_dir(zone: str, seed: int) -> Path:
    return REPLAY_ROOT / fit_parent_id(zone, seed)


def replay_child_id(policy_mode: str, evaluation_price_id: str) -> str:
    mode = str(policy_mode)
    price_id = str(evaluation_price_id)
    if mode not in POLICY_MODES or price_id not in {
        row[0] for row in core.FORMAL_PRICE_ROWS
    }:
        raise RuntimeError("replay child mode/evaluation price越出冻结范围")
    return f"{mode}__{price_id}"


def selector_plan_rows(phase: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scope = phase["formal_scope"]
    parents: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    for zone in scope["zones"]:
        for seed in scope["seeds"]:
            parent_id = fit_parent_id(zone, seed)
            parents.append(
                {
                    "task_id": f"fit::{parent_id}",
                    "stage": "fit",
                    "zone": zone,
                    "seed": seed,
                    "dependency": "endpoint_root_seal",
                    "expected_price_children": 5,
                }
            )
            parents.append(
                {
                    "task_id": f"replay::{parent_id}",
                    "stage": "replay",
                    "zone": zone,
                    "seed": seed,
                    "dependency": f"fit::{parent_id}+endpoint_root_seal",
                    "expected_price_children": 10,
                }
            )
            for price_id in scope["price_ids"]:
                children.append(
                    {
                        "child_id": f"fit::{parent_id}::{price_id}",
                        "stage": "fit",
                        "zone": zone,
                        "seed": seed,
                        "price_id": price_id,
                        "dependency": "endpoint_root_seal+compact_source_identity",
                    }
                )
            for policy_mode in POLICY_MODES:
                for evaluation_price_id in scope["price_ids"]:
                    policy_price_id = (
                        evaluation_price_id
                        if policy_mode == "RESELECT_EACH_RATIO"
                        else MAIN_PRICE_ID
                    )
                    children.append(
                        {
                            "child_id": (
                                f"replay::{parent_id}::"
                                f"{replay_child_id(policy_mode, evaluation_price_id)}"
                            ),
                            "stage": "replay",
                            "zone": zone,
                            "seed": seed,
                            "policy_mode": policy_mode,
                            "evaluation_price_id": evaluation_price_id,
                            "policy_price_id": policy_price_id,
                            "dependency": (
                                f"fit::{parent_id}::{policy_price_id}+endpoint_root_seal+"
                                "frozen_linucb_configuration"
                            ),
                        }
                    )
    parents.extend(
        [
            {
                "task_id": "aggregate::global",
                "stage": "aggregate",
                "zone": "ALL",
                "seed": -1,
                "dependency": "30_replay_parent_seals",
                "expected_price_children": 300,
            },
            {
                "task_id": "qa::global",
                "stage": "qa",
                "zone": "ALL",
                "seed": -1,
                "dependency": "aggregate_seal+30_replay_parent_seals",
                "expected_price_children": 300,
            },
        ]
    )
    parent_frame = pd.DataFrame(parents)
    child_frame = pd.DataFrame(children)
    if len(parent_frame) != 62 or len(child_frame) != 450:
        raise RuntimeError("selector计划必须精确为62 parent（含aggregate/QA）及450 children")
    return parent_frame, child_frame


def write_plan() -> dict[str, Any]:
    phase, main = load_configs()
    identity, audits = execution_identity(phase, main, require_release=False)
    parents, children = selector_plan_rows(phase)
    revision = identity["execution_identity_sha256"]
    directory = CONTROL_ROOT / "plan_revisions" / revision
    manifest_path = directory / "manifest.json"
    payload = {
        "schema": "TEST_CLARA_GEFCOM_6A_SELECTOR_PLAN_V1",
        "status": "PASS" if audits["reviewer_release"]["status"] == "PASS" else "BLOCKED",
        "scientific_execution_allowed": audits["reviewer_release"]["status"] == "PASS",
        "blocked_reason": (
            None
            if audits["reviewer_release"]["status"] == "PASS"
            else audits["reviewer_release"].get("reason")
        ),
        "execution_identity": identity,
        "parent_task_count": len(parents),
        "price_child_count": len(children),
        "fit_parent_count": 30,
        "replay_parent_count": 30,
        "fit_price_child_count": 150,
        "replay_price_child_count": 300,
        "parent_ledger_sha256": None,
        "child_ledger_sha256": None,
        "generated_at_utc": utc_now(),
    }
    if manifest_path.exists():
        existing = load_json(manifest_path)
        for field in (
            "execution_identity",
            "parent_task_count",
            "price_child_count",
            "fit_parent_count",
            "replay_parent_count",
            "fit_price_child_count",
            "replay_price_child_count",
        ):
            if existing.get(field) != payload.get(field):
                raise RuntimeError("既有selector计划revision身份冲突")
        return {**existing, "manifest_path": str(manifest_path), "resumed": True}
    directory.mkdir(parents=True, exist_ok=False)
    atomic_csv(directory / "parent_ledger.csv", parents)
    atomic_csv(directory / "price_child_ledger.csv", children)
    payload["parent_ledger_sha256"] = sha256_file(directory / "parent_ledger.csv")
    payload["child_ledger_sha256"] = sha256_file(directory / "price_child_ledger.csv")
    atomic_json(manifest_path, payload)
    return {**payload, "manifest_path": str(manifest_path), "resumed": False}


def issue_preflight_token(report: Mapping[str, Any], report_path: Path) -> dict[str, Any]:
    official_report_path = Path(report_path).resolve()
    if (
        official_report_path.name != "manifest.json"
        or official_report_path.parent.parent.resolve() != PREFLIGHT_ROOT.resolve()
        or not official_report_path.is_file()
    ):
        raise RuntimeError("selector token只能由官方preflight report签发")
    stored_report = load_json(official_report_path)
    expected_report_fields = {
        "schema",
        "status",
        "scientific_execution_allowed",
        "require_reviewer_release",
        "failures",
        "execution_identity",
        "audits",
        "generated_at_utc",
    }
    if (
        set(stored_report) != expected_report_fields
        or stored_report.get("schema")
        != "TEST_CLARA_GEFCOM_6A_SELECTOR_PREFLIGHT_V1"
        or stored_report.get("status") != "PASS"
        or not bool(stored_report.get("scientific_execution_allowed"))
        or not bool(stored_report.get("require_reviewer_release"))
        or stored_report.get("failures") != []
        or stored_report.get("execution_identity") != report.get("execution_identity")
    ):
        raise RuntimeError("selector token只能由exact formal PASS report签发")
    if report.get("status") != "PASS" or not report.get("scientific_execution_allowed"):
        raise RuntimeError("只有exact PASS且review release通过的preflight可签token")
    payload = {
        "schema": "TEST_CLARA_GEFCOM_6A_SELECTOR_PREFLIGHT_TOKEN_V1",
        "status": "PASS",
        "execution_identity": report["execution_identity"],
        "execution_identity_sha256": report["execution_identity"][
            "execution_identity_sha256"
        ],
        "preflight_report_path": str(official_report_path),
        "preflight_report_sha256": sha256_file(official_report_path),
        "issued_at_utc": utc_now(),
    }
    token_id = canonical_sha256(payload)
    directory = TOKEN_ROOT / token_id
    path = directory / "manifest.json"
    if path.exists():
        existing = load_json(path)
        if existing != payload:
            raise RuntimeError("selector token ID内容冲突")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        atomic_json(path, payload)
    return {
        "token_id": token_id,
        "token_path": str(path),
        "token_sha256": sha256_file(path),
    }


def preflight(*, write_report: bool, require_release: bool = True) -> dict[str, Any]:
    phase, main = load_configs()
    failures: list[str] = []
    identity: dict[str, Any] | None = None
    audits: dict[str, Any] = {}
    try:
        identity, audits = execution_identity(
            phase, main, require_release=require_release
        )
    except Exception as error:
        failures.append(f"{type(error).__name__}: {error}")
        # Produce actionable status without authorizing a token.
        for name, function in (
            ("environment", lambda: environment_audit(phase)),
            ("endpoint_root", lambda: endpoint_root_audit(phase)),
        ):
            try:
                audits[name] = function()
            except Exception as current:
                audits[name] = {"status": "FAIL", "error": str(current)}
    report = {
        "schema": "TEST_CLARA_GEFCOM_6A_SELECTOR_PREFLIGHT_V1",
        "status": "PASS" if not failures else "BLOCKED",
        "scientific_execution_allowed": not failures and require_release,
        "require_reviewer_release": bool(require_release),
        "failures": failures,
        "execution_identity": identity,
        "audits": audits,
        "generated_at_utc": utc_now(),
    }
    if write_report:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = PREFLIGHT_ROOT / timestamp / "manifest.json"
        atomic_json(path, report)
        report["report_path"] = str(path)
        report["report_sha256"] = sha256_file(path)
        if report["status"] == "PASS" and report["scientific_execution_allowed"]:
            report["token"] = issue_preflight_token(report, path)
    return report


def validate_preflight_token(
    token_path: Path,
    *,
    expected_token_sha256: str,
    zone: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    path = Path(token_path).resolve()
    if (
        path.name != "manifest.json"
        or path.parent.parent.resolve() != TOKEN_ROOT.resolve()
        or not path.is_file()
        or not is_sha256(expected_token_sha256)
        or sha256_file(path) != str(expected_token_sha256)
    ):
        raise RuntimeError("selector preflight token路径/文件SHA失配")
    token = load_json(path)
    expected_token_fields = {
        "schema",
        "status",
        "execution_identity",
        "execution_identity_sha256",
        "preflight_report_path",
        "preflight_report_sha256",
        "issued_at_utc",
    }
    if (
        set(token) != expected_token_fields
        or token.get("schema")
        != "TEST_CLARA_GEFCOM_6A_SELECTOR_PREFLIGHT_TOKEN_V1"
        or token.get("status") != "PASS"
    ):
        raise RuntimeError("selector preflight token schema/status失配")
    token_id = canonical_sha256(token)
    if path.parent.name != token_id:
        raise RuntimeError("selector token目录ID与canonical payload SHA失配")
    report_path = Path(token["preflight_report_path"]).resolve()
    if (
        report_path.name != "manifest.json"
        or report_path.parent.parent.resolve() != PREFLIGHT_ROOT.resolve()
        or not report_path.is_file()
        or not is_sha256(token["preflight_report_sha256"])
        or sha256_file(report_path) != str(token["preflight_report_sha256"])
    ):
        raise RuntimeError("selector token绑定的preflight report失效")
    report = load_json(report_path)
    expected_report_fields = {
        "schema",
        "status",
        "scientific_execution_allowed",
        "require_reviewer_release",
        "failures",
        "execution_identity",
        "audits",
        "generated_at_utc",
    }
    if (
        set(report) != expected_report_fields
        or report.get("schema") != "TEST_CLARA_GEFCOM_6A_SELECTOR_PREFLIGHT_V1"
        or report.get("status") != "PASS"
        or not bool(report.get("scientific_execution_allowed"))
        or not bool(report.get("require_reviewer_release"))
        or report.get("failures") != []
        or report.get("execution_identity") != token.get("execution_identity")
        or str(report.get("execution_identity", {}).get("execution_identity_sha256"))
        != str(token.get("execution_identity_sha256"))
    ):
        raise RuntimeError("selector token绑定report未授权科学执行")
    phase, main = load_configs()
    current, _ = execution_identity(phase, main, require_release=True)
    if (
        current != token.get("execution_identity")
        or current["execution_identity_sha256"]
        != token.get("execution_identity_sha256")
    ):
        raise RuntimeError("selector token与当前代码/输入/环境/root/release身份漂移")
    if zone is not None and str(zone) not in core.FORMAL_ZONES:
        raise RuntimeError("selector worker zone越出冻结范围")
    if seed is not None and int(seed) not in range(3):
        raise RuntimeError("selector worker seed越出冻结范围")
    return token


def _temporary_entries(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(
        child.name for child in path.iterdir() if ".tmp__" in child.name
    )


def selector_status() -> dict[str, Any]:
    phase, main = load_configs()
    prices = set(phase["formal_scope"]["price_ids"])
    replay_children_expected = {
        replay_child_id(mode, price_id)
        for mode in POLICY_MODES
        for price_id in phase["formal_scope"]["price_ids"]
    }
    execution: dict[str, Any] | None = None
    validation_gate_error = ""
    try:
        execution, _ = execution_identity(phase, main, require_release=True)
    except Exception as error:  # status reports the fail-closed gate; it never authorizes.
        validation_gate_error = f"{type(error).__name__}: {error}"
    artifacts = _artifacts_module() if execution is not None else None
    endpoint_root_sha = (
        str(execution["endpoint_root_audit"]["sha256"])
        if execution is not None
        else ""
    )
    execution_sha = (
        str(execution["execution_identity_sha256"])
        if execution is not None
        else ""
    )
    rows: list[dict[str, Any]] = []
    for stage, root, seal_name in (
        ("fit", FIT_ROOT, "fit_parent_manifest.json"),
        ("replay", REPLAY_ROOT, "replay_parent_manifest.json"),
    ):
        for zone in core.FORMAL_ZONES:
            for seed in range(3):
                parent = root / fit_parent_id(zone, seed)
                children = parent / "children"
                child_entries = (
                    {
                        child.name: child
                        for child in children.iterdir()
                        if ".tmp__" not in child.name
                    }
                    if children.is_dir()
                    else {}
                )
                observed = set(child_entries)
                expected_children = (
                    prices if stage == "fit" else replay_children_expected
                )
                temporary = sorted(
                    {
                        str(path.relative_to(parent))
                        for path in parent.rglob("*")
                        if ".tmp__" in path.name
                    }
                ) if parent.is_dir() else []
                extra = sorted(observed - expected_children)
                missing = sorted(expected_children - observed)
                invalid_children = sorted(
                    price_id
                    for price_id in observed & expected_children
                    if (
                        not child_entries[price_id].is_dir()
                        or not (child_entries[price_id] / "manifest.json").is_file()
                    )
                )
                allowed_parent_entries = {"children", seal_name}
                if stage == "replay":
                    allowed_parent_entries.add("target_input_manifest.json")
                parent_extra = (
                    sorted(
                        child.name
                        for child in parent.iterdir()
                        if ".tmp__" not in child.name
                        and child.name not in allowed_parent_entries
                    )
                    if parent.is_dir()
                    else []
                )
                sealed = (parent / seal_name).is_file()
                obvious_block = bool(
                    temporary
                    or extra
                    or invalid_children
                    or parent_extra
                    or (sealed and missing)
                    or (
                        stage == "replay"
                        and bool(observed)
                        and not (parent / "target_input_manifest.json").is_file()
                    )
                )
                status = (
                    "BLOCKED"
                    if obvious_block
                    else "SEALED_UNVALIDATED"
                    if sealed and not missing
                    else "PARTIAL"
                    if observed
                    else "MISSING"
                )
                validation_error = ""
                if status == "SEALED_UNVALIDATED" and artifacts is not None:
                    try:
                        if stage == "fit":
                            artifacts.load_fit_parent(
                                parent / seal_name,
                                expected_zone=zone,
                                expected_seed=seed,
                                expected_endpoint_root_sha256=endpoint_root_sha,
                                expected_execution_identity_sha256=execution_sha,
                                formal_identity=True,
                            )
                        else:
                            target_manifest_path = parent / "target_input_manifest.json"
                            artifacts.load_replay_parent(
                                parent / seal_name,
                                expected_zone=zone,
                                expected_seed=seed,
                                expected_fit_parent_manifest_path=(
                                    FIT_ROOT
                                    / fit_parent_id(zone, seed)
                                    / "fit_parent_manifest.json"
                                ),
                                expected_endpoint_root_sha256=endpoint_root_sha,
                                expected_target_input_manifest_path=target_manifest_path,
                                expected_target_input_manifest_sha256=sha256_file(
                                    target_manifest_path
                                ),
                                expected_execution_identity_sha256=execution_sha,
                                formal_identity=True,
                            )
                        status = "VALIDATED_COMPLETE"
                    except Exception as error:
                        status = "BLOCKED"
                        validation_error = f"{type(error).__name__}: {error}"
                rows.append(
                    {
                        "stage": stage,
                        "zone": zone,
                        "seed": seed,
                        "status": status,
                        "sealed": sealed,
                        "complete_child_count": len(observed & expected_children),
                        "missing_prices": missing,
                        "extra_children": extra,
                        "temporary_entries": temporary,
                        "invalid_children": invalid_children,
                        "extra_parent_entries": parent_extra,
                        "validation_error": validation_error,
                    }
                )
    fit_validated = sum(
        row["status"] == "VALIDATED_COMPLETE" and row["stage"] == "fit"
        for row in rows
    )
    replay_validated = sum(
        row["status"] == "VALIDATED_COMPLETE" and row["stage"] == "replay"
        for row in rows
    )
    aggregate_sealed = (AGGREGATE_ROOT / "manifest.json").is_file()
    qa_sealed = (QA_ROOT / "manifest.json").is_file()
    aggregate_validated = False
    qa_validated = False
    aggregate_error = ""
    qa_error = ""
    aggregate_blocked = (
        (AGGREGATE_ROOT.exists() and not aggregate_sealed)
        or bool(
            list(
                AGGREGATE_ROOT.parent.glob(
                    f".{AGGREGATE_ROOT.name}.tmp__*"
                )
            )
        )
    )
    qa_blocked = (
        (QA_ROOT.exists() and not qa_sealed)
        or bool(list(QA_ROOT.parent.glob(f".{QA_ROOT.name}.tmp__*")))
    )
    if execution is not None and fit_validated == 30 and replay_validated == 30:
        if aggregate_sealed:
            try:
                load_aggregate_root(execution, deep_inputs=False)
                aggregate_validated = True
            except Exception as error:
                aggregate_blocked = True
                aggregate_error = f"{type(error).__name__}: {error}"
        if qa_sealed and aggregate_validated:
            try:
                load_qa_root(execution, validate_aggregate=False)
                qa_validated = True
            except Exception as error:
                qa_blocked = True
                qa_error = f"{type(error).__name__}: {error}"
    any_blocked = (
        any(row["status"] == "BLOCKED" for row in rows)
        or aggregate_blocked
        or qa_blocked
    )
    fully_complete = (
        fit_validated == 30
        and replay_validated == 30
        and aggregate_validated
        and qa_validated
    )
    return {
        "schema": "TEST_CLARA_GEFCOM_6A_SELECTOR_STATUS_V1",
        "status": (
            "BLOCKED" if any_blocked else "PASS" if fully_complete else "INCOMPLETE"
        ),
        "rows": rows,
        "validation_gate_status": "PASS" if execution is not None else "BLOCKED",
        "validation_gate_error": validation_gate_error,
        "fit_validated_complete": fit_validated,
        "replay_validated_complete": replay_validated,
        "fit_sealed_unvalidated": sum(
            row["status"] == "SEALED_UNVALIDATED" and row["stage"] == "fit"
            for row in rows
        ),
        "replay_sealed_unvalidated": sum(
            row["status"] == "SEALED_UNVALIDATED" and row["stage"] == "replay"
            for row in rows
        ),
        "aggregate_status": (
            "VALIDATED_COMPLETE"
            if aggregate_validated
            else "SEALED_UNVALIDATED"
            if aggregate_sealed
            else "BLOCKED"
            if aggregate_blocked
            else "MISSING"
        ),
        "aggregate_validation_error": aggregate_error,
        "qa_status": (
            "VALIDATED_COMPLETE"
            if qa_validated
            else "SEALED_UNVALIDATED"
            if qa_sealed
            else "BLOCKED"
            if qa_blocked
            else "MISSING"
        ),
        "qa_validation_error": qa_error,
    }


def _coverage_partition_label(value: float) -> str:
    text = format(float(value), ".12g")
    return text.replace("-", "m").replace(".", "p")


def load_target_parent_inputs(
    *,
    zone: str,
    seed: int,
    endpoint_root_sha256: str,
    execution_identity_sha256: str,
    token_path: Path,
    token_sha256: str,
) -> dict[str, Any]:
    """Load 20 official target streams and seal their 220 compact partitions."""

    phase, _ = load_configs()
    root_path = resolved_package_path(
        phase["paths"]["endpoint_root_manifest_relative_path"]
    )
    root_manifest, root_units = core.validate_endpoint_root_seal(
        root_path, expected_endpoint_root_sha256=endpoint_root_sha256
    )
    horizon_frames: dict[int, pd.DataFrame] = {}
    partition_records: dict[str, dict[str, Any]] = {}
    stream_audits: dict[str, dict[str, Any]] = {}
    unit_records: dict[str, dict[str, Any]] = {}
    all_frames: list[pd.DataFrame] = []
    for horizon in core.FORMAL_HORIZONS:
        predictor_frames: dict[str, pd.DataFrame] = {}
        for predictor in core.FORMAL_PREDICTORS:
            facts, audit_capability = target_loader.load_causal_target_stream(
                evaluation_zone=zone,
                predictor=predictor,
                horizon=horizon,
                seed=int(seed),
                expected_endpoint_root_sha256=endpoint_root_sha256,
            )
            if not audit_capability.verified_by_formal_loader():
                raise RuntimeError("target loader未返回正式capability")
            audit = dict(audit_capability)
            stream_key = f"H{int(horizon):02d}__{predictor}"
            if stream_key in stream_audits:
                raise RuntimeError("target stream key重复")
            stream_audits[stream_key] = audit
            predictor_frames[predictor] = facts
            for coverage in core.FORMAL_COVERAGES:
                part = facts[
                    facts["target_coverage"].astype(float).eq(float(coverage))
                ].copy()
                if part.empty:
                    raise RuntimeError("target partition为空")
                partition_id = (
                    f"H{int(horizon):02d}__{predictor}__"
                    f"C{_coverage_partition_label(coverage)}"
                )
                signature = core.event_id_signature(
                    part["event_id"], include_canonical_sha256=True
                )
                partition_records[partition_id] = {
                    "zone_or_farm": str(zone),
                    "seed": int(seed),
                    "horizon_steps": int(horizon),
                    "predictor": str(predictor),
                    "target_coverage": float(coverage),
                    **signature,
                }
        unit_id = f"{zone}__seed{int(seed)}__H{int(horizon):02d}"
        expected_count = int(root_units[unit_id]["event_count"])
        unit_audit = target_loader.assert_target_unit_conservation(
            predictor_frames, expected_event_count=expected_count
        )
        horizon_frame = pd.concat(
            [predictor_frames[predictor] for predictor in core.FORMAL_PREDICTORS],
            ignore_index=True,
            sort=False,
        )
        horizon_frames[int(horizon)] = horizon_frame
        all_frames.append(horizon_frame)
        unit_records[f"H{int(horizon):02d}"] = {
            "unit_id": unit_id,
            "unit_manifest_sha256": str(
                root_units[unit_id]["unit_manifest_sha256"]
            ),
            "expected_event_count": expected_count,
            "observed_event_count": int(unit_audit["event_count"]),
            "cold_start_event_count": int(unit_audit["cold_start_event_count"]),
            "event_signature": dict(unit_audit["event_signature"]),
        }
    if len(stream_audits) != 20 or len(unit_records) != 5 or len(partition_records) != 220:
        raise RuntimeError("target parent必须闭合20 streams/5 units/220 partitions")
    combined_event_ids = pd.concat(
        [frame["event_id"].astype(str) for frame in all_frames],
        ignore_index=True,
    )
    if combined_event_ids.duplicated().any():
        raise RuntimeError("target parent五时距event_id重复")
    combined_signature = core.event_id_signature(
        combined_event_ids, include_canonical_sha256=True
    )
    expected_parent_count = sum(
        int(record["expected_event_count"]) for record in unit_records.values()
    )
    if len(combined_event_ids) != expected_parent_count:
        raise RuntimeError("target parent五时距事件数与endpoint root rows失配")
    payload = {
        "schema": "TEST_CLARA_GEFCOM_6A_TARGET_INPUT_PARENT_V1",
        "status": "PASS",
        "evaluation_zone": str(zone),
        "seed": int(seed),
        "endpoint_root_path": str(root_path),
        "endpoint_root_sha256": str(endpoint_root_sha256),
        "execution_identity_sha256": str(execution_identity_sha256),
        "target_loader_module_sha256": sha256_file(TARGET_LOADER_PATH),
        "stream_count": len(stream_audits),
        "stream_audits": stream_audits,
        "unit_count": len(unit_records),
        "unit_records": unit_records,
        "partition_count": len(partition_records),
        "partition_records": partition_records,
        "combined_event_signature": combined_signature,
    }
    manifest = {**payload, "payload_sha256": canonical_sha256(payload)}
    parent = replay_parent_dir(zone, seed)
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / "target_input_manifest.json"
    validate_preflight_token(
        token_path,
        expected_token_sha256=token_sha256,
        zone=zone,
        seed=seed,
    )
    if path.exists():
        existing = load_json(path)
        if existing != manifest or sha256_file(path) != hashlib.sha256(
            canonical_json_bytes(manifest)
        ).hexdigest():
            raise RuntimeError("既有target input manifest与当前完整输入身份冲突")
    else:
        atomic_json(path, manifest)
    # Revalidate the root at the end of the multi-file load.  Individual stream
    # loaders already rehash every leaf before/after each read.
    core.validate_endpoint_root_seal(
        root_path, expected_endpoint_root_sha256=endpoint_root_sha256
    )
    return {
        "horizon_frames": horizon_frames,
        "partition_records": partition_records,
        "partition_signatures": {
            partition_id: {
                field: record[field]
                for field in (
                    "event_count",
                    "hash_sum_u64",
                    "hash_xor_u64",
                    "hash_square_sum_u64",
                    "canonical_sorted_event_id_sha256",
                )
            }
            for partition_id, record in partition_records.items()
        },
        "combined_event_signature": combined_signature,
        "manifest": manifest,
        "manifest_path": path,
        "manifest_sha256": sha256_file(path),
        "endpoint_root_manifest": root_manifest,
    }


def materialize_target_partition(
    target: Mapping[str, Any], partition_id: str
) -> pd.DataFrame:
    """Materialize one root-pinned partition from the five resident horizons."""

    record = dict(target["partition_records"][str(partition_id)])
    horizon = int(record["horizon_steps"])
    predictor = str(record["predictor"])
    coverage = float(record["target_coverage"])
    source = target["horizon_frames"][horizon]
    part = source.loc[
        source["predictor"].astype(str).eq(predictor)
        & source["target_coverage"].astype(float).eq(coverage)
    ].copy()
    observed = core.event_id_signature(
        part["event_id"], include_canonical_sha256=True
    )
    expected = {
        field: record[field]
        for field in (
            "event_count",
            "hash_sum_u64",
            "hash_xor_u64",
            "hash_square_sum_u64",
            "canonical_sorted_event_id_sha256",
        )
    }
    if observed != expected:
        raise RuntimeError(f"target partition现场成员签名失配: {partition_id}")
    return part


def run_fit_parent(
    *,
    zone: str,
    seed: int,
    token_path: Path,
    token_sha256: str,
) -> dict[str, Any]:
    """Build/resume one formal heldout-zone×seed five-price fit parent."""

    token = validate_preflight_token(
        token_path,
        expected_token_sha256=token_sha256,
        zone=zone,
        seed=seed,
    )
    phase, main = load_configs()
    endpoint_root_sha = token["execution_identity"]["endpoint_root_audit"]["sha256"]
    artifacts = _artifacts_module()
    parent = fit_parent_dir(zone, seed)
    children_dir = parent / "children"
    parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_entries(children_dir)
    if temporary:
        raise RuntimeError(f"fit parent含残留temp，拒绝续跑: {temporary}")
    prices = core.price_specs_from_config(main)
    expected_prices = {price.price_id for price in prices}
    observed_children = (
        {child.name for child in children_dir.iterdir() if child.is_dir()}
        if children_dir.is_dir()
        else set()
    )
    if observed_children - expected_prices:
        raise RuntimeError(f"fit parent含额外child: {sorted(observed_children-expected_prices)}")
    _, states = index_contract_audit(phase)
    contracts = core.six_action_contracts(core.load_frozen_contracts())
    source_zones = tuple(sorted(set(core.FORMAL_ZONES) - {str(zone)}))
    builder = core.SixActionStatsBuilder(
        states=states,
        prices=prices,
        heldout_zone=str(zone),
        source_zones=source_zones,
        seed=int(seed),
        horizons=core.FORMAL_HORIZONS,
        predictors=core.FORMAL_PREDICTORS,
        coverages=core.FORMAL_COVERAGES,
        actions=core.SIX_ACTIONS,
        require_causal_source_audit=True,
    )
    for source_zone in source_zones:
        for predictor in core.FORMAL_PREDICTORS:
            for horizon in core.FORMAL_HORIZONS:
                paths = core.formal_causal_source_paths(
                    source_zone=source_zone,
                    predictor=predictor,
                    horizon=horizon,
                    seed=int(seed),
                )
                facts, audit = core.load_causal_source_stream(
                    source_fact_path=paths["source_fact"],
                    candidate_bundle_dir=paths["candidate_bundle_dir"],
                    endpoint_addon_path=paths["endpoint_addon"],
                    endpoint_root_manifest_path=paths["endpoint_root_manifest"],
                    expected_endpoint_root_sha256=endpoint_root_sha,
                    width_thresholds_path=paths["width_thresholds"],
                    outer_heldout_zone=str(zone),
                    source_zone=source_zone,
                    predictor=predictor,
                    horizon=horizon,
                    seed=int(seed),
                    coverages=core.FORMAL_COVERAGES,
                    verify_leaf_hashes=True,
                )
                builder.add_stream(
                    facts,
                    source_zone=source_zone,
                    predictor=predictor,
                    horizon=horizon,
                    seed=int(seed),
                    artifact_identity=audit,
                )
                del facts
    stats = builder.finalize()
    if stats.audit.get("endpoint_root_manifest_sha256_set") != [endpoint_root_sha]:
        raise RuntimeError("fit compact source未统一绑定当前endpoint root")
    v4_config = load_json(core.FOUR_VERSION_CONFIG_PATH)
    prepared = core.prepare_clara_fit(
        stats,
        contracts=contracts,
        v4_config=v4_config,
        require_formal_identity=True,
    )
    cart_config = core.frozen_zone_baseline_contract(zone, "CARTBestAction")[
        "configuration"
    ]
    execution_sha = token["execution_identity_sha256"]
    execution_payload = execution_identity_payload(token["execution_identity"])
    child_manifests: list[Path] = []
    for price in prices:
        child_path = children_dir / price.price_id
        if child_path.is_dir():
            validate_preflight_token(
                token_path,
                expected_token_sha256=token_sha256,
                zone=zone,
                seed=seed,
            )
            artifacts.load_fit_price_child(
                child_path,
                expected_zone=zone,
                expected_seed=int(seed),
                expected_price_id=price.price_id,
                expected_compact_source_identity_sha256=stats.audit["identity_sha256"],
                expected_endpoint_root_sha256=endpoint_root_sha,
                expected_state_universe=states,
                expected_execution_identity_sha256=execution_sha,
                formal_identity=True,
            )
        else:
            child = core.fit_price_conditioned_selectors(
                prepared,
                price=price,
                v4_config=v4_config,
                cart_config=cart_config,
                require_formal_identity=True,
            )
            # Recheck the exact capability after the potentially long fit and
            # immediately before the writer creates its atomic final child.
            validate_preflight_token(
                token_path,
                expected_token_sha256=token_sha256,
                zone=zone,
                seed=seed,
            )
            artifacts.write_fit_price_child(
                parent,
                child,
                endpoint_root_sha256=endpoint_root_sha,
                compact_source_identity_sha256=stats.audit["identity_sha256"],
                expected_state_universe=states,
                compact_source_audit=stats.audit,
                compact_stream_audit=stats.stream_audit,
                execution_identity_sha256=execution_sha,
                execution_identity=execution_payload,
                formal_identity=True,
            )
        child_manifests.append(child_path / "manifest.json")
    validate_preflight_token(
        token_path,
        expected_token_sha256=token_sha256,
        zone=zone,
        seed=seed,
    )
    return artifacts.seal_fit_parent(
        parent,
        stats=stats,
        child_manifest_paths=child_manifests,
        endpoint_root_sha256=endpoint_root_sha,
        execution_identity_sha256=execution_sha,
        execution_identity=execution_payload,
        formal_identity=True,
    )


def run_replay_parent(
    *,
    zone: str,
    seed: int,
    token_path: Path,
    token_sha256: str,
) -> dict[str, Any]:
    """Build/resume one globally coordinated five-horizon replay parent.

    The implementation is enabled only when the frozen core exposes its formal
    price-child accumulator finalizer; no non-formal fallback is permitted.
    """

    validate_preflight_token(
        token_path,
        expected_token_sha256=token_sha256,
        zone=zone,
        seed=seed,
    )
    if not hasattr(core.StreamingMetricAccumulator, "finalize_price_child"):
        raise RuntimeError(
            "BLOCKED: core缺正式zone-seed-price accumulator finalizer；"
            "禁止使用formal_identity=False绕过"
        )
    return _run_replay_parent_dual_mode_implemented(
        zone=zone,
        seed=seed,
        token_path=token_path,
        token_sha256=token_sha256,
    )


def _run_replay_parent_dual_mode_implemented(
    *, zone: str, seed: int, token_path: Path, token_sha256: str
) -> dict[str, Any]:
    """Resume one exact two-mode x five-evaluation-price replay parent."""

    token = validate_preflight_token(
        token_path,
        expected_token_sha256=token_sha256,
        zone=zone,
        seed=seed,
    )
    phase, main = load_configs()
    artifacts = _artifacts_module()
    execution_identity_full = token["execution_identity"]
    execution_sha = str(token["execution_identity_sha256"])
    execution_payload = execution_identity_payload(execution_identity_full)
    endpoint_root_sha = str(
        execution_identity_full["endpoint_root_audit"]["sha256"]
    )
    target = load_target_parent_inputs(
        zone=zone,
        seed=seed,
        endpoint_root_sha256=endpoint_root_sha,
        execution_identity_sha256=execution_sha,
        token_path=token_path,
        token_sha256=token_sha256,
    )
    combined_signature = dict(target["combined_event_signature"])
    expected_event_count = int(combined_signature["event_count"])
    parent = replay_parent_dir(zone, seed)
    children_dir = parent / "children"
    children_dir.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_entries(children_dir)
    if temporary:
        raise RuntimeError(f"replay parent含残留temp，拒绝续跑: {temporary}")
    prices = core.price_specs_from_config(main)
    price_by_id = {price.price_id: price for price in prices}
    expected_price_ids = tuple(price_by_id)
    expected_child_keys = tuple(
        replay_child_id(mode, price_id)
        for mode in POLICY_MODES
        for price_id in expected_price_ids
    )
    observed_children = {
        child.name for child in children_dir.iterdir() if child.is_dir()
    }
    if observed_children - set(expected_child_keys):
        raise RuntimeError(
            f"replay parent含额外child: {sorted(observed_children-set(expected_child_keys))}"
        )
    _, states = index_contract_audit(phase)
    fit_parent_path = fit_parent_dir(zone, seed) / "fit_parent_manifest.json"
    if not fit_parent_path.is_file():
        raise FileNotFoundError("replay要求同zone×seed fit parent seal")
    fit_parent_manifest = load_json(fit_parent_path)
    if (
        fit_parent_manifest.get("status") != "PASS"
        or str(fit_parent_manifest.get("heldout_zone")) != str(zone)
        or int(fit_parent_manifest.get("seed", -1)) != int(seed)
        or str(fit_parent_manifest.get("endpoint_root_sha256"))
        != endpoint_root_sha
        or str(fit_parent_manifest.get("execution_identity_sha256"))
        != execution_sha
        or tuple(fit_parent_manifest.get("price_child_identity_sha256", {}))
        != expected_price_ids
    ):
        raise RuntimeError("replay引用的fit parent轴/root/execution/五价失配")
    contracts = core.six_action_contracts(core.load_frozen_contracts())
    linucb_config = core.frozen_zone_baseline_contract(zone, "LinUCB")[
        "configuration"
    ]
    source_zones = tuple(sorted(set(core.FORMAL_ZONES) - {str(zone)}))
    loaded_fits: dict[str, Any] = {}
    fit_paths: dict[str, Path] = {}
    fit_identity_sha: dict[str, str] = {}
    fit_manifest_sha: dict[str, str] = {}
    for price in prices:
        fit_child_path = (
            fit_parent_dir(zone, seed)
            / "children"
            / price.price_id
            / "manifest.json"
        )
        fit_child_identity_sha = str(
            fit_parent_manifest["price_child_identity_sha256"][price.price_id]
        )
        fit_child_manifest_sha = str(
            fit_parent_manifest["price_child_manifest_sha256"][price.price_id]
        )
        if sha256_file(fit_child_path) != fit_child_manifest_sha:
            raise RuntimeError("replay same-price fit child manifest SHA漂移")
        loaded_fit = artifacts.load_fit_price_child(
            fit_child_path,
            expected_zone=zone,
            expected_seed=int(seed),
            expected_price_id=price.price_id,
            expected_compact_source_identity_sha256=str(
                fit_parent_manifest["compact_source_identity_sha256"]
            ),
            expected_endpoint_root_sha256=endpoint_root_sha,
            expected_execution_identity_sha256=execution_sha,
            expected_state_universe=states,
            formal_identity=True,
        )
        loaded_fits[price.price_id] = loaded_fit
        fit_paths[price.price_id] = fit_child_path
        fit_identity_sha[price.price_id] = fit_child_identity_sha
        fit_manifest_sha[price.price_id] = fit_child_manifest_sha

    child_manifest_paths: dict[str, Path] = {}
    missing_scopes: list[tuple[str, str, str]] = []
    for policy_mode in POLICY_MODES:
        for evaluation_price_id in expected_price_ids:
            policy_price_id = (
                evaluation_price_id
                if policy_mode == POLICY_MODES[0]
                else MAIN_PRICE_ID
            )
            child_key = replay_child_id(policy_mode, evaluation_price_id)
            replay_child_manifest = children_dir / child_key / "manifest.json"
            if replay_child_manifest.is_file():
                validate_preflight_token(
                    token_path,
                    expected_token_sha256=token_sha256,
                    zone=zone,
                    seed=seed,
                )
                artifacts.load_replay_price_child(
                    replay_child_manifest,
                    expected_zone=zone,
                    expected_seed=int(seed),
                    expected_policy_mode=policy_mode,
                    expected_evaluation_price_id=evaluation_price_id,
                    expected_policy_price_id=policy_price_id,
                    expected_fit_child_identity_sha256=fit_identity_sha[
                        policy_price_id
                    ],
                    expected_fit_child_manifest_sha256=fit_manifest_sha[
                        policy_price_id
                    ],
                    expected_endpoint_root_sha256=endpoint_root_sha,
                    expected_target_input_manifest_path=target["manifest_path"],
                    expected_target_input_manifest_sha256=target["manifest_sha256"],
                    expected_execution_identity_sha256=execution_sha,
                    expected_event_signature=combined_signature,
                    formal_identity=True,
                )
                child_manifest_paths[child_key] = replay_child_manifest
            else:
                missing_scopes.append(
                    (policy_mode, evaluation_price_id, policy_price_id)
                )

    if missing_scopes:
        required_policy_ids = tuple(
            price_id
            for price_id in expected_price_ids
            if any(scope[2] == price_id for scope in missing_scopes)
        )
        linucb = core.replay_all_price_conditioned_linucb(
            target["horizon_frames"],
            prices=tuple(price_by_id[price_id] for price_id in required_policy_ids),
            contracts=contracts,
            exploration_alpha=float(linucb_config["exploration_alpha"]),
            l2_regularization=float(linucb_config["l2_regularization"]),
            evaluation_zone=zone,
            seed=int(seed),
            source_zones=source_zones,
            expected_horizons=core.FORMAL_HORIZONS,
            expected_predictors=core.FORMAL_PREDICTORS,
            expected_coverages=core.FORMAL_COVERAGES,
            expected_event_count=expected_event_count,
            expected_event_signature=combined_signature,
            retain_decisions=True,
            require_formal_identity=True,
            fit_child_sha256_by_price={
                price_id: fit_identity_sha[price_id]
                for price_id in required_policy_ids
            },
            endpoint_root_sha256=endpoint_root_sha,
        )
        linucb_maps: dict[str, pd.Series] = {}
        for policy_price_id in required_policy_ids:
            decisions = linucb.decisions_by_price[policy_price_id]
            if (
                len(decisions) != expected_event_count
                or decisions["event_id"].astype(str).duplicated().any()
            ):
                raise RuntimeError(
                    f"LinUCB {policy_price_id}决策事件集不完整"
                )
            linucb_maps[policy_price_id] = decisions.set_index(
                decisions["event_id"].astype(str)
            )["selected_action"]
        accumulators = {
            replay_child_id(mode, evaluation_price_id):
            core.StreamingMetricAccumulator(
                retain_reduced_rows=True,
                expected_partition_signatures=target["partition_signatures"],
                formal_identity=True,
            )
            for mode, evaluation_price_id, _ in missing_scopes
        }
        for partition_id in sorted(target["partition_records"]):
            facts = materialize_target_partition(target, partition_id)
            predictions: dict[str, tuple[Any, Any]] = {}
            for policy_price_id in required_policy_ids:
                loaded_fit = loaded_fits[policy_price_id]
                predictions[policy_price_id] = (
                    core.predict_clara_actions_with_diagnostics(
                        facts,
                        loaded_fit.clara_decisions,
                        loaded_fit.clara_action_evidence,
                    ),
                    core.predict_cart_actions_with_capability(
                        facts, loaded_fit.cart_selector
                    ),
                )
            event_ids = facts["event_id"].astype(str)
            for policy_mode, evaluation_price_id, policy_price_id in missing_scopes:
                clara_result, cart_result = predictions[policy_price_id]
                linucb_actions = linucb_maps[policy_price_id].reindex(
                    event_ids
                ).to_numpy(dtype=object)
                if pd.isna(linucb_actions).any():
                    raise RuntimeError("LinUCB决策未覆盖target partition")
                method_actions = {
                    "CLARA_6A": clara_result.selected_actions,
                    "CART_6A": cart_result.selected_actions,
                    "LinUCB_6A": linucb_actions,
                    "TunedSingleConformal": np.full(
                        len(facts), "TunedSingleConformal", dtype=object
                    ),
                    "EqualEndpointEnsemble": np.full(
                        len(facts), "EqualEndpointEnsemble", dtype=object
                    ),
                    "FixedStatic": np.full(len(facts), "Static", dtype=object),
                    "FixedACI": np.full(len(facts), "ACI", dtype=object),
                    "FixedAgACI": np.full(len(facts), "AgACI", dtype=object),
                    "FixedEnbPI_RH": np.full(
                        len(facts), "EnbPI_RH", dtype=object
                    ),
                }
                accumulator = accumulators[
                    replay_child_id(policy_mode, evaluation_price_id)
                ]
                for method in core.FORMAL_METHODS:
                    accumulator.update(
                        evaluation_price=price_by_id[evaluation_price_id],
                        policy_mode=policy_mode,
                        policy_price=price_by_id[policy_price_id],
                        method=method,
                        facts=facts,
                        selected_actions=method_actions[method],
                        clara_event_diagnostics=(
                            clara_result if method == "CLARA_6A" else None
                        ),
                        cart_event_actions=(
                            cart_result if method == "CART_6A" else None
                        ),
                        linucb_replay_result=(
                            linucb if method == "LinUCB_6A" else None
                        ),
                        partition_id=partition_id,
                    )
            del facts, predictions

        for policy_mode, evaluation_price_id, policy_price_id in missing_scopes:
            child_key = replay_child_id(policy_mode, evaluation_price_id)
            compact = accumulators[child_key].finalize_price_child(
                price=price_by_id[evaluation_price_id],
                policy_mode=policy_mode,
                policy_price=price_by_id[policy_price_id],
                expected_event_count=expected_event_count,
                expected_event_signature=combined_signature,
                expected_methods=core.FORMAL_METHODS,
                linucb_replay_result=linucb,
            )
            validate_preflight_token(
                token_path,
                expected_token_sha256=token_sha256,
                zone=zone,
                seed=seed,
            )
            artifacts.write_replay_price_child(
                parent,
                policy_mode=policy_mode,
                evaluation_price=price_by_id[evaluation_price_id],
                policy_price=price_by_id[policy_price_id],
                evaluation_zone=zone,
                seed=int(seed),
                metric_result=compact,
                partition_records=target["partition_records"],
                fit_child_manifest_path=fit_paths[policy_price_id],
                endpoint_root_sha256=endpoint_root_sha,
                expected_event_signature=combined_signature,
                execution_identity_sha256=execution_sha,
                execution_identity=execution_payload,
                target_input_manifest_path=target["manifest_path"],
                target_input_manifest_sha256=target["manifest_sha256"],
                formal_identity=True,
            )
            child_manifest_paths[child_key] = (
                children_dir / child_key / "manifest.json"
            )
    validate_preflight_token(
        token_path,
        expected_token_sha256=token_sha256,
        zone=zone,
        seed=seed,
    )
    return artifacts.seal_replay_parent(
        parent,
        fit_parent_manifest_path=fit_parent_path,
        child_manifest_paths=child_manifest_paths,
        endpoint_root_sha256=endpoint_root_sha,
        expected_event_signature_by_evaluation_price={
            price.price_id: combined_signature for price in prices
        },
        execution_identity_sha256=execution_sha,
        execution_identity=execution_payload,
        target_input_manifest_path=target["manifest_path"],
        target_input_manifest_sha256=target["manifest_sha256"],
        formal_identity=True,
    )


def _metric_means(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    denominator = output["event_count"].to_numpy(dtype=float)
    if np.any(denominator <= 0):
        raise RuntimeError("compact metric含非正event_count")
    means = {
        "mean_errf": "errf_sum",
        "mean_reserve_up": "reserve_up_sum",
        "mean_reserve_down": "reserve_down_sum",
        "mean_miss_upper": "miss_upper_sum",
        "mean_miss_lower": "miss_lower_sum",
        "empirical_coverage": "covered_sum",
        "mean_coverage_target": "coverage_target_sum",
        "coverage_gap": "coverage_gap_sum",
        "mean_width": "width_sum",
        "mean_interval_score": "interval_score_sum",
    }
    missing = sorted(set(means.values()) - set(output.columns))
    if missing:
        raise RuntimeError(f"compact metric缺冻结加性列: {missing}")
    for result, source in means.items():
        output[result] = output[source].to_numpy(dtype=float) / denominator
    return output


def _safe_ratio(
    numerator: pd.Series | np.ndarray,
    denominator: pd.Series | np.ndarray,
) -> np.ndarray:
    top = np.asarray(numerator, dtype=float)
    bottom = np.asarray(denominator, dtype=float)
    return np.divide(
        top,
        bottom,
        out=np.full(len(top), np.nan, dtype=float),
        where=bottom > 0.0,
    )


def _diagnostic_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive non-additive diagnostics only from frozen integer sums."""

    output = _metric_means(frame)
    output["TOWR"] = _safe_ratio(
        output["towr_exceedance_count"], output["rolling_window_count"]
    )
    output["TUWR"] = _safe_ratio(
        output["tuwr_exceedance_count"], output["rolling_window_count"]
    )
    output["ARD"] = _safe_ratio(
        output["rolling_absolute_deviation_sum"],
        output["rolling_window_count"],
    )
    output["support_backoff_rate"] = _safe_ratio(
        output["support_backoff_count"], output["support_applicable_count"]
    )
    output["guardrail_any_exclusion_rate"] = _safe_ratio(
        output["guardrail_any_exclusion_count"],
        output["guardrail_applicable_count"],
    )
    output["guardrail_action_exclusion_rate"] = _safe_ratio(
        output["guardrail_excluded_action_count"],
        output["guardrail_applicable_count"].to_numpy(dtype=float)
        * len(core.SIX_ACTIONS),
    )
    output["guardrail_empty_rate"] = _safe_ratio(
        output["guardrail_empty_count"], output["guardrail_applicable_count"]
    )
    output["selected_action_guardrail_fail_rate"] = _safe_ratio(
        output["selected_action_guardrail_fail_count"],
        output["guardrail_applicable_count"],
    )
    return output


def _entropy_table(
    actions: pd.DataFrame, *, include_zone: bool
) -> pd.DataFrame:
    """Recompute regime-specific entropy from action counts."""

    base = [*RESULT_SCOPE_FIELDS, "method"]
    if include_zone:
        base.append("zone_or_farm")
    required = set(base) | {"ramp_state", "selected_action", "action_count"}
    if not required.issubset(actions):
        raise RuntimeError("action entropy缺动作充分统计")
    overall = (
        actions.groupby([*base, "selected_action"], sort=True, as_index=False)[
            "action_count"
        ]
        .sum()
        .assign(regime="overall")
    )
    strata = (
        actions.groupby(
            [*base, "ramp_state", "selected_action"],
            sort=True,
            as_index=False,
        )["action_count"]
        .sum()
        .rename(columns={"ramp_state": "regime"})
    )
    combined = pd.concat([overall, strata], ignore_index=True, sort=False)
    group = [*base, "regime"]
    totals = combined.groupby(group, sort=True)["action_count"].transform("sum")
    probability = combined["action_count"].to_numpy(dtype=float) / totals.to_numpy(
        dtype=float
    )
    combined["entropy_component"] = np.where(
        probability > 0.0, -probability * np.log(probability), 0.0
    )
    result = (
        combined.groupby(group, sort=True, as_index=False)
        .agg(
            action_count=("action_count", "sum"),
            action_entropy_nats=("entropy_component", "sum"),
        )
        .reset_index(drop=True)
    )
    result["action_entropy_normalized"] = (
        result["action_entropy_nats"] / np.log(len(core.SIX_ACTIONS))
    )
    return result


DIAGNOSTIC_ADDITIVE_COLUMNS = (
    "event_count",
    "errf_sum",
    "reserve_up_sum",
    "reserve_down_sum",
    "miss_upper_sum",
    "miss_lower_sum",
    "covered_sum",
    "coverage_target_sum",
    "coverage_gap_sum",
    "width_sum",
    "interval_score_sum",
    "rolling_window_count",
    "towr_exceedance_count",
    "tuwr_exceedance_count",
    "rolling_absolute_deviation_sum",
    "support_applicable_count",
    "support_backoff_count",
    "guardrail_applicable_count",
    "guardrail_any_exclusion_count",
    "guardrail_excluded_action_count",
    "guardrail_empty_count",
    "selected_action_guardrail_fail_count",
)


def aggregate_diagnostic_metrics(
    diagnostics: pd.DataFrame, actions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Close selected-endpoint secondary metrics without averaging rates."""

    keys = [*RESULT_SCOPE_FIELDS, "method", "zone_or_farm", "regime"]
    required = set(keys) | set(DIAGNOSTIC_ADDITIVE_COLUMNS)
    if not required.issubset(diagnostics):
        raise RuntimeError(
            f"diagnostic compact缺列: {sorted(required-set(diagnostics))}"
        )
    if set(diagnostics["regime"].astype(str)) != {"overall", "ordinary", "ramp"}:
        raise RuntimeError("diagnostic regime未闭合overall/ordinary/ramp")
    zone_raw = (
        diagnostics.groupby(keys, sort=True, as_index=False)[
            list(DIAGNOSTIC_ADDITIVE_COLUMNS)
        ]
        .sum()
        .reset_index(drop=True)
    )
    summary_identity = [*RESULT_SCOPE_FIELDS, "method", "regime"]
    counts = zone_raw.groupby(summary_identity, sort=True)[
        "zone_or_farm"
    ].nunique()
    if len(counts) != 2 * 5 * len(core.FORMAL_METHODS) * 3 or not counts.eq(10).all():
        raise RuntimeError("diagnostic zone未闭合2mode×5价×9方法×3regime×10区")
    zone = _diagnostic_rates(zone_raw)
    zone = zone.merge(
        _entropy_table(actions, include_zone=True),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if zone[["action_entropy_nats", "action_entropy_normalized"]].isna().any().any():
        raise RuntimeError("diagnostic zone动作熵未闭合")

    event_means = (
        "mean_errf",
        "mean_reserve_up",
        "mean_reserve_down",
        "mean_miss_upper",
        "mean_miss_lower",
        "empirical_coverage",
        "mean_coverage_target",
        "coverage_gap",
        "mean_width",
        "mean_interval_score",
    )
    equal_zone = (
        zone.groupby(summary_identity, sort=True, as_index=False)[
            list(event_means)
        ]
        .mean()
        .reset_index(drop=True)
    )
    pooled_raw = (
        zone_raw.groupby(
            summary_identity, sort=True, as_index=False
        )[list(DIAGNOSTIC_ADDITIVE_COLUMNS)]
        .sum()
        .reset_index(drop=True)
    )
    pooled = _diagnostic_rates(pooled_raw)
    entropy = _entropy_table(actions, include_zone=False)
    rate_columns = (
        "TOWR",
        "TUWR",
        "ARD",
        "support_backoff_rate",
        "guardrail_any_exclusion_rate",
        "guardrail_action_exclusion_rate",
        "guardrail_empty_rate",
        "selected_action_guardrail_fail_rate",
    )
    identity = summary_identity
    secondary = equal_zone.merge(
        pooled[[*identity, *DIAGNOSTIC_ADDITIVE_COLUMNS, *rate_columns]],
        on=identity,
        how="left",
        validate="one_to_one",
    ).merge(entropy, on=identity, how="left", validate="one_to_one")
    secondary["zone_count"] = 10
    secondary["event_metric_aggregation_identity"] = (
        "EVENT_MEAN_WITHIN_ZONE_THEN_EQUAL_WEIGHT_TEN_ZONES"
    )
    secondary["rolling_and_guardrail_aggregation_identity"] = (
        "RATIO_OF_FROZEN_INTEGER_SUFFICIENT_COUNTS_ACROSS_TEN_ZONES"
    )
    descriptive = pooled.merge(
        entropy, on=identity, how="left", validate="one_to_one"
    )
    descriptive["aggregation_identity"] = (
        "EVENT_WEIGHTED_ACROSS_ZONES_DESCRIPTIVE_ONLY"
    )
    clara = secondary[secondary["method"].astype(str).eq("CLARA_6A")].copy()
    if len(clara) != 30 or not clara["support_applicable_count"].gt(0).all():
        raise RuntimeError("CLARA support/guardrail summary未闭合2mode×5价×3regime")
    return zone, secondary, descriptive, clara.reset_index(drop=True)


def zone_balanced_overall_metrics(
    cells: pd.DataFrame,
    *,
    numeric_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute the frozen event-within-zone, equal-weight-across-zone estimand."""

    group_fields = [*RESULT_SCOPE_FIELDS, "method"]
    zone_raw = (
        cells.groupby(
            [*group_fields, "zone_or_farm"],
            sort=True,
            as_index=False,
        )[list(numeric_columns)]
        .sum()
        .reset_index(drop=True)
    )
    zone_counts = zone_raw.groupby(group_fields, sort=True)[
        "zone_or_farm"
    ].agg(lambda values: len(set(str(value) for value in values)))
    if (
        len(zone_counts) != 90
        or not zone_counts.eq(10).all()
        or set(zone_raw["zone_or_farm"].astype(str)) != set(core.FORMAL_ZONES)
    ):
        raise RuntimeError("zone-balanced总体统计未闭合5价×9方法×10 zones")
    zone_means = _metric_means(zone_raw)
    mean_columns = (
        "mean_errf",
        "mean_reserve_up",
        "mean_reserve_down",
        "mean_miss_upper",
        "mean_miss_lower",
        "empirical_coverage",
        "mean_coverage_target",
        "coverage_gap",
        "mean_width",
        "mean_interval_score",
    )
    if not np.isfinite(zone_means[list(mean_columns)].to_numpy(dtype=float)).all():
        raise RuntimeError("zone-balanced总体统计含非有限zone mean")
    balanced = (
        zone_means.groupby(group_fields, sort=True, as_index=False)[
            list(mean_columns)
        ]
        .mean()
        .reset_index(drop=True)
    )
    totals = (
        zone_raw.groupby(group_fields, sort=True, as_index=False)[
            ["event_count"]
        ]
        .sum()
        .reset_index(drop=True)
    )
    balanced = balanced.merge(
        totals, on=group_fields, how="left", validate="one_to_one"
    )
    balanced["zone_count"] = 10
    balanced["aggregation_identity"] = (
        "EVENT_MEAN_WITHIN_ZONE_THEN_EQUAL_WEIGHT_TEN_ZONES"
    )
    balanced["errf_rank_within_price"] = balanced.groupby(
        ["policy_mode", "evaluation_price_id"]
    )[
        "mean_errf"
    ].rank(method="min", ascending=True).astype(int)

    event_weighted_raw = (
        cells.groupby(group_fields, sort=True, as_index=False)[
            list(numeric_columns)
        ]
        .sum()
        .reset_index(drop=True)
    )
    event_weighted = _metric_means(event_weighted_raw)
    event_weighted["aggregation_identity"] = (
        "EVENT_WEIGHTED_ACROSS_ZONES_DESCRIPTIVE_ONLY"
    )
    return balanced, event_weighted, zone_means


def paired_inference_from_zone_metrics(
    zone_metrics: pd.DataFrame,
    *,
    bootstrap_replicates: int = 5000,
    random_seed: int = 2026082802,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the frozen 660-cell/zone paired bootstrap and exact sign test."""

    required = {
        *RESULT_SCOPE_FIELDS,
        "zone_or_farm",
        "method",
        "zone_mean_errf",
    }
    if not required.issubset(zone_metrics) or zone_metrics.empty:
        raise RuntimeError("paired inference缺双mode zone compact metrics")
    current = zone_metrics.copy()
    if current[list(required)].isna().any().any() or current.duplicated(
        [*RESULT_SCOPE_FIELDS, "zone_or_farm", "method"]
    ).any():
        raise RuntimeError("paired inference输入含NaN/重复键")
    comparators = tuple(
        method for method in core.FORMAL_METHODS if method != "CLARA_6A"
    )
    rng = np.random.default_rng(int(random_seed))
    zone_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    for policy_mode in POLICY_MODES:
        for evaluation_price_id in [row[0] for row in core.FORMAL_PRICE_ROWS]:
            policy_price_id = (
                evaluation_price_id
                if policy_mode == "RESELECT_EACH_RATIO"
                else MAIN_PRICE_ID
            )
            price_frame = current[
                current["policy_mode"].astype(str).eq(policy_mode)
                & current["evaluation_price_id"].astype(str).eq(
                    evaluation_price_id
                )
                & current["policy_price_id"].astype(str).eq(policy_price_id)
            ]
            pivot = price_frame.pivot(
                index="zone_or_farm", columns="method", values="zone_mean_errf"
            ).reindex(index=core.FORMAL_ZONES, columns=core.FORMAL_METHODS)
            if pivot.isna().any().any():
                raise RuntimeError(
                    "paired inference未闭合2mode×5价×10 zones×9 methods"
                )
            for comparator in comparators:
                clara = pivot["CLARA_6A"].to_numpy(dtype=float)
                baseline = pivot[comparator].to_numpy(dtype=float)
                differences = clara - baseline
                if not np.isfinite(differences).all():
                    raise RuntimeError("paired inference差值非有限")
                for zone, clara_value, baseline_value, difference in zip(
                    core.FORMAL_ZONES, clara, baseline, differences
                ):
                    zone_rows.append(
                        {
                            "policy_mode": policy_mode,
                            "evaluation_price_id": evaluation_price_id,
                            "policy_price_id": policy_price_id,
                            "comparator": comparator,
                            "zone_or_farm": zone,
                            "clara_mean_errf": float(clara_value),
                            "comparator_mean_errf": float(baseline_value),
                            "paired_difference_clara_minus_comparator": float(
                                difference
                            ),
                        }
                    )
                observed = float(differences.mean())
                samples = rng.integers(
                    0,
                    len(differences),
                    size=(int(bootstrap_replicates), len(differences)),
                )
                bootstrap = differences[samples].mean(axis=1)
                signs = np.where(
                    (
                        (np.arange(1 << len(differences))[:, None]
                        >> np.arange(len(differences)))
                        & 1
                    )
                    == 1,
                    1.0,
                    -1.0,
                )
                null_values = (signs * differences[None, :]).mean(axis=1)
                extreme = int(
                    (np.abs(null_values) >= abs(observed) - 1e-15).sum()
                )
                baseline_mean = float(baseline.mean())
                inference_rows.append(
                    {
                        "policy_mode": policy_mode,
                        "evaluation_price_id": evaluation_price_id,
                        "policy_price_id": policy_price_id,
                        "comparator": comparator,
                        "zone_count": len(differences),
                        "mean_paired_difference_clara_minus_comparator": observed,
                        "relative_difference_percent": (
                            np.nan
                            if baseline_mean == 0.0
                            else 100.0 * observed / baseline_mean
                        ),
                        "zone_bootstrap_ci95_lower": float(
                            np.quantile(bootstrap, 0.025)
                        ),
                        "zone_bootstrap_ci95_upper": float(
                            np.quantile(bootstrap, 0.975)
                        ),
                        "exact_sign_flip_extreme_count": extreme,
                        "exact_sign_flip_configuration_count": len(null_values),
                        "exact_sign_flip_p_value": extreme / float(len(null_values)),
                        "bootstrap_replicates": int(bootstrap_replicates),
                        "bootstrap_seed": int(random_seed),
                    }
                )
    inference = pd.DataFrame(inference_rows)
    inference["holm_adjusted_p_value"] = np.nan
    inference["holm_reject_0p05"] = False
    for _, index in inference.groupby(
        ["policy_mode", "evaluation_price_id"], sort=False
    ).groups.items():
        indices = list(index)
        ordered = sorted(indices, key=lambda value: inference.at[value, "exact_sign_flip_p_value"])
        running = 0.0
        m = len(ordered)
        for rank, row_index in enumerate(ordered):
            adjusted = min(
                1.0,
                max(
                    running,
                    (m - rank)
                    * float(inference.at[row_index, "exact_sign_flip_p_value"]),
                ),
            )
            running = adjusted
            inference.at[row_index, "holm_adjusted_p_value"] = adjusted
            inference.at[row_index, "holm_reject_0p05"] = adjusted <= 0.05
    return pd.DataFrame(zone_rows), inference.sort_values(
        [
            "policy_mode",
            "evaluation_price_id",
            "holm_adjusted_p_value",
            "comparator",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def equal_design_zone_metrics(cells: pd.DataFrame) -> pd.DataFrame:
    """Average exact seed×predictor×horizon×coverage cells within zone."""

    keys = [
        *RESULT_SCOPE_FIELDS,
        "method",
        "zone_or_farm",
        "seed",
        "predictor",
        "horizon_steps",
        "target_coverage",
    ]
    required = set(keys) | {"event_count", "errf_sum"}
    if not required.issubset(cells):
        raise RuntimeError("equal-design inference缺设计格/误差字段")
    design = (
        cells.groupby(keys, sort=True, as_index=False)[["event_count", "errf_sum"]]
        .sum()
        .reset_index(drop=True)
    )
    expected_cells = 3 * len(core.FORMAL_PREDICTORS) * len(core.FORMAL_HORIZONS) * len(
        core.FORMAL_COVERAGES
    )
    counts = design.groupby(
        [*RESULT_SCOPE_FIELDS, "method", "zone_or_farm"], sort=True
    ).size()
    if (
        len(counts) != 2 * 5 * len(core.FORMAL_METHODS) * len(core.FORMAL_ZONES)
        or not counts.eq(expected_cells).all()
        or set(design["seed"].astype(int)) != {0, 1, 2}
        or set(design["predictor"].astype(str)) != set(core.FORMAL_PREDICTORS)
        or set(design["horizon_steps"].astype(int)) != set(core.FORMAL_HORIZONS)
        or set(design["target_coverage"].astype(float))
        != set(core.FORMAL_COVERAGES)
    ):
        raise RuntimeError("equal-design inference未闭合exact 660格/区/方法/价格")
    if np.any(design["event_count"].to_numpy(dtype=np.int64) <= 0):
        raise RuntimeError("equal-design inference含空设计格")
    design["design_cell_mean_errf"] = (
        design["errf_sum"].to_numpy(dtype=float)
        / design["event_count"].to_numpy(dtype=float)
    )
    zone = (
        design.groupby(
            [*RESULT_SCOPE_FIELDS, "method", "zone_or_farm"],
            sort=True,
            as_index=False,
        )
        .agg(
            zone_mean_errf=("design_cell_mean_errf", "mean"),
            design_cell_count=("design_cell_mean_errf", "size"),
            source_event_count=("event_count", "sum"),
        )
        .reset_index(drop=True)
    )
    zone["aggregation_identity"] = (
        "EVENT_MEAN_WITHIN_660_DESIGN_CELLS_THEN_EQUAL_CELL_MEAN_WITHIN_ZONE"
    )
    return zone


def _assert_compact_exact_multiset(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    drop_columns: Sequence[str],
    label: str,
) -> None:
    if not set(drop_columns).issubset(left) or not set(drop_columns).issubset(
        right
    ):
        raise RuntimeError(f"{label}: 缺少双mode身份列")
    left_value = left.drop(columns=list(drop_columns))
    right_value = right.drop(columns=list(drop_columns))
    if list(left_value.columns) != list(right_value.columns):
        raise RuntimeError(f"{label}: compact列漂移")
    sort_columns = list(left_value.columns)
    try:
        pd.testing.assert_frame_equal(
            left_value.sort_values(
                sort_columns, kind="mergesort"
            ).reset_index(drop=True),
            right_value.sort_values(
                sort_columns, kind="mergesort"
            ).reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError as error:
        raise RuntimeError(f"{label}: compact exact代数门失败: {error}") from error


def assert_cross_mode_child_algebra(children: Sequence[Any]) -> None:
    """Enforce the frozen R05 and deterministic cross-mode identities."""

    compact_names = (
        "cell_metrics",
        "action_counts",
        "paired_blocks",
        "diagnostic_metrics",
        "clara_state_counts",
        "event_conservation",
    )
    by_key: dict[tuple[str, int, str, str], Any] = {}
    for child in children:
        manifest = dict(child.manifest)
        key = (
            str(manifest.get("evaluation_zone")),
            int(manifest.get("seed", -1)),
            str(manifest.get("policy_mode")),
            str(manifest.get("evaluation_price_id")),
        )
        expected_policy_price = (
            key[3] if key[2] == POLICY_MODES[0] else MAIN_PRICE_ID
        )
        if (
            key in by_key
            or key[0] not in core.FORMAL_ZONES
            or key[1] not in range(3)
            or key[2] not in POLICY_MODES
            or key[3] not in {row[0] for row in core.FORMAL_PRICE_ROWS}
            or str(manifest.get("policy_price_id")) != expected_policy_price
        ):
            raise RuntimeError(f"cross-mode replay child轴/策略价串线: {key}")
        for name in compact_names:
            frame = getattr(child, name)
            if not set(RESULT_SCOPE_FIELDS).issubset(frame):
                raise RuntimeError(f"replay child {key} 的{name}缺双mode轴")
            scope = set(
                zip(
                    frame["policy_mode"].astype(str),
                    frame["evaluation_price_id"].astype(str),
                    frame["policy_price_id"].astype(str),
                )
            )
            if scope != {(key[2], key[3], expected_policy_price)}:
                raise RuntimeError(f"replay child {key} 的{name}内部轴串线")
        by_key[key] = child
    parent_axes = {(key[0], key[1]) for key in by_key}
    if len(parent_axes) not in {1, 30}:
        raise RuntimeError("cross-mode algebra仅接受exact1或30个zone×seed parent")
    expected_keys = {
        (zone, seed, mode, price_id)
        for zone, seed in parent_axes
        for mode in POLICY_MODES
        for price_id in (row[0] for row in core.FORMAL_PRICE_ROWS)
    }
    if set(by_key) != expected_keys:
        raise RuntimeError("cross-mode algebra要求每parent exact2×5 replay children")
    deterministic = {
        "TunedSingleConformal",
        "EqualEndpointEnsemble",
        "FixedStatic",
        "FixedACI",
        "FixedAgACI",
        "FixedEnbPI_RH",
    }
    for zone, seed in sorted(parent_axes):
            r05_reselected = by_key[
                (zone, seed, POLICY_MODES[0], MAIN_PRICE_ID)
            ]
            r05_fixed = by_key[(zone, seed, POLICY_MODES[1], MAIN_PRICE_ID)]
            for name in compact_names:
                _assert_compact_exact_multiset(
                    getattr(r05_reselected, name),
                    getattr(r05_fixed, name),
                    drop_columns=("policy_mode",),
                    label=f"R05 cross-mode {zone}/S{seed}/{name}",
                )
            for evaluation_price_id in (
                row[0] for row in core.FORMAL_PRICE_ROWS
            ):
                reselected = by_key[
                    (zone, seed, POLICY_MODES[0], evaluation_price_id)
                ]
                fixed = by_key[
                    (zone, seed, POLICY_MODES[1], evaluation_price_id)
                ]
                for name in compact_names:
                    left = getattr(reselected, name)
                    right = getattr(fixed, name)
                    if "method" in left:
                        left = left[left["method"].astype(str).isin(deterministic)]
                        right = right[right["method"].astype(str).isin(deterministic)]
                    _assert_compact_exact_multiset(
                        left,
                        right,
                        drop_columns=("policy_mode", "policy_price_id"),
                        label=(
                            f"deterministic cross-mode {zone}/S{seed}/"
                            f"{evaluation_price_id}/{name}"
                        ),
                    )
            reference_actions = r05_fixed.action_counts
            for evaluation_price_id in (
                row[0] for row in core.FORMAL_PRICE_ROWS
            ):
                _assert_compact_exact_multiset(
                    reference_actions,
                    by_key[
                        (zone, seed, POLICY_MODES[1], evaluation_price_id)
                    ].action_counts,
                    drop_columns=RESULT_SCOPE_FIELDS,
                    label=(
                        f"fixed-scan action counts {zone}/S{seed}/"
                        f"{evaluation_price_id}"
                    ),
                )


def aggregate_compact_children(
    children: Iterable[Any],
) -> dict[str, pd.DataFrame]:
    """Reduce validated replay children without retaining event-level rows."""

    cell_parts: list[pd.DataFrame] = []
    action_parts: list[pd.DataFrame] = []
    diagnostic_parts: list[pd.DataFrame] = []
    clara_state_parts: list[pd.DataFrame] = []
    conservation_parts: list[pd.DataFrame] = []
    zone_block_parts: list[pd.DataFrame] = []
    index_rows: list[dict[str, Any]] = []
    child_count = 0
    parent_batch: list[Any] = []
    for loaded in children:
        parent_batch.append(loaded)
        manifest = dict(loaded.manifest)
        child_count += 1
        cell = loaded.cell_metrics.copy()
        actions = loaded.action_counts.copy()
        blocks = loaded.paired_blocks.copy()
        diagnostics = loaded.diagnostic_metrics.copy()
        clara_states = loaded.clara_state_counts.copy()
        conservation = loaded.event_conservation.copy()
        cell_parts.append(cell)
        action_parts.append(actions)
        diagnostic_parts.append(diagnostics)
        clara_state_parts.append(clara_states)
        conservation_parts.append(conservation)
        zone_blocks = (
            blocks.groupby(
                [*RESULT_SCOPE_FIELDS, "zone_or_farm", "method"],
                sort=True,
                as_index=False,
            )[["event_count", "errf_sum"]]
            .sum()
            .reset_index(drop=True)
        )
        zone_block_parts.append(zone_blocks)
        files = manifest["files"]
        manifest_sha = str(manifest.get("_manifest_sha256", ""))
        manifest_relative_path = str(manifest.get("_manifest_relative_path", ""))
        if not is_sha256(manifest_sha) or not manifest_relative_path:
            raise RuntimeError("aggregate replay child缺deep-loader manifest路径/SHA")
        index_rows.append(
            {
                "evaluation_zone": manifest["evaluation_zone"],
                "seed": int(manifest["seed"]),
                "policy_mode": manifest["policy_mode"],
                "evaluation_price_id": manifest["evaluation_price_id"],
                "policy_price_id": manifest["policy_price_id"],
                "replay_child_identity_sha256": manifest[
                    "replay_child_identity_sha256"
                ],
                "manifest_relative_path": manifest_relative_path,
                "manifest_sha256": manifest_sha,
                "cell_metrics_relative_path": files["cell_metrics"]["relative_path"],
                "cell_metrics_sha256": files["cell_metrics"]["sha256"],
                "cell_metrics_row_count": int(files["cell_metrics"]["row_count"]),
                "action_counts_relative_path": files["action_counts"]["relative_path"],
                "action_counts_sha256": files["action_counts"]["sha256"],
                "action_counts_row_count": int(files["action_counts"]["row_count"]),
                "paired_blocks_relative_path": files["paired_blocks"]["relative_path"],
                "paired_blocks_sha256": files["paired_blocks"]["sha256"],
                "paired_blocks_row_count": int(files["paired_blocks"]["row_count"]),
                "diagnostic_metrics_relative_path": files["diagnostic_metrics"]["relative_path"],
                "diagnostic_metrics_sha256": files["diagnostic_metrics"]["sha256"],
                "diagnostic_metrics_row_count": int(files["diagnostic_metrics"]["row_count"]),
                "clara_state_counts_relative_path": files["clara_state_counts"]["relative_path"],
                "clara_state_counts_sha256": files["clara_state_counts"]["sha256"],
                "clara_state_counts_row_count": int(files["clara_state_counts"]["row_count"]),
            }
        )
        del blocks
        if len(parent_batch) == 10:
            assert_cross_mode_child_algebra(parent_batch)
            parent_batch.clear()
            del loaded
        elif len(parent_batch) > 10:
            raise RuntimeError("aggregate child stream未按exact parent十子单元排序")
    if parent_batch:
        raise RuntimeError("aggregate child stream在parent边界截断")
    if child_count != 300:
        raise RuntimeError(f"aggregate必须精确消费300 replay children: {child_count}")
    cells = pd.concat(cell_parts, ignore_index=True, sort=False)
    actions = pd.concat(action_parts, ignore_index=True, sort=False)
    diagnostics = pd.concat(diagnostic_parts, ignore_index=True, sort=False)
    clara_states = pd.concat(clara_state_parts, ignore_index=True, sort=False)
    conservation = pd.concat(conservation_parts, ignore_index=True, sort=False)
    numeric_cells = [
        column
        for column in cells.columns
        if column
        not in {
            *RESULT_SCOPE_FIELDS,
            "method",
            *core.StreamingMetricAccumulator.CELL_FIELDS,
        }
    ]
    cells = (
        cells.groupby(
            [*RESULT_SCOPE_FIELDS, "method", *core.StreamingMetricAccumulator.CELL_FIELDS],
            sort=True,
            as_index=False,
        )[numeric_cells]
        .sum()
        .reset_index(drop=True)
    )
    actions = (
        actions.groupby(
            [
                *RESULT_SCOPE_FIELDS,
                "method",
                *core.StreamingMetricAccumulator.CELL_FIELDS,
                "selected_action",
            ],
            sort=True,
            as_index=False,
        )["action_count"]
        .sum()
        .reset_index(drop=True)
    )
    diagnostic_keys = [
        *RESULT_SCOPE_FIELDS,
        "method",
        *core.StreamingMetricAccumulator.DIAGNOSTIC_FIELDS,
    ]
    if diagnostics.duplicated(diagnostic_keys).any():
        raise RuntimeError("aggregate diagnostic compact键重复")
    diagnostics = diagnostics.sort_values(
        diagnostic_keys, kind="mergesort"
    ).reset_index(drop=True)
    clara_state_keys = [
        *RESULT_SCOPE_FIELDS,
        "method",
        *core.StreamingMetricAccumulator.CLARA_STATE_COUNT_FIELDS,
    ]
    if set(clara_states["method"].astype(str)) != {"CLARA_6A"}:
        raise RuntimeError("aggregate CLARA state compact含非CLARA方法")
    clara_states = (
        clara_states.groupby(clara_state_keys, sort=True, as_index=False)[
            "event_count"
        ]
        .sum()
        .reset_index(drop=True)
    )
    conservation = (
        conservation.groupby([*RESULT_SCOPE_FIELDS, "method"], sort=True, as_index=False)[
            ["event_count", "hash_sum_u64", "hash_xor_u64", "hash_square_sum_u64"]
        ]
        .agg(
            {
                "event_count": "sum",
                "hash_sum_u64": lambda values: int(sum(int(value) for value in values) % (1 << 64)),
                "hash_xor_u64": lambda values: int(np.bitwise_xor.reduce(np.asarray(values, dtype=np.uint64))),
                "hash_square_sum_u64": lambda values: int(sum(int(value) for value in values) % (1 << 64)),
            }
        )
        .reset_index(drop=True)
    )
    if (
        len(conservation) != 90
        or not conservation["event_count"].eq(23_024_760).all()
        or set(conservation["evaluation_price_id"].astype(str))
        != {row[0] for row in core.FORMAL_PRICE_ROWS}
        or set(conservation["policy_mode"].astype(str)) != set(POLICY_MODES)
        or set(conservation["method"].astype(str)) != set(core.FORMAL_METHODS)
    ):
        raise RuntimeError("aggregate全局9方法×2mode×5价×23,024,760事件不守恒")
    diagnostic_overall = diagnostics[
        diagnostics["regime"].astype(str).eq("overall")
    ].groupby([*RESULT_SCOPE_FIELDS, "method"], sort=True)["event_count"].sum()
    diagnostic_strata = diagnostics[
        diagnostics["regime"].astype(str).isin(("ordinary", "ramp"))
    ].groupby([*RESULT_SCOPE_FIELDS, "method"], sort=True)["event_count"].sum()
    state_totals = clara_states.groupby(
        list(RESULT_SCOPE_FIELDS), sort=True
    )["event_count"].sum()
    if (
        len(diagnostic_overall) != 90
        or not diagnostic_overall.eq(23_024_760).all()
        or not diagnostic_strata.eq(23_024_760).all()
        or len(state_totals) != 10
        or not state_totals.eq(23_024_760).all()
    ):
        raise RuntimeError("aggregate diagnostic/state与全target事件数不守恒")
    overall, event_weighted_overall, zone_metric_means = (
        zone_balanced_overall_metrics(cells, numeric_columns=numeric_cells)
    )
    stratified_raw = (
        cells.groupby(
            [
                *RESULT_SCOPE_FIELDS,
                "method",
                "zone_or_farm",
                "predictor",
                "horizon_steps",
                "target_coverage",
            ],
            sort=True,
            as_index=False,
        )[numeric_cells]
        .sum()
        .reset_index(drop=True)
    )
    stratified = _metric_means(stratified_raw)
    action_summary = pd.concat(
        [
            actions.groupby(
                [*RESULT_SCOPE_FIELDS, "method", "selected_action"],
                sort=True,
                as_index=False,
            )["action_count"].sum().assign(regime="overall"),
            actions.groupby(
                [*RESULT_SCOPE_FIELDS, "method", "ramp_state", "selected_action"],
                sort=True,
                as_index=False,
            )["action_count"].sum().rename(columns={"ramp_state": "regime"}),
        ],
        ignore_index=True,
        sort=False,
    )
    action_summary = action_summary.sort_values(
        [*RESULT_SCOPE_FIELDS, "method", "regime", "selected_action"],
        kind="mergesort",
    ).reset_index(drop=True)
    action_totals = action_summary.groupby(
        [*RESULT_SCOPE_FIELDS, "method", "regime"]
    )["action_count"].transform("sum")
    action_summary["action_share"] = action_summary["action_count"] / action_totals
    (
        diagnostic_zone,
        secondary_zone,
        secondary_descriptive,
        clara_summary,
    ) = aggregate_diagnostic_metrics(diagnostics, actions)
    block_zone_totals = (
        pd.concat(zone_block_parts, ignore_index=True, sort=False)
        .groupby([*RESULT_SCOPE_FIELDS, "zone_or_farm", "method"], sort=True, as_index=False)[
            ["event_count", "errf_sum"]
        ]
        .sum()
        .reset_index(drop=True)
    )
    cell_zone_totals = zone_metric_means[
        [*RESULT_SCOPE_FIELDS, "zone_or_farm", "method", "event_count", "errf_sum"]
    ].sort_values([*RESULT_SCOPE_FIELDS, "zone_or_farm", "method"], kind="mergesort").reset_index(drop=True)
    block_zone_totals = block_zone_totals.sort_values(
        [*RESULT_SCOPE_FIELDS, "zone_or_farm", "method"], kind="mergesort"
    ).reset_index(drop=True)
    if not cell_zone_totals.equals(block_zone_totals):
        numeric_equal = np.allclose(
            cell_zone_totals[["event_count", "errf_sum"]].to_numpy(dtype=float),
            block_zone_totals[["event_count", "errf_sum"]].to_numpy(dtype=float),
            atol=1e-10,
            rtol=1e-12,
        )
        identity_equal = cell_zone_totals[
            [*RESULT_SCOPE_FIELDS, "zone_or_farm", "method"]
        ].equals(block_zone_totals[[*RESULT_SCOPE_FIELDS, "zone_or_farm", "method"]])
        if not numeric_equal or not identity_equal:
            raise RuntimeError("paired blocks与cell metrics的zone代数不闭合")
    design_zone_metrics = equal_design_zone_metrics(cells)
    zone_differences, inference = paired_inference_from_zone_metrics(
        design_zone_metrics
    )
    artifact_index = pd.DataFrame(index_rows).sort_values(
        ["evaluation_zone", "seed", "policy_mode", "evaluation_price_id"], kind="mergesort"
    ).reset_index(drop=True)
    expected_index = {
        (zone, seed, policy_mode, evaluation_price_id)
        for zone in core.FORMAL_ZONES
        for seed in range(3)
        for policy_mode in POLICY_MODES
        for evaluation_price_id in (row[0] for row in core.FORMAL_PRICE_ROWS)
    }
    observed_index = set(
        zip(
            artifact_index["evaluation_zone"].astype(str),
            artifact_index["seed"].astype(int),
            artifact_index["policy_mode"].astype(str),
            artifact_index["evaluation_price_id"].astype(str),
        )
    )
    if (
        observed_index != expected_index
        or artifact_index["manifest_relative_path"].astype(str).duplicated().any()
        or not artifact_index["manifest_sha256"].map(is_sha256).all()
    ):
        raise RuntimeError("aggregate compact artifact index轴/路径/SHA不闭合")
    return {
        "compact_artifact_index": artifact_index,
        "cell_metrics": cells,
        "action_counts": actions,
        "diagnostic_metrics": diagnostics,
        "clara_state_counts": clara_states,
        "event_conservation": conservation,
        "overall_ranking": overall.sort_values(
            ["policy_mode", "evaluation_price_id", "errf_rank_within_price", "method"], kind="mergesort"
        ).reset_index(drop=True),
        "event_weighted_overall_descriptive": event_weighted_overall,
        "zone_metric_means": zone_metric_means,
        "zone_design_equal_metrics": design_zone_metrics,
        "stratified_metrics": stratified,
        "action_share": action_summary,
        "diagnostic_zone_metrics": diagnostic_zone,
        "secondary_zone_balanced": secondary_zone,
        "secondary_event_weighted_descriptive": secondary_descriptive,
        "clara_support_guardrail_summary": clara_summary,
        "paired_zone_differences": zone_differences,
        "paired_inference": inference,
    }


def _selector_relative_path(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(SELECTOR_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError(f"selector工件路径逃逸output root: {path}") from error


def _manifest_payload_sha(payload: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(payload)
    output["payload_sha256"] = canonical_sha256(output)
    return output


def _verify_manifest_payload_sha(
    manifest: Mapping[str, Any], *, label: str
) -> None:
    observed = str(manifest.get("payload_sha256", ""))
    payload = {
        str(key): value
        for key, value in manifest.items()
        if str(key) != "payload_sha256"
    }
    if not is_sha256(observed) or canonical_sha256(payload) != observed:
        raise RuntimeError(f"{label} payload SHA失配")


def _iter_deep_replay_children(
    execution: Mapping[str, Any],
    *,
    records: dict[str, dict[str, Any]],
    memory_audit: dict[str, Any],
    token_path: Path | None = None,
    token_sha256: str | None = None,
) -> Iterable[Any]:
    """Yield one validated child at a time, retaining at most one 10-child parent."""

    artifacts = _artifacts_module()
    execution_sha = str(execution["execution_identity_sha256"])
    endpoint_root_sha = str(execution["endpoint_root_audit"]["sha256"])
    if records or memory_audit:
        raise RuntimeError("deep replay stream审计容器必须从空状态开始")
    process = psutil.Process()
    child_count = 0
    memory_audit.update(
        {
            "schema": "TEST_CLARA_GEFCOM_6A_REPLAY_STREAM_MEMORY_AUDIT_V1",
            "status": "IN_PROGRESS",
            "retained_parent_limit": 1,
            "retained_loaded_child_limit": 10,
            "max_loaded_parent_count_observed": 0,
            "max_loaded_parent_child_count": 0,
            "max_loaded_parent_paired_block_rows": 0,
            "total_paired_block_rows_seen": 0,
            "paired_block_retention_identity": (
                "REDUCE_EACH_CHILD_IMMEDIATELY_TO_ZONE_METHOD_SCOPE_TOTALS;"
                "NEVER_CONCAT_RAW_PAIRED_BLOCKS_ACROSS_CHILDREN"
            ),
            "full_raw_paired_materialization_forbidden": True,
            "peak_process_rss_bytes_observed": int(
                process.memory_info().rss
            ),
        }
    )
    for zone in core.FORMAL_ZONES:
        for seed in range(3):
            if token_path is not None:
                validate_preflight_token(
                    token_path,
                    expected_token_sha256=str(token_sha256),
                    zone=zone,
                    seed=seed,
                )
            parent_id = fit_parent_id(zone, seed)
            replay_manifest_path = (
                REPLAY_ROOT / parent_id / "replay_parent_manifest.json"
            )
            fit_manifest_path = FIT_ROOT / parent_id / "fit_parent_manifest.json"
            target_manifest_path = REPLAY_ROOT / parent_id / "target_input_manifest.json"
            for required in (
                replay_manifest_path,
                fit_manifest_path,
                target_manifest_path,
            ):
                if not required.is_file():
                    raise RuntimeError(f"aggregate缺少已封存输入: {required}")
            target_sha = sha256_file(target_manifest_path)
            loaded_parent = artifacts.load_replay_parent(
                replay_manifest_path,
                expected_zone=zone,
                expected_seed=seed,
                expected_fit_parent_manifest_path=fit_manifest_path,
                expected_endpoint_root_sha256=endpoint_root_sha,
                expected_target_input_manifest_path=target_manifest_path,
                expected_target_input_manifest_sha256=target_sha,
                expected_execution_identity_sha256=execution_sha,
                formal_identity=True,
            )
            if len(loaded_parent.children) != 10:
                raise RuntimeError("deep replay stream parent非exact10 children")
            memory_audit["max_loaded_parent_count_observed"] = 1
            assert_cross_mode_child_algebra(
                tuple(loaded_parent.children.values())
            )
            parent_paired_rows = int(
                sum(
                    len(child.paired_blocks)
                    for child in loaded_parent.children.values()
                )
            )
            memory_audit["max_loaded_parent_child_count"] = max(
                int(memory_audit["max_loaded_parent_child_count"]),
                len(loaded_parent.children),
            )
            memory_audit["max_loaded_parent_paired_block_rows"] = max(
                int(memory_audit["max_loaded_parent_paired_block_rows"]),
                parent_paired_rows,
            )
            memory_audit["total_paired_block_rows_seen"] = int(
                memory_audit["total_paired_block_rows_seen"]
            ) + parent_paired_rows
            memory_audit["peak_process_rss_bytes_observed"] = max(
                int(memory_audit["peak_process_rss_bytes_observed"]),
                int(process.memory_info().rss),
            )
            if token_path is not None:
                validate_preflight_token(
                    token_path,
                    expected_token_sha256=str(token_sha256),
                    zone=zone,
                    seed=seed,
                )
            child_records: dict[str, dict[str, str]] = {}
            for policy_mode in POLICY_MODES:
                for evaluation_price_id in (
                    row[0] for row in core.FORMAL_PRICE_ROWS
                ):
                    child_key = replay_child_id(
                        policy_mode, evaluation_price_id
                    )
                    policy_price_id = (
                        evaluation_price_id
                        if policy_mode == POLICY_MODES[0]
                        else MAIN_PRICE_ID
                    )
                    child_manifest_path = (
                        REPLAY_ROOT
                        / parent_id
                        / "children"
                        / child_key
                        / "manifest.json"
                    )
                    child_sha = sha256_file(child_manifest_path)
                    loaded_child = loaded_parent.children.pop(child_key)
                    loaded_child.manifest["_manifest_sha256"] = child_sha
                    loaded_child.manifest["_manifest_relative_path"] = (
                        _selector_relative_path(child_manifest_path)
                    )
                    if (
                        str(loaded_child.manifest.get("policy_mode"))
                        != policy_mode
                        or str(
                            loaded_child.manifest.get("evaluation_price_id")
                        )
                        != evaluation_price_id
                        or str(loaded_child.manifest.get("policy_price_id"))
                        != policy_price_id
                    ):
                        raise RuntimeError(
                            f"aggregate replay child双mode轴串线: {child_key}"
                        )
                    child_records[child_key] = {
                        "policy_mode": policy_mode,
                        "evaluation_price_id": evaluation_price_id,
                        "policy_price_id": policy_price_id,
                        "manifest_relative_path": _selector_relative_path(
                            child_manifest_path
                        ),
                        "manifest_sha256": child_sha,
                        "identity_sha256": str(
                            loaded_child.manifest[
                                "replay_child_identity_sha256"
                            ]
                        ),
                    }
                    child_count += 1
                    yield loaded_child
                    del loaded_child
            if loaded_parent.children:
                raise RuntimeError("deep replay stream未释放parent全部children")
            records[parent_id] = {
                "zone": zone,
                "seed": seed,
                "replay_parent_manifest_relative_path": _selector_relative_path(
                    replay_manifest_path
                ),
                "replay_parent_manifest_sha256": sha256_file(
                    replay_manifest_path
                ),
                "fit_parent_manifest_relative_path": _selector_relative_path(
                    fit_manifest_path
                ),
                "fit_parent_manifest_sha256": sha256_file(fit_manifest_path),
                "target_input_manifest_relative_path": _selector_relative_path(
                    target_manifest_path
                ),
                "target_input_manifest_sha256": target_sha,
                "policy_evaluation_children": child_records,
            }
            del loaded_parent
            memory_audit["peak_process_rss_bytes_observed"] = max(
                int(memory_audit["peak_process_rss_bytes_observed"]),
                int(process.memory_info().rss),
            )
    if len(records) != 30 or child_count != 300:
        raise RuntimeError("aggregate deep replay输入未闭合30 parent/300 child")
    memory_audit.update(
        {
            "status": "PASS",
            "parent_count": len(records),
            "child_count": child_count,
        }
    )


def _aggregate_summary_markdown(outputs: Mapping[str, pd.DataFrame]) -> str:
    overall = outputs["overall_ranking"]
    inference = outputs["paired_inference"]
    lines = [
        "# GEFCom six-action compact aggregate",
        "",
        "Overall rankings use event means within each held-out zone followed by equal weight across the ten zones.",
        "Paired CLARA comparisons first average the exact 660 seed×predictor×horizon×coverage design-cell event means within each zone, then compare the ten zone-level values.",
        "Selected-endpoint TOWR/TUWR/ARD and guardrail rates are reconstructed from integer numerator/denominator sufficient statistics; candidate-history reliability is never reused.",
        "The event-weighted table is descriptive only. No event-level method-by-policy-mode-by-evaluation-price table is materialized.",
        "",
        "## Contract closure",
        "",
        f"- Method-policy-mode-evaluation-price ranking rows: {len(overall)} (expected 90)",
        f"- CLARA paired comparisons: {len(inference)} (expected 80)",
        "- Exact sign-flip configurations per comparison: 1024",
        "- Paired zone-bootstrap replicates per comparison: 5000",
        "",
        "## Files",
        "",
        "See `overall_ranking.csv`, `paired_inference.csv`, and the Parquet sufficient-statistic layers indexed by `compact_artifact_index.parquet`.",
        "",
    ]
    return "\n".join(lines)


def _write_aggregate_staging(
    staging: Path,
    *,
    outputs: Mapping[str, pd.DataFrame],
    execution: Mapping[str, Any],
    replay_parent_records: Mapping[str, Mapping[str, Any]],
    stream_memory_audit: Mapping[str, Any],
) -> dict[str, Any]:
    staging.mkdir(parents=False, exist_ok=False)
    file_records: dict[str, dict[str, Any]] = {}
    for name in AGGREGATE_OUTPUT_NAMES:
        frame = outputs[name]
        path = staging / f"{name}.parquet"
        atomic_parquet(path, frame)
        file_records[name] = {
            "relative_path": path.name,
            "sha256": sha256_file(path),
            "row_count": len(frame),
            "columns": list(frame.columns),
        }
    csv_records: dict[str, dict[str, Any]] = {}
    for name in (
        "overall_ranking",
        "action_share",
        "secondary_zone_balanced",
        "clara_support_guardrail_summary",
        "paired_inference",
    ):
        path = staging / f"{name}.csv"
        atomic_csv(path, outputs[name])
        csv_records[name] = {
            "relative_path": path.name,
            "sha256": sha256_file(path),
            "row_count": len(outputs[name]),
        }
    summary_path = staging / "SUMMARY.md"
    atomic_text(summary_path, _aggregate_summary_markdown(outputs))
    statistics = statistics_contract_audit()
    conservation = outputs["event_conservation"]
    manifest = _manifest_payload_sha(
        {
            "schema": AGGREGATE_SCHEMA,
            "status": "PASS",
            "created_at_utc": utc_now(),
            "execution_identity_sha256": str(
                execution["execution_identity_sha256"]
            ),
            "endpoint_root_sha256": str(
                execution["endpoint_root_audit"]["sha256"]
            ),
            "statistics_contract_sha256": statistics[
                "statistics_contract_sha256"
            ],
            "statistics_contract": statistics,
            "replay_parent_count": len(replay_parent_records),
            "replay_child_count": 300,
            "input_replay_parents": dict(replay_parent_records),
            "stream_memory_audit": dict(stream_memory_audit),
            "output_files": file_records,
            "csv_files": csv_records,
            "summary_markdown": {
                "relative_path": summary_path.name,
                "sha256": sha256_file(summary_path),
            },
            "method_policy_mode_price_row_count": len(conservation),
            "event_count_per_method_policy_mode_price": 23_024_760,
            "method_event_policy_mode_price_score_count": 2_072_228_400,
            "full_event_materialization": False,
            "primary_aggregation_identity": (
                "EVENT_MEAN_WITHIN_ZONE_THEN_EQUAL_WEIGHT_TEN_ZONES"
            ),
            "paired_inference_aggregation_identity": (
                "EVENT_MEAN_WITHIN_EXACT_660_DESIGN_CELLS_THEN_EQUAL_CELL_MEAN_WITHIN_ZONE_THEN_TEN_ZONE_PAIRING"
            ),
            "selected_reliability_identity": (
                "SELECTED_ENDPOINT_168H_ROLLING_COUNTS;RATIO_OF_INTEGER_SUFFICIENT_STATISTICS"
            ),
        }
    )
    atomic_json(staging / "manifest.json", manifest)
    return manifest


def load_aggregate_root(
    execution: Mapping[str, Any],
    *,
    deep_inputs: bool,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], list[Any]]:
    manifest_path = AGGREGATE_ROOT / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("aggregate root manifest缺失")
    if _temporary_entries(AGGREGATE_ROOT) or any(
        ".tmp__" in path.name for path in AGGREGATE_ROOT.rglob("*")
    ):
        raise RuntimeError("aggregate root含残留temp")
    manifest = load_json(manifest_path)
    _verify_manifest_payload_sha(manifest, label="aggregate manifest")
    statistics = statistics_contract_audit()
    expected_fields = {
        "schema",
        "status",
        "created_at_utc",
        "execution_identity_sha256",
        "endpoint_root_sha256",
        "statistics_contract_sha256",
        "statistics_contract",
        "replay_parent_count",
        "replay_child_count",
        "input_replay_parents",
        "stream_memory_audit",
        "output_files",
        "csv_files",
        "summary_markdown",
        "method_policy_mode_price_row_count",
        "event_count_per_method_policy_mode_price",
        "method_event_policy_mode_price_score_count",
        "full_event_materialization",
        "primary_aggregation_identity",
        "paired_inference_aggregation_identity",
        "selected_reliability_identity",
        "payload_sha256",
    }
    if set(manifest) != expected_fields:
        raise RuntimeError("aggregate manifest schema字段漂移")
    if (
        manifest.get("schema") != AGGREGATE_SCHEMA
        or manifest.get("status") != "PASS"
        or str(manifest.get("execution_identity_sha256"))
        != str(execution["execution_identity_sha256"])
        or str(manifest.get("endpoint_root_sha256"))
        != str(execution["endpoint_root_audit"]["sha256"])
        or str(manifest.get("statistics_contract_sha256"))
        != statistics["statistics_contract_sha256"]
        or manifest.get("statistics_contract") != statistics
        or int(manifest.get("replay_parent_count", -1)) != 30
        or int(manifest.get("replay_child_count", -1)) != 300
        or dict(manifest.get("stream_memory_audit", {})).get("status")
        != "PASS"
        or int(
            dict(manifest.get("stream_memory_audit", {})).get(
                "retained_parent_limit", -1
            )
        )
        != 1
        or int(
            dict(manifest.get("stream_memory_audit", {})).get(
                "max_loaded_parent_count_observed", -1
            )
        )
        > 1
        or int(
            dict(manifest.get("stream_memory_audit", {})).get(
                "max_loaded_parent_child_count", -1
            )
        )
        > 10
        or int(
            dict(manifest.get("stream_memory_audit", {})).get(
                "parent_count", -1
            )
        )
        != 30
        or int(
            dict(manifest.get("stream_memory_audit", {})).get(
                "child_count", -1
            )
        )
        != 300
        or not bool(
            dict(manifest.get("stream_memory_audit", {})).get(
                "full_raw_paired_materialization_forbidden", False
            )
        )
        or int(manifest.get("method_policy_mode_price_row_count", -1)) != 90
        or int(
            manifest.get("event_count_per_method_policy_mode_price", -1)
        )
        != 23_024_760
        or int(
            manifest.get("method_event_policy_mode_price_score_count", -1)
        )
        != 2_072_228_400
        or bool(manifest.get("full_event_materialization"))
        or manifest.get("primary_aggregation_identity")
        != "EVENT_MEAN_WITHIN_ZONE_THEN_EQUAL_WEIGHT_TEN_ZONES"
        or manifest.get("paired_inference_aggregation_identity")
        != "EVENT_MEAN_WITHIN_EXACT_660_DESIGN_CELLS_THEN_EQUAL_CELL_MEAN_WITHIN_ZONE_THEN_TEN_ZONE_PAIRING"
        or manifest.get("selected_reliability_identity")
        != "SELECTED_ENDPOINT_168H_ROLLING_COUNTS;RATIO_OF_INTEGER_SUFFICIENT_STATISTICS"
    ):
        raise RuntimeError("aggregate manifest科学身份/守恒门失配")
    parent_records = dict(manifest.get("input_replay_parents", {}))
    expected_parent_ids = {
        fit_parent_id(zone, seed)
        for zone in core.FORMAL_ZONES
        for seed in range(3)
    }
    if set(parent_records) != expected_parent_ids:
        raise RuntimeError("aggregate input parent账本不是exact30")
    for parent_id, record in parent_records.items():
        replay_path = SELECTOR_ROOT / str(
            record["replay_parent_manifest_relative_path"]
        )
        if (
            not replay_path.is_file()
            or sha256_file(replay_path)
            != str(record["replay_parent_manifest_sha256"])
        ):
            raise RuntimeError(f"aggregate input parent SHA漂移: {parent_id}")
    outputs: dict[str, pd.DataFrame] = {}
    output_records = dict(manifest.get("output_files", {}))
    if set(output_records) != set(AGGREGATE_OUTPUT_NAMES):
        raise RuntimeError("aggregate output文件集不精确")
    allowed_names = {"manifest.json", "SUMMARY.md"}
    for name in AGGREGATE_OUTPUT_NAMES:
        record = dict(output_records[name])
        path = AGGREGATE_ROOT / str(record.get("relative_path", ""))
        allowed_names.add(path.name)
        if (
            path.parent.resolve() != AGGREGATE_ROOT.resolve()
            or not path.is_file()
            or sha256_file(path) != str(record.get("sha256"))
        ):
            raise RuntimeError(f"aggregate output SHA/路径失配: {name}")
        frame = pd.read_parquet(path)
        if (
            len(frame) != int(record.get("row_count", -1))
            or list(frame.columns) != list(record.get("columns", []))
        ):
            raise RuntimeError(f"aggregate output schema/行数失配: {name}")
        outputs[name] = frame
    for name, record in dict(manifest.get("csv_files", {})).items():
        if name not in {
            "overall_ranking",
            "action_share",
            "secondary_zone_balanced",
            "clara_support_guardrail_summary",
            "paired_inference",
        }:
            raise RuntimeError("aggregate CSV文件集含额外项")
        path = AGGREGATE_ROOT / str(record.get("relative_path", ""))
        allowed_names.add(path.name)
        if (
            path.parent.resolve() != AGGREGATE_ROOT.resolve()
            or not path.is_file()
            or sha256_file(path) != str(record.get("sha256"))
        ):
            raise RuntimeError(f"aggregate CSV SHA/路径失配: {name}")
    summary = dict(manifest.get("summary_markdown", {}))
    summary_path = AGGREGATE_ROOT / str(summary.get("relative_path", ""))
    if (
        summary_path.parent.resolve() != AGGREGATE_ROOT.resolve()
        or not summary_path.is_file()
        or sha256_file(summary_path) != str(summary.get("sha256"))
    ):
        raise RuntimeError("aggregate summary markdown SHA/路径失配")
    observed_names = {path.name for path in AGGREGATE_ROOT.iterdir()}
    if observed_names != allowed_names:
        raise RuntimeError("aggregate root含额外或缺失文件")
    conservation = outputs["event_conservation"]
    overall = outputs["overall_ranking"]
    index = outputs["compact_artifact_index"]
    inference = outputs["paired_inference"]
    diagnostic = outputs["diagnostic_metrics"]
    secondary = outputs["secondary_zone_balanced"]
    clara_states = outputs["clara_state_counts"]
    diagnostic_overall = diagnostic[
        diagnostic["regime"].astype(str).eq("overall")
    ].groupby([*RESULT_SCOPE_FIELDS, "method"], sort=True)["event_count"].sum()
    state_totals = clara_states.groupby(
        list(RESULT_SCOPE_FIELDS), sort=True
    )["event_count"].sum()
    if (
        len(conservation) != 90
        or not conservation["event_count"].eq(23_024_760).all()
        or len(overall) != 90
        or not overall["zone_count"].eq(10).all()
        or not overall["aggregation_identity"].eq(
            "EVENT_MEAN_WITHIN_ZONE_THEN_EQUAL_WEIGHT_TEN_ZONES"
        ).all()
        or len(index) != 300
        or not index["manifest_sha256"].map(is_sha256).all()
        or index["manifest_relative_path"].astype(str).eq("").any()
        or len(inference) != 80
        or not inference["exact_sign_flip_configuration_count"].eq(1024).all()
        or not np.allclose(
            inference["exact_sign_flip_p_value"],
            inference["exact_sign_flip_extreme_count"] / 1024.0,
            atol=0.0,
            rtol=0.0,
        )
        or len(diagnostic_overall) != 90
        or not diagnostic_overall.eq(23_024_760).all()
        or len(state_totals) != 10
        or not state_totals.eq(23_024_760).all()
        or len(secondary) != 270
        or set(secondary["regime"].astype(str))
        != {"overall", "ordinary", "ramp"}
    ):
        raise RuntimeError("aggregate output全局统计/守恒门失配")
    if deep_inputs:
        current_records: dict[str, dict[str, Any]] = {}
        current_memory_audit: dict[str, Any] = {}
        for child in _iter_deep_replay_children(
            execution,
            records=current_records,
            memory_audit=current_memory_audit,
        ):
            del child
        if current_records != parent_records:
            raise RuntimeError("aggregate账本与当前deep replay parent闭合失败")
        if current_memory_audit.get("status") != "PASS":
            raise RuntimeError("aggregate deep input流式内存审计未PASS")
    return manifest, outputs, []


def aggregate_results(
    *, token_path: Path, token_sha256: str
) -> dict[str, Any]:
    token = validate_preflight_token(
        token_path, expected_token_sha256=token_sha256
    )
    execution = dict(token["execution_identity"])
    residual = sorted(
        path.name
        for path in AGGREGATE_ROOT.parent.glob(f".{AGGREGATE_ROOT.name}.tmp__*")
    )
    if residual:
        raise RuntimeError(f"aggregate残留staging目录阻断: {residual}")
    if AGGREGATE_ROOT.exists():
        manifest, _, _ = load_aggregate_root(execution, deep_inputs=True)
        return {
            "schema": AGGREGATE_SCHEMA,
            "status": "COMPLETE",
            "resumed": True,
            "manifest_path": str(AGGREGATE_ROOT / "manifest.json"),
            "manifest_sha256": sha256_file(AGGREGATE_ROOT / "manifest.json"),
            "payload_sha256": manifest["payload_sha256"],
        }
    parent_records: dict[str, dict[str, Any]] = {}
    stream_memory_audit: dict[str, Any] = {}
    replay_children = _iter_deep_replay_children(
        execution,
        records=parent_records,
        memory_audit=stream_memory_audit,
        token_path=token_path,
        token_sha256=token_sha256,
    )
    outputs = aggregate_compact_children(replay_children)
    if stream_memory_audit.get("status") != "PASS":
        raise RuntimeError("aggregate replay child stream未完整消费")
    validate_preflight_token(token_path, expected_token_sha256=token_sha256)
    AGGREGATE_ROOT.parent.mkdir(parents=True, exist_ok=True)
    staging = AGGREGATE_ROOT.parent / (
        f".{AGGREGATE_ROOT.name}.tmp__{uuid.uuid4().hex}"
    )
    _write_aggregate_staging(
        staging,
        outputs=outputs,
        execution=execution,
        replay_parent_records=parent_records,
        stream_memory_audit=stream_memory_audit,
    )
    validate_preflight_token(token_path, expected_token_sha256=token_sha256)
    os.replace(staging, AGGREGATE_ROOT)
    manifest, _, _ = load_aggregate_root(execution, deep_inputs=False)
    return {
        "schema": AGGREGATE_SCHEMA,
        "status": "COMPLETE",
        "resumed": False,
        "manifest_path": str(AGGREGATE_ROOT / "manifest.json"),
        "manifest_sha256": sha256_file(AGGREGATE_ROOT / "manifest.json"),
        "payload_sha256": manifest["payload_sha256"],
    }


def load_qa_root(
    execution: Mapping[str, Any], *, validate_aggregate: bool
) -> dict[str, Any]:
    qa_module = _independent_qa_module()
    if str(qa_module.module_sha256()) != str(
        execution["code_identity"]["module_file_sha256"]["independent_qa"]
    ):
        raise RuntimeError("QA root当前独立QA模块与execution identity失配")
    manifest_path = QA_ROOT / "manifest.json"
    checks_path = QA_ROOT / "qa_checks.csv"
    summary_path = QA_ROOT / "QA_SUMMARY.md"
    if not all(path.is_file() for path in (manifest_path, checks_path, summary_path)):
        raise RuntimeError("QA root缺manifest/checks/summary")
    if any(".tmp__" in path.name for path in QA_ROOT.rglob("*")):
        raise RuntimeError("QA root含残留temp")
    observed_names = {path.name for path in QA_ROOT.iterdir()}
    if observed_names != {"manifest.json", "qa_checks.csv", "QA_SUMMARY.md"}:
        raise RuntimeError("QA root含额外或缺失文件")
    manifest = load_json(manifest_path)
    _verify_manifest_payload_sha(manifest, label="QA manifest")
    expected_fields = {
        "schema",
        "status",
        "created_at_utc",
        "execution_identity_sha256",
        "endpoint_root_sha256",
        "aggregate_manifest_relative_path",
        "aggregate_manifest_sha256",
        "aggregate_payload_sha256",
        "independent_qa_module_path",
        "independent_qa_module_sha256",
        "replay_parent_count",
        "replay_child_count",
        "input_replay_parents",
        "stream_memory_audit",
        "independent_streaming_identity",
        "method_policy_mode_price_row_count",
        "event_count_per_method_policy_mode_price",
        "method_event_policy_mode_price_score_count",
        "check_count",
        "failed_check_count",
        "qa_checks",
        "qa_summary",
        "payload_sha256",
    }
    if set(manifest) != expected_fields:
        raise RuntimeError("QA manifest schema字段漂移")
    aggregate_path = AGGREGATE_ROOT / "manifest.json"
    if (
        manifest.get("schema") != QA_SCHEMA
        or manifest.get("status") != "PASS"
        or str(manifest.get("execution_identity_sha256"))
        != str(execution["execution_identity_sha256"])
        or str(manifest.get("endpoint_root_sha256"))
        != str(execution["endpoint_root_audit"]["sha256"])
        or Path(
            SELECTOR_ROOT / str(manifest.get("aggregate_manifest_relative_path", ""))
        ).resolve()
        != aggregate_path.resolve()
        or not aggregate_path.is_file()
        or sha256_file(aggregate_path)
        != str(manifest.get("aggregate_manifest_sha256"))
        or int(manifest.get("replay_parent_count", -1)) != 30
        or int(manifest.get("replay_child_count", -1)) != 300
        or dict(manifest.get("stream_memory_audit", {})).get("status")
        != "PASS"
        or int(
            dict(manifest.get("stream_memory_audit", {})).get(
                "retained_parent_limit", -1
            )
        )
        != 1
        or int(
            dict(manifest.get("stream_memory_audit", {})).get(
                "max_loaded_parent_child_count", -1
            )
        )
        > 10
        or int(
            dict(manifest.get("stream_memory_audit", {})).get(
                "parent_count", -1
            )
        )
        != 30
        or int(
            dict(manifest.get("stream_memory_audit", {})).get(
                "child_count", -1
            )
        )
        != 300
        or manifest.get("independent_streaming_identity")
        != "SECOND_INDEPENDENT_PARENT_STREAM;MAX_ONE_PARENT_TEN_RAW_CHILDREN;PAIRED_REDUCED_PER_CHILD_TO_ZONE_TOTALS"
        or int(manifest.get("method_policy_mode_price_row_count", -1)) != 90
        or int(
            manifest.get("event_count_per_method_policy_mode_price", -1)
        )
        != 23_024_760
        or int(
            manifest.get("method_event_policy_mode_price_score_count", -1)
        )
        != 2_072_228_400
        or int(manifest.get("failed_check_count", -1)) != 0
        or int(manifest.get("check_count", -1)) <= 0
        or Path(str(manifest.get("independent_qa_module_path", ""))).resolve()
        != INDEPENDENT_QA_PATH.resolve()
        or str(manifest.get("independent_qa_module_sha256"))
        != sha256_file(INDEPENDENT_QA_PATH)
    ):
        raise RuntimeError("QA manifest身份/守恒门失配")
    qa_parent_records = dict(manifest.get("input_replay_parents", {}))
    expected_parent_ids = {
        fit_parent_id(zone, seed)
        for zone in core.FORMAL_ZONES
        for seed in range(3)
    }
    if set(qa_parent_records) != expected_parent_ids:
        raise RuntimeError("QA input replay parent账本不是exact30")
    checks_record = dict(manifest.get("qa_checks", {}))
    summary_record = dict(manifest.get("qa_summary", {}))
    if (
        checks_record.get("relative_path") != checks_path.name
        or sha256_file(checks_path) != str(checks_record.get("sha256"))
        or summary_record.get("relative_path") != summary_path.name
        or sha256_file(summary_path) != str(summary_record.get("sha256"))
    ):
        raise RuntimeError("QA checks/summary SHA失配")
    checks = pd.read_csv(checks_path)
    if (
        len(checks) != int(manifest["check_count"])
        or set(checks.columns) != {"check_id", "status", "detail"}
        or not checks["status"].astype(str).eq("PASS").all()
        or checks["check_id"].astype(str).duplicated().any()
    ):
        raise RuntimeError("QA checks表未全PASS或结构漂移")
    if validate_aggregate:
        aggregate, _, _ = load_aggregate_root(execution, deep_inputs=True)
        if str(aggregate.get("payload_sha256")) != str(
            manifest.get("aggregate_payload_sha256")
        ):
            raise RuntimeError("QA与当前aggregate payload SHA失配")
        if qa_parent_records != dict(aggregate.get("input_replay_parents", {})):
            raise RuntimeError("QA独立输入账本与aggregate输入账本失配")
    return manifest


def run_qa(*, token_path: Path, token_sha256: str) -> dict[str, Any]:
    token = validate_preflight_token(
        token_path, expected_token_sha256=token_sha256
    )
    execution = dict(token["execution_identity"])
    residual = sorted(
        path.name for path in QA_ROOT.parent.glob(f".{QA_ROOT.name}.tmp__*")
    )
    if residual:
        raise RuntimeError(f"QA残留staging目录阻断: {residual}")
    if QA_ROOT.exists():
        manifest = load_qa_root(execution, validate_aggregate=True)
        return {
            "schema": QA_SCHEMA,
            "status": "COMPLETE",
            "resumed": True,
            "manifest_path": str(QA_ROOT / "manifest.json"),
            "manifest_sha256": sha256_file(QA_ROOT / "manifest.json"),
            "payload_sha256": manifest["payload_sha256"],
        }
    aggregate_manifest, stored_outputs, _ = load_aggregate_root(
        execution, deep_inputs=False
    )
    validate_preflight_token(token_path, expected_token_sha256=token_sha256)
    independent_qa = _independent_qa_module()
    qa_parent_records: dict[str, dict[str, Any]] = {}
    qa_stream_memory_audit: dict[str, Any] = {}
    replay_children = _iter_deep_replay_children(
        execution,
        records=qa_parent_records,
        memory_audit=qa_stream_memory_audit,
        token_path=token_path,
        token_sha256=token_sha256,
    )
    checks_frame = independent_qa.validate_aggregate_outputs(
        replay_children, stored_outputs
    )
    if qa_stream_memory_audit.get("status") != "PASS":
        raise RuntimeError("independent QA replay child stream未完整消费")
    if qa_parent_records != dict(
        aggregate_manifest.get("input_replay_parents", {})
    ):
        raise RuntimeError("independent QA输入账本与aggregate输入账本失配")
    if (
        checks_frame.empty
        or set(checks_frame.columns) != {"check_id", "status", "detail"}
        or not checks_frame["status"].astype(str).eq("PASS").all()
        or checks_frame["check_id"].astype(str).duplicated().any()
    ):
        raise RuntimeError("independent QA模块未返回唯一全PASS checks")
    validate_preflight_token(token_path, expected_token_sha256=token_sha256)
    QA_ROOT.parent.mkdir(parents=True, exist_ok=True)
    staging = QA_ROOT.parent / f".{QA_ROOT.name}.tmp__{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    checks_path = staging / "qa_checks.csv"
    atomic_csv(checks_path, checks_frame)
    summary_path = staging / "QA_SUMMARY.md"
    atomic_text(
        summary_path,
        "\n".join(
            [
                "# Independent QA: GEFCom six-action selector run",
                "",
                f"All {len(checks_frame)} independent checks passed.",
                "",
                "A separately SHA-pinned QA module consumed a second independent stream of all 30 replay parents and 300 policy-mode/evaluation-price children. It retained at most one ten-child parent, reduced every paired-block frame immediately to zone sufficient totals, independently recomputed every compact aggregate table, and closed 9 methods × 2 policy modes × 5 evaluation prices × 23,024,760 events without event-level materialization.",
                "",
            ]
        ),
    )
    manifest = _manifest_payload_sha(
        {
            "schema": QA_SCHEMA,
            "status": "PASS",
            "created_at_utc": utc_now(),
            "execution_identity_sha256": str(
                execution["execution_identity_sha256"]
            ),
            "endpoint_root_sha256": str(
                execution["endpoint_root_audit"]["sha256"]
            ),
            "aggregate_manifest_relative_path": _selector_relative_path(
                AGGREGATE_ROOT / "manifest.json"
            ),
            "aggregate_manifest_sha256": sha256_file(
                AGGREGATE_ROOT / "manifest.json"
            ),
            "aggregate_payload_sha256": aggregate_manifest["payload_sha256"],
            "independent_qa_module_path": str(INDEPENDENT_QA_PATH.resolve()),
            "independent_qa_module_sha256": sha256_file(INDEPENDENT_QA_PATH),
            "replay_parent_count": 30,
            "replay_child_count": 300,
            "input_replay_parents": qa_parent_records,
            "stream_memory_audit": qa_stream_memory_audit,
            "independent_streaming_identity": (
                "SECOND_INDEPENDENT_PARENT_STREAM;MAX_ONE_PARENT_TEN_RAW_CHILDREN;"
                "PAIRED_REDUCED_PER_CHILD_TO_ZONE_TOTALS"
            ),
            "method_policy_mode_price_row_count": 90,
            "event_count_per_method_policy_mode_price": 23_024_760,
            "method_event_policy_mode_price_score_count": 2_072_228_400,
            "check_count": len(checks_frame),
            "failed_check_count": 0,
            "qa_checks": {
                "relative_path": checks_path.name,
                "sha256": sha256_file(checks_path),
            },
            "qa_summary": {
                "relative_path": summary_path.name,
                "sha256": sha256_file(summary_path),
            },
        }
    )
    atomic_json(staging / "manifest.json", manifest)
    validate_preflight_token(token_path, expected_token_sha256=token_sha256)
    os.replace(staging, QA_ROOT)
    final = load_qa_root(execution, validate_aggregate=False)
    return {
        "schema": QA_SCHEMA,
        "status": "COMPLETE",
        "resumed": False,
        "manifest_path": str(QA_ROOT / "manifest.json"),
        "manifest_sha256": sha256_file(QA_ROOT / "manifest.json"),
        "payload_sha256": final["payload_sha256"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    subparsers.add_parser("status")
    design = subparsers.add_parser("design-preflight")
    design.add_argument("--write-report", action="store_true")
    formal = subparsers.add_parser("preflight")
    formal.add_argument("--write-report", action="store_true")
    for name in ("fit-one", "replay-one"):
        current = subparsers.add_parser(name)
        current.add_argument("--zone", required=True, choices=core.FORMAL_ZONES)
        current.add_argument("--seed", required=True, type=int, choices=range(3))
        current.add_argument("--token", required=True, type=Path)
        current.add_argument("--token-sha256", required=True)
    for name in ("aggregate", "qa"):
        current = subparsers.add_parser(name)
        current.add_argument("--token", required=True, type=Path)
        current.add_argument("--token-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "plan":
        result = write_plan()
    elif args.command == "status":
        result = selector_status()
    elif args.command == "design-preflight":
        result = preflight(write_report=args.write_report, require_release=False)
    elif args.command == "preflight":
        result = preflight(write_report=args.write_report, require_release=True)
    elif args.command == "fit-one":
        result = run_fit_parent(
            zone=args.zone,
            seed=args.seed,
            token_path=args.token,
            token_sha256=args.token_sha256,
        )
    elif args.command == "replay-one":
        result = run_replay_parent(
            zone=args.zone,
            seed=args.seed,
            token_path=args.token,
            token_sha256=args.token_sha256,
        )
    elif args.command == "aggregate":
        result = aggregate_results(
            token_path=args.token, token_sha256=args.token_sha256
        )
    elif args.command == "qa":
        result = run_qa(token_path=args.token, token_sha256=args.token_sha256)
    else:  # pragma: no cover
        raise RuntimeError(f"未知命令: {args.command}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, default=str))
    return 0 if result.get("status") in {"PASS", "COMPLETE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
