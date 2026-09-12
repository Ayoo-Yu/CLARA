"""Atomic, fail-closed artifacts for the GEFCom five-price six-action selectors.

This module deliberately contains no fitting or replay algorithm.  It serializes
only compact policies/audits, binds every artifact to the scientific identities
emitted by :mod:`gefcom_six_action_selector_core_v1`, and never persists the full
method-event-price score table.  CART is stored as numeric tree arrays plus JSON;
no pickle/joblib payload is accepted or executed.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier, _tree


TEST_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = TEST_ROOT.parent
AUTH_CODE = REVISION_ROOT / "_权威代码" / "code"
for directory in (Path(__file__).parent, AUTH_CODE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import gefcom_six_action_selector_core_v1 as core  # noqa: E402
from baseline_common import FrozenStateEncoder  # noqa: E402
from clara_event_contract import load_frozen_contracts  # noqa: E402
from extended_selector_adapters import ExtendedCartSelector  # noqa: E402


IMPORTED_WRITER_MODULE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
IMPORTED_CORE_MODULE_SHA256 = hashlib.sha256(
    Path(core.__file__).read_bytes()
).hexdigest()


SCHEMA = "TEST_CLARA_GEFCOM_6A_SELECTOR_ARTIFACTS_V1"
FIT_CHILD_SCHEMA = "TEST_CLARA_GEFCOM_6A_FIT_PRICE_ARTIFACT_V1"
FIT_PARENT_SCHEMA = "TEST_CLARA_GEFCOM_6A_FIT_PARENT_ARTIFACT_V1"
REPLAY_CHILD_SCHEMA = "TEST_CLARA_GEFCOM_6A_REPLAY_POLICY_EVALUATION_ARTIFACT_V2"
REPLAY_PARENT_SCHEMA = "TEST_CLARA_GEFCOM_6A_REPLAY_DUAL_POLICY_PARENT_ARTIFACT_V2"
CART_MODEL_SCHEMA = "TEST_CLARA_GEFCOM_6A_CART_TREE_ARRAYS_V1"
CART_GINI_NUMERIC_ATOL = 1e-10

FORMAL_SELECTOR_ROOT = (
    TEST_ROOT / "results_raw" / "gefcom_six_action_price_full_v1" / "selector_v1"
)
FORMAL_FIT_ROOT = FORMAL_SELECTOR_ROOT / "fit_parents"
FORMAL_REPLAY_ROOT = FORMAL_SELECTOR_ROOT / "replay_parents"
FORMAL_AGGREGATE_ROOT = FORMAL_SELECTOR_ROOT / "aggregate_v1"
FORMAL_QA_ROOT = FORMAL_SELECTOR_ROOT / "qa_v1"
FORMAL_INDEX_CONTRACT_PATH = (
    REVISION_ROOT
    / "07_GEFCom完整主实验"
    / "configs"
    / "s07_landscape_index_contract_v1.json"
)
FORMAL_INDEX_CONTRACT_SHA256 = (
    "4a42f7980b523328dbe0853a19516d40173726178028a43821cf91ffeeb0152b"
)
FORMAL_STATE_COUNT = 6_600
FORMAL_STATE_UNIVERSE_SHA256 = (
    "c80bee4a80b9a4073caa5fbc0f418cb5c9b33572b037ca808b14aaed752639c6"
)
FORMAL_INDEX_IDENTITY = {
    "contract_id": "S07_SOURCE_LANDSCAPE_INDEX_V1",
    "contract_version": "1.0.0",
    "status": "FROZEN_FOR_PREFLIGHT",
    "parent_execution_contract_sha256": "e967f5904163da208c225d292782ed98d50015f64823291f341fab7dec02bea3",
    "causal_event_protocol_sha256": "194f9c7d52efe7f59c91e4c4e8bba0a17bb3712240dfb69832ce5cd875ecc22b",
    "source_fact_root_manifest_sha256": "30a0f86d6f775c7c32a3654117807dfa9acabc8187c2bdefeb8440e1466ef9e7",
    "source_fact_ledger_sha256": "bcb6da55c70cba48c36456ed4f4cfcb6fd1baea4352c2f94969456361d0cde4a",
}

FIT_FILES = {
    "clara_decisions": "clara_decisions.parquet",
    "clara_action_evidence": "clara_action_evidence.parquet",
    "clara_audit": "clara_audit.json",
    "cart_model": "cart_model.json",
    "cart_tree_arrays": "cart_tree_arrays.npz",
    "cart_compact_statistics": "cart_compact_statistics.parquet",
    "cart_probe": "cart_probe.parquet",
    "cart_audit": "cart_audit.json",
    "child_audit": "child_audit.json",
    "compact_source_audit": "compact_source_audit.json",
    "compact_stream_audit": "compact_stream_audit.parquet",
}
REPLAY_FILES = {
    "cell_metrics": "cell_metrics.parquet",
    "action_counts": "action_counts.parquet",
    "paired_blocks": "paired_blocks.parquet",
    "diagnostic_metrics": "diagnostic_metrics.parquet",
    "clara_state_counts": "clara_state_counts.parquet",
    "event_conservation": "event_conservation.parquet",
    "metric_finalize_audit": "metric_finalize_audit.json",
    "linucb_replay_audit": "linucb_replay_audit.json",
}

FIT_MANIFEST_FIELDS = {
    "schema",
    "status",
    "heldout_zone",
    "seed",
    "price_id",
    "price_sha256",
    "action_library_sha256",
    "compact_source_identity_sha256",
    "endpoint_root_sha256",
    "execution_identity_sha256",
    "execution_identity",
    "fit_child_identity",
    "fit_child_identity_sha256",
    "state_universe_sha256",
    "state_count",
    "index_contract_sha256",
    "index_contract_identity",
    "config_identities",
    "core_module_sha256",
    "writer_module_sha256",
    "software_versions",
    "files",
    "payload_sha256",
}
FIT_PARENT_FIELDS = {
    "schema",
    "status",
    "heldout_zone",
    "seed",
    "action_library_sha256",
    "compact_source_identity_sha256",
    "endpoint_root_sha256",
    "execution_identity_sha256",
    "execution_identity",
    "fit_parent_identity",
    "fit_parent_identity_sha256",
    "price_child_identity_sha256",
    "price_child_manifest_sha256",
    "price_child_count",
    "payload_sha256",
}
REPLAY_MANIFEST_FIELDS = {
    "schema",
    "status",
    "evaluation_zone",
    "seed",
    "policy_mode",
    "evaluation_price_id",
    "evaluation_price_sha256",
    "policy_price_id",
    "policy_price_sha256",
    "policy_evaluation_scope_sha256",
    "action_library_sha256",
    "fit_child_identity_sha256",
    "fit_child_manifest_sha256",
    "endpoint_root_sha256",
    "target_input_manifest_path",
    "target_input_manifest_sha256",
    "execution_identity_sha256",
    "execution_identity",
    "replay_child_identity",
    "replay_child_identity_sha256",
    "policy_replay_child_identity",
    "policy_replay_child_identity_sha256",
    "metric_result_capability_sha256",
    "expected_event_signature",
    "partition_records",
    "files",
    "payload_sha256",
}
REPLAY_PARENT_FIELDS = {
    "schema",
    "status",
    "evaluation_zone",
    "seed",
    "action_library_sha256",
    "fit_parent_manifest_sha256",
    "fit_parent_identity_sha256",
    "endpoint_root_sha256",
    "target_input_manifest_path",
    "target_input_manifest_sha256",
    "execution_identity_sha256",
    "execution_identity",
    "price_child_identity_sha256",
    "price_child_manifest_sha256",
    "price_child_count",
    "payload_sha256",
}
FILE_RECORD_FIELDS = {"relative_path", "sha256", "byte_count", "row_count"}
SIGNATURE_FIELDS = {
    "event_count",
    "hash_sum_u64",
    "hash_xor_u64",
    "hash_square_sum_u64",
    "canonical_sorted_event_id_sha256",
}
PARTITION_RECORD_FIELDS = {
    "zone_or_farm",
    "seed",
    "horizon_steps",
    "predictor",
    "target_coverage",
    *SIGNATURE_FIELDS,
}
TARGET_INPUT_FIELDS = {
    "schema",
    "status",
    "evaluation_zone",
    "seed",
    "endpoint_root_path",
    "endpoint_root_sha256",
    "execution_identity_sha256",
    "target_loader_module_sha256",
    "stream_count",
    "stream_audits",
    "unit_count",
    "unit_records",
    "partition_count",
    "partition_records",
    "combined_event_signature",
    "payload_sha256",
}
TARGET_STREAM_AUDIT_FIELDS = {
    "schema",
    "status",
    "evaluation_zone",
    "predictor",
    "horizon",
    "seed",
    "partition_id",
    "raw_s06_event_count",
    "endpoint_root_target_event_count",
    "s06_events_outside_endpoint_root_count",
    "cold_start_event_count",
    "target_event_count",
    "historically_releasable_feedback_event_count",
    "terminal_or_not_yet_releasable_event_count",
    "target_event_filter_applied",
    "strict_maturity_used_for_feedback_release_only",
    "strict_mature_action_count",
    "heldout_zone_used_only_as_target",
    "raw_width_threshold_outer_fold",
    "endpoint_root_manifest_sha256",
    "endpoint_unit_manifest_sha256",
    "endpoint_unit_expected_event_count",
    "event_signature",
    "leaf_sha256",
    "leaf_hashes_verified",
    "toctou_recheck_passed",
    "source_anchor_audit_sha256",
    "migration_manifest_sha256",
}
TARGET_UNIT_RECORD_FIELDS = {
    "unit_id",
    "unit_manifest_sha256",
    "expected_event_count",
    "observed_event_count",
    "cold_start_event_count",
    "event_signature",
}
METRIC_FINALIZE_AUDIT_FIELDS = {
    "schema",
    "status",
    "scope",
    "policy_mode",
    "evaluation_price_id",
    "evaluation_price_sha256",
    "policy_price_id",
    "policy_price_sha256",
    "policy_evaluation_scope_sha256",
    "policy_feedback_uses_evaluation_price",
    "endpoint_score_uses_evaluation_price",
    "method_count",
    "methods",
    "trusted_partition_count",
    "trusted_partition_ids",
    "trusted_partition_signatures_sha256",
    "evaluation_zone",
    "seed",
    "expected_event_count_per_method",
    "expected_event_signature",
    "update_count",
    "input_method_event_score_count",
    "event_rows_retained",
    "retained_reduced_rows",
    "cell_metric_row_count",
    "action_count_row_count",
    "paired_block_row_count",
    "diagnostic_metric_row_count",
    "clara_state_count_row_count",
    "event_conservation_row_count",
    "rolling_reliability_contract",
    "candidate_historical_reliability_used_as_final_metric",
    "diagnostic_rates_recoverable_from_integer_sufficient_statistics",
    "method_policy_identities",
    "full_target_event_set_preserved",
    "terminal_pending_feedback_rows_dropped",
    "linucb_feedback_release_rule",
}
LINUCB_REPLAY_AUDIT_FIELDS = {
    "schema",
    "status",
    "method",
    "evaluation_zone",
    "seed",
    "price",
    "actions",
    "horizons",
    "global_issue_time_coordinator",
    "horizon_independent_restart",
    "fresh_state_initialized_for_this_price",
    "cross_price_state_shared",
    "strict_delayed_feedback",
    "candidate_action_count_per_event",
    "candidate_event_action_count",
    "selected_action_count",
    "selected_action_library_closed",
    "retained_decisions_sha256",
    "action_library_sha256",
    "price_sha256",
    "global_replay_scope_sha256",
    "derived_contract_sha256",
    "linucb_policy_identity",
    "replay_child_identity",
    "terminal_feedback_drain",
    "terminal_drain_decision_effect",
    "event_count",
    "adaptation_event_count",
    "final_event_count",
    "feedback_count_before_terminal_drain",
    "pending_feedback_count",
    "final_feedback_drain",
    "future_feedback_violation_count",
    "within_issue_feedback_use_count",
    "action_update_count",
    "context_dimension",
    "exploration_alpha",
    "l2_regularization",
    "batch_scoring",
    "feedback_update_order",
}
CLARA_AUDIT_FIELDS = {
    "schema", "status", "method", "heldout_zone",
    "heldout_zone_used_in_fit", "seed", "price", "source_zones",
    "training_horizons", "state_count", "actions",
    "adaptive_n_min_top_two_action_scope", "adaptive_nu_median_action_scope",
    "guardrail_safe_set_action_scope", "four_action_then_append_shortcut_used",
    "price_conditioned_risk", "adaptive_guardrail_thresholds",
    "v4_actual_configuration_sha256", "v4_adaptive_subcontract_sha256",
    "derived_contract_sha256", "n_min_counts", "nu_counts",
    "guardrail_empty_state_count", "selected_action_state_counts",
}
CART_AUDIT_FIELDS = {
    "schema", "status", "method", "heldout_zone",
    "heldout_zone_used_in_fit", "seed", "price", "actions",
    "eventwise_labels_recomputed_for_price", "source_event_count",
    "observed_compact_state_count", "expanded_training_row_count",
    "feature_count", "observed_classes", "tree_depth", "tree_leaf_count",
    "configuration", "actual_configuration_sha256",
    "frozen_zone_configuration_sha256", "derived_contract_sha256",
    "best_action_counts",
}
FIT_CHILD_AUDIT_FIELDS = {
    "schema", "status", "heldout_zone", "seed", "price_id",
    "price_sha256", "action_library_sha256", "fit_child_identity_sha256",
    "clara_v4_adaptive_subcontract_sha256",
    "cart_frozen_zone_configuration_sha256", "clara_state_count",
    "cart_source_event_count", "price_independent_compact_reused",
    "cross_price_policy_state_shared", "atomic_commit_unit",
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("JSON artifact禁止NaN/Inf")
        return float(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"artifact JSON含不可序列化类型: {type(value)!r}")


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _json_ready(dict(payload)),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _payload_with_sha(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _json_ready(dict(payload))
    return {**normalized, "payload_sha256": canonical_sha256(normalized)}


def _verify_payload_sha(payload: Mapping[str, Any], *, fields: set[str], label: str) -> None:
    if set(payload) != fields:
        raise RuntimeError(f"{label} schema字段失配: extra={sorted(set(payload)-fields)}, missing={sorted(fields-set(payload))}")
    expected = str(payload.get("payload_sha256"))
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if not _is_sha256(expected) or canonical_sha256(body) != expected:
        raise RuntimeError(f"{label} payload SHA失配")


def _verify_embedded_identity(
    identity: Mapping[str, Any], sha_field: str, *, schema: str | None = None
) -> str:
    if sha_field not in identity:
        raise RuntimeError(f"identity缺少{sha_field}")
    if schema is not None and identity.get("schema") != schema:
        raise RuntimeError("identity schema失配")
    observed = str(identity[sha_field])
    body = {key: value for key, value in identity.items() if key != sha_field}
    # Core scientific identities use canonical JSON without a trailing newline.
    core_digest = hashlib.sha256(
        json.dumps(
            _json_ready(body), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if not _is_sha256(observed) or core_digest != observed:
        raise RuntimeError(f"identity现场复算失败: {sha_field}")
    return observed


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(payload))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact根必须为object: {path}")
    return value


def _module_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def _core_sha256() -> str:
    return sha256_file(Path(core.__file__).resolve())


def _software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _price_from_id(price_id: str) -> core.PriceSpec:
    for row in core.FORMAL_PRICE_ROWS:
        if row[0] == str(price_id):
            return core.PriceSpec(
                price_id=row[0],
                miss_to_capacity_ratio=row[1],
                theta=row[2],
                base_submission_price=row[3],
            )
    raise RuntimeError(f"未知正式价格: {price_id}")


def _formal_price_ids() -> tuple[str, ...]:
    return tuple(row[0] for row in core.FORMAL_PRICE_ROWS)


def replay_child_key(policy_mode: str, evaluation_price_id: str) -> str:
    if str(policy_mode) not in core.FORMAL_POLICY_MODES:
        raise ValueError(f"未知replay policy mode: {policy_mode}")
    if str(evaluation_price_id) not in _formal_price_ids():
        raise ValueError(f"未知replay evaluation price: {evaluation_price_id}")
    return f"{policy_mode}__{evaluation_price_id}"


def formal_replay_child_keys() -> tuple[str, ...]:
    return tuple(
        replay_child_key(mode, price_id)
        for mode in core.FORMAL_POLICY_MODES
        for price_id in _formal_price_ids()
    )


def _assert_no_temporary_entries(container: Path) -> None:
    if not container.exists():
        return
    temporary = sorted(
        str(path) for path in container.iterdir() if ".tmp__" in path.name
    )
    if temporary:
        raise RuntimeError(f"检测到未裁决临时artifact，拒绝自动清理/继续: {temporary}")


def _assert_child_load_location(
    directory: Path,
    *,
    expected_child_name: str,
    staged: bool,
) -> None:
    """Bind a deep load either to the final child or its sole staged directory."""

    child = Path(directory)
    if staged:
        expected_prefix = f".{expected_child_name}.tmp__"
        if (
            not child.is_dir()
            or not child.name.startswith(expected_prefix)
            or len(child.name) <= len(expected_prefix)
        ):
            raise RuntimeError("staged artifact目录名/类型失配")
        temporary = [
            path.resolve()
            for path in child.parent.iterdir()
            if ".tmp__" in path.name
        ]
        if temporary != [child.resolve()]:
            raise RuntimeError("staged artifact不是children容器中的唯一临时目录")
        return
    if child.name != str(expected_child_name):
        raise RuntimeError("artifact final child目录名失配")
    _assert_no_temporary_entries(child.parent)


def _assert_child_container(
    children_dir: Path, *, require_exact: bool, kind: str = "fit"
) -> None:
    _assert_no_temporary_entries(children_dir)
    if not children_dir.exists():
        if require_exact:
            raise RuntimeError("parent seal缺少children目录")
        return
    observed = {path.name for path in children_dir.iterdir() if path.is_dir()}
    files = {path.name for path in children_dir.iterdir() if not path.is_dir()}
    allowed = (
        set(formal_replay_child_keys())
        if str(kind) == "replay"
        else set(_formal_price_ids())
    )
    if files or not observed.issubset(allowed) or (require_exact and observed != allowed):
        raise RuntimeError(
            f"children容器含extra/非法/缺失child: dirs={sorted(observed)}, files={sorted(files)}"
        )


def _assert_parent_container(parent: Path, seal_name: str, *, allow_missing_seal: bool) -> None:
    if not parent.exists():
        return
    observed = {path.name for path in parent.iterdir()}
    is_replay = seal_name == "replay_parent_manifest.json"
    allowed = {"children", seal_name}
    if is_replay:
        allowed.add("target_input_manifest.json")
    if not observed.issubset(allowed) or (
        not allow_missing_seal and seal_name not in observed
    ) or (
        is_replay and "target_input_manifest.json" not in observed
    ):
        raise RuntimeError(f"parent容器含extra或seal缺失: {sorted(observed)}")
    if "children" in observed and not (parent / "children").is_dir():
        raise RuntimeError("parent children必须为目录")
    if seal_name in observed and not (parent / seal_name).is_file():
        raise RuntimeError("parent seal必须为文件")
    if is_replay and not (parent / "target_input_manifest.json").is_file():
        raise RuntimeError("replay parent target input manifest必须为文件")


def _assert_formal_parent_path(parent_dir: Path, *, kind: str, zone: str, seed: int) -> None:
    root = FORMAL_FIT_ROOT if kind == "fit" else FORMAL_REPLAY_ROOT
    expected = root / f"{zone}__seed{int(seed)}"
    if Path(parent_dir).resolve() != expected.resolve():
        raise RuntimeError(f"正式{kind} artifact拒绝非官方输出路径")


def _reject_nonformal_official_root(
    path: Path, *, formal_identity: bool, kind: str
) -> None:
    """Never let a non-formal call plant bytes under a formal selector root."""

    resolved = Path(path).resolve()
    roots = {
        "fit": FORMAL_FIT_ROOT.resolve(),
        "replay": FORMAL_REPLAY_ROOT.resolve(),
        "selector": FORMAL_SELECTOR_ROOT.resolve(),
    }
    root = roots[kind]
    if not formal_identity and (resolved == root or resolved.is_relative_to(root)):
        raise RuntimeError(
            f"官方{kind} root禁止formal_identity=False写入/封存"
        )


def _dataframe_sha(frame: pd.DataFrame, columns: Sequence[str], sort_by: Sequence[str]) -> str:
    selected = frame.loc[:, list(columns)].sort_values(list(sort_by), kind="mergesort")
    records = [_json_ready(row) for row in selected.to_dict("records")]
    return canonical_sha256({"columns": list(columns), "records": records})


def _state_universe_sha(frame: pd.DataFrame) -> str:
    columns = ("full_state_code", *core.STATE_FIELDS)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"状态全集缺列: {missing}")
    selected = frame.loc[:, list(columns)].sort_values(
        "full_state_code", kind="mergesort"
    )
    payload = {
        "columns": list(columns),
        "records": [_json_ready(row) for row in selected.to_dict("records")],
    }
    # Match the frozen core/config scientific identity (no trailing newline).
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def official_state_universe() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = FORMAL_INDEX_CONTRACT_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FORMAL_INDEX_CONTRACT_SHA256:
        raise RuntimeError("官方landscape index contract SHA失配")
    index_contract = json.loads(raw.decode("utf-8"))
    observed_identity = {
        key: index_contract.get(key) for key in FORMAL_INDEX_IDENTITY
    }
    if observed_identity != FORMAL_INDEX_IDENTITY:
        raise RuntimeError("官方landscape index contract身份/parent链失配")
    contracts = core.six_action_contracts(load_frozen_contracts())
    states = core.state_universe_from_contract(contracts, index_contract)
    observed_sha = _state_universe_sha(states)
    if len(states) != FORMAL_STATE_COUNT or observed_sha != FORMAL_STATE_UNIVERSE_SHA256:
        raise RuntimeError("官方6600 state universe身份失配")
    return states, observed_identity


def _validate_execution_identity(
    execution_identity_sha256: str,
    execution_identity: Mapping[str, Any] | None,
    *,
    formal_identity: bool,
) -> None:
    if not _is_sha256(execution_identity_sha256):
        raise RuntimeError("execution identity SHA非法")
    if execution_identity is not None and canonical_sha256(
        dict(execution_identity)
    ) != str(execution_identity_sha256):
        raise RuntimeError("execution identity payload现场复算失配")
    if not formal_identity:
        return
    if execution_identity is None:
        raise RuntimeError("formal artifact拒绝裸execution SHA；必须提供live payload")
    if (
        sha256_file(Path(__file__)) != IMPORTED_WRITER_MODULE_SHA256
        or sha256_file(Path(core.__file__)) != IMPORTED_CORE_MODULE_SHA256
    ):
        raise RuntimeError("artifact/core在当前进程import后发生TOCTOU漂移")
    import importlib

    runner = importlib.import_module("run_gefcom_six_action_selectors_v1")
    phase, main = runner.load_configs()
    live, _ = runner.execution_identity(phase, main, require_release=True)
    live_payload = runner.execution_identity_payload(live)
    if (
        str(live["execution_identity_sha256"]) != str(execution_identity_sha256)
        or dict(live_payload) != dict(execution_identity)
    ):
        raise RuntimeError("artifact commit前live code/config/root/release/environment身份漂移")
    module_hashes = dict(
        dict(execution_identity.get("code_identity", {})).get(
            "module_file_sha256", {}
        )
    )
    if (
        str(module_hashes.get("selector_artifacts"))
        != IMPORTED_WRITER_MODULE_SHA256
        or str(module_hashes.get("selector_core")) != IMPORTED_CORE_MODULE_SHA256
    ):
        raise RuntimeError("execution identity未绑定当前已import artifact/core字节")


def _validate_state_universe(
    decisions: pd.DataFrame,
    evidence: pd.DataFrame,
    expected_states: pd.DataFrame,
) -> str:
    columns = ["full_state_code", *core.STATE_FIELDS]
    expected = expected_states.loc[:, columns].sort_values("full_state_code", kind="mergesort").reset_index(drop=True)
    observed = decisions.loc[:, columns].sort_values("full_state_code", kind="mergesort").reset_index(drop=True)
    if observed.duplicated(columns).any() or not observed.equals(expected):
        raise RuntimeError("CLARA decisions未完整覆盖冻结state universe")
    if "action" not in evidence.columns:
        raise RuntimeError("CLARA action evidence缺少action")
    evidence_keys = evidence[["full_state_code", "action"]].copy()
    if evidence_keys.duplicated().any():
        raise RuntimeError("CLARA action evidence状态×动作重复")
    counts = evidence_keys.groupby("full_state_code", sort=False)["action"].agg(
        lambda values: tuple(values.astype(str))
    )
    if set(counts.index.astype(int)) != set(expected["full_state_code"].astype(int)) or any(
        set(values) != set(core.SIX_ACTIONS) or len(values) != len(core.SIX_ACTIONS)
        for values in counts
    ):
        raise RuntimeError("CLARA action evidence未对每状态完整覆盖六动作")
    selected = set(decisions["selected_action"].astype(str))
    if not selected.issubset(set(core.SIX_ACTIONS)):
        raise RuntimeError("CLARA decisions含动作库外值")
    return _state_universe_sha(expected)


def _independent_six_action_selection(
    evidence: pd.DataFrame,
    thresholds: Mapping[str, Any],
    contracts: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute the frozen six-action guardrail without legacy module globals.

    ``run_four_clara_versions`` intentionally retains its four-action ``ACTIONS``
    default.  The formal fitter temporarily replaces that global while fitting,
    but an artifact validator must neither depend on nor mutate ambient process
    state.  Keep this replay explicit and independent so 6600 x 6 evidence can
    never be silently interpreted as 9900 x 4 evidence.
    """

    required = {
        "full_state_code",
        "action_order",
        "target_coverage",
        "coverage_point",
        "tuwr_point",
        "ard_point",
        "risk_score",
        "cold_start_guardrail",
    }
    missing = sorted(required - set(evidence.columns))
    if missing:
        raise RuntimeError(f"CLARA六动作独立重算缺列: {missing}")
    action_count = len(core.SIX_ACTIONS)
    if len(evidence) % action_count:
        raise RuntimeError("CLARA六动作证据行数不能被6整除")
    ordered = evidence.sort_values(
        ["full_state_code", "action_order"], kind="mergesort"
    ).reset_index(drop=True)
    state_count = len(ordered) // action_count
    state_codes = ordered["full_state_code"].to_numpy(dtype=np.int64).reshape(
        state_count, action_count
    )
    action_orders = ordered["action_order"].to_numpy(dtype=np.int64).reshape(
        state_count, action_count
    )
    if not np.array_equal(
        state_codes,
        np.broadcast_to(
            np.arange(state_count, dtype=np.int64)[:, None], state_codes.shape
        ),
    ):
        raise RuntimeError("CLARA六动作独立重算状态编号不连续")
    if not np.array_equal(
        action_orders,
        np.broadcast_to(
            np.arange(action_count, dtype=np.int64), action_orders.shape
        ),
    ):
        raise RuntimeError("CLARA六动作独立重算动作顺序身份失配")

    tolerance = float(
        contracts.protocol["selection_contract"]["score_tie_tolerance"]
    )
    target = ordered["target_coverage"].to_numpy(dtype=np.float64).reshape(
        state_count, action_count
    )
    coverage = ordered["coverage_point"].to_numpy(dtype=np.float64).reshape(
        state_count, action_count
    )
    tuwr = ordered["tuwr_point"].to_numpy(dtype=np.float64).reshape(
        state_count, action_count
    )
    ard = ordered["ard_point"].to_numpy(dtype=np.float64).reshape(
        state_count, action_count
    )
    risk = ordered["risk_score"].to_numpy(dtype=np.float64).reshape(
        state_count, action_count
    )
    cold_matrix = ordered["cold_start_guardrail"].to_numpy(dtype=bool).reshape(
        state_count, action_count
    )
    if not np.all(cold_matrix == cold_matrix[:, :1]):
        raise RuntimeError("CLARA六动作cold-start状态内不一致")
    cold = cold_matrix[:, 0]

    safe = coverage >= (
        target - float(thresholds["coverage_shortfall_epsilon"])
    )
    complete = ~cold
    safe[complete] &= (
        np.isfinite(tuwr[complete])
        & (tuwr[complete] <= float(thresholds["tuwr_upper"]))
        & np.isfinite(ard[complete])
        & (ard[complete] <= float(thresholds["ard_upper"]))
    )
    empty = ~safe.any(axis=1)
    candidates = safe.copy()
    candidates[empty] = True

    minimum_risk = np.min(np.where(candidates, risk, np.inf), axis=1)
    candidates &= risk <= minimum_risk[:, None] + tolerance
    maximum_coverage = np.max(np.where(candidates, coverage, -np.inf), axis=1)
    candidates &= coverage >= maximum_coverage[:, None] - tolerance
    for values, name in ((tuwr, "TUWR"), (ard, "ARD")):
        candidate_count = candidates.sum(axis=1)
        finite = np.isfinite(values)
        finite_count = (candidates & finite).sum(axis=1)
        mixed = (finite_count > 0) & (finite_count < candidate_count)
        if mixed.any():
            raise RuntimeError(f"CLARA六动作并列候选{name}可用性不一致")
        applicable = (finite_count == candidate_count) & (candidate_count > 0)
        minimum = np.min(
            np.where(candidates & finite, values, np.inf), axis=1
        )
        candidates &= (~applicable[:, None]) | (
            finite & (values <= minimum[:, None] + tolerance)
        )
    if not candidates.any(axis=1).all():
        raise RuntimeError("CLARA六动作独立重算产生空候选")
    selected = np.argmax(candidates, axis=1).astype(np.int64)
    return selected, empty, safe


def _validate_fit_policy_semantics(
    *,
    decisions: pd.DataFrame,
    evidence: pd.DataFrame,
    selector: ExtendedCartSelector,
    compact_statistics: pd.DataFrame,
    clara_audit: Mapping[str, Any],
    cart_audit: Mapping[str, Any],
    child_audit: Mapping[str, Any],
    fit_identity: Mapping[str, Any],
    zone: str,
    seed: int,
    price: core.PriceSpec,
    formal_identity: bool,
) -> None:
    """Recompute policy/audit invariants; self-consistent hashes are insufficient."""

    if (
        set(clara_audit) != CLARA_AUDIT_FIELDS
        or set(cart_audit) != CART_AUDIT_FIELDS
        or set(child_audit) != FIT_CHILD_AUDIT_FIELDS
    ):
        raise RuntimeError("fit child CLARA/CART/child audit exact schema失配")
    contracts = core.six_action_contracts(load_frozen_contracts())
    derived = core.derived_six_action_contract_identity(contracts)
    fit_sha = str(fit_identity.get("fit_child_identity_sha256"))
    expected_common = (
        clara_audit.get("schema") == core.SCHEMA
        and cart_audit.get("schema") == core.SCHEMA
        and child_audit.get("schema") == core.SCHEMA
        and clara_audit.get("status") == "PASS"
        and cart_audit.get("status") == "PASS"
        and child_audit.get("status") == "PASS"
        and clara_audit.get("method") == "CLARA_6A"
        and cart_audit.get("method") == "CART_6A"
        and str(clara_audit.get("heldout_zone")) == str(zone)
        and str(cart_audit.get("heldout_zone")) == str(zone)
        and str(child_audit.get("heldout_zone")) == str(zone)
        and int(clara_audit.get("seed", -1)) == int(seed)
        and int(cart_audit.get("seed", -1)) == int(seed)
        and int(child_audit.get("seed", -1)) == int(seed)
        and not bool(clara_audit.get("heldout_zone_used_in_fit"))
        and not bool(cart_audit.get("heldout_zone_used_in_fit"))
        and dict(clara_audit.get("price", {})) == price.as_dict()
        and dict(cart_audit.get("price", {})) == price.as_dict()
        and str(child_audit.get("price_id")) == price.price_id
        and str(child_audit.get("price_sha256")) == core.price_sha256(price)
        and tuple(clara_audit.get("actions", [])) == core.SIX_ACTIONS
        and tuple(cart_audit.get("actions", [])) == core.SIX_ACTIONS
        and str(child_audit.get("action_library_sha256"))
        == core.action_library_sha256()
        and str(child_audit.get("fit_child_identity_sha256")) == fit_sha
        and str(clara_audit.get("derived_contract_sha256"))
        == str(fit_identity.get("derived_contract_sha256"))
        and str(cart_audit.get("derived_contract_sha256"))
        == str(fit_identity.get("derived_contract_sha256"))
        and int(clara_audit.get("state_count", -1)) == len(decisions)
        and int(child_audit.get("clara_state_count", -1)) == len(decisions)
        and bool(child_audit.get("price_independent_compact_reused"))
        and not bool(child_audit.get("cross_price_policy_state_shared"))
        and child_audit.get("atomic_commit_unit") == "heldout_zone_seed_price"
    )
    if not expected_common:
        raise RuntimeError("fit child audit轴/价格/动作/派生合同失配")
    if formal_identity and str(fit_identity.get("derived_contract_sha256")) != str(
        derived["derived_contract_sha256"]
    ):
        raise RuntimeError("fit child未绑定官方六动作派生合同")

    ordered_evidence = evidence.sort_values(
        ["full_state_code", "action_order"], kind="mergesort"
    ).reset_index(drop=True)
    ordered_decisions = decisions.sort_values(
        "full_state_code", kind="mergesort"
    ).reset_index(drop=True)
    if (
        len(ordered_evidence) != len(ordered_decisions) * len(core.SIX_ACTIONS)
        or set(ordered_evidence["price_id"].astype(str)) != {price.price_id}
        or not np.array_equal(
            ordered_evidence["action_order"].to_numpy(dtype=np.int64),
            np.tile(np.arange(len(core.SIX_ACTIONS), dtype=np.int64), len(ordered_decisions)),
        )
        or not np.array_equal(
            ordered_evidence["action"].astype(str).to_numpy(),
            np.tile(np.asarray(core.SIX_ACTIONS, dtype=str), len(ordered_decisions)),
        )
    ):
        raise RuntimeError("CLARA evidence state×六动作顺序不闭合")
    thresholds = dict(clara_audit.get("adaptive_guardrail_thresholds", {}))
    selected_index, empty, safe = _independent_six_action_selection(
        ordered_evidence, thresholds, contracts
    )
    selected = np.asarray(core.SIX_ACTIONS, dtype=object)[selected_index]
    if (
        not np.array_equal(
            ordered_decisions["selected_action"].astype(str).to_numpy(),
            selected.astype(str),
        )
        or not np.array_equal(
            ordered_decisions["guardrail_empty_fallback"].astype(bool).to_numpy(),
            empty,
        )
        or not np.array_equal(
            ordered_evidence["guardrail_pass"].astype(bool).to_numpy(),
            safe.reshape(-1),
        )
        or not np.array_equal(
            ordered_decisions["selected_action_guardrail_pass"].astype(bool).to_numpy(),
            safe[np.arange(len(selected_index)), selected_index],
        )
    ):
        raise RuntimeError("CLARA decisions与evidence/guardrail重算失配")
    selected_counts = {
        action: int((selected.astype(str) == action).sum())
        for action in core.SIX_ACTIONS
    }
    n_min_counts = {
        str(value): int((ordered_decisions["selected_n_min"].astype(int) == int(value)).sum())
        for value in sorted(ordered_decisions["selected_n_min"].astype(int).unique())
    }
    nu_counts = {
        str(value): int((ordered_decisions["selected_nu"].astype(int) == int(value)).sum())
        for value in sorted(ordered_decisions["selected_nu"].astype(int).unique())
    }
    if (
        dict(clara_audit.get("selected_action_state_counts", {})) != selected_counts
        or dict(clara_audit.get("n_min_counts", {})) != n_min_counts
        or dict(clara_audit.get("nu_counts", {})) != nu_counts
        or int(clara_audit.get("guardrail_empty_state_count", -1)) != int(empty.sum())
        or int(clara_audit.get("adaptive_n_min_top_two_action_scope", -1)) != 6
        or int(clara_audit.get("adaptive_nu_median_action_scope", -1)) != 6
        or int(clara_audit.get("guardrail_safe_set_action_scope", -1)) != 6
        or bool(clara_audit.get("four_action_then_append_shortcut_used"))
        or not bool(clara_audit.get("price_conditioned_risk"))
        or str(clara_audit.get("v4_actual_configuration_sha256"))
        != str(fit_identity.get("clara_configuration_sha256"))
        or str(clara_audit.get("v4_adaptive_subcontract_sha256"))
        != str(fit_identity.get("clara_v4_adaptive_subcontract_sha256"))
        or str(child_audit.get("clara_v4_adaptive_subcontract_sha256"))
        != str(fit_identity.get("clara_v4_adaptive_subcontract_sha256"))
    ):
        raise RuntimeError("CLARA audit计数/配置身份失配")

    required_stats = {
        *core.STATE_FIELDS,
        "event_count",
        *(f"{action}__best_count" for action in core.SIX_ACTIONS),
    }
    if set(compact_statistics.columns) != required_stats or compact_statistics.empty:
        raise RuntimeError("CART compact statistics exact schema/空值失配")
    integer_columns = [
        "event_count", *(f"{action}__best_count" for action in core.SIX_ACTIONS)
    ]
    numeric = compact_statistics[integer_columns].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(numeric).all()
        or np.any(numeric < 0.0)
        or not np.array_equal(numeric, np.floor(numeric))
        or compact_statistics[list(core.STATE_FIELDS)].duplicated().any()
        or not np.array_equal(numeric[:, 1:].sum(axis=1), numeric[:, 0])
    ):
        raise RuntimeError("CART compact statistics计数/状态不闭合")
    best_counts = {
        action: int(compact_statistics[f"{action}__best_count"].sum())
        for action in core.SIX_ACTIONS
    }
    source_event_count = int(compact_statistics["event_count"].sum())
    if (
        int(cart_audit.get("source_event_count", -1)) != source_event_count
        or int(child_audit.get("cart_source_event_count", -1)) != source_event_count
        or int(cart_audit.get("observed_compact_state_count", -1))
        != len(compact_statistics)
        or dict(cart_audit.get("best_action_counts", {})) != best_counts
        or tuple(cart_audit.get("observed_classes", []))
        != tuple(str(value) for value in selector.classifier.classes_)
        or int(cart_audit.get("tree_depth", -1)) != selector.classifier.get_depth()
        or int(cart_audit.get("tree_leaf_count", -1))
        != selector.classifier.get_n_leaves()
        or int(cart_audit.get("feature_count", -1))
        != len(selector.encoder.feature_names)
        or dict(cart_audit.get("configuration", {})) != dict(selector.configuration)
        or str(cart_audit.get("actual_configuration_sha256"))
        != str(fit_identity.get("cart_configuration_sha256"))
        or str(cart_audit.get("frozen_zone_configuration_sha256"))
        != str(fit_identity.get("cart_frozen_zone_configuration_sha256"))
        or str(child_audit.get("cart_frozen_zone_configuration_sha256"))
        != str(fit_identity.get("cart_frozen_zone_configuration_sha256"))
        or not bool(cart_audit.get("eventwise_labels_recomputed_for_price"))
    ):
        raise RuntimeError("CART model/stats/audit/config身份失配")
    configuration = dict(selector.configuration)
    classifier = selector.classifier
    if (
        tuple(selector.actions) != core.SIX_ACTIONS
        or classifier.max_depth != configuration.get("max_depth")
        or int(classifier.min_samples_leaf)
        != int(configuration.get("min_samples_leaf", -1))
        or int(classifier.random_state) != int(configuration.get("random_state", -1))
    ):
        raise RuntimeError("CART safe model参数与配置失配")
    if formal_identity:
        frozen = core.frozen_zone_baseline_contract(zone, "CARTBestAction")
        v4_config = json.loads(
            core.FOUR_VERSION_CONFIG_PATH.read_text(encoding="utf-8")
        )
        v4 = core.validate_formal_v4_configuration(v4_config)
        if (
            dict(frozen["configuration"]) != configuration
            or str(frozen["frozen_configuration_sha256"])
            != str(fit_identity.get("cart_frozen_zone_configuration_sha256"))
            or core._canonical_digest(configuration)
            != str(fit_identity.get("cart_configuration_sha256"))
            or str(v4["actual_configuration_sha256"])
            != str(fit_identity.get("clara_configuration_sha256"))
            or str(v4["v4_adaptive_subcontract_sha256"])
            != str(fit_identity.get("clara_v4_adaptive_subcontract_sha256"))
            or str(fit_identity.get("algorithm_closure_sha256"))
            != str(core.algorithm_dependency_closure()["algorithm_closure_sha256"])
        ):
            raise RuntimeError("fit policy未绑定V4/CART registry/算法closure")


def _validate_compact_source_provenance(
    source_audit: Mapping[str, Any],
    stream_audit: pd.DataFrame,
    *,
    zone: str,
    seed: int,
    compact_source_identity_sha256: str,
    endpoint_root_sha256: str,
) -> None:
    body = {
        key: value
        for key, value in source_audit.items()
        if key
        not in {
            "identity_sha256",
            "maximum_base_price_errf_regression_error",
            "array_bytes",
            "status",
        }
    }
    observed_identity = hashlib.sha256(
        json.dumps(_json_ready(body), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if (
        source_audit.get("status") != "PASS"
        or str(source_audit.get("heldout_zone")) != str(zone)
        or int(source_audit.get("seed", -1)) != int(seed)
        or str(source_audit.get("identity_sha256")) != observed_identity
        or observed_identity != str(compact_source_identity_sha256)
        or int(source_audit.get("stream_count", -1)) != 180
        or int(source_audit.get("strict_causal_stream_count", -1)) != 180
        or list(source_audit.get("endpoint_root_manifest_sha256_set", []))
        != [str(endpoint_root_sha256)]
    ):
        raise RuntimeError("compact source正式因果/身份审计失配")
    required = {
        "source_zone",
        "predictor",
        "horizon",
        "seed",
        "status",
        "strict_maturity_filter_applied",
        "leaf_hashes_verified",
        "event_id_hash_verified",
        "outer_heldout_zone_used_as_source",
        "endpoint_root_seal_verified",
        "endpoint_root_manifest_sha256",
    }
    if set(stream_audit.columns) < required or len(stream_audit) != 180:
        raise RuntimeError("compact stream audit列或行数失配")
    expected_streams = {
        (source, predictor, horizon, int(seed))
        for source in core.FORMAL_ZONES
        if source != str(zone)
        for predictor in core.FORMAL_PREDICTORS
        for horizon in core.FORMAL_HORIZONS
    }
    observed_streams = set(
        stream_audit[["source_zone", "predictor", "horizon", "seed"]].itertuples(index=False, name=None)
    )
    if observed_streams != expected_streams or not (
        stream_audit["status"].eq("PASS").all()
        and stream_audit["strict_maturity_filter_applied"].astype(bool).all()
        and stream_audit["leaf_hashes_verified"].astype(bool).all()
        and stream_audit["event_id_hash_verified"].astype(bool).all()
        and (~stream_audit["outer_heldout_zone_used_as_source"].astype(bool)).all()
        and stream_audit["endpoint_root_seal_verified"].astype(bool).all()
        and stream_audit["endpoint_root_manifest_sha256"].astype(str).eq(str(endpoint_root_sha256)).all()
    ):
        raise RuntimeError("compact stream audit 9×5×4因果矩阵不闭合")


def _cart_model_payload(selector: ExtendedCartSelector, derived_contract_sha256: str) -> dict[str, Any]:
    classifier = selector.classifier
    if not isinstance(classifier, DecisionTreeClassifier) or not hasattr(classifier, "tree_"):
        raise RuntimeError("CART artifact只接受已拟合DecisionTreeClassifier")
    params = classifier.get_params(deep=False)
    allowed_params = {
        "ccp_alpha",
        "class_weight",
        "criterion",
        "max_depth",
        "max_features",
        "max_leaf_nodes",
        "min_impurity_decrease",
        "min_samples_leaf",
        "min_samples_split",
        "min_weight_fraction_leaf",
        "monotonic_cst",
        "random_state",
        "splitter",
    }
    if set(params) != allowed_params:
        raise RuntimeError("CART sklearn参数schema漂移")
    payload = {
        "schema": CART_MODEL_SCHEMA,
        "actions": list(selector.actions),
        "configuration": dict(selector.configuration),
        "classifier_parameters": params,
        "classes": classifier.classes_.astype(str).tolist(),
        "n_classes": np.atleast_1d(classifier.n_classes_).astype(int).tolist(),
        "n_features_in": int(classifier.n_features_in_),
        "n_outputs": int(classifier.n_outputs_),
        "max_features_fitted": int(classifier.max_features_),
        "tree_max_depth": int(classifier.tree_.max_depth),
        "tree_node_count": int(classifier.tree_.node_count),
        "derived_contract_sha256": str(derived_contract_sha256),
    }
    return _payload_with_sha(payload)


def _write_cart(selector: ExtendedCartSelector, directory: Path, derived_contract_sha256: str) -> None:
    if tuple(selector.actions) != core.SIX_ACTIONS:
        raise RuntimeError("CART artifact动作库失配")
    model = _cart_model_payload(selector, derived_contract_sha256)
    state = selector.classifier.tree_.__getstate__()
    np.savez_compressed(
        directory / FIT_FILES["cart_tree_arrays"],
        nodes=np.asarray(state["nodes"]),
        values=np.asarray(state["values"], dtype=np.float64),
    )
    _write_json(directory / FIT_FILES["cart_model"], model)


def _validate_cart_gini_impurity(
    impurity: np.ndarray,
    values: np.ndarray,
    n_classes: np.ndarray,
) -> None:
    """Validate Gini metadata bounds without altering sklearn tree arrays."""

    observed = np.asarray(impurity, dtype=np.float64)
    class_counts = np.asarray(n_classes, dtype=np.intp)
    if (
        observed.ndim != 1
        or values.ndim != 3
        or values.shape[:2] != (len(observed), len(class_counts))
        or len(class_counts) != 1
        or int(class_counts[0]) <= 1
        or values.shape[2] < int(class_counts[0])
        or not np.isfinite(observed).all()
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
    ):
        raise RuntimeError("CART Gini impurity输入结构/数值失配")
    active = values[:, :, : int(class_counts[0])]
    totals = active.sum(axis=2)
    if not np.isfinite(totals).all() or np.any(totals <= 0.0):
        raise RuntimeError("CART Gini节点类别权重非法")
    maximum = 1.0 - 1.0 / float(class_counts[0])
    if (
        np.any(observed < -CART_GINI_NUMERIC_ATOL)
        or np.any(observed > maximum + CART_GINI_NUMERIC_ATOL)
    ):
        raise RuntimeError("CART Gini impurity超出数值容差边界")


def _load_cart(directory: Path) -> ExtendedCartSelector:
    model = _load_json(directory / FIT_FILES["cart_model"])
    expected_fields = {
        "schema",
        "actions",
        "configuration",
        "classifier_parameters",
        "classes",
        "n_classes",
        "n_features_in",
        "n_outputs",
        "max_features_fitted",
        "tree_max_depth",
        "tree_node_count",
        "derived_contract_sha256",
        "payload_sha256",
    }
    _verify_payload_sha(model, fields=expected_fields, label="CART model")
    if model["schema"] != CART_MODEL_SCHEMA or tuple(model["actions"]) != core.SIX_ACTIONS:
        raise RuntimeError("CART model身份失配")
    contracts = core.six_action_contracts(load_frozen_contracts())
    derived = core.derived_six_action_contract_identity(contracts)
    if str(model["derived_contract_sha256"]) != str(derived["derived_contract_sha256"]):
        raise RuntimeError("CART model派生合同失配")
    with np.load(directory / FIT_FILES["cart_tree_arrays"], allow_pickle=False) as archive:
        if set(archive.files) != {"nodes", "values"}:
            raise RuntimeError("CART tree arrays schema失配")
        nodes = np.asarray(archive["nodes"])
        values = np.asarray(archive["values"], dtype=np.float64)
    nodes.setflags(write=False)
    values.setflags(write=False)
    node_count = int(model["tree_node_count"])
    n_outputs = int(model["n_outputs"])
    n_classes = np.asarray(model["n_classes"], dtype=np.intp)
    classes = np.asarray(model["classes"], dtype=str)
    if (
        nodes.dtype != _tree.NODE_DTYPE
        or nodes.ndim != 1
        or len(nodes) != node_count
        or values.shape != (node_count, n_outputs, int(n_classes.max()))
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or len(classes) != int(n_classes[0])
        or len(set(classes.tolist())) != len(classes)
        or not set(classes.tolist()).issubset(set(core.SIX_ACTIONS))
    ):
        raise RuntimeError("CART tree arrays内容失配")
    if str(model["classifier_parameters"].get("criterion")) != "gini":
        raise RuntimeError("CART artifact仅接受冻结Gini criterion")
    _validate_cart_gini_impurity(
        nodes["impurity"], values, n_classes
    )
    left = nodes["left_child"].astype(np.int64)
    right = nodes["right_child"].astype(np.int64)
    feature = nodes["feature"].astype(np.int64)
    leaf = (left == _tree.TREE_LEAF) & (right == _tree.TREE_LEAF)
    internal = ~leaf
    if (
        node_count <= 0
        or np.any((left[internal] < 0) | (left[internal] >= node_count))
        or np.any((right[internal] < 0) | (right[internal] >= node_count))
        or np.any(feature[leaf] != _tree.TREE_UNDEFINED)
        or np.any(
            (feature[internal] < 0)
            | (feature[internal] >= int(model["n_features_in"]))
        )
        or not np.isfinite(nodes["threshold"]).all()
        or np.any(nodes["n_node_samples"] < 0)
        or np.any(nodes["weighted_n_node_samples"] < 0.0)
        or not set(nodes["missing_go_to_left"].astype(int)).issubset({0, 1})
    ):
        raise RuntimeError("CART tree拓扑/数值不变量失配")
    tree = _tree.Tree(
        int(model["n_features_in"]),
        n_classes,
        n_outputs,
    )
    tree.__setstate__(
        {
            "max_depth": int(model["tree_max_depth"]),
            "node_count": int(model["tree_node_count"]),
            "nodes": nodes,
            "values": values,
        }
    )
    classifier = DecisionTreeClassifier(**dict(model["classifier_parameters"]))
    classifier.classes_ = classes
    classifier.n_classes_ = np.int64(model["n_classes"][0])
    classifier.n_features_in_ = int(model["n_features_in"])
    classifier.n_outputs_ = int(model["n_outputs"])
    classifier.max_features_ = int(model["max_features_fitted"])
    classifier.tree_ = tree
    if (
        int(classifier.tree_.node_count) != node_count
        or int(classifier.tree_.max_depth) != int(model["tree_max_depth"])
    ):
        raise RuntimeError("CART tree安全重建后结构失配")
    return ExtendedCartSelector(
        encoder=FrozenStateEncoder(contracts),
        classifier=classifier,
        actions=core.SIX_ACTIONS,
        configuration=dict(model["configuration"]),
    )


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def _file_records(directory: Path, names: Mapping[str, str], frames: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, relative in names.items():
        path = directory / relative
        if not path.is_file():
            raise RuntimeError(f"artifact缺少预期文件: {relative}")
        output[key] = {
            "relative_path": relative,
            "sha256": sha256_file(path),
            "byte_count": path.stat().st_size,
            "row_count": int(len(frames[key])) if key in frames else None,
        }
    return output


def _verify_files(directory: Path, records: Mapping[str, Any], expected_names: Mapping[str, str]) -> None:
    if set(records) != set(expected_names):
        raise RuntimeError("artifact file记录schema失配")
    expected_disk = {"manifest.json", *expected_names.values()}
    observed_disk = {path.name for path in directory.iterdir()}
    if observed_disk != expected_disk:
        raise RuntimeError(f"artifact目录含未知/缺失文件: {sorted(observed_disk ^ expected_disk)}")
    for key, relative in expected_names.items():
        record = records[key]
        if set(record) != FILE_RECORD_FIELDS or str(record["relative_path"]) != relative:
            raise RuntimeError(f"artifact file record失配: {key}")
        path = (directory / relative).resolve()
        if path.parent != directory.resolve() or not path.is_file():
            raise RuntimeError(f"artifact file路径逃逸或缺失: {key}")
        if (
            not _is_sha256(record["sha256"])
            or sha256_file(path) != str(record["sha256"])
            or path.stat().st_size != int(record["byte_count"])
        ):
            raise RuntimeError(f"artifact file现场SHA/大小失配: {key}")


@dataclass(frozen=True)
class LoadedFitPriceArtifact:
    manifest: dict[str, Any]
    clara_decisions: pd.DataFrame
    clara_action_evidence: pd.DataFrame
    cart_selector: ExtendedCartSelector
    cart_compact_statistics: pd.DataFrame
    compact_source_audit: dict[str, Any]
    compact_stream_audit: pd.DataFrame


def write_fit_price_child(
    parent_dir: Path,
    child: core.PriceFitChildResult,
    *,
    endpoint_root_sha256: str,
    compact_source_identity_sha256: str,
    expected_state_universe: pd.DataFrame,
    compact_source_audit: Mapping[str, Any],
    compact_stream_audit: pd.DataFrame,
    execution_identity_sha256: str,
    execution_identity: Mapping[str, Any] | None = None,
    formal_identity: bool = True,
) -> dict[str, Any]:
    """Atomically commit one independently resumable ``zone x seed x price`` fit."""

    if formal_identity:
        core.verify_formal_fit_result_capability(child)
    identity = dict(child.identity)
    fit_sha = _verify_embedded_identity(
        identity,
        "fit_child_identity_sha256",
        schema="TEST_CLARA_GEFCOM_6A_FIT_PRICE_CHILD_IDENTITY_V1",
    )
    zone = str(identity.get("heldout_zone"))
    seed = int(identity.get("seed", -1))
    price_id = str(identity.get("price_id"))
    price = _price_from_id(price_id)
    if (
        str(identity.get("price_sha256")) != core.price_sha256(price)
        or str(identity.get("action_library_sha256")) != core.action_library_sha256()
        or str(identity.get("compact_source_identity_sha256")) != str(compact_source_identity_sha256)
        or not _is_sha256(endpoint_root_sha256)
    ):
        raise RuntimeError("fit child价格/动作/compact/endpoint身份失配")
    _reject_nonformal_official_root(
        Path(parent_dir), formal_identity=formal_identity, kind="fit"
    )
    _validate_execution_identity(
        execution_identity_sha256,
        execution_identity,
        formal_identity=formal_identity,
    )
    parent = Path(parent_dir)
    if formal_identity:
        _assert_formal_parent_path(parent, kind="fit", zone=zone, seed=seed)
    children_dir = parent / "children"
    children_dir.mkdir(parents=True, exist_ok=True)
    _assert_child_container(children_dir, require_exact=False)
    target = children_dir / price_id
    if (parent / "fit_parent_manifest.json").exists() and not target.exists():
        raise RuntimeError("fit parent已封存，禁止补写child")
    if target.exists():
        loaded = load_fit_price_child(
            target,
            expected_zone=zone,
            expected_seed=seed,
            expected_price_id=price_id,
            expected_compact_source_identity_sha256=compact_source_identity_sha256,
            expected_endpoint_root_sha256=endpoint_root_sha256,
            expected_execution_identity_sha256=execution_identity_sha256,
            expected_state_universe=expected_state_universe,
            formal_identity=formal_identity,
        )
        if loaded.manifest["fit_child_identity_sha256"] != fit_sha:
            raise RuntimeError("既有fit child为不同科学身份，拒绝覆盖")
        return {**loaded.manifest, "manifest_path": str(target / "manifest.json"), "manifest_sha256": sha256_file(target / "manifest.json"), "resumed": True}

    if formal_identity:
        official_states, index_identity = official_state_universe()
        caller_state_sha = _state_universe_sha(expected_state_universe)
        if (
            len(expected_state_universe) != FORMAL_STATE_COUNT
            or caller_state_sha != FORMAL_STATE_UNIVERSE_SHA256
        ):
            raise RuntimeError("formal fit writer拒绝caller自报state universe")
        expected_state_universe = official_states
        index_sha = FORMAL_INDEX_CONTRACT_SHA256
    else:
        index_identity = None
        index_sha = None
    decisions = child.clara.decisions.copy()
    evidence = child.clara.action_evidence.copy()
    if set(decisions["price_id"].astype(str)) != {price_id} or set(evidence["price_id"].astype(str)) != {price_id}:
        raise RuntimeError("fit child CLARA表混入其他价格")
    state_sha = _validate_state_universe(decisions, evidence, expected_state_universe)
    _validate_compact_source_provenance(
        compact_source_audit,
        compact_stream_audit,
        zone=zone,
        seed=seed,
        compact_source_identity_sha256=compact_source_identity_sha256,
        endpoint_root_sha256=endpoint_root_sha256,
    )
    if formal_identity:
        core.validate_endpoint_root_seal(
            core.FORMAL_ENDPOINT_ROOT_MANIFEST,
            expected_endpoint_root_sha256=endpoint_root_sha256,
        )
    derived_sha = str(identity.get("derived_contract_sha256"))
    if not _is_sha256(derived_sha):
        raise RuntimeError("fit child未绑定六动作派生合同")
    _validate_fit_policy_semantics(
        decisions=decisions,
        evidence=evidence,
        selector=child.cart.selector,
        compact_statistics=child.cart.compact_statistics,
        clara_audit=child.clara.audit,
        cart_audit=child.cart.audit,
        child_audit=child.audit,
        fit_identity=identity,
        zone=zone,
        seed=seed,
        price=price,
        formal_identity=formal_identity,
    )
    tmp = children_dir / f".{price_id}.tmp__{uuid.uuid4().hex}"
    tmp.mkdir()
    try:
        frames = {
            "clara_decisions": decisions,
            "clara_action_evidence": evidence,
            "cart_compact_statistics": child.cart.compact_statistics.copy(),
            "compact_stream_audit": compact_stream_audit.copy(),
        }
        # Probe every state; a partial probe cannot detect a corrupted branch used
        # only outside an arbitrary head sample.
        probe = decisions.loc[:, ["full_state_code", *core.STATE_FIELDS]].copy()
        probe.insert(0, "event_id", [f"cart-artifact-probe-{index:04d}" for index in range(len(probe))])
        selected, scores = core.predict_cart_actions(probe, child.cart.selector)
        probe["expected_action"] = selected
        probe["expected_score"] = scores
        frames["cart_probe"] = probe
        for key, frame in frames.items():
            _write_parquet(tmp / FIT_FILES[key], frame)
        _write_json(tmp / FIT_FILES["clara_audit"], dict(child.clara.audit))
        _write_json(tmp / FIT_FILES["cart_audit"], dict(child.cart.audit))
        _write_json(tmp / FIT_FILES["child_audit"], dict(child.audit))
        _write_json(tmp / FIT_FILES["compact_source_audit"], dict(compact_source_audit))
        _write_cart(child.cart.selector, tmp, derived_sha)
        config_fields = (
            "clara_configuration_sha256",
            "cart_configuration_sha256",
            "clara_v4_adaptive_subcontract_sha256",
            "cart_frozen_zone_configuration_sha256",
            "derived_contract_sha256",
        )
        config_identities = {field: str(identity.get(field, "")) for field in config_fields}
        if any(not _is_sha256(value) for value in config_identities.values()):
            raise RuntimeError("fit child配置身份不完整")
        file_records = _file_records(tmp, FIT_FILES, frames)
        manifest = _payload_with_sha(
            {
                "schema": FIT_CHILD_SCHEMA,
                "status": "PASS",
                "heldout_zone": zone,
                "seed": seed,
                "price_id": price_id,
                "price_sha256": core.price_sha256(price),
                "action_library_sha256": core.action_library_sha256(),
                "compact_source_identity_sha256": str(compact_source_identity_sha256),
                "endpoint_root_sha256": str(endpoint_root_sha256),
                "execution_identity_sha256": str(execution_identity_sha256),
                "execution_identity": (
                    None if execution_identity is None else dict(execution_identity)
                ),
                "fit_child_identity": identity,
                "fit_child_identity_sha256": fit_sha,
                "state_universe_sha256": state_sha,
                "state_count": len(expected_state_universe),
                "index_contract_sha256": index_sha,
                "index_contract_identity": index_identity,
                "config_identities": config_identities,
                "core_module_sha256": _core_sha256(),
                "writer_module_sha256": _module_sha256(),
                "software_versions": _software_versions(),
                "files": file_records,
            }
        )
        _write_json(tmp / "manifest.json", manifest)
        staged_loaded = load_fit_price_child(
            tmp,
            expected_zone=zone,
            expected_seed=seed,
            expected_price_id=price_id,
            expected_compact_source_identity_sha256=compact_source_identity_sha256,
            expected_endpoint_root_sha256=endpoint_root_sha256,
            expected_execution_identity_sha256=execution_identity_sha256,
            expected_state_universe=expected_state_universe,
            formal_identity=formal_identity,
            _staged=True,
        )
        if staged_loaded.manifest["fit_child_identity_sha256"] != fit_sha:
            raise RuntimeError("staged fit child深验身份失配")
        staged_manifest_sha = sha256_file(tmp / "manifest.json")
        # The expensive caller-side fit and artifact serialization can span a
        # control-file change.  Revalidate the live execution capability at the
        # actual atomic commit boundary, not only at function entry.
        _validate_execution_identity(
            execution_identity_sha256,
            execution_identity,
            formal_identity=formal_identity,
        )
        os.replace(tmp, target)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp)
        raise
    return {
        **staged_loaded.manifest,
        "manifest_path": str(target / "manifest.json"),
        "manifest_sha256": staged_manifest_sha,
        "resumed": False,
    }


def load_fit_price_child(
    path: Path,
    *,
    expected_zone: str,
    expected_seed: int,
    expected_price_id: str,
    expected_compact_source_identity_sha256: str,
    expected_endpoint_root_sha256: str,
    expected_execution_identity_sha256: str,
    expected_state_universe: pd.DataFrame,
    expected_config_identities: Mapping[str, str] | None = None,
    formal_identity: bool = True,
    _staged: bool = False,
) -> LoadedFitPriceArtifact:
    if not _is_sha256(expected_execution_identity_sha256):
        raise RuntimeError("expected execution identity SHA非法")
    directory = Path(path).parent if Path(path).name == "manifest.json" else Path(path)
    if formal_identity:
        _assert_formal_parent_path(directory.parent.parent, kind="fit", zone=expected_zone, seed=expected_seed)
    _assert_child_load_location(
        directory,
        expected_child_name=expected_price_id,
        staged=bool(_staged),
    )
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"fit child manifest不存在: {manifest_path}")
    manifest = _load_json(manifest_path)
    _verify_payload_sha(manifest, fields=FIT_MANIFEST_FIELDS, label="fit child manifest")
    price = _price_from_id(expected_price_id)
    if (
        manifest["schema"] != FIT_CHILD_SCHEMA
        or manifest["status"] != "PASS"
        or str(manifest["heldout_zone"]) != str(expected_zone)
        or int(manifest["seed"]) != int(expected_seed)
        or str(manifest["price_id"]) != str(expected_price_id)
        or str(manifest["price_sha256"]) != core.price_sha256(price)
        or str(manifest["action_library_sha256"]) != core.action_library_sha256()
        or str(manifest["compact_source_identity_sha256"]) != str(expected_compact_source_identity_sha256)
        or str(manifest["endpoint_root_sha256"]) != str(expected_endpoint_root_sha256)
        or str(manifest["execution_identity_sha256"])
        != str(expected_execution_identity_sha256)
        or str(manifest["core_module_sha256"]) != _core_sha256()
        or str(manifest["writer_module_sha256"]) != _module_sha256()
    ):
        raise RuntimeError("fit child manifest轴/科学身份失配")
    _validate_execution_identity(
        str(manifest["execution_identity_sha256"]),
        manifest.get("execution_identity"),
        formal_identity=formal_identity,
    )
    identity = dict(manifest["fit_child_identity"])
    fit_sha = _verify_embedded_identity(identity, "fit_child_identity_sha256", schema="TEST_CLARA_GEFCOM_6A_FIT_PRICE_CHILD_IDENTITY_V1")
    if fit_sha != str(manifest["fit_child_identity_sha256"]):
        raise RuntimeError("fit child manifest与embedded identity SHA失配")
    expected_embedded_configs = {
        field: str(identity.get(field, ""))
        for field in (
            "clara_configuration_sha256",
            "cart_configuration_sha256",
            "clara_v4_adaptive_subcontract_sha256",
            "cart_frozen_zone_configuration_sha256",
            "derived_contract_sha256",
        )
    }
    if (
        dict(manifest["config_identities"]) != expected_embedded_configs
        or any(not _is_sha256(value) for value in expected_embedded_configs.values())
        or str(identity.get("heldout_zone")) != str(expected_zone)
        or int(identity.get("seed", -1)) != int(expected_seed)
        or str(identity.get("price_id")) != price.price_id
        or str(identity.get("price_sha256")) != core.price_sha256(price)
        or str(identity.get("action_library_sha256"))
        != core.action_library_sha256()
        or tuple(identity.get("ordered_actions", [])) != core.SIX_ACTIONS
        or str(identity.get("compact_source_identity_sha256"))
        != str(expected_compact_source_identity_sha256)
    ):
        raise RuntimeError("fit child embedded/outer/config identity不闭合")
    if expected_config_identities is not None and dict(manifest["config_identities"]) != {
        str(key): str(value) for key, value in expected_config_identities.items()
    }:
        raise RuntimeError("fit child配置身份失配")
    _verify_files(directory, manifest["files"], FIT_FILES)
    if formal_identity:
        official_states, official_index_identity = official_state_universe()
        if (
            str(manifest["index_contract_sha256"])
            != FORMAL_INDEX_CONTRACT_SHA256
            or dict(manifest["index_contract_identity"])
            != official_index_identity
            or str(manifest["state_universe_sha256"])
            != FORMAL_STATE_UNIVERSE_SHA256
            or int(manifest["state_count"]) != FORMAL_STATE_COUNT
            or _state_universe_sha(expected_state_universe)
            != FORMAL_STATE_UNIVERSE_SHA256
        ):
            raise RuntimeError("fit child官方index/state identity失配")
        expected_state_universe = official_states
    frames = {
        key: pd.read_parquet(directory / relative)
        for key, relative in FIT_FILES.items()
        if relative.endswith(".parquet")
    }
    for key, frame in frames.items():
        row_count = manifest["files"][key]["row_count"]
        if row_count is None or len(frame) != int(row_count):
            raise RuntimeError(f"fit child parquet行数失配: {key}")
    state_sha = _validate_state_universe(
        frames["clara_decisions"], frames["clara_action_evidence"], expected_state_universe
    )
    if state_sha != str(manifest["state_universe_sha256"]) or len(expected_state_universe) != int(manifest["state_count"]):
        raise RuntimeError("fit child state universe SHA/计数失配")
    source_audit = _load_json(directory / FIT_FILES["compact_source_audit"])
    _validate_compact_source_provenance(
        source_audit,
        frames["compact_stream_audit"],
        zone=expected_zone,
        seed=expected_seed,
        compact_source_identity_sha256=expected_compact_source_identity_sha256,
        endpoint_root_sha256=expected_endpoint_root_sha256,
    )
    if formal_identity:
        core.validate_endpoint_root_seal(
            core.FORMAL_ENDPOINT_ROOT_MANIFEST,
            expected_endpoint_root_sha256=expected_endpoint_root_sha256,
        )
    selector = _load_cart(directory)
    clara_audit = _load_json(directory / FIT_FILES["clara_audit"])
    cart_audit = _load_json(directory / FIT_FILES["cart_audit"])
    child_audit = _load_json(directory / FIT_FILES["child_audit"])
    _validate_fit_policy_semantics(
        decisions=frames["clara_decisions"],
        evidence=frames["clara_action_evidence"],
        selector=selector,
        compact_statistics=frames["cart_compact_statistics"],
        clara_audit=clara_audit,
        cart_audit=cart_audit,
        child_audit=child_audit,
        fit_identity=identity,
        zone=expected_zone,
        seed=expected_seed,
        price=price,
        formal_identity=formal_identity,
    )
    probe = frames["cart_probe"]
    selected, scores = core.predict_cart_actions(probe, selector)
    if not np.array_equal(selected.astype(str), probe["expected_action"].astype(str).to_numpy()) or not np.allclose(
        scores, probe["expected_score"].to_numpy(dtype=float), atol=1e-12, rtol=0.0
    ):
        raise RuntimeError("CART safe tree reload probe失配")
    return LoadedFitPriceArtifact(
        manifest=manifest,
        clara_decisions=frames["clara_decisions"],
        clara_action_evidence=frames["clara_action_evidence"],
        cart_selector=selector,
        cart_compact_statistics=frames["cart_compact_statistics"],
        compact_source_audit=source_audit,
        compact_stream_audit=frames["compact_stream_audit"],
    )


def _manifest_result(path: Path, manifest: Mapping[str, Any], *, resumed: bool) -> dict[str, Any]:
    return {
        **dict(manifest),
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "resumed": bool(resumed),
    }


def _commit_manifest(path: Path, manifest: Mapping[str, Any]) -> bool:
    """Write one immutable manifest; return True only when an equal seal resumed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_temporary_entries(path.parent)
    if path.exists():
        if path.read_bytes() != canonical_json_bytes(manifest):
            raise RuntimeError(f"既有manifest身份不同，拒绝覆盖: {path}")
        return True
    temporary = path.parent / f".{path.name}.tmp__{uuid.uuid4().hex}"
    try:
        temporary.write_bytes(canonical_json_bytes(manifest))
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return False


def _load_fit_manifest_header(path: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = Path(path) if Path(path).name == "manifest.json" else Path(path) / "manifest.json"
    directory = manifest_path.parent
    _assert_no_temporary_entries(directory.parent)
    manifest = _load_json(manifest_path)
    _verify_payload_sha(manifest, fields=FIT_MANIFEST_FIELDS, label="fit child manifest")
    if manifest.get("schema") != FIT_CHILD_SCHEMA or manifest.get("status") != "PASS":
        raise RuntimeError("fit child manifest状态失配")
    identity_sha = _verify_embedded_identity(
        dict(manifest["fit_child_identity"]),
        "fit_child_identity_sha256",
        schema="TEST_CLARA_GEFCOM_6A_FIT_PRICE_CHILD_IDENTITY_V1",
    )
    if identity_sha != str(manifest["fit_child_identity_sha256"]):
        raise RuntimeError("fit child embedded identity交叉接线")
    _verify_files(directory, manifest["files"], FIT_FILES)
    return manifest, manifest_path


def seal_fit_parent(
    parent_dir: Path,
    *,
    stats: core.SixActionSufficientStats,
    child_manifest_paths: Sequence[Path],
    endpoint_root_sha256: str,
    execution_identity_sha256: str,
    execution_identity: Mapping[str, Any] | None = None,
    formal_identity: bool = True,
) -> dict[str, Any]:
    """Seal a fit parent only after all exact five independently valid children."""

    _reject_nonformal_official_root(
        Path(parent_dir), formal_identity=formal_identity, kind="fit"
    )
    if formal_identity:
        core.assert_formal_gefcom_identity(stats)
    zone = str(stats.heldout_zone)
    seed = int(stats.seed)
    compact_sha = str(stats.audit["identity_sha256"])
    if formal_identity:
        _assert_formal_parent_path(Path(parent_dir), kind="fit", zone=zone, seed=seed)
    _validate_execution_identity(
        execution_identity_sha256,
        execution_identity,
        formal_identity=formal_identity,
    )
    price_ids = _formal_price_ids()
    _assert_parent_container(Path(parent_dir), "fit_parent_manifest.json", allow_missing_seal=True)
    _assert_child_container(Path(parent_dir) / "children", require_exact=True)
    paths = tuple(Path(value) for value in child_manifest_paths)
    if len(paths) != 5:
        raise RuntimeError("fit parent必须提供exact五价child")
    child_manifest_shas: dict[str, str] = {}
    child_identity_shas: dict[str, str] = {}
    for price_id, manifest_path in zip(price_ids, paths):
        expected_path = Path(parent_dir) / "children" / price_id / "manifest.json"
        if manifest_path.resolve() != expected_path.resolve():
            raise RuntimeError("fit parent child manifest路径交叉接线")
        loaded = load_fit_price_child(
            manifest_path,
            expected_zone=zone,
            expected_seed=seed,
            expected_price_id=price_id,
            expected_compact_source_identity_sha256=compact_sha,
            expected_endpoint_root_sha256=endpoint_root_sha256,
            expected_execution_identity_sha256=execution_identity_sha256,
            expected_state_universe=stats.states,
            formal_identity=formal_identity,
        )
        observed_child_sha = str(loaded.manifest["fit_child_identity_sha256"])
        child_identity_shas[price_id] = observed_child_sha
        child_manifest_shas[price_id] = sha256_file(manifest_path)
    parent_identity = core.fit_parent_identity(stats, child_identity_shas)
    parent_sha = _verify_embedded_identity(
        parent_identity,
        "fit_parent_identity_sha256",
        schema="TEST_CLARA_GEFCOM_6A_FIT_PARENT_IDENTITY_V1",
    )
    manifest = _payload_with_sha(
        {
            "schema": FIT_PARENT_SCHEMA,
            "status": "PASS",
            "heldout_zone": zone,
            "seed": seed,
            "action_library_sha256": core.action_library_sha256(),
            "compact_source_identity_sha256": compact_sha,
            "endpoint_root_sha256": str(endpoint_root_sha256),
            "execution_identity_sha256": str(execution_identity_sha256),
            "execution_identity": (
                None if execution_identity is None else dict(execution_identity)
            ),
            "fit_parent_identity": parent_identity,
            "fit_parent_identity_sha256": parent_sha,
            "price_child_identity_sha256": child_identity_shas,
            "price_child_manifest_sha256": child_manifest_shas,
            "price_child_count": 5,
        }
    )
    manifest_path = Path(parent_dir) / "fit_parent_manifest.json"
    _validate_execution_identity(
        execution_identity_sha256,
        execution_identity,
        formal_identity=formal_identity,
    )
    resumed = _commit_manifest(manifest_path, manifest)
    _assert_parent_container(Path(parent_dir), "fit_parent_manifest.json", allow_missing_seal=False)
    return _manifest_result(manifest_path, manifest, resumed=resumed)


CELL_COLUMNS = core.StreamingMetricAccumulator.CELL_METRIC_COLUMNS
ACTION_COLUMNS = core.StreamingMetricAccumulator.ACTION_COUNT_COLUMNS
BLOCK_COLUMNS = core.StreamingMetricAccumulator.PAIRED_BLOCK_COLUMNS
DIAGNOSTIC_COLUMNS = core.StreamingMetricAccumulator.DIAGNOSTIC_METRIC_COLUMNS
CLARA_STATE_COUNT_COLUMNS = core.StreamingMetricAccumulator.CLARA_STATE_COUNT_COLUMNS
CONSERVATION_COLUMNS = (
    *core.POLICY_SCOPE_FIELDS,
    "method",
    "event_count",
    "hash_sum_u64",
    "hash_xor_u64",
    "hash_square_sum_u64",
)


def _validate_replay_frames(
    frames: Mapping[str, pd.DataFrame],
    *,
    zone: str,
    seed: int,
    policy_mode: str,
    evaluation_price_id: str,
    policy_price_id: str,
    expected_signature: Mapping[str, Any],
    partition_records: Mapping[str, Mapping[str, Any]],
) -> None:
    expected_frame_names = {
        "cell_metrics",
        "action_counts",
        "paired_blocks",
        "diagnostic_metrics",
        "clara_state_counts",
        "event_conservation",
    }
    if set(frames) != expected_frame_names:
        raise RuntimeError("replay child必须恰含六类compact frame")
    expected_columns = {
        "cell_metrics": CELL_COLUMNS,
        "action_counts": ACTION_COLUMNS,
        "paired_blocks": BLOCK_COLUMNS,
        "diagnostic_metrics": DIAGNOSTIC_COLUMNS,
        "clara_state_counts": CLARA_STATE_COUNT_COLUMNS,
        "event_conservation": CONSERVATION_COLUMNS,
    }
    for name, frame in frames.items():
        if tuple(frame.columns) != tuple(expected_columns[name]) or frame.empty:
            raise RuntimeError(f"replay compact frame schema/空值失配: {name}")
        if name != "diagnostic_metrics" and frame.isna().any().any():
            raise RuntimeError(f"replay compact frame含NaN: {name}")
        expected_methods = (
            {"CLARA_6A"}
            if name == "clara_state_counts"
            else set(core.FORMAL_METHODS)
        )
        if (
            set(frame["policy_mode"].astype(str)) != {str(policy_mode)}
            or set(frame["evaluation_price_id"].astype(str))
            != {str(evaluation_price_id)}
            or set(frame["policy_price_id"].astype(str)) != {str(policy_price_id)}
            or set(frame["method"].astype(str)) != expected_methods
        ):
            raise RuntimeError(f"replay compact frame价格/九方法不闭合: {name}")
        if name != "event_conservation" and (
            set(frame["zone_or_farm"].astype(str)) != {str(zone)}
            or set(frame["seed"].astype(int)) != {int(seed)}
        ):
            raise RuntimeError(f"replay compact frame zone/seed交叉接线: {name}")
    if set(expected_signature) != SIGNATURE_FIELDS or not _is_sha256(
        expected_signature.get("canonical_sorted_event_id_sha256")
    ):
        raise RuntimeError("replay expected event signature字段不完整")
    expected_count = int(expected_signature["event_count"])
    totals = {
        "cell_metrics": frames["cell_metrics"].groupby("method")["event_count"].sum(),
        "action_counts": frames["action_counts"].groupby("method")["action_count"].sum(),
        "paired_blocks": frames["paired_blocks"].groupby("method")["event_count"].sum(),
        "diagnostic_metrics": frames["diagnostic_metrics"]
        .loc[lambda value: value["regime"].astype(str).eq("overall")]
        .groupby("method")["event_count"]
        .sum(),
        "event_conservation": frames["event_conservation"].set_index("method")["event_count"],
    }
    if any(set(series.index.astype(str)) != set(core.FORMAL_METHODS) or not series.eq(expected_count).all() for series in totals.values()):
        raise RuntimeError("replay compact四表事件数不守恒")
    conservation = frames["event_conservation"].set_index("method")
    for field in ("hash_sum_u64", "hash_xor_u64", "hash_square_sum_u64"):
        if not conservation[field].map(str).eq(str(expected_signature[field])).all():
            raise RuntimeError(f"replay compact event signature失配: {field}")
    if not set(frames["action_counts"]["selected_action"].astype(str)).issubset(set(core.SIX_ACTIONS)):
        raise RuntimeError("replay action counts含六动作库外值")
    finite_nonnegative = {
        "cell_metrics": (
            "event_count",
            "errf_sum",
            "reserve_up_sum",
            "reserve_down_sum",
            "miss_upper_sum",
            "miss_lower_sum",
            "covered_sum",
            "coverage_target_sum",
            "width_sum",
            "interval_score_sum",
        ),
        "action_counts": ("action_count",),
        "paired_blocks": ("event_count", "errf_sum", "covered_sum", "width_sum"),
    }
    for name, columns in finite_nonnegative.items():
        numeric = frames[name].loc[:, list(columns)].to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or np.any(numeric < 0.0):
            raise RuntimeError(f"replay compact frame数值非法: {name}")
    if not np.isfinite(
        frames["cell_metrics"]["coverage_gap_sum"].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("replay cell coverage_gap_sum非有限")
    _validate_partition_records(
        partition_records,
        zone=zone,
        seed=seed,
        expected_signature=expected_signature,
    )
    partition_keys = ["horizon_steps", "predictor", "target_coverage"]
    expected_partition_counts = {
        (
            int(record["horizon_steps"]),
            str(record["predictor"]),
            float(record["target_coverage"]),
        ): int(record["event_count"])
        for record in partition_records.values()
    }
    cell = frames["cell_metrics"]
    actions = frames["action_counts"]
    blocks = frames["paired_blocks"]
    diagnostics = frames["diagnostic_metrics"]
    clara_states = frames["clara_state_counts"]
    if (
        cell.duplicated([*core.POLICY_SCOPE_FIELDS, "method", *core.StreamingMetricAccumulator.CELL_FIELDS]).any()
        or actions.duplicated(
            [
                *core.POLICY_SCOPE_FIELDS,
                "method",
                *core.StreamingMetricAccumulator.CELL_FIELDS,
                "selected_action",
            ]
        ).any()
        or blocks.duplicated(
            [*core.POLICY_SCOPE_FIELDS, "method", *core.StreamingMetricAccumulator.BLOCK_FIELDS]
        ).any()
        or diagnostics.duplicated(
            [
                *core.POLICY_SCOPE_FIELDS,
                "method",
                *core.StreamingMetricAccumulator.DIAGNOSTIC_FIELDS,
            ]
        ).any()
        or clara_states.duplicated(
            [
                *core.POLICY_SCOPE_FIELDS,
                "method",
                *core.StreamingMetricAccumulator.CLARA_STATE_COUNT_FIELDS,
            ]
        ).any()
        or frames["event_conservation"].duplicated([*core.POLICY_SCOPE_FIELDS, "method"]).any()
    ):
        raise RuntimeError("replay compact frame含重复聚合键")
    expected_cells = set(expected_partition_counts)
    for name, frame, count_column in (
        ("cell_metrics", cell, "event_count"),
        ("action_counts", actions, "action_count"),
        ("paired_blocks", blocks, "event_count"),
    ):
        observed_cells = set(
            frame[partition_keys]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        observed_cells = {
            (int(horizon), str(predictor), float(coverage))
            for horizon, predictor, coverage in observed_cells
        }
        if observed_cells != expected_cells:
            raise RuntimeError(f"replay {name}未覆盖exact 220 partitions")
        totals_by_partition = frame.groupby(
            ["method", *partition_keys], sort=True, dropna=False
        )[count_column].sum()
        for method in core.FORMAL_METHODS:
            method_totals = totals_by_partition.loc[method]
            for key, expected_count in expected_partition_counts.items():
                lookup = (key[0], key[1], key[2])
                if int(method_totals.loc[lookup]) != expected_count:
                    raise RuntimeError(
                        f"replay {name}逐partition事件数失配: {method}/{key}"
                    )
    diagnostic_overall = diagnostics[
        diagnostics["regime"].astype(str).eq("overall")
    ]
    observed_diagnostic_cells = {
        (int(horizon), str(predictor), float(coverage))
        for horizon, predictor, coverage in diagnostic_overall[partition_keys]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    if observed_diagnostic_cells != expected_cells:
        raise RuntimeError("replay diagnostic overall未覆盖exact 220 partitions")
    diagnostic_partition_totals = diagnostic_overall.groupby(
        ["method", *partition_keys], sort=True, dropna=False
    )["event_count"].sum()
    for method in core.FORMAL_METHODS:
        for partition_key, partition_count in expected_partition_counts.items():
            if int(diagnostic_partition_totals.loc[(method, *partition_key)]) != int(
                partition_count
            ):
                raise RuntimeError(
                    f"replay diagnostic overall partition数失配: {method}/{partition_key}"
                )
    observed_state_cells = {
        (int(horizon), str(predictor), float(coverage))
        for horizon, predictor, coverage in clara_states[partition_keys]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    if observed_state_cells != expected_cells:
        raise RuntimeError("replay CLARA state counts未覆盖exact 220 partitions")
    state_partition_totals = clara_states.groupby(
        partition_keys, sort=True, dropna=False
    )["event_count"].sum()
    for partition_key, partition_count in expected_partition_counts.items():
        if int(state_partition_totals.loc[partition_key]) != int(partition_count):
            raise RuntimeError(
                f"replay CLARA state partition数失配: {partition_key}"
            )
    cell_keys = [
        *core.POLICY_SCOPE_FIELDS,
        "method",
        *core.StreamingMetricAccumulator.CELL_FIELDS,
    ]
    action_by_cell = actions.groupby(cell_keys, sort=True, as_index=False)[
        "action_count"
    ].sum()
    cell_action = cell[cell_keys + ["event_count"]].merge(
        action_by_cell,
        on=cell_keys,
        how="outer",
        validate="one_to_one",
    )
    if cell_action.isna().any().any() or not np.array_equal(
        cell_action["event_count"].to_numpy(dtype=np.int64),
        cell_action["action_count"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError("replay cell与action count代数不闭合")
    block_by_cell = blocks.groupby(cell_keys, sort=True, as_index=False)[
        ["event_count", "errf_sum", "covered_sum", "width_sum"]
    ].sum()
    cell_block = cell[
        cell_keys + ["event_count", "errf_sum", "covered_sum", "width_sum"]
    ].merge(
        block_by_cell,
        on=cell_keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_cell", "_block"),
    )
    for field in ("event_count", "errf_sum", "covered_sum", "width_sum"):
        if not np.allclose(
            cell_block[f"{field}_cell"].to_numpy(dtype=float),
            cell_block[f"{field}_block"].to_numpy(dtype=float),
            atol=1e-10,
            rtol=1e-12,
        ):
            raise RuntimeError(f"replay cell与paired block代数失配: {field}")
    if (
        np.any(cell["event_count"].to_numpy(dtype=np.int64) <= 0)
        or np.any(cell["covered_sum"] > cell["event_count"])
        or not np.allclose(
            cell["coverage_target_sum"],
            cell["target_coverage"] * cell["event_count"],
            atol=1e-12,
            rtol=0.0,
        )
        or not np.allclose(
            cell["coverage_gap_sum"],
            cell["covered_sum"] - cell["coverage_target_sum"],
            atol=1e-12,
            rtol=0.0,
        )
    ):
        raise RuntimeError("replay cell metric计数边界不闭合")
    # Run the same exact algebra used before core signs the in-process metric
    # capability.  Artifact write/load both invoke this on the serialized frames.
    core.StreamingMetricAccumulator._validate_diagnostic_outputs(
        diagnostics=diagnostics,
        clara_states=clara_states,
        cell_metrics=cell,
        action_counts=actions,
    )


def _validate_partition_records(
    records: Mapping[str, Mapping[str, Any]],
    *,
    zone: str,
    seed: int,
    expected_signature: Mapping[str, Any],
) -> None:
    expected_cells = {
        (horizon, predictor, coverage)
        for horizon in core.FORMAL_HORIZONS
        for predictor in core.FORMAL_PREDICTORS
        for coverage in core.FORMAL_COVERAGES
    }
    if not isinstance(records, Mapping) or len(records) != len(expected_cells):
        raise RuntimeError("replay partition必须完整覆盖5h×4pred×11coverage")
    observed_cells: set[tuple[int, str, float]] = set()
    combined = {
        "event_count": 0,
        "hash_sum_u64": 0,
        "hash_xor_u64": 0,
        "hash_square_sum_u64": 0,
    }
    modulus = 1 << 64
    for partition_id, record in records.items():
        if not str(partition_id) or set(record) != PARTITION_RECORD_FIELDS:
            raise RuntimeError("replay partition record schema失配")
        if (
            str(record["zone_or_farm"]) != str(zone)
            or int(record["seed"]) != int(seed)
            or not _is_sha256(record["canonical_sorted_event_id_sha256"])
            or int(record["event_count"]) <= 0
        ):
            raise RuntimeError("replay partition zone/seed/signature失配")
        coverage_label = format(float(record["target_coverage"]), ".12g").replace(
            "-", "m"
        ).replace(".", "p")
        expected_partition_id = (
            f"H{int(record['horizon_steps']):02d}__{record['predictor']}__"
            f"C{coverage_label}"
        )
        if str(partition_id) != expected_partition_id:
            raise RuntimeError("replay partition ID与轴字段失配")
        observed_cells.add(
            (
                int(record["horizon_steps"]),
                str(record["predictor"]),
                float(record["target_coverage"]),
            )
        )
        combined["event_count"] += int(record["event_count"])
        combined["hash_sum_u64"] = (
            combined["hash_sum_u64"] + int(record["hash_sum_u64"])
        ) % modulus
        combined["hash_xor_u64"] ^= int(record["hash_xor_u64"])
        combined["hash_square_sum_u64"] = (
            combined["hash_square_sum_u64"]
            + int(record["hash_square_sum_u64"])
        ) % modulus
    if observed_cells != expected_cells:
        raise RuntimeError("replay partition axis含重复/缺失/越界cell")
    for field in combined:
        if int(combined[field]) != int(expected_signature[field]):
            raise RuntimeError(f"replay partition聚合签名失配: {field}")


def _combined_numeric_signature(
    signatures: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    result = {
        "event_count": 0,
        "hash_sum_u64": 0,
        "hash_xor_u64": 0,
        "hash_square_sum_u64": 0,
    }
    modulus = 1 << 64
    for signature in signatures:
        result["event_count"] += int(signature["event_count"])
        result["hash_sum_u64"] = (
            result["hash_sum_u64"] + int(signature["hash_sum_u64"])
        ) % modulus
        result["hash_xor_u64"] ^= int(signature["hash_xor_u64"])
        result["hash_square_sum_u64"] = (
            result["hash_square_sum_u64"]
            + int(signature["hash_square_sum_u64"])
        ) % modulus
    return result


def validate_target_input_manifest(
    path: Path,
    *,
    expected_zone: str,
    expected_seed: int,
    expected_endpoint_root_sha256: str,
    expected_execution_identity_sha256: str,
    expected_partition_records: Mapping[str, Mapping[str, Any]],
    expected_combined_signature: Mapping[str, Any],
    formal_identity: bool,
) -> tuple[dict[str, Any], str]:
    manifest_path = Path(path)
    if formal_identity:
        expected_path = (
            FORMAL_REPLAY_ROOT
            / f"{expected_zone}__seed{int(expected_seed)}"
            / "target_input_manifest.json"
        )
        if manifest_path.resolve() != expected_path.resolve():
            raise RuntimeError("target input manifest拒绝非官方路径")
    manifest = _load_json(manifest_path)
    _verify_payload_sha(manifest, fields=TARGET_INPUT_FIELDS, label="target input manifest")
    import importlib

    loader = importlib.import_module("gefcom_six_action_selector_target_loader_v1")
    loader_sha = sha256_file(Path(loader.__file__))
    if (
        manifest.get("schema") != "TEST_CLARA_GEFCOM_6A_TARGET_INPUT_PARENT_V1"
        or manifest.get("status") != "PASS"
        or str(manifest.get("evaluation_zone")) != str(expected_zone)
        or int(manifest.get("seed", -1)) != int(expected_seed)
        or (
            formal_identity
            and Path(manifest.get("endpoint_root_path", "")).resolve()
            != core.FORMAL_ENDPOINT_ROOT_MANIFEST.resolve()
        )
        or str(manifest.get("endpoint_root_sha256"))
        != str(expected_endpoint_root_sha256)
        or str(manifest.get("execution_identity_sha256"))
        != str(expected_execution_identity_sha256)
        or str(manifest.get("target_loader_module_sha256")) != loader_sha
        or int(manifest.get("stream_count", -1)) != 20
        or int(manifest.get("unit_count", -1)) != 5
        or int(manifest.get("partition_count", -1)) != 220
        or dict(manifest.get("partition_records", {}))
        != {str(key): dict(value) for key, value in expected_partition_records.items()}
        or dict(manifest.get("combined_event_signature", {}))
        != dict(expected_combined_signature)
    ):
        raise RuntimeError("target input manifest axis/code/root/execution身份失配")
    _validate_partition_records(
        dict(manifest["partition_records"]),
        zone=expected_zone,
        seed=expected_seed,
        expected_signature=expected_combined_signature,
    )
    stream_audits = dict(manifest.get("stream_audits", {}))
    expected_stream_ids = {
        f"H{horizon:02d}__{predictor}"
        for horizon in core.FORMAL_HORIZONS
        for predictor in core.FORMAL_PREDICTORS
    }
    if set(stream_audits) != expected_stream_ids:
        raise RuntimeError("target input stream audit 5h×4pred矩阵失配")
    for horizon in core.FORMAL_HORIZONS:
        for predictor in core.FORMAL_PREDICTORS:
            stream_key = f"H{horizon:02d}__{predictor}"
            audit = dict(stream_audits[stream_key])
            if (
                set(audit) != TARGET_STREAM_AUDIT_FIELDS
                or audit.get("schema") != loader.SCHEMA
                or audit.get("status") != "PASS"
                or str(audit.get("evaluation_zone")) != str(expected_zone)
                or str(audit.get("predictor")) != predictor
                or int(audit.get("horizon", -1)) != horizon
                or int(audit.get("seed", -1)) != int(expected_seed)
                or bool(audit.get("target_event_filter_applied"))
                or not bool(audit.get("strict_maturity_used_for_feedback_release_only"))
                or int(audit.get("target_event_count", -1))
                != int(audit.get("endpoint_root_target_event_count", -2))
                # Some individual predictor streams can legitimately have no
                # cold row.  Cold events are required and conserved at the
                # endpoint unit (four-predictor) level below.
                or int(audit.get("cold_start_event_count", -1)) < 0
                or not bool(audit.get("leaf_hashes_verified"))
                or not bool(audit.get("toctou_recheck_passed"))
                or str(audit.get("endpoint_root_manifest_sha256"))
                != str(expected_endpoint_root_sha256)
                or set(dict(audit.get("event_signature", {}))) != SIGNATURE_FIELDS
            ):
                raise RuntimeError(f"target stream audit身份/完整成员集失配: {stream_key}")
            if formal_identity:
                paths = loader.formal_target_paths(
                    evaluation_zone=expected_zone,
                    predictor=predictor,
                    horizon=horizon,
                    seed=expected_seed,
                )
                bundle = paths["candidate_bundle_dir"]
                live_leaf_paths = {
                    "source_fact": paths["source_fact"],
                    "event_core": bundle / "event_core.parquet",
                    "candidate_intervals": bundle / "candidate_intervals.parquet",
                    "feedback_trace": bundle / "feedback_trace.parquet",
                    "endpoint_addon": paths["endpoint_addon"],
                    "width_thresholds": paths["width_thresholds"],
                }
                leaf_sha = dict(audit.get("leaf_sha256", {}))
                if set(leaf_sha) != set(live_leaf_paths) or any(
                    not live_path.is_file()
                    or sha256_file(live_path) != str(leaf_sha[name])
                    for name, live_path in live_leaf_paths.items()
                ):
                    raise RuntimeError(
                        f"target stream leaf现场SHA失配: {stream_key}"
                    )
            cell_signatures = [
                record
                for record in manifest["partition_records"].values()
                if int(record["horizon_steps"]) == horizon
                and str(record["predictor"]) == predictor
            ]
            combined = _combined_numeric_signature(cell_signatures)
            if any(
                int(combined[field]) != int(audit["event_signature"][field])
                for field in combined
            ):
                raise RuntimeError(f"target stream audit与11 coverage partitions失配: {stream_key}")
    unit_records = dict(manifest.get("unit_records", {}))
    if set(unit_records) != {f"H{horizon:02d}" for horizon in core.FORMAL_HORIZONS}:
        raise RuntimeError("target input unit records五时距不完整")
    for horizon in core.FORMAL_HORIZONS:
        key = f"H{horizon:02d}"
        record = dict(unit_records[key])
        signatures = [
            stream_audits[f"H{horizon:02d}__{predictor}"]["event_signature"]
            for predictor in core.FORMAL_PREDICTORS
        ]
        combined = _combined_numeric_signature(signatures)
        if (
            set(record) != TARGET_UNIT_RECORD_FIELDS
            or str(record.get("unit_id"))
            != f"{expected_zone}__seed{int(expected_seed)}__H{horizon:02d}"
            or not _is_sha256(record.get("unit_manifest_sha256"))
            or int(record.get("expected_event_count", -1))
            != int(record.get("observed_event_count", -2))
            or int(record.get("cold_start_event_count", 0)) <= 0
            or any(
                int(combined[field]) != int(record["event_signature"][field])
                for field in combined
            )
        ):
            raise RuntimeError(f"target input unit record/root成员集失配: {key}")
    if formal_identity:
        _, root_units = core.validate_endpoint_root_seal(
            core.FORMAL_ENDPOINT_ROOT_MANIFEST,
            expected_endpoint_root_sha256=expected_endpoint_root_sha256,
        )
    else:
        root_units = {
            str(record["unit_id"]): {
                "unit_manifest_sha256": record["unit_manifest_sha256"],
                "event_count": record["expected_event_count"],
            }
            for record in unit_records.values()
        }
    for horizon in core.FORMAL_HORIZONS:
        unit_id = f"{expected_zone}__seed{int(expected_seed)}__H{horizon:02d}"
        record = unit_records[f"H{horizon:02d}"]
        unit_manifest_path = core.FORMAL_ENDPOINT_ROOT / unit_id / "manifest.json"
        stream_unit_shas = {
            str(stream_audits[f"H{horizon:02d}__{predictor}"]["endpoint_unit_manifest_sha256"])
            for predictor in core.FORMAL_PREDICTORS
        }
        if (
            str(root_units[unit_id]["unit_manifest_sha256"])
            != str(record["unit_manifest_sha256"])
            or int(root_units[unit_id]["event_count"])
            != int(record["expected_event_count"])
            or stream_unit_shas != {str(record["unit_manifest_sha256"])}
            or (
                formal_identity
                and (
                    not unit_manifest_path.is_file()
                    or sha256_file(unit_manifest_path)
                    != str(record["unit_manifest_sha256"])
                )
            )
        ):
            raise RuntimeError("target input unit与endpoint root row失配")
    return manifest, sha256_file(manifest_path)


def _validate_metric_finalize_audit(
    audit: Mapping[str, Any],
    *,
    policy_mode: str,
    evaluation_price: core.PriceSpec,
    policy_price: core.PriceSpec,
    expected_signature: Mapping[str, Any],
    partition_records: Mapping[str, Mapping[str, Any]],
    frames: Mapping[str, pd.DataFrame],
    zone: str,
    seed: int,
    fit_artifact: LoadedFitPriceArtifact | None = None,
) -> None:
    if set(audit) != METRIC_FINALIZE_AUDIT_FIELDS:
        raise RuntimeError("metric finalize audit exact schema失配")
    scope_identity = core.policy_evaluation_scope(
        policy_mode=policy_mode,
        evaluation_price=evaluation_price,
        policy_price=policy_price,
        require_formal_identity=True,
    )
    if (
        audit.get("schema") != core.SCHEMA
        or audit.get("status") != "PASS"
        or audit.get("scope") != "ZONE_SEED_PRICE_CHILD"
        or str(audit.get("policy_mode")) != str(policy_mode)
        or str(audit.get("evaluation_price_id")) != evaluation_price.price_id
        or str(audit.get("evaluation_price_sha256"))
        != core.price_sha256(evaluation_price)
        or str(audit.get("policy_price_id")) != policy_price.price_id
        or str(audit.get("policy_price_sha256")) != core.price_sha256(policy_price)
        or str(audit.get("policy_evaluation_scope_sha256"))
        != str(scope_identity["policy_evaluation_scope_sha256"])
        or bool(audit.get("policy_feedback_uses_evaluation_price"))
        or not bool(audit.get("endpoint_score_uses_evaluation_price"))
        or int(audit.get("method_count", -1)) != len(core.FORMAL_METHODS)
        or tuple(audit.get("methods", [])) != core.FORMAL_METHODS
        or int(audit.get("trusted_partition_count", -1))
        != len(core.FORMAL_HORIZONS) * len(core.FORMAL_PREDICTORS) * len(core.FORMAL_COVERAGES)
        or tuple(audit.get("trusted_partition_ids", []))
        != tuple(sorted(partition_records))
        or str(audit.get("trusted_partition_signatures_sha256"))
        != hashlib.sha256(
            json.dumps(
                {
                    partition_id: {
                        field: partition_records[partition_id][field]
                        for field in SIGNATURE_FIELDS
                    }
                    for partition_id in sorted(partition_records)
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        or str(audit.get("evaluation_zone")) != str(zone)
        or int(audit.get("seed", -1)) != int(seed)
        or int(audit.get("expected_event_count_per_method", -1))
        != int(expected_signature["event_count"])
        or dict(audit.get("expected_event_signature", {}))
        != dict(expected_signature)
        or int(audit.get("event_rows_retained", -1)) != 0
        or not bool(audit.get("retained_reduced_rows"))
        or int(audit.get("update_count", -1))
        != len(partition_records) * len(core.FORMAL_METHODS)
        or int(audit.get("input_method_event_score_count", -1))
        != int(expected_signature["event_count"]) * len(core.FORMAL_METHODS)
        or int(audit.get("cell_metric_row_count", -1))
        != len(frames["cell_metrics"])
        or int(audit.get("action_count_row_count", -1))
        != len(frames["action_counts"])
        or int(audit.get("paired_block_row_count", -1))
        != len(frames["paired_blocks"])
        or int(audit.get("diagnostic_metric_row_count", -1))
        != len(frames["diagnostic_metrics"])
        or int(audit.get("clara_state_count_row_count", -1))
        != len(frames["clara_state_counts"])
        or int(audit.get("event_conservation_row_count", -1))
        != len(frames["event_conservation"])
        or int(audit.get("event_conservation_row_count", -1))
        != len(core.FORMAL_METHODS)
        or not bool(audit.get("full_target_event_set_preserved"))
        or bool(audit.get("terminal_pending_feedback_rows_dropped"))
        or audit.get("linucb_feedback_release_rule")
        != "label_available_timestamp <= issue_time; no within-issue use"
        or audit.get("rolling_reliability_contract")
        != (
            "SELECTED_ENDPOINT_COVERED_SEQUENCE_SORTED_BY_ISSUE_TIMESTAMP_EVENT_ID;"
            "168H_WINDOW; OVERALL_ORDINARY_RAMP_RECOMPUTED_PER_METHOD_CELL"
        )
        or bool(audit.get("candidate_historical_reliability_used_as_final_metric"))
        or not bool(
            audit.get("diagnostic_rates_recoverable_from_integer_sufficient_statistics")
        )
    ):
        raise RuntimeError("metric finalize audit身份/完整target守恒失配")
    policies = dict(audit.get("method_policy_identities", {}))
    if set(policies) != set(core.FORMAL_METHODS):
        raise RuntimeError("metric finalize audit九方法policy identity不闭合")
    for method, identity in policies.items():
        binding = dict(identity)
        expected_binding_sha = hashlib.sha256(
            json.dumps(
                {
                    key: value
                    for key, value in binding.items()
                    if key != "policy_binding_sha256"
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if (
            str(binding.get("policy_binding_sha256")) != expected_binding_sha
            or str(binding.get("policy_mode")) != str(policy_mode)
            or str(binding.get("policy_price_id")) != policy_price.price_id
            or str(binding.get("policy_price_sha256"))
            != core.price_sha256(policy_price)
        ):
            raise RuntimeError(f"metric method policy binding SHA失配: {method}")
    for method, action in core.FORMAL_DETERMINISTIC_METHOD_ACTIONS.items():
        if (
            policies[method].get("schema")
            != "TEST_CLARA_GEFCOM_6A_DETERMINISTIC_POLICY_BINDING_V1"
            or policies[method].get("fixed_action") != action
        ):
            raise RuntimeError(f"metric确定性方法policy语义失配: {method}")
    if (
        policies["CLARA_6A"].get("schema")
        != "TEST_CLARA_GEFCOM_6A_CLARA_REPLAY_POLICY_BINDING_V1"
        or policies["CART_6A"].get("schema")
        != "TEST_CLARA_GEFCOM_6A_CART_REPLAY_POLICY_BINDING_V1"
        or policies["LinUCB_6A"].get("schema")
        != "TEST_CLARA_GEFCOM_6A_LINUCB_REPLAY_POLICY_BINDING_V1"
        or not all(
            _is_sha256(value)
            for method in ("CLARA_6A", "CART_6A", "LinUCB_6A")
            for key, value in policies[method].items()
            if key.endswith("sha256")
        )
    ):
        raise RuntimeError("metric learned-selector policy binding字段失配")
    if fit_artifact is not None and (
        str(policies["CLARA_6A"].get("clara_decisions_sha256"))
        != core._dataframe_content_sha256(fit_artifact.clara_decisions)
        or str(policies["CLARA_6A"].get("clara_action_evidence_sha256"))
        != core._dataframe_content_sha256(fit_artifact.clara_action_evidence)
        or str(policies["CART_6A"].get("cart_selector_sha256"))
        != core._cart_selector_content_sha256(fit_artifact.cart_selector)
    ):
        raise RuntimeError("metric CLARA/CART policy与same-price fit artifact失配")
    if fit_artifact is not None:
        state_attributes = [
            "full_state_code",
            "support_backoff_level",
            "support_backoff_name",
            "guardrail_empty_fallback",
            "selected_action_guardrail_pass",
            "selected_action",
        ]
        expected_state = fit_artifact.clara_decisions.loc[
            :, state_attributes
        ].copy()
        safe_counts = (
            fit_artifact.clara_action_evidence.groupby(
                "full_state_code", sort=True, dropna=False
            )["guardrail_pass"]
            .sum()
            .astype(np.int64)
            .rename("guardrail_safe_action_count")
            .reset_index()
        )
        expected_state = expected_state.merge(
            safe_counts,
            on="full_state_code",
            how="left",
            validate="one_to_one",
        )
        expected_state["guardrail_excluded_action_count"] = (
            len(core.SIX_ACTIONS)
            - expected_state["guardrail_safe_action_count"].astype(int)
        )
        observed_state = frames["clara_state_counts"].loc[
            :,
            [
                "full_state_code",
                "support_backoff_level",
                "support_backoff_name",
                "guardrail_safe_action_count",
                "guardrail_excluded_action_count",
                "guardrail_empty_fallback",
                "selected_action_guardrail_pass",
                "selected_action",
            ],
        ].drop_duplicates()
        joined = observed_state.merge(
            expected_state,
            on="full_state_code",
            how="left",
            validate="many_to_one",
            suffixes=("_observed", "_expected"),
        )
        compare_fields = [field for field in observed_state if field != "full_state_code"]
        if joined.isna().any().any() or any(
            not joined[f"{field}_observed"].astype(str).equals(
                joined[f"{field}_expected"].astype(str)
            )
            for field in compare_fields
        ):
            raise RuntimeError("replay CLARA state diagnostics与live fit policy失配")


def _validate_linucb_replay_audit(
    audit: Mapping[str, Any],
    *,
    zone: str,
    seed: int,
    price: core.PriceSpec,
    event_count: int,
    formal_identity: bool,
) -> dict[str, Any]:
    if set(audit) != LINUCB_REPLAY_AUDIT_FIELDS:
        raise RuntimeError("LinUCB replay audit exact bounded schema失配")
    replay_identity = dict(audit.get("replay_child_identity", {}))
    _verify_embedded_identity(
        replay_identity,
        "replay_child_identity_sha256",
        schema="TEST_CLARA_GEFCOM_6A_REPLAY_PRICE_CHILD_IDENTITY_V1",
    )
    policy_identity = dict(audit.get("linucb_policy_identity", {}))
    policy_sha = _verify_embedded_identity(
        policy_identity,
        "policy_identity_sha256",
        schema="TEST_CLARA_LINUCB_6A_POLICY_IDENTITY_V1",
    )
    updates = {
        str(action): int(value)
        for action, value in dict(audit.get("action_update_count", {})).items()
    }
    feedback_count = int(audit.get("feedback_count_before_terminal_drain", -1))
    pending_count = int(audit.get("pending_feedback_count", -1))
    alpha = float(audit.get("exploration_alpha", np.nan))
    l2 = float(audit.get("l2_regularization", np.nan))
    if (
        audit.get("schema") != core.SCHEMA
        or audit.get("status") != "PASS"
        or audit.get("method") != "LinUCB_6A"
        or str(audit.get("evaluation_zone")) != str(zone)
        or int(audit.get("seed", -1)) != int(seed)
        or str(audit.get("price_sha256")) != core.price_sha256(price)
        or dict(audit.get("price", {})) != price.as_dict()
        or tuple(audit.get("actions", [])) != core.SIX_ACTIONS
        or tuple(audit.get("horizons", [])) != core.FORMAL_HORIZONS
        or not bool(audit.get("global_issue_time_coordinator"))
        or bool(audit.get("horizon_independent_restart"))
        or not bool(audit.get("fresh_state_initialized_for_this_price"))
        or bool(audit.get("cross_price_state_shared"))
        or not bool(audit.get("strict_delayed_feedback"))
        or int(audit.get("candidate_action_count_per_event", -1)) != 6
        or int(audit.get("candidate_event_action_count", -1)) != event_count * 6
        or int(audit.get("selected_action_count", -1)) != event_count
        or not bool(audit.get("selected_action_library_closed"))
        or not _is_sha256(audit.get("retained_decisions_sha256"))
        or int(audit.get("event_count", -1)) != event_count
        or int(audit.get("final_event_count", -1)) != event_count
        or int(audit.get("future_feedback_violation_count", -1)) != 0
        or int(audit.get("within_issue_feedback_use_count", -1)) != 0
        or bool(audit.get("terminal_feedback_drain"))
        or bool(audit.get("final_feedback_drain"))
        or audit.get("terminal_drain_decision_effect")
        != "NONE_AFTER_LAST_RELEASE"
        or str(audit.get("action_library_sha256")) != core.action_library_sha256()
        or str(audit.get("global_replay_scope_sha256"))
        != core.global_replay_scope_sha256()
        or set(updates) != set(core.SIX_ACTIONS)
        or any(value < 0 for value in updates.values())
        or sum(updates.values()) != feedback_count
        or feedback_count < 0
        or pending_count < 0
        or feedback_count + pending_count != event_count
        or int(audit.get("adaptation_event_count", -1)) != 0
        or int(audit.get("context_dimension", -1)) <= 0
        or not np.isfinite(alpha)
        or alpha <= 0.0
        or not np.isfinite(l2)
        or l2 <= 0.0
        or not bool(audit.get("batch_scoring"))
        or audit.get("feedback_update_order")
        != "identical_eventwise_heap_order"
        or str(policy_identity.get("policy_identity_sha256")) != policy_sha
        or str(replay_identity.get("linucb_policy_identity_sha256")) != policy_sha
        or str(policy_identity.get("price_id")) != price.price_id
        or str(policy_identity.get("price_sha256")) != core.price_sha256(price)
        or str(policy_identity.get("evaluation_zone")) != str(zone)
        or tuple(policy_identity.get("ordered_actions", [])) != core.SIX_ACTIONS
        or tuple(policy_identity.get("global_issue_time_horizons", []))
        != core.FORMAL_HORIZONS
        or tuple(policy_identity.get("global_issue_time_predictors", []))
        != core.FORMAL_PREDICTORS
        or tuple(float(value) for value in policy_identity.get("global_issue_time_coverages", []))
        != core.FORMAL_COVERAGES
        or dict(policy_identity.get("global_scope_key", {}))
        != {"theta_id": price.price_id, "seed": int(seed), "evaluation_zone": str(zone)}
        or float(policy_identity.get("exploration_alpha", np.nan)) != alpha
        or float(policy_identity.get("l2_regularization", np.nan)) != l2
        or not bool(policy_identity.get("fresh_state_per_price"))
        or bool(policy_identity.get("cross_price_state_shared"))
    ):
        raise RuntimeError("LinUCB replay audit因果/动作/完整target身份失配")
    if formal_identity:
        frozen = core.frozen_zone_baseline_contract(zone, "LinUCB")
        derived = core.derived_six_action_contract_identity(
            core.six_action_contracts(load_frozen_contracts())
        )
        if (
            dict(policy_identity.get("frozen_zone_configuration", {})) != frozen
            or str(policy_identity.get("frozen_zone_configuration_sha256"))
            != str(frozen["frozen_configuration_sha256"])
            or dict(frozen.get("configuration", {}))
            != {"exploration_alpha": alpha, "l2_regularization": l2}
            or str(policy_identity.get("derived_contract_sha256"))
            != str(derived["derived_contract_sha256"])
            or str(audit.get("derived_contract_sha256"))
            != str(derived["derived_contract_sha256"])
            or str(policy_identity.get("algorithm_closure_sha256"))
            != str(core.algorithm_dependency_closure()["algorithm_closure_sha256"])
        ):
            raise RuntimeError("LinUCB audit未绑定registry/派生合同/算法closure")
    return replay_identity


@dataclass(frozen=True)
class LoadedReplayPriceArtifact:
    manifest: dict[str, Any]
    cell_metrics: pd.DataFrame
    action_counts: pd.DataFrame
    paired_blocks: pd.DataFrame
    diagnostic_metrics: pd.DataFrame
    clara_state_counts: pd.DataFrame
    event_conservation: pd.DataFrame
    metric_finalize_audit: dict[str, Any]
    linucb_replay_audit: dict[str, Any]


@dataclass(frozen=True)
class LoadedFitParentArtifact:
    manifest: dict[str, Any]
    manifest_path: Path
    children: Mapping[str, LoadedFitPriceArtifact]


@dataclass(frozen=True)
class LoadedReplayParentArtifact:
    manifest: dict[str, Any]
    manifest_path: Path
    children: Mapping[str, LoadedReplayPriceArtifact]
    fit_parent: LoadedFitParentArtifact
    target_input_manifest: dict[str, Any]


def write_replay_price_child(
    parent_dir: Path,
    *,
    policy_mode: str,
    evaluation_price: core.PriceSpec,
    policy_price: core.PriceSpec,
    evaluation_zone: str,
    seed: int,
    metric_result: core.StreamingMetricResult,
    partition_records: Mapping[str, Mapping[str, Any]],
    fit_child_manifest_path: Path,
    endpoint_root_sha256: str,
    target_input_manifest_path: Path,
    target_input_manifest_sha256: str,
    expected_event_signature: Mapping[str, Any],
    execution_identity_sha256: str,
    execution_identity: Mapping[str, Any] | None = None,
    formal_identity: bool = True,
) -> dict[str, Any]:
    """Commit compact replay outputs only; event-level decisions are not accepted."""

    _reject_nonformal_official_root(
        Path(parent_dir), formal_identity=formal_identity, kind="replay"
    )
    scope_identity = core.policy_evaluation_scope(
        policy_mode=policy_mode,
        evaluation_price=evaluation_price,
        policy_price=policy_price,
        require_formal_identity=formal_identity,
    )
    metric_capability_identity = None
    if formal_identity:
        metric_capability_identity = core.verify_formal_metric_result_capability(
            metric_result, require_linucb_audit=True
        )
    if metric_result.linucb_replay_audit is None:
        raise RuntimeError("replay artifact缺少LinUCB replay audit")
    metric_finalize_audit = metric_result.audit
    linucb_replay_audit = metric_result.linucb_replay_audit
    compact_frames = {
        "cell_metrics": metric_result.cell_metrics,
        "action_counts": metric_result.action_counts,
        "paired_blocks": metric_result.paired_blocks,
        "diagnostic_metrics": metric_result.diagnostic_metrics,
        "clara_state_counts": metric_result.clara_state_counts,
        "event_conservation": metric_result.event_conservation,
    }
    _validate_execution_identity(
        execution_identity_sha256,
        execution_identity,
        formal_identity=formal_identity,
    )
    if formal_identity:
        core.assert_formal_price_subset((evaluation_price,))
        core.assert_formal_price_subset((policy_price,))
        _assert_formal_parent_path(Path(parent_dir), kind="replay", zone=evaluation_zone, seed=seed)
    fit_manifest, fit_manifest_path = _load_fit_manifest_header(fit_child_manifest_path)
    expected_fit_path = FORMAL_FIT_ROOT / f"{evaluation_zone}__seed{int(seed)}" / "children" / policy_price.price_id / "manifest.json"
    if formal_identity and fit_manifest_path.resolve() != expected_fit_path.resolve():
        raise RuntimeError("replay child拒绝非官方same-price fit manifest")
    if (
        str(fit_manifest["heldout_zone"]) != str(evaluation_zone)
        or int(fit_manifest["seed"]) != int(seed)
        or str(fit_manifest["price_id"]) != policy_price.price_id
        or str(fit_manifest["price_sha256"]) != core.price_sha256(policy_price)
        or str(fit_manifest["endpoint_root_sha256"]) != str(endpoint_root_sha256)
    ):
        raise RuntimeError("replay child与fit child/endpoint root交叉接线")
    if formal_identity:
        fit_states, _ = official_state_universe()
    else:
        fit_states = pd.read_parquet(
            fit_manifest_path.parent / FIT_FILES["clara_decisions"]
        ).loc[:, ["full_state_code", *core.STATE_FIELDS]]
    # Deep-load the same-price fit artifact before accepting any replay lineage;
    # this revalidates source provenance, endpoint root, safe CART and full policy.
    loaded_fit_artifact = load_fit_price_child(
        fit_manifest_path,
        expected_zone=evaluation_zone,
        expected_seed=seed,
        expected_price_id=policy_price.price_id,
        expected_compact_source_identity_sha256=str(
            fit_manifest["compact_source_identity_sha256"]
        ),
        expected_endpoint_root_sha256=endpoint_root_sha256,
        expected_execution_identity_sha256=str(
            fit_manifest["execution_identity_sha256"]
        ),
        expected_state_universe=fit_states,
        formal_identity=formal_identity,
    )
    _validate_partition_records(
        partition_records,
        zone=evaluation_zone,
        seed=seed,
        expected_signature=expected_event_signature,
    )
    frames = {str(key): value.copy() for key, value in compact_frames.items()}
    _validate_metric_finalize_audit(
        metric_finalize_audit,
        policy_mode=policy_mode,
        evaluation_price=evaluation_price,
        policy_price=policy_price,
        expected_signature=expected_event_signature,
        partition_records=partition_records,
        frames=frames,
        zone=evaluation_zone,
        seed=seed,
        fit_artifact=loaded_fit_artifact,
    )
    policy_replay_identity = _validate_linucb_replay_audit(
        linucb_replay_audit,
        zone=evaluation_zone,
        seed=seed,
        price=policy_price,
        event_count=int(expected_event_signature["event_count"]),
        formal_identity=formal_identity,
    )
    linucb_policy_binding = dict(metric_finalize_audit["method_policy_identities"])[
        "LinUCB_6A"
    ]
    if str(linucb_policy_binding.get("linucb_decisions_sha256")) != str(
        linucb_replay_audit.get("retained_decisions_sha256")
    ):
        raise RuntimeError("metric LinUCB动作与replay audit决策内容串线")
    replay_identity = core.replay_policy_evaluation_child_identity(
        policy_mode=policy_mode,
        evaluation_price=evaluation_price,
        policy_price=policy_price,
        evaluation_zone=evaluation_zone,
        seed=seed,
        fit_child_sha256=str(fit_manifest["fit_child_identity_sha256"]),
        endpoint_root_sha256=endpoint_root_sha256,
        policy_replay_child_identity=policy_replay_identity,
    )
    replay_sha = str(replay_identity["replay_child_identity_sha256"])
    if (
        str(replay_identity.get("evaluation_zone")) != str(evaluation_zone)
        or int(replay_identity.get("seed", -1)) != int(seed)
        or str(replay_identity.get("policy_mode")) != str(policy_mode)
        or str(replay_identity.get("evaluation_price_id"))
        != evaluation_price.price_id
        or str(replay_identity.get("evaluation_price_sha256"))
        != core.price_sha256(evaluation_price)
        or str(replay_identity.get("policy_price_id")) != policy_price.price_id
        or str(replay_identity.get("policy_price_sha256"))
        != core.price_sha256(policy_price)
        or str(replay_identity.get("fit_child_sha256")) != str(fit_manifest["fit_child_identity_sha256"])
        or str(replay_identity.get("endpoint_root_sha256")) != str(endpoint_root_sha256)
    ):
        raise RuntimeError("replay child identity same-price/zone/seed/root失配")
    _validate_replay_frames(
        frames,
        zone=evaluation_zone,
        seed=seed,
        policy_mode=policy_mode,
        evaluation_price_id=evaluation_price.price_id,
        policy_price_id=policy_price.price_id,
        expected_signature=expected_event_signature,
        partition_records=partition_records,
    )
    parent = Path(parent_dir)
    expected_target_input_path = parent / "target_input_manifest.json"
    if Path(target_input_manifest_path).resolve() != expected_target_input_path.resolve():
        raise RuntimeError("replay child target input manifest路径交叉接线")
    _, observed_target_input_sha = validate_target_input_manifest(
        Path(target_input_manifest_path),
        expected_zone=evaluation_zone,
        expected_seed=seed,
        expected_endpoint_root_sha256=endpoint_root_sha256,
        expected_execution_identity_sha256=execution_identity_sha256,
        expected_partition_records=partition_records,
        expected_combined_signature=expected_event_signature,
        formal_identity=formal_identity,
    )
    if (
        not _is_sha256(target_input_manifest_sha256)
        or observed_target_input_sha != str(target_input_manifest_sha256)
    ):
        raise RuntimeError("replay child target input manifest SHA失配")
    _assert_parent_container(
        parent, "replay_parent_manifest.json", allow_missing_seal=True
    )
    children_dir = parent / "children"
    children_dir.mkdir(parents=True, exist_ok=True)
    _assert_child_container(children_dir, require_exact=False, kind="replay")
    child_key = replay_child_key(policy_mode, evaluation_price.price_id)
    target = children_dir / child_key
    if (parent / "replay_parent_manifest.json").exists() and not target.exists():
        raise RuntimeError("replay parent已封存，禁止补写child")
    if target.exists():
        loaded = load_replay_price_child(
            target,
            expected_zone=evaluation_zone,
            expected_seed=seed,
            expected_policy_mode=policy_mode,
            expected_evaluation_price_id=evaluation_price.price_id,
            expected_policy_price_id=policy_price.price_id,
            expected_fit_child_identity_sha256=str(fit_manifest["fit_child_identity_sha256"]),
            expected_fit_child_manifest_sha256=sha256_file(fit_manifest_path),
            expected_endpoint_root_sha256=endpoint_root_sha256,
            expected_target_input_manifest_path=target_input_manifest_path,
            expected_target_input_manifest_sha256=target_input_manifest_sha256,
            expected_execution_identity_sha256=execution_identity_sha256,
            expected_event_signature=expected_event_signature,
            formal_identity=formal_identity,
        )
        if loaded.manifest["replay_child_identity_sha256"] != replay_sha:
            raise RuntimeError("既有replay child为不同科学身份，拒绝覆盖")
        return _manifest_result(target / "manifest.json", loaded.manifest, resumed=True)
    tmp = children_dir / f".{child_key}.tmp__{uuid.uuid4().hex}"
    tmp.mkdir()
    try:
        for name, frame in frames.items():
            _write_parquet(tmp / REPLAY_FILES[name], frame)
        _write_json(
            tmp / REPLAY_FILES["metric_finalize_audit"],
            dict(metric_finalize_audit),
        )
        _write_json(
            tmp / REPLAY_FILES["linucb_replay_audit"],
            dict(linucb_replay_audit),
        )
        file_records = _file_records(tmp, REPLAY_FILES, frames)
        manifest = _payload_with_sha(
            {
                "schema": REPLAY_CHILD_SCHEMA,
                "status": "PASS",
                "evaluation_zone": str(evaluation_zone),
                "seed": int(seed),
                "policy_mode": str(policy_mode),
                "evaluation_price_id": evaluation_price.price_id,
                "evaluation_price_sha256": core.price_sha256(evaluation_price),
                "policy_price_id": policy_price.price_id,
                "policy_price_sha256": core.price_sha256(policy_price),
                "policy_evaluation_scope_sha256": str(
                    scope_identity["policy_evaluation_scope_sha256"]
                ),
                "action_library_sha256": core.action_library_sha256(),
                "fit_child_identity_sha256": str(fit_manifest["fit_child_identity_sha256"]),
                "fit_child_manifest_sha256": sha256_file(fit_manifest_path),
                "endpoint_root_sha256": str(endpoint_root_sha256),
                "target_input_manifest_path": str(
                    Path(target_input_manifest_path).resolve()
                ),
                "target_input_manifest_sha256": str(
                    target_input_manifest_sha256
                ),
                "execution_identity_sha256": str(execution_identity_sha256),
                "execution_identity": (
                    None if execution_identity is None else dict(execution_identity)
                ),
                "replay_child_identity": replay_identity,
                "replay_child_identity_sha256": replay_sha,
                "policy_replay_child_identity": policy_replay_identity,
                "policy_replay_child_identity_sha256": str(
                    policy_replay_identity["replay_child_identity_sha256"]
                ),
                "metric_result_capability_sha256": str(
                    metric_capability_identity["capability_sha256"]
                    if metric_capability_identity is not None
                    else ""
                ),
                "expected_event_signature": dict(expected_event_signature),
                "partition_records": {
                    str(key): dict(value) for key, value in partition_records.items()
                },
                "files": file_records,
            }
        )
        _write_json(tmp / "manifest.json", manifest)
        staged_loaded = load_replay_price_child(
            tmp,
            expected_zone=evaluation_zone,
            expected_seed=seed,
            expected_policy_mode=policy_mode,
            expected_evaluation_price_id=evaluation_price.price_id,
            expected_policy_price_id=policy_price.price_id,
            expected_fit_child_identity_sha256=str(
                fit_manifest["fit_child_identity_sha256"]
            ),
            expected_fit_child_manifest_sha256=sha256_file(fit_manifest_path),
            expected_endpoint_root_sha256=endpoint_root_sha256,
            expected_target_input_manifest_path=target_input_manifest_path,
            expected_target_input_manifest_sha256=target_input_manifest_sha256,
            expected_execution_identity_sha256=execution_identity_sha256,
            expected_event_signature=expected_event_signature,
            formal_identity=formal_identity,
            _staged=True,
        )
        if staged_loaded.manifest["replay_child_identity_sha256"] != replay_sha:
            raise RuntimeError("staged replay child深验身份失配")
        staged_manifest_sha = sha256_file(tmp / "manifest.json")
        _validate_execution_identity(
            execution_identity_sha256,
            execution_identity,
            formal_identity=formal_identity,
        )
        _, final_target_input_sha = validate_target_input_manifest(
            Path(target_input_manifest_path),
            expected_zone=evaluation_zone,
            expected_seed=seed,
            expected_endpoint_root_sha256=endpoint_root_sha256,
            expected_execution_identity_sha256=execution_identity_sha256,
            expected_partition_records=partition_records,
            expected_combined_signature=expected_event_signature,
            formal_identity=formal_identity,
        )
        if final_target_input_sha != str(target_input_manifest_sha256):
            raise RuntimeError("target input manifest在replay child提交前漂移")
        os.replace(tmp, target)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp)
        raise
    return {
        **staged_loaded.manifest,
        "manifest_path": str(target / "manifest.json"),
        "manifest_sha256": staged_manifest_sha,
        "resumed": False,
    }


def load_replay_price_child(
    path: Path,
    *,
    expected_zone: str,
    expected_seed: int,
    expected_policy_mode: str,
    expected_evaluation_price_id: str,
    expected_policy_price_id: str,
    expected_fit_child_identity_sha256: str,
    expected_fit_child_manifest_sha256: str,
    expected_endpoint_root_sha256: str,
    expected_target_input_manifest_path: Path,
    expected_target_input_manifest_sha256: str,
    expected_execution_identity_sha256: str,
    expected_event_signature: Mapping[str, Any],
    formal_identity: bool = True,
    _staged: bool = False,
) -> LoadedReplayPriceArtifact:
    directory = Path(path).parent if Path(path).name == "manifest.json" else Path(path)
    if formal_identity:
        _assert_formal_parent_path(directory.parent.parent, kind="replay", zone=expected_zone, seed=expected_seed)
    _assert_parent_container(
        directory.parent.parent,
        "replay_parent_manifest.json",
        allow_missing_seal=True,
    )
    expected_child_key = replay_child_key(
        expected_policy_mode, expected_evaluation_price_id
    )
    _assert_child_load_location(
        directory,
        expected_child_name=expected_child_key,
        staged=bool(_staged),
    )
    manifest = _load_json(directory / "manifest.json")
    _verify_payload_sha(manifest, fields=REPLAY_MANIFEST_FIELDS, label="replay child manifest")
    evaluation_price = _price_from_id(expected_evaluation_price_id)
    policy_price = _price_from_id(expected_policy_price_id)
    scope_identity = core.policy_evaluation_scope(
        policy_mode=expected_policy_mode,
        evaluation_price=evaluation_price,
        policy_price=policy_price,
        require_formal_identity=formal_identity,
    )
    if (
        manifest.get("schema") != REPLAY_CHILD_SCHEMA
        or manifest.get("status") != "PASS"
        or str(manifest.get("evaluation_zone")) != str(expected_zone)
        or int(manifest.get("seed", -1)) != int(expected_seed)
        or (
            not bool(_staged)
            and directory.name != expected_child_key
        )
        or str(manifest.get("policy_mode")) != str(expected_policy_mode)
        or str(manifest.get("evaluation_price_id"))
        != expected_evaluation_price_id
        or str(manifest.get("evaluation_price_sha256"))
        != core.price_sha256(evaluation_price)
        or str(manifest.get("policy_price_id")) != expected_policy_price_id
        or str(manifest.get("policy_price_sha256"))
        != core.price_sha256(policy_price)
        or str(manifest.get("policy_evaluation_scope_sha256"))
        != str(scope_identity["policy_evaluation_scope_sha256"])
        or str(manifest.get("action_library_sha256")) != core.action_library_sha256()
        or str(manifest.get("fit_child_identity_sha256")) != str(expected_fit_child_identity_sha256)
        or str(manifest.get("fit_child_manifest_sha256")) != str(expected_fit_child_manifest_sha256)
        or str(manifest.get("endpoint_root_sha256")) != str(expected_endpoint_root_sha256)
        or Path(manifest.get("target_input_manifest_path", "")).resolve()
        != Path(expected_target_input_manifest_path).resolve()
        or str(manifest.get("target_input_manifest_sha256"))
        != str(expected_target_input_manifest_sha256)
        or str(manifest.get("execution_identity_sha256"))
        != str(expected_execution_identity_sha256)
        or dict(manifest.get("expected_event_signature", {})) != dict(expected_event_signature)
    ):
        raise RuntimeError("replay child manifest轴/fit/root/signature失配")
    _validate_execution_identity(
        str(manifest["execution_identity_sha256"]),
        manifest.get("execution_identity"),
        formal_identity=formal_identity,
    )
    replay_sha = _verify_embedded_identity(
        dict(manifest["replay_child_identity"]),
        "replay_child_identity_sha256",
        schema="TEST_CLARA_GEFCOM_6A_REPLAY_POLICY_EVALUATION_CHILD_IDENTITY_V1",
    )
    if replay_sha != str(manifest["replay_child_identity_sha256"]):
        raise RuntimeError("replay child embedded identity SHA失配")
    policy_replay_sha = _verify_embedded_identity(
        dict(manifest["policy_replay_child_identity"]),
        "replay_child_identity_sha256",
        schema="TEST_CLARA_GEFCOM_6A_REPLAY_PRICE_CHILD_IDENTITY_V1",
    )
    if policy_replay_sha != str(manifest["policy_replay_child_identity_sha256"]):
        raise RuntimeError("policy replay child embedded identity SHA失配")
    _verify_files(directory, manifest["files"], REPLAY_FILES)
    _validate_partition_records(
        dict(manifest["partition_records"]),
        zone=expected_zone,
        seed=expected_seed,
        expected_signature=expected_event_signature,
    )
    target_path = Path(expected_target_input_manifest_path)
    expected_parent_target = directory.parent.parent / "target_input_manifest.json"
    if target_path.resolve() != expected_parent_target.resolve():
        raise RuntimeError("replay child target input manifest不在同一parent")
    _, observed_target_sha = validate_target_input_manifest(
        target_path,
        expected_zone=expected_zone,
        expected_seed=expected_seed,
        expected_endpoint_root_sha256=expected_endpoint_root_sha256,
        expected_execution_identity_sha256=expected_execution_identity_sha256,
        expected_partition_records=dict(manifest["partition_records"]),
        expected_combined_signature=expected_event_signature,
        formal_identity=formal_identity,
    )
    if observed_target_sha != str(expected_target_input_manifest_sha256):
        raise RuntimeError("replay child target input manifest现场SHA失配")
    frames = {
        key: pd.read_parquet(directory / relative)
        for key, relative in REPLAY_FILES.items()
        if relative.endswith(".parquet")
    }
    for key, frame in frames.items():
        if len(frame) != int(manifest["files"][key]["row_count"]):
            raise RuntimeError(f"replay child parquet行数失配: {key}")
    _validate_replay_frames(
        frames,
        zone=expected_zone,
        seed=expected_seed,
        policy_mode=expected_policy_mode,
        evaluation_price_id=expected_evaluation_price_id,
        policy_price_id=expected_policy_price_id,
        expected_signature=expected_event_signature,
        partition_records=dict(manifest["partition_records"]),
    )
    metric_finalize_audit = _load_json(
        directory / REPLAY_FILES["metric_finalize_audit"]
    )
    linucb_replay_audit = _load_json(
        directory / REPLAY_FILES["linucb_replay_audit"]
    )
    loaded_fit_artifact = None
    if formal_identity:
        fit_manifest_path = (
            FORMAL_FIT_ROOT
            / f"{expected_zone}__seed{int(expected_seed)}"
            / "children"
            / expected_policy_price_id
            / "manifest.json"
        )
        if (
            not fit_manifest_path.is_file()
            or sha256_file(fit_manifest_path)
            != str(expected_fit_child_manifest_sha256)
        ):
            raise RuntimeError("replay child same-price fit manifest现场SHA失配")
        fit_header, _ = _load_fit_manifest_header(fit_manifest_path)
        fit_states, _ = official_state_universe()
        loaded_fit_artifact = load_fit_price_child(
            fit_manifest_path,
            expected_zone=expected_zone,
            expected_seed=expected_seed,
            expected_price_id=expected_policy_price_id,
            expected_compact_source_identity_sha256=str(
                fit_header["compact_source_identity_sha256"]
            ),
            expected_endpoint_root_sha256=expected_endpoint_root_sha256,
            expected_execution_identity_sha256=expected_execution_identity_sha256,
            expected_state_universe=fit_states,
            formal_identity=True,
        )
    _validate_metric_finalize_audit(
        metric_finalize_audit,
        policy_mode=expected_policy_mode,
        evaluation_price=evaluation_price,
        policy_price=policy_price,
        expected_signature=expected_event_signature,
        partition_records=dict(manifest["partition_records"]),
        frames=frames,
        zone=expected_zone,
        seed=expected_seed,
        fit_artifact=loaded_fit_artifact,
    )
    observed_policy_replay_identity = _validate_linucb_replay_audit(
        linucb_replay_audit,
        zone=expected_zone,
        seed=expected_seed,
        price=policy_price,
        event_count=int(expected_event_signature["event_count"]),
        formal_identity=formal_identity,
    )
    linucb_policy_binding = dict(metric_finalize_audit["method_policy_identities"])[
        "LinUCB_6A"
    ]
    if str(linucb_policy_binding.get("linucb_decisions_sha256")) != str(
        linucb_replay_audit.get("retained_decisions_sha256")
    ):
        raise RuntimeError("metric LinUCB动作与replay audit决策内容串线")
    reconstructed_replay_identity = core.replay_policy_evaluation_child_identity(
        policy_mode=expected_policy_mode,
        evaluation_price=evaluation_price,
        policy_price=policy_price,
        evaluation_zone=expected_zone,
        seed=expected_seed,
        fit_child_sha256=expected_fit_child_identity_sha256,
        endpoint_root_sha256=expected_endpoint_root_sha256,
        policy_replay_child_identity=observed_policy_replay_identity,
    )
    if (
        reconstructed_replay_identity != dict(manifest["replay_child_identity"])
        or str(observed_policy_replay_identity["replay_child_identity_sha256"])
        != policy_replay_sha
    ):
        raise RuntimeError("LinUCB audit与replay manifest identity失配")
    reconstructed_metric = core.StreamingMetricResult(
        cell_metrics=frames["cell_metrics"],
        action_counts=frames["action_counts"],
        paired_blocks=frames["paired_blocks"],
        diagnostic_metrics=frames["diagnostic_metrics"],
        clara_state_counts=frames["clara_state_counts"],
        event_conservation=frames["event_conservation"],
        audit=metric_finalize_audit,
        linucb_replay_audit=linucb_replay_audit,
    )
    reconstructed_metric_identity = core._metric_result_capability_identity(
        reconstructed_metric
    )
    if str(manifest.get("metric_result_capability_sha256")) != str(
        reconstructed_metric_identity["capability_sha256"]
    ):
        raise RuntimeError("replay metric capability SHA现场重算失配")
    return LoadedReplayPriceArtifact(
        manifest=manifest,
        cell_metrics=frames["cell_metrics"],
        action_counts=frames["action_counts"],
        paired_blocks=frames["paired_blocks"],
        diagnostic_metrics=frames["diagnostic_metrics"],
        clara_state_counts=frames["clara_state_counts"],
        event_conservation=frames["event_conservation"],
        metric_finalize_audit=metric_finalize_audit,
        linucb_replay_audit=linucb_replay_audit,
    )


def _load_fit_parent_deep(
    path: Path,
    *,
    expected_endpoint_root_sha256: str,
    expected_execution_identity_sha256: str | None = None,
    expected_zone: str | None = None,
    expected_seed: int | None = None,
    formal_identity: bool,
) -> LoadedFitParentArtifact:
    manifest_path = (
        Path(path)
        if Path(path).name == "fit_parent_manifest.json"
        else Path(path) / "fit_parent_manifest.json"
    )
    fit_parent = _load_json(manifest_path)
    _verify_payload_sha(fit_parent, fields=FIT_PARENT_FIELDS, label="fit parent manifest")
    parent_identity_sha = _verify_embedded_identity(
        dict(fit_parent["fit_parent_identity"]),
        "fit_parent_identity_sha256",
        schema="TEST_CLARA_GEFCOM_6A_FIT_PARENT_IDENTITY_V1",
    )
    zone = str(fit_parent.get("heldout_zone"))
    seed = int(fit_parent.get("seed", -1))
    if (
        fit_parent.get("schema") != FIT_PARENT_SCHEMA
        or fit_parent.get("status") != "PASS"
        or parent_identity_sha != str(fit_parent.get("fit_parent_identity_sha256"))
        or str(fit_parent.get("action_library_sha256"))
        != core.action_library_sha256()
        or str(fit_parent.get("endpoint_root_sha256"))
        != str(expected_endpoint_root_sha256)
        or int(fit_parent.get("price_child_count", -1)) != 5
        or set(fit_parent.get("price_child_identity_sha256", {}))
        != set(_formal_price_ids())
        or set(fit_parent.get("price_child_manifest_sha256", {}))
        != set(_formal_price_ids())
        or (
            expected_zone is not None
            and zone != str(expected_zone)
        )
        or (
            expected_seed is not None
            and seed != int(expected_seed)
        )
        or (
            expected_execution_identity_sha256 is not None
            and str(fit_parent.get("execution_identity_sha256"))
            != str(expected_execution_identity_sha256)
        )
    ):
        raise RuntimeError("fit parent深层身份/五价/root失配")
    _validate_execution_identity(
        str(fit_parent["execution_identity_sha256"]),
        fit_parent.get("execution_identity"),
        formal_identity=formal_identity,
    )
    parent = manifest_path.parent
    if formal_identity:
        expected_path = (
            FORMAL_FIT_ROOT / f"{zone}__seed{seed}" / "fit_parent_manifest.json"
        )
        if manifest_path.resolve() != expected_path.resolve():
            raise RuntimeError("fit parent拒绝非官方路径")
        states, _ = official_state_universe()
    else:
        first = parent / "children" / _formal_price_ids()[0] / FIT_FILES["clara_decisions"]
        states = pd.read_parquet(first).loc[:, ["full_state_code", *core.STATE_FIELDS]]
    _assert_parent_container(parent, "fit_parent_manifest.json", allow_missing_seal=False)
    _assert_child_container(parent / "children", require_exact=True)
    loaded_children: dict[str, LoadedFitPriceArtifact] = {}
    for price_id in _formal_price_ids():
        child_path = parent / "children" / price_id / "manifest.json"
        child_manifest = _load_json(child_path)
        loaded = load_fit_price_child(
            child_path,
            expected_zone=zone,
            expected_seed=seed,
            expected_price_id=price_id,
            expected_compact_source_identity_sha256=str(
                fit_parent["compact_source_identity_sha256"]
            ),
            expected_endpoint_root_sha256=expected_endpoint_root_sha256,
            expected_execution_identity_sha256=str(
                fit_parent["execution_identity_sha256"]
            ),
            expected_state_universe=states,
            formal_identity=formal_identity,
        )
        if (
            str(loaded.manifest["fit_child_identity_sha256"])
            != str(fit_parent["price_child_identity_sha256"][price_id])
            or sha256_file(child_path)
            != str(fit_parent["price_child_manifest_sha256"][price_id])
        ):
            raise RuntimeError("fit parent与当前price child SHA不闭合")
        loaded_children[price_id] = loaded
    return LoadedFitParentArtifact(
        manifest=fit_parent,
        manifest_path=manifest_path,
        children=loaded_children,
    )


def load_fit_parent(
    path: Path,
    *,
    expected_zone: str,
    expected_seed: int,
    expected_endpoint_root_sha256: str,
    expected_execution_identity_sha256: str,
    formal_identity: bool = True,
) -> LoadedFitParentArtifact:
    """Deep-load an immutable five-price fit parent and every policy payload."""

    return _load_fit_parent_deep(
        path,
        expected_endpoint_root_sha256=expected_endpoint_root_sha256,
        expected_execution_identity_sha256=expected_execution_identity_sha256,
        expected_zone=expected_zone,
        expected_seed=expected_seed,
        formal_identity=formal_identity,
    )


def _exact_frame_without_scope(
    frame: pd.DataFrame, *, drop_fields: Sequence[str]
) -> pd.DataFrame:
    result = frame.drop(columns=list(drop_fields)).copy()
    sort_fields = list(result.columns)
    return result.sort_values(sort_fields, kind="mergesort").reset_index(drop=True)


def _assert_exact_frame(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    drop_fields: Sequence[str],
    label: str,
) -> None:
    try:
        pd.testing.assert_frame_equal(
            _exact_frame_without_scope(left, drop_fields=drop_fields),
            _exact_frame_without_scope(right, drop_fields=drop_fields),
            check_exact=True,
            check_dtype=True,
            check_like=False,
        )
    except AssertionError as exc:
        raise RuntimeError(f"{label} exact不变量失配") from exc


def _validate_dual_mode_children(
    children: Mapping[str, LoadedReplayPriceArtifact],
) -> None:
    expected = set(formal_replay_child_keys())
    if set(children) != expected:
        raise RuntimeError("双policy mode replay child矩阵不闭合")
    frame_fields = (
        "cell_metrics",
        "action_counts",
        "paired_blocks",
        "diagnostic_metrics",
        "clara_state_counts",
        "event_conservation",
    )
    main_reselected = children[
        replay_child_key(core.POLICY_MODE_RESELECTED, core.FORMAL_MAIN_PRICE_ID)
    ]
    main_fixed = children[
        replay_child_key(core.POLICY_MODE_FIXED_MAIN, core.FORMAL_MAIN_PRICE_ID)
    ]
    for field in frame_fields:
        _assert_exact_frame(
            getattr(main_reselected, field),
            getattr(main_fixed, field),
            drop_fields=("policy_mode",),
            label=f"R05双mode/{field}",
        )

    deterministic_methods = set(core.FORMAL_DETERMINISTIC_METHOD_ACTIONS)
    for price_id in _formal_price_ids():
        reselected = children[
            replay_child_key(core.POLICY_MODE_RESELECTED, price_id)
        ]
        fixed = children[replay_child_key(core.POLICY_MODE_FIXED_MAIN, price_id)]
        for field in (
            "cell_metrics",
            "action_counts",
            "paired_blocks",
            "diagnostic_metrics",
            "event_conservation",
        ):
            left = getattr(reselected, field)
            right = getattr(fixed, field)
            left = left[left["method"].astype(str).isin(deterministic_methods)]
            right = right[right["method"].astype(str).isin(deterministic_methods)]
            _assert_exact_frame(
                left,
                right,
                drop_fields=core.POLICY_SCOPE_FIELDS,
                label=f"确定性方法双mode/{price_id}/{field}",
            )

    fixed_policy_bindings: dict[str, Any] | None = None
    fixed_action_counts: pd.DataFrame | None = None
    for price_id in _formal_price_ids():
        child = children[replay_child_key(core.POLICY_MODE_FIXED_MAIN, price_id)]
        policies = dict(child.metric_finalize_audit["method_policy_identities"])
        if fixed_policy_bindings is None:
            fixed_policy_bindings = policies
        elif policies != fixed_policy_bindings:
            raise RuntimeError("fixed mode五评估价policy binding漂移")
        if fixed_action_counts is None:
            fixed_action_counts = child.action_counts
        else:
            _assert_exact_frame(
                fixed_action_counts,
                child.action_counts,
                drop_fields=core.POLICY_SCOPE_FIELDS,
                label="fixed mode五评估价动作计数",
            )


def seal_replay_parent(
    parent_dir: Path,
    *,
    fit_parent_manifest_path: Path,
    child_manifest_paths: Mapping[str, Path],
    endpoint_root_sha256: str,
    target_input_manifest_path: Path,
    target_input_manifest_sha256: str,
    expected_event_signature_by_evaluation_price: Mapping[str, Mapping[str, Any]],
    execution_identity_sha256: str,
    execution_identity: Mapping[str, Any] | None = None,
    formal_identity: bool = True,
) -> dict[str, Any]:
    """Seal replay parent only after the exact two-mode x five-price children validate."""

    _reject_nonformal_official_root(
        Path(parent_dir), formal_identity=formal_identity, kind="replay"
    )

    loaded_fit_parent = _load_fit_parent_deep(
        Path(fit_parent_manifest_path),
        expected_endpoint_root_sha256=endpoint_root_sha256,
        expected_execution_identity_sha256=execution_identity_sha256,
        formal_identity=formal_identity,
    )
    fit_parent = loaded_fit_parent.manifest
    fit_parent_path = loaded_fit_parent.manifest_path
    zone = str(fit_parent["heldout_zone"])
    seed = int(fit_parent["seed"])
    parent = Path(parent_dir)
    _validate_execution_identity(
        execution_identity_sha256,
        execution_identity,
        formal_identity=formal_identity,
    )
    if formal_identity:
        _assert_formal_parent_path(parent, kind="replay", zone=zone, seed=seed)
        expected_fit_parent_path = FORMAL_FIT_ROOT / f"{zone}__seed{seed}" / "fit_parent_manifest.json"
        if fit_parent_path.resolve() != expected_fit_parent_path.resolve():
            raise RuntimeError("replay parent拒绝非官方fit parent")
    price_ids = _formal_price_ids()
    child_keys = formal_replay_child_keys()
    _assert_parent_container(parent, "replay_parent_manifest.json", allow_missing_seal=True)
    _assert_child_container(parent / "children", require_exact=True, kind="replay")
    if (
        tuple(child_manifest_paths) != child_keys
        or tuple(expected_event_signature_by_evaluation_price) != price_ids
    ):
        raise RuntimeError(
            "replay parent必须按冻结顺序提供exact双mode×五价child/signature"
        )
    expected_target_path = parent / "target_input_manifest.json"
    if Path(target_input_manifest_path).resolve() != expected_target_path.resolve():
        raise RuntimeError("replay parent target input manifest路径交叉接线")
    first_signature = dict(
        expected_event_signature_by_evaluation_price[price_ids[0]]
    )
    if any(
        dict(expected_event_signature_by_evaluation_price[price_id])
        != first_signature
        for price_id in price_ids[1:]
    ):
        raise RuntimeError("replay parent五价必须共用同一target成员集")
    target_manifest, observed_target_sha = validate_target_input_manifest(
        Path(target_input_manifest_path),
        expected_zone=zone,
        expected_seed=seed,
        expected_endpoint_root_sha256=endpoint_root_sha256,
        expected_execution_identity_sha256=execution_identity_sha256,
        expected_partition_records=dict(
            _load_json(Path(child_manifest_paths[child_keys[0]]))["partition_records"]
        ),
        expected_combined_signature=first_signature,
        formal_identity=formal_identity,
    )
    if (
        not _is_sha256(target_input_manifest_sha256)
        or observed_target_sha != str(target_input_manifest_sha256)
    ):
        raise RuntimeError("replay parent target input manifest SHA失配")
    child_manifest_shas: dict[str, str] = {}
    child_identity_shas: dict[str, str] = {}
    loaded_children: dict[str, LoadedReplayPriceArtifact] = {}
    for child_key in child_keys:
        policy_mode, evaluation_price_id = child_key.rsplit("__", 1)
        policy_price_id = (
            evaluation_price_id
            if policy_mode == core.POLICY_MODE_RESELECTED
            else core.FORMAL_MAIN_PRICE_ID
        )
        path = Path(child_manifest_paths[child_key])
        expected_path = parent / "children" / child_key / "manifest.json"
        if path.resolve() != expected_path.resolve():
            raise RuntimeError("replay parent child路径交叉接线")
        fit_child_sha = str(
            fit_parent["price_child_identity_sha256"][policy_price_id]
        )
        fit_child_manifest_sha = str(
            fit_parent["price_child_manifest_sha256"][policy_price_id]
        )
        loaded = load_replay_price_child(
            path,
            expected_zone=zone,
            expected_seed=seed,
            expected_policy_mode=policy_mode,
            expected_evaluation_price_id=evaluation_price_id,
            expected_policy_price_id=policy_price_id,
            expected_fit_child_identity_sha256=fit_child_sha,
            expected_fit_child_manifest_sha256=fit_child_manifest_sha,
            expected_endpoint_root_sha256=endpoint_root_sha256,
            expected_target_input_manifest_path=target_input_manifest_path,
            expected_target_input_manifest_sha256=target_input_manifest_sha256,
            expected_execution_identity_sha256=execution_identity_sha256,
            expected_event_signature=expected_event_signature_by_evaluation_price[
                evaluation_price_id
            ],
            formal_identity=formal_identity,
        )
        child_identity_shas[child_key] = str(
            loaded.manifest["replay_child_identity_sha256"]
        )
        child_manifest_shas[child_key] = sha256_file(path)
        loaded_children[child_key] = loaded
    _validate_dual_mode_children(loaded_children)
    manifest = _payload_with_sha(
        {
            "schema": REPLAY_PARENT_SCHEMA,
            "status": "PASS",
            "evaluation_zone": zone,
            "seed": seed,
            "action_library_sha256": core.action_library_sha256(),
            "fit_parent_manifest_sha256": sha256_file(fit_parent_path),
            "fit_parent_identity_sha256": str(fit_parent["fit_parent_identity_sha256"]),
            "endpoint_root_sha256": str(endpoint_root_sha256),
            "target_input_manifest_path": str(
                Path(target_input_manifest_path).resolve()
            ),
            "target_input_manifest_sha256": str(
                target_input_manifest_sha256
            ),
            "execution_identity_sha256": str(execution_identity_sha256),
            "execution_identity": (
                None if execution_identity is None else dict(execution_identity)
            ),
            "price_child_identity_sha256": child_identity_shas,
            "price_child_manifest_sha256": child_manifest_shas,
            "price_child_count": 10,
        }
    )
    manifest_path = parent / "replay_parent_manifest.json"
    _validate_execution_identity(
        execution_identity_sha256,
        execution_identity,
        formal_identity=formal_identity,
    )
    _, final_target_sha = validate_target_input_manifest(
        Path(target_input_manifest_path),
        expected_zone=zone,
        expected_seed=seed,
        expected_endpoint_root_sha256=endpoint_root_sha256,
        expected_execution_identity_sha256=execution_identity_sha256,
        expected_partition_records=dict(target_manifest["partition_records"]),
        expected_combined_signature=first_signature,
        formal_identity=formal_identity,
    )
    if final_target_sha != str(target_input_manifest_sha256):
        raise RuntimeError("target input manifest在replay parent seal前漂移")
    resumed = _commit_manifest(manifest_path, manifest)
    _assert_parent_container(parent, "replay_parent_manifest.json", allow_missing_seal=False)
    return _manifest_result(manifest_path, manifest, resumed=resumed)


def load_replay_parent(
    path: Path,
    *,
    expected_zone: str,
    expected_seed: int,
    expected_fit_parent_manifest_path: Path,
    expected_endpoint_root_sha256: str,
    expected_target_input_manifest_path: Path,
    expected_target_input_manifest_sha256: str,
    expected_execution_identity_sha256: str,
    formal_identity: bool = True,
) -> LoadedReplayParentArtifact:
    """Deep-load a replay parent, lineage and all ten mode x evaluation children."""

    manifest_path = (
        Path(path)
        if Path(path).name == "replay_parent_manifest.json"
        else Path(path) / "replay_parent_manifest.json"
    )
    parent = manifest_path.parent
    manifest = _load_json(manifest_path)
    _verify_payload_sha(
        manifest, fields=REPLAY_PARENT_FIELDS, label="replay parent manifest"
    )
    if formal_identity:
        _assert_formal_parent_path(
            parent, kind="replay", zone=expected_zone, seed=expected_seed
        )
    _assert_parent_container(
        parent, "replay_parent_manifest.json", allow_missing_seal=False
    )
    _assert_child_container(parent / "children", require_exact=True, kind="replay")
    if (
        manifest.get("schema") != REPLAY_PARENT_SCHEMA
        or manifest.get("status") != "PASS"
        or str(manifest.get("evaluation_zone")) != str(expected_zone)
        or int(manifest.get("seed", -1)) != int(expected_seed)
        or str(manifest.get("action_library_sha256"))
        != core.action_library_sha256()
        or str(manifest.get("endpoint_root_sha256"))
        != str(expected_endpoint_root_sha256)
        or str(manifest.get("execution_identity_sha256"))
        != str(expected_execution_identity_sha256)
        or Path(manifest.get("target_input_manifest_path", "")).resolve()
        != Path(expected_target_input_manifest_path).resolve()
        or str(manifest.get("target_input_manifest_sha256"))
        != str(expected_target_input_manifest_sha256)
        or int(manifest.get("price_child_count", -1)) != 10
        or set(manifest.get("price_child_identity_sha256", {}))
        != set(formal_replay_child_keys())
        or set(manifest.get("price_child_manifest_sha256", {}))
        != set(formal_replay_child_keys())
    ):
        raise RuntimeError("replay parent轴/root/execution/target/五价身份失配")
    _validate_execution_identity(
        expected_execution_identity_sha256,
        manifest.get("execution_identity"),
        formal_identity=formal_identity,
    )
    fit_parent_path = Path(expected_fit_parent_manifest_path)
    if (
        not fit_parent_path.is_file()
        or sha256_file(fit_parent_path)
        != str(manifest.get("fit_parent_manifest_sha256"))
    ):
        raise RuntimeError("replay parent与fit parent manifest SHA失配")
    loaded_fit = load_fit_parent(
        fit_parent_path,
        expected_zone=expected_zone,
        expected_seed=expected_seed,
        expected_endpoint_root_sha256=expected_endpoint_root_sha256,
        expected_execution_identity_sha256=expected_execution_identity_sha256,
        formal_identity=formal_identity,
    )
    if str(loaded_fit.manifest["fit_parent_identity_sha256"]) != str(
        manifest.get("fit_parent_identity_sha256")
    ):
        raise RuntimeError("replay parent与fit parent identity交叉接线")
    target_path = Path(expected_target_input_manifest_path)
    if target_path.resolve() != (parent / "target_input_manifest.json").resolve():
        raise RuntimeError("replay parent target input不在官方容器")
    target_header = _load_json(target_path)
    target_manifest, observed_target_sha = validate_target_input_manifest(
        target_path,
        expected_zone=expected_zone,
        expected_seed=expected_seed,
        expected_endpoint_root_sha256=expected_endpoint_root_sha256,
        expected_execution_identity_sha256=expected_execution_identity_sha256,
        expected_partition_records=dict(target_header.get("partition_records", {})),
        expected_combined_signature=dict(
            target_header.get("combined_event_signature", {})
        ),
        formal_identity=formal_identity,
    )
    if observed_target_sha != str(expected_target_input_manifest_sha256):
        raise RuntimeError("replay parent target input现场SHA失配")
    combined_signature = dict(target_manifest["combined_event_signature"])
    children: dict[str, LoadedReplayPriceArtifact] = {}
    for child_key in formal_replay_child_keys():
        policy_mode, evaluation_price_id = child_key.rsplit("__", 1)
        policy_price_id = (
            evaluation_price_id
            if policy_mode == core.POLICY_MODE_RESELECTED
            else core.FORMAL_MAIN_PRICE_ID
        )
        child_path = parent / "children" / child_key / "manifest.json"
        fit_child = loaded_fit.children[policy_price_id].manifest
        loaded_child = load_replay_price_child(
            child_path,
            expected_zone=expected_zone,
            expected_seed=expected_seed,
            expected_policy_mode=policy_mode,
            expected_evaluation_price_id=evaluation_price_id,
            expected_policy_price_id=policy_price_id,
            expected_fit_child_identity_sha256=str(
                fit_child["fit_child_identity_sha256"]
            ),
            expected_fit_child_manifest_sha256=sha256_file(
                loaded_fit.manifest_path.parent
                / "children"
                / policy_price_id
                / "manifest.json"
            ),
            expected_endpoint_root_sha256=expected_endpoint_root_sha256,
            expected_target_input_manifest_path=target_path,
            expected_target_input_manifest_sha256=expected_target_input_manifest_sha256,
            expected_execution_identity_sha256=expected_execution_identity_sha256,
            expected_event_signature=combined_signature,
            formal_identity=formal_identity,
        )
        if (
            sha256_file(child_path)
            != str(manifest["price_child_manifest_sha256"][child_key])
            or str(loaded_child.manifest["replay_child_identity_sha256"])
            != str(manifest["price_child_identity_sha256"][child_key])
        ):
            raise RuntimeError("replay parent与当前price child SHA不闭合")
        children[child_key] = loaded_child
    _validate_dual_mode_children(children)
    return LoadedReplayParentArtifact(
        manifest=manifest,
        manifest_path=manifest_path,
        children=children,
        fit_parent=loaded_fit,
        target_input_manifest=target_manifest,
    )


__all__ = [
    "FORMAL_SELECTOR_ROOT",
    "FORMAL_FIT_ROOT",
    "FORMAL_REPLAY_ROOT",
    "FORMAL_AGGREGATE_ROOT",
    "FORMAL_QA_ROOT",
    "LoadedFitPriceArtifact",
    "LoadedReplayPriceArtifact",
    "LoadedFitParentArtifact",
    "LoadedReplayParentArtifact",
    "canonical_json_bytes",
    "canonical_sha256",
    "sha256_file",
    "replay_child_key",
    "formal_replay_child_keys",
    "write_fit_price_child",
    "load_fit_price_child",
    "load_fit_parent",
    "seal_fit_parent",
    "write_replay_price_child",
    "load_replay_price_child",
    "load_replay_parent",
    "seal_replay_parent",
]
