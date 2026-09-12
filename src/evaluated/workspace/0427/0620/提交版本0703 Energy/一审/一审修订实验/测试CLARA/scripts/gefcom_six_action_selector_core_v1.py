"""Price-conditioned six-action selector core for the five-horizon GEFCom run.

The module is intentionally independent from orchestration and artifact writing.  A
``heldout-zone x seed`` fit scans each retained source stream once, reduces it to
price-independent state/action sufficient statistics, and then derives all five
price policies without rereading event facts.  The implementation reuses the sealed
GEFCom landscape accelerator, the V4 adaptive-support implementation, the extended
CART adapter types, and the strict delayed-feedback LinUCB accelerator.

This is a *new* five-horizon/six-action identity.  It does not claim bitwise identity
with the historical 24-horizon/four-action V4 decisions.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import sys
import threading
from collections.abc import Iterator, Mapping as ABCMapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


TEST_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = TEST_ROOT.parent
AUTH_CODE = REVISION_ROOT / "_权威代码" / "code"
S07_SCRIPTS = REVISION_ROOT / "07_GEFCom完整主实验" / "scripts"
S09_SCRIPTS = REVISION_ROOT / "09_商业场站外部验证" / "scripts"
PRICE_SCRIPTS = (
    REVISION_ROOT / "08_敏感性分析与消融" / "01_价格敏感性" / "scripts"
)
FROZEN_REGISTRY_PATH = (
    REVISION_ROOT
    / "06_基线实现与训练区调参"
    / "results_verified"
    / "s07_execution_registry_v2"
    / "s07_baseline_execution_registry.parquet"
)
FROZEN_REGISTRY_SHA256 = (
    "26d554902c94e46e30bf7afa42aa75cad0f2f2a5a4353955b90b75acf5bff0d5"
)
FOUR_VERSION_CONFIG_PATH = TEST_ROOT / "configs" / "test_clara_four_versions_v1.json"
FOUR_VERSION_CONFIG_SHA256 = (
    "29ec03b10940ea4e53632802a40bdede094a12cd66582703bba29fcb69e71572"
)
GEFCOM_SIX_ACTION_CONFIG_PATH = (
    TEST_ROOT / "configs" / "test_clara_gefcom_six_action_full_v1.json"
)
FORMAL_S03_ROOT = (
    REVISION_ROOT / "03_基础预测与候选区间重建" / "results_raw" / "full_rebuild_v1"
)
FORMAL_S06_ROOT = (
    REVISION_ROOT
    / "06_基线实现与训练区调参"
    / "results_raw"
    / "nested_source_selection_v1"
)
FORMAL_ENDPOINT_ROOT = (
    TEST_ROOT / "results_raw" / "gefcom_six_action_price_full_v1" / "endpoint_units"
)
FORMAL_ENDPOINT_ROOT_MANIFEST = (
    FORMAL_ENDPOINT_ROOT
    / "root_manifest__B_UNIFIED_FIVE_HORIZON_NEW_IDENTITY.json"
)
for directory in (AUTH_CODE, S07_SCRIPTS, S09_SCRIPTS, PRICE_SCRIPTS, Path(__file__).parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from baseline_compact_training import (  # noqa: E402
    COMPACT_STATE_FIELDS,
    _capped_classification_expansion,
    _encoded_compact_states,
)
from baseline_training import validate_frozen_selector_config  # noqa: E402
from clara_event_contract import FrozenContracts, load_frozen_contracts  # noqa: E402
import s07_decision_cache_accelerator as landscape_accelerator  # noqa: E402
import run_four_clara_versions as adaptive_v4  # noqa: E402
from extended_selector_adapters import (  # noqa: E402
    ExtendedCartSelector,
    predict_cart_local_actions,
    run_linucb_local_actions,
)


SCHEMA = "TEST_CLARA_GEFCOM_6A_SELECTOR_CORE_V1"
SIX_ACTIONS = (
    "Static",
    "ACI",
    "AgACI",
    "EnbPI_RH",
    "TunedSingleConformal",
    "EqualEndpointEnsemble",
)
FORMAL_HORIZONS = (1, 3, 6, 12, 24)
FORMAL_PREDICTORS = ("Ridge", "GBR", "MLP", "QRLSTM")
FORMAL_ZONES = tuple(f"zone{index}" for index in range(1, 11))
FORMAL_COVERAGES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
FORMAL_METHODS = (
    "CLARA_6A",
    "CART_6A",
    "LinUCB_6A",
    "TunedSingleConformal",
    "EqualEndpointEnsemble",
    "FixedStatic",
    "FixedACI",
    "FixedAgACI",
    "FixedEnbPI_RH",
)
FORMAL_DETERMINISTIC_METHOD_ACTIONS = {
    "TunedSingleConformal": "TunedSingleConformal",
    "EqualEndpointEnsemble": "EqualEndpointEnsemble",
    "FixedStatic": "Static",
    "FixedACI": "ACI",
    "FixedAgACI": "AgACI",
    "FixedEnbPI_RH": "EnbPI_RH",
}
POLICY_MODE_RESELECTED = "RESELECT_EACH_RATIO"
POLICY_MODE_FIXED_MAIN = "FIXED_MAIN_RATIO_R05P6179775281"
FORMAL_POLICY_MODES = (POLICY_MODE_RESELECTED, POLICY_MODE_FIXED_MAIN)
FORMAL_MAIN_PRICE_ID = "R05P6179775281"
POLICY_SCOPE_FIELDS = (
    "policy_mode",
    "evaluation_price_id",
    "policy_price_id",
)
FORMAL_PRICE_ROWS = (
    ("R01", 1.0, (3.56, 3.56, 3.56, 3.56), False),
    ("R02", 2.0, (3.56, 3.56, 7.12, 7.12), False),
    ("R05P6179775281", 5.6179775281, (3.56, 3.56, 20.0, 20.0), True),
    ("R10", 10.0, (3.56, 3.56, 35.6, 35.6), False),
    ("R20", 20.0, (3.56, 3.56, 71.2, 71.2), False),
)
STATE_FIELDS = tuple(COMPACT_STATE_FIELDS)
COMPONENT_FIELDS = ("capacity_exposure", "miss_exposure")
_ACTION_PATCH_LOCK = threading.RLock()
_PATH_SHA_CACHE: dict[str, str] = {}
_FORMAL_FIT_RESULT_CAPABILITY = object()
_FORMAL_METRIC_RESULT_CAPABILITY = object()
_FORMAL_LINUCB_RESULT_CAPABILITY = object()
_FORMAL_CLARA_PREPARATION_CAPABILITY = object()
_FORMAL_CLARA_EVENT_DIAGNOSTIC_CAPABILITY = object()
_FORMAL_CART_EVENT_ACTION_CAPABILITY = object()
_CAUSAL_LOADER_CAPABILITY = object()


ALGORITHM_DEPENDENCY_PATHS = (
    Path(__file__).resolve(),
    AUTH_CODE / "baseline_common.py",
    AUTH_CODE / "baseline_compact_training.py",
    AUTH_CODE / "baseline_training.py",
    AUTH_CODE / "baseline_conformal.py",
    AUTH_CODE / "clara_errf.py",
    AUTH_CODE / "clara_event_contract.py",
    AUTH_CODE / "source_tuning_facts.py",
    S07_SCRIPTS / "s07_decision_cache_accelerator.py",
    Path(__file__).resolve().parent / "run_four_clara_versions.py",
    PRICE_SCRIPTS / "build_price_decision_caches.py",
    PRICE_SCRIPTS / "price_sensitivity_common.py",
    Path(__file__).resolve().parent / "extended_selector_adapters.py",
    S09_SCRIPTS / "s09_linucb_batch_accelerator.py",
    S09_SCRIPTS / "s09_minimum_external_core.py",
)


def _canonical_digest(payload: Any) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _runner_newline_canonical_digest(payload: Any) -> str:
    raw = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_path_cached(path: Path) -> str:
    key = str(Path(path).resolve())
    if key not in _PATH_SHA_CACHE:
        _PATH_SHA_CACHE[key] = _sha256_path(Path(path))
    return _PATH_SHA_CACHE[key]


def _ndarray_content_identity(values: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.tobytes(order="C"))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "data_sha256": digest.hexdigest(),
    }


def _structured_ndarray_named_field_identity(values: np.ndarray) -> dict[str, Any]:
    """Hash structured-array semantics while excluding unnamed padding bytes.

    sklearn's tree node dtype contains implementation padding whose bytes are not
    initialized deterministically when a tree is reconstructed.  The padding has
    no predictive meaning, so bind the complete named-field layout and each named
    field's values instead of hashing the raw record buffer.
    """

    array = np.asarray(values)
    if array.dtype.names is None:
        raise ValueError("structured ndarray identity要求命名字段dtype")
    fields: list[dict[str, Any]] = []
    for name in array.dtype.names:
        field_info = array.dtype.fields[name]
        if field_info is None:
            raise RuntimeError(f"structured ndarray字段描述缺失: {name}")
        field_dtype = np.dtype(field_info[0])
        fields.append(
            {
                "name": str(name),
                "dtype": field_dtype.str,
                "offset": int(field_info[1]),
                "values": _ndarray_content_identity(array[name]),
            }
        )
    return {
        "schema": "TEST_CLARA_NAMED_FIELD_NDARRAY_IDENTITY_V1",
        "shape": list(array.shape),
        "dtype_itemsize": int(array.dtype.itemsize),
        "dtype_is_aligned": bool(array.dtype.isalignedstruct),
        "fields": fields,
    }


def _state_universe_content_sha256(states: pd.DataFrame) -> str:
    ordered = states.sort_values("full_state_code", kind="mergesort").reset_index(
        drop=True
    )
    return _canonical_digest(
        {
            "columns": ["full_state_code", *STATE_FIELDS],
            "records": ordered[["full_state_code", *STATE_FIELDS]].to_dict("records"),
        }
    )


def _dataframe_content_sha256(frame: pd.DataFrame) -> str:
    """Stable in-process content identity used by formal result capabilities."""

    digest = hashlib.sha256()
    digest.update(_canonical_digest(list(frame.columns)).encode("ascii"))
    digest.update(
        _canonical_digest([str(dtype) for dtype in frame.dtypes]).encode("ascii")
    )
    digest.update(
        np.ascontiguousarray(
            pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
        ).tobytes(order="C")
    )
    return digest.hexdigest()


def _cart_selector_content_sha256(selector: ExtendedCartSelector) -> str:
    classifier = selector.classifier
    if not hasattr(classifier, "tree_"):
        raise RuntimeError("CART selector未拟合")
    payload = {
        "actions": list(selector.actions),
        "configuration": dict(selector.configuration),
        "feature_names": list(selector.encoder.feature_names),
        "classes": [str(value) for value in classifier.classes_],
        "parameters": classifier.get_params(deep=False),
        "tree_nodes": _structured_ndarray_named_field_identity(
            classifier.tree_.__getstate__()["nodes"]
        ),
        "tree_values": _ndarray_content_identity(classifier.tree_.__getstate__()["values"]),
    }
    return _canonical_digest(payload)


def _statistics_array_content_identity(
    arrays: Mapping[str, np.ndarray],
    cart_best_count: np.ndarray,
) -> tuple[dict[str, Any], str]:
    content = {
        name: _ndarray_content_identity(values)
        for name, values in sorted(arrays.items())
    }
    content["cart_best_count"] = _ndarray_content_identity(cart_best_count)
    return content, _canonical_digest(content)


def algorithm_dependency_closure() -> dict[str, Any]:
    """Hash the explicit direct/transitive algorithm implementation closure."""

    rows: list[dict[str, str]] = []
    for path in ALGORITHM_DEPENDENCY_PATHS:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise RuntimeError(f"六动作算法依赖缺失: {resolved}")
        try:
            relative = resolved.relative_to(REVISION_ROOT.resolve()).as_posix()
        except ValueError:
            relative = str(resolved)
        rows.append({"relative_path": relative, "sha256": _sha256_path(resolved)})
    rows = sorted(rows, key=lambda row: row["relative_path"])
    payload = {
        "schema": "TEST_CLARA_GEFCOM_6A_ALGORITHM_CLOSURE_V1",
        "dependencies": rows,
        "dependency_count": len(rows),
    }
    return {**payload, "algorithm_closure_sha256": _canonical_digest(payload)}


def action_library_sha256(actions: Sequence[str] = SIX_ACTIONS) -> str:
    action_order = tuple(str(value) for value in actions)
    if action_order != SIX_ACTIONS:
        raise ValueError("六动作库顺序失配")
    return _canonical_digest(
        {
            "schema": "TEST_CLARA_6A_ACTION_LIBRARY_IDENTITY_V1",
            "tie_order": list(action_order),
            "action_count": len(action_order),
        }
    )


def global_replay_scope_contract(
    *,
    horizons: Sequence[int] = FORMAL_HORIZONS,
    predictors: Sequence[str] = FORMAL_PREDICTORS,
    coverages: Sequence[float] = FORMAL_COVERAGES,
) -> dict[str, Any]:
    """Return the frozen protocol-level scope for one global LinUCB replay.

    This is deliberately distinct from the runtime coordinator key
    ``(price, seed, evaluation_zone)``.  Config, ledgers, policy identities, and
    replay children must all pin this same canonical payload.
    """

    normalized_horizons = tuple(int(value) for value in horizons)
    normalized_predictors = tuple(str(value) for value in predictors)
    normalized_coverages = tuple(float(value) for value in coverages)
    if normalized_horizons != FORMAL_HORIZONS:
        raise ValueError("LinUCB全局scope时距顺序失配")
    if normalized_predictors != FORMAL_PREDICTORS:
        raise ValueError("LinUCB全局scope预测器顺序失配")
    if normalized_coverages != FORMAL_COVERAGES:
        raise ValueError("LinUCB全局scope覆盖率顺序失配")
    return {
        "schema": "TEST_CLARA_6A_GLOBAL_REPLAY_SCOPE_V1",
        "coordination_unit": "heldout_zone_seed_price",
        "horizons": list(normalized_horizons),
        "predictors": list(normalized_predictors),
        "coverages": list(normalized_coverages),
        "order": "issue_time_global",
        "initialize_once": True,
    }


def global_replay_scope_sha256(
    *,
    horizons: Sequence[int] = FORMAL_HORIZONS,
    predictors: Sequence[str] = FORMAL_PREDICTORS,
    coverages: Sequence[float] = FORMAL_COVERAGES,
) -> str:
    """Canonical no-trailing-newline digest of the frozen replay scope."""

    return _canonical_digest(
        global_replay_scope_contract(
            horizons=horizons,
            predictors=predictors,
            coverages=coverages,
        )
    )


def frozen_zone_baseline_contract(
    heldout_zone: str,
    baseline_id: str,
) -> dict[str, Any]:
    """Load and bind one zone's immutable CART/LinUCB registry row."""

    zone = str(heldout_zone)
    baseline = str(baseline_id)
    if zone not in FORMAL_ZONES or baseline not in {"CARTBestAction", "LinUCB"}:
        raise ValueError("冻结registry查询zone或baseline非法")
    if _sha256_path(FROZEN_REGISTRY_PATH) != FROZEN_REGISTRY_SHA256:
        raise RuntimeError("S07冻结execution registry SHA失配")
    registry = pd.read_parquet(FROZEN_REGISTRY_PATH)
    rows = registry[
        registry["outer_heldout_zone"].astype(str).eq(zone)
        & registry["baseline_id"].astype(str).eq(baseline)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"S07 registry行不唯一: {zone}/{baseline}")
    row = rows.iloc[0]
    configuration = json.loads(str(row["selected_configuration_json"]))
    selected_identity = {
        "baseline_id": baseline,
        "selected_family": str(row["selected_family"]),
        "configuration": configuration,
    }
    if (
        str(row["registry_schema"]) != "S07_BASELINE_EXECUTION_REGISTRY_V2"
        or str(row["execution_status"]) != "FROZEN_PENDING_S07"
        or not bool(row["deployable"])
        or int(row["source_zone_count"]) != 9
        or set(json.loads(str(row["source_zones_json"])))
        != set(FORMAL_ZONES) - {zone}
        or _canonical_digest(selected_identity)
        != str(row["selected_configuration_sha256"])
        or not str(row["selected_configuration_id"])
        or not str(row["registry_row_sha256"])
    ):
        raise RuntimeError(f"S07 registry冻结行身份失配: {zone}/{baseline}")
    payload = {
        "schema": "TEST_CLARA_6A_FROZEN_ZONE_BASELINE_CONTRACT_V1",
        "registry_sha256": FROZEN_REGISTRY_SHA256,
        "outer_heldout_zone": zone,
        "source_zones": sorted(set(FORMAL_ZONES) - {zone}),
        "baseline_id": baseline,
        "selected_family": str(row["selected_family"]),
        "selected_configuration_id": str(row["selected_configuration_id"]),
        "selected_configuration_sha256": str(row["selected_configuration_sha256"]),
        "registry_row_sha256": str(row["registry_row_sha256"]),
        "configuration": configuration,
        "source_fit_scope": str(row["source_fit_scope"]),
        "selector_online_update_on_heldout": bool(
            row["selector_online_update_on_heldout"]
        ),
        "candidate_input_scope": str(row["candidate_input_scope"]),
    }
    return {**payload, "frozen_configuration_sha256": _canonical_digest(payload)}


def clara_v4_adaptive_contract() -> dict[str, Any]:
    """Freeze the legacy V4 rule inputs plus the new all-six action scope."""

    if _sha256_path(FOUR_VERSION_CONFIG_PATH) != FOUR_VERSION_CONFIG_SHA256:
        raise RuntimeError("CLARA四版本冻结协议SHA失配")
    config = json.loads(FOUR_VERSION_CONFIG_PATH.read_text(encoding="utf-8"))
    if (
        config.get("schema") != "TEST_CLARA_FOUR_VERSION_PROTOCOL_V1"
        or config.get("status")
        != "FROZEN_BEFORE_ADAPTIVE_HELDOUT_PERFORMANCE_READ"
        or config.get("version_contracts", {}).get("V4_FULLY_ADAPTIVE")
        != {"adaptive_support": True, "adaptive_guardrail": True}
    ):
        raise RuntimeError("CLARA V4冻结协议状态失配")
    payload = {
        "schema": "TEST_CLARA_6A_V4_ADAPTIVE_SUBCONTRACT_V1",
        "source_protocol_sha256": FOUR_VERSION_CONFIG_SHA256,
        "version": "V4_FULLY_ADAPTIVE",
        "fixed_support": config["fixed_support"],
        "adaptive_support": config["adaptive_support"],
        "adaptive_guardrail": config["adaptive_guardrail"],
        "version_contract": config["version_contracts"]["V4_FULLY_ADAPTIVE"],
        "action_library_sha256": action_library_sha256(),
        "adaptive_statistics_action_scope": "ALL_SIX_ACTIONS",
    }
    return {
        **payload,
        "v4_adaptive_subcontract_sha256": _canonical_digest(payload),
        "full_protocol_configuration_sha256": _canonical_digest(config),
    }


def validate_formal_v4_configuration(config: Mapping[str, Any]) -> dict[str, Any]:
    frozen = json.loads(FOUR_VERSION_CONFIG_PATH.read_text(encoding="utf-8"))
    contract = clara_v4_adaptive_contract()
    actual_sha256 = _canonical_digest(dict(config))
    if actual_sha256 != contract["full_protocol_configuration_sha256"]:
        raise RuntimeError("CLARA V4实际配置与冻结完整协议失配")
    if dict(config) != frozen:
        raise RuntimeError("CLARA V4实际配置结构与冻结协议失配")
    return {**contract, "actual_configuration_sha256": actual_sha256}


def price_sha256(price: "PriceSpec") -> str:
    price.validate()
    return _canonical_digest(
        {"schema": "TEST_CLARA_PRICE_IDENTITY_V1", **price.as_dict()}
    )


def linucb_policy_identity(
    *,
    price: "PriceSpec",
    evaluation_zone: str,
    seed: int,
    exploration_alpha: float,
    l2_regularization: float,
    horizons: Sequence[int],
    predictors: Sequence[str] = FORMAL_PREDICTORS,
    coverages: Sequence[float] = FORMAL_COVERAGES,
    actions: Sequence[str] = SIX_ACTIONS,
    frozen_configuration_contract: Mapping[str, Any] | None = None,
    derived_contract_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Identity that fixes the new 6A library missing from the legacy config id."""

    action_hash = action_library_sha256(actions)
    scope_contract = global_replay_scope_contract(
        horizons=horizons,
        predictors=predictors,
        coverages=coverages,
    )
    scope_sha256 = _canonical_digest(scope_contract)
    closure = algorithm_dependency_closure()
    if frozen_configuration_contract is None:
        frozen_configuration_contract = frozen_zone_baseline_contract(
            str(evaluation_zone), "LinUCB"
        )
    frozen_configuration = dict(frozen_configuration_contract)
    if derived_contract_identity is None or not str(
        derived_contract_identity.get("derived_contract_sha256") or ""
    ):
        raise RuntimeError("LinUCB policy缺少官方六动作派生合同身份")
    if (
        str(frozen_configuration.get("outer_heldout_zone"))
        != str(evaluation_zone)
        or str(frozen_configuration.get("baseline_id")) != "LinUCB"
        or dict(frozen_configuration.get("configuration", {}))
        != {
            "exploration_alpha": float(exploration_alpha),
            "l2_regularization": float(l2_regularization),
        }
        or not str(frozen_configuration.get("frozen_configuration_sha256") or "")
    ):
        raise RuntimeError("LinUCB实际alpha/lambda与持出区registry冻结配置失配")
    payload = {
        "schema": "TEST_CLARA_LINUCB_6A_POLICY_IDENTITY_V1",
        "method": "LinUCB_6A",
        "action_library_sha256": action_hash,
        "ordered_actions": list(SIX_ACTIONS),
        "algorithm_closure_sha256": closure["algorithm_closure_sha256"],
        "derived_contract_sha256": str(
            derived_contract_identity["derived_contract_sha256"]
        ),
        "frozen_zone_configuration_sha256": frozen_configuration[
            "frozen_configuration_sha256"
        ],
        "frozen_zone_configuration": frozen_configuration,
        "price_sha256": price_sha256(price),
        "price_id": price.price_id,
        "theta": [float(value) for value in price.theta],
        "evaluation_zone": str(evaluation_zone),
        "global_scope_key": {
            "theta_id": price.price_id,
            "seed": int(seed),
            "evaluation_zone": str(evaluation_zone),
        },
        "global_replay_scope_contract": scope_contract,
        "global_replay_scope_sha256": scope_sha256,
        "global_issue_time_horizons": [int(value) for value in horizons],
        "global_issue_time_predictors": [str(value) for value in predictors],
        "global_issue_time_coverages": [float(value) for value in coverages],
        "exploration_alpha": float(exploration_alpha),
        "l2_regularization": float(l2_regularization),
        "fresh_state_per_price": True,
        "cross_price_state_shared": False,
    }
    return {**payload, "policy_identity_sha256": _canonical_digest(payload)}


def fit_price_child_identity(
    *,
    stats: "SixActionSufficientStats",
    price: "PriceSpec",
    clara_configuration_sha256: str,
    cart_configuration_sha256: str,
    clara_v4_adaptive_subcontract_sha256: str,
    cart_frozen_zone_configuration_sha256: str,
    derived_contract_sha256: str,
) -> dict[str, Any]:
    """Build the immutable identity for one ``zone x seed x price`` fit child."""

    _assert_compact_content_current(stats)
    closure = algorithm_dependency_closure()
    payload = {
        "schema": "TEST_CLARA_GEFCOM_6A_FIT_PRICE_CHILD_IDENTITY_V1",
        "heldout_zone": stats.heldout_zone,
        "seed": stats.seed,
        "price_id": price.price_id,
        "price_sha256": price_sha256(price),
        "theta": [float(value) for value in price.theta],
        "action_library_sha256": action_library_sha256(stats.actions),
        "ordered_actions": list(stats.actions),
        "algorithm_closure_sha256": closure["algorithm_closure_sha256"],
        "compact_source_identity_sha256": str(stats.audit["identity_sha256"]),
        "training_horizons": list(stats.horizons),
        "source_zones": list(stats.source_zones),
        "clara_configuration_sha256": str(clara_configuration_sha256),
        "cart_configuration_sha256": str(cart_configuration_sha256),
        "clara_v4_adaptive_subcontract_sha256": str(
            clara_v4_adaptive_subcontract_sha256
        ),
        "cart_frozen_zone_configuration_sha256": str(
            cart_frozen_zone_configuration_sha256
        ),
        "derived_contract_sha256": str(derived_contract_sha256),
    }
    if any(
        not payload[field]
        for field in (
            "clara_configuration_sha256",
            "cart_configuration_sha256",
            "clara_v4_adaptive_subcontract_sha256",
            "cart_frozen_zone_configuration_sha256",
            "derived_contract_sha256",
        )
    ):
        raise ValueError("逐价fit child配置身份不得为空")
    return {**payload, "fit_child_identity_sha256": _canonical_digest(payload)}


def replay_price_child_identity(
    *,
    price: "PriceSpec",
    evaluation_zone: str,
    seed: int,
    fit_child_sha256: str,
    endpoint_root_sha256: str,
    linucb_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Pin a replay child to its same-price fit, endpoints, theta, and 6A policy."""

    if not fit_child_sha256 or not endpoint_root_sha256:
        raise ValueError("逐价replay child缺少fit或endpoint root身份")
    if str(linucb_identity.get("price_id")) != price.price_id:
        raise ValueError("试图将LinUCB策略接到其他价格replay child")
    if str(linucb_identity.get("price_sha256")) != price_sha256(price):
        raise ValueError("LinUCB策略价格哈希与replay child失配")
    if str(linucb_identity.get("action_library_sha256")) != action_library_sha256():
        raise ValueError("LinUCB策略动作库哈希失配")
    if str(linucb_identity.get("global_replay_scope_sha256")) != (
        global_replay_scope_sha256()
    ):
        raise ValueError("LinUCB策略全局回放scope哈希失配")
    if not str(linucb_identity.get("frozen_zone_configuration_sha256") or ""):
        raise ValueError("LinUCB策略未绑定持出区冻结超参")
    expected_scope_key = {
        "theta_id": price.price_id,
        "seed": int(seed),
        "evaluation_zone": str(evaluation_zone),
    }
    frozen_zone_configuration = dict(
        linucb_identity.get("frozen_zone_configuration", {})
    )
    if (
        str(linucb_identity.get("evaluation_zone")) != str(evaluation_zone)
        or dict(linucb_identity.get("global_scope_key", {})) != expected_scope_key
        or str(frozen_zone_configuration.get("outer_heldout_zone"))
        != str(evaluation_zone)
        or not str(linucb_identity.get("derived_contract_sha256") or "")
    ):
        raise ValueError("LinUCB policy与replay child的zone/seed/price/contract交叉接线")
    payload = {
        "schema": "TEST_CLARA_GEFCOM_6A_REPLAY_PRICE_CHILD_IDENTITY_V1",
        "evaluation_zone": str(evaluation_zone),
        "seed": int(seed),
        "price_id": price.price_id,
        "price_sha256": price_sha256(price),
        "theta": [float(value) for value in price.theta],
        "action_library_sha256": action_library_sha256(),
        "fit_child_sha256": str(fit_child_sha256),
        "endpoint_root_sha256": str(endpoint_root_sha256),
        "linucb_policy_identity_sha256": str(
            linucb_identity["policy_identity_sha256"]
        ),
        "global_replay_scope_sha256": str(
            linucb_identity["global_replay_scope_sha256"]
        ),
        "frozen_zone_configuration_sha256": str(
            linucb_identity["frozen_zone_configuration_sha256"]
        ),
        "derived_contract_sha256": str(
            linucb_identity["derived_contract_sha256"]
        ),
        "global_scope_key": dict(linucb_identity["global_scope_key"]),
    }
    return {**payload, "replay_child_identity_sha256": _canonical_digest(payload)}


def replay_policy_evaluation_child_identity(
    *,
    policy_mode: str,
    evaluation_price: "PriceSpec",
    policy_price: "PriceSpec",
    evaluation_zone: str,
    seed: int,
    fit_child_sha256: str,
    endpoint_root_sha256: str,
    policy_replay_child_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Wrap one policy-price replay for one independently scored evaluation theta."""

    scope = policy_evaluation_scope(
        policy_mode=policy_mode,
        evaluation_price=evaluation_price,
        policy_price=policy_price,
        require_formal_identity=True,
    )
    policy_identity = dict(policy_replay_child_identity)
    expected_sha = _canonical_digest(
        {
            key: value
            for key, value in policy_identity.items()
            if key != "replay_child_identity_sha256"
        }
    )
    if (
        policy_identity.get("schema")
        != "TEST_CLARA_GEFCOM_6A_REPLAY_PRICE_CHILD_IDENTITY_V1"
        or str(policy_identity.get("replay_child_identity_sha256")) != expected_sha
        or str(policy_identity.get("evaluation_zone")) != str(evaluation_zone)
        or int(policy_identity.get("seed", -1)) != int(seed)
        or str(policy_identity.get("price_id")) != policy_price.price_id
        or str(policy_identity.get("price_sha256")) != price_sha256(policy_price)
        or str(policy_identity.get("fit_child_sha256")) != str(fit_child_sha256)
        or str(policy_identity.get("endpoint_root_sha256"))
        != str(endpoint_root_sha256)
    ):
        raise RuntimeError(
            "policy replay identity与mode/evaluation/policy fit身份串线"
        )
    payload = {
        "schema": "TEST_CLARA_GEFCOM_6A_REPLAY_POLICY_EVALUATION_CHILD_IDENTITY_V1",
        "evaluation_zone": str(evaluation_zone),
        "seed": int(seed),
        "policy_mode": str(scope["policy_mode"]),
        "evaluation_price_id": evaluation_price.price_id,
        "evaluation_price_sha256": price_sha256(evaluation_price),
        "policy_price_id": policy_price.price_id,
        "policy_price_sha256": price_sha256(policy_price),
        "policy_evaluation_scope_sha256": str(
            scope["policy_evaluation_scope_sha256"]
        ),
        "action_library_sha256": action_library_sha256(),
        "fit_child_sha256": str(fit_child_sha256),
        "endpoint_root_sha256": str(endpoint_root_sha256),
        "policy_replay_child_identity_sha256": expected_sha,
        "linucb_policy_identity_sha256": str(
            policy_identity["linucb_policy_identity_sha256"]
        ),
    }
    return {
        **payload,
        "replay_child_identity_sha256": _canonical_digest(payload),
    }


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label}缺少字段: {missing}")


def _validate_event_ids(frame: pd.DataFrame, label: str) -> None:
    _require_columns(frame, ("event_id",), label)
    raw = frame["event_id"]
    if raw.isna().any():
        raise ValueError(f"{label} event_id含空值")
    text_values = raw.astype(str)
    if text_values.str.strip().eq("").any() or text_values.duplicated().any():
        raise ValueError(f"{label} event_id必须非空唯一")


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def assert_formal_price_grid(prices: Sequence["PriceSpec"]) -> None:
    observed = tuple(prices)
    if len(observed) != len(FORMAL_PRICE_ROWS):
        raise RuntimeError("正式价格网格数量失配")
    for price, expected in zip(observed, FORMAL_PRICE_ROWS):
        price.validate()
        expected_id, expected_ratio, expected_theta, expected_base = expected
        if (
            price.price_id != expected_id
            or not np.isclose(price.miss_to_capacity_ratio, expected_ratio, atol=1e-12, rtol=0.0)
            or not np.allclose(price.theta, expected_theta, atol=1e-12, rtol=0.0)
            or bool(price.base_submission_price) != bool(expected_base)
        ):
            raise RuntimeError(f"正式价格网格身份失配: {price.price_id}")


def assert_formal_price_subset(prices: Sequence["PriceSpec"]) -> None:
    """Validate a non-empty ordered subset for atomic price-child resume."""

    observed = tuple(prices)
    if not observed:
        raise RuntimeError("正式逐价回放子集不得为空")
    expected_by_id = {row[0]: row for row in FORMAL_PRICE_ROWS}
    positions: list[int] = []
    for price in observed:
        price.validate()
        expected = expected_by_id.get(price.price_id)
        if expected is None:
            raise RuntimeError(f"正式价格子集含未冻结ID: {price.price_id}")
        expected_price = PriceSpec(expected[0], expected[1], expected[2], expected[3])
        if price_sha256(price) != price_sha256(expected_price):
            raise RuntimeError(f"正式价格子集身份失配: {price.price_id}")
        positions.append(tuple(row[0] for row in FORMAL_PRICE_ROWS).index(price.price_id))
    if len(set(positions)) != len(positions) or positions != sorted(positions):
        raise RuntimeError("正式价格子集必须唯一且保持冻结顺序")


def _finite(values: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{label}含非有限值")
    return result


def _timestamp_us(values: pd.Series, label: str) -> np.ndarray:
    parsed = pd.to_datetime(values, errors="raise")
    result = parsed.to_numpy(dtype="datetime64[us]").astype(np.int64)
    if np.any(result == np.iinfo(np.int64).min):
        raise ValueError(f"{label}含NaT")
    return result


@dataclass(frozen=True)
class PriceSpec:
    price_id: str
    miss_to_capacity_ratio: float
    theta: tuple[float, float, float, float]
    base_submission_price: bool = False

    @property
    def capacity_weight(self) -> float:
        return float(self.theta[0])

    @property
    def miss_weight(self) -> float:
        return float(self.theta[2])

    def validate(self) -> None:
        if not self.price_id:
            raise ValueError("价格标识不得为空")
        theta = np.asarray(self.theta, dtype=np.float64)
        if len(theta) != 4 or not np.isfinite(theta).all() or np.any(theta < 0.0):
            raise ValueError(f"价格向量非法: {self.price_id}")
        if theta[0] != theta[1] or theta[2] != theta[3]:
            raise ValueError(
                f"当前compact分量只支持对称容量/缺口价格: {self.price_id}"
            )
        if self.capacity_weight <= 0.0:
            raise ValueError(f"容量权重必须为正: {self.price_id}")
        observed_ratio = self.miss_weight / self.capacity_weight
        if not np.isclose(
            observed_ratio,
            float(self.miss_to_capacity_ratio),
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError(f"价格比与theta不一致: {self.price_id}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "price_id": self.price_id,
            "miss_to_capacity_ratio": float(self.miss_to_capacity_ratio),
            "theta": [float(value) for value in self.theta],
            "base_submission_price": bool(self.base_submission_price),
        }


def price_specs_from_config(config: Mapping[str, Any]) -> tuple[PriceSpec, ...]:
    rows = config.get("price_grid")
    if not isinstance(rows, list) or not rows:
        raise ValueError("配置缺少价格网格")
    output = tuple(
        PriceSpec(
            price_id=str(row["price_id"]),
            miss_to_capacity_ratio=float(row["miss_to_capacity_ratio"]),
            theta=tuple(float(value) for value in row["theta"]),
            base_submission_price=bool(row.get("base_submission_price", False)),
        )
        for row in rows
    )
    if len({row.price_id for row in output}) != len(output):
        raise ValueError("价格标识重复")
    for row in output:
        row.validate()
    return output


def formal_price_spec(price_id: str) -> PriceSpec:
    for row in FORMAL_PRICE_ROWS:
        if str(row[0]) == str(price_id):
            return PriceSpec(
                price_id=row[0],
                miss_to_capacity_ratio=row[1],
                theta=row[2],
                base_submission_price=row[3],
            )
    raise ValueError(f"未知正式价格: {price_id}")


def policy_evaluation_scope(
    *,
    policy_mode: str,
    evaluation_price: PriceSpec,
    policy_price: PriceSpec | None = None,
    require_formal_identity: bool = True,
) -> dict[str, Any]:
    """Validate and identify one policy-mode x evaluation-price child.

    Policy fitting/replay always uses ``policy_price``.  Endpoint scoring always
    uses ``evaluation_price``.  The fixed deployment scan therefore reuses only
    the frozen main-price policy while never feeding evaluation-price losses back
    into CLARA/CART/LinUCB.
    """

    mode = str(policy_mode)
    if mode not in FORMAL_POLICY_MODES:
        raise ValueError(f"未知policy mode: {mode}")
    evaluation_price.validate()
    resolved_policy = (
        evaluation_price
        if policy_price is None and mode == POLICY_MODE_RESELECTED
        else (
            formal_price_spec(FORMAL_MAIN_PRICE_ID)
            if policy_price is None
            else policy_price
        )
    )
    resolved_policy.validate()
    if mode == POLICY_MODE_RESELECTED:
        if price_sha256(resolved_policy) != price_sha256(evaluation_price):
            raise RuntimeError("RESELECT_EACH_RATIO要求policy price等于evaluation price")
    elif resolved_policy.price_id != FORMAL_MAIN_PRICE_ID:
        raise RuntimeError("FIXED_MAIN_RATIO必须使用R05主价格策略")
    if require_formal_identity:
        assert_formal_price_subset((evaluation_price,))
        assert_formal_price_subset((resolved_policy,))
        expected_policy = (
            evaluation_price
            if mode == POLICY_MODE_RESELECTED
            else formal_price_spec(FORMAL_MAIN_PRICE_ID)
        )
        if price_sha256(resolved_policy) != price_sha256(expected_policy):
            raise RuntimeError("policy/evaluation价格身份与冻结双mode合同失配")
    payload = {
        "schema": "TEST_CLARA_GEFCOM_6A_POLICY_EVALUATION_SCOPE_V1",
        "policy_mode": mode,
        "evaluation_price_id": evaluation_price.price_id,
        "evaluation_price_sha256": price_sha256(evaluation_price),
        "policy_price_id": resolved_policy.price_id,
        "policy_price_sha256": price_sha256(resolved_policy),
        "policy_feedback_uses_evaluation_price": False,
        "endpoint_score_uses_evaluation_price": True,
    }
    return {**payload, "policy_evaluation_scope_sha256": _canonical_digest(payload)}


def six_action_contracts(
    contracts: FrozenContracts,
    actions: Sequence[str] = SIX_ACTIONS,
) -> FrozenContracts:
    """Return an in-memory derived contract with the six-action tie order."""

    action_order = tuple(str(action) for action in actions)
    if action_order != SIX_ACTIONS:
        raise ValueError("正式六动作顺序失配")
    protocol = copy.deepcopy(contracts.protocol)
    action_contract = protocol["action_contract"]
    action_contract["actions_in_tie_order"] = list(action_order)
    implementation_map = dict(action_contract.get("implementation_map", {}))
    implementation_map.update(
        {
            "TunedSingleConformal": "FrozenTunedSingleConformalEndpoint",
            "EqualEndpointEnsemble": "FrozenEqualEndpointEnsemble",
        }
    )
    action_contract["implementation_map"] = implementation_map
    action_contract["derived_selector_identity"] = SCHEMA
    return replace(contracts, protocol=protocol)


def derived_six_action_contract_identity(
    contracts: FrozenContracts,
) -> dict[str, Any]:
    """Validate and identify the sole official six-action derived contract."""

    official = load_frozen_contracts()
    expected = six_action_contracts(official)
    file_fields = (
        ("protocol_path", "protocol_sha256"),
        ("base_protocol_path", "base_protocol_sha256"),
        ("baseline_path", "baseline_sha256"),
        ("output_path", "output_sha256"),
        ("base_output_path", "base_output_sha256"),
    )
    for path_field, sha_field in file_fields:
        path = Path(getattr(contracts, path_field))
        if (
            path.resolve() != Path(getattr(expected, path_field)).resolve()
            or _sha256_path(path) != str(getattr(expected, sha_field))
            or str(getattr(contracts, sha_field)) != str(getattr(expected, sha_field))
        ):
            raise RuntimeError(f"六动作派生合同官方文件身份失配: {path_field}")
    object_fields = (
        "protocol",
        "base_protocol",
        "baseline_registry",
        "output_contract",
        "base_output_contract",
    )
    for field in object_fields:
        if getattr(contracts, field) != getattr(expected, field):
            raise RuntimeError(f"六动作派生合同内容漂移: {field}")
    payload = {
        "schema": "TEST_CLARA_6A_DERIVED_FROZEN_CONTRACT_V1",
        "actions": list(SIX_ACTIONS),
        "action_library_sha256": action_library_sha256(),
        "object_sha256": {
            field: _canonical_digest(getattr(contracts, field))
            for field in object_fields
        },
        "official_file_sha256": {
            sha_field: str(getattr(contracts, sha_field))
            for _, sha_field in file_fields
        },
    }
    return {**payload, "derived_contract_sha256": _canonical_digest(payload)}


def state_universe_from_contract(
    contracts: FrozenContracts,
    index_contract: Mapping[str, Any],
) -> pd.DataFrame:
    _, states = landscape_accelerator.state_frames(
        dict(index_contract), contracts.protocol
    )
    _validate_state_universe(states)
    return states


def _validate_state_universe(states: pd.DataFrame) -> None:
    _require_columns(states, ("full_state_code", *STATE_FIELDS), "查询状态全集")
    if states[list(STATE_FIELDS)].duplicated().any():
        raise ValueError("查询状态全集键重复")
    ordered = states.sort_values("full_state_code", kind="mergesort")
    codes = ordered["full_state_code"].to_numpy(dtype=np.int64)
    if not np.array_equal(codes, np.arange(len(ordered), dtype=np.int64)):
        raise ValueError("查询状态编号必须从0连续")


def endpoint_components(
    *,
    target: np.ndarray,
    schedule: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    cadence_minutes: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return symmetric capacity and miss exposures before price weighting."""

    target_values = _finite(target, "target")
    schedule_values = _finite(schedule, "schedule")
    lower_values = _finite(lower, "lower")
    upper_values = _finite(upper, "upper")
    if not (
        len(target_values)
        == len(schedule_values)
        == len(lower_values)
        == len(upper_values)
    ):
        raise ValueError("端点分量数组长度失配")
    if np.any(lower_values > upper_values):
        raise ValueError("区间上下界倒置")
    # Preserve the frozen endpoint ERRF semantics exactly.  GEFCom point
    # forecasts (schedule/base_center) can be slightly outside [0, 1], and the
    # historical endpoint contract intentionally scores those raw values rather
    # than clipping them.  Finite values plus non-inverted intervals are the
    # authoritative admissibility conditions here.
    scale = float(cadence_minutes) / 60.0
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("数据节拍非法")
    capacity = scale * (
        np.maximum(upper_values - schedule_values, 0.0)
        + np.maximum(schedule_values - lower_values, 0.0)
    )
    miss = scale * (
        np.maximum(target_values - upper_values, 0.0)
        + np.maximum(lower_values - target_values, 0.0)
    )
    return capacity, miss


def attach_price_losses(
    facts: pd.DataFrame,
    price: PriceSpec,
    *,
    actions: Sequence[str] = SIX_ACTIONS,
    cadence_minutes: float = 60.0,
    copy_frame: bool = True,
) -> pd.DataFrame:
    """Attach six price-specific ERRF columns to a target/source event frame."""

    price.validate()
    action_order = tuple(str(action) for action in actions)
    required = {"target_after_maturity", "schedule_proxy"}
    for action in action_order:
        required.update(
            {
                f"{action}__candidate_lower",
                f"{action}__candidate_upper",
            }
        )
    _require_columns(facts, required, "价格损失事件")
    output = facts.copy() if copy_frame else facts
    target = output["target_after_maturity"].to_numpy(dtype=np.float64)
    schedule = output["schedule_proxy"].to_numpy(dtype=np.float64)
    for action in action_order:
        capacity, miss = endpoint_components(
            target=target,
            schedule=schedule,
            lower=output[f"{action}__candidate_lower"].to_numpy(dtype=np.float64),
            upper=output[f"{action}__candidate_upper"].to_numpy(dtype=np.float64),
            cadence_minutes=cadence_minutes,
        )
        output[f"{action}__errf"] = (
            price.capacity_weight * capacity + price.miss_weight * miss
        )
    return output


def validated_fold_width_thresholds(
    thresholds: pd.DataFrame,
    facts: pd.DataFrame,
    *,
    outer_heldout_zone: str,
    predictor: str,
    seed: int,
) -> pd.DataFrame:
    """Select an exact fold threshold map, rejecting duplicates before any drop."""

    threshold_keys = ["predictor", "horizon_group", "target_coverage", "seed"]
    required = {
        "outer_heldout_zone",
        *threshold_keys,
        "raw_width_q33",
        "raw_width_q67",
        "heldout_zone_in_threshold",
    }
    _require_columns(thresholds, required, "折级宽度阈值")
    _require_columns(facts, threshold_keys, "源流阈值键")
    selected = thresholds[
        thresholds["outer_heldout_zone"].astype(str).eq(str(outer_heldout_zone))
        & thresholds["seed"].astype(int).eq(int(seed))
        & thresholds["predictor"].astype(str).eq(str(predictor))
    ][
        [
            *threshold_keys,
            "raw_width_q33",
            "raw_width_q67",
            "heldout_zone_in_threshold",
        ]
    ].copy()
    if selected[threshold_keys].duplicated().any():
        raise RuntimeError("折级宽度阈值键重复或冲突，拒绝静默去重")
    required_keys = facts[threshold_keys].drop_duplicates()
    key_audit = required_keys.merge(
        selected[threshold_keys],
        on=threshold_keys,
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    if not key_audit["_merge"].eq("both").all():
        raise RuntimeError("折级宽度阈值未覆盖源流全部状态键")
    if selected["heldout_zone_in_threshold"].astype(bool).any():
        raise RuntimeError("折级宽度阈值混入outer-heldout zone")
    numeric = selected[["raw_width_q33", "raw_width_q67"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or np.any(numeric[:, 0] > numeric[:, 1]):
        raise RuntimeError("折级宽度阈值含非有限值或q33>q67")
    return selected


def validate_endpoint_root_seal(
    endpoint_root_manifest_path: Path,
    *,
    expected_endpoint_root_sha256: str,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    """Validate the complete formal 10-zone x 3-seed x 5-horizon root seal."""

    path = Path(endpoint_root_manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"新增端点root seal不存在: {path}")
    # Hash and parse one immutable byte snapshot.  The formal root is small, and
    # avoiding the process cache closes both stale-cache and hash/read TOCTOU
    # windows during a long-running coordinator process.
    root_bytes = path.read_bytes()
    observed_root_sha256 = hashlib.sha256(root_bytes).hexdigest()
    if (
        not str(expected_endpoint_root_sha256)
        or observed_root_sha256 != str(expected_endpoint_root_sha256)
    ):
        raise RuntimeError("新增端点root seal SHA与调度器pin失配")
    manifest = json.loads(root_bytes.decode("utf-8"))
    expected_unit_ids = {
        f"{zone}__seed{seed}__H{horizon:02d}"
        for zone in FORMAL_ZONES
        for seed in range(3)
        for horizon in FORMAL_HORIZONS
    }
    unit_rows = manifest.get("unit_manifests")
    if not isinstance(unit_rows, list):
        raise RuntimeError("新增端点root seal缺少unit manifests")
    unit_ids = [str(row.get("unit_id")) for row in unit_rows]
    if len(unit_ids) != len(set(unit_ids)):
        raise RuntimeError("新增端点root seal unit ID重复")
    unit_map = {str(row["unit_id"]): row for row in unit_rows}
    identities = manifest.get("identities")
    config = json.loads(GEFCOM_SIX_ACTION_CONFIG_PATH.read_text(encoding="utf-8"))
    endpoint_contract = dict(config.get("endpoint_contract", {}))
    expected_dependency_hashes = {
        str(input_id): str(config["inputs"][str(input_id)]["sha256"])
        for input_id in endpoint_contract.get("identity_input_ids", [])
    }
    endpoint_algorithm_path = TEST_ROOT / str(
        endpoint_contract.get("algorithm_module", "")
    )
    # Reuse the producer's live identity function instead of maintaining a
    # second, subtly different canonicalization/schema in the consumer.  The
    # runner imports this module lazily, so resolving it here is cycle-safe.
    endpoint_runner = importlib.import_module("run_gefcom_six_action_full_v1")
    expected_endpoint_identity = dict(
        endpoint_runner.endpoint_artifact_identity(config)
    )
    runner_closure = (
        identities.get("endpoint_runner_code_closure", {})
        if isinstance(identities, Mapping)
        else {}
    )
    closure_functions = (
        runner_closure.get("function_sha256", {})
        if isinstance(runner_closure, Mapping)
        else {}
    )
    closure_constants = (
        runner_closure.get("constants", {})
        if isinstance(runner_closure, Mapping)
        else {}
    )
    closure_valid = (
        isinstance(runner_closure, Mapping)
        and set(runner_closure)
        == {
            "root_functions",
            "function_sha256",
            "function_count",
            "constants",
            "constant_count",
            "closure_sha256",
        }
        and isinstance(runner_closure.get("root_functions"), list)
        and bool(runner_closure.get("root_functions"))
        and isinstance(closure_functions, Mapping)
        and bool(closure_functions)
        and all(_is_sha256(value) for value in closure_functions.values())
        and int(runner_closure.get("function_count", -1)) == len(closure_functions)
        and isinstance(closure_constants, Mapping)
        and bool(closure_constants)
        and int(runner_closure.get("constant_count", -1)) == len(closure_constants)
        and str(runner_closure.get("closure_sha256"))
        == _runner_newline_canonical_digest(
            {"functions": dict(closure_functions), "constants": dict(closure_constants)}
        )
    )
    identity_contract_valid = (
        isinstance(identities, Mapping)
        and dict(identities) == expected_endpoint_identity
        and set(identities) == set(expected_endpoint_identity)
        and str(identities.get("endpoint_contract_sha256"))
        == _canonical_digest(endpoint_contract)
        and str(identities.get("regression_gate_sha256"))
        == _canonical_digest(config.get("regression_gate", {}))
        and dict(identities.get("endpoint_dependency_hashes", {}))
        == expected_dependency_hashes
        and str(identities.get("endpoint_dependency_set_sha256"))
        == _canonical_digest(expected_dependency_hashes)
        and endpoint_algorithm_path.is_file()
        and str(identities.get("endpoint_algorithm_module_sha256"))
        == _sha256_path(endpoint_algorithm_path)
        and closure_valid
        and str(
            expected_dependency_hashes.get("migration_full_sha256_manifest", "")
        )
        == "0f5c18097f13f3f7c6fe3b6a1ff09c6a54d1afebca8860b9dcff9dcafab1c2f4"
    )
    if (
        manifest.get("schema") != "TEST_CLARA_GEFCOM_6A_ENDPOINT_ROOT_SEAL_V1"
        or manifest.get("status") != "PASS"
        or manifest.get("training_support_identity")
        != "B_UNIFIED_FIVE_HORIZON_NEW_IDENTITY"
        or tuple(int(value) for value in manifest.get("training_horizons", []))
        != FORMAL_HORIZONS
        or int(manifest.get("unit_count", -1)) != 150
        or set(unit_map) != expected_unit_ids
        or int(manifest.get("total_event_count", -1)) != 23_024_760
        or int(manifest.get("expected_total_event_count", -1)) != 23_024_760
        or int(manifest.get("total_event_action_count", -1)) != 46_049_520
        or int(manifest.get("coverage_unit_count", -1)) != 1_650
        or int(manifest.get("predictor_unit_count", -1)) != 600
        or int(manifest.get("addon_action_unit_count", -1)) != 300
        or not identity_contract_valid
    ):
        raise RuntimeError("新增端点root seal未完整闭合10×3×5正式库")
    if any(
        not _is_sha256(row.get("unit_manifest_sha256"))
        or int(row.get("event_count", -1)) != int(row.get("expected_event_count", -2))
        for row in unit_rows
    ):
        raise RuntimeError("新增端点root seal逐unit身份或事件数未闭合")
    return manifest, unit_map


def _assert_endpoint_unit_root_identity(
    addon_manifest: Mapping[str, Any],
    endpoint_root_manifest: Mapping[str, Any],
    *,
    source_zone: str,
    horizon: int,
    seed: int,
) -> None:
    """Reject a scientifically different endpoint unit under a valid root."""

    if (
        addon_manifest.get("schema") != "TEST_CLARA_GEFCOM_6A_ENDPOINT_UNIT_V1"
        or addon_manifest.get("status") != "PASS"
        or str(addon_manifest.get("zone")) != str(source_zone)
        or int(addon_manifest.get("horizon", -1)) != int(horizon)
        or int(addon_manifest.get("seed", -1)) != int(seed)
        or addon_manifest.get("identities")
        != endpoint_root_manifest.get("identities")
    ):
        raise RuntimeError("新增端点unit manifest状态或身份失配")


@dataclass(frozen=True)
class VerifiedCausalSourceAudit(ABCMapping[str, Any]):
    """In-process capability emitted only after the formal loader closes all gates."""

    _payload: Mapping[str, Any] | str
    _capability: object

    def __post_init__(self) -> None:
        payload = (
            self._payload
            if isinstance(self._payload, str)
            else json.dumps(
                dict(self._payload),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        )
        object.__setattr__(self, "_payload", payload)

    def _decoded(self) -> dict[str, Any]:
        return json.loads(str(self._payload))

    def __getitem__(self, key: str) -> Any:
        return self._decoded()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._decoded())

    def __len__(self) -> int:
        return len(self._decoded())

    def verified_by_formal_loader(self) -> bool:
        return self._capability is _CAUSAL_LOADER_CAPABILITY


def formal_causal_source_paths(
    *, source_zone: str, predictor: str, horizon: int, seed: int
) -> dict[str, Path]:
    stream_id = f"{predictor}-H{int(horizon):02d}-{source_zone}-S{int(seed)}"
    unit_id = f"{source_zone}__seed{int(seed)}__H{int(horizon):02d}"
    return {
        "source_fact": FORMAL_S06_ROOT
        / "source_facts"
        / f"predictor={predictor}"
        / f"zone={source_zone}"
        / f"horizon={int(horizon):02d}"
        / f"seed={int(seed)}"
        / "facts.parquet",
        "candidate_bundle_dir": FORMAL_S03_ROOT / "bundles" / predictor / stream_id,
        "endpoint_addon": FORMAL_ENDPOINT_ROOT / unit_id / "endpoint_addons.parquet",
        "endpoint_root_manifest": FORMAL_ENDPOINT_ROOT_MANIFEST,
        "width_thresholds": FORMAL_S06_ROOT / "fold_width_thresholds.parquet",
    }


def load_causal_source_stream(
    *,
    source_fact_path: Path,
    candidate_bundle_dir: Path,
    endpoint_addon_path: Path,
    endpoint_root_manifest_path: Path,
    expected_endpoint_root_sha256: str,
    width_thresholds_path: Path,
    outer_heldout_zone: str,
    source_zone: str,
    predictor: str,
    horizon: int,
    seed: int,
    coverages: Sequence[float],
    verify_leaf_hashes: bool = True,
) -> tuple[pd.DataFrame, VerifiedCausalSourceAudit]:
    """Load one source stream under the strict-maturity and fold-state contract.

    Raw S06 facts still contain a small terminal tail not present as four-action
    strictly mature feedback in S03.  This loader is the only supported formal path:
    it filters that tail, merges mature targets/schedules, derives raw-width state
    using the *outer heldout* fold thresholds, then joins the two addon actions.
    """

    fact_path = Path(source_fact_path)
    bundle_dir = Path(candidate_bundle_dir)
    addon_path = Path(endpoint_addon_path)
    threshold_path = Path(width_thresholds_path)
    endpoint_root_path = Path(endpoint_root_manifest_path)
    if tuple(float(value) for value in coverages) != FORMAL_COVERAGES:
        raise RuntimeError("正式因果源loader必须使用冻结十一覆盖率")
    if (
        str(source_zone) not in FORMAL_ZONES
        or str(outer_heldout_zone) not in FORMAL_ZONES
        or str(source_zone) == str(outer_heldout_zone)
        or str(predictor) not in FORMAL_PREDICTORS
        or int(horizon) not in FORMAL_HORIZONS
        or int(seed) not in range(3)
    ):
        raise RuntimeError("正式因果源loader的zone/predictor/horizon/seed身份失配")
    official_paths = formal_causal_source_paths(
        source_zone=str(source_zone),
        predictor=str(predictor),
        horizon=int(horizon),
        seed=int(seed),
    )
    supplied_paths = {
        "source_fact": fact_path,
        "candidate_bundle_dir": bundle_dir,
        "endpoint_addon": addon_path,
        "endpoint_root_manifest": endpoint_root_path,
        "width_thresholds": threshold_path,
    }
    path_mismatch = {
        name: {"supplied": str(path.resolve()), "official": str(official_paths[name].resolve())}
        for name, path in supplied_paths.items()
        if path.resolve() != official_paths[name].resolve()
    }
    if path_mismatch:
        raise RuntimeError(f"正式因果源loader拒绝非官方冻结路径: {path_mismatch}")
    fact_manifest_path = fact_path.parent / "manifest.json"
    bundle_manifest_path = bundle_dir / "manifest.json"
    addon_manifest_path = addon_path.parent / "manifest.json"
    paths = {
        "source_fact": fact_path,
        "source_fact_manifest": fact_manifest_path,
        "bundle_manifest": bundle_manifest_path,
        "event_core": bundle_dir / "event_core.parquet",
        "candidate_intervals": bundle_dir / "candidate_intervals.parquet",
        "feedback_trace": bundle_dir / "feedback_trace.parquet",
        "endpoint_addon": addon_path,
        "endpoint_addon_manifest": addon_manifest_path,
        "endpoint_source_anchor": addon_path.parent / "source_anchor_audit.json",
        "endpoint_root_manifest": endpoint_root_path,
        "width_thresholds": threshold_path,
    }
    missing = {name: str(path) for name, path in paths.items() if not path.is_file()}
    if missing:
        raise FileNotFoundError(f"正式源流输入缺失: {missing}")
    fact_manifest = json.loads(fact_manifest_path.read_text(encoding="utf-8"))
    bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    addon_manifest = json.loads(addon_manifest_path.read_text(encoding="utf-8"))
    endpoint_root_manifest, root_units = validate_endpoint_root_seal(
        endpoint_root_path,
        expected_endpoint_root_sha256=str(expected_endpoint_root_sha256),
    )
    expected_identity = {
        "predictor": str(predictor),
        "zone": str(source_zone),
        "horizon": int(horizon),
        "seed": int(seed),
    }
    if fact_manifest.get("status") != "COMPLETE_VALIDATED":
        raise RuntimeError("正式源流S06 manifest未通过")
    if bundle_manifest.get("status") != "COMPLETE_VALIDATED":
        raise RuntimeError("正式源流S03 manifest未通过")
    _assert_endpoint_unit_root_identity(
        addon_manifest,
        endpoint_root_manifest,
        source_zone=str(source_zone),
        horizon=int(horizon),
        seed=int(seed),
    )
    unit_id = f"{source_zone}__seed{int(seed)}__H{int(horizon):02d}"
    if unit_id not in root_units or str(root_units[unit_id]["unit_manifest_sha256"]) != _sha256_path(
        addon_manifest_path
    ):
        raise RuntimeError("新增端点unit未被root seal准确pin")
    identities = addon_manifest.get("identities")
    dependency_hashes = (
        identities.get("endpoint_dependency_hashes", {})
        if isinstance(identities, Mapping)
        else {}
    )
    if (
        not isinstance(dependency_hashes, Mapping)
        or str(dependency_hashes.get("fold_width_thresholds"))
        != _sha256_path_cached(threshold_path)
    ):
        raise RuntimeError("折级宽度阈值SHA与端点unit冻结依赖失配")
    source_anchor_path = paths["endpoint_source_anchor"]
    if str(addon_manifest.get("source_anchor_audit_json_sha256")) != (
        _sha256_path_cached(source_anchor_path)
    ):
        raise RuntimeError("端点source anchor未被unit manifest准确pin")
    source_anchor = json.loads(source_anchor_path.read_text(encoding="utf-8"))
    anchor_records = source_anchor.get("records")
    if (
        source_anchor.get("schema")
        != "TEST_CLARA_GEFCOM_6A_ENDPOINT_SOURCE_ANCHOR_AUDIT_V1"
        or source_anchor.get("status") != "PASS"
        or source_anchor.get("unit_id") != unit_id
        or str(source_anchor.get("migration_manifest_sha256"))
        != str(
            endpoint_root_manifest["identities"]["endpoint_dependency_hashes"][
                "migration_full_sha256_manifest"
            ]
        )
        or not isinstance(anchor_records, list)
        or int(source_anchor.get("checked_file_count", -1)) != 32
        or len(anchor_records) != 32
        or len({str(row.get("relative_path")) for row in anchor_records}) != 32
        or {str(row.get("stream_id")) for row in anchor_records}
        != {
            f"{item}-H{int(horizon):02d}-{source_zone}-S{int(seed)}"
            for item in FORMAL_PREDICTORS
        }
    ):
        raise RuntimeError("端点source anchor审计未完整闭合4预测器×8文件")
    current_stream_id = f"{predictor}-H{int(horizon):02d}-{source_zone}-S{int(seed)}"
    current_records = [
        row for row in anchor_records if str(row.get("stream_id")) == current_stream_id
    ]
    if len(current_records) != 8:
        raise RuntimeError("当前源流source anchor记录数失配")
    normalized_records = [
        {
            **row,
            "normalized_path": str(row.get("relative_path", "")).replace("\\", "/"),
        }
        for row in current_records
    ]
    anchored_current_paths = {
        "source_fact": fact_path,
        "source_fact_manifest": fact_manifest_path,
        "bundle_manifest": bundle_manifest_path,
        "event_core": paths["event_core"],
        "candidate_intervals": paths["candidate_intervals"],
        "feedback_trace": paths["feedback_trace"],
    }
    for name, current_path in anchored_current_paths.items():
        basename = current_path.name
        is_source = name.startswith("source_fact")
        matches = [
            row
            for row in normalized_records
            if row["normalized_path"].endswith("/" + basename)
            and (("/source_facts/" in row["normalized_path"]) == is_source)
            and (("/bundles/" in row["normalized_path"]) == (not is_source))
        ]
        if len(matches) != 1 or str(matches[0].get("sha256")) != _sha256_path_cached(
            current_path
        ):
            raise RuntimeError(f"当前源流文件与迁移source anchor失配: {name}")
    for field, expected in expected_identity.items():
        observed_fact = fact_manifest.get(field)
        observed_bundle = bundle_manifest.get(field)
        if field in {"horizon", "seed"}:
            observed_fact = int(observed_fact)
            observed_bundle = int(observed_bundle)
        else:
            observed_fact = str(observed_fact)
            observed_bundle = str(observed_bundle)
        if observed_fact != expected or observed_bundle != expected:
            raise RuntimeError(f"正式源流manifest身份失配: {field}")
    if str(source_zone) == str(outer_heldout_zone):
        raise RuntimeError("正式源流不得使用outer-heldout zone")
    observed_hashes: dict[str, str] = {}
    if verify_leaf_hashes:
        for name in (
            "source_fact",
            "event_core",
            "candidate_intervals",
            "feedback_trace",
            "endpoint_addon",
            "width_thresholds",
        ):
            observed_hashes[name] = _sha256_path_cached(paths[name])
        expected_hashes = {
            "source_fact": str(fact_manifest["facts_sha256"]),
            "event_core": str(bundle_manifest["event_core_sha256"]),
            "candidate_intervals": str(bundle_manifest["candidate_intervals_sha256"]),
            "feedback_trace": str(bundle_manifest["feedback_trace_sha256"]),
            "endpoint_addon": str(addon_manifest["endpoint_addons_parquet_sha256"]),
        }
        mismatch = {
            name: {"expected": expected, "observed": observed_hashes.get(name)}
            for name, expected in expected_hashes.items()
            if observed_hashes.get(name) != expected
        }
        if mismatch:
            raise RuntimeError(f"正式源流leaf hash失配: {mismatch}")
        if str(fact_manifest["input_bundle_manifest_sha256"]) != _sha256_path_cached(
            bundle_manifest_path
        ):
            raise RuntimeError("S06与S03 bundle manifest身份链断裂")

    facts = pd.read_parquet(fact_path)
    _validate_event_ids(facts, "S06源事件")
    facts["event_id"] = facts["event_id"].astype(str)
    raw_event_count = len(facts)
    if raw_event_count != int(fact_manifest["complete_case_event_count"]):
        raise RuntimeError("S06源事件行数与manifest失配")
    feedback = pd.read_parquet(
        paths["feedback_trace"],
        columns=["event_id", "action", "eligible_by_strict_time_rule"],
    )
    feedback["event_id"] = feedback["event_id"].astype(str)
    feedback["action"] = feedback["action"].astype(str)
    eligible = feedback[feedback["eligible_by_strict_time_rule"].astype(bool)].copy()
    if eligible[["event_id", "action"]].duplicated().any():
        raise RuntimeError("S03严格成熟反馈键重复")
    action_sets = eligible.groupby("event_id", sort=False)["action"].agg(
        lambda values: tuple(sorted(set(values)))
    )
    expected_actions = tuple(sorted(SIX_ACTIONS[:4]))
    partial = action_sets[action_sets.map(lambda values: values != expected_actions)]
    if len(partial):
        raise RuntimeError(
            f"S03严格成熟反馈存在非四动作事件: {partial.head(3).to_dict()}"
        )
    mature_ids = set(action_sets.index.astype(str)) & set(facts["event_id"])
    facts = facts[facts["event_id"].isin(mature_ids)].copy()
    if len(eligible[eligible["event_id"].isin(mature_ids)]) != len(facts) * 4:
        raise RuntimeError("S03严格成熟四动作反馈不守恒")
    s06_unmatured_count = raw_event_count - len(facts)
    if s06_unmatured_count < 0:
        raise RuntimeError("S06未成熟事件计数非法")

    event_core = pd.read_parquet(
        paths["event_core"], columns=["event_id", "target", "base_center"]
    )
    event_core["event_id"] = event_core["event_id"].astype(str)
    event_core = event_core[event_core["event_id"].isin(mature_ids)].copy()
    if len(event_core) != len(facts) or event_core["event_id"].duplicated().any():
        raise RuntimeError("成熟event_core映射不完整")
    facts = facts.merge(
        event_core.rename(
            columns={"target": "target_after_maturity", "base_center": "schedule_proxy"}
        ),
        on="event_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    thresholds = pd.read_parquet(threshold_path)
    threshold_keys = ["predictor", "horizon_group", "target_coverage", "seed"]
    selected_thresholds = validated_fold_width_thresholds(
        thresholds,
        facts,
        outer_heldout_zone=str(outer_heldout_zone),
        predictor=str(predictor),
        seed=int(seed),
    )
    facts = facts.merge(
        selected_thresholds,
        on=threshold_keys,
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if facts[["raw_width_q33", "raw_width_q67"]].isna().any().any():
        raise RuntimeError("折级宽度阈值映射不完整")
    if facts["heldout_zone_in_threshold"].astype(bool).any():
        raise RuntimeError("折级宽度阈值混入outer-heldout zone")
    widths = facts["raw_width_value"].to_numpy(dtype=np.float64)
    facts["raw_width_state"] = np.where(
        widths <= facts["raw_width_q33"].to_numpy(dtype=np.float64),
        "narrow",
        np.where(
            widths <= facts["raw_width_q67"].to_numpy(dtype=np.float64),
            "medium",
            "wide",
        ),
    )

    addon_columns = [
        "event_id",
        "zone_or_farm",
        "predictor",
        "seed",
        "horizon_steps",
        "target_coverage",
        *(
            f"{action}__{field}"
            for action in SIX_ACTIONS[4:]
            for field in (
                "candidate_lower",
                "candidate_upper",
                "covered",
                "tuwr_indicator",
                "ard_value",
            )
        ),
    ]
    addons = pd.read_parquet(addon_path, columns=addon_columns)
    addons["event_id"] = addons["event_id"].astype(str)
    addons = addons[
        addons["event_id"].isin(mature_ids)
        & addons["zone_or_farm"].astype(str).eq(str(source_zone))
        & addons["predictor"].astype(str).eq(str(predictor))
        & addons["seed"].astype(int).eq(int(seed))
        & addons["horizon_steps"].astype(int).eq(int(horizon))
    ].copy()
    if len(addons) != len(facts) or addons["event_id"].duplicated().any():
        raise RuntimeError("新增六动作端点与严格成熟事件集失配")
    addon_values = [
        "event_id",
        *[column for column in addon_columns if "__" in column],
    ]
    facts = facts.merge(
        addons[addon_values],
        on="event_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if set(facts["target_coverage"].astype(float)) != set(
        float(value) for value in coverages
    ):
        raise RuntimeError("正式源流十一覆盖率不完整")
    complete_hash = _canonical_digest(sorted(facts["event_id"].astype(str)))
    audit = {
        "schema": "TEST_CLARA_GEFCOM_6A_CAUSAL_SOURCE_STREAM_AUDIT_V1",
        "status": "PASS",
        "source_zone": str(source_zone),
        "outer_heldout_zone": str(outer_heldout_zone),
        "outer_heldout_zone_used_as_source": False,
        "predictor": str(predictor),
        "horizon": int(horizon),
        "seed": int(seed),
        "strict_maturity_filter_applied": True,
        "mature_action_count": 4,
        "raw_s06_event_count": raw_event_count,
        "strict_mature_event_count": len(facts),
        "s06_unmatured_event_count": s06_unmatured_count,
        "strict_mature_feedback_row_count": len(facts) * 4,
        "complete_case_event_id_sha256": complete_hash,
        "raw_width_threshold_outer_fold": str(outer_heldout_zone),
        "leaf_hashes_verified": bool(verify_leaf_hashes),
        "leaf_sha256": observed_hashes,
        "verified_paths": {
            name: str(path.resolve())
            for name, path in paths.items()
            if name
            in {
                "source_fact",
                "event_core",
                "candidate_intervals",
                "feedback_trace",
                "endpoint_addon",
                "width_thresholds",
                "endpoint_root_manifest",
                "endpoint_source_anchor",
            }
        },
        "source_fact_manifest_sha256": _sha256_path_cached(fact_manifest_path),
        "bundle_manifest_sha256": _sha256_path_cached(bundle_manifest_path),
        "endpoint_manifest_sha256": _sha256_path_cached(addon_manifest_path),
        "endpoint_root_manifest_sha256": _sha256_path_cached(endpoint_root_path),
        "width_thresholds_sha256": _sha256_path_cached(threshold_path),
        "source_anchor_audit_sha256": _sha256_path_cached(source_anchor_path),
        "migration_manifest_sha256": str(
            source_anchor.get("migration_manifest_sha256")
        ),
        "endpoint_root_seal_verified": True,
    }
    return facts.reset_index(drop=True), VerifiedCausalSourceAudit(
        _payload=audit,
        _capability=_CAUSAL_LOADER_CAPABILITY,
    )


@dataclass(frozen=True)
class SixActionSufficientStats:
    schema: str
    actions: tuple[str, ...]
    prices: tuple[PriceSpec, ...]
    heldout_zone: str
    source_zones: tuple[str, ...]
    seed: int
    horizons: tuple[int, ...]
    predictors: tuple[str, ...]
    coverages: tuple[float, ...]
    states: pd.DataFrame
    arrays: dict[str, np.ndarray]
    cart_best_count: np.ndarray
    stream_audit: pd.DataFrame
    audit: dict[str, Any]

    def price_index(self, price_id: str) -> int:
        matches = [
            index for index, row in enumerate(self.prices) if row.price_id == str(price_id)
        ]
        if len(matches) != 1:
            raise KeyError(f"价格标识不唯一或不存在: {price_id}")
        return matches[0]

    def arrays_for_price(self, price: PriceSpec | str) -> dict[str, np.ndarray]:
        if isinstance(price, str):
            current = self.prices[self.price_index(price)]
        else:
            price.validate()
            stored = self.prices[self.price_index(price.price_id)]
            if price_sha256(price) != price_sha256(stored):
                raise ValueError(
                    f"价格ID与充分统计中的ratio/theta/base身份失配: "
                    f"{price.price_id}"
                )
            current = stored
        output = {
            name: values
            for name, values in self.arrays.items()
            if name not in {"capacity_sum", "miss_sum"}
        }
        output["errf_sum"] = (
            current.capacity_weight * self.arrays["capacity_sum"]
            + current.miss_weight * self.arrays["miss_sum"]
        )
        return output


def _assert_compact_content_current(stats: SixActionSufficientStats) -> None:
    _, current_array_sha256 = _statistics_array_content_identity(
        stats.arrays, stats.cart_best_count
    )
    current_state_sha256 = _state_universe_content_sha256(stats.states)
    if current_array_sha256 != str(
        stats.audit.get("sufficient_statistics_content_sha256")
    ):
        raise RuntimeError("compact充分统计数组在finalize后发生变异")
    if current_state_sha256 != str(stats.audit.get("state_universe_content_sha256")):
        raise RuntimeError("compact状态全集在finalize后发生变异")


class SixActionStatsBuilder:
    """One-pass, bounded-memory reducer for a heldout-zone/seed source archive."""

    def __init__(
        self,
        *,
        states: pd.DataFrame,
        prices: Sequence[PriceSpec],
        heldout_zone: str,
        source_zones: Sequence[str],
        seed: int,
        horizons: Sequence[int] = FORMAL_HORIZONS,
        predictors: Sequence[str] = FORMAL_PREDICTORS,
        coverages: Sequence[float] = FORMAL_COVERAGES,
        actions: Sequence[str] = SIX_ACTIONS,
        cadence_minutes: float = 60.0,
        require_causal_source_audit: bool = True,
    ) -> None:
        _validate_state_universe(states)
        self.states = states.sort_values("full_state_code", kind="mergesort").reset_index(
            drop=True
        )
        self.prices = tuple(prices)
        for price in self.prices:
            price.validate()
        if len(self.prices) == 0 or len({row.price_id for row in self.prices}) != len(
            self.prices
        ):
            raise ValueError("价格网格为空或标识重复")
        self.actions = tuple(str(value) for value in actions)
        if self.actions != SIX_ACTIONS:
            raise ValueError("六动作顺序失配")
        self.heldout_zone = str(heldout_zone)
        self.source_zones = tuple(sorted(str(value) for value in source_zones))
        if self.heldout_zone in self.source_zones:
            raise ValueError("外层持出区出现在源区集合")
        if len(set(self.source_zones)) != len(self.source_zones):
            raise ValueError("源区集合重复")
        self.seed = int(seed)
        self.horizons = tuple(sorted(int(value) for value in horizons))
        self.predictors = tuple(str(value) for value in predictors)
        self.coverages = tuple(float(value) for value in coverages)
        self.cadence_minutes = float(cadence_minutes)
        self.require_causal_source_audit = bool(require_causal_source_audit)
        formal_price_grid = True
        try:
            assert_formal_price_grid(self.prices)
        except (ValueError, RuntimeError):
            formal_price_grid = False
        formal_shape = (
            self.horizons == FORMAL_HORIZONS
            and self.predictors == FORMAL_PREDICTORS
            and self.coverages == FORMAL_COVERAGES
            and len(self.source_zones) == 9
            and set(self.source_zones) == set(FORMAL_ZONES) - {self.heldout_zone}
            and formal_price_grid
        )
        if formal_shape and not self.require_causal_source_audit:
            raise RuntimeError(
                "正式GEFCom builder禁止关闭strict-maturity/leaf-hash因果门"
            )
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("训练时距重复")
        if len(set(self.predictors)) != len(self.predictors):
            raise ValueError("预测器重复")
        if len(set(self.coverages)) != len(self.coverages):
            raise ValueError("覆盖率重复")
        self._state_lookup = self.states[
            ["full_state_code", *STATE_FIELDS]
        ].copy()
        zone_count = len(self.source_zones)
        state_count = len(self.states)
        action_count = len(self.actions)
        price_count = len(self.prices)
        maximum = np.iinfo(np.int64).max
        minimum = np.iinfo(np.int64).min
        self.arrays: dict[str, np.ndarray] = {
            "event_count": np.zeros((zone_count, state_count), dtype=np.int64),
            "issue_min_us": np.full((zone_count, state_count), maximum, dtype=np.int64),
            "issue_max_us": np.full((zone_count, state_count), minimum, dtype=np.int64),
            "guardrail_count": np.zeros((zone_count, state_count), dtype=np.int64),
            "guardrail_issue_min_us": np.full(
                (zone_count, state_count), maximum, dtype=np.int64
            ),
            "guardrail_issue_max_us": np.full(
                (zone_count, state_count), minimum, dtype=np.int64
            ),
            "capacity_sum": np.zeros(
                (action_count, zone_count, state_count), dtype=np.float64
            ),
            "miss_sum": np.zeros(
                (action_count, zone_count, state_count), dtype=np.float64
            ),
            "covered_sum": np.zeros(
                (action_count, zone_count, state_count), dtype=np.float64
            ),
            "guardrail_covered_sum": np.zeros(
                (action_count, zone_count, state_count), dtype=np.float64
            ),
            "tuwr_sum": np.zeros(
                (action_count, zone_count, state_count), dtype=np.float64
            ),
            "ard_sum": np.zeros(
                (action_count, zone_count, state_count), dtype=np.float64
            ),
        }
        self.cart_best_count = np.zeros(
            (price_count, action_count, zone_count, state_count), dtype=np.int64
        )
        self._streams: set[tuple[str, str, int, int]] = set()
        self._stream_rows: list[dict[str, Any]] = []
        self._event_count = 0
        self._maximum_base_errf_error = 0.0
        self._finalized = False

    def _state_codes(self, frame: pd.DataFrame) -> np.ndarray:
        keyed = frame.loc[:, list(STATE_FIELDS)].copy()
        keyed["_row_order"] = np.arange(len(keyed), dtype=np.int64)
        mapped = keyed.merge(
            self._state_lookup,
            on=list(STATE_FIELDS),
            how="left",
            validate="many_to_one",
            sort=False,
        ).sort_values("_row_order", kind="mergesort")
        if mapped["full_state_code"].isna().any():
            sample = mapped[mapped["full_state_code"].isna()][list(STATE_FIELDS)]
            raise RuntimeError(
                f"源事件状态不在冻结全集，样例={sample.head(3).to_dict('records')}"
            )
        return mapped["full_state_code"].to_numpy(dtype=np.int64)

    def add_stream(
        self,
        facts: pd.DataFrame,
        *,
        source_zone: str,
        predictor: str,
        horizon: int,
        seed: int,
        endpoint_addons: pd.DataFrame | None = None,
        artifact_identity: Mapping[str, Any] | None = None,
    ) -> None:
        if self._finalized:
            raise RuntimeError("充分统计已封闭，不得继续追加")
        zone = str(source_zone)
        current_predictor = str(predictor)
        current_horizon = int(horizon)
        current_seed = int(seed)
        stream_key = (zone, current_predictor, current_horizon, current_seed)
        if stream_key in self._streams:
            raise RuntimeError(f"源流重复追加: {stream_key}")
        if (
            zone not in self.source_zones
            or current_predictor not in self.predictors
            or current_horizon not in self.horizons
            or current_seed != self.seed
        ):
            raise ValueError(f"源流超出冻结范围: {stream_key}")
        if facts.empty:
            raise ValueError(f"源流为空: {stream_key}")
        if self.require_causal_source_audit and (
            not isinstance(artifact_identity, VerifiedCausalSourceAudit)
            or not artifact_identity.verified_by_formal_loader()
        ):
            raise RuntimeError(
                f"正式源流只接受load_causal_source_stream返回的验证capability: "
                f"{stream_key}"
            )
        identity = dict(artifact_identity or {})
        if self.require_causal_source_audit:
            required_audit = {
                "schema",
                "status",
                "strict_maturity_filter_applied",
                "mature_action_count",
                "raw_s06_event_count",
                "strict_mature_event_count",
                "s06_unmatured_event_count",
                "complete_case_event_id_sha256",
                "source_zone",
                "outer_heldout_zone",
                "predictor",
                "horizon",
                "seed",
                "leaf_hashes_verified",
                "leaf_sha256",
                "verified_paths",
                "outer_heldout_zone_used_as_source",
                "endpoint_root_seal_verified",
                "endpoint_root_manifest_sha256",
                "width_thresholds_sha256",
                "source_anchor_audit_sha256",
                "migration_manifest_sha256",
            }
            missing_audit = sorted(required_audit - set(identity))
            if missing_audit:
                raise RuntimeError(
                    f"正式源流缺少因果成熟审计: {stream_key}/{missing_audit}"
                )
            if (
                identity["schema"]
                != "TEST_CLARA_GEFCOM_6A_CAUSAL_SOURCE_STREAM_AUDIT_V1"
                or identity["status"] != "PASS"
                or not bool(identity["strict_maturity_filter_applied"])
                or int(identity["mature_action_count"]) != 4
                or str(identity["source_zone"]) != zone
                or str(identity["outer_heldout_zone"]) != self.heldout_zone
                or str(identity["predictor"]) != current_predictor
                or int(identity["horizon"]) != current_horizon
                or int(identity["seed"]) != current_seed
                or int(identity["strict_mature_event_count"]) != len(facts)
                or int(identity["raw_s06_event_count"])
                - int(identity["s06_unmatured_event_count"])
                != len(facts)
                or not str(identity["complete_case_event_id_sha256"])
                or not bool(identity["leaf_hashes_verified"])
                or bool(identity["outer_heldout_zone_used_as_source"])
                or not bool(identity["endpoint_root_seal_verified"])
                or any(
                    not str(identity.get(field) or "")
                    for field in (
                        "endpoint_root_manifest_sha256",
                        "width_thresholds_sha256",
                        "source_anchor_audit_sha256",
                        "migration_manifest_sha256",
                    )
                )
            ):
                raise RuntimeError(f"正式源流因果成熟审计失配: {stream_key}")
            expected_leaf_names = {
                "source_fact",
                "event_core",
                "candidate_intervals",
                "feedback_trace",
                "endpoint_addon",
                "width_thresholds",
            }
            leaf_sha256 = identity.get("leaf_sha256")
            verified_paths = identity.get("verified_paths")
            if (
                not isinstance(leaf_sha256, Mapping)
                or set(leaf_sha256) != expected_leaf_names
                or any(not _is_sha256(value) for value in leaf_sha256.values())
                or not isinstance(verified_paths, Mapping)
                or not expected_leaf_names.issubset(verified_paths)
                or not {
                    "endpoint_root_manifest",
                    "endpoint_source_anchor",
                }.issubset(verified_paths)
            ):
                raise RuntimeError(f"正式源流leaf/path审计不完整: {stream_key}")
            for leaf_name in expected_leaf_names:
                leaf_path = Path(str(verified_paths[leaf_name]))
                if not leaf_path.is_file() or _sha256_path(leaf_path) != str(
                    leaf_sha256[leaf_name]
                ):
                    raise RuntimeError(
                        f"正式源流leaf文件与审计SHA失配: "
                        f"{stream_key}/{leaf_name}"
                    )
            attested_paths = {
                "endpoint_root_manifest": "endpoint_root_manifest_sha256",
                "endpoint_source_anchor": "source_anchor_audit_sha256",
                "width_thresholds": "width_thresholds_sha256",
            }
            for path_name, sha_field in attested_paths.items():
                attested_path = Path(str(verified_paths[path_name]))
                if not attested_path.is_file() or _sha256_path(attested_path) != str(
                    identity[sha_field]
                ):
                    raise RuntimeError(
                        f"正式源流证明文件SHA失配: {stream_key}/{path_name}"
                    )
        frame = facts.copy()
        _validate_event_ids(frame, f"源流{stream_key}")
        frame["event_id"] = frame["event_id"].astype(str)
        if endpoint_addons is not None:
            addons = endpoint_addons.copy()
            _require_columns(addons, ("event_id",), "新增动作端点")
            _validate_event_ids(addons, f"新增动作端点{stream_key}")
            addons["event_id"] = addons["event_id"].astype(str)
            addon_columns = [
                "event_id",
                *(
                    f"{action}__{field}"
                    for action in self.actions[4:]
                    for field in (
                        "candidate_lower",
                        "candidate_upper",
                        "covered",
                        "tuwr_indicator",
                        "ard_value",
                    )
                ),
            ]
            _require_columns(addons, addon_columns, "新增动作端点")
            if addons["event_id"].duplicated().any():
                raise ValueError(f"新增动作端点事件重复: {stream_key}")
            frame = frame.merge(
                addons[addon_columns],
                on="event_id",
                how="left",
                validate="one_to_one",
                sort=False,
            )
        required = {
            "event_id",
            "zone_or_farm",
            "predictor",
            "horizon_steps",
            "seed",
            "target_coverage",
            "issue_timestamp",
            "target_after_maturity",
            "schedule_proxy",
            *STATE_FIELDS,
        }
        for action in self.actions:
            required.update(
                {
                    f"{action}__candidate_lower",
                    f"{action}__candidate_upper",
                    f"{action}__covered",
                    f"{action}__tuwr_indicator",
                    f"{action}__ard_value",
                }
            )
        _require_columns(frame, required, "六动作源事件")
        observed_event_hash = _canonical_digest(sorted(frame["event_id"].astype(str)))
        event_id_hash_verified = observed_event_hash == str(
            identity.get("complete_case_event_id_sha256", "")
        )
        if self.require_causal_source_audit and not event_id_hash_verified:
            raise RuntimeError(f"源流事件集哈希与因果loader审计失配: {stream_key}")
        identity_checks = {
            "zone_or_farm": zone,
            "predictor": current_predictor,
            "horizon_steps": current_horizon,
            "seed": current_seed,
        }
        for field, expected in identity_checks.items():
            observed = set(frame[field].astype(type(expected)).tolist())
            if observed != {expected}:
                raise ValueError(
                    f"源流身份失配: {stream_key}/{field}/{sorted(observed)}"
                )
        observed_coverages = set(frame["target_coverage"].astype(float))
        if observed_coverages != set(self.coverages):
            raise ValueError(
                f"源流覆盖率不完整: {stream_key}/{sorted(observed_coverages)}"
            )
        state_codes = self._state_codes(frame)
        zone_index = self.source_zones.index(zone)
        state_count = len(self.states)
        issue_us = _timestamp_us(frame["issue_timestamp"], "issue_timestamp")
        target = _finite(
            frame["target_after_maturity"].to_numpy(dtype=np.float64),
            "target_after_maturity",
        )
        schedule = _finite(
            frame["schedule_proxy"].to_numpy(dtype=np.float64), "schedule_proxy"
        )
        if np.any((target < 0.0) | (target > 1.0)):
            raise ValueError(f"成熟目标超出[0,1]: {stream_key}")
        # Preserve the frozen GEFCom endpoint contract: schedule/base_center is
        # an unconstrained finite point forecast and is intentionally scored
        # without clipping.  Real source streams contain both negative and
        # above-one values; only the mature target retains a [0, 1] domain gate.
        schedule_outside_unit_count = int(
            np.count_nonzero((schedule < 0.0) | (schedule > 1.0))
        )

        capacity_matrix = np.empty((len(frame), len(self.actions)), dtype=np.float64)
        miss_matrix = np.empty_like(capacity_matrix)
        covered_matrix = np.empty_like(capacity_matrix)
        tuwr_matrix = np.empty_like(capacity_matrix)
        ard_matrix = np.empty_like(capacity_matrix)
        reference_guardrail: np.ndarray | None = None
        cold = frame["rolling_state"].astype(str).eq("cold_start").to_numpy()
        for action_index, action in enumerate(self.actions):
            lower = frame[f"{action}__candidate_lower"].to_numpy(dtype=np.float64)
            upper = frame[f"{action}__candidate_upper"].to_numpy(dtype=np.float64)
            capacity, miss = endpoint_components(
                target=target,
                schedule=schedule,
                lower=lower,
                upper=upper,
                cadence_minutes=self.cadence_minutes,
            )
            observed_covered = frame[f"{action}__covered"].astype(bool).to_numpy()
            computed_covered = (lower <= target) & (target <= upper)
            if not np.array_equal(observed_covered, computed_covered):
                raise RuntimeError(f"覆盖标记与端点不一致: {stream_key}/{action}")
            tuwr = frame[f"{action}__tuwr_indicator"].to_numpy(dtype=np.float64)
            ard = frame[f"{action}__ard_value"].to_numpy(dtype=np.float64)
            if (
                not np.array_equal(np.isnan(tuwr), cold)
                or not np.array_equal(np.isnan(ard), cold)
                or not np.isfinite(tuwr[~cold]).all()
                or not np.isfinite(ard[~cold]).all()
                or np.any((tuwr[~cold] < 0.0) | (tuwr[~cold] > 1.0))
                or np.any((ard[~cold] < 0.0) | (ard[~cold] > 1.0))
            ):
                raise RuntimeError(
                    f"TUWR/ARD必须cold严格NaN、noncold有限且在[0,1]: "
                    f"{stream_key}/{action}"
                )
            guardrail = ~cold
            if reference_guardrail is None:
                reference_guardrail = guardrail
            elif not np.array_equal(reference_guardrail, guardrail):
                raise RuntimeError(f"六动作Guardrail缺失模式失配: {stream_key}")
            capacity_matrix[:, action_index] = capacity
            miss_matrix[:, action_index] = miss
            covered_matrix[:, action_index] = observed_covered.astype(np.float64)
            tuwr_matrix[:, action_index] = tuwr
            ard_matrix[:, action_index] = ard
            if f"{action}__errf" in frame.columns:
                base_prices = [row for row in self.prices if row.base_submission_price]
                if len(base_prices) == 1:
                    expected_errf = (
                        base_prices[0].capacity_weight * capacity
                        + base_prices[0].miss_weight * miss
                    )
                    observed_errf = frame[f"{action}__errf"].to_numpy(dtype=np.float64)
                    difference = float(np.max(np.abs(expected_errf - observed_errf)))
                    self._maximum_base_errf_error = max(
                        self._maximum_base_errf_error, difference
                    )
                    if not np.allclose(
                        expected_errf, observed_errf, atol=1e-10, rtol=1e-10
                    ):
                        raise RuntimeError(
                            f"基准价ERRF分量回归失败: {stream_key}/{action}/{difference}"
                        )
        if reference_guardrail is None:
            raise RuntimeError("六动作Guardrail掩码未构建")
        if not np.array_equal(~reference_guardrail, cold):
            raise RuntimeError(f"可靠性缺失与cold_start状态不同步: {stream_key}")

        self.arrays["event_count"][zone_index] += np.bincount(
            state_codes, minlength=state_count
        ).astype(np.int64)
        np.minimum.at(self.arrays["issue_min_us"][zone_index], state_codes, issue_us)
        np.maximum.at(self.arrays["issue_max_us"][zone_index], state_codes, issue_us)
        if reference_guardrail.any():
            guardrail_codes = state_codes[reference_guardrail]
            guardrail_times = issue_us[reference_guardrail]
            self.arrays["guardrail_count"][zone_index] += np.bincount(
                guardrail_codes, minlength=state_count
            ).astype(np.int64)
            np.minimum.at(
                self.arrays["guardrail_issue_min_us"][zone_index],
                guardrail_codes,
                guardrail_times,
            )
            np.maximum.at(
                self.arrays["guardrail_issue_max_us"][zone_index],
                guardrail_codes,
                guardrail_times,
            )
        for action_index in range(len(self.actions)):
            self.arrays["capacity_sum"][action_index, zone_index] += np.bincount(
                state_codes,
                weights=capacity_matrix[:, action_index],
                minlength=state_count,
            )
            self.arrays["miss_sum"][action_index, zone_index] += np.bincount(
                state_codes,
                weights=miss_matrix[:, action_index],
                minlength=state_count,
            )
            self.arrays["covered_sum"][action_index, zone_index] += np.bincount(
                state_codes,
                weights=covered_matrix[:, action_index],
                minlength=state_count,
            )
            if reference_guardrail.any():
                guardrail_codes = state_codes[reference_guardrail]
                self.arrays["guardrail_covered_sum"][
                    action_index, zone_index
                ] += np.bincount(
                    guardrail_codes,
                    weights=covered_matrix[reference_guardrail, action_index],
                    minlength=state_count,
                )
                self.arrays["tuwr_sum"][action_index, zone_index] += np.bincount(
                    guardrail_codes,
                    weights=tuwr_matrix[reference_guardrail, action_index],
                    minlength=state_count,
                )
                self.arrays["ard_sum"][action_index, zone_index] += np.bincount(
                    guardrail_codes,
                    weights=ard_matrix[reference_guardrail, action_index],
                    minlength=state_count,
                )
        for price_index, price in enumerate(self.prices):
            losses = (
                price.capacity_weight * capacity_matrix
                + price.miss_weight * miss_matrix
            )
            selected = np.argmin(losses, axis=1)
            for action_index in range(len(self.actions)):
                mask = selected == action_index
                if mask.any():
                    self.cart_best_count[
                        price_index, action_index, zone_index
                    ] += np.bincount(
                        state_codes[mask], minlength=state_count
                    ).astype(np.int64)
        self._streams.add(stream_key)
        self._event_count += len(frame)
        self._stream_rows.append(
            {
                "source_zone": zone,
                "predictor": current_predictor,
                "horizon": current_horizon,
                "seed": current_seed,
                "event_count": len(frame),
                "coverage_count": len(observed_coverages),
                "guardrail_event_count": int(reference_guardrail.sum()),
                "target_unit_interval_verified": True,
                "schedule_proxy_finite_verified": True,
                "schedule_proxy_minimum": float(np.min(schedule)),
                "schedule_proxy_maximum": float(np.max(schedule)),
                "schedule_proxy_outside_unit_interval_count": (
                    schedule_outside_unit_count
                ),
                "artifact_identity_sha256": _canonical_digest(identity),
                "artifact_identity": json.dumps(
                    identity, ensure_ascii=False, sort_keys=True, default=str
                ),
                "strict_maturity_filter_applied": bool(
                    identity.get("strict_maturity_filter_applied", False)
                ),
                "leaf_hashes_verified": bool(identity.get("leaf_hashes_verified", False)),
                "event_id_hash_verified": bool(event_id_hash_verified),
                "outer_heldout_zone_used_as_source": bool(
                    identity.get("outer_heldout_zone_used_as_source", True)
                ),
                "endpoint_root_seal_verified": bool(
                    identity.get("endpoint_root_seal_verified", False)
                ),
                "endpoint_root_manifest_sha256": str(
                    identity.get("endpoint_root_manifest_sha256", "")
                ),
                "width_thresholds_sha256": str(
                    identity.get("width_thresholds_sha256", "")
                ),
                "source_anchor_audit_sha256": str(
                    identity.get("source_anchor_audit_sha256", "")
                ),
                "migration_manifest_sha256": str(
                    identity.get("migration_manifest_sha256", "")
                ),
                "status": "PASS",
            }
        )

    def finalize(self) -> SixActionSufficientStats:
        if self._finalized:
            raise RuntimeError("充分统计builder不允许重复finalize")
        expected = {
            (zone, predictor, horizon, self.seed)
            for zone in self.source_zones
            for predictor in self.predictors
            for horizon in self.horizons
        }
        missing = sorted(expected - self._streams)
        unexpected = sorted(self._streams - expected)
        if missing or unexpected:
            raise RuntimeError(
                f"源流矩阵不完整: missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        event_counts = self.arrays["event_count"]
        best_totals = self.cart_best_count.sum(axis=1)
        expected_totals = np.broadcast_to(
            event_counts[None, :, :], best_totals.shape
        )
        if not np.array_equal(best_totals, expected_totals):
            raise RuntimeError("CART五价最优动作计数不闭合")
        if int(event_counts.sum()) != self._event_count:
            raise RuntimeError("充分统计事件数不守恒")
        self._finalized = True
        stream_audit = pd.DataFrame(self._stream_rows).sort_values(
            ["source_zone", "predictor", "horizon"], kind="mergesort"
        ).reset_index(drop=True)
        stream_identity_records = stream_audit[
            [
                "source_zone",
                "predictor",
                "horizon",
                "seed",
                "event_count",
                "target_unit_interval_verified",
                "schedule_proxy_finite_verified",
                "schedule_proxy_minimum",
                "schedule_proxy_maximum",
                "schedule_proxy_outside_unit_interval_count",
                "artifact_identity_sha256",
                "endpoint_root_manifest_sha256",
                "width_thresholds_sha256",
                "source_anchor_audit_sha256",
                "migration_manifest_sha256",
            ]
        ].to_dict("records")
        provenance_sets = {
            field: sorted(
                {
                    str(value)
                    for value in stream_audit[field].tolist()
                    if str(value)
                }
            )
            for field in (
                "endpoint_root_manifest_sha256",
                "width_thresholds_sha256",
                "migration_manifest_sha256",
            )
        }
        if self.require_causal_source_audit and any(
            len(provenance_sets[field]) != 1 for field in provenance_sets
        ):
            raise RuntimeError(
                "正式compact source的endpoint-root/threshold/migration身份不唯一"
            )
        array_content, sufficient_statistics_content_sha256 = (
            _statistics_array_content_identity(self.arrays, self.cart_best_count)
        )
        state_universe_content_sha256 = _state_universe_content_sha256(self.states)
        identity = {
            "schema": SCHEMA,
            "heldout_zone": self.heldout_zone,
            "source_zones": list(self.source_zones),
            "seed": self.seed,
            "horizons": list(self.horizons),
            "predictors": list(self.predictors),
            "coverages": list(self.coverages),
            "actions": list(self.actions),
            "prices": [row.as_dict() for row in self.prices],
            "stream_count": len(stream_audit),
            "event_count": self._event_count,
            "heldout_zone_used_in_fit": False,
            "price_independent_event_scan_count": 1,
            "cart_labels_recomputed_per_price": True,
            "causal_source_gate_required": bool(self.require_causal_source_audit),
            "strict_causal_stream_count": int(
                stream_audit["strict_maturity_filter_applied"].astype(bool).sum()
            ),
            "stream_artifact_identities": stream_identity_records,
            "stream_artifact_identity_set_sha256": _canonical_digest(
                stream_identity_records
            ),
            "endpoint_root_manifest_sha256_set": provenance_sets[
                "endpoint_root_manifest_sha256"
            ],
            "width_thresholds_sha256_set": provenance_sets[
                "width_thresholds_sha256"
            ],
            "migration_manifest_sha256_set": provenance_sets[
                "migration_manifest_sha256"
            ],
            "sufficient_statistics_array_identities": array_content,
            "sufficient_statistics_content_sha256": (
                sufficient_statistics_content_sha256
            ),
            "state_universe_content_sha256": state_universe_content_sha256,
        }
        audit = {
            **identity,
            "identity_sha256": _canonical_digest(identity),
            "maximum_base_price_errf_regression_error": float(
                self._maximum_base_errf_error
            ),
            "array_bytes": int(
                sum(values.nbytes for values in self.arrays.values())
                + self.cart_best_count.nbytes
            ),
            "status": "PASS",
        }
        immutable_arrays = {
            name: values.copy() for name, values in self.arrays.items()
        }
        for values in immutable_arrays.values():
            values.setflags(write=False)
        immutable_cart_best_count = self.cart_best_count.copy()
        immutable_cart_best_count.setflags(write=False)
        return SixActionSufficientStats(
            schema=SCHEMA,
            actions=self.actions,
            prices=self.prices,
            heldout_zone=self.heldout_zone,
            source_zones=self.source_zones,
            seed=self.seed,
            horizons=self.horizons,
            predictors=self.predictors,
            coverages=self.coverages,
            states=self.states.copy(deep=True),
            arrays=immutable_arrays,
            cart_best_count=immutable_cart_best_count,
            stream_audit=stream_audit,
            audit=audit,
        )


def assert_formal_gefcom_identity(stats: SixActionSufficientStats) -> None:
    _assert_compact_content_current(stats)
    expected_sources = set(FORMAL_ZONES) - {stats.heldout_zone}
    failures: list[str] = []
    if stats.heldout_zone not in FORMAL_ZONES:
        failures.append("heldout_zone")
    if int(stats.seed) not in {0, 1, 2}:
        failures.append("seed")
    if stats.actions != SIX_ACTIONS:
        failures.append("actions")
    if stats.horizons != FORMAL_HORIZONS:
        failures.append("horizons")
    if stats.predictors != FORMAL_PREDICTORS:
        failures.append("predictors")
    if stats.coverages != FORMAL_COVERAGES:
        failures.append("coverages")
    if set(stats.source_zones) != expected_sources or len(stats.source_zones) != 9:
        failures.append("source_zones")
    try:
        assert_formal_price_grid(stats.prices)
    except (ValueError, RuntimeError):
        failures.append("prices")
    if not bool(stats.audit.get("causal_source_gate_required")):
        failures.append("causal_source_gate")
    if int(stats.audit.get("stream_count", -1)) != 180:
        failures.append("stream_count")
    if any(
        len(stats.audit.get(field, [])) != 1
        for field in (
            "endpoint_root_manifest_sha256_set",
            "width_thresholds_sha256_set",
            "migration_manifest_sha256_set",
        )
    ):
        failures.append("provenance_identity_sets")
    if int(stats.audit.get("strict_causal_stream_count", -1)) != int(
        stats.audit.get("stream_count", -2)
    ):
        failures.append("strict_causal_stream_count")
    required_stream_fields = {
        "status",
        "strict_maturity_filter_applied",
        "leaf_hashes_verified",
        "event_id_hash_verified",
        "outer_heldout_zone_used_as_source",
        "endpoint_root_seal_verified",
        "endpoint_root_manifest_sha256",
        "width_thresholds_sha256",
        "source_anchor_audit_sha256",
        "migration_manifest_sha256",
        "target_unit_interval_verified",
        "schedule_proxy_finite_verified",
        "schedule_proxy_minimum",
        "schedule_proxy_maximum",
        "schedule_proxy_outside_unit_interval_count",
    }
    if not required_stream_fields.issubset(stats.stream_audit.columns):
        failures.append("stream_audit_fields")
    elif not (
        stats.stream_audit["status"].eq("PASS").all()
        and stats.stream_audit["strict_maturity_filter_applied"].astype(bool).all()
        and stats.stream_audit["target_unit_interval_verified"].astype(bool).all()
        and stats.stream_audit["schedule_proxy_finite_verified"].astype(bool).all()
        and stats.stream_audit["leaf_hashes_verified"].astype(bool).all()
        and stats.stream_audit["event_id_hash_verified"].astype(bool).all()
        and (~stats.stream_audit["outer_heldout_zone_used_as_source"].astype(bool)).all()
        and stats.stream_audit["endpoint_root_seal_verified"].astype(bool).all()
    ):
        failures.append("stream_audit_verification")
    else:
        observed_streams = set(
            stats.stream_audit[
                ["source_zone", "predictor", "horizon", "seed"]
            ].itertuples(index=False, name=None)
        )
        expected_streams = {
            (zone, predictor, horizon, stats.seed)
            for zone in stats.source_zones
            for predictor in FORMAL_PREDICTORS
            for horizon in FORMAL_HORIZONS
        }
        sha_fields = (
            "artifact_identity_sha256",
            "endpoint_root_manifest_sha256",
            "width_thresholds_sha256",
            "source_anchor_audit_sha256",
            "migration_manifest_sha256",
        )
        if observed_streams != expected_streams or any(
            not stats.stream_audit[field].map(_is_sha256).all()
            for field in sha_fields
        ):
            failures.append("stream_axis_or_sha_closure")
    if failures:
        raise RuntimeError(f"正式GEFCom五时距六动作身份失败: {failures}")


@dataclass(frozen=True)
class ClaraPreparation:
    stats: SixActionSufficientStats
    contracts: FrozenContracts
    mappings: tuple[np.ndarray, ...]
    group_frames: tuple[pd.DataFrame, ...]
    profiles: dict[int, pd.DataFrame]
    audit: dict[str, Any]
    capability_identity: dict[str, Any] | None = None
    _capability: object | None = None


def _clara_preparation_capability_identity(
    preparation: ClaraPreparation,
) -> dict[str, Any]:
    payload = {
        "schema": "TEST_CLARA_GEFCOM_6A_FORMAL_CLARA_PREPARATION_CAPABILITY_V1",
        "compact_source_identity_sha256": str(
            preparation.stats.audit["identity_sha256"]
        ),
        "derived_contract_sha256": str(
            preparation.audit["derived_contract_sha256"]
        ),
        "v4_actual_configuration_sha256": str(
            preparation.audit["v4_actual_configuration_sha256"]
        ),
        "v4_adaptive_subcontract_sha256": str(
            preparation.audit["v4_adaptive_subcontract_sha256"]
        ),
        "mappings": [
            _ndarray_content_identity(values) for values in preparation.mappings
        ],
        "group_frames": [
            _dataframe_content_sha256(frame) for frame in preparation.group_frames
        ],
        "profiles": {
            str(key): _dataframe_content_sha256(frame)
            for key, frame in sorted(preparation.profiles.items())
        },
        "audit_sha256": _canonical_digest(preparation.audit),
    }
    return {**payload, "capability_sha256": _canonical_digest(payload)}


def _verify_formal_clara_preparation(
    preparation: ClaraPreparation,
    *,
    v4_config: Mapping[str, Any],
) -> dict[str, Any]:
    if preparation._capability is not _FORMAL_CLARA_PREPARATION_CAPABILITY:
        raise RuntimeError("formal fitter拒绝nonformal CLARA preparation")
    observed = _clara_preparation_capability_identity(preparation)
    if preparation.capability_identity != observed:
        raise RuntimeError("formal CLARA preparation在prepare后发生漂移")
    assert_formal_gefcom_identity(preparation.stats)
    v4 = validate_formal_v4_configuration(v4_config)
    derived = derived_six_action_contract_identity(preparation.contracts)
    if (
        not bool(preparation.audit.get("formal_identity"))
        or str(preparation.audit.get("v4_actual_configuration_sha256"))
        != str(v4["actual_configuration_sha256"])
        or str(preparation.audit.get("v4_adaptive_subcontract_sha256"))
        != str(v4["v4_adaptive_subcontract_sha256"])
        or str(preparation.audit.get("derived_contract_sha256"))
        != str(derived["derived_contract_sha256"])
    ):
        raise RuntimeError("formal CLARA preparation的V4/派生合同身份失配")
    return observed


@dataclass(frozen=True)
class ClaraFitResult:
    decisions: pd.DataFrame
    action_evidence: pd.DataFrame
    audit: dict[str, Any]


def prepare_clara_fit(
    stats: SixActionSufficientStats,
    *,
    contracts: FrozenContracts,
    v4_config: Mapping[str, Any],
    require_formal_identity: bool = True,
) -> ClaraPreparation:
    if require_formal_identity:
        assert_formal_gefcom_identity(stats)
        v4_identity = validate_formal_v4_configuration(v4_config)
        contract_identity = derived_six_action_contract_identity(contracts)
    else:
        nonformal_v4_contract = {
            "schema": "TEST_CLARA_6A_NONFORMAL_V4_CONFIGURATION_V1",
            "configuration": dict(v4_config),
        }
        v4_identity = {
            "actual_configuration_sha256": _canonical_digest(dict(v4_config)),
            "v4_adaptive_subcontract_sha256": _canonical_digest(
                nonformal_v4_contract
            ),
        }
        contract_identity = {
            "derived_contract_sha256": _canonical_digest(contracts.protocol)
        }
    if tuple(contracts.actions) != SIX_ACTIONS:
        raise ValueError("CLARA拟合合同未扩展为六动作")
    adaptive = v4_config.get("adaptive_support")
    if not isinstance(adaptive, Mapping):
        raise ValueError("V4配置缺少adaptive_support")
    candidates = tuple(int(value) for value in adaptive["n_min_candidates"])
    if not candidates or int(adaptive["n_min_no_candidate_rule"]) not in candidates:
        raise ValueError("V4 n_min候选或回退规则非法")
    mappings, group_frames = landscape_accelerator.factor_level_mappings(
        stats.states, contracts
    )
    # Support profiles depend only on counts/timestamps, not on price-weighted ERRF.
    reference_arrays = stats.arrays_for_price(stats.prices[0])
    reference_levels = landscape_accelerator.build_level_aggregates(
        reference_arrays, mappings, group_frames
    )
    profiles: dict[int, pd.DataFrame] = {}
    for n_min in candidates:
        current_contracts = adaptive_v4.parameterized_contracts(
            contracts,
            n_min=n_min,
            nu=int(adaptive["diagnostic_nu"]),
        )
        profiles[n_min] = landscape_accelerator.select_support_levels(
            stats.states,
            current_contracts,
            mappings,
            reference_levels,
        )
    audit = {
        "schema": SCHEMA,
        "status": "PASS",
        "heldout_zone": stats.heldout_zone,
        "seed": stats.seed,
        "source_zone_count": len(stats.source_zones),
        "source_event_count": int(stats.arrays["event_count"].sum()),
        "state_count": len(stats.states),
        "action_count": len(stats.actions),
        "n_min_candidates": list(candidates),
        "support_profile_reused_across_prices": True,
        "support_profile_price_independence_basis": "counts_and_timestamps_only",
        "formal_identity": bool(require_formal_identity),
        "v4_actual_configuration_sha256": v4_identity[
            "actual_configuration_sha256"
        ],
        "v4_adaptive_subcontract_sha256": v4_identity[
            "v4_adaptive_subcontract_sha256"
        ],
        "derived_contract_sha256": contract_identity["derived_contract_sha256"],
    }
    preparation = ClaraPreparation(
        stats=stats,
        contracts=contracts,
        mappings=tuple(mappings),
        group_frames=tuple(group_frames),
        profiles=profiles,
        audit=audit,
    )
    if not require_formal_identity:
        return preparation
    capability_identity = _clara_preparation_capability_identity(preparation)
    return ClaraPreparation(
        stats=preparation.stats,
        contracts=preparation.contracts,
        mappings=preparation.mappings,
        group_frames=preparation.group_frames,
        profiles=preparation.profiles,
        audit=preparation.audit,
        capability_identity=capability_identity,
        _capability=_FORMAL_CLARA_PREPARATION_CAPABILITY,
    )


def fit_price_conditioned_clara(
    prepared: ClaraPreparation,
    *,
    price: PriceSpec,
    v4_config: Mapping[str, Any],
) -> ClaraFitResult:
    """Fit one fully adaptive CLARA policy with all six actions in every rule."""

    price.validate()
    stats = prepared.stats
    if bool(prepared.audit.get("formal_identity")):
        _verify_formal_clara_preparation(prepared, v4_config=v4_config)
        v4_identity = validate_formal_v4_configuration(v4_config)
        if (
            v4_identity["actual_configuration_sha256"]
            != prepared.audit["v4_actual_configuration_sha256"]
            or v4_identity["v4_adaptive_subcontract_sha256"]
            != prepared.audit["v4_adaptive_subcontract_sha256"]
        ):
            raise RuntimeError("CLARA prepare/fit V4冻结配置身份失配")
        if (
            derived_six_action_contract_identity(prepared.contracts)[
                "derived_contract_sha256"
            ]
            != prepared.audit["derived_contract_sha256"]
        ):
            raise RuntimeError("CLARA prepare/fit六动作派生合同身份失配")
    if price.price_id not in {row.price_id for row in stats.prices}:
        raise ValueError(f"CLARA价格不在冻结网格: {price.price_id}")
    price_arrays = stats.arrays_for_price(price)
    levels = landscape_accelerator.build_level_aggregates(
        price_arrays,
        list(prepared.mappings),
        list(prepared.group_frames),
    )
    # The sealed adaptive helper keeps ACTIONS as a module constant.  Patch it only
    # under a process-local lock; formal workers are separate processes.
    with _ACTION_PATCH_LOCK:
        original_actions = adaptive_v4.ACTIONS
        try:
            adaptive_v4.ACTIONS = SIX_ACTIONS
            (
                _,
                evidence,
                selected_n_min,
                selected_nu,
                selected_margin,
                selected_nu_ratio,
            ) = adaptive_v4.adaptive_support_evidence(
                states=stats.states,
                contracts=prepared.contracts,
                mappings=list(prepared.mappings),
                levels=levels,
                profiles=prepared.profiles,
                config=dict(v4_config),
            )
            thresholds = adaptive_v4.adaptive_guardrail_thresholds(
                price.miss_to_capacity_ratio, dict(v4_config)
            )
            selected, empty, safe = adaptive_v4.select_actions(
                evidence, thresholds, prepared.contracts
            )
            profile = adaptive_v4.stitch_profile(
                prepared.profiles, selected_n_min
            )
        finally:
            adaptive_v4.ACTIONS = original_actions
    ordered_evidence = evidence.sort_values(
        ["full_state_code", "action_order"], kind="mergesort"
    ).reset_index(drop=True)
    if len(ordered_evidence) != len(stats.states) * len(SIX_ACTIONS):
        raise RuntimeError("CLARA六动作证据行数失配")
    ordered_evidence["guardrail_pass"] = safe.reshape(-1)
    ordered_evidence.insert(0, "price_id", price.price_id)
    ordered_evidence.insert(
        1, "miss_to_capacity_ratio", price.miss_to_capacity_ratio
    )
    decisions = profile[
        ["full_state_code", *STATE_FIELDS, "support_backoff_level", "support_backoff_name"]
    ].copy()
    decisions["selected_action"] = np.asarray(SIX_ACTIONS, dtype=object)[selected]
    decisions["selected_n_min"] = selected_n_min
    decisions["selected_nu"] = selected_nu
    decisions["risk_margin_se_ratio"] = selected_margin
    decisions["nu_signal_noise_ratio"] = selected_nu_ratio
    decisions["guardrail_empty_fallback"] = empty
    decisions["selected_action_guardrail_pass"] = safe[
        np.arange(len(selected)), selected
    ]
    decisions.insert(0, "miss_to_capacity_ratio", price.miss_to_capacity_ratio)
    decisions.insert(0, "price_id", price.price_id)
    action_counts = np.bincount(selected, minlength=len(SIX_ACTIONS))
    audit = {
        "schema": SCHEMA,
        "status": "PASS",
        "method": "CLARA_6A",
        "heldout_zone": stats.heldout_zone,
        "heldout_zone_used_in_fit": False,
        "seed": stats.seed,
        "price": price.as_dict(),
        "source_zones": list(stats.source_zones),
        "training_horizons": list(stats.horizons),
        "state_count": len(decisions),
        "actions": list(SIX_ACTIONS),
        "adaptive_n_min_top_two_action_scope": len(SIX_ACTIONS),
        "adaptive_nu_median_action_scope": len(SIX_ACTIONS),
        "guardrail_safe_set_action_scope": len(SIX_ACTIONS),
        "four_action_then_append_shortcut_used": False,
        "price_conditioned_risk": True,
        "adaptive_guardrail_thresholds": thresholds,
        "v4_actual_configuration_sha256": prepared.audit[
            "v4_actual_configuration_sha256"
        ],
        "v4_adaptive_subcontract_sha256": prepared.audit[
            "v4_adaptive_subcontract_sha256"
        ],
        "derived_contract_sha256": prepared.audit["derived_contract_sha256"],
        "n_min_counts": {
            str(value): int((selected_n_min == int(value)).sum())
            for value in sorted(set(selected_n_min.tolist()))
        },
        "nu_counts": {
            str(value): int((selected_nu == int(value)).sum())
            for value in sorted(set(selected_nu.tolist()))
        },
        "guardrail_empty_state_count": int(empty.sum()),
        "selected_action_state_counts": {
            action: int(action_counts[index])
            for index, action in enumerate(SIX_ACTIONS)
        },
    }
    return ClaraFitResult(
        decisions=decisions,
        action_evidence=ordered_evidence,
        audit=audit,
    )


def fit_all_price_conditioned_clara(
    prepared: ClaraPreparation,
    *,
    v4_config: Mapping[str, Any],
) -> dict[str, ClaraFitResult]:
    return {
        price.price_id: fit_price_conditioned_clara(
            prepared, price=price, v4_config=v4_config
        )
        for price in prepared.stats.prices
    }


@dataclass(frozen=True)
class CartFitResult:
    selector: ExtendedCartSelector
    compact_statistics: pd.DataFrame
    audit: dict[str, Any]


def fit_price_conditioned_cart(
    stats: SixActionSufficientStats,
    *,
    price: PriceSpec,
    contracts: FrozenContracts,
    config: Mapping[str, Any],
    require_formal_identity: bool = True,
) -> CartFitResult:
    """Refit CART from exact eventwise six-action labels for one price."""

    if require_formal_identity:
        assert_formal_gefcom_identity(stats)
        contract_identity = derived_six_action_contract_identity(contracts)
    else:
        contract_identity = {
            "derived_contract_sha256": _canonical_digest(contracts.protocol)
        }
    if tuple(contracts.actions) != SIX_ACTIONS:
        raise ValueError("CART拟合合同未扩展为六动作")
    actual_cart_configuration = dict(config)
    if require_formal_identity:
        frozen_cart_configuration = frozen_zone_baseline_contract(
            stats.heldout_zone, "CARTBestAction"
        )
        if actual_cart_configuration != dict(
            frozen_cart_configuration["configuration"]
        ):
            raise RuntimeError("CART实际配置与持出区S07 registry冻结行失配")
    else:
        nonformal_payload = {
            "schema": "TEST_CLARA_6A_NONFORMAL_CART_CONFIGURATION_V1",
            "outer_heldout_zone": stats.heldout_zone,
            "baseline_id": "CARTBestAction",
            "configuration": actual_cart_configuration,
        }
        frozen_cart_configuration = {
            **nonformal_payload,
            "frozen_configuration_sha256": _canonical_digest(nonformal_payload),
        }
    price_index = stats.price_index(price.price_id)
    price.validate()
    stored_price = stats.prices[price_index]
    if price_sha256(price) != price_sha256(stored_price):
        raise ValueError(
            f"CART价格ID与充分统计中的ratio/theta/base身份失配: "
            f"{price.price_id}"
        )
    fitting_config = dict(config)
    random_state = int(fitting_config.pop("random_state", 0))
    if random_state != 0:
        raise ValueError("CART必须保持冻结随机种子0")
    validate_frozen_selector_config(
        contracts=contracts,
        baseline_id="CARTBestAction",
        config=fitting_config,
    )
    event_count = stats.arrays["event_count"].sum(axis=0).astype(np.int64)
    best_counts = stats.cart_best_count[price_index].sum(axis=1).astype(np.int64)
    if not np.array_equal(best_counts.sum(axis=0), event_count):
        raise RuntimeError(f"CART逐价标签计数不闭合: {price.price_id}")
    observed = event_count > 0
    collapsed = stats.states.loc[observed, list(STATE_FIELDS)].copy().reset_index(
        drop=True
    )
    collapsed["event_count"] = event_count[observed]
    for action_index, action in enumerate(SIX_ACTIONS):
        collapsed[f"{action}__best_count"] = best_counts[action_index, observed]
    encoder, _, feature_matrix = _encoded_compact_states(
        contracts=contracts,
        collapsed=collapsed,
    )
    class_counts = np.column_stack(
        [
            collapsed[f"{action}__best_count"].to_numpy(dtype=np.int64)
            for action in SIX_ACTIONS
        ]
    )
    features, labels, sample_weights = _capped_classification_expansion(
        feature_matrix=feature_matrix,
        class_counts_by_state=class_counts,
        class_labels=SIX_ACTIONS,
        min_samples_leaf=int(fitting_config["min_samples_leaf"]),
        balanced=fitting_config["class_weight"] == "balanced",
    )
    classifier = DecisionTreeClassifier(
        max_depth=fitting_config["max_depth"],
        min_samples_leaf=int(fitting_config["min_samples_leaf"]),
        class_weight=None,
        random_state=0,
    )
    classifier.fit(features, labels, sample_weight=sample_weights)
    selector = ExtendedCartSelector(
        encoder=encoder,
        classifier=classifier,
        actions=SIX_ACTIONS,
        configuration={
            "max_depth": fitting_config["max_depth"],
            "min_samples_leaf": int(fitting_config["min_samples_leaf"]),
            "class_weight": fitting_config["class_weight"],
            "random_state": 0,
        },
    )
    audit = {
        "schema": SCHEMA,
        "status": "PASS",
        "method": "CART_6A",
        "heldout_zone": stats.heldout_zone,
        "heldout_zone_used_in_fit": False,
        "seed": stats.seed,
        "price": price.as_dict(),
        "actions": list(SIX_ACTIONS),
        "eventwise_labels_recomputed_for_price": True,
        "source_event_count": int(event_count.sum()),
        "observed_compact_state_count": len(collapsed),
        "expanded_training_row_count": int(features.shape[0]),
        "feature_count": int(features.shape[1]),
        "observed_classes": [str(value) for value in classifier.classes_],
        "tree_depth": int(classifier.get_depth()),
        "tree_leaf_count": int(classifier.get_n_leaves()),
        "configuration": selector.configuration,
        "actual_configuration_sha256": _canonical_digest(
            actual_cart_configuration
        ),
        "frozen_zone_configuration_sha256": frozen_cart_configuration[
            "frozen_configuration_sha256"
        ],
        "derived_contract_sha256": contract_identity["derived_contract_sha256"],
        "best_action_counts": {
            action: int(best_counts[index].sum())
            for index, action in enumerate(SIX_ACTIONS)
        },
    }
    return CartFitResult(
        selector=selector,
        compact_statistics=collapsed,
        audit=audit,
    )


def fit_all_price_conditioned_cart(
    stats: SixActionSufficientStats,
    *,
    contracts: FrozenContracts,
    config: Mapping[str, Any],
    require_formal_identity: bool = True,
) -> dict[str, CartFitResult]:
    return {
        price.price_id: fit_price_conditioned_cart(
            stats,
            price=price,
            contracts=contracts,
            config=config,
            require_formal_identity=require_formal_identity,
        )
        for price in stats.prices
    }


@dataclass(frozen=True)
class PriceFitChildResult:
    price: PriceSpec
    clara: ClaraFitResult
    cart: CartFitResult
    identity: dict[str, Any]
    audit: dict[str, Any]
    capability_identity: dict[str, Any] | None = None
    _capability: object | None = None


def _fit_result_capability_identity(
    *,
    price: PriceSpec,
    clara: ClaraFitResult,
    cart: CartFitResult,
    identity: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": "TEST_CLARA_GEFCOM_6A_FORMAL_FIT_RESULT_CAPABILITY_V1",
        "fit_child_identity_sha256": str(identity["fit_child_identity_sha256"]),
        "compact_source_identity_sha256": str(
            identity["compact_source_identity_sha256"]
        ),
        "price_sha256": price_sha256(price),
        "clara_decisions_sha256": _dataframe_content_sha256(clara.decisions),
        "clara_action_evidence_sha256": _dataframe_content_sha256(
            clara.action_evidence
        ),
        "clara_audit_sha256": _canonical_digest(clara.audit),
        "cart_selector_sha256": _cart_selector_content_sha256(cart.selector),
        "cart_compact_statistics_sha256": _dataframe_content_sha256(
            cart.compact_statistics
        ),
        "cart_audit_sha256": _canonical_digest(cart.audit),
        "child_audit_sha256": _canonical_digest(dict(audit)),
    }
    return {**payload, "capability_sha256": _canonical_digest(payload)}


def verify_formal_fit_result_capability(child: PriceFitChildResult) -> dict[str, Any]:
    """Verify that the formal fitter issued this exact in-memory policy result."""

    if child._capability is not _FORMAL_FIT_RESULT_CAPABILITY:
        raise RuntimeError("formal fit artifact只接受core fitter签发的capability")
    observed = _fit_result_capability_identity(
        price=child.price,
        clara=child.clara,
        cart=child.cart,
        identity=child.identity,
        audit=child.audit,
    )
    if child.capability_identity != observed:
        raise RuntimeError("formal fit result capability内容重算失配")
    return observed


def fit_price_conditioned_selectors(
    prepared: ClaraPreparation,
    *,
    price: PriceSpec,
    v4_config: Mapping[str, Any],
    cart_config: Mapping[str, Any],
    require_formal_identity: bool = True,
) -> PriceFitChildResult:
    """Fit and identify one independently sealable ``zone x seed x price`` child."""

    if require_formal_identity:
        _verify_formal_clara_preparation(prepared, v4_config=v4_config)
    clara = fit_price_conditioned_clara(
        prepared,
        price=price,
        v4_config=v4_config,
    )
    cart = fit_price_conditioned_cart(
        prepared.stats,
        price=price,
        contracts=prepared.contracts,
        config=cart_config,
        require_formal_identity=require_formal_identity,
    )
    if clara.audit["derived_contract_sha256"] != cart.audit[
        "derived_contract_sha256"
    ]:
        raise RuntimeError("CLARA/CART六动作派生合同身份不一致")
    identity = fit_price_child_identity(
        stats=prepared.stats,
        price=price,
        clara_configuration_sha256=_canonical_digest(dict(v4_config)),
        cart_configuration_sha256=_canonical_digest(dict(cart_config)),
        clara_v4_adaptive_subcontract_sha256=str(
            clara.audit["v4_adaptive_subcontract_sha256"]
        ),
        cart_frozen_zone_configuration_sha256=str(
            cart.audit["frozen_zone_configuration_sha256"]
        ),
        derived_contract_sha256=str(clara.audit["derived_contract_sha256"]),
    )
    audit = {
        "schema": SCHEMA,
        "status": "PASS",
        "heldout_zone": prepared.stats.heldout_zone,
        "seed": prepared.stats.seed,
        "price_id": price.price_id,
        "price_sha256": price_sha256(price),
        "action_library_sha256": action_library_sha256(),
        "fit_child_identity_sha256": identity["fit_child_identity_sha256"],
        "clara_v4_adaptive_subcontract_sha256": clara.audit[
            "v4_adaptive_subcontract_sha256"
        ],
        "cart_frozen_zone_configuration_sha256": cart.audit[
            "frozen_zone_configuration_sha256"
        ],
        "clara_state_count": len(clara.decisions),
        "cart_source_event_count": int(cart.audit["source_event_count"]),
        "price_independent_compact_reused": True,
        "cross_price_policy_state_shared": False,
        "atomic_commit_unit": "heldout_zone_seed_price",
    }
    capability_identity = None
    capability = None
    if require_formal_identity:
        capability_identity = _fit_result_capability_identity(
            price=price,
            clara=clara,
            cart=cart,
            identity=identity,
            audit=audit,
        )
        capability = _FORMAL_FIT_RESULT_CAPABILITY
    return PriceFitChildResult(
        price=price,
        clara=clara,
        cart=cart,
        identity=identity,
        audit=audit,
        capability_identity=capability_identity,
        _capability=capability,
    )


def fit_all_price_conditioned_selectors(
    prepared: ClaraPreparation,
    *,
    v4_config: Mapping[str, Any],
    cart_config: Mapping[str, Any],
    require_formal_identity: bool = True,
    child_sink: Callable[[PriceFitChildResult], None] | None = None,
    retain_children: bool = True,
) -> dict[str, PriceFitChildResult]:
    """Fit five independent price children from one reusable compact parent."""

    if not retain_children and child_sink is None:
        raise ValueError("不保留fit children时必须提供child_sink")
    children: dict[str, PriceFitChildResult] = {}
    for price in prepared.stats.prices:
        child = fit_price_conditioned_selectors(
            prepared,
            price=price,
            v4_config=v4_config,
            cart_config=cart_config,
            require_formal_identity=require_formal_identity,
        )
        if child_sink is not None:
            child_sink(child)
        if retain_children:
            children[price.price_id] = child
    return children


def fit_parent_identity(
    stats: SixActionSufficientStats,
    child_identity_sha256_by_price: Mapping[str, str],
) -> dict[str, Any]:
    """Seal a parent only after every configured price child is present."""

    expected = {row.price_id for row in stats.prices}
    observed = {str(value) for value in child_identity_sha256_by_price}
    if observed != expected:
        raise RuntimeError(
            f"fit parent不得在五价child未齐时封印: missing={sorted(expected-observed)}"
        )
    children = {
        price_id: str(child_identity_sha256_by_price[price_id])
        for price_id in sorted(expected)
    }
    if any(not value for value in children.values()):
        raise ValueError("fit parent含空child SHA")
    payload = {
        "schema": "TEST_CLARA_GEFCOM_6A_FIT_PARENT_IDENTITY_V1",
        "heldout_zone": stats.heldout_zone,
        "seed": stats.seed,
        "compact_source_identity_sha256": str(stats.audit["identity_sha256"]),
        "action_library_sha256": action_library_sha256(),
        "price_child_identity_sha256": children,
        "price_child_count": len(children),
    }
    return {**payload, "fit_parent_identity_sha256": _canonical_digest(payload)}


def predict_clara_actions(
    events: pd.DataFrame,
    decisions: pd.DataFrame,
) -> np.ndarray:
    _require_columns(events, ("event_id", *STATE_FIELDS), "CLARA目标事件")
    _require_columns(decisions, (*STATE_FIELDS, "selected_action"), "CLARA状态决策")
    frame = events.loc[:, ["event_id", *STATE_FIELDS]].copy()
    frame["_row_order"] = np.arange(len(frame), dtype=np.int64)
    lookup = decisions.loc[:, [*STATE_FIELDS, "selected_action"]].drop_duplicates(
        list(STATE_FIELDS)
    )
    mapped = frame.merge(
        lookup,
        on=list(STATE_FIELDS),
        how="left",
        validate="many_to_one",
        sort=False,
    ).sort_values("_row_order", kind="mergesort")
    if mapped["selected_action"].isna().any():
        sample = mapped[mapped["selected_action"].isna()][list(STATE_FIELDS)]
        raise RuntimeError(
            f"CLARA目标状态映射缺失，样例={sample.head(3).to_dict('records')}"
        )
    selected = mapped["selected_action"].astype(str).to_numpy()
    invalid = sorted(set(selected) - set(SIX_ACTIONS))
    if invalid:
        raise RuntimeError(f"CLARA预测产生六动作库外值: {invalid}")
    return selected


def predict_cart_actions(
    events: pd.DataFrame,
    result: CartFitResult | ExtendedCartSelector,
) -> tuple[np.ndarray, np.ndarray]:
    selector = result.selector if isinstance(result, CartFitResult) else result
    return predict_cart_local_actions(events, selector)


@dataclass(frozen=True)
class CartEventActionResult:
    selected_actions: np.ndarray
    scores: np.ndarray
    capability_identity: dict[str, Any]
    _capability: object | None = None


def _cart_event_action_capability_identity(
    result: CartEventActionResult,
    *,
    ordered_event_ids: Sequence[str],
    selector_sha256: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": "TEST_CLARA_GEFCOM_6A_CART_EVENT_ACTION_CAPABILITY_V1",
        "ordered_event_id_sha256": _canonical_digest(
            [str(value) for value in ordered_event_ids]
        ),
        "event_count": len(ordered_event_ids),
        "selected_actions_sha256": _canonical_digest(
            _ndarray_content_identity(
                np.asarray(result.selected_actions, dtype="U64")
            )
        ),
        "scores_sha256": _canonical_digest(
            _ndarray_content_identity(np.asarray(result.scores, dtype=np.float64))
        ),
    }
    if selector_sha256 is not None:
        payload["cart_selector_sha256"] = str(selector_sha256)
    return {**payload, "capability_sha256": _canonical_digest(payload)}


def verify_cart_event_action_capability(
    result: CartEventActionResult,
    *,
    ordered_event_ids: Sequence[str],
) -> dict[str, Any]:
    if result._capability is not _FORMAL_CART_EVENT_ACTION_CAPABILITY:
        raise RuntimeError("formal CART评估缺core predictor capability")
    expected = dict(result.capability_identity)
    if not _is_sha256(expected.get("cart_selector_sha256")):
        raise RuntimeError("CART event capability未绑定selector内容")
    observed = _cart_event_action_capability_identity(
        result,
        ordered_event_ids=ordered_event_ids,
        selector_sha256=str(expected["cart_selector_sha256"]),
    )
    if observed != expected:
        raise RuntimeError("CART event capability内容重算失配")
    return observed


def predict_cart_actions_with_capability(
    events: pd.DataFrame,
    result: CartFitResult | ExtendedCartSelector,
) -> CartEventActionResult:
    selector = result.selector if isinstance(result, CartFitResult) else result
    selected, scores = predict_cart_local_actions(events, selector)
    provisional = CartEventActionResult(
        selected_actions=np.asarray(selected, dtype=object).astype(str),
        scores=np.asarray(scores, dtype=np.float64),
        capability_identity={},
    )
    identity = _cart_event_action_capability_identity(
        provisional,
        ordered_event_ids=events["event_id"].astype(str).tolist(),
        selector_sha256=_cart_selector_content_sha256(selector),
    )
    return CartEventActionResult(
        selected_actions=provisional.selected_actions,
        scores=provisional.scores,
        capability_identity=identity,
        _capability=_FORMAL_CART_EVENT_ACTION_CAPABILITY,
    )


def event_id_signature(
    event_ids: pd.Series | Sequence[str],
    *,
    include_canonical_sha256: bool = False,
) -> dict[str, Any]:
    """Order-independent event multiset signature for conservation gates."""

    raw = event_ids.copy() if isinstance(event_ids, pd.Series) else pd.Series(
        list(event_ids), dtype="object"
    )
    if raw.isna().any():
        raise ValueError("event signature含空event_id")
    series = raw.astype(str)
    if series.str.strip().eq("").any() or series.duplicated().any():
        raise ValueError("event signature要求event_id非空且唯一")
    values = pd.util.hash_pandas_object(
        series, index=False, categorize=False
    ).to_numpy(dtype=np.uint64)
    modulus = 1 << 64
    total = int(sum(int(value) for value in values) % modulus)
    square = int(
        sum((int(value) * int(value)) % modulus for value in values) % modulus
    )
    xor = 0
    for value in values:
        xor ^= int(value)
    result: dict[str, Any] = {
        "event_count": len(values),
        "hash_sum_u64": total,
        "hash_xor_u64": xor,
        "hash_square_sum_u64": square,
    }
    if include_canonical_sha256:
        result["canonical_sorted_event_id_sha256"] = _canonical_digest(
            sorted(series.tolist())
        )
    return result


@dataclass(frozen=True)
class LinUCBReplayResult:
    decisions_by_price: dict[str, pd.DataFrame]
    audit_by_price: dict[str, dict[str, Any]]
    audit: dict[str, Any]
    capability_identity_by_price: dict[str, dict[str, Any]] | None = None
    _capability: object | None = None


def _linucb_result_capability_identity(
    result: LinUCBReplayResult, price_id: str
) -> dict[str, Any]:
    if price_id not in result.decisions_by_price or price_id not in result.audit_by_price:
        raise RuntimeError("LinUCB result capability缺少指定价格")
    # A price child is an independently resumable scientific unit.  The root
    # audit also records the *requested* price subset for the current process;
    # binding that call-local list here would make an otherwise identical R05
    # trajectory acquire a different identity when resumed alone after a
    # five-price run.  Keep the complete root audit as parent-level evidence,
    # while pinning only the price-subset-invariant coordinator scope in every
    # child capability.
    coordinator_scope_fields = (
        "schema",
        "status",
        "method",
        "evaluation_zone",
        "seed",
        "each_price_fresh_state",
        "cross_price_learning",
        "global_horizons",
        "event_count_per_price",
        "event_signature",
        "global_replay_scope_contract",
        "global_replay_scope_sha256",
        "derived_contract_sha256",
    )
    if set(coordinator_scope_fields) - set(result.audit):
        raise RuntimeError("LinUCB root audit缺少稳定coordinator scope字段")
    coordinator_scope_audit = {
        field: result.audit[field] for field in coordinator_scope_fields
    }
    payload = {
        "schema": "TEST_CLARA_GEFCOM_6A_FORMAL_LINUCB_RESULT_CAPABILITY_V2",
        "price_id": str(price_id),
        "decisions_sha256": _dataframe_content_sha256(
            result.decisions_by_price[price_id]
        ),
        "price_audit_sha256": _canonical_digest(result.audit_by_price[price_id]),
        "coordinator_scope_audit_sha256": _canonical_digest(
            coordinator_scope_audit
        ),
    }
    return {**payload, "capability_sha256": _canonical_digest(payload)}


def verify_formal_linucb_result_capability(
    result: LinUCBReplayResult, price_id: str
) -> dict[str, Any]:
    if result._capability is not _FORMAL_LINUCB_RESULT_CAPABILITY:
        raise RuntimeError("formal LinUCB audit只接受core replay签发的capability")
    observed = _linucb_result_capability_identity(result, str(price_id))
    identities = dict(result.capability_identity_by_price or {})
    if identities.get(str(price_id)) != observed:
        raise RuntimeError("formal LinUCB result capability内容重算失配")
    return observed


def replay_all_price_conditioned_linucb(
    horizon_facts: Mapping[int, pd.DataFrame] | Sequence[pd.DataFrame],
    *,
    prices: Sequence[PriceSpec],
    contracts: FrozenContracts,
    exploration_alpha: float,
    l2_regularization: float,
    evaluation_zone: str,
    seed: int,
    source_zones: Sequence[str],
    expected_horizons: Sequence[int] = FORMAL_HORIZONS,
    expected_predictors: Sequence[str] = FORMAL_PREDICTORS,
    expected_coverages: Sequence[float] = FORMAL_COVERAGES,
    expected_event_count: int | None = None,
    expected_event_signature: Mapping[str, Any] | None = None,
    cadence_minutes: float = 60.0,
    result_sink: Callable[[PriceSpec, pd.DataFrame, Mapping[str, Any]], None]
    | None = None,
    retain_decisions: bool = False,
    require_formal_identity: bool = True,
    fit_child_sha256_by_price: Mapping[str, str] | None = None,
    endpoint_root_sha256: str | None = None,
) -> LinUCBReplayResult:
    """Replay LinUCB once per price over one global five-horizon issue timeline.

    Horizon frames may be loaded/checkpointed separately, but they are concatenated
    before every price replay.  Restarting LinUCB independently by horizon is rejected
    because the frozen GEFCom coordinator scope is ``(theta_id, seed)``.
    """

    if tuple(contracts.actions) != SIX_ACTIONS:
        raise ValueError("LinUCB回放合同未扩展为六动作")
    contract_identity = derived_six_action_contract_identity(contracts)
    if (
        not np.isfinite(float(exploration_alpha))
        or float(exploration_alpha) < 0.0
        or not np.isfinite(float(l2_regularization))
        or float(l2_regularization) <= 0.0
    ):
        raise ValueError("LinUCB alpha/lambda必须为有限且lambda>0、alpha>=0")
    frozen_linucb_configuration = None
    if require_formal_identity:
        frozen_linucb_configuration = frozen_zone_baseline_contract(
            str(evaluation_zone), "LinUCB"
        )
        if dict(frozen_linucb_configuration["configuration"]) != {
            "exploration_alpha": float(exploration_alpha),
            "l2_regularization": float(l2_regularization),
        }:
            raise RuntimeError("LinUCB超参与持出区S07 registry冻结行失配")
    else:
        nonformal_payload = {
            "schema": "TEST_CLARA_6A_NONFORMAL_LINUCB_CONFIGURATION_V1",
            "outer_heldout_zone": str(evaluation_zone),
            "baseline_id": "LinUCB",
            "configuration": {
                "exploration_alpha": float(exploration_alpha),
                "l2_regularization": float(l2_regularization),
            },
        }
        frozen_linucb_configuration = {
            **nonformal_payload,
            "frozen_configuration_sha256": _canonical_digest(nonformal_payload),
        }
    if not retain_decisions and result_sink is None:
        raise ValueError("不保留LinUCB决策时必须提供result_sink")
    expected = tuple(sorted(int(value) for value in expected_horizons))
    if isinstance(horizon_facts, Mapping):
        normalized_mapping: dict[int, pd.DataFrame] = {}
        for raw_key, value in horizon_facts.items():
            key = int(raw_key)
            if key in normalized_mapping:
                raise ValueError(f"LinUCB时距分块键重复: {key}")
            normalized_mapping[key] = value
        observed_horizons = tuple(sorted(normalized_mapping))
        frames = []
        for value in observed_horizons:
            current = normalized_mapping[value]
            _require_columns(current, ("horizon_steps",), "LinUCB时距分块")
            actual = set(current["horizon_steps"].astype(int))
            if actual != {value}:
                raise RuntimeError(
                    f"LinUCB时距分块键与内容失配: key={value}, actual={sorted(actual)}"
                )
            frames.append(current)
    else:
        frames = list(horizon_facts)
        observed_values: list[int] = []
        for frame in frames:
            _require_columns(frame, ("horizon_steps",), "LinUCB时距分块")
            values = set(frame["horizon_steps"].astype(int))
            if len(values) != 1:
                raise ValueError("LinUCB时距分块必须只含一个时距")
            observed_values.append(next(iter(values)))
        observed_horizons = tuple(sorted(observed_values))
    if observed_horizons != expected:
        raise RuntimeError(
            f"LinUCB全局协调时距不完整: observed={observed_horizons}, expected={expected}"
        )
    if require_formal_identity and expected != FORMAL_HORIZONS:
        raise RuntimeError("LinUCB正式身份必须使用1/3/6/12/24h五时距")
    frame = pd.concat(frames, ignore_index=True, sort=False)
    required = {
        "event_id",
        "zone_or_farm",
        "seed",
        "horizon_steps",
        "issue_timestamp",
        "label_timestamp",
        "label_available_timestamp",
        "target_after_maturity",
        "schedule_proxy",
        *STATE_FIELDS,
    }
    for action in SIX_ACTIONS:
        required.update(
            {f"{action}__candidate_lower", f"{action}__candidate_upper"}
        )
    _require_columns(frame, required, "LinUCB全局回放事件")
    _validate_event_ids(frame, "LinUCB全局回放")
    frame["event_id"] = frame["event_id"].astype(str)
    if set(frame["zone_or_farm"].astype(str)) != {str(evaluation_zone)}:
        raise ValueError("LinUCB回放混入非目标区")
    if set(frame["seed"].astype(int)) != {int(seed)}:
        raise ValueError("LinUCB回放混入其他随机种子")
    predictors = tuple(str(value) for value in expected_predictors)
    coverages = tuple(float(value) for value in expected_coverages)
    if require_formal_identity and (
        predictors != FORMAL_PREDICTORS or coverages != FORMAL_COVERAGES
    ):
        raise RuntimeError("LinUCB正式回放必须使用四预测器×十一覆盖率")
    if set(frame["predictor"].astype(str)) != set(predictors):
        raise RuntimeError("LinUCB全局回放预测器范围失配")
    if set(frame["target_coverage"].astype(float)) != set(coverages):
        raise RuntimeError("LinUCB全局回放覆盖率范围失配")
    observed_cells = set(
        frame[["horizon_steps", "predictor", "target_coverage"]].drop_duplicates().itertuples(
            index=False, name=None
        )
    )
    expected_cells = {
        (int(horizon), str(predictor), float(coverage))
        for horizon in expected
        for predictor in predictors
        for coverage in coverages
    }
    if observed_cells != expected_cells:
        raise RuntimeError(
            f"LinUCB全局回放时距×预测器×覆盖率矩阵不完整: "
            f"missing={list(sorted(expected_cells-observed_cells))[:5]}"
        )
    observed_event_signature = event_id_signature(
        frame["event_id"], include_canonical_sha256=True
    )
    if require_formal_identity and (
        expected_event_count is None or expected_event_signature is None
    ):
        raise RuntimeError(
            "LinUCB正式回放必须pin endpoint-root派生的可信事件数与签名"
        )
    if expected_event_count is not None and len(frame) != int(expected_event_count):
        raise RuntimeError("LinUCB回放事件数与endpoint-root失配")
    if expected_event_signature is not None:
        required_signature_fields = {
            "event_count",
            "hash_sum_u64",
            "hash_xor_u64",
            "hash_square_sum_u64",
            "canonical_sorted_event_id_sha256",
        }
        if require_formal_identity and set(expected_event_signature) != required_signature_fields:
            raise RuntimeError(
                "LinUCB正式endpoint event signature字段必须完整且无额外字段"
            )
        for field in required_signature_fields:
            if field not in expected_event_signature or str(
                observed_event_signature[field]
            ) != str(expected_event_signature[field]):
                raise RuntimeError(f"LinUCB回放事件签名失配: {field}")
    sources = tuple(sorted(str(value) for value in source_zones))
    if str(evaluation_zone) in sources:
        raise ValueError("LinUCB回放的源区列表含目标区")
    if require_formal_identity and (
        len(sources) != 9 or set(sources) != set(FORMAL_ZONES) - {str(evaluation_zone)}
    ):
        raise RuntimeError("LinUCB正式留一区源区身份失配")
    price_rows = tuple(prices)
    if len({row.price_id for row in price_rows}) != len(price_rows):
        raise ValueError("LinUCB价格标识重复")
    if require_formal_identity:
        assert_formal_price_subset(price_rows)
    if require_formal_identity:
        if fit_child_sha256_by_price is None or endpoint_root_sha256 is None:
            raise RuntimeError(
                "正式LinUCB replay必须pin同价fit child与endpoint root SHA"
            )
        expected_price_ids = {row.price_id for row in price_rows}
        if set(str(value) for value in fit_child_sha256_by_price) != expected_price_ids:
            raise RuntimeError("正式replay的逐价fit child SHA矩阵不完整")
    decisions_by_price: dict[str, pd.DataFrame] = {}
    audit_by_price: dict[str, dict[str, Any]] = {}
    for price in price_rows:
        # A fresh accelerator call allocates new A^-1/b state for every action.
        priced = attach_price_losses(
            frame,
            price,
            cadence_minutes=cadence_minutes,
            copy_frame=True,
        )
        selected, engine_audit = run_linucb_local_actions(
            final_facts=priced,
            adaptation_facts=None,  # GEFCom target replay starts without target adaptation.
            contracts=contracts,
            exploration_alpha=float(exploration_alpha),
            l2_regularization=float(l2_regularization),
            actions=SIX_ACTIONS,
        )
        if (
            int(engine_audit["future_feedback_violation_count"]) != 0
            or int(engine_audit["within_issue_feedback_use_count"]) != 0
        ):
            raise RuntimeError(f"LinUCB因果审计失败: {price.price_id}")
        update_counts = {
            str(action): int(value)
            for action, value in dict(engine_audit["action_update_count"]).items()
        }
        if set(update_counts) != set(SIX_ACTIONS):
            raise RuntimeError(f"LinUCB六臂反馈审计不完整: {price.price_id}")
        if sum(update_counts.values()) != int(
            engine_audit["feedback_count_before_terminal_drain"]
        ):
            raise RuntimeError(f"LinUCB反馈更新计数不闭合: {price.price_id}")
        policy_identity = linucb_policy_identity(
            price=price,
            evaluation_zone=str(evaluation_zone),
            seed=int(seed),
            exploration_alpha=float(exploration_alpha),
            l2_regularization=float(l2_regularization),
            horizons=observed_horizons,
            predictors=predictors,
            coverages=coverages,
            actions=SIX_ACTIONS,
            frozen_configuration_contract=frozen_linucb_configuration,
            derived_contract_identity=contract_identity,
        )
        replay_identity = None
        if fit_child_sha256_by_price is not None and endpoint_root_sha256 is not None:
            replay_identity = replay_price_child_identity(
                price=price,
                evaluation_zone=str(evaluation_zone),
                seed=int(seed),
                fit_child_sha256=str(fit_child_sha256_by_price[price.price_id]),
                endpoint_root_sha256=str(endpoint_root_sha256),
                linucb_identity=policy_identity,
            )
        decision = frame[
            ["event_id", "horizon_steps", "predictor", "target_coverage"]
        ].copy()
        decision["selected_action"] = selected
        decision.insert(0, "price_id", price.price_id)
        retained_decisions_sha256 = _dataframe_content_sha256(decision)
        audit = {
            "schema": SCHEMA,
            "status": "PASS",
            "method": "LinUCB_6A",
            "evaluation_zone": str(evaluation_zone),
            "seed": int(seed),
            "price": price.as_dict(),
            "actions": list(SIX_ACTIONS),
            "horizons": list(observed_horizons),
            "global_issue_time_coordinator": True,
            "horizon_independent_restart": False,
            "fresh_state_initialized_for_this_price": True,
            "cross_price_state_shared": False,
            "strict_delayed_feedback": True,
            "candidate_action_count_per_event": len(SIX_ACTIONS),
            "candidate_event_action_count": len(frame) * len(SIX_ACTIONS),
            "selected_action_count": len(selected),
            "selected_action_library_closed": True,
            "retained_decisions_sha256": retained_decisions_sha256,
            "action_library_sha256": action_library_sha256(),
            "price_sha256": price_sha256(price),
            "global_replay_scope_sha256": global_replay_scope_sha256(),
            "derived_contract_sha256": contract_identity[
                "derived_contract_sha256"
            ],
            "linucb_policy_identity": policy_identity,
            "replay_child_identity": replay_identity,
            "terminal_feedback_drain": False,
            "terminal_drain_decision_effect": "NONE_AFTER_LAST_RELEASE",
            **dict(engine_audit),
        }
        audit_by_price[price.price_id] = audit
        if result_sink is not None:
            result_sink(price, decision, audit)
        if retain_decisions:
            decisions_by_price[price.price_id] = decision
        del priced
    root_audit = {
        "schema": SCHEMA,
        "status": "PASS",
        "method": "LinUCB_6A",
        "evaluation_zone": str(evaluation_zone),
        "seed": int(seed),
        "price_ids": [row.price_id for row in price_rows],
        "price_replay_count": len(price_rows),
        "each_price_fresh_state": True,
        "cross_price_learning": False,
        "global_horizons": list(observed_horizons),
        "event_count_per_price": len(frame),
        "event_signature": observed_event_signature,
        "global_replay_scope_contract": global_replay_scope_contract(),
        "global_replay_scope_sha256": global_replay_scope_sha256(),
        "derived_contract_sha256": contract_identity["derived_contract_sha256"],
    }
    result = LinUCBReplayResult(
        decisions_by_price=decisions_by_price,
        audit_by_price=audit_by_price,
        audit=root_audit,
    )
    if not require_formal_identity or not retain_decisions:
        return result
    capability_identities = {
        price_id: _linucb_result_capability_identity(result, price_id)
        for price_id in result.audit_by_price
    }
    return LinUCBReplayResult(
        decisions_by_price=result.decisions_by_price,
        audit_by_price=result.audit_by_price,
        audit=result.audit,
        capability_identity_by_price=capability_identities,
        _capability=_FORMAL_LINUCB_RESULT_CAPABILITY,
    )


def projected_output_bytes(
    *,
    event_count_per_method_price: int,
    method_count: int,
    price_count: int,
    estimated_event_row_bytes: int = 96,
    cell_count_per_method_price: int | None = None,
    estimated_cell_row_bytes: int = 256,
) -> dict[str, Any]:
    """Project full event materialization versus sufficient-statistic storage."""

    event_count = int(event_count_per_method_price)
    methods = int(method_count)
    prices = int(price_count)
    row_bytes = int(estimated_event_row_bytes)
    if min(event_count, methods, prices, row_bytes) <= 0:
        raise ValueError("输出容量估算参数必须为正")
    raw_rows = event_count * methods * prices
    raw_bytes = raw_rows * row_bytes
    cells = None if cell_count_per_method_price is None else int(
        cell_count_per_method_price
    )
    sufficient_bytes = (
        None
        if cells is None
        else cells * methods * prices * int(estimated_cell_row_bytes)
    )
    return {
        "schema": SCHEMA,
        "method_event_price_row_count": raw_rows,
        "projected_full_event_bytes": raw_bytes,
        "projected_full_event_gib": raw_bytes / (1024.0**3),
        "estimated_event_row_bytes": row_bytes,
        "projected_sufficient_stat_bytes": sufficient_bytes,
        "projected_sufficient_stat_gib": (
            None if sufficient_bytes is None else sufficient_bytes / (1024.0**3)
        ),
        "full_event_materialization_allowed": False,
        "required_storage_mode": "STREAMED_CELL_AND_PAIRED_BLOCK_SUFFICIENT_STATISTICS",
    }


def assert_no_full_event_materialization(
    projection: Mapping[str, Any],
    *,
    free_bytes: int,
    maximum_free_disk_fraction: float = 0.25,
) -> None:
    """Fail closed when a proposed event table would consume unsafe disk space."""

    projected = int(projection["projected_full_event_bytes"])
    free = int(free_bytes)
    fraction = float(maximum_free_disk_fraction)
    if free <= 0 or not (0.0 < fraction < 1.0):
        raise ValueError("磁盘安全门参数非法")
    if projected > free * fraction:
        raise RuntimeError(
            "禁止全量method-event-price物化: "
            f"projected={projected / (1024.0**3):.3f} GiB, "
            f"allowed={free * fraction / (1024.0**3):.3f} GiB"
        )


@dataclass(frozen=True)
class StreamingMetricResult:
    cell_metrics: pd.DataFrame
    action_counts: pd.DataFrame
    paired_blocks: pd.DataFrame
    diagnostic_metrics: pd.DataFrame
    clara_state_counts: pd.DataFrame
    event_conservation: pd.DataFrame
    audit: dict[str, Any]
    linucb_replay_audit: dict[str, Any] | None = None
    capability_identity: dict[str, Any] | None = None
    _capability: object | None = None


def _metric_result_capability_identity(
    result: StreamingMetricResult,
) -> dict[str, Any]:
    payload = {
        "schema": "TEST_CLARA_GEFCOM_6A_FORMAL_REPLAY_RESULT_CAPABILITY_V1",
        "evaluation_zone": result.audit.get("evaluation_zone"),
        "seed": result.audit.get("seed"),
        "policy_mode": result.audit.get("policy_mode"),
        "evaluation_price_id": result.audit.get("evaluation_price_id"),
        "evaluation_price_sha256": result.audit.get("evaluation_price_sha256"),
        "policy_price_id": result.audit.get("policy_price_id"),
        "policy_price_sha256": result.audit.get("policy_price_sha256"),
        "policy_evaluation_scope_sha256": result.audit.get(
            "policy_evaluation_scope_sha256"
        ),
        "trusted_partition_signatures_sha256": result.audit.get(
            "trusted_partition_signatures_sha256"
        ),
        "cell_metrics_sha256": _dataframe_content_sha256(result.cell_metrics),
        "action_counts_sha256": _dataframe_content_sha256(result.action_counts),
        "paired_blocks_sha256": _dataframe_content_sha256(result.paired_blocks),
        "diagnostic_metrics_sha256": _dataframe_content_sha256(
            result.diagnostic_metrics
        ),
        "clara_state_counts_sha256": _dataframe_content_sha256(
            result.clara_state_counts
        ),
        "event_conservation_sha256": _dataframe_content_sha256(
            result.event_conservation
        ),
        "metric_finalize_audit_sha256": _canonical_digest(result.audit),
        "linucb_replay_audit_sha256": (
            None
            if result.linucb_replay_audit is None
            else _canonical_digest(result.linucb_replay_audit)
        ),
    }
    return {**payload, "capability_sha256": _canonical_digest(payload)}


def verify_formal_metric_result_capability(
    result: StreamingMetricResult, *, require_linucb_audit: bool = True
) -> dict[str, Any]:
    """Verify a formal accumulator-issued replay result at first-write time."""

    if result._capability is not _FORMAL_METRIC_RESULT_CAPABILITY:
        raise RuntimeError("formal replay artifact只接受core finalizer签发的capability")
    if require_linucb_audit and result.linucb_replay_audit is None:
        raise RuntimeError("formal replay result capability未绑定LinUCB audit")
    observed = _metric_result_capability_identity(result)
    if result.capability_identity != observed:
        raise RuntimeError("formal replay result capability内容重算失配")
    return observed


def rolling_reliability_sufficient_stats(
    covered: Sequence[bool] | np.ndarray,
    *,
    target_coverage: float,
    cadence_minutes: float = 60.0,
    window_hours: float = 168.0,
) -> dict[str, Any]:
    """Exact compact sufficient statistics for the frozen rolling reliability.

    The input must already be in the authoritative ``(issue_timestamp,
    event_id)`` order.  Integer cumulative sums make this linear in the number
    of events while remaining numerically equivalent to the historical
    ``np.convolve`` implementation.
    """

    values = np.asarray(covered)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("滚动可靠性序列必须为非空一维数组")
    numeric = values.astype(np.float64)
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError("滚动可靠性covered必须为0/1")
    target = float(target_coverage)
    cadence = float(cadence_minutes)
    hours = float(window_hours)
    if (
        not np.isfinite(target)
        or not 0.0 < target < 1.0
        or not np.isfinite(cadence)
        or cadence <= 0.0
        or not np.isfinite(hours)
        or hours <= 0.0
    ):
        raise ValueError("滚动可靠性参数非法")
    window = int(math.ceil(hours * 60.0 / cadence))
    if len(numeric) < window:
        return {
            "rolling_window_size": window,
            "rolling_window_count": 0,
            "towr_exceedance_count": 0,
            "tuwr_exceedance_count": 0,
            "rolling_absolute_deviation_sum": 0.0,
            "TOWR": math.nan,
            "TUWR": math.nan,
            "ARD": math.nan,
        }
    integer = numeric.astype(np.int64, copy=False)
    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(integer, dtype=np.int64))
    )
    rolling = (cumulative[window:] - cumulative[:-window]).astype(
        np.float64
    ) / float(window)
    gaps = rolling - target
    tolerance = 1.96 * math.sqrt(target * (1.0 - target) / float(window))
    over = int(np.count_nonzero(gaps > tolerance))
    under = int(np.count_nonzero(gaps < -tolerance))
    absolute_sum = float(np.abs(gaps).sum(dtype=np.float64))
    count = int(len(gaps))
    return {
        "rolling_window_size": window,
        "rolling_window_count": count,
        "towr_exceedance_count": over,
        "tuwr_exceedance_count": under,
        "rolling_absolute_deviation_sum": absolute_sum,
        "TOWR": float(over / count),
        "TUWR": float(under / count),
        "ARD": float(absolute_sum / count),
    }


CLARA_EVENT_DIAGNOSTIC_FIELDS = (
    "full_state_code",
    "support_backoff_level",
    "support_backoff_name",
    "guardrail_safe_action_count",
    "guardrail_excluded_action_count",
    "guardrail_empty_fallback",
    "selected_action_guardrail_pass",
    "selected_action",
)


@dataclass(frozen=True)
class ClaraEventDiagnosticResult:
    selected_actions: np.ndarray
    diagnostics: pd.DataFrame
    capability_identity: dict[str, Any]
    _capability: object | None = None


def _clara_event_diagnostic_capability_identity(
    result: ClaraEventDiagnosticResult,
    *,
    ordered_event_ids: Sequence[str],
    decisions: pd.DataFrame | None = None,
    action_evidence: pd.DataFrame | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": "TEST_CLARA_GEFCOM_6A_EVENT_DIAGNOSTIC_CAPABILITY_V1",
        "ordered_event_id_sha256": _canonical_digest(
            [str(value) for value in ordered_event_ids]
        ),
        "event_count": len(ordered_event_ids),
        "selected_actions_sha256": _canonical_digest(
            _ndarray_content_identity(
                np.asarray(result.selected_actions, dtype="U64")
            )
        ),
        "diagnostics_sha256": _dataframe_content_sha256(result.diagnostics),
    }
    if decisions is not None or action_evidence is not None:
        if decisions is None or action_evidence is None:
            raise RuntimeError("CLARA event diagnostics策略输入身份不完整")
        payload["clara_decisions_sha256"] = _dataframe_content_sha256(decisions)
        payload["clara_action_evidence_sha256"] = _dataframe_content_sha256(
            action_evidence
        )
    return {**payload, "capability_sha256": _canonical_digest(payload)}


def verify_clara_event_diagnostic_capability(
    result: ClaraEventDiagnosticResult,
    *,
    ordered_event_ids: Sequence[str],
) -> dict[str, Any]:
    if result._capability is not _FORMAL_CLARA_EVENT_DIAGNOSTIC_CAPABILITY:
        raise RuntimeError("formal CLARA event diagnostics缺core helper capability")
    observed = _clara_event_diagnostic_capability_identity(
        result, ordered_event_ids=ordered_event_ids
    )
    expected = dict(result.capability_identity)
    # The helper also binds the exact fit-policy input digests.  They cannot be
    # recomputed from the compact output, but must remain part of the identity.
    for field in ("clara_decisions_sha256", "clara_action_evidence_sha256"):
        if not _is_sha256(expected.get(field)):
            raise RuntimeError("CLARA event diagnostics未绑定完整fit policy")
        observed[field] = expected[field]
    observed["capability_sha256"] = _canonical_digest(
        {key: value for key, value in observed.items() if key != "capability_sha256"}
    )
    if expected != observed:
        raise RuntimeError("CLARA event diagnostics capability内容重算失配")
    return observed


def predict_clara_actions_with_diagnostics(
    events: pd.DataFrame,
    decisions: pd.DataFrame,
    action_evidence: pd.DataFrame,
) -> ClaraEventDiagnosticResult:
    """Map a sealed CLARA policy and its guardrail/support audit to events."""

    decision_fields = (
        "full_state_code",
        *STATE_FIELDS,
        "support_backoff_level",
        "support_backoff_name",
        "guardrail_empty_fallback",
        "selected_action_guardrail_pass",
        "selected_action",
    )
    _require_columns(events, ("event_id", *STATE_FIELDS), "CLARA目标事件诊断")
    _require_columns(decisions, decision_fields, "CLARA状态诊断")
    _require_columns(
        action_evidence,
        ("full_state_code", "action_order", "guardrail_pass"),
        "CLARA护栏证据",
    )
    if decisions[list(STATE_FIELDS)].duplicated().any() or decisions[
        "full_state_code"
    ].duplicated().any():
        raise RuntimeError("CLARA状态诊断决策键重复")
    evidence = action_evidence.sort_values(
        ["full_state_code", "action_order"], kind="mergesort"
    ).copy()
    grouped = evidence.groupby("full_state_code", sort=True, dropna=False)
    if (
        grouped.size().ne(len(SIX_ACTIONS)).any()
        or not np.array_equal(
            evidence["action_order"].to_numpy(dtype=np.int64),
            np.tile(
                np.arange(len(SIX_ACTIONS), dtype=np.int64),
                len(decisions),
            ),
        )
    ):
        raise RuntimeError("CLARA护栏证据未按状态×六动作闭合")
    safe_counts = (
        grouped["guardrail_pass"]
        .sum()
        .astype(np.int64)
        .rename("guardrail_safe_action_count")
        .reset_index()
    )
    state_lookup = decisions.loc[:, list(decision_fields)].merge(
        safe_counts,
        on="full_state_code",
        how="left",
        validate="one_to_one",
    )
    state_lookup["guardrail_excluded_action_count"] = (
        len(SIX_ACTIONS)
        - state_lookup["guardrail_safe_action_count"].to_numpy(dtype=np.int64)
    )
    if (
        state_lookup["guardrail_safe_action_count"].isna().any()
        or np.any(
            (state_lookup["guardrail_safe_action_count"] < 0)
            | (state_lookup["guardrail_safe_action_count"] > len(SIX_ACTIONS))
        )
        or not np.array_equal(
            state_lookup["guardrail_empty_fallback"].astype(bool).to_numpy(),
            state_lookup["guardrail_safe_action_count"].eq(0).to_numpy(),
        )
    ):
        raise RuntimeError("CLARA护栏safe/empty诊断不闭合")
    frame = events.loc[:, ["event_id", *STATE_FIELDS]].copy()
    frame["_row_order"] = np.arange(len(frame), dtype=np.int64)
    mapped = frame.merge(
        state_lookup,
        on=list(STATE_FIELDS),
        how="left",
        validate="many_to_one",
        sort=False,
    ).sort_values("_row_order", kind="mergesort")
    if mapped[list(CLARA_EVENT_DIAGNOSTIC_FIELDS)].isna().any().any():
        raise RuntimeError("CLARA目标事件诊断映射不完整")
    selected = mapped["selected_action"].astype(str).to_numpy()
    if sorted(set(selected) - set(SIX_ACTIONS)):
        raise RuntimeError("CLARA事件诊断含六动作库外值")
    diagnostics = mapped.loc[:, list(CLARA_EVENT_DIAGNOSTIC_FIELDS)].reset_index(
        drop=True
    )
    diagnostics["full_state_code"] = diagnostics["full_state_code"].astype(
        np.int64
    )
    diagnostics["support_backoff_level"] = diagnostics[
        "support_backoff_level"
    ].astype(np.int64)
    diagnostics["guardrail_safe_action_count"] = diagnostics[
        "guardrail_safe_action_count"
    ].astype(np.int64)
    diagnostics["guardrail_excluded_action_count"] = diagnostics[
        "guardrail_excluded_action_count"
    ].astype(np.int64)
    diagnostics["guardrail_empty_fallback"] = diagnostics[
        "guardrail_empty_fallback"
    ].astype(bool)
    diagnostics["selected_action_guardrail_pass"] = diagnostics[
        "selected_action_guardrail_pass"
    ].astype(bool)
    diagnostics["selected_action"] = diagnostics["selected_action"].astype(str)
    provisional = ClaraEventDiagnosticResult(
        selected_actions=selected,
        diagnostics=diagnostics,
        capability_identity={},
    )
    identity = _clara_event_diagnostic_capability_identity(
        provisional,
        ordered_event_ids=events["event_id"].astype(str).tolist(),
        decisions=decisions,
        action_evidence=action_evidence,
    )
    return ClaraEventDiagnosticResult(
        selected_actions=selected,
        diagnostics=diagnostics,
        capability_identity=identity,
        _capability=_FORMAL_CLARA_EVENT_DIAGNOSTIC_CAPABILITY,
    )


class StreamingMetricAccumulator:
    """Bounded-output accumulator for nine-method, five-price evaluation.

    ``update`` consumes an event chunk and immediately reduces it to exact cells,
    action counts, and paired UTC-day block sums.  It never retains endpoint/event
    rows.  A caller may supply ``row_sink`` and set ``retain_reduced_rows=False`` to
    stream even the reduced partitions directly to its own atomic artifact writer.
    """

    CELL_FIELDS = (
        "zone_or_farm",
        "predictor",
        "horizon_steps",
        "seed",
        "target_coverage",
        "ramp_state",
    )
    # Retain all prespecified analysis axes so predictor/horizon/coverage-stratified
    # paired inference remains recoverable.  Coarser overall/ramp blocks are derived
    # by exact downstream summation, never by rereading event rows.
    BLOCK_FIELDS = (
        "zone_or_farm",
        "predictor",
        "horizon_steps",
        "seed",
        "target_coverage",
        "_utc_day",
        "ramp_state",
    )
    DIAGNOSTIC_FIELDS = (
        "zone_or_farm",
        "predictor",
        "horizon_steps",
        "seed",
        "target_coverage",
        "regime",
    )
    CLARA_STATE_COUNT_FIELDS = (
        "zone_or_farm",
        "predictor",
        "horizon_steps",
        "seed",
        "target_coverage",
        "ramp_state",
        *CLARA_EVENT_DIAGNOSTIC_FIELDS,
    )
    CELL_METRIC_COLUMNS = (
        *POLICY_SCOPE_FIELDS,
        "method",
        *CELL_FIELDS,
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
    )
    ACTION_COUNT_COLUMNS = (
        *POLICY_SCOPE_FIELDS,
        "method",
        *CELL_FIELDS,
        "selected_action",
        "action_count",
    )
    PAIRED_BLOCK_COLUMNS = (
        *POLICY_SCOPE_FIELDS,
        "method",
        *BLOCK_FIELDS,
        "event_count",
        "errf_sum",
        "covered_sum",
        "width_sum",
    )
    DIAGNOSTIC_METRIC_COLUMNS = (
        *POLICY_SCOPE_FIELDS,
        "method",
        *DIAGNOSTIC_FIELDS,
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
        "rolling_window_size",
        "rolling_window_count",
        "towr_exceedance_count",
        "tuwr_exceedance_count",
        "rolling_absolute_deviation_sum",
        "TOWR",
        "TUWR",
        "ARD",
        "action_entropy_nats",
        "action_entropy_normalized",
        "support_applicable_count",
        "support_backoff_count",
        "support_backoff_rate",
        "guardrail_applicable_count",
        "guardrail_any_exclusion_count",
        "guardrail_excluded_action_count",
        "guardrail_empty_count",
        "selected_action_guardrail_fail_count",
        "guardrail_any_exclusion_rate",
        "guardrail_action_exclusion_rate",
        "guardrail_empty_rate",
        "selected_action_guardrail_fail_rate",
    )
    CLARA_STATE_COUNT_COLUMNS = (
        *POLICY_SCOPE_FIELDS,
        "method",
        *CLARA_STATE_COUNT_FIELDS,
        "event_count",
    )

    def __init__(
        self,
        *,
        row_sink: Callable[[str, pd.DataFrame], None] | None = None,
        retain_reduced_rows: bool = True,
        cadence_minutes: float = 60.0,
        expected_partition_signatures: Mapping[str, Mapping[str, Any]] | None = None,
        formal_identity: bool = True,
    ) -> None:
        if not retain_reduced_rows and row_sink is None:
            raise ValueError("不保留约减行时必须提供row_sink")
        self.row_sink = row_sink
        self.retain_reduced_rows = bool(retain_reduced_rows)
        self.cadence_minutes = float(cadence_minutes)
        self.formal_identity = bool(formal_identity)
        if not np.isfinite(self.cadence_minutes) or self.cadence_minutes <= 0.0:
            raise ValueError("评估数据节拍非法")
        if self.formal_identity and self.cadence_minutes != 60.0:
            raise RuntimeError("GEFCom正式评估必须使用60分钟节拍/168小时窗")
        self._cell_parts: list[pd.DataFrame] = []
        self._action_parts: list[pd.DataFrame] = []
        self._block_parts: list[pd.DataFrame] = []
        self._diagnostic_parts: list[pd.DataFrame] = []
        self._clara_state_parts: list[pd.DataFrame] = []
        self._conservation: dict[tuple[str, str, str, str], dict[str, int]] = {}
        self.expected_partition_signatures = {
            str(key): dict(value)
            for key, value in dict(expected_partition_signatures or {}).items()
        }
        signature_fields = {
            "event_count",
            "hash_sum_u64",
            "hash_xor_u64",
            "hash_square_sum_u64",
            "canonical_sorted_event_id_sha256",
        }
        if self.formal_identity and not self.expected_partition_signatures:
            raise RuntimeError("正式流式评估必须提供endpoint-root可信partition签名")
        for partition_id, signature in self.expected_partition_signatures.items():
            if set(signature) != signature_fields or not str(
                signature.get("canonical_sorted_event_id_sha256") or ""
            ):
                raise RuntimeError(
                    f"可信partition签名字段不完整: {partition_id}"
                )
        self._seen_partitions: dict[tuple[str, str, str, str], set[str]] = {}
        self._update_count = 0
        self._input_event_score_count = 0
        self._formal_child_scope: tuple[str, int] | None = None
        self._formal_policy_scope: dict[str, Any] | None = None
        self._method_policy_identities: dict[str, dict[str, Any]] = {}
        self._linucb_action_map_by_price: dict[str, pd.Series] = {}

    @staticmethod
    def _selected_values(
        facts: pd.DataFrame,
        selected_actions: np.ndarray,
        field: str,
        *,
        allow_missing_columns: bool = False,
    ) -> np.ndarray:
        selected = np.asarray(selected_actions, dtype=object).astype(str)
        output = np.full(len(facts), np.nan, dtype=np.float64)
        for action in SIX_ACTIONS:
            mask = selected == action
            if not mask.any():
                continue
            column = f"{action}__{field}"
            if column not in facts.columns:
                if allow_missing_columns:
                    continue
                raise ValueError(f"选中动作事件缺少字段: {column}")
            output[mask] = facts.loc[mask, column].to_numpy(dtype=np.float64)
        return output

    @staticmethod
    def _event_signature(event_ids: pd.Series) -> dict[str, int]:
        return event_id_signature(event_ids)

    @staticmethod
    def _combine_signature(target: dict[str, int], current: Mapping[str, int]) -> None:
        modulus = 1 << 64
        target["event_count"] += int(current["event_count"])
        target["hash_sum_u64"] = (
            target["hash_sum_u64"] + int(current["hash_sum_u64"])
        ) % modulus
        target["hash_xor_u64"] ^= int(current["hash_xor_u64"])
        target["hash_square_sum_u64"] = (
            target["hash_square_sum_u64"]
            + int(current["hash_square_sum_u64"])
        ) % modulus

    def _emit(self, kind: str, frame: pd.DataFrame) -> None:
        if self.row_sink is not None:
            self.row_sink(kind, frame)
        if self.retain_reduced_rows:
            if kind == "cell_metrics":
                self._cell_parts.append(frame)
            elif kind == "action_counts":
                self._action_parts.append(frame)
            elif kind == "paired_blocks":
                self._block_parts.append(frame)
            elif kind == "diagnostic_metrics":
                self._diagnostic_parts.append(frame)
            elif kind == "clara_state_counts":
                self._clara_state_parts.append(frame)
            else:
                raise ValueError(f"未知约减输出类型: {kind}")

    @staticmethod
    def _action_entropy(actions: np.ndarray) -> tuple[float, float]:
        counts = np.asarray(
            [np.count_nonzero(actions == action) for action in SIX_ACTIONS],
            dtype=np.float64,
        )
        if counts.sum() <= 0.0:
            raise ValueError("动作熵计算缺少事件")
        probabilities = counts[counts > 0.0] / counts.sum()
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        return entropy, float(entropy / math.log(len(SIX_ACTIONS)))

    def _register_policy_identity(
        self, method: str, identity: Mapping[str, Any]
    ) -> None:
        normalized = {str(key): value for key, value in dict(identity).items()}
        normalized["policy_binding_sha256"] = _canonical_digest(
            {
                key: value
                for key, value in normalized.items()
                if key != "policy_binding_sha256"
            }
        )
        previous = self._method_policy_identities.get(str(method))
        if previous is not None and previous != normalized:
            raise RuntimeError(f"同一方法跨partition policy identity漂移: {method}")
        self._method_policy_identities[str(method)] = normalized

    @staticmethod
    def _validated_clara_event_diagnostics(
        diagnostics: pd.DataFrame,
        *,
        selected_actions: np.ndarray,
        row_count: int,
    ) -> pd.DataFrame:
        if not isinstance(diagnostics, pd.DataFrame):
            raise TypeError("CLARA event diagnostics必须为DataFrame")
        if tuple(diagnostics.columns) != CLARA_EVENT_DIAGNOSTIC_FIELDS:
            raise RuntimeError("CLARA event diagnostics exact schema失配")
        if len(diagnostics) != int(row_count) or diagnostics.isna().any().any():
            raise RuntimeError("CLARA event diagnostics行数/空值失配")
        frame = diagnostics.reset_index(drop=True).copy()
        integer_fields = (
            "full_state_code",
            "support_backoff_level",
            "guardrail_safe_action_count",
            "guardrail_excluded_action_count",
        )
        integer = frame.loc[:, list(integer_fields)].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(integer).all()
            or not np.array_equal(integer, np.floor(integer))
            or np.any(frame["full_state_code"].to_numpy(dtype=np.int64) < 0)
            or np.any(frame["support_backoff_level"].to_numpy(dtype=np.int64) < 0)
            or np.any(
                frame["guardrail_safe_action_count"].to_numpy(dtype=np.int64)
                + frame["guardrail_excluded_action_count"].to_numpy(dtype=np.int64)
                != len(SIX_ACTIONS)
            )
            or np.any(
                (frame["guardrail_safe_action_count"].to_numpy(dtype=np.int64) < 0)
                | (
                    frame["guardrail_safe_action_count"].to_numpy(dtype=np.int64)
                    > len(SIX_ACTIONS)
                )
            )
        ):
            raise RuntimeError("CLARA event diagnostics数值/六动作safe-set不闭合")
        if (
            not np.array_equal(
                frame["selected_action"].astype(str).to_numpy(),
                selected_actions.astype(str),
            )
            or sorted(set(frame["selected_action"].astype(str)) - set(SIX_ACTIONS))
            or not np.array_equal(
                frame["guardrail_empty_fallback"].astype(bool).to_numpy(),
                frame["guardrail_safe_action_count"].eq(0).to_numpy(),
            )
        ):
            raise RuntimeError("CLARA event diagnostics与动作/护栏决策不闭合")
        return frame

    def _diagnostic_rows(
        self,
        *,
        policy_scope: Mapping[str, Any],
        method: str,
        facts: pd.DataFrame,
        actions: np.ndarray,
        errf: np.ndarray,
        reserve_up: np.ndarray,
        reserve_down: np.ndarray,
        miss_upper: np.ndarray,
        miss_lower: np.ndarray,
        covered: np.ndarray,
        width: np.ndarray,
        interval_score: np.ndarray,
        clara_diagnostics: pd.DataFrame | None,
    ) -> pd.DataFrame:
        base_axes = [field for field in self.DIAGNOSTIC_FIELDS if field != "regime"]
        if any(facts[field].nunique(dropna=False) != 1 for field in base_axes):
            raise RuntimeError("滚动可靠性update必须为单一zone×pred×h×seed×coverage partition")
        axis = {field: facts.iloc[0][field] for field in base_axes}
        issue = pd.to_datetime(facts["issue_timestamp"], errors="raise", utc=True)
        event_ids = facts["event_id"].astype(str).to_numpy()
        ramp = facts["ramp_state"].astype(str).to_numpy()
        target_coverage = float(axis["target_coverage"])
        rows: list[dict[str, Any]] = []
        for regime in ("overall", "ordinary", "ramp"):
            mask = (
                np.ones(len(facts), dtype=bool)
                if regime == "overall"
                else ramp == regime
            )
            if not mask.any():
                continue
            positions = np.flatnonzero(mask)
            ordered_local = np.lexsort(
                (
                    event_ids[positions],
                    issue.iloc[positions].astype("int64").to_numpy(),
                )
            )
            ordered_positions = positions[ordered_local]
            reliability = rolling_reliability_sufficient_stats(
                covered[ordered_positions],
                target_coverage=target_coverage,
                cadence_minutes=self.cadence_minutes,
            )
            entropy, normalized_entropy = self._action_entropy(actions[positions])
            event_count = int(len(positions))
            support_applicable = event_count if clara_diagnostics is not None else 0
            guardrail_applicable = support_applicable
            if clara_diagnostics is None:
                support_backoff_count = 0
                guardrail_any_exclusion_count = 0
                guardrail_excluded_action_count = 0
                guardrail_empty_count = 0
                selected_guardrail_fail_count = 0
            else:
                local = clara_diagnostics.iloc[positions]
                support_backoff_count = int(
                    local["support_backoff_level"].astype(int).gt(0).sum()
                )
                excluded = local["guardrail_excluded_action_count"].to_numpy(
                    dtype=np.int64
                )
                guardrail_any_exclusion_count = int(np.count_nonzero(excluded > 0))
                guardrail_excluded_action_count = int(excluded.sum())
                guardrail_empty_count = int(
                    local["guardrail_empty_fallback"].astype(bool).sum()
                )
                selected_guardrail_fail_count = int(
                    (~local["selected_action_guardrail_pass"].astype(bool)).sum()
                )
            rolling_count = int(reliability["rolling_window_count"])
            covered_sum = int(covered[positions].sum())
            target_sum = float(target_coverage * event_count)
            rows.append(
                {
                    **{
                        field: str(policy_scope[field])
                        for field in POLICY_SCOPE_FIELDS
                    },
                    "method": str(method),
                    **axis,
                    "regime": regime,
                    "event_count": event_count,
                    "errf_sum": float(errf[positions].sum()),
                    "reserve_up_sum": float(reserve_up[positions].sum()),
                    "reserve_down_sum": float(reserve_down[positions].sum()),
                    "miss_upper_sum": float(miss_upper[positions].sum()),
                    "miss_lower_sum": float(miss_lower[positions].sum()),
                    "covered_sum": covered_sum,
                    "coverage_target_sum": target_sum,
                    "coverage_gap_sum": float(covered_sum - target_sum),
                    "width_sum": float(width[positions].sum()),
                    "interval_score_sum": float(interval_score[positions].sum()),
                    "rolling_window_size": int(reliability["rolling_window_size"]),
                    "rolling_window_count": rolling_count,
                    "towr_exceedance_count": int(
                        reliability["towr_exceedance_count"]
                    ),
                    "tuwr_exceedance_count": int(
                        reliability["tuwr_exceedance_count"]
                    ),
                    "rolling_absolute_deviation_sum": float(
                        reliability["rolling_absolute_deviation_sum"]
                    ),
                    "TOWR": float(reliability["TOWR"]),
                    "TUWR": float(reliability["TUWR"]),
                    "ARD": float(reliability["ARD"]),
                    "action_entropy_nats": entropy,
                    "action_entropy_normalized": normalized_entropy,
                    "support_applicable_count": support_applicable,
                    "support_backoff_count": support_backoff_count,
                    "support_backoff_rate": (
                        float(support_backoff_count / support_applicable)
                        if support_applicable
                        else math.nan
                    ),
                    "guardrail_applicable_count": guardrail_applicable,
                    "guardrail_any_exclusion_count": guardrail_any_exclusion_count,
                    "guardrail_excluded_action_count": guardrail_excluded_action_count,
                    "guardrail_empty_count": guardrail_empty_count,
                    "selected_action_guardrail_fail_count": selected_guardrail_fail_count,
                    "guardrail_any_exclusion_rate": (
                        float(guardrail_any_exclusion_count / guardrail_applicable)
                        if guardrail_applicable
                        else math.nan
                    ),
                    "guardrail_action_exclusion_rate": (
                        float(
                            guardrail_excluded_action_count
                            / (guardrail_applicable * len(SIX_ACTIONS))
                        )
                        if guardrail_applicable
                        else math.nan
                    ),
                    "guardrail_empty_rate": (
                        float(guardrail_empty_count / guardrail_applicable)
                        if guardrail_applicable
                        else math.nan
                    ),
                    "selected_action_guardrail_fail_rate": (
                        float(selected_guardrail_fail_count / guardrail_applicable)
                        if guardrail_applicable
                        else math.nan
                    ),
                }
            )
        return pd.DataFrame(rows)

    def update(
        self,
        *,
        method: str,
        facts: pd.DataFrame,
        selected_actions: Sequence[str],
        price: PriceSpec | None = None,
        evaluation_price: PriceSpec | None = None,
        policy_mode: str = POLICY_MODE_RESELECTED,
        policy_price: PriceSpec | None = None,
        selected_lower: np.ndarray | None = None,
        selected_upper: np.ndarray | None = None,
        clara_event_diagnostics: ClaraEventDiagnosticResult | pd.DataFrame | None = None,
        cart_event_actions: CartEventActionResult | None = None,
        linucb_replay_result: LinUCBReplayResult | None = None,
        partition_id: str | None = None,
    ) -> dict[str, Any]:
        if evaluation_price is None:
            if price is None:
                raise ValueError("update缺少evaluation price")
            evaluation_price = price
        elif price is not None and price_sha256(price) != price_sha256(
            evaluation_price
        ):
            raise RuntimeError("price别名与evaluation_price失配")
        evaluation_price.validate()
        scope_identity = policy_evaluation_scope(
            policy_mode=policy_mode,
            evaluation_price=evaluation_price,
            policy_price=policy_price,
            require_formal_identity=self.formal_identity,
        )
        resolved_policy_price = (
            evaluation_price
            if scope_identity["policy_price_id"] == evaluation_price.price_id
            else formal_price_spec(str(scope_identity["policy_price_id"]))
        )
        price = evaluation_price
        method_id = str(method)
        if not method_id:
            raise ValueError("方法标识不得为空")
        if self.formal_identity:
            if self._formal_policy_scope is None:
                self._formal_policy_scope = dict(scope_identity)
            elif self._formal_policy_scope != dict(scope_identity):
                raise RuntimeError("单一formal accumulator混入不同policy mode/双价格")
            if method_id not in FORMAL_METHODS:
                raise RuntimeError("正式流式评估方法超出冻结九方法")
        required = {
            "event_id",
            "issue_timestamp",
            "target_after_maturity",
            "schedule_proxy",
            "rolling_state",
            *self.CELL_FIELDS,
        }
        _require_columns(facts, required, "流式评估事件")
        if facts.empty:
            raise ValueError("流式评估事件为空")
        _validate_event_ids(facts, "单次流式评估块")
        if facts[list(required)].isna().any().any():
            raise ValueError("流式评估身份/指标关键列含空值")
        if self.formal_identity:
            frozen_axes = {
                "zone_or_farm": set(FORMAL_ZONES),
                "predictor": set(FORMAL_PREDICTORS),
                "horizon_steps": set(FORMAL_HORIZONS),
                "seed": {0, 1, 2},
                "target_coverage": set(FORMAL_COVERAGES),
                "ramp_state": {"ordinary", "ramp"},
                "rolling_state": {
                    "undercoverage_pressure",
                    "overcoverage_pressure",
                    "volatile",
                    "stable",
                    "cold_start",
                },
            }
            for field, allowed in frozen_axes.items():
                observed = set(facts[field].tolist())
                if field in {"horizon_steps", "seed"}:
                    observed = {int(value) for value in observed}
                elif field == "target_coverage":
                    observed = {float(value) for value in observed}
                else:
                    observed = {str(value) for value in observed}
                if not observed or not observed.issubset(allowed):
                    raise RuntimeError(
                        f"正式流式评估轴越出冻结范围: {field}/{observed}"
                    )
            scopes = set(
                facts[["zone_or_farm", "seed"]].itertuples(
                    index=False, name=None
                )
            )
            if len(scopes) != 1:
                raise RuntimeError("逐价正式评估chunk必须属于单一zone×seed")
            current_scope = next(iter(scopes))
            normalized_scope = (str(current_scope[0]), int(current_scope[1]))
            if self._formal_child_scope is None:
                self._formal_child_scope = normalized_scope
            elif self._formal_child_scope != normalized_scope:
                raise RuntimeError("逐价正式评估跨zone/seed串线")
        actions = np.asarray(selected_actions, dtype=object).astype(str)
        if len(actions) != len(facts):
            raise ValueError("流式评估动作数组长度失配")
        invalid = sorted(set(actions) - set(SIX_ACTIONS))
        if invalid:
            raise ValueError(f"流式评估含六动作库外值: {invalid}")
        # Policy-source identity deliberately excludes evaluation theta.  In the
        # fixed deployment scan the same R05 policy/trajectory must therefore
        # have one identical binding across all five evaluation prices.
        policy_scope_binding = {
            "policy_mode": str(scope_identity["policy_mode"]),
            "policy_price_id": str(scope_identity["policy_price_id"]),
            "policy_price_sha256": str(scope_identity["policy_price_sha256"]),
        }
        if method_id == "CLARA_6A":
            if cart_event_actions is not None or linucb_replay_result is not None:
                raise RuntimeError("CLARA评估接入了其他方法policy capability")
            if clara_event_diagnostics is None:
                if self.formal_identity:
                    raise RuntimeError(
                        "正式CLARA评估必须绑定full-state/support/guardrail事件诊断"
                    )
                clara_diagnostics = None
            else:
                if isinstance(
                    clara_event_diagnostics, ClaraEventDiagnosticResult
                ):
                    clara_identity = verify_clara_event_diagnostic_capability(
                        clara_event_diagnostics,
                        ordered_event_ids=facts["event_id"].astype(str).tolist(),
                    )
                    if not np.array_equal(
                        np.asarray(
                            clara_event_diagnostics.selected_actions,
                            dtype=object,
                        ).astype(str),
                        actions,
                    ):
                        raise RuntimeError(
                            "CLARA event diagnostics capability与selected_actions失配"
                        )
                    diagnostic_frame = clara_event_diagnostics.diagnostics
                    self._register_policy_identity(
                        method_id,
                        {
                            "schema": "TEST_CLARA_GEFCOM_6A_CLARA_REPLAY_POLICY_BINDING_V1",
                            **policy_scope_binding,
                            "clara_decisions_sha256": clara_identity[
                                "clara_decisions_sha256"
                            ],
                            "clara_action_evidence_sha256": clara_identity[
                                "clara_action_evidence_sha256"
                            ],
                        },
                    )
                else:
                    if self.formal_identity:
                        raise RuntimeError(
                            "formal CLARA评估拒绝裸diagnostics DataFrame"
                        )
                    diagnostic_frame = clara_event_diagnostics
                clara_diagnostics = self._validated_clara_event_diagnostics(
                    diagnostic_frame,
                    selected_actions=actions,
                    row_count=len(facts),
                )
        elif method_id == "CART_6A":
            if clara_event_diagnostics is not None or linucb_replay_result is not None:
                raise RuntimeError("CART评估接入了其他方法policy capability")
            if cart_event_actions is None:
                if self.formal_identity:
                    raise RuntimeError("正式CART评估必须绑定core predictor capability")
            else:
                cart_identity = verify_cart_event_action_capability(
                    cart_event_actions,
                    ordered_event_ids=facts["event_id"].astype(str).tolist(),
                )
                if not np.array_equal(
                    np.asarray(cart_event_actions.selected_actions, dtype=object).astype(
                        str
                    ),
                    actions,
                ):
                    raise RuntimeError("CART predictor capability与selected_actions失配")
                self._register_policy_identity(
                    method_id,
                    {
                        "schema": "TEST_CLARA_GEFCOM_6A_CART_REPLAY_POLICY_BINDING_V1",
                        **policy_scope_binding,
                        "cart_selector_sha256": cart_identity[
                            "cart_selector_sha256"
                        ],
                    },
                )
            clara_diagnostics = None
        elif method_id == "LinUCB_6A":
            if clara_event_diagnostics is not None or cart_event_actions is not None:
                raise RuntimeError("LinUCB评估接入了其他方法policy capability")
            if linucb_replay_result is None:
                if self.formal_identity:
                    raise RuntimeError("正式LinUCB评估必须绑定同价replay capability")
            else:
                linucb_identity = verify_formal_linucb_result_capability(
                    linucb_replay_result, resolved_policy_price.price_id
                )
                action_map = self._linucb_action_map_by_price.get(
                    resolved_policy_price.price_id
                )
                if action_map is None:
                    decisions = linucb_replay_result.decisions_by_price[
                        resolved_policy_price.price_id
                    ].copy()
                    _require_columns(
                        decisions, ("event_id", "selected_action"), "LinUCB决策闭合"
                    )
                    _validate_event_ids(decisions, "LinUCB决策闭合")
                    action_map = decisions.set_index(
                        decisions["event_id"].astype(str)
                    )["selected_action"].astype(str)
                    self._linucb_action_map_by_price[
                        resolved_policy_price.price_id
                    ] = action_map
                expected_actions = action_map.reindex(
                    facts["event_id"].astype(str)
                )
                if expected_actions.isna().any() or not np.array_equal(
                    expected_actions.to_numpy(dtype=object).astype(str), actions
                ):
                    raise RuntimeError("LinUCB实际计分动作与replay decisions不闭合")
                self._register_policy_identity(
                    method_id,
                    {
                        "schema": "TEST_CLARA_GEFCOM_6A_LINUCB_REPLAY_POLICY_BINDING_V1",
                        **policy_scope_binding,
                        "linucb_decisions_sha256": linucb_identity[
                            "decisions_sha256"
                        ],
                        "linucb_result_capability_sha256": linucb_identity[
                            "capability_sha256"
                        ],
                    },
                )
            clara_diagnostics = None
        elif method_id in FORMAL_DETERMINISTIC_METHOD_ACTIONS:
            if (
                clara_event_diagnostics is not None
                or cart_event_actions is not None
                or linucb_replay_result is not None
            ):
                raise RuntimeError("确定性方法不得接入selector capability")
            expected_action = FORMAL_DETERMINISTIC_METHOD_ACTIONS[method_id]
            if set(actions) != {expected_action}:
                raise RuntimeError(
                    f"确定性方法动作语义失配: {method_id}/{expected_action}"
                )
            self._register_policy_identity(
                method_id,
                {
                    "schema": "TEST_CLARA_GEFCOM_6A_DETERMINISTIC_POLICY_BINDING_V1",
                    **policy_scope_binding,
                    "fixed_action": expected_action,
                },
            )
            clara_diagnostics = None
        else:
            if clara_event_diagnostics is not None:
                raise RuntimeError("非CLARA方法不得伪装CLARA support/guardrail诊断")
            clara_diagnostics = None
        expected_lower = self._selected_values(facts, actions, "candidate_lower")
        expected_upper = self._selected_values(facts, actions, "candidate_upper")
        lower = expected_lower if selected_lower is None else _finite(
            selected_lower, "selected_lower"
        )
        upper = expected_upper if selected_upper is None else _finite(
            selected_upper, "selected_upper"
        )
        lower = _finite(lower, "selected_lower")
        upper = _finite(upper, "selected_upper")
        if len(lower) != len(facts) or len(upper) != len(facts):
            raise ValueError("流式评估端点长度失配")
        if not np.allclose(lower, expected_lower, atol=1e-12, rtol=0.0) or not np.allclose(
            upper, expected_upper, atol=1e-12, rtol=0.0
        ):
            raise RuntimeError("流式评估override端点与selected_action候选端点失配")
        if np.any(lower > upper):
            raise ValueError("流式评估端点倒置")
        target = facts["target_after_maturity"].to_numpy(dtype=np.float64)
        schedule = facts["schedule_proxy"].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(target).all()
            or not np.isfinite(schedule).all()
        ):
            raise ValueError("流式评估target/schedule必须有限")
        scale = self.cadence_minutes / 60.0
        reserve_up = scale * np.maximum(upper - schedule, 0.0)
        reserve_down = scale * np.maximum(schedule - lower, 0.0)
        miss_upper = scale * np.maximum(target - upper, 0.0)
        miss_lower = scale * np.maximum(lower - target, 0.0)
        errf = (
            price.theta[0] * reserve_up
            + price.theta[1] * reserve_down
            + price.theta[2] * miss_upper
            + price.theta[3] * miss_lower
        )
        covered = ((lower <= target) & (target <= upper)).astype(np.int64)
        width = upper - lower
        alpha = 1.0 - facts["target_coverage"].to_numpy(dtype=np.float64)
        if np.any(alpha <= 0.0):
            raise ValueError("区间分数alpha非正")
        interval_score = width + (2.0 / alpha) * (
            np.maximum(lower - target, 0.0) + np.maximum(target - upper, 0.0)
        )
        if not np.isfinite(
            np.column_stack(
                [
                    reserve_up,
                    reserve_down,
                    miss_upper,
                    miss_lower,
                    errf,
                    width,
                    interval_score,
                ]
            )
        ).all():
            raise RuntimeError("流式评估指标含非有限值")
        if np.any(
            np.column_stack(
                [
                    reserve_up,
                    reserve_down,
                    miss_upper,
                    miss_lower,
                    errf,
                    width,
                    interval_score,
                ]
            )
            < 0.0
        ):
            raise RuntimeError("流式评估暴露/损失/宽度含负值")
        metric = facts.loc[:, list(self.CELL_FIELDS)].copy()
        metric["event_count"] = 1
        metric["errf_sum"] = errf
        metric["reserve_up_sum"] = reserve_up
        metric["reserve_down_sum"] = reserve_down
        metric["miss_upper_sum"] = miss_upper
        metric["miss_lower_sum"] = miss_lower
        metric["covered_sum"] = covered
        metric["coverage_target_sum"] = facts["target_coverage"].to_numpy(
            dtype=np.float64
        )
        metric["coverage_gap_sum"] = (
            covered
            - facts["target_coverage"].to_numpy(dtype=np.float64)
        )
        metric["width_sum"] = width
        metric["interval_score_sum"] = interval_score
        numeric_columns = [
            column for column in metric.columns if column not in self.CELL_FIELDS
        ]
        cell = (
            metric.groupby(list(self.CELL_FIELDS), sort=True, as_index=False)[
                numeric_columns
            ]
            .sum()
            .reset_index(drop=True)
        )
        cell.insert(0, "method", method_id)
        for field in reversed(POLICY_SCOPE_FIELDS):
            cell.insert(0, field, str(scope_identity[field]))

        action_frame = facts.loc[:, list(self.CELL_FIELDS)].copy()
        action_frame["selected_action"] = actions
        action_frame["action_count"] = 1
        action_counts = (
            action_frame.groupby(
                [*self.CELL_FIELDS, "selected_action"], sort=True, as_index=False
            )["action_count"]
            .sum()
            .reset_index(drop=True)
        )
        action_counts.insert(0, "method", method_id)
        for field in reversed(POLICY_SCOPE_FIELDS):
            action_counts.insert(0, field, str(scope_identity[field]))

        block = facts.loc[
            :,
            [
                "zone_or_farm",
                "predictor",
                "horizon_steps",
                "seed",
                "target_coverage",
                "issue_timestamp",
                "ramp_state",
            ],
        ].copy()
        block["_utc_day"] = pd.to_datetime(
            block["issue_timestamp"], errors="raise", utc=True
        ).dt.floor("D")
        block["event_count"] = 1
        block["errf_sum"] = errf
        block["covered_sum"] = covered
        block["width_sum"] = width
        blocks = (
            block.groupby(list(self.BLOCK_FIELDS), sort=True, as_index=False)[
                ["event_count", "errf_sum", "covered_sum", "width_sum"]
            ]
            .sum()
            .reset_index(drop=True)
        )
        blocks.insert(0, "method", method_id)
        for field in reversed(POLICY_SCOPE_FIELDS):
            blocks.insert(0, field, str(scope_identity[field]))

        diagnostic_metrics = self._diagnostic_rows(
            policy_scope=scope_identity,
            method=method_id,
            facts=facts,
            actions=actions,
            errf=errf,
            reserve_up=reserve_up,
            reserve_down=reserve_down,
            miss_upper=miss_upper,
            miss_lower=miss_lower,
            covered=covered,
            width=width,
            interval_score=interval_score,
            clara_diagnostics=clara_diagnostics,
        )
        if clara_diagnostics is None:
            clara_state_counts = pd.DataFrame()
        else:
            clara_state_frame = facts.loc[
                :,
                [
                    "zone_or_farm",
                    "predictor",
                    "horizon_steps",
                    "seed",
                    "target_coverage",
                    "ramp_state",
                ],
            ].reset_index(drop=True)
            clara_state_frame = pd.concat(
                [clara_state_frame, clara_diagnostics.reset_index(drop=True)],
                axis=1,
            )
            clara_state_frame["event_count"] = 1
            clara_state_counts = (
                clara_state_frame.groupby(
                    list(self.CLARA_STATE_COUNT_FIELDS),
                    sort=True,
                    dropna=False,
                    as_index=False,
                )["event_count"]
                .sum()
                .reset_index(drop=True)
            )
            clara_state_counts.insert(0, "method", method_id)
            for field in reversed(POLICY_SCOPE_FIELDS):
                clara_state_counts.insert(0, field, str(scope_identity[field]))

        output_schemas = (
            ("cell_metrics", cell, self.CELL_METRIC_COLUMNS),
            ("action_counts", action_counts, self.ACTION_COUNT_COLUMNS),
            ("paired_blocks", blocks, self.PAIRED_BLOCK_COLUMNS),
            (
                "diagnostic_metrics",
                diagnostic_metrics,
                self.DIAGNOSTIC_METRIC_COLUMNS,
            ),
        )
        if not clara_state_counts.empty:
            output_schemas = (
                *output_schemas,
                (
                    "clara_state_counts",
                    clara_state_counts,
                    self.CLARA_STATE_COUNT_COLUMNS,
                ),
            )
        for output_name, output_frame, expected_columns in output_schemas:
            if tuple(output_frame.columns) != tuple(expected_columns):
                raise RuntimeError(
                    f"流式评估compact exact schema失配: {output_name}"
                )

        signature = event_id_signature(
            facts["event_id"], include_canonical_sha256=True
        )
        key = (
            str(scope_identity["policy_mode"]),
            str(scope_identity["evaluation_price_id"]),
            str(scope_identity["policy_price_id"]),
            method_id,
        )
        if self.expected_partition_signatures:
            if partition_id is None:
                raise RuntimeError("正式流式评估必须提供可信partition_id")
            partition_key = str(partition_id)
            if partition_key not in self.expected_partition_signatures:
                raise RuntimeError(f"未登记的评估partition: {partition_key}")
            seen = self._seen_partitions.setdefault(key, set())
            if partition_key in seen:
                raise RuntimeError(
                    f"同一方法×价格partition重复计入: {key}/{partition_key}"
                )
            expected_signature = self.expected_partition_signatures[partition_key]
            for field in (
                "event_count",
                "hash_sum_u64",
                "hash_xor_u64",
                "hash_square_sum_u64",
                "canonical_sorted_event_id_sha256",
            ):
                if str(signature[field]) != str(expected_signature[field]):
                    raise RuntimeError(
                        f"评估partition事件签名失配: {partition_key}/{field}"
                    )
            seen.add(partition_key)
        target_signature = self._conservation.setdefault(
            key,
            {
                "event_count": 0,
                "hash_sum_u64": 0,
                "hash_xor_u64": 0,
                "hash_square_sum_u64": 0,
            },
        )
        self._combine_signature(target_signature, signature)
        self._emit("cell_metrics", cell)
        self._emit("action_counts", action_counts)
        self._emit("paired_blocks", blocks)
        self._emit("diagnostic_metrics", diagnostic_metrics)
        if not clara_state_counts.empty:
            self._emit("clara_state_counts", clara_state_counts)
        self._update_count += 1
        self._input_event_score_count += len(facts)
        return {
            "status": "PASS",
            **{field: str(scope_identity[field]) for field in POLICY_SCOPE_FIELDS},
            "method": method_id,
            "event_count": len(facts),
            "cell_row_count": len(cell),
            "action_count_row_count": len(action_counts),
            "paired_block_row_count": len(blocks),
            "diagnostic_metric_row_count": len(diagnostic_metrics),
            "clara_state_count_row_count": len(clara_state_counts),
            "event_signature": signature,
            "event_rows_retained": 0,
        }

    @staticmethod
    def _collapse_parts(
        parts: list[pd.DataFrame], keys: Sequence[str]
    ) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame()
        combined = pd.concat(parts, ignore_index=True, sort=False)
        numeric = [column for column in combined.columns if column not in keys]
        return (
            combined.groupby(list(keys), sort=True, as_index=False)[numeric]
            .sum()
            .reset_index(drop=True)
        )

    @staticmethod
    def _concat_unique_parts(
        parts: list[pd.DataFrame], keys: Sequence[str]
    ) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame()
        combined = pd.concat(parts, ignore_index=True, sort=False)
        if combined.duplicated(list(keys)).any():
            raise RuntimeError("非线性诊断compact键重复，禁止求和掩盖")
        return combined.sort_values(list(keys), kind="mergesort").reset_index(drop=True)

    @classmethod
    def _validate_diagnostic_outputs(
        cls,
        *,
        diagnostics: pd.DataFrame,
        clara_states: pd.DataFrame,
        cell_metrics: pd.DataFrame,
        action_counts: pd.DataFrame,
    ) -> None:
        """Close all non-additive diagnostics against additive sufficient stats."""

        if (
            tuple(diagnostics.columns) != cls.DIAGNOSTIC_METRIC_COLUMNS
            or tuple(clara_states.columns) != cls.CLARA_STATE_COUNT_COLUMNS
            or diagnostics.empty
            or clara_states.empty
        ):
            raise RuntimeError("诊断compact exact schema/空值失配")
        key = [*POLICY_SCOPE_FIELDS, "method", *cls.DIAGNOSTIC_FIELDS[:-1]]
        if diagnostics.duplicated([*key, "regime"]).any():
            raise RuntimeError("诊断compact聚合键重复")
        count_fields = (
            "event_count",
            "covered_sum",
            "rolling_window_size",
            "rolling_window_count",
            "towr_exceedance_count",
            "tuwr_exceedance_count",
            "support_applicable_count",
            "support_backoff_count",
            "guardrail_applicable_count",
            "guardrail_any_exclusion_count",
            "guardrail_excluded_action_count",
            "guardrail_empty_count",
            "selected_action_guardrail_fail_count",
        )
        counts = diagnostics.loc[:, list(count_fields)].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(counts).all()
            or not np.array_equal(counts, np.floor(counts))
            or np.any(counts < 0.0)
            or np.any(diagnostics["event_count"].to_numpy(dtype=np.int64) <= 0)
            or not diagnostics["rolling_window_size"].eq(168).all()
            or np.any(diagnostics["covered_sum"] > diagnostics["event_count"])
            or np.any(
                diagnostics["towr_exceedance_count"]
                > diagnostics["rolling_window_count"]
            )
            or np.any(
                diagnostics["tuwr_exceedance_count"]
                > diagnostics["rolling_window_count"]
            )
        ):
            raise RuntimeError("诊断compact计数边界不闭合")
        target_sum = (
            diagnostics["target_coverage"].to_numpy(dtype=np.float64)
            * diagnostics["event_count"].to_numpy(dtype=np.float64)
        )
        if (
            not np.allclose(
                diagnostics["coverage_target_sum"], target_sum, atol=1e-12, rtol=0.0
            )
            or not np.allclose(
                diagnostics["coverage_gap_sum"],
                diagnostics["covered_sum"].to_numpy(dtype=np.float64) - target_sum,
                atol=1e-12,
                rtol=0.0,
            )
        ):
            raise RuntimeError("诊断coverage target/gap代数不闭合")
        rolling = diagnostics["rolling_window_count"].to_numpy(dtype=np.int64)
        has_window = rolling > 0
        for value, numerator in (
            ("TOWR", "towr_exceedance_count"),
            ("TUWR", "tuwr_exceedance_count"),
            ("ARD", "rolling_absolute_deviation_sum"),
        ):
            values = diagnostics[value].to_numpy(dtype=np.float64)
            if (
                not np.array_equal(np.isnan(values), ~has_window)
                or not np.isfinite(values[has_window]).all()
                or not np.allclose(
                    values[has_window],
                    diagnostics.loc[has_window, numerator].to_numpy(dtype=np.float64)
                    / rolling[has_window],
                    atol=1e-12,
                    rtol=0.0,
                )
            ):
                raise RuntimeError(f"最终selected-sequence {value}不可由充分统计恢复")
        entropy = diagnostics[
            ["action_entropy_nats", "action_entropy_normalized"]
        ].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(entropy).all()
            or np.any(entropy < -1e-15)
            or np.any(entropy[:, 1] > 1.0 + 1e-15)
            or not np.allclose(
                entropy[:, 1], entropy[:, 0] / math.log(len(SIX_ACTIONS)), atol=1e-12
            )
        ):
            raise RuntimeError("动作熵充分统计不闭合")
        clara_mask = diagnostics["method"].astype(str).eq("CLARA_6A").to_numpy()
        event_count = diagnostics["event_count"].to_numpy(dtype=np.int64)
        support_applicable = diagnostics["support_applicable_count"].to_numpy(
            dtype=np.int64
        )
        guardrail_applicable = diagnostics["guardrail_applicable_count"].to_numpy(
            dtype=np.int64
        )
        rate_specs = (
            ("support_backoff_rate", "support_backoff_count", support_applicable),
            (
                "guardrail_any_exclusion_rate",
                "guardrail_any_exclusion_count",
                guardrail_applicable,
            ),
            (
                "guardrail_action_exclusion_rate",
                "guardrail_excluded_action_count",
                guardrail_applicable * len(SIX_ACTIONS),
            ),
            ("guardrail_empty_rate", "guardrail_empty_count", guardrail_applicable),
            (
                "selected_action_guardrail_fail_rate",
                "selected_action_guardrail_fail_count",
                guardrail_applicable,
            ),
        )
        if (
            not np.array_equal(support_applicable[clara_mask], event_count[clara_mask])
            or not np.array_equal(
                guardrail_applicable[clara_mask], event_count[clara_mask]
            )
            or np.any(support_applicable[~clara_mask] != 0)
            or np.any(guardrail_applicable[~clara_mask] != 0)
        ):
            raise RuntimeError("support/guardrail applicability不闭合")
        for rate_field, count_field, denominator in rate_specs:
            values = diagnostics[rate_field].to_numpy(dtype=np.float64)
            applicable = denominator > 0
            if (
                not np.array_equal(np.isnan(values), ~applicable)
                or not np.allclose(
                    values[applicable],
                    diagnostics.loc[applicable, count_field].to_numpy(dtype=np.float64)
                    / denominator[applicable],
                    atol=1e-12,
                    rtol=0.0,
                )
            ):
                raise RuntimeError(f"support/guardrail rate不可恢复: {rate_field}")

        additive = (
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
        )
        cell_key = [*POLICY_SCOPE_FIELDS, "method", *cls.CELL_FIELDS[:-1]]
        stratified = diagnostics[
            diagnostics["regime"].astype(str).isin(("ordinary", "ramp"))
        ].copy()
        stratified = stratified.rename(columns={"regime": "ramp_state"})
        joined = stratified[[*cell_key, "ramp_state", *additive]].merge(
            cell_metrics[[*cell_key, "ramp_state", *additive]],
            on=[*cell_key, "ramp_state"],
            how="outer",
            validate="one_to_one",
            suffixes=("_diagnostic", "_cell"),
        )
        if joined.isna().any().any() or any(
            not np.allclose(
                joined[f"{field}_diagnostic"],
                joined[f"{field}_cell"],
                atol=1e-10,
                rtol=1e-12,
            )
            for field in additive
        ):
            raise RuntimeError("诊断ramp与cell additive充分统计失配")
        overall = diagnostics[diagnostics["regime"].astype(str).eq("overall")]
        collapsed = cell_metrics.groupby(cell_key, sort=True, as_index=False)[
            list(additive)
        ].sum()
        joined = overall[[*cell_key, *additive]].merge(
            collapsed,
            on=cell_key,
            how="outer",
            validate="one_to_one",
            suffixes=("_diagnostic", "_cell"),
        )
        if joined.isna().any().any() or any(
            not np.allclose(
                joined[f"{field}_diagnostic"],
                joined[f"{field}_cell"],
                atol=1e-10,
                rtol=1e-12,
            )
            for field in additive
        ):
            raise RuntimeError("诊断overall从stratified cell不可恢复")

        state_numeric = clara_states[
            [
                "full_state_code",
                "support_backoff_level",
                "guardrail_safe_action_count",
                "guardrail_excluded_action_count",
                "event_count",
            ]
        ].to_numpy(dtype=np.float64)
        if (
            set(clara_states["method"].astype(str)) != {"CLARA_6A"}
            or not np.isfinite(state_numeric).all()
            or not np.array_equal(state_numeric, np.floor(state_numeric))
            or np.any(state_numeric < 0.0)
            or np.any(clara_states["event_count"].to_numpy(dtype=np.int64) <= 0)
            or np.any(
                clara_states["guardrail_safe_action_count"].to_numpy(dtype=np.int64)
                + clara_states["guardrail_excluded_action_count"].to_numpy(
                    dtype=np.int64
                )
                != len(SIX_ACTIONS)
            )
            or not np.array_equal(
                clara_states["guardrail_empty_fallback"].astype(bool).to_numpy(),
                clara_states["guardrail_safe_action_count"].eq(0).to_numpy(),
            )
            or not np.array_equal(
                clara_states["selected_action_guardrail_pass"].astype(bool).to_numpy(),
                clara_states["guardrail_safe_action_count"].gt(0).to_numpy(),
            )
            or sorted(set(clara_states["selected_action"].astype(str)) - set(SIX_ACTIONS))
        ):
            raise RuntimeError("CLARA full-state/support/guardrail counts不闭合")

        # Recompute CLARA support/guardrail counts from full-state sufficient rows
        # for every overall/ordinary/ramp diagnostic cell.
        state = clara_states.copy()
        state["support_applicable_count"] = state["event_count"]
        state["support_backoff_count"] = (
            state["support_backoff_level"].astype(int).gt(0) * state["event_count"]
        )
        state["guardrail_applicable_count"] = state["event_count"]
        state["guardrail_any_exclusion_count"] = (
            state["guardrail_excluded_action_count"].astype(int).gt(0)
            * state["event_count"]
        )
        state["guardrail_excluded_action_count"] = (
            state["guardrail_excluded_action_count"].astype(int)
            * state["event_count"]
        )
        state["guardrail_empty_count"] = (
            state["guardrail_empty_fallback"].astype(bool) * state["event_count"]
        )
        state["selected_action_guardrail_fail_count"] = (
            ~state["selected_action_guardrail_pass"].astype(bool)
        ) * state["event_count"]
        diagnostic_count_fields = (
            "support_applicable_count",
            "support_backoff_count",
            "guardrail_applicable_count",
            "guardrail_any_exclusion_count",
            "guardrail_excluded_action_count",
            "guardrail_empty_count",
            "selected_action_guardrail_fail_count",
        )
        state_key = [*POLICY_SCOPE_FIELDS, "method", *cls.DIAGNOSTIC_FIELDS[:-1]]
        state_stratified = state.rename(columns={"ramp_state": "regime"}).groupby(
            [*state_key, "regime"], sort=True, as_index=False
        )[list(diagnostic_count_fields)].sum()
        state_overall = state.groupby(state_key, sort=True, as_index=False)[
            list(diagnostic_count_fields)
        ].sum()
        state_overall["regime"] = "overall"
        reconstructed = pd.concat(
            [state_overall, state_stratified], ignore_index=True, sort=False
        )
        clara_diagnostic = diagnostics[clara_mask]
        joined = clara_diagnostic[
            [*state_key, "regime", *diagnostic_count_fields]
        ].merge(
            reconstructed,
            on=[*state_key, "regime"],
            how="outer",
            validate="one_to_one",
            suffixes=("_diagnostic", "_state"),
        )
        if joined.isna().any().any() or any(
            not np.array_equal(
                joined[f"{field}_diagnostic"].to_numpy(dtype=np.int64),
                joined[f"{field}_state"].to_numpy(dtype=np.int64),
            )
            for field in diagnostic_count_fields
        ):
            raise RuntimeError("CLARA state counts与support/guardrail rates不闭合")

        # Recompute per-cell action entropy from the independent action-count table.
        action_base_key = [
            *POLICY_SCOPE_FIELDS,
            "method",
            *cls.DIAGNOSTIC_FIELDS[:-1],
        ]
        entropy_rows: list[dict[str, Any]] = []
        for regime in ("overall", "ordinary", "ramp"):
            source = (
                action_counts
                if regime == "overall"
                else action_counts[
                    action_counts["ramp_state"].astype(str).eq(regime)
                ]
            )
            for values, group in source.groupby(action_base_key, sort=True, dropna=False):
                expanded = np.repeat(
                    group["selected_action"].astype(str).to_numpy(),
                    group["action_count"].to_numpy(dtype=np.int64),
                )
                entropy_nats, entropy_normalized = cls._action_entropy(expanded)
                entropy_rows.append(
                    {
                        **dict(zip(action_base_key, values)),
                        "regime": regime,
                        "action_entropy_nats_expected": entropy_nats,
                        "action_entropy_normalized_expected": entropy_normalized,
                    }
                )
        entropy_expected = pd.DataFrame(entropy_rows)
        joined = diagnostics[
            [
                *action_base_key,
                "regime",
                "action_entropy_nats",
                "action_entropy_normalized",
            ]
        ].merge(
            entropy_expected,
            on=[*action_base_key, "regime"],
            how="outer",
            validate="one_to_one",
        )
        if joined.isna().any().any() or not np.allclose(
            joined["action_entropy_nats"],
            joined["action_entropy_nats_expected"],
            atol=1e-12,
            rtol=0.0,
        ) or not np.allclose(
            joined["action_entropy_normalized"],
            joined["action_entropy_normalized_expected"],
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError("诊断action entropy与action counts不闭合")

    def finalize_price_child(
        self,
        *,
        price: PriceSpec,
        policy_mode: str = POLICY_MODE_RESELECTED,
        policy_price: PriceSpec | None = None,
        expected_event_count: int,
        expected_event_signature: Mapping[str, Any],
        expected_methods: Sequence[str] = FORMAL_METHODS,
        linucb_replay_result: LinUCBReplayResult | None = None,
    ) -> StreamingMetricResult:
        """Formally close one resumable ``zone x seed x price`` metric child.

        This is not a non-formal escape hatch.  It retains every formal gate that
        is meaningful at price-child scope: the exact price identity, all nine
        methods, the complete trusted five-horizon partition set, and the full
        target event count/signature.  The existing :meth:`finalize` remains the
        only global 9-method x 5-price x 23,024,760-event seal.
        """

        if not self.formal_identity:
            raise RuntimeError("逐价正式封存要求formal accumulator")
        price.validate()
        scope_identity = policy_evaluation_scope(
            policy_mode=policy_mode,
            evaluation_price=price,
            policy_price=policy_price,
            require_formal_identity=True,
        )
        resolved_policy_price = formal_price_spec(
            str(scope_identity["policy_price_id"])
        )
        if self._formal_policy_scope != scope_identity:
            raise RuntimeError("finalize与update的policy mode/双价格身份串线")
        if tuple(expected_methods) != FORMAL_METHODS:
            raise RuntimeError("逐价正式封存必须包含冻结九方法")
        event_count = int(expected_event_count)
        signature_fields = {
            "event_count",
            "hash_sum_u64",
            "hash_xor_u64",
            "hash_square_sum_u64",
            "canonical_sorted_event_id_sha256",
        }
        if (
            event_count <= 0
            or set(expected_event_signature) != signature_fields
            or int(expected_event_signature["event_count"]) != event_count
            or not _is_sha256(
                expected_event_signature["canonical_sorted_event_id_sha256"]
            )
            or not self.expected_partition_signatures
            or len(self.expected_partition_signatures)
            != len(FORMAL_HORIZONS) * len(FORMAL_PREDICTORS) * len(FORMAL_COVERAGES)
        ):
            raise RuntimeError(
                "逐价正式封存缺少完整target事件数/签名/5h×4pred×11coverage partition证明"
            )
        trusted_partition_total = {
            "event_count": 0,
            "hash_sum_u64": 0,
            "hash_xor_u64": 0,
            "hash_square_sum_u64": 0,
        }
        for signature in self.expected_partition_signatures.values():
            self._combine_signature(trusted_partition_total, signature)
        for field in (
            "event_count",
            "hash_sum_u64",
            "hash_xor_u64",
            "hash_square_sum_u64",
        ):
            if int(trusted_partition_total[field]) != int(
                expected_event_signature[field]
            ):
                raise RuntimeError(
                    f"逐价正式封存的partition与完整target签名失配: {field}"
                )
        expected_keys = {
            (
                str(scope_identity["policy_mode"]),
                str(scope_identity["evaluation_price_id"]),
                str(scope_identity["policy_price_id"]),
                method,
            )
            for method in FORMAL_METHODS
        }
        if set(self._conservation) != expected_keys:
            raise RuntimeError(
                f"逐价正式方法矩阵不闭合: missing={sorted(expected_keys-set(self._conservation))}"
            )
        expected_partitions = set(self.expected_partition_signatures)
        conservation_rows: list[dict[str, Any]] = []
        for key in sorted(expected_keys):
            if self._seen_partitions.get(key, set()) != expected_partitions:
                raise RuntimeError(
                    f"逐价正式五时距partition不闭合: {key}/"
                    f"missing={sorted(expected_partitions-self._seen_partitions.get(key,set()))}"
                )
            signature = self._conservation[key]
            for field in (
                "event_count",
                "hash_sum_u64",
                "hash_xor_u64",
                "hash_square_sum_u64",
            ):
                if int(signature[field]) != int(expected_event_signature[field]):
                    raise RuntimeError(
                        f"逐价正式完整target事件守恒失败: {key}/{field}"
                    )
            conservation_rows.append(
                {
                    "policy_mode": key[0],
                    "evaluation_price_id": key[1],
                    "policy_price_id": key[2],
                    "method": key[3],
                    **signature,
                }
            )
        conservation = pd.DataFrame(conservation_rows)
        if self.retain_reduced_rows:
            cell = self._collapse_parts(
                self._cell_parts,
                [*POLICY_SCOPE_FIELDS, "method", *self.CELL_FIELDS],
            )
            actions = self._collapse_parts(
                self._action_parts,
                [
                    *POLICY_SCOPE_FIELDS,
                    "method",
                    *self.CELL_FIELDS,
                    "selected_action",
                ],
            )
            blocks = self._collapse_parts(
                self._block_parts,
                [*POLICY_SCOPE_FIELDS, "method", *self.BLOCK_FIELDS],
            )
            diagnostics = self._concat_unique_parts(
                self._diagnostic_parts,
                [*POLICY_SCOPE_FIELDS, "method", *self.DIAGNOSTIC_FIELDS],
            )
            clara_states = self._collapse_parts(
                self._clara_state_parts,
                [*POLICY_SCOPE_FIELDS, "method", *self.CLARA_STATE_COUNT_FIELDS],
            )
            totals = {
                "cell": cell.groupby("method")["event_count"].sum(),
                "actions": actions.groupby("method")["action_count"].sum(),
                "blocks": blocks.groupby("method")["event_count"].sum(),
            }
            if any(
                set(values.index.astype(str)) != set(FORMAL_METHODS)
                or not values.eq(event_count).all()
                for values in totals.values()
            ):
                raise RuntimeError("逐价正式compact cell/action/block事件数不守恒")
            overall = diagnostics[diagnostics["regime"].astype(str).eq("overall")]
            stratified = diagnostics[
                diagnostics["regime"].astype(str).isin(("ordinary", "ramp"))
            ]
            if (
                set(diagnostics["method"].astype(str)) != set(FORMAL_METHODS)
                or set(diagnostics["regime"].astype(str))
                - {"overall", "ordinary", "ramp"}
                or len(overall)
                != len(FORMAL_METHODS)
                * len(FORMAL_HORIZONS)
                * len(FORMAL_PREDICTORS)
                * len(FORMAL_COVERAGES)
                or not overall.groupby("method")["event_count"].sum().eq(event_count).all()
                or not stratified.groupby("method")["event_count"].sum().eq(event_count).all()
                or set(clara_states["method"].astype(str)) != {"CLARA_6A"}
                or int(clara_states["event_count"].sum()) != event_count
            ):
                raise RuntimeError(
                    "逐价正式滚动可靠性/状态诊断充分统计不闭合"
                )
            self._validate_diagnostic_outputs(
                diagnostics=diagnostics,
                clara_states=clara_states,
                cell_metrics=cell,
                action_counts=actions,
            )
            if set(self._method_policy_identities) != set(FORMAL_METHODS):
                raise RuntimeError("逐价正式九方法policy capability矩阵不闭合")
        else:
            cell = pd.DataFrame()
            actions = pd.DataFrame()
            blocks = pd.DataFrame()
            diagnostics = pd.DataFrame()
            clara_states = pd.DataFrame()
        audit = {
            "schema": SCHEMA,
            "status": "PASS",
            "scope": "ZONE_SEED_PRICE_CHILD",
            "policy_mode": str(scope_identity["policy_mode"]),
            "evaluation_price_id": str(scope_identity["evaluation_price_id"]),
            "evaluation_price_sha256": str(
                scope_identity["evaluation_price_sha256"]
            ),
            "policy_price_id": str(scope_identity["policy_price_id"]),
            "policy_price_sha256": str(scope_identity["policy_price_sha256"]),
            "policy_evaluation_scope_sha256": str(
                scope_identity["policy_evaluation_scope_sha256"]
            ),
            "policy_feedback_uses_evaluation_price": False,
            "endpoint_score_uses_evaluation_price": True,
            "method_count": len(FORMAL_METHODS),
            "methods": list(FORMAL_METHODS),
            "trusted_partition_count": len(expected_partitions),
            "trusted_partition_ids": sorted(expected_partitions),
            "trusted_partition_signatures_sha256": _canonical_digest(
                {
                    partition_id: self.expected_partition_signatures[partition_id]
                    for partition_id in sorted(expected_partitions)
                }
            ),
            "evaluation_zone": (
                None if self._formal_child_scope is None else self._formal_child_scope[0]
            ),
            "seed": (
                None if self._formal_child_scope is None else self._formal_child_scope[1]
            ),
            "expected_event_count_per_method": event_count,
            "expected_event_signature": dict(expected_event_signature),
            "update_count": self._update_count,
            "input_method_event_score_count": self._input_event_score_count,
            "event_rows_retained": 0,
            "retained_reduced_rows": bool(self.retain_reduced_rows),
            "cell_metric_row_count": len(cell),
            "action_count_row_count": len(actions),
            "paired_block_row_count": len(blocks),
            "diagnostic_metric_row_count": len(diagnostics),
            "clara_state_count_row_count": len(clara_states),
            "event_conservation_row_count": len(conservation),
            "rolling_reliability_contract": (
                "SELECTED_ENDPOINT_COVERED_SEQUENCE_SORTED_BY_ISSUE_TIMESTAMP_EVENT_ID;"
                "168H_WINDOW; OVERALL_ORDINARY_RAMP_RECOMPUTED_PER_METHOD_CELL"
            ),
            "candidate_historical_reliability_used_as_final_metric": False,
            "diagnostic_rates_recoverable_from_integer_sufficient_statistics": True,
            "method_policy_identities": {
                method: self._method_policy_identities[method]
                for method in FORMAL_METHODS
            },
            "full_target_event_set_preserved": True,
            "terminal_pending_feedback_rows_dropped": False,
            "linucb_feedback_release_rule": "label_available_timestamp <= issue_time; no within-issue use",
        }
        linucb_replay_audit = None
        if linucb_replay_result is not None:
            linucb_capability_identity = verify_formal_linucb_result_capability(
                linucb_replay_result, resolved_policy_price.price_id
            )
            registered_linucb = self._method_policy_identities.get("LinUCB_6A")
            if registered_linucb is None or (
                str(registered_linucb.get("linucb_decisions_sha256"))
                != str(linucb_capability_identity["decisions_sha256"])
                or str(
                    registered_linucb.get("linucb_result_capability_sha256")
                )
                != str(linucb_capability_identity["capability_sha256"])
            ):
                raise RuntimeError(
                    "LinUCB update动作来源与finalize replay capability串线"
                )
            linucb_replay_audit = dict(
                linucb_replay_result.audit_by_price[resolved_policy_price.price_id]
            )
            if str(linucb_replay_audit.get("retained_decisions_sha256")) != str(
                linucb_capability_identity["decisions_sha256"]
            ):
                raise RuntimeError("LinUCB replay audit未绑定实际决策内容")
        result = StreamingMetricResult(
            cell_metrics=cell,
            action_counts=actions,
            paired_blocks=blocks,
            diagnostic_metrics=diagnostics,
            clara_state_counts=clara_states,
            event_conservation=conservation,
            audit=audit,
            linucb_replay_audit=(
                None
                if linucb_replay_audit is None
                else linucb_replay_audit
            ),
        )
        capability_identity = _metric_result_capability_identity(result)
        return StreamingMetricResult(
            cell_metrics=result.cell_metrics,
            action_counts=result.action_counts,
            paired_blocks=result.paired_blocks,
            diagnostic_metrics=result.diagnostic_metrics,
            clara_state_counts=result.clara_state_counts,
            event_conservation=result.event_conservation,
            audit=result.audit,
            linucb_replay_audit=result.linucb_replay_audit,
            capability_identity=capability_identity,
            _capability=_FORMAL_METRIC_RESULT_CAPABILITY,
        )

    def finalize(
        self,
        *,
        expected_methods: Sequence[str] | None = None,
        expected_policy_modes: Sequence[str] | None = None,
        expected_prices: Sequence[str] | None = None,
        expected_event_count_per_method_price: int | None = None,
        expected_event_signature_by_price: Mapping[str, Mapping[str, Any]] | None = None,
        formal_identity: bool = True,
    ) -> StreamingMetricResult:
        if bool(formal_identity) != self.formal_identity:
            raise RuntimeError("流式评估constructor/finalize正式身份标志失配")
        conservation_rows = [
            {
                "policy_mode": policy_mode,
                "evaluation_price_id": evaluation_price_id,
                "policy_price_id": policy_price_id,
                "method": method,
                **signature,
            }
            for (
                policy_mode,
                evaluation_price_id,
                policy_price_id,
                method,
            ), signature in sorted(self._conservation.items())
        ]
        conservation = pd.DataFrame(conservation_rows)
        if formal_identity and (
            expected_methods is None
            or expected_policy_modes is None
            or expected_prices is None
            or expected_event_count_per_method_price is None
            or expected_event_signature_by_price is None
        ):
            raise RuntimeError(
                "正式评估finalize必须提供方法/价格/事件数/可信event signature"
            )
        if formal_identity:
            formal_price_ids = tuple(row[0] for row in FORMAL_PRICE_ROWS)
            signature_fields = {
                "event_count",
                "hash_sum_u64",
                "hash_xor_u64",
                "hash_square_sum_u64",
                "canonical_sorted_event_id_sha256",
            }
            if (
                tuple(expected_methods or ()) != FORMAL_METHODS
                or tuple(expected_policy_modes or ()) != FORMAL_POLICY_MODES
                or tuple(expected_prices or ()) != formal_price_ids
                or int(expected_event_count_per_method_price or -1) != 23_024_760
                or set(expected_event_signature_by_price or {})
                != set(formal_price_ids)
                or any(
                    set(signature) != signature_fields
                    or int(signature["event_count"]) != 23_024_760
                    or not str(signature["canonical_sorted_event_id_sha256"])
                    for signature in (expected_event_signature_by_price or {}).values()
                )
            ):
                raise RuntimeError(
                    "正式评估不允许缩小九方法×双mode×五价×23,024,760事件闭合矩阵"
                )
        if expected_methods is not None and expected_prices is not None:
            modes = tuple(expected_policy_modes or (POLICY_MODE_RESELECTED,))
            expected_keys = {
                (
                    str(mode),
                    str(price),
                    (
                        str(price)
                        if str(mode) == POLICY_MODE_RESELECTED
                        else FORMAL_MAIN_PRICE_ID
                    ),
                    str(method),
                )
                for mode in modes
                for price in expected_prices
                for method in expected_methods
            }
            observed_keys = set(
                conservation[[*POLICY_SCOPE_FIELDS, "method"]].itertuples(
                    index=False, name=None
                )
            )
            if observed_keys != expected_keys:
                raise RuntimeError(
                    f"评估方法×价格闭合失败: missing={sorted(expected_keys-observed_keys)}"
                )
        if expected_event_count_per_method_price is not None and not conservation.empty:
            if not conservation["event_count"].eq(
                int(expected_event_count_per_method_price)
            ).all():
                raise RuntimeError("评估方法×价格事件数不守恒")
        if expected_event_signature_by_price is not None and not conservation.empty:
            for row in conservation.itertuples(index=False):
                if str(row.evaluation_price_id) not in expected_event_signature_by_price:
                    raise RuntimeError(
                        f"缺少可信评估价格event signature: {row.evaluation_price_id}"
                    )
                expected_signature = expected_event_signature_by_price[
                    str(row.evaluation_price_id)
                ]
                for field in (
                    "event_count",
                    "hash_sum_u64",
                    "hash_xor_u64",
                    "hash_square_sum_u64",
                ):
                    if int(getattr(row, field)) != int(expected_signature[field]):
                        raise RuntimeError(
                            f"方法事件集与可信endpoint签名失配: "
                            f"{row.policy_mode}/{row.evaluation_price_id}/"
                            f"{row.policy_price_id}/{row.method}/{field}"
                        )
        if not conservation.empty:
            signature_fields = [
                "event_count",
                "hash_sum_u64",
                "hash_xor_u64",
                "hash_square_sum_u64",
            ]
            for child_scope, part in conservation.groupby(
                list(POLICY_SCOPE_FIELDS), sort=True
            ):
                if len(part[signature_fields].drop_duplicates()) != 1:
                    raise RuntimeError(
                        f"同mode×评估价各方法事件集不一致: {child_scope}"
                    )
        if formal_identity and self.expected_partition_signatures:
            expected_partition_ids = set(self.expected_partition_signatures)
            for key in set(self._conservation):
                observed_partition_ids = self._seen_partitions.get(key, set())
                if observed_partition_ids != expected_partition_ids:
                    raise RuntimeError(
                        f"方法×价格partition闭合失败: {key}/"
                        f"missing={sorted(expected_partition_ids-observed_partition_ids)}"
                    )
        if self.retain_reduced_rows:
            cell = self._collapse_parts(
                self._cell_parts,
                [*POLICY_SCOPE_FIELDS, "method", *self.CELL_FIELDS],
            )
            actions = self._collapse_parts(
                self._action_parts,
                [*POLICY_SCOPE_FIELDS, "method", *self.CELL_FIELDS, "selected_action"],
            )
            blocks = self._collapse_parts(
                self._block_parts,
                [*POLICY_SCOPE_FIELDS, "method", *self.BLOCK_FIELDS],
            )
            diagnostics = self._concat_unique_parts(
                self._diagnostic_parts,
                [*POLICY_SCOPE_FIELDS, "method", *self.DIAGNOSTIC_FIELDS],
            )
            clara_states = self._collapse_parts(
                self._clara_state_parts,
                [*POLICY_SCOPE_FIELDS, "method", *self.CLARA_STATE_COUNT_FIELDS],
            )
        else:
            cell = pd.DataFrame()
            actions = pd.DataFrame()
            blocks = pd.DataFrame()
            diagnostics = pd.DataFrame()
            clara_states = pd.DataFrame()
        audit = {
            "schema": SCHEMA,
            "status": "PASS",
            "update_count": self._update_count,
            "input_method_event_score_count": self._input_event_score_count,
            "event_rows_retained": 0,
            "retained_reduced_rows": bool(self.retain_reduced_rows),
            "cell_metric_row_count": len(cell),
            "action_count_row_count": len(actions),
            "paired_block_row_count": len(blocks),
            "diagnostic_metric_row_count": len(diagnostics),
            "clara_state_count_row_count": len(clara_states),
            "event_conservation_row_count": len(conservation),
            "paired_test_input": (
                "UTC_DAY_METHOD_SUMS_JOINED_WITHIN_PRICE_ZONE_PREDICTOR_HORIZON_"
                "SEED_COVERAGE_RAMP; COARSER_LEVELS_DERIVED_BY_EXACT_SUMMATION"
            ),
        }
        return StreamingMetricResult(
            cell_metrics=cell,
            action_counts=actions,
            paired_blocks=blocks,
            diagnostic_metrics=diagnostics,
            clara_state_counts=clara_states,
            event_conservation=conservation,
            audit=audit,
        )


def readonly_small_sample_preflight(
    *,
    source_fact_path: Path,
    endpoint_addon_path: Path,
    event_core_path: Path,
    feedback_trace_path: Path | None = None,
    width_thresholds_path: Path,
    heldout_zone: str,
    prices: Sequence[PriceSpec],
    sample_rows: int = 512,
) -> dict[str, Any]:
    """Read-only schema/alignment/component preflight for one retained stream.

    Missing endpoint cache returns ``BLOCKED`` (never ``PASS``).  Content or causal
    contract mismatches raise immediately.  The function never writes an artifact.
    """

    import pyarrow.parquet as pq

    fact_path = Path(source_fact_path)
    addon_path = Path(endpoint_addon_path)
    core_path = Path(event_core_path)
    feedback_path = (
        Path(feedback_trace_path)
        if feedback_trace_path is not None
        else core_path.parent / "feedback_trace.parquet"
    )
    threshold_path = Path(width_thresholds_path)
    for label, path in (
        ("source_fact", fact_path),
        ("event_core", core_path),
        ("feedback_trace", feedback_path),
        ("width_thresholds", threshold_path),
    ):
        if not path.is_file():
            return {
                "schema": SCHEMA,
                "status": "BLOCKED",
                "ready": False,
                "reason": f"{label.upper()}_MISSING",
                "path": str(path),
                "read_only": True,
            }
    all_fact_ids = pd.read_parquet(fact_path, columns=["event_id"])
    all_fact_ids["event_id"] = all_fact_ids["event_id"].astype(str)
    if all_fact_ids["event_id"].duplicated().any():
        raise RuntimeError("小样本preflight检测到S06 event_id重复")
    feedback = pd.read_parquet(
        feedback_path,
        columns=["event_id", "action", "eligible_by_strict_time_rule"],
    )
    feedback["event_id"] = feedback["event_id"].astype(str)
    feedback["action"] = feedback["action"].astype(str)
    eligible = feedback[feedback["eligible_by_strict_time_rule"].astype(bool)].copy()
    if eligible[["event_id", "action"]].duplicated().any():
        raise RuntimeError("小样末preflight严格成熟反馈键重复")
    action_sets = eligible.groupby("event_id", sort=False)["action"].agg(
        lambda values: tuple(sorted(set(values)))
    )
    expected_actions = tuple(sorted(SIX_ACTIONS[:4]))
    if not action_sets.map(lambda values: values == expected_actions).all():
        raise RuntimeError("小样本preflight含非四动作成熟反馈")
    mature_ids = set(action_sets.index.astype(str)) & set(
        all_fact_ids["event_id"].astype(str)
    )
    raw_s06_event_count = len(all_fact_ids)
    strict_mature_event_count = len(mature_ids)
    s06_unmatured_event_count = raw_s06_event_count - strict_mature_event_count
    if len(eligible[eligible["event_id"].isin(mature_ids)]) != strict_mature_event_count * 4:
        raise RuntimeError("小样本preflight严格成熟四动作反馈不守恒")
    parquet = pq.ParquetFile(fact_path)
    batches = parquet.iter_batches(batch_size=max(1, int(sample_rows)))
    try:
        facts = next(batches).to_pandas()
    except StopIteration as error:
        raise RuntimeError("小样本preflight源事件为空") from error
    facts["event_id"] = facts["event_id"].astype(str)
    facts = facts[facts["event_id"].isin(mature_ids)].copy()
    if facts.empty:
        facts = pd.read_parquet(fact_path)
        facts["event_id"] = facts["event_id"].astype(str)
        facts = facts[facts["event_id"].isin(mature_ids)].head(
            max(1, int(sample_rows))
        )
    fact_required = {
        "event_id",
        "zone_or_farm",
        "predictor",
        "horizon_steps",
        "seed",
        "target_coverage",
        "horizon_group",
        "raw_width_value",
        *STATE_FIELDS[:-1],
    }
    for action in SIX_ACTIONS[:4]:
        fact_required.update(
            {
                f"{action}__candidate_lower",
                f"{action}__candidate_upper",
                f"{action}__covered",
                f"{action}__tuwr_indicator",
                f"{action}__ard_value",
            }
        )
    _require_columns(facts, fact_required, "小样本S06源事件")
    if not addon_path.is_file():
        return {
            "schema": SCHEMA,
            "status": "BLOCKED",
            "ready": False,
            "reason": "ENDPOINT_ADDON_MISSING",
            "source_fact_path": str(fact_path),
            "endpoint_addon_path": str(addon_path),
            "sample_event_count": len(facts),
            "source_fact_schema_pass": True,
            "strict_maturity_filter_applied": True,
            "mature_action_count": 4,
            "raw_s06_event_count": raw_s06_event_count,
            "strict_mature_event_count": strict_mature_event_count,
            "s06_unmatured_event_count": s06_unmatured_event_count,
            "complete_case_event_id_sha256": _canonical_digest(sorted(mature_ids)),
            "read_only": True,
        }
    event_ids = set(facts["event_id"].astype(str))
    addon_columns = [
        "event_id",
        *(
            f"{action}__{field}"
            for action in SIX_ACTIONS[4:]
            for field in (
                "candidate_lower",
                "candidate_upper",
                "covered",
                "tuwr_indicator",
                "ard_value",
            )
        ),
    ]
    addons = pd.read_parquet(addon_path, columns=addon_columns)
    addons = addons[addons["event_id"].astype(str).isin(event_ids)].copy()
    if len(addons) != len(facts) or addons["event_id"].duplicated().any():
        raise RuntimeError("小样本新增端点事件映射不完整")
    core = pd.read_parquet(core_path, columns=["event_id", "target", "base_center"])
    core = core[core["event_id"].astype(str).isin(event_ids)].copy()
    if len(core) != len(facts) or core["event_id"].duplicated().any():
        raise RuntimeError("小样本event_core事件映射不完整")
    combined = facts.merge(addons, on="event_id", how="left", validate="one_to_one")
    combined = combined.merge(
        core.rename(
            columns={"target": "target_after_maturity", "base_center": "schedule_proxy"}
        ),
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    thresholds = pd.read_parquet(threshold_path)
    seed = int(combined["seed"].iloc[0])
    predictor = str(combined["predictor"].iloc[0])
    threshold_keys = ["predictor", "horizon_group", "target_coverage", "seed"]
    selected_thresholds = validated_fold_width_thresholds(
        thresholds,
        combined,
        outer_heldout_zone=str(heldout_zone),
        predictor=predictor,
        seed=seed,
    )
    combined = combined.merge(
        selected_thresholds,
        on=threshold_keys,
        how="left",
        validate="many_to_one",
    )
    if combined[["raw_width_q33", "raw_width_q67"]].isna().any().any():
        raise RuntimeError("小样本折级宽度阈值映射不完整")
    if combined["heldout_zone_in_threshold"].astype(bool).any():
        raise RuntimeError("小样本宽度阈值混入持出区")
    width = combined["raw_width_value"].to_numpy(dtype=np.float64)
    combined["raw_width_state"] = np.where(
        width <= combined["raw_width_q33"].to_numpy(dtype=np.float64),
        "narrow",
        np.where(
            width <= combined["raw_width_q67"].to_numpy(dtype=np.float64),
            "medium",
            "wide",
        ),
    )
    target = combined["target_after_maturity"].to_numpy(dtype=np.float64)
    schedule = combined["schedule_proxy"].to_numpy(dtype=np.float64)
    component_maximum = 0.0
    component_by_action: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for action in SIX_ACTIONS:
        lower = combined[f"{action}__candidate_lower"].to_numpy(dtype=np.float64)
        upper = combined[f"{action}__candidate_upper"].to_numpy(dtype=np.float64)
        capacity, miss = endpoint_components(
            target=target,
            schedule=schedule,
            lower=lower,
            upper=upper,
        )
        component_by_action[action] = (capacity, miss)
        component_maximum = max(
            component_maximum, float(np.max(capacity)), float(np.max(miss))
        )
        expected_covered = (lower <= target) & (target <= upper)
        if not np.array_equal(
            expected_covered,
            combined[f"{action}__covered"].astype(bool).to_numpy(),
        ):
            raise RuntimeError(f"小样本覆盖标记回归失败: {action}")
    price_rows = tuple(prices)
    import gefcom_six_action_endpoint_contract_v1 as endpoint_contract

    price_regression_maximum = 0.0
    for price in price_rows:
        price.validate()
        attached = attach_price_losses(combined, price, copy_frame=True)
        for action in SIX_ACTIONS:
            direct = endpoint_contract.endpoint_errf(
                target=target,
                schedule=schedule,
                lower=combined[f"{action}__candidate_lower"].to_numpy(
                    dtype=np.float64
                ),
                upper=combined[f"{action}__candidate_upper"].to_numpy(
                    dtype=np.float64
                ),
                theta=price.theta,
            )
            observed = attached[f"{action}__errf"].to_numpy(dtype=np.float64)
            delta = float(np.max(np.abs(observed - direct)))
            price_regression_maximum = max(price_regression_maximum, delta)
            if delta > 1e-12:
                raise RuntimeError(
                    f"五价compact分解与权威endpoint_errf失配: {price.price_id}/{action}"
                )
    base_prices = [price for price in price_rows if price.base_submission_price]
    if len(base_prices) != 1:
        raise RuntimeError("小样本preflight必须有exact一个base submission price")
    base_price = base_prices[0]
    base_stored_regression_maximum = 0.0
    for action in SIX_ACTIONS[:4]:
        capacity, miss = component_by_action[action]
        observed = (
            base_price.capacity_weight * capacity
            + base_price.miss_weight * miss
        )
        stored = combined[f"{action}__errf"].to_numpy(dtype=np.float64)
        delta = float(np.max(np.abs(observed - stored)))
        base_stored_regression_maximum = max(
            base_stored_regression_maximum, delta
        )
        if delta > 1e-12:
            raise RuntimeError(f"旧四动作stored ERRF回归失配: {action}")
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "ready": True,
        "read_only": True,
        "source_fact_path": str(fact_path),
        "endpoint_addon_path": str(addon_path),
        "event_core_path": str(core_path),
        "feedback_trace_path": str(feedback_path),
        "width_thresholds_path": str(threshold_path),
        "heldout_zone": str(heldout_zone),
        "sample_event_count": len(combined),
        "raw_s06_event_count": raw_s06_event_count,
        "strict_mature_event_count": strict_mature_event_count,
        "s06_unmatured_event_count": s06_unmatured_event_count,
        "complete_case_event_id_sha256": _canonical_digest(sorted(mature_ids)),
        "actions": list(SIX_ACTIONS),
        "price_ids": [row.price_id for row in price_rows],
        "maximum_unweighted_component": component_maximum,
        "schedule_proxy_minimum": float(np.min(schedule)),
        "schedule_proxy_maximum": float(np.max(schedule)),
        "five_price_endpoint_errf_maximum_error": price_regression_maximum,
        "base_price_stored_errf_maximum_error": base_stored_regression_maximum,
        "writes_performed": 0,
    }


__all__ = [
    "SCHEMA",
    "SIX_ACTIONS",
    "FORMAL_HORIZONS",
    "FORMAL_PREDICTORS",
    "FORMAL_ZONES",
    "FORMAL_COVERAGES",
    "FORMAL_METHODS",
    "FORMAL_DETERMINISTIC_METHOD_ACTIONS",
    "FORMAL_POLICY_MODES",
    "POLICY_MODE_RESELECTED",
    "POLICY_MODE_FIXED_MAIN",
    "FORMAL_MAIN_PRICE_ID",
    "POLICY_SCOPE_FIELDS",
    "FORMAL_PRICE_ROWS",
    "FROZEN_REGISTRY_PATH",
    "FROZEN_REGISTRY_SHA256",
    "FOUR_VERSION_CONFIG_PATH",
    "FOUR_VERSION_CONFIG_SHA256",
    "PriceSpec",
    "SixActionSufficientStats",
    "SixActionStatsBuilder",
    "VerifiedCausalSourceAudit",
    "ClaraPreparation",
    "ClaraFitResult",
    "CartFitResult",
    "PriceFitChildResult",
    "verify_formal_fit_result_capability",
    "LinUCBReplayResult",
    "verify_formal_linucb_result_capability",
    "StreamingMetricResult",
    "verify_formal_metric_result_capability",
    "StreamingMetricAccumulator",
    "ClaraEventDiagnosticResult",
    "CartEventActionResult",
    "algorithm_dependency_closure",
    "action_library_sha256",
    "global_replay_scope_contract",
    "global_replay_scope_sha256",
    "frozen_zone_baseline_contract",
    "clara_v4_adaptive_contract",
    "validate_formal_v4_configuration",
    "price_sha256",
    "linucb_policy_identity",
    "fit_price_child_identity",
    "replay_price_child_identity",
    "replay_policy_evaluation_child_identity",
    "assert_formal_price_grid",
    "assert_formal_price_subset",
    "price_specs_from_config",
    "formal_price_spec",
    "policy_evaluation_scope",
    "six_action_contracts",
    "derived_six_action_contract_identity",
    "state_universe_from_contract",
    "endpoint_components",
    "attach_price_losses",
    "validated_fold_width_thresholds",
    "validate_endpoint_root_seal",
    "load_causal_source_stream",
    "assert_formal_gefcom_identity",
    "prepare_clara_fit",
    "fit_price_conditioned_clara",
    "fit_all_price_conditioned_clara",
    "fit_price_conditioned_cart",
    "fit_all_price_conditioned_cart",
    "fit_price_conditioned_selectors",
    "fit_all_price_conditioned_selectors",
    "fit_parent_identity",
    "predict_clara_actions",
    "predict_clara_actions_with_diagnostics",
    "predict_cart_actions",
    "predict_cart_actions_with_capability",
    "event_id_signature",
    "rolling_reliability_sufficient_stats",
    "replay_all_price_conditioned_linucb",
    "projected_output_bytes",
    "assert_no_full_event_materialization",
    "readonly_small_sample_preflight",
]
