"""GEFCom 2014 六动作、十一覆盖率、五价格的版本化实验入口。

当前修订实现三个可独立验收的部分：冻结输入/环境预检、完整任务账本、
以及 150 个价格无关的新增动作端点缓存。价格条件化的 CLARA/CART 拟合、
LinUCB 回放和最终聚合保留为显式未实现阶段；本脚本不会把骨架标成完成。

所有新工件只写入 ``测试CLARA/results_raw/gefcom_six_action_price_full_v1``。
任何已存在但不完整或身份不匹配的单元都会触发 fail-closed，不会被覆盖。
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import re
import shutil
import sys
import time
import traceback
import textwrap
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from gefcom_six_action_endpoint_contract_v1 import (
    aligned_endpoints as _aligned_endpoints,
    endpoint_errf,
    reliability_arrays as _endpoint_reliability_arrays,
)
TEST_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = TEST_ROOT.parent
PACKAGE_ROOT = REVISION_ROOT.parents[5]
RUNNER_PATH = Path(__file__).resolve()
CONFIG_PATH = TEST_ROOT / "configs" / "test_clara_gefcom_six_action_full_v1.json"
OUTPUT_ROOT = TEST_ROOT / "results_raw" / "gefcom_six_action_price_full_v1"
ENDPOINT_ROOT = OUTPUT_ROOT / "endpoint_units"
CONTROL_ROOT = OUTPUT_ROOT / "control"

AUTH_CODE = REVISION_ROOT / "_权威代码" / "code"
BASELINE_SCRIPTS = REVISION_ROOT / "基线实验补充" / "scripts"
S03_ROOT = REVISION_ROOT / "03_基础预测与候选区间重建" / "results_raw" / "full_rebuild_v1"
S06_FACT_ROOT = (
    REVISION_ROOT
    / "06_基线实现与训练区调参"
    / "results_raw"
    / "nested_source_selection_v1"
)
REGISTRY_PATH = (
    REVISION_ROOT
    / "06_基线实现与训练区调参"
    / "results_verified"
    / "s07_execution_registry_v2"
    / "s07_baseline_execution_registry.parquet"
)

ORIGINAL_ACTIONS = ("Static", "ACI", "AgACI", "EnbPI_RH")
ADDON_ACTIONS = ("TunedSingleConformal", "EqualEndpointEnsemble")
HARD_METHODS = (
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
HARD_COVERAGES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
HARD_HORIZONS = (1, 3, 6, 12, 24)
BASE_PRICE_THETA = (3.56, 3.56, 20.0, 20.0)
TSC_FROZEN_FAMILY = "SAOCP"
TSC_FROZEN_CONFIGURATION_ID = "996f625def857d7772788a87f7a8e4ebe87202ded973bcde2462122dde17cf7d"
TSC_FROZEN_CONFIGURATION = {
    "learning_rate": 0.05,
    "temperature": 0.1,
    "window_hours": [24.0, 48.0, 96.0, 168.0],
}
ENDPOINT_BUILD_IMPLEMENTATION_ENABLED = True
APPROVED_TRAINING_STATUS = "AUTHOR_APPROVED_B_FROZEN"
ACTION_LIBRARY_CONTRACT = {
    "schema": "TEST_CLARA_6A_ACTION_LIBRARY_IDENTITY_V1",
    "tie_order": [*ORIGINAL_ACTIONS, *ADDON_ACTIONS],
    "action_count": 6,
}
ACTION_LIBRARY_SHA256 = "e8cc20e0ee0c76f63a78188ee7d2557037aa75994c866f997773e14695e9aa24"
ENDPOINT_CODE_CLOSURE_FUNCTIONS = (
    "preflight",
    "sha256_file",
    "canonical_json_bytes",
    "load_json",
    "endpoint_unit_id",
    "load_endpoint_config",
    "validate_endpoint_runtime_paths",
    "validate_endpoint_scope_contract",
    "endpoint_execution_scope_authorization",
    "validate_endpoint_task_scope_authorization",
    "issue_preflight_gate_token",
    "tuned_single_conformal_registry_audit",
    "artifact_hash_audit",
    "migration_sha256_anchor_index",
    "migration_relative_key",
    "_legacy_anchor_metadata",
    "_validate_legacy_reference_row",
    "legacy_reference_anchor_audit",
    "legacy_reference_record",
    "endpoint_unit_source_anchor_audit",
    "expected_endpoint_unit_event_count",
    "_baseline_module",
    "_reliability_arrays",
    "_legacy_regression",
    "_complete_endpoint_manifest",
    "validate_preflight_gate_token",
    "run_endpoint_unit",
    "endpoint_status",
    "seal_endpoint_root",
)


def _selector_core_module() -> Any:
    """Lazy import keeps endpoint-only plan/preflight independent of selector source edits."""

    return importlib.import_module("gefcom_six_action_selector_core_v1")


def selector_core_action_library_sha256(*args: Any, **kwargs: Any) -> str:
    return str(_selector_core_module().action_library_sha256(*args, **kwargs))


def selector_core_algorithm_dependency_closure() -> dict[str, Any]:
    return dict(_selector_core_module().algorithm_dependency_closure())


def selector_core_clara_v4_adaptive_contract() -> dict[str, Any]:
    return dict(_selector_core_module().clara_v4_adaptive_contract())


def selector_core_frozen_zone_baseline_contract(zone: str, baseline_id: str) -> dict[str, Any]:
    return dict(_selector_core_module().frozen_zone_baseline_contract(zone, baseline_id))


def selector_core_global_replay_scope_contract() -> dict[str, Any]:
    return dict(_selector_core_module().global_replay_scope_contract())


def selector_core_global_replay_scope_sha256() -> str:
    return str(_selector_core_module().global_replay_scope_sha256())


def selector_core_price_sha256(price: Any) -> str:
    return str(_selector_core_module().price_sha256(price))


def selector_core_price_specs_from_config(config: dict[str, Any]) -> list[Any]:
    return list(_selector_core_module().price_specs_from_config(config))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_identity_sha256(payload: Any) -> str:
    """Match the selector core's versioned identity digest (no trailing newline)."""

    return sha256_bytes(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    temporary.write_bytes(canonical_json_bytes(payload))
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    frame.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
    os.replace(temporary, path)


def load_config() -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    validate_scope_contract(config)
    return config


def load_endpoint_config() -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    validate_endpoint_scope_contract(config)
    validate_endpoint_runtime_paths(config)
    return config


def config_identity() -> dict[str, str]:
    return {
        "config_sha256": sha256_file(CONFIG_PATH),
        "runner_sha256": sha256_file(RUNNER_PATH),
    }


def endpoint_code_closure_identity() -> dict[str, Any]:
    """AST-expand every runner-local dependency from endpoint roots and bind globals."""

    functions: dict[str, str] = {}
    referenced_globals: set[str] = set()
    pending = list(ENDPOINT_CODE_CLOSURE_FUNCTIONS)
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        function = globals().get(name)
        if function is None or not inspect.isfunction(function):
            raise RuntimeError(f"端点代码闭包函数不存在或不是Python函数: {name}")
        try:
            source = textwrap.dedent(inspect.getsource(function))
            tree = ast.parse(source)
        except (OSError, TypeError, SyntaxError) as error:
            raise RuntimeError(f"无法封存端点函数源码: {name}") from error
        seen.add(name)
        functions[name] = sha256_bytes(source.encode("utf-8"))
        loaded_names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        for referenced in loaded_names:
            value = globals().get(referenced)
            if inspect.isfunction(value) and referenced not in seen:
                pending.append(referenced)
            if function.__module__ == __name__ and referenced.isupper() and referenced in globals():
                referenced_globals.add(referenced)

    def normalize_constant(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value.resolve())
        if isinstance(value, dict):
            return {str(key): normalize_constant(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize_constant(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        raise RuntimeError(f"端点闭包全局常量不可规范化: {type(value).__name__}")

    constants = {
        name: normalize_constant(globals()[name])
        for name in sorted(referenced_globals)
    }
    return {
        "root_functions": list(ENDPOINT_CODE_CLOSURE_FUNCTIONS),
        "function_sha256": dict(sorted(functions.items())),
        "function_count": len(functions),
        "constants": constants,
        "constant_count": len(constants),
        "closure_sha256": sha256_bytes(
            canonical_json_bytes({"functions": functions, "constants": constants})
        ),
    }


def endpoint_artifact_identity(config: dict[str, Any]) -> dict[str, Any]:
    contract = config["endpoint_contract"]
    module_path = TEST_ROOT / str(contract["algorithm_module"])
    if not module_path.is_file():
        raise FileNotFoundError(f"稳定端点算法模块不存在: {module_path}")
    dependency_hashes: dict[str, str] = {}
    for input_id in contract["identity_input_ids"]:
        spec = config["inputs"][str(input_id)]
        dependency_hashes[str(input_id)] = str(spec["sha256"])
    return {
        "endpoint_contract_sha256": canonical_identity_sha256(contract),
        "regression_gate_sha256": canonical_identity_sha256(config["regression_gate"]),
        "endpoint_algorithm_module_sha256": sha256_file(module_path),
        "endpoint_runner_code_closure": endpoint_code_closure_identity(),
        "endpoint_dependency_hashes": dependency_hashes,
        "endpoint_dependency_set_sha256": canonical_identity_sha256(dependency_hashes),
    }


def endpoint_execution_scope_authorization(config: dict[str, Any]) -> dict[str, Any]:
    """Stable endpoint-only execution scope; deliberately excludes selector/replay code."""

    scope = config["scope"]
    training = config["training_support_identity"]
    endpoint = config["endpoint_contract"]
    payload = {
        "schema": "TEST_CLARA_GEFCOM_6A_ENDPOINT_EXECUTION_SCOPE_AUTHORIZATION_V1",
        "training_support_status": training["status"],
        "chosen_training_support_identity": training["chosen_identity"],
        "author_approved": bool(training["author_approved"]),
        "scientific_execution_allowed": bool(training["scientific_execution_allowed"]),
        "author_approval_record": training["author_approval_record"],
        "option_horizons": training["option_horizons"],
        "zones": scope["zones"],
        "seeds": scope["seeds"],
        "predictors": scope["predictors"],
        "evaluation_horizons": scope["horizons"],
        "target_coverages": scope["target_coverages"],
        "endpoint_actions": endpoint["actions"],
        "regression_horizons": endpoint["regression_horizons"],
        "endpoint_build_enabled": bool(config["execution"]["endpoint_build_enabled"]),
    }
    return {
        "payload": payload,
        "sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def selector_core_artifact_identity() -> dict[str, Any]:
    module_path = TEST_ROOT / "scripts" / "gefcom_six_action_selector_core_v1.py"
    closure = selector_core_algorithm_dependency_closure()
    return {
        "selector_core_module_sha256": sha256_file(module_path),
        "selector_algorithm_closure_sha256": closure["algorithm_closure_sha256"],
        "selector_algorithm_dependency_count": int(closure["dependency_count"]),
        "action_library_sha256": selector_core_action_library_sha256(),
        "global_replay_scope_sha256": selector_core_global_replay_scope_sha256(),
    }


def validate_endpoint_scope_contract(config: dict[str, Any]) -> None:
    """Validate only frozen endpoint axes/authorization; never import selector core."""

    scope = config["scope"]
    training_contract = config["training_support_identity"]
    if (
        training_contract.get("status") != APPROVED_TRAINING_STATUS
        or training_contract.get("chosen_identity")
        != "B_UNIFIED_FIVE_HORIZON_NEW_IDENTITY"
    ):
        raise RuntimeError("作者批准的 B 训练支持身份状态失配")
    approval_record = training_contract.get("author_approval_record") or {}
    if (
        training_contract.get("author_approved") is not True
        or training_contract.get("scientific_execution_allowed") is not True
        or approval_record.get("status") != "APPROVED"
        or approval_record.get("approved_option")
        != "B_UNIFIED_FIVE_HORIZON_NEW_IDENTITY"
        or approval_record.get("approved_by") != "author"
        or not str(approval_record.get("approved_at_utc") or "").strip()
        or not str(approval_record.get("scope_statement") or "").strip()
    ):
        raise RuntimeError("作者批准记录已撤销、缺失或与 B 身份不一致")
    observed_option_horizons = {
        str(key): tuple(int(value) for value in values)
        for key, values in training_contract.get("option_horizons", {}).items()
    }
    if observed_option_horizons != {
        "A_FULL_24_HORIZON_SOURCE_SUPPORT": tuple(range(1, 25)),
        "B_UNIFIED_FIVE_HORIZON_NEW_IDENTITY": HARD_HORIZONS,
    }:
        raise RuntimeError("A/B 训练支持时距集合失配")
    zones = tuple(str(value) for value in scope["zones"])
    predictors = tuple(str(value) for value in scope["predictors"])
    seeds = tuple(int(value) for value in scope["seeds"])
    horizons = tuple(int(value) for value in scope["horizons"])
    coverages = tuple(float(value) for value in scope["target_coverages"])
    actions = tuple(str(value) for value in scope["actions"])
    price_grid = config["price_grid"]
    if zones != tuple(f"zone{value}" for value in range(1, 11)):
        raise RuntimeError("GEFCom 区域矩阵不是冻结的 zone1-zone10")
    if predictors != ("Ridge", "GBR", "MLP", "QRLSTM") or seeds != (0, 1, 2):
        raise RuntimeError("预测器或随机种子矩阵失配")
    if horizons != HARD_HORIZONS or coverages != HARD_COVERAGES:
        raise RuntimeError("五预测时距或十一覆盖率矩阵失配")
    if actions != (*ORIGINAL_ACTIONS, *ADDON_ACTIONS):
        raise RuntimeError("六动作矩阵失配")
    action_identity = config["selector_contract"]["linucb_six_action_identity"]
    if (
        action_identity.get("action_library_contract") != ACTION_LIBRARY_CONTRACT
        or action_identity.get("ordered_action_library_sha256") != ACTION_LIBRARY_SHA256
        or canonical_identity_sha256(ACTION_LIBRARY_CONTRACT) != ACTION_LIBRARY_SHA256
    ):
        raise RuntimeError("稳定六动作库合同身份失配")
    observed_price_ids = tuple(str(row["price_id"]) for row in price_grid)
    if observed_price_ids != ("R01", "R02", "R05P6179775281", "R10", "R20"):
        raise RuntimeError("五价格矩阵失配")
    base_prices = [
        str(row["price_id"])
        for row in price_grid
        if bool(row.get("base_submission_price", False))
    ]
    if base_prices != ["R05P6179775281"]:
        raise RuntimeError("基准提交价格标志失配")
    expected_endpoint_counts = {
        str(key): int(value)
        for key, value in scope["expected_endpoint_unit_count_by_training_identity"].items()
    }
    if expected_endpoint_counts != {
        "A_FULL_24_HORIZON_SOURCE_SUPPORT": len(zones) * len(seeds) * 24,
        "B_UNIFIED_FIVE_HORIZON_NEW_IDENTITY": len(zones) * len(seeds) * len(horizons),
    }:
        raise RuntimeError("A/B 训练身份的端点缓存单元数合同失配")
    endpoint_contract = config["endpoint_contract"]
    if (
        tuple(endpoint_contract["actions"]) != ADDON_ACTIONS
        or tuple(float(value) for value in endpoint_contract["target_coverages"]) != HARD_COVERAGES
        or not bool(config["execution"]["endpoint_build_enabled"])
        or not ENDPOINT_BUILD_IMPLEMENTATION_ENABLED
    ):
        raise RuntimeError("端点合同或禁用门失配")
    regression_gate = config["regression_gate"]
    if (
        tuple(float(value) for value in endpoint_contract.get("regression_coverages", []))
        != (0.5, 0.9, 0.99)
        or tuple(int(value) for value in endpoint_contract.get("regression_horizons", []))
        != HARD_HORIZONS
        or tuple(float(value) for value in regression_gate.get("coverages", []))
        != (0.5, 0.9, 0.99)
        or tuple(str(value) for value in regression_gate.get("methods", [])) != ADDON_ACTIONS
        or float(regression_gate.get("endpoint_and_base_price_metric_absolute_tolerance", -1.0))
        != 1e-12
        or regression_gate.get("reference_relative_path")
        != "基线实验补充/results_raw/baseline_supplement_v2/units"
        or regression_gate.get("verified_manifest_input_id")
        != "legacy_regression_verified_manifest"
        or regression_gate.get("verified_input_audit_input_id")
        != "legacy_regression_input_audit"
        or regression_gate.get("independent_qa_input_id")
        != "legacy_regression_independent_qa"
        or int(regression_gate.get("expected_reference_unit_count", -1)) != 150
        or int(regression_gate.get("expected_reference_unique_event_count", -1)) != 6_279_480
        or regression_gate.get("must_pass_for_every_endpoint_unit") is not True
    ):
        raise RuntimeError("旧三覆盖率回归门完整合同失配")
    storage = config["output_storage_contract"]
    if storage.get("mode") != "COMPACT_ONLY_NO_FULL_METHOD_EVENT_PRICE_MATERIALIZATION":
        raise RuntimeError("紧凑输出合同失配：禁止全量方法-事件-价格物化")
    disk = config["execution"]["disk_capacity_contract"]
    active = str(disk["active_stage"])
    if active not in disk["stage_projected_bytes"] or float(disk["safety_margin_fraction"]) < 0:
        raise RuntimeError("磁盘阶段投影合同失配")
    if config["environment"].get("formal_python_flags") != ["-s"]:
        raise RuntimeError("正式 Python 启动参数必须冻结为 -s 以隔离 user-site")


def validate_endpoint_runtime_paths(config: dict[str, Any]) -> None:
    expected = {
        "AUTH_CODE": (
            REVISION_ROOT / config["inputs"]["baseline_common_module"]["relative_path"]
        ).resolve().parent,
        "BASELINE_SCRIPTS": (
            REVISION_ROOT / config["inputs"]["baseline_supplement_runner"]["relative_path"]
        ).resolve().parent,
        "S03_ROOT": (
            REVISION_ROOT / config["inputs"]["s03_root_manifest"]["relative_path"]
        ).resolve().parent,
        "S06_FACT_ROOT": (
            REVISION_ROOT / config["inputs"]["s06_source_fact_root_manifest"]["relative_path"]
        ).resolve().parent,
        "REGISTRY_PATH": (
            REVISION_ROOT / config["inputs"]["s07_execution_registry"]["relative_path"]
        ).resolve(),
        "PACKAGE_ROOT": (
            REVISION_ROOT / config["inputs"]["migration_full_sha256_manifest"]["relative_path"]
        ).resolve().parent,
    }
    observed = {
        "AUTH_CODE": AUTH_CODE.resolve(),
        "BASELINE_SCRIPTS": BASELINE_SCRIPTS.resolve(),
        "S03_ROOT": S03_ROOT.resolve(),
        "S06_FACT_ROOT": S06_FACT_ROOT.resolve(),
        "REGISTRY_PATH": REGISTRY_PATH.resolve(),
        "PACKAGE_ROOT": PACKAGE_ROOT.resolve(),
    }
    failures = {name: (str(observed[name]), str(path)) for name, path in expected.items() if observed[name] != path}
    if failures:
        raise RuntimeError(f"端点运行时根路径偏离冻结输入位置: {failures}")


def validate_scope_contract(config: dict[str, Any]) -> None:
    validate_endpoint_scope_contract(config)
    scope = config["scope"]
    actions = tuple(str(value) for value in scope["actions"])
    endpoint_contract = config["endpoint_contract"]
    selector_contract = config["selector_contract"]
    linucb_identity = selector_contract["linucb_six_action_identity"]
    if (
        tuple(selector_contract["action_library"]) != actions
        or selector_contract["adaptive_statistics_action_scope"] != "ALL_SIX_ACTIONS"
        or "original four actions" not in selector_contract["prohibited_shortcut"]
        or bool(selector_contract["linucb_bundle_level_independence_allowed"])
    ):
        raise RuntimeError("六动作自适应选择器合同失配")
    if (
        linucb_identity["ordered_action_library_sha256"]
        != canonical_identity_sha256(linucb_identity["action_library_contract"])
        or linucb_identity["ordered_action_library_sha256"]
        != selector_core_action_library_sha256(actions)
        or linucb_identity["global_replay_scope_sha256"]
        != canonical_identity_sha256(linucb_identity["global_replay_scope_contract"])
        or linucb_identity["global_replay_scope_contract"]
        != selector_core_global_replay_scope_contract()
        or linucb_identity["global_replay_scope_sha256"]
        != selector_core_global_replay_scope_sha256()
    ):
        raise RuntimeError("六动作 LinUCB 动作库或全局回放范围身份失配")
    v4_contract = selector_core_clara_v4_adaptive_contract()
    if (
        v4_contract["source_protocol_sha256"]
        != config["inputs"]["v4_adaptive_protocol"]["sha256"]
        or v4_contract["action_library_sha256"]
        != linucb_identity["ordered_action_library_sha256"]
        or v4_contract["adaptive_statistics_action_scope"] != "ALL_SIX_ACTIONS"
    ):
        raise RuntimeError("CLARA V4 冻结子合同与六动作新身份失配")
    methods = tuple(str(value) for value in scope["methods"])
    price_grid = config["price_grid"]
    zones = tuple(str(value) for value in scope["zones"])
    seeds = tuple(int(value) for value in scope["seeds"])
    if methods != HARD_METHODS:
        raise RuntimeError("九方法矩阵失配")
    if int(scope["expected_selector_unit_count"]) != len(zones) * len(seeds):
        raise RuntimeError("选择器拟合单元数合同失配")
    if int(scope["expected_replay_unit_count"]) != len(zones) * len(seeds):
        raise RuntimeError("目标回放单元数合同失配")
    if int(scope["expected_price_child_count"]) != 2 * len(zones) * len(seeds) * len(price_grid):
        raise RuntimeError("逐价子清单数量合同失配")
    expected_scores = (
        int(scope["expected_target_event_count_per_method_price"])
        * len(methods)
        * len(price_grid)
    )
    if expected_scores != int(scope["expected_method_event_score_count"]):
        raise RuntimeError("方法-事件-价格评分总数合同失配")
    checkpoint = config["checkpoint_contract"]
    if (
        "LinUCB has no fitted policy artifact" not in checkpoint["selector_price_subcheckpoint"]
        or "globally replaying LinUCB per price" not in checkpoint["replay_unit"]
        or checkpoint.get("price_child_commit")
        != "WRITE_CHILD_TO_UNIQUE_TEMP_DIRECTORY_THEN_ATOMIC_RENAME"
        or checkpoint.get("parent_commit")
        != "INCREMENTAL_CONTAINER_WITH_SEPARATE_ATOMIC_SEAL_AFTER_ALL_FIVE_CHILD_MANIFEST_SHA256_PASS"
    ):
        raise RuntimeError("选择器/回放父子 checkpoint 语义失配")
    storage = config["output_storage_contract"]
    if str(scope["expected_method_event_score_count"]) not in str(
        storage.get("prohibited_materialization")
    ).replace(",", ""):
        raise RuntimeError("紧凑输出评分总数合同失配")


def validate_endpoint_task_scope_authorization(
    config: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    """Reject revoked authorization or a worker task outside the frozen B axes."""

    validate_endpoint_scope_contract(config)
    zone = str(task.get("zone") or "")
    try:
        seed = int(task["seed"])
        horizon = int(task["horizon"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("端点 worker 任务缺少合法 zone/seed/horizon") from error
    chosen = str(config["training_support_identity"]["chosen_identity"])
    if (
        zone not in {str(value) for value in config["scope"]["zones"]}
        or seed not in {int(value) for value in config["scope"]["seeds"]}
        or horizon not in set(training_horizons(config, chosen))
    ):
        raise RuntimeError("端点 worker 任务轴越出冻结授权 scope")
    return endpoint_execution_scope_authorization(config)


def input_hash_audit(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for input_id, spec in config["inputs"].items():
        path = REVISION_ROOT / str(spec["relative_path"])
        if not path.is_file():
            raise FileNotFoundError(f"冻结输入不存在: {path}")
        observed = sha256_file(path)
        expected = str(spec["sha256"])
        rows.append(
            {
                "input_id": input_id,
                "relative_path": str(spec["relative_path"]),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "bytes": int(path.stat().st_size),
                "status": "PASS" if observed == expected else "FAIL",
            }
        )
    audit = pd.DataFrame(rows)
    if audit["status"].ne("PASS").any():
        failures = audit[audit["status"].ne("PASS")].to_dict("records")
        raise RuntimeError(f"冻结输入哈希失配: {failures}")
    return audit


def endpoint_input_hash_audit(
    config: dict[str, Any],
    full_audit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    expected_ids = {str(value) for value in config["endpoint_contract"]["identity_input_ids"]}
    if full_audit is None:
        rows: list[dict[str, Any]] = []
        for input_id in sorted(expected_ids):
            spec = config["inputs"][input_id]
            path = REVISION_ROOT / str(spec["relative_path"])
            if not path.is_file():
                raise FileNotFoundError(f"端点冻结输入不存在: {path}")
            observed = sha256_file(path)
            expected = str(spec["sha256"])
            rows.append(
                {
                    "input_id": input_id,
                    "relative_path": str(spec["relative_path"]),
                    "expected_sha256": expected,
                    "observed_sha256": observed,
                    "bytes": int(path.stat().st_size),
                    "status": "PASS" if observed == expected else "FAIL",
                }
            )
        audit = pd.DataFrame(rows)
    else:
        audit = full_audit.copy()
    selected = audit[audit["input_id"].astype(str).isin(expected_ids)].copy()
    if set(selected["input_id"].astype(str)) != expected_ids or selected["status"].ne("PASS").any():
        raise RuntimeError("端点 gate 冻结输入子集不完整")
    return selected.sort_values("input_id", kind="stable").reset_index(drop=True)


def environment_audit(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    expected_executable = Path(str(config["environment"]["formal_python_executable"])).resolve()
    observed_executable = Path(sys.executable).resolve()
    rows.append(
        {
            "component": "python_executable",
            "expected": str(expected_executable),
            "observed": str(observed_executable),
            "status": "PASS" if observed_executable == expected_executable else "FAIL",
        }
    )
    expected_python = str(config["environment"]["python"])
    observed_python = platform.python_version()
    rows.append(
        {
            "component": "python",
            "expected": expected_python,
            "observed": observed_python,
            "status": "PASS" if observed_python == expected_python else "FAIL",
        }
    )
    for package, expected in config["environment"]["packages"].items():
        try:
            observed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            observed = "MISSING"
        rows.append(
            {
                "component": package,
                "expected": str(expected),
                "observed": observed,
                "status": "PASS" if observed == str(expected) else "FAIL",
            }
        )
    for variable, expected in config["environment"]["required_process_environment"].items():
        observed = os.environ.get(str(variable), "MISSING")
        rows.append(
            {
                "component": f"env:{variable}",
                "expected": str(expected),
                "observed": observed,
                "status": "PASS" if observed == str(expected) else "FAIL",
            }
        )
    rows.append(
        {
            "component": "python_no_user_site_flag",
            "expected": "1",
            "observed": "1" if bool(sys.flags.no_user_site) else "0",
            "status": "PASS" if bool(sys.flags.no_user_site) else "FAIL",
        }
    )
    return pd.DataFrame(rows)


def training_horizons(config: dict[str, Any], identity: str) -> tuple[int, ...]:
    mapping = config["training_support_identity"]["option_horizons"]
    if identity not in mapping:
        raise ValueError(f"未知训练支持身份: {identity}")
    return tuple(int(value) for value in mapping[identity])


def expected_stream_ids(config: dict[str, Any], horizons: Sequence[int]) -> set[str]:
    scope = config["scope"]
    return {
        f"{predictor}-H{int(horizon):02d}-{zone}-S{int(seed)}"
        for predictor in scope["predictors"]
        for zone in scope["zones"]
        for horizon in horizons
        for seed in scope["seeds"]
    }


def inventory_audit(
    config: dict[str, Any],
    *,
    horizons: Sequence[int],
    include_legacy_regression: bool,
) -> pd.DataFrame:
    expected = expected_stream_ids(config, horizons)
    source_manifests = sorted((S06_FACT_ROOT / "source_facts").glob("predictor=*/zone=*/horizon=*/seed=*/manifest.json"))
    bundle_manifests = sorted((S03_ROOT / "bundles").glob("*/*/manifest.json"))
    base_manifests = sorted((S03_ROOT / "base").glob("*/*/manifest.json"))
    legacy_root = REVISION_ROOT / str(config["regression_gate"]["reference_relative_path"])
    legacy_manifests = sorted(legacy_root.glob("*/unit_manifest.json"))

    source_ids: set[str] = set()
    bad_source_status: list[str] = []
    for path in source_manifests:
        manifest = load_json(path)
        stream_id = f"{manifest['predictor']}-H{int(manifest['horizon']):02d}-{manifest['zone']}-S{int(manifest['seed'])}"
        source_ids.add(stream_id)
        if manifest.get("status") != "COMPLETE_VALIDATED":
            bad_source_status.append(stream_id)
    bundle_ids = {path.parent.name for path in bundle_manifests}
    base_ids = {path.parent.name for path in base_manifests}
    rows: list[dict[str, Any]] = [
        {
            "inventory": "s06_selected_source_facts",
            "expected_count": len(expected),
            "observed_count": len(source_manifests),
            "missing_count": len(expected - source_ids),
            "extra_count": len(source_ids - expected),
            "bad_status_count": len(bad_source_status),
        },
        {
            "inventory": "s03_selected_candidate_bundles",
            "expected_count": len(expected),
            "observed_count": len(bundle_manifests),
            "missing_count": len(expected - bundle_ids),
            "extra_count": len(bundle_ids - expected),
            "bad_status_count": 0,
        },
        {
            "inventory": "s03_selected_base_streams",
            "expected_count": len(expected),
            "observed_count": len(base_manifests),
            "missing_count": len(expected - base_ids),
            "extra_count": len(base_ids - expected),
            "bad_status_count": 0,
        },
    ]
    if include_legacy_regression:
        expected_legacy = (
            len(config["scope"]["zones"])
            * len(config["scope"]["seeds"])
            * len(config["scope"]["horizons"])
        )
        rows.append(
            {
                "inventory": "legacy_three_coverage_regression_units",
                "expected_count": expected_legacy,
                "observed_count": len(legacy_manifests),
                "missing_count": max(0, expected_legacy - len(legacy_manifests)),
                "extra_count": max(0, len(legacy_manifests) - expected_legacy),
                "bad_status_count": 0,
            }
        )
    audit = pd.DataFrame(rows)
    audit["status"] = np.where(
        audit[["expected_count", "observed_count"]].nunique(axis=1).eq(1)
        & audit[["missing_count", "extra_count", "bad_status_count"]].sum(axis=1).eq(0),
        "PASS",
        "FAIL",
    )
    return audit


def migration_sha256_anchor_index(config: dict[str, Any]) -> dict[str, str]:
    """Load the package-root migration inventory as an independent file identity anchor."""

    spec = config["inputs"]["migration_full_sha256_manifest"]
    path = (REVISION_ROOT / str(spec["relative_path"])).resolve()
    frame = pd.read_csv(path, dtype=str)
    expected_columns = {"relative_path", "size_bytes", "sha256"}
    if set(frame.columns) != expected_columns:
        raise RuntimeError("FULL_SHA256_MANIFEST.csv 字段失配")
    if len(frame) != int(spec["expected_entry_count"]):
        raise RuntimeError("FULL_SHA256_MANIFEST.csv 条目数失配")
    keys = frame["relative_path"].astype(str).map(
        lambda value: value.replace("\\", "/").casefold()
    )
    if keys.duplicated().any() or frame["sha256"].astype(str).str.fullmatch(r"[0-9A-Fa-f]{64}").ne(True).any():
        raise RuntimeError("FULL_SHA256_MANIFEST.csv 路径重复或 SHA256 非法")
    return {
        str(key): str(value).lower()
        for key, value in zip(keys, frame["sha256"].astype(str), strict=True)
    }


def migration_relative_key(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(PACKAGE_ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(f"冻结工件越出迁移包根: {path}") from error
    return relative.as_posix().casefold()


def artifact_hash_audit(
    config: dict[str, Any],
    *,
    horizons: Sequence[int],
) -> dict[str, Any]:
    """Verify every selected S03/S06 payload, but only after inventory is complete."""

    inventory = inventory_audit(
        config,
        horizons=horizons,
        include_legacy_regression=False,
    )
    if inventory["status"].ne("PASS").any():
        return {
            "status": "NOT_RUN_INVENTORY_INCOMPLETE",
            "checked_stream_count": 0,
            "mismatch_count": 0,
        }
    migration_anchor = migration_sha256_anchor_index(config)
    checked = 0
    anchor_checked_file_count = 0
    mismatches: list[dict[str, str]] = []
    for predictor in config["scope"]["predictors"]:
        for zone in config["scope"]["zones"]:
            for horizon in horizons:
                for seed in config["scope"]["seeds"]:
                    stream_id = f"{predictor}-H{int(horizon):02d}-{zone}-S{int(seed)}"
                    source_dir = (
                        S06_FACT_ROOT
                        / "source_facts"
                        / f"predictor={predictor}"
                        / f"zone={zone}"
                        / f"horizon={int(horizon):02d}"
                        / f"seed={int(seed)}"
                    )
                    source_manifest = load_json(source_dir / "manifest.json")
                    bundle_dir = S03_ROOT / "bundles" / str(predictor) / stream_id
                    bundle_manifest = load_json(bundle_dir / "manifest.json")
                    base_dir = S03_ROOT / "base" / str(predictor) / stream_id
                    base_manifest = load_json(base_dir / "manifest.json")
                    identities = (
                        source_manifest.get("stream_id") == stream_id
                        and source_manifest.get("status") == "COMPLETE_VALIDATED"
                        and bundle_manifest.get("bundle_id") == stream_id
                        and bundle_manifest.get("status") == "COMPLETE_VALIDATED"
                        and base_manifest.get("status") == "COMPLETE_VALIDATED"
                        and str(base_manifest.get("predictor")) == str(predictor)
                        and str(base_manifest.get("zone")) == str(zone)
                        and int(base_manifest.get("horizon", -1)) == int(horizon)
                        and int(base_manifest.get("seed", -1)) == int(seed)
                    )
                    checks = {
                        "s06_manifest": (
                            source_dir / "manifest.json",
                            None,
                        ),
                        "s06_facts": (
                            source_dir / "facts.parquet",
                            str(source_manifest.get("facts_sha256")),
                        ),
                        "s03_event_core": (
                            bundle_dir / "event_core.parquet",
                            str(bundle_manifest.get("event_core_sha256")),
                        ),
                        "s03_candidate_intervals": (
                            bundle_dir / "candidate_intervals.parquet",
                            str(bundle_manifest.get("candidate_intervals_sha256")),
                        ),
                        "s03_feedback_trace": (
                            bundle_dir / "feedback_trace.parquet",
                            str(bundle_manifest.get("feedback_trace_sha256")),
                        ),
                        "s03_bundle_manifest": (
                            bundle_dir / "manifest.json",
                            None,
                        ),
                        "s03_base_manifest": (
                            base_dir / "manifest.json",
                            None,
                        ),
                        "s03_base_predictions": (
                            base_dir / "base_predictions.parquet",
                            str(base_manifest.get("output_sha256")),
                        ),
                    }
                    if not identities:
                        mismatches.append({"stream_id": stream_id, "artifact": "manifest_identity"})
                    for artifact, (path, expected_hash) in checks.items():
                        if not path.is_file():
                            mismatches.append({"stream_id": stream_id, "artifact": artifact})
                            continue
                        observed_hash = sha256_file(path)
                        if expected_hash is not None and observed_hash != expected_hash:
                            mismatches.append(
                                {"stream_id": stream_id, "artifact": f"{artifact}:leaf_manifest"}
                            )
                        anchored_hash = migration_anchor.get(migration_relative_key(path))
                        anchor_checked_file_count += 1
                        if anchored_hash is None or observed_hash != anchored_hash:
                            mismatches.append(
                                {"stream_id": stream_id, "artifact": f"{artifact}:migration_anchor"}
                            )
                    checked += 1
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "checked_stream_count": checked,
        "migration_anchor_checked_file_count": anchor_checked_file_count,
        "migration_anchor_manifest_sha256": str(
            config["inputs"]["migration_full_sha256_manifest"]["sha256"]
        ),
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:20],
    }


def tuned_single_conformal_registry_audit(config: dict[str, Any]) -> dict[str, Any]:
    """Endpoint-only TSC registry gate with no CART/LinUCB/core dependency."""

    expected_registry = (
        REVISION_ROOT / config["inputs"]["s07_execution_registry"]["relative_path"]
    ).resolve()
    if (
        REGISTRY_PATH.resolve() != expected_registry
        or sha256_file(REGISTRY_PATH)
        != str(config["inputs"]["s07_execution_registry"]["sha256"])
    ):
        raise RuntimeError("TunedSingleConformal 注册表路径或SHA偏离冻结输入")
    registry = pd.read_parquet(REGISTRY_PATH)
    tuned = registry[registry["baseline_id"].astype(str).eq("TunedSingleConformal")].copy()
    zones = set(str(value) for value in config["scope"]["zones"])
    if set(tuned["outer_heldout_zone"].astype(str)) != zones or len(tuned) != len(zones):
        raise RuntimeError("TunedSingleConformal 十折注册表范围失配")
    tuned_row_sha_by_zone: dict[str, str] = {}
    for record in tuned.to_dict(orient="records"):
        zone = str(record["outer_heldout_zone"])
        source_zones = json.loads(str(record["source_zones_json"]))
        configuration = json.loads(str(record["selected_configuration_json"]))
        selected_identity = {
            "baseline_id": "TunedSingleConformal",
            "selected_family": TSC_FROZEN_FAMILY,
            "configuration": TSC_FROZEN_CONFIGURATION,
        }
        row_payload = dict(record)
        observed_row_sha = str(row_payload.pop("registry_row_sha256"))

        def native(value: Any) -> Any:
            if isinstance(value, dict):
                return {str(key): native(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [native(item) for item in value]
            if isinstance(value, np.generic):
                return value.item()
            return value

        if (
            record.get("registry_schema") != "S07_BASELINE_EXECUTION_REGISTRY_V2"
            or record.get("execution_status") != "FROZEN_PENDING_S07"
            or not bool(record.get("deployable"))
            or int(record.get("source_zone_count", -1)) != 9
            or set(str(value) for value in source_zones) != zones - {zone}
            or str(record.get("selected_family")) != TSC_FROZEN_FAMILY
            or str(record.get("selected_configuration_id")) != TSC_FROZEN_CONFIGURATION_ID
            or configuration != TSC_FROZEN_CONFIGURATION
            or str(record.get("selected_configuration_sha256"))
            != canonical_identity_sha256(selected_identity)
            or observed_row_sha != canonical_identity_sha256(native(row_payload))
        ):
            raise RuntimeError(f"TunedSingleConformal 冻结语义或逐行身份失配: {zone}")
        tuned_row_sha_by_zone[zone] = observed_row_sha
    return {
        "status": "PASS",
        "fold_count": len(tuned),
        "family": TSC_FROZEN_FAMILY,
        "configuration_id": TSC_FROZEN_CONFIGURATION_ID,
        "configuration_json": TSC_FROZEN_CONFIGURATION,
        "selected_configuration_sha256": canonical_identity_sha256(
            {
                "baseline_id": "TunedSingleConformal",
                "selected_family": TSC_FROZEN_FAMILY,
                "configuration": TSC_FROZEN_CONFIGURATION,
            }
        ),
        "registry_row_sha256_by_zone": tuned_row_sha_by_zone,
    }


def registry_audit(config: dict[str, Any]) -> dict[str, Any]:
    registry = pd.read_parquet(REGISTRY_PATH)
    tuned = registry[registry["baseline_id"].astype(str).eq("TunedSingleConformal")].copy()
    zones = set(str(value) for value in config["scope"]["zones"])
    if set(tuned["outer_heldout_zone"].astype(str)) != zones or len(tuned) != len(zones):
        raise RuntimeError("TunedSingleConformal 十折注册表范围失配")
    family = set(tuned["selected_family"].astype(str))
    configuration_ids = set(tuned["selected_configuration_id"].astype(str))
    configuration_json = set(tuned["selected_configuration_json"].astype(str))
    if len(family) != 1 or len(configuration_ids) != 1 or len(configuration_json) != 1:
        raise RuntimeError("TunedSingleConformal 十折冻结配置不唯一")
    tuned_row_sha_by_zone: dict[str, str] = {}
    for record in tuned.to_dict(orient="records"):
        zone = str(record["outer_heldout_zone"])
        source_zones = json.loads(str(record["source_zones_json"]))
        configuration = json.loads(str(record["selected_configuration_json"]))
        selected_identity = {
            "baseline_id": "TunedSingleConformal",
            "selected_family": TSC_FROZEN_FAMILY,
            "configuration": TSC_FROZEN_CONFIGURATION,
        }
        row_payload = dict(record)
        observed_row_sha = str(row_payload.pop("registry_row_sha256"))

        def native(value: Any) -> Any:
            if isinstance(value, dict):
                return {str(key): native(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [native(item) for item in value]
            if isinstance(value, np.generic):
                return value.item()
            return value

        expected_row_sha = canonical_identity_sha256(native(row_payload))
        if (
            record.get("registry_schema") != "S07_BASELINE_EXECUTION_REGISTRY_V2"
            or record.get("execution_status") != "FROZEN_PENDING_S07"
            or not bool(record.get("deployable"))
            or int(record.get("source_zone_count", -1)) != 9
            or len(source_zones) != 9
            or set(str(value) for value in source_zones) != zones - {zone}
            or str(record.get("selected_family")) != TSC_FROZEN_FAMILY
            or str(record.get("selected_configuration_id"))
            != TSC_FROZEN_CONFIGURATION_ID
            or configuration != TSC_FROZEN_CONFIGURATION
            or str(record.get("selected_configuration_sha256"))
            != canonical_identity_sha256(selected_identity)
            or observed_row_sha != expected_row_sha
        ):
            raise RuntimeError(f"TunedSingleConformal 冻结语义或逐行身份失配: {zone}")
        tuned_row_sha_by_zone[zone] = observed_row_sha
    linucb = registry[registry["baseline_id"].astype(str).eq("LinUCB")].copy()
    if (
        len(linucb) != len(zones)
        or set(linucb["outer_heldout_zone"].astype(str)) != zones
        or linucb["outer_heldout_zone"].astype(str).duplicated().any()
    ):
        raise RuntimeError("LinUCB 十折 alpha/lambda 冻结配置范围失配")
    linucb_by_zone: dict[str, Any] = {}
    for row in linucb.itertuples(index=False):
        parsed = json.loads(str(row.selected_configuration_json))
        if set(parsed) != {"exploration_alpha", "l2_regularization"}:
            raise RuntimeError("LinUCB 冻结 alpha/lambda 字段失配")
        core_contract = selector_core_frozen_zone_baseline_contract(
            str(row.outer_heldout_zone), "LinUCB"
        )
        if (
            core_contract["configuration"] != parsed
            or core_contract["selected_configuration_id"]
            != str(row.selected_configuration_id)
        ):
            raise RuntimeError("LinUCB runner/core 逐区冻结配置失配")
        linucb_by_zone[str(row.outer_heldout_zone)] = {
            "configuration": parsed,
            "legacy_configuration_id": str(row.selected_configuration_id),
            "heldout_zone_alpha_lambda_sha256": canonical_identity_sha256(parsed),
            "frozen_configuration_sha256": core_contract["frozen_configuration_sha256"],
        }
    cart = registry[registry["baseline_id"].astype(str).eq("CARTBestAction")].copy()
    if (
        len(cart) != len(zones)
        or set(cart["outer_heldout_zone"].astype(str)) != zones
        or cart["outer_heldout_zone"].astype(str).duplicated().any()
    ):
        raise RuntimeError("CARTBestAction 十折冻结配置范围失配")
    cart_by_zone: dict[str, Any] = {}
    for row in cart.itertuples(index=False):
        parsed = json.loads(str(row.selected_configuration_json))
        if set(parsed) != {"class_weight", "max_depth", "min_samples_leaf", "random_state"}:
            raise RuntimeError("CARTBestAction 冻结配置字段失配")
        core_contract = selector_core_frozen_zone_baseline_contract(
            str(row.outer_heldout_zone), "CARTBestAction"
        )
        if (
            core_contract["configuration"] != parsed
            or core_contract["selected_configuration_id"]
            != str(row.selected_configuration_id)
        ):
            raise RuntimeError("CART runner/core 逐区冻结配置失配")
        cart_by_zone[str(row.outer_heldout_zone)] = {
            "configuration": parsed,
            "legacy_configuration_id": str(row.selected_configuration_id),
            "heldout_zone_cart_configuration_sha256": canonical_identity_sha256(parsed),
            "frozen_configuration_sha256": core_contract["frozen_configuration_sha256"],
        }
    return {
        "status": "PASS",
        "fold_count": len(tuned),
        "family": next(iter(family)),
        "configuration_id": next(iter(configuration_ids)),
        "configuration_json": json.loads(next(iter(configuration_json))),
        "selected_configuration_sha256": canonical_identity_sha256(
            {
                "baseline_id": "TunedSingleConformal",
                "selected_family": TSC_FROZEN_FAMILY,
                "configuration": TSC_FROZEN_CONFIGURATION,
            }
        ),
        "registry_row_sha256_by_zone": tuned_row_sha_by_zone,
        "linucb_configuration_by_zone": linucb_by_zone,
        "cart_configuration_by_zone": cart_by_zone,
    }


def price_protocol_audit(config: dict[str, Any]) -> dict[str, Any]:
    protocol_path = REVISION_ROOT / str(config["inputs"]["price_protocol"]["relative_path"])
    protocol = load_json(protocol_path)
    if (
        protocol.get("schema") != "S08_PRICE_SENSITIVITY_PROTOCOL_V1"
        or protocol.get("version") != "1.0.0"
        or protocol.get("status")
        != "FROZEN_BEFORE_PRICE_SENSITIVITY_PERFORMANCE_READ"
    ):
        raise RuntimeError("冻结 S08 价格协议 schema/version/status 失配")
    left = [
        (
            str(row["price_id"]),
            float(row["miss_to_capacity_ratio"]),
            tuple(float(x) for x in row["theta"]),
            bool(row.get("base_submission_price", False)),
        )
        for row in config["price_grid"]
    ]
    right = [
        (
            str(row["price_id"]),
            float(row["miss_to_capacity_ratio"]),
            tuple(float(x) for x in row["theta"]),
            bool(row.get("base_submission_price", False)),
        )
        for row in protocol["price_grid"]
    ]
    if left != right or [row[0] for row in left if row[3]] != ["R05P6179775281"]:
        raise RuntimeError("新运行配置与冻结 S08 五价格合同失配")
    return {"status": "PASS", "price_count": len(left), "price_ids": [row[0] for row in left]}


def training_support_audit(
    config: dict[str, Any],
    *,
    verify_artifact_hashes: bool,
) -> dict[str, Any]:
    contract = config["training_support_identity"]
    chosen = contract.get("chosen_identity")
    allowed = (
        "A_FULL_24_HORIZON_SOURCE_SUPPORT",
        "B_UNIFIED_FIVE_HORIZON_NEW_IDENTITY",
    )
    approved = bool(contract.get("author_approved"))
    scientific = bool(contract.get("scientific_execution_allowed"))
    approval_record = dict(contract.get("author_approval_record") or {})
    record_valid = bool(
        approval_record.get("status") == "APPROVED"
        and approval_record.get("approved_option") == chosen
        and approval_record.get("approved_by") == "author"
        and str(approval_record.get("approved_at_utc") or "").strip()
        and str(approval_record.get("scope_statement") or "").strip()
    )
    status_valid = contract.get("status") == {
        "A_FULL_24_HORIZON_SOURCE_SUPPORT": "AUTHOR_APPROVED_A_FROZEN",
        "B_UNIFIED_FIVE_HORIZON_NEW_IDENTITY": APPROVED_TRAINING_STATUS,
    }.get(str(chosen))
    option_inventory: dict[str, Any] = {}
    for identity in allowed:
        horizons = training_horizons(config, identity)
        audit = inventory_audit(
            config,
            horizons=horizons,
            include_legacy_regression=False,
        )
        option_inventory[identity] = {
            "status": "PASS" if audit["status"].eq("PASS").all() else "FAIL",
            "horizons": list(horizons),
            "expected_stream_count": len(expected_stream_ids(config, horizons)),
            "rows": audit.to_dict("records"),
        }
    chosen_inventory_pass = bool(
        chosen in option_inventory and option_inventory[str(chosen)]["status"] == "PASS"
    )
    authorization_pass = bool(
        chosen in allowed and approved and scientific and record_valid and status_valid
    )
    if authorization_pass and chosen_inventory_pass and verify_artifact_hashes:
        hash_audit = artifact_hash_audit(
            config,
            horizons=training_horizons(config, str(chosen)),
        )
    elif authorization_pass and chosen_inventory_pass:
        hash_audit = {
            "status": "NOT_RUN_TEST_MODE",
            "checked_stream_count": 0,
            "mismatch_count": 0,
        }
    else:
        hash_audit = {
            "status": "NOT_RUN_GATE_BLOCKED",
            "checked_stream_count": 0,
            "mismatch_count": 0,
        }
    expected_hash_status = "PASS" if verify_artifact_hashes else "NOT_RUN_TEST_MODE"
    passed = bool(
        authorization_pass
        and chosen_inventory_pass
        and hash_audit["status"] == expected_hash_status
    )
    reasons: list[str] = []
    if chosen not in allowed:
        reasons.append("NO_VALID_CHOSEN_IDENTITY")
    if not approved or not scientific:
        reasons.append("APPROVAL_BOOLEANS_NOT_BOTH_TRUE")
    if not record_valid:
        reasons.append("MATCHING_AUTHOR_APPROVAL_RECORD_MISSING_OR_INVALID")
    if not status_valid:
        reasons.append("FROZEN_AUTHOR_APPROVAL_STATUS_MISMATCH")
    if chosen in allowed and not chosen_inventory_pass:
        reasons.append("CHOSEN_IDENTITY_INVENTORY_INCOMPLETE")
    if authorization_pass and chosen_inventory_pass and hash_audit["status"] != expected_hash_status:
        reasons.append("CHOSEN_IDENTITY_ARTIFACT_HASH_GATE_NOT_PASS")
    return {
        "status": (
            "PASS"
            if passed and verify_artifact_hashes
            else "PASS_METADATA_ONLY_ARTIFACT_HASHES_NOT_RUN"
            if passed
            else "BLOCKED_TRAINING_SUPPORT_GATE"
        ),
        "chosen_identity": chosen,
        "author_approved": approved,
        "scientific_execution_allowed": scientific,
        "approval_record_valid": record_valid,
        "approval_status_valid": status_valid,
        "authorization_pass": authorization_pass,
        "chosen_inventory_pass": chosen_inventory_pass,
        "legacy_identity": str(contract["legacy_identity"]),
        "available_migration_inventory": str(contract["available_migration_inventory"]),
        "option_inventory": option_inventory,
        "artifact_hash_audit": hash_audit,
        "blocking_reasons": reasons,
    }


def disk_capacity_audit(config: dict[str, Any]) -> dict[str, Any]:
    contract = config["execution"]["disk_capacity_contract"]
    active_stage = str(contract["active_stage"])
    stage = contract["stage_projected_bytes"][active_stage]
    persistent = int(stage["persistent_output_bytes"])
    temporary = int(stage["peak_concurrent_temporary_bytes"])
    safety_fraction = float(contract["safety_margin_fraction"])
    reserve = int(contract["fixed_reserve_bytes"])
    required = int(np.ceil(reserve + (persistent + temporary) * (1.0 + safety_fraction)))
    free = int(shutil.disk_usage(TEST_ROOT).free)
    return {
        "status": "PASS" if free >= required else "FAIL",
        "active_stage": active_stage,
        "projected_persistent_output_bytes": persistent,
        "projected_peak_concurrent_temporary_bytes": temporary,
        "safety_margin_fraction": safety_fraction,
        "fixed_reserve_bytes": reserve,
        "required_free_bytes": required,
        "observed_free_bytes": free,
        "required_free_gib": float(required / (1024**3)),
        "observed_free_gib": float(free / (1024**3)),
        "formula": str(contract["required_free_bytes_formula"]),
    }


def reusable_preflight_token(base_payload: dict[str, Any]) -> tuple[Path, dict[str, Any]] | None:
    """Reuse an unchanged exact gate token so interrupted multi-invocation runs can seal."""

    token_root = CONTROL_ROOT / "preflight_tokens"
    if not token_root.exists():
        return None
    stable_fields = (
        "schema",
        "status",
        "endpoint_artifact_identity",
        "endpoint_execution_scope_authorization_sha256",
        "training_support_identity",
        "training_support_artifact_hash_status",
        "training_support_checked_stream_count",
        "environment_audit_sha256",
        "endpoint_input_hash_audit_sha256",
        "inventory_audit_sha256",
        "legacy_reference_anchor_audit_sha256",
        "registry_audit_sha256",
        "disk_contract_required_free_bytes",
    )
    for token_path in sorted(token_root.glob("*/manifest.json"), reverse=True):
        try:
            token = load_json(token_path)
            if any(token.get(field) != base_payload.get(field) for field in stable_fields):
                continue
            report_path = Path(str(token.get("preflight_report_path"))).resolve()
            if (
                not report_path.is_file()
                or sha256_file(report_path) != token.get("preflight_report_sha256")
                or load_json(report_path).get("status") != "PASS"
            ):
                continue
            return token_path, token
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return None


def issue_preflight_gate_token(token_payload: dict[str, Any]) -> dict[str, str]:
    """Write one immutable token; its canonical identity and file digest are audited separately."""

    token_id = canonical_identity_sha256(token_payload)
    token_path = CONTROL_ROOT / "preflight_tokens" / token_id / "manifest.json"
    if token_path.exists():
        if load_json(token_path) != token_payload:
            raise RuntimeError("既有 preflight token 内容失配")
    else:
        atomic_json(token_path, token_payload)
    return {
        "path": str(token_path),
        "sha256": sha256_file(token_path),
        "token_id": token_id,
    }


def preflight(
    *,
    require_exact_environment: bool,
    write_report: bool,
    issue_token: bool = False,
) -> dict[str, Any]:
    config = load_endpoint_config()
    endpoint_inputs = endpoint_input_hash_audit(config)
    input_audit = endpoint_inputs.copy()
    inventory = inventory_audit(
        config,
        horizons=tuple(int(value) for value in config["scope"]["horizons"]),
        include_legacy_regression=True,
    )
    if inventory["status"].ne("PASS").any():
        raise RuntimeError(f"五时距评价库存失配: {inventory.to_dict('records')}")
    environment = environment_audit(config)
    registry = tuned_single_conformal_registry_audit(config)
    prices = price_protocol_audit(config)
    training_support = training_support_audit(config, verify_artifact_hashes=True)
    legacy_reference = legacy_reference_anchor_audit(
        config,
        verify_event_files=True,
    )
    disk = disk_capacity_audit(config)
    disk_pass = disk["status"] == "PASS"
    environment_pass = bool(environment["status"].eq("PASS").all())
    status = "PASS"
    if not disk_pass or (require_exact_environment and not environment_pass):
        status = "FAIL"
    elif not environment_pass:
        status = "WARN_ENVIRONMENT_DRIFT_DEVELOPMENT_ONLY"
    if training_support["status"] != "PASS" and status != "FAIL":
        status = "BLOCKED_TRAINING_SUPPORT_GATE"
    report = {
        "schema": "TEST_CLARA_GEFCOM_6A_PREFLIGHT_V1",
        "status": status,
        "generated_at_utc": utc_now(),
        **config_identity(),
        "require_exact_environment": require_exact_environment,
        "environment": environment.to_dict("records"),
        "input_hash_audit": input_audit.to_dict("records"),
        "endpoint_input_hash_audit": endpoint_inputs.to_dict("records"),
        "inventory_audit": inventory.to_dict("records"),
        "registry_audit": registry,
        "price_protocol_audit": prices,
        "training_support_audit": training_support,
        "legacy_reference_anchor_audit": legacy_reference,
        "disk_capacity_audit": disk,
    }
    report_path: Path | None = None
    if write_report:
        report_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        report_dir = CONTROL_ROOT / "preflight_runs" / report_id
        report_dir.mkdir(parents=True, exist_ok=False)
        report_path = report_dir / "manifest.json"
        atomic_json(report_path, report)
    if status == "FAIL":
        failures = environment[environment["status"].ne("PASS")].to_dict("records")
        raise RuntimeError(
            f"预检失败: environment={failures}, "
            f"free_disk_gib={disk['observed_free_gib']:.3f}, "
            f"required_disk_gib={disk['required_free_gib']:.3f}"
        )
    if status == "BLOCKED_TRAINING_SUPPORT_GATE":
        raise RuntimeError(
            "正式执行已 fail-closed：训练支持身份、库存或全库/迁移锚哈希门未通过；"
            f"reasons={training_support['blocking_reasons']}"
        )
    if issue_token:
        if not write_report or report_path is None:
            raise RuntimeError("签发 preflight token 必须同时写入不可变 preflight report")
        token_base = {
            "schema": "TEST_CLARA_GEFCOM_6A_ENDPOINT_PREFLIGHT_GATE_TOKEN_V2",
            "status": "PASS",
            "endpoint_artifact_identity": endpoint_artifact_identity(config),
            "endpoint_execution_scope_authorization_sha256": (
                endpoint_execution_scope_authorization(config)["sha256"]
            ),
            "training_support_identity": training_support["chosen_identity"],
            "training_support_artifact_hash_status": training_support["artifact_hash_audit"]["status"],
            "training_support_checked_stream_count": training_support["artifact_hash_audit"]["checked_stream_count"],
            "environment_audit_sha256": sha256_bytes(
                canonical_json_bytes(environment.to_dict("records"))
            ),
            "endpoint_input_hash_audit_sha256": sha256_bytes(
                canonical_json_bytes(endpoint_inputs.to_dict("records"))
            ),
            "inventory_audit_sha256": sha256_bytes(
                canonical_json_bytes(inventory.to_dict("records"))
            ),
            "legacy_reference_anchor_audit_sha256": sha256_bytes(
                canonical_json_bytes(legacy_reference)
            ),
            "registry_audit_sha256": sha256_bytes(canonical_json_bytes(registry)),
            "disk_contract_required_free_bytes": int(disk["required_free_bytes"]),
        }
        reusable = reusable_preflight_token(token_base)
        if reusable is not None:
            token_path, _ = reusable
            report["preflight_token"] = {
                "path": str(token_path),
                "sha256": sha256_file(token_path),
                "token_id": token_path.parent.name,
                "reused": True,
            }
            return report
        token_payload = {
            **token_base,
            "issued_at_utc": utc_now(),
            "disk_capacity_audit": disk,
            "preflight_report_path": str(report_path),
            "preflight_report_sha256": sha256_file(report_path),
        }
        issued_token = issue_preflight_gate_token(token_payload)
        report["preflight_token"] = {
            **issued_token,
            "reused": False,
        }
    return report


def endpoint_unit_id(zone: str, seed: int, horizon: int) -> str:
    return f"{zone}__seed{int(seed)}__H{int(horizon):02d}"


def selector_unit_id(zone: str, seed: int) -> str:
    return f"{zone}__seed{int(seed)}__selector_coordinator"


def replay_unit_id(zone: str, seed: int) -> str:
    return f"{zone}__seed{int(seed)}__global_replay_coordinator"


def clara_six_action_contract_identity(config: dict[str, Any]) -> dict[str, Any]:
    adaptive = config["adaptive_clara"]
    selector = config["selector_contract"]
    payload = {
        "schema": "TEST_CLARA_6A_ADAPTIVE_CONTRACT_IDENTITY_V1",
        "training_support_identity": config["training_support_identity"]["chosen_identity"],
        "source_v4_protocol_sha256": config["inputs"]["v4_adaptive_protocol"]["sha256"],
        "source_version": adaptive["version"],
        "n_min_candidates": adaptive["n_min_candidates"],
        "n_min_no_candidate_rule": adaptive["n_min_no_candidate_rule"],
        "nu_candidates": adaptive["nu_candidates"],
        "guardrail_interpolation_axis": adaptive["guardrail_interpolation_axis"],
        "action_library_sha256": selector["linucb_six_action_identity"][
            "ordered_action_library_sha256"
        ],
        "adaptive_statistics_action_scope": selector["adaptive_statistics_action_scope"],
        "n_min_top_two_margin_rule": selector["n_min_top_two_margin_rule"],
        "nu_signal_rule": selector["nu_signal_rule"],
        "guardrail_safe_set_rule": selector["guardrail_safe_set_rule"],
        "tie_break_rule": selector["tie_break_rule"],
        "price_conditioning": selector["price_conditioning"],
    }
    return {
        **payload,
        "clara_six_action_contract_sha256": canonical_identity_sha256(payload),
    }


def endpoint_tasks(config: dict[str, Any], identity: str) -> list[dict[str, Any]]:
    scope = config["scope"]
    return [
        {"zone": str(zone), "seed": int(seed), "horizon": int(horizon)}
        for zone in scope["zones"]
        for seed in scope["seeds"]
        for horizon in training_horizons(config, identity)
    ]


def build_task_ledger(config: dict[str, Any], identity: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    support_horizons = training_horizons(config, identity)
    for task in endpoint_tasks(config, identity):
        identifier = endpoint_unit_id(**task)
        rows.append(
            {
                "phase": "endpoint_cache",
                "unit_id": identifier,
                **task,
                "price_id": "",
                "dependency": "",
                "implementation": "IMPLEMENTED",
            }
        )
    for zone in config["scope"]["zones"]:
        for seed in config["scope"]["seeds"]:
            source_dependencies = ";".join(
                endpoint_unit_id(str(source_zone), int(seed), int(horizon))
                for source_zone in config["scope"]["zones"]
                if str(source_zone) != str(zone)
                for horizon in support_horizons
            )
            selector_id = selector_unit_id(str(zone), int(seed))
            rows.append(
                {
                    "phase": "selector_fit",
                    "unit_id": selector_id,
                    "zone": str(zone),
                    "seed": int(seed),
                    "horizon": "ALL_TRAINING_HORIZONS",
                    "price_id": "ALL_FIVE_PRICES_WITH_ATOMIC_CHILDREN",
                    "dependency": source_dependencies,
                    "parent_seal_rule": "incremental valid children; seal only after five CLARA/CART price-child manifests PASS",
                    "implementation": "PLANNED_NOT_IMPLEMENTED",
                }
            )
            target_dependencies = ";".join(
                endpoint_unit_id(str(zone), int(seed), int(horizon))
                for horizon in config["scope"]["horizons"]
            )
            rows.append(
                {
                    "phase": "target_replay",
                    "unit_id": replay_unit_id(str(zone), int(seed)),
                    "zone": str(zone),
                    "seed": int(seed),
                    "horizon": "ALL_FIVE_EVALUATION_HORIZONS_GLOBAL_ISSUE_TIME",
                    "price_id": "ALL_FIVE_PRICES_WITH_ATOMIC_CHILDREN",
                    "dependency": f"SAME_PRICE_SELECTOR_CHILD_STREAMING;{target_dependencies}",
                    "parent_seal_rule": "incremental valid children; seal only after five same-price replay child manifests PASS",
                    "implementation": "PLANNED_NOT_IMPLEMENTED",
                }
            )
    rows.extend(
        [
            {
                "phase": "aggregate",
                "unit_id": "aggregate_all",
                "zone": "",
                "seed": np.nan,
                "horizon": np.nan,
                "price_id": "",
                "dependency": "all_target_replay_units",
                "implementation": "PLANNED_NOT_IMPLEMENTED",
            },
            {
                "phase": "independent_qa",
                "unit_id": "independent_qa_all",
                "zone": "",
                "seed": np.nan,
                "horizon": np.nan,
                "price_id": "",
                "dependency": "aggregate_all",
                "implementation": "PLANNED_NOT_IMPLEMENTED",
            },
        ]
    )
    ledger = pd.DataFrame(rows)
    ledger.insert(0, "training_support_identity", identity)
    return ledger


def build_price_child_ledger(config: dict[str, Any], identity: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    linucb_identity = config["selector_contract"]["linucb_six_action_identity"]
    registry_all = registry_audit(config)
    linucb_registry = registry_all["linucb_configuration_by_zone"]
    cart_registry = registry_all["cart_configuration_by_zone"]
    clara_identity = clara_six_action_contract_identity(config)
    v4_contract = selector_core_clara_v4_adaptive_contract()
    price_specs = {
        price.price_id: price for price in selector_core_price_specs_from_config(config)
    }
    action_library_sha = str(linucb_identity["ordered_action_library_sha256"])
    global_scope_sha = str(linucb_identity["global_replay_scope_sha256"])
    endpoint_root_dependency = f"endpoint_units/root_manifest__{identity}.json#SHA256_REQUIRED_AT_EXECUTION"
    for zone in config["scope"]["zones"]:
        for seed in config["scope"]["seeds"]:
            selector_parent = selector_unit_id(str(zone), int(seed))
            replay_parent = replay_unit_id(str(zone), int(seed))
            for price in config["price_grid"]:
                price_id = str(price["price_id"])
                theta_sha = canonical_identity_sha256(
                    {
                        "schema": "TEST_CLARA_THETA_VECTOR_IDENTITY_V1",
                        "price_id": price_id,
                        "theta": [float(value) for value in price["theta"]],
                    }
                )
                price_sha = selector_core_price_sha256(price_specs[price_id])
                selector_child = f"{selector_parent}__{price_id}"
                replay_child = f"{replay_parent}__{price_id}"
                rows.extend(
                    [
                        {
                            "training_support_identity": identity,
                            "phase": "selector_fit_price_child",
                            "parent_unit_id": selector_parent,
                            "child_unit_id": selector_child,
                            "zone": str(zone),
                            "seed": int(seed),
                            "price_id": price_id,
                            "dependency": endpoint_root_dependency,
                            "action_library_sha256": action_library_sha,
                            "theta_sha256": theta_sha,
                            "price_sha256": price_sha,
                            "global_replay_scope_sha256": "",
                            "heldout_zone_alpha_lambda_sha256": "",
                            "heldout_zone_cart_configuration_sha256": cart_registry[str(zone)][
                                "heldout_zone_cart_configuration_sha256"
                            ],
                            "heldout_zone_cart_legacy_configuration_id": cart_registry[str(zone)][
                                "legacy_configuration_id"
                            ],
                            "cart_frozen_zone_configuration_sha256": cart_registry[str(zone)][
                                "frozen_configuration_sha256"
                            ],
                            "clara_v4_source_protocol_sha256": config["inputs"][
                                "v4_adaptive_protocol"
                            ]["sha256"],
                            "clara_six_action_contract_sha256": clara_identity[
                                "clara_six_action_contract_sha256"
                            ],
                            "clara_v4_adaptive_subcontract_sha256": v4_contract[
                                "v4_adaptive_subcontract_sha256"
                            ],
                            "s07_registry_sha256": config["inputs"]["s07_execution_registry"]["sha256"],
                            "contract": "independent six-action CLARA/CART refit only; no LinUCB policy artifact",
                        },
                        {
                            "training_support_identity": identity,
                            "phase": "target_replay_price_child",
                            "parent_unit_id": replay_parent,
                            "child_unit_id": replay_child,
                            "zone": str(zone),
                            "seed": int(seed),
                            "price_id": price_id,
                            "dependency": f"{selector_child}#MANIFEST_SHA256_REQUIRED;{endpoint_root_dependency}",
                            "action_library_sha256": action_library_sha,
                            "theta_sha256": theta_sha,
                            "price_sha256": price_sha,
                            "global_replay_scope_sha256": global_scope_sha,
                            "heldout_zone_alpha_lambda_sha256": linucb_registry[str(zone)][
                                "heldout_zone_alpha_lambda_sha256"
                            ],
                            "heldout_zone_linucb_legacy_configuration_id": linucb_registry[str(zone)][
                                "legacy_configuration_id"
                            ],
                            "linucb_frozen_zone_configuration_sha256": linucb_registry[str(zone)][
                                "frozen_configuration_sha256"
                            ],
                            "heldout_zone_cart_configuration_sha256": "",
                            "heldout_zone_cart_legacy_configuration_id": "",
                            "cart_frozen_zone_configuration_sha256": "",
                            "clara_v4_source_protocol_sha256": config["inputs"][
                                "v4_adaptive_protocol"
                            ]["sha256"],
                            "clara_six_action_contract_sha256": clara_identity[
                                "clara_six_action_contract_sha256"
                            ],
                            "clara_v4_adaptive_subcontract_sha256": v4_contract[
                                "v4_adaptive_subcontract_sha256"
                            ],
                            "s07_registry_sha256": config["inputs"]["s07_execution_registry"]["sha256"],
                            "contract": "same-price selector child plus endpoint-root seal; initialize one six-action LinUCB and replay globally across all 5 horizons×4 predictors×11 coverages; then horizon slices",
                        },
                    ]
                )
    return pd.DataFrame(rows)


def write_endpoint_plan() -> dict[str, Any]:
    """Write a B-only endpoint plan whose identity excludes runner and selector-core SHAs."""

    config = load_endpoint_config()
    endpoint_inputs = endpoint_input_hash_audit(config)
    tuned_single_conformal_registry_audit(config)
    price_protocol_audit(config)
    training = training_support_audit(config, verify_artifact_hashes=False)
    if not str(training["status"]).startswith("PASS"):
        raise RuntimeError("endpoint-only 计划被训练支持门阻断")
    identity = str(config["training_support_identity"]["chosen_identity"])
    tasks = endpoint_tasks(config, identity)
    rows = [
        {
            "unit_id": endpoint_unit_id(**task),
            "training_support_identity": identity,
            **task,
        }
        for task in tasks
    ]
    ledger = pd.DataFrame(rows)
    identity_payload = {
        "schema": "TEST_CLARA_GEFCOM_6A_ENDPOINT_PLAN_IDENTITY_V1",
        "training_support_identity": identity,
        "endpoint_artifact_identity": endpoint_artifact_identity(config),
        "endpoint_execution_scope_authorization_sha256": (
            endpoint_execution_scope_authorization(config)["sha256"]
        ),
        "endpoint_input_hash_audit_sha256": sha256_bytes(
            canonical_json_bytes(endpoint_inputs.to_dict("records"))
        ),
        "environment_contract_sha256": sha256_bytes(
            canonical_json_bytes(config["environment"])
        ),
        "action_library_sha256": ACTION_LIBRARY_SHA256,
        "unit_count": len(rows),
    }
    revision_key = canonical_identity_sha256(identity_payload)
    revision_root = CONTROL_ROOT / "endpoint_plan_revisions" / revision_key
    ledger_path = revision_root / "endpoint_task_ledger.csv"
    manifest_path = revision_root / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if (
            manifest.get("identity") != identity_payload
            or not ledger_path.is_file()
            or sha256_file(ledger_path) != manifest.get("endpoint_task_ledger_sha256")
        ):
            raise RuntimeError("既有 endpoint-only 计划身份或账本哈希失配")
        return {**manifest, "resumed": True}
    if revision_root.exists():
        raise RuntimeError("endpoint-only 计划目录存在但未封存，拒绝覆盖")
    atomic_csv(ledger_path, ledger)
    manifest = {
        "schema": "TEST_CLARA_GEFCOM_6A_ENDPOINT_PLAN_V1",
        "status": "PASS",
        "generated_at_utc": utc_now(),
        "identity": identity_payload,
        "endpoint_task_ledger_path": str(ledger_path),
        "endpoint_task_ledger_sha256": sha256_file(ledger_path),
    }
    atomic_json(manifest_path, manifest)
    return manifest


def write_plan() -> dict[str, Any]:
    config = load_config()
    input_hash_audit(config)
    registry_audit(config)
    price_protocol_audit(config)
    training_support = training_support_audit(config, verify_artifact_hashes=False)
    identities = {**config_identity(), **selector_core_artifact_identity()}
    revision_key = (
        f"{identities['config_sha256'][:12]}_{identities['runner_sha256'][:12]}_"
        f"{identities['selector_core_module_sha256'][:12]}_"
        f"{identities['selector_algorithm_closure_sha256'][:12]}"
    )
    revision_root = CONTROL_ROOT / "plan_revisions" / revision_key
    master_path = revision_root / "master_manifest.json"
    options = (
        "A_FULL_24_HORIZON_SOURCE_SUPPORT",
        "B_UNIFIED_FIVE_HORIZON_NEW_IDENTITY",
    )
    if master_path.exists():
        master = load_json(master_path)
        if any(master.get(key) != value for key, value in identities.items()):
            raise RuntimeError("计划修订目录身份失配，拒绝复用")
        for identity in options:
            ledger_path = revision_root / identity / "task_ledger.csv"
            expected_hash = master["option_plans"][identity]["task_ledger_sha256"]
            if not ledger_path.is_file() or sha256_file(ledger_path) != expected_hash:
                raise RuntimeError("计划修订账本哈希失配，拒绝复用")
            child_path = revision_root / identity / "price_child_ledger.csv"
            expected_child_hash = master["option_plans"][identity]["price_child_ledger_sha256"]
            if not child_path.is_file() or sha256_file(child_path) != expected_child_hash:
                raise RuntimeError("计划修订逐价子账本哈希失配，拒绝复用")
        return {**master, "resumed": True}
    if revision_root.exists():
        raise RuntimeError("计划修订目录存在但未封存，拒绝覆盖")
    option_plans: dict[str, Any] = {}
    for identity in options:
        ledger = build_task_ledger(config, identity)
        child_ledger = build_price_child_ledger(config, identity)
        expected = {
            "endpoint_cache": int(
                config["scope"]["expected_endpoint_unit_count_by_training_identity"][identity]
            ),
            "selector_fit": int(config["scope"]["expected_selector_unit_count"]),
            "target_replay": int(config["scope"]["expected_replay_unit_count"]),
            "aggregate": 1,
            "independent_qa": 1,
        }
        observed = ledger.groupby("phase").size().to_dict()
        if observed != expected:
            raise RuntimeError(
                f"{identity} 任务账本单元守恒失败: observed={observed}, expected={expected}"
            )
        if (
            len(child_ledger) != int(config["scope"]["expected_price_child_count"])
            or child_ledger.groupby("phase").size().to_dict()
            != {
                "selector_fit_price_child": 150,
                "target_replay_price_child": 150,
            }
        ):
            raise RuntimeError(f"{identity} 逐价子账本数量不守恒")
        option_root = revision_root / identity
        ledger_path = option_root / "task_ledger.csv"
        child_path = option_root / "price_child_ledger.csv"
        atomic_csv(ledger_path, ledger)
        atomic_csv(child_path, child_ledger)
        phase_counts = (
            ledger.groupby(["phase", "implementation"], sort=True)
            .size()
            .rename("unit_count")
            .reset_index()
        )
        option_manifest = {
            "schema": "TEST_CLARA_GEFCOM_6A_OPTION_TASK_PLAN_V2",
            "status": (
                "ENDPOINT_PHASE_EXECUTABLE_AFTER_PASSING_PREFLIGHT"
                if identity == config["training_support_identity"].get("chosen_identity")
                and training_support["option_inventory"][identity]["status"] == "PASS"
                else "BLOCKED_OPTION_NOT_APPROVED_OR_INVENTORY_INCOMPLETE"
            ),
            "training_support_identity": identity,
            **identities,
            "task_ledger_sha256": sha256_file(ledger_path),
            "price_child_ledger_sha256": sha256_file(child_path),
            "phase_counts": phase_counts.to_dict("records"),
            "total_task_count": len(ledger),
            "price_child_count": len(child_ledger),
            "implemented_task_count": int(ledger["implementation"].eq("IMPLEMENTED").sum()),
            "disabled_or_unimplemented_task_count": int(
                ledger["implementation"].ne("IMPLEMENTED").sum()
            ),
        }
        atomic_json(option_root / "manifest.json", option_manifest)
        option_plans[identity] = option_manifest
    master = {
        "schema": "TEST_CLARA_GEFCOM_6A_DUAL_OPTION_TASK_PLAN_V2",
        "status": "B_ENDPOINT_PHASE_READY_PENDING_PASSING_PREFLIGHT",
        "generated_at_utc": utc_now(),
        **identities,
        "chosen_identity": config["training_support_identity"].get("chosen_identity"),
        "training_support_audit": training_support,
        "option_plans": option_plans,
        "supersedes_legacy_fixed_five_horizon_plan": str(CONTROL_ROOT / "plan_manifest.json"),
    }
    atomic_json(master_path, master)
    return master


def _baseline_module() -> Any:
    config = load_endpoint_config()
    for path in (AUTH_CODE, BASELINE_SCRIPTS):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    import run_baseline_supplement as baseline  # type: ignore
    module_inputs = {
        "run_baseline_supplement": "baseline_supplement_runner",
        "baseline_supplement_common": "baseline_supplement_common",
        "baseline_common": "baseline_common_module",
        "baseline_conformal": "baseline_conformal_module",
        "baseline_ensemble": "baseline_ensemble_module",
        "baseline_sequential": "baseline_sequential_module",
        "baseline_training": "baseline_training_module",
        "clara_errf": "clara_errf_module",
        "source_tuning_facts": "source_tuning_facts_module",
        "clara_event_contract": "event_contract_module",
    }
    for module_name, input_id in module_inputs.items():
        module = importlib.import_module(module_name)
        observed_path = Path(str(module.__file__)).resolve()
        expected_path = (
            REVISION_ROOT / str(config["inputs"][input_id]["relative_path"])
        ).resolve()
        if (
            observed_path != expected_path
            or sha256_file(observed_path) != str(config["inputs"][input_id]["sha256"])
        ):
            raise RuntimeError(
                f"端点 baseline 导入模块路径或SHA偏离冻结身份: {module_name}"
            )
    return baseline


def _reliability_arrays(
    events: pd.DataFrame,
    covered: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    return _endpoint_reliability_arrays(
        events,
        covered,
        cadence_minutes=60.0,
    )


def _legacy_anchor_metadata(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    gate = config["regression_gate"]
    verified_path = REVISION_ROOT / str(
        config["inputs"][gate["verified_manifest_input_id"]]["relative_path"]
    )
    audit_path = REVISION_ROOT / str(
        config["inputs"][gate["verified_input_audit_input_id"]]["relative_path"]
    )
    qa_path = REVISION_ROOT / str(
        config["inputs"][gate["independent_qa_input_id"]]["relative_path"]
    )
    verified = load_json(verified_path)
    qa = load_json(qa_path)
    audit = pd.read_csv(audit_path, dtype={"unit_id": str, "event_results_sha256": str, "status": str})
    expected_count = int(gate["expected_reference_unit_count"])
    expected_events = int(gate["expected_reference_unique_event_count"])
    required_columns = {
        "unit_id",
        "event_count",
        "cell_count",
        "maximum_errf_recomputation_error",
        "event_results_sha256",
        "status",
    }
    verified_sha = sha256_file(verified_path)
    audit_sha = sha256_file(audit_path)
    if (
        verified.get("schema") != "PAPER_EQUIVALENT_REAGGREGATION_V1"
        or verified.get("status") != "READY_FOR_INDEPENDENT_QA"
        or int(verified.get("unit_count", -1)) != expected_count
        or int(verified.get("unique_event_count", -1)) != expected_events
        or verified.get("outputs", {}).get("input_audit.csv") != audit_sha
        or qa.get("schema") != "QA_PAPER_EQUIVALENT_REAGGREGATION_V1"
        or qa.get("status") != "PASS"
        or int(qa.get("failed", -1)) != 0
        or qa.get("manifest_sha256") != verified_sha
        or set(audit.columns) != required_columns
        or len(audit) != expected_count
        or audit["unit_id"].duplicated().any()
        or audit["status"].astype(str).ne("PASS").any()
        or int(audit["event_count"].astype(int).sum()) != expected_events
        or audit["event_results_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").ne(True).any()
    ):
        raise RuntimeError("三覆盖率回归参考的验证清单、input_audit 或独立 QA 锚失配")
    return verified, qa, audit


def _validate_legacy_reference_row(
    config: dict[str, Any],
    row: Any,
    *,
    verify_event_file: bool,
) -> dict[str, Any]:
    unit_id = str(row.unit_id)
    match = re.fullmatch(r"(zone(?:10|[1-9]))__seed([0-2])__H(01|03|06|12|24)", unit_id)
    if match is None:
        raise RuntimeError(f"回归参考 unit_id 非法: {unit_id}")
    zone, seed_text, horizon_text = match.groups()
    seed, horizon = int(seed_text), int(horizon_text)
    unit_root = (
        REVISION_ROOT
        / str(config["regression_gate"]["reference_relative_path"])
        / unit_id
    )
    manifest_path = unit_root / "unit_manifest.json"
    event_path = unit_root / "event_results.parquet"
    if not manifest_path.is_file() or not event_path.is_file():
        raise FileNotFoundError(f"回归参考单元不完整: {unit_root}")
    manifest = load_json(manifest_path)
    expected_hash = str(row.event_results_sha256)
    expected_events = int(row.event_count)
    if (
        manifest.get("schema") != "BASELINE_SUPPLEMENT_UNIT_V1"
        or manifest.get("status") != "PASS"
        or manifest.get("unit_id") != unit_id
        or manifest.get("zone") != zone
        or int(manifest.get("seed", -1)) != seed
        or int(manifest.get("horizon", -1)) != horizon
        or int(manifest.get("event_count", -1)) != expected_events
        or manifest.get("event_results_sha256") != expected_hash
    ):
        raise RuntimeError(f"回归参考单元 manifest 身份失配: {unit_id}")
    if verify_event_file and sha256_file(event_path) != expected_hash:
        raise RuntimeError(f"回归参考 event_results SHA256 失配: {unit_id}")
    return {
        "unit_id": unit_id,
        "zone": zone,
        "seed": seed,
        "horizon": horizon,
        "event_count": expected_events,
        "event_results_path": str(event_path),
        "event_results_sha256": expected_hash,
        "unit_manifest_sha256": sha256_file(manifest_path),
    }


def legacy_reference_anchor_audit(
    config: dict[str, Any],
    *,
    verify_event_files: bool,
) -> dict[str, Any]:
    verified, qa, audit = _legacy_anchor_metadata(config)
    records = [
        _validate_legacy_reference_row(
            config,
            row,
            verify_event_file=verify_event_files,
        )
        for row in audit.itertuples(index=False)
    ]
    return {
        "status": "PASS",
        "verified_manifest_sha256": str(qa["manifest_sha256"]),
        "input_audit_sha256": str(verified["outputs"]["input_audit.csv"]),
        "independent_qa_status": str(qa["status"]),
        "unit_count": len(records),
        "unique_event_count": int(sum(row["event_count"] for row in records)),
        "verified_event_file_count": len(records) if verify_event_files else 0,
        "reference_set_sha256": sha256_bytes(canonical_json_bytes(records)),
    }


def legacy_reference_record(
    config: dict[str, Any],
    *,
    zone: str,
    seed: int,
    horizon: int,
) -> dict[str, Any]:
    _, _, audit = _legacy_anchor_metadata(config)
    unit_id = endpoint_unit_id(zone, seed, horizon)
    selected = audit[audit["unit_id"].astype(str).eq(unit_id)]
    if len(selected) != 1:
        raise RuntimeError(f"已验证 input_audit 中回归参考不唯一: {unit_id}")
    return _validate_legacy_reference_row(
        config,
        next(selected.itertuples(index=False)),
        verify_event_file=True,
    )


def _legacy_regression(
    *,
    config: dict[str, Any],
    events: pd.DataFrame,
    addons: pd.DataFrame,
    zone: str,
    seed: int,
    horizon: int,
) -> dict[str, Any]:
    anchor = legacy_reference_record(
        config,
        zone=zone,
        seed=seed,
        horizon=horizon,
    )
    reference_path = Path(anchor["event_results_path"])
    reference = pd.read_parquet(reference_path)
    coverages = set(float(value) for value in config["regression_gate"]["coverages"])
    reference = reference[reference["target_coverage"].astype(float).isin(coverages)].copy()
    addon_value_columns = [
        "event_id",
        *[
            column
            for action in ADDON_ACTIONS
            for column in (
                f"{action}__candidate_lower",
                f"{action}__candidate_upper",
                f"{action}__covered",
                f"{action}__tuwr_indicator",
                f"{action}__ard_value",
            )
        ],
    ]
    base = events[["event_id", "target_coverage", "target_after_maturity", "schedule_proxy"]].merge(
        addons[addon_value_columns],
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    base = base[base["target_coverage"].astype(float).isin(coverages)].copy()
    expected_reference_events = int(anchor["event_count"])
    reference_counts = reference.groupby("target_coverage", sort=False).size().to_dict()
    base_counts = base.groupby("target_coverage", sort=False).size().to_dict()
    if (
        len(reference) != expected_reference_events
        or len(base) != expected_reference_events
        or reference.empty
        or base.empty
        or reference["event_id"].astype(str).duplicated().any()
        or base["event_id"].astype(str).duplicated().any()
        or set(float(value) for value in reference_counts) != coverages
        or set(float(value) for value in base_counts) != coverages
        or any(int(value) <= 0 for value in reference_counts.values())
        or any(int(value) <= 0 for value in base_counts.values())
        or set(base["event_id"].astype(str)) != set(reference["event_id"].astype(str))
    ):
        raise RuntimeError("新增动作端点与三覆盖率回归参考事件集合失配")
    tolerance = float(config["regression_gate"]["endpoint_and_base_price_metric_absolute_tolerance"])
    rows: list[dict[str, Any]] = []
    mapping = {
        "TunedSingleConformal": "tuned",
        "EqualEndpointEnsemble": "ensemble",
    }
    for action, prefix in mapping.items():
        left = base[["event_id", "target_after_maturity", "schedule_proxy", f"{action}__candidate_lower", f"{action}__candidate_upper", f"{action}__covered"]].copy()
        left["computed_errf"] = endpoint_errf(
            target=left["target_after_maturity"].to_numpy(dtype=float),
            schedule=left["schedule_proxy"].to_numpy(dtype=float),
            lower=left[f"{action}__candidate_lower"].to_numpy(dtype=float),
            upper=left[f"{action}__candidate_upper"].to_numpy(dtype=float),
            theta=BASE_PRICE_THETA,
        )
        joined = left.merge(
            reference[["event_id", f"{prefix}_lower", f"{prefix}_upper", f"{prefix}_covered", f"{prefix}_errf"]],
            on="event_id",
            how="inner",
            validate="one_to_one",
        )
        if len(joined) != expected_reference_events or joined.empty:
            raise RuntimeError(f"三覆盖率回归连接事件数失配: {action}")
        maximum = 0.0
        mismatch = 0
        for candidate_column, reference_column in (
            (f"{action}__candidate_lower", f"{prefix}_lower"),
            (f"{action}__candidate_upper", f"{prefix}_upper"),
            ("computed_errf", f"{prefix}_errf"),
        ):
            delta = np.abs(joined[candidate_column].to_numpy(dtype=float) - joined[reference_column].to_numpy(dtype=float))
            maximum = max(maximum, float(delta.max(initial=0.0)))
            mismatch += int((delta > tolerance).sum())
        mismatch += int(
            joined[f"{action}__covered"].astype(bool).ne(joined[f"{prefix}_covered"].astype(bool)).sum()
        )
        rows.append(
            {
                "method": action,
                "event_count": len(joined),
                "mismatch_count": mismatch,
                "maximum_absolute_difference": maximum,
                "tolerance": tolerance,
                "status": "PASS" if mismatch == 0 else "FAIL",
            }
        )
    if any(row["status"] != "PASS" for row in rows):
        raise RuntimeError(f"三覆盖率新增动作端点回归失败: {rows}")
    return {
        "status": "PASS",
        "reference_path": str(reference_path),
        "reference_sha256": str(anchor["event_results_sha256"]),
        "reference_unit_manifest_sha256": str(anchor["unit_manifest_sha256"]),
        "checks": rows,
    }


def expected_endpoint_unit_event_count(
    config: dict[str, Any],
    *,
    zone: str,
    seed: int,
    horizon: int,
) -> int:
    """Derive the exact post-maturity-filter event total from frozen inputs."""

    total = 0
    for predictor in config["scope"]["predictors"]:
        manifest_path = (
            S06_FACT_ROOT
            / "source_facts"
            / f"predictor={predictor}"
            / f"zone={zone}"
            / f"horizon={int(horizon):02d}"
            / f"seed={int(seed)}"
            / "manifest.json"
        )
        if not manifest_path.is_file():
            raise FileNotFoundError(f"S06 端点事件守恒 manifest 不存在: {manifest_path}")
        manifest = load_json(manifest_path)
        stream_id = f"{predictor}-H{int(horizon):02d}-{zone}-S{int(seed)}"
        count = int(manifest.get("complete_case_event_count", -1))
        if (
            manifest.get("manifest_schema") != "S06_SOURCE_TUNING_FACT_PARTITION_V1"
            or manifest.get("status") != "COMPLETE_VALIDATED"
            or manifest.get("stream_id") != stream_id
            or manifest.get("predictor") != predictor
            or manifest.get("zone") != zone
            or int(manifest.get("seed", -1)) != int(seed)
            or int(manifest.get("horizon", -1)) != int(horizon)
            or count <= 0
        ):
            raise RuntimeError(f"S06 端点事件守恒 manifest 身份失配: {stream_id}")
        total += count
    if total <= 0:
        raise RuntimeError("端点单元预期事件数非正")
    regression_horizons = set(
        int(value) for value in config["endpoint_contract"]["regression_horizons"]
    )
    if int(horizon) in regression_horizons:
        _, _, audit = _legacy_anchor_metadata(config)
        unit_id = endpoint_unit_id(zone, seed, horizon)
        selected = audit[audit["unit_id"].astype(str).eq(unit_id)]
        if len(selected) != 1:
            raise RuntimeError(f"端点事件守恒的已验证三覆盖率参考不唯一: {unit_id}")
        legacy_events = int(selected.iloc[0]["event_count"])
        reference_coverage_count = len(config["regression_gate"]["coverages"])
        target_coverage_count = len(config["scope"]["target_coverages"])
        numerator = legacy_events * target_coverage_count
        if numerator % reference_coverage_count:
            raise RuntimeError(f"三覆盖率到十一覆盖率的事件数不可整除: {unit_id}")
        exact = numerator // reference_coverage_count
        if exact > total:
            raise RuntimeError(f"成熟反馈过滤后事件数超过 S06 complete-case 上界: {unit_id}")
        return exact

    # A 路线的19个非参考时距没有旧回归结果；在恢复库存后从
    # S06 facts 与 S03 strict-feedback 直接计数，不伪造 regression PASS。
    exact = 0
    coverages = set(float(value) for value in config["scope"]["target_coverages"])
    for predictor in config["scope"]["predictors"]:
        stream_id = f"{predictor}-H{int(horizon):02d}-{zone}-S{int(seed)}"
        fact_path = (
            S06_FACT_ROOT
            / "source_facts"
            / f"predictor={predictor}"
            / f"zone={zone}"
            / f"horizon={int(horizon):02d}"
            / f"seed={int(seed)}"
            / "facts.parquet"
        )
        feedback_path = S03_ROOT / "bundles" / str(predictor) / stream_id / "feedback_trace.parquet"
        facts = pd.read_parquet(fact_path, columns=["event_id", "target_coverage"])
        facts = facts[facts["target_coverage"].astype(float).isin(coverages)]
        feedback = pd.read_parquet(
            feedback_path,
            columns=["event_id", "action", "eligible_by_strict_time_rule"],
        )
        fact_ids = set(facts["event_id"].astype(str))
        eligible = feedback[
            feedback["eligible_by_strict_time_rule"].astype(bool)
            & feedback["event_id"].astype(str).isin(fact_ids)
        ]
        mature_counts = eligible.groupby("event_id", sort=False)["action"].nunique()
        mature_ids = set(mature_counts[mature_counts.eq(len(ORIGINAL_ACTIONS))].index.astype(str))
        exact += int(facts["event_id"].astype(str).isin(mature_ids).sum())
    if exact <= 0 or exact > total:
        raise RuntimeError("非参考时距端点事件数守恒失败")
    return exact


def endpoint_unit_source_anchor_audit(
    config: dict[str, Any],
    *,
    zone: str,
    seed: int,
    horizon: int,
) -> dict[str, Any]:
    anchor = migration_sha256_anchor_index(config)
    records: list[dict[str, Any]] = []
    for predictor in config["scope"]["predictors"]:
        stream_id = f"{predictor}-H{int(horizon):02d}-{zone}-S{int(seed)}"
        source_dir = (
            S06_FACT_ROOT
            / "source_facts"
            / f"predictor={predictor}"
            / f"zone={zone}"
            / f"horizon={int(horizon):02d}"
            / f"seed={int(seed)}"
        )
        bundle_dir = S03_ROOT / "bundles" / str(predictor) / stream_id
        base_dir = S03_ROOT / "base" / str(predictor) / stream_id
        paths = (
            source_dir / "manifest.json",
            source_dir / "facts.parquet",
            bundle_dir / "manifest.json",
            bundle_dir / "event_core.parquet",
            bundle_dir / "candidate_intervals.parquet",
            bundle_dir / "feedback_trace.parquet",
            base_dir / "manifest.json",
            base_dir / "base_predictions.parquet",
        )
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"端点单元冻结源工件不存在: {path}")
            key = migration_relative_key(path)
            observed = sha256_file(path)
            expected = anchor.get(key)
            if expected is None or observed != expected:
                raise RuntimeError(f"端点单元冻结源与迁移根锚失配: {path}")
            records.append(
                {
                    "stream_id": stream_id,
                    "relative_path": key,
                    "sha256": observed,
                }
            )
    return {
        "schema": "TEST_CLARA_GEFCOM_6A_ENDPOINT_SOURCE_ANCHOR_AUDIT_V1",
        "status": "PASS",
        "unit_id": endpoint_unit_id(zone, seed, horizon),
        "migration_manifest_sha256": config["inputs"]["migration_full_sha256_manifest"]["sha256"],
        "checked_file_count": len(records),
        "records_sha256": sha256_bytes(canonical_json_bytes(records)),
        "records": records,
    }


def _complete_endpoint_manifest(
    path: Path,
    identities: dict[str, Any],
    *,
    config: dict[str, Any],
    zone: str,
    seed: int,
    horizon: int,
) -> dict[str, Any] | None:
    manifest_path = path / "manifest.json"
    required = (
        "endpoint_addons.parquet",
        "event_input_audit.csv",
        "bundle_input_audit.csv",
        "conformal_audit.json",
        "regression_audit.json",
        "source_anchor_audit.json",
    )
    if not manifest_path.is_file() or any(not (path / name).is_file() for name in required):
        return None
    manifest = load_json(manifest_path)
    expected_unit_id = endpoint_unit_id(zone, seed, horizon)
    regression_expected = int(horizon) in set(
        int(value) for value in config["endpoint_contract"]["regression_horizons"]
    )
    expected_regression_status = "PASS" if regression_expected else "NOT_APPLICABLE_NON_REFERENCE_HORIZON"
    expected_events = expected_endpoint_unit_event_count(
        config,
        zone=zone,
        seed=seed,
        horizon=horizon,
    )
    if (
        manifest.get("schema") != "TEST_CLARA_GEFCOM_6A_ENDPOINT_UNIT_V1"
        or manifest.get("status") != "PASS"
        or manifest.get("identities") != identities
        or manifest.get("unit_id") != expected_unit_id
        or manifest.get("zone") != str(zone)
        or int(manifest.get("seed", -1)) != int(seed)
        or int(manifest.get("horizon", -1)) != int(horizon)
        or int(manifest.get("event_count", -1)) != expected_events
        or int(manifest.get("coverage_count", -1)) != len(config["scope"]["target_coverages"])
        or int(manifest.get("predictor_count", -1)) != len(config["scope"]["predictors"])
        or int(manifest.get("addon_action_count", -1)) != len(ADDON_ACTIONS)
        or manifest.get("legacy_regression_status") != expected_regression_status
        or re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("preflight_token_id") or ""))
        is None
        or not str(manifest.get("preflight_token_sha256") or "")
        or not str(manifest.get("preflight_report_sha256") or "")
    ):
        return None
    for name in required:
        key = name.replace(".", "_") + "_sha256"
        if manifest.get(key) != sha256_file(path / name):
            return None
    endpoint_keys = pd.read_parquet(
        path / "endpoint_addons.parquet",
        columns=["event_id", "zone_or_farm", "predictor", "seed", "horizon_steps", "target_coverage"],
    )
    if (
        len(endpoint_keys) != int(manifest["event_count"])
        or endpoint_keys["event_id"].duplicated().any()
        or endpoint_keys["zone_or_farm"].astype(str).ne(str(zone)).any()
        or endpoint_keys["seed"].astype(int).ne(int(seed)).any()
        or endpoint_keys["horizon_steps"].astype(int).ne(int(horizon)).any()
        or endpoint_keys["predictor"].nunique() != len(config["scope"]["predictors"])
        or endpoint_keys["target_coverage"].nunique() != len(config["scope"]["target_coverages"])
    ):
        return None
    regression = load_json(path / "regression_audit.json")
    if regression.get("status") != expected_regression_status:
        return None
    if regression_expected and (
        manifest.get("legacy_regression_reference_path") != regression.get("reference_path")
        or manifest.get("legacy_regression_reference_sha256") != regression.get("reference_sha256")
        or manifest.get("legacy_regression_reference_unit_manifest_sha256")
        != regression.get("reference_unit_manifest_sha256")
        or not str(regression.get("reference_sha256") or "")
        or not str(regression.get("reference_unit_manifest_sha256") or "")
    ):
        return None
    source_anchor = load_json(path / "source_anchor_audit.json")
    if (
        source_anchor.get("schema")
        != "TEST_CLARA_GEFCOM_6A_ENDPOINT_SOURCE_ANCHOR_AUDIT_V1"
        or source_anchor.get("status") != "PASS"
        or source_anchor.get("unit_id") != expected_unit_id
        or source_anchor.get("migration_manifest_sha256")
        != config["inputs"]["migration_full_sha256_manifest"]["sha256"]
        or int(source_anchor.get("checked_file_count", -1))
        != len(config["scope"]["predictors"]) * 8
        or source_anchor.get("records_sha256")
        != sha256_bytes(canonical_json_bytes(source_anchor.get("records")))
    ):
        return None
    try:
        current_source_anchor = endpoint_unit_source_anchor_audit(
            config,
            zone=str(zone),
            seed=int(seed),
            horizon=int(horizon),
        )
    except (OSError, RuntimeError, KeyError, ValueError):
        return None
    if current_source_anchor != source_anchor:
        return None
    return manifest


def validate_preflight_gate_token(
    config: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    scope_authorization = validate_endpoint_task_scope_authorization(config, task)
    token_value = task.get("preflight_token_path")
    expected_token_hash = str(task.get("preflight_token_sha256") or "")
    if not token_value or not expected_token_hash:
        raise RuntimeError("端点 worker 缺少封存 preflight gate token")
    token_path = Path(str(token_value)).resolve()
    token_root = (CONTROL_ROOT / "preflight_tokens").resolve()
    if not token_path.is_relative_to(token_root) or token_path.name != "manifest.json":
        raise RuntimeError("preflight gate token 路径越出冻结 token 根")
    if not token_path.is_file() or sha256_file(token_path) != expected_token_hash:
        raise RuntimeError("preflight gate token 文件或哈希失配")
    token = load_json(token_path)
    expected_token_id = canonical_identity_sha256(token)
    if token_path.parent.name != expected_token_id:
        raise RuntimeError("preflight gate token canonical identity 与目录名失配")
    if (
        token.get("schema") != "TEST_CLARA_GEFCOM_6A_ENDPOINT_PREFLIGHT_GATE_TOKEN_V2"
        or token.get("status") != "PASS"
        or token.get("endpoint_artifact_identity") != endpoint_artifact_identity(config)
        or token.get("endpoint_execution_scope_authorization_sha256")
        != scope_authorization["sha256"]
        or token.get("training_support_identity")
        != config["training_support_identity"].get("chosen_identity")
        or token.get("training_support_artifact_hash_status") != "PASS"
        or int(token.get("training_support_checked_stream_count", -1))
        != len(expected_stream_ids(config, training_horizons(config, str(token["training_support_identity"]))))
    ):
        raise RuntimeError("preflight gate token 的配置、算法、训练身份或全库存哈希门失配")
    report_path = Path(str(token.get("preflight_report_path"))).resolve()
    report_root = (CONTROL_ROOT / "preflight_runs").resolve()
    report = load_json(report_path) if report_path.is_file() else {}
    if (
        not report_path.is_relative_to(report_root)
        or not report_path.is_file()
        or sha256_file(report_path) != token.get("preflight_report_sha256")
        or report.get("status") != "PASS"
        or sha256_bytes(canonical_json_bytes(report.get("legacy_reference_anchor_audit")))
        != token.get("legacy_reference_anchor_audit_sha256")
        or report.get("disk_capacity_audit", {}).get("status") != "PASS"
    ):
        raise RuntimeError("preflight gate token 引用的不可变报告失配")
    current_registry = tuned_single_conformal_registry_audit(config)
    current_registry_sha = sha256_bytes(canonical_json_bytes(current_registry))
    if (
        current_registry_sha != token.get("registry_audit_sha256")
        or report.get("registry_audit") != current_registry
    ):
        raise RuntimeError("端点 worker TSC 注册表语义审计与受信 preflight 失配")
    environment = environment_audit(config)
    if environment["status"].ne("PASS").any():
        raise RuntimeError("端点 worker 当前环境已偏离封存复刻环境")
    if sha256_bytes(canonical_json_bytes(environment.to_dict("records"))) != token.get(
        "environment_audit_sha256"
    ):
        raise RuntimeError("端点 worker 环境审计摘要与 token 失配")
    inputs = endpoint_input_hash_audit(config)
    if sha256_bytes(canonical_json_bytes(inputs.to_dict("records"))) != token.get(
        "endpoint_input_hash_audit_sha256"
    ):
        raise RuntimeError("端点 worker 输入哈希审计摘要与 token 失配")
    current_disk = disk_capacity_audit(config)
    if (
        current_disk["status"] != "PASS"
        or int(current_disk["required_free_bytes"])
        != int(token.get("disk_contract_required_free_bytes", -1))
    ):
        raise RuntimeError("端点 worker 当前磁盘容量或冻结投影合同失配")
    identity = str(token["training_support_identity"])
    if int(task["horizon"]) not in set(training_horizons(config, identity)):
        raise RuntimeError("端点 worker 任务时距越出已批准训练身份")
    return token


def run_endpoint_unit(task: dict[str, Any]) -> dict[str, Any]:
    if not ENDPOINT_BUILD_IMPLEMENTATION_ENABLED:
        raise RuntimeError(
            "端点写入口已硬禁用：当前版本仅允许 plan/preflight；"
            "必须先完成封存 preflight token 与 resume gate 的独立复审"
        )
    started = time.perf_counter()
    config = load_endpoint_config()
    preflight_token = validate_preflight_gate_token(config, task)
    identities = endpoint_artifact_identity(config)
    zone = str(task["zone"])
    seed = int(task["seed"])
    horizon = int(task["horizon"])
    identifier = endpoint_unit_id(zone, seed, horizon)
    final_dir = ENDPOINT_ROOT / identifier
    complete = (
        _complete_endpoint_manifest(
            final_dir,
            identities,
            config=config,
            zone=zone,
            seed=seed,
            horizon=horizon,
        )
        if final_dir.exists()
        else None
    )
    if complete is not None:
        return {**complete, "resumed": True}
    if final_dir.exists():
        raise RuntimeError(f"既有新增动作端点单元不完整或身份失配，拒绝覆盖: {final_dir}")
    ENDPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary_dir = ENDPOINT_ROOT / f".{identifier}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    temporary_dir.mkdir(parents=True, exist_ok=False)
    try:
        source_anchor_audit = endpoint_unit_source_anchor_audit(
            config,
            zone=zone,
            seed=seed,
            horizon=horizon,
        )
        baseline = _baseline_module()
        coverages = [float(value) for value in config["scope"]["target_coverages"]]
        predictors = [str(value) for value in config["scope"]["predictors"]]
        events, event_input_audit = baseline.load_existing_event_states(
            zone=zone,
            seed=seed,
            horizon=horizon,
            predictors=predictors,
            target_coverages=coverages,
        )
        events = events.reset_index(drop=True)
        expected_events = expected_endpoint_unit_event_count(
            config,
            zone=zone,
            seed=seed,
            horizon=horizon,
        )
        if (
            len(events) != expected_events
            or events["event_id"].duplicated().any()
            or set(events["target_coverage"].astype(float)) != set(coverages)
            or set(events["predictor"].astype(str)) != set(predictors)
            or events["zone_or_farm"].astype(str).ne(zone).any()
            or events["seed"].astype(int).ne(seed).any()
            or events["horizon_steps"].astype(int).ne(horizon).any()
        ):
            raise RuntimeError(f"端点单元事件范围失配: {identifier}")
        candidates, bundle_input_audit = baseline.load_candidate_evidence(
            events=events,
            zone=zone,
            seed=seed,
            horizon=horizon,
            predictors=predictors,
        )
        contracts = baseline.load_frozen_contracts()
        registry_all = pd.read_parquet(REGISTRY_PATH)
        registry = registry_all[
            registry_all["outer_heldout_zone"].astype(str).eq(zone)
        ].set_index("baseline_id")
        tuned, conformal_audit_rows = baseline.tuned_conformal_decisions(
            contracts=contracts,
            events=events,
            registry_row=registry.loc["TunedSingleConformal"],
            zone=zone,
            seed=seed,
            horizon=horizon,
            predictors=predictors,
        )
        ensemble = baseline.ensemble_decisions(
            contracts=contracts,
            events=events,
            candidates=candidates,
        )
        addons = events[
            ["event_id", "zone_or_farm", "predictor", "seed", "horizon_steps", "target_coverage"]
        ].copy()
        reliability_audits: dict[str, Any] = {}
        for action, decisions in (
            ("TunedSingleConformal", tuned),
            ("EqualEndpointEnsemble", ensemble),
        ):
            lower, upper = _aligned_endpoints(events, decisions, action)
            covered = (
                (lower <= events["target_after_maturity"].to_numpy(dtype=np.float64))
                & (events["target_after_maturity"].to_numpy(dtype=np.float64) <= upper)
            )
            tuwr, ard, reliability_audit = _reliability_arrays(events, covered)
            addons[f"{action}__candidate_lower"] = lower
            addons[f"{action}__candidate_upper"] = upper
            addons[f"{action}__covered"] = covered
            addons[f"{action}__tuwr_indicator"] = tuwr
            addons[f"{action}__ard_value"] = ard
            reliability_audits[action] = reliability_audit
        if len(addons) != len(events) or addons["event_id"].duplicated().any():
            raise RuntimeError("新增动作端点输出事件不守恒")
        if horizon in set(int(value) for value in config["endpoint_contract"]["regression_horizons"]):
            regression = _legacy_regression(
                config=config,
                events=events,
                addons=addons,
                zone=zone,
                seed=seed,
                horizon=horizon,
            )
        else:
            regression = {
                "status": "NOT_APPLICABLE_NON_REFERENCE_HORIZON",
                "reason": "legacy three-coverage reference exists only for evaluation horizons",
            }
        atomic_parquet(temporary_dir / "endpoint_addons.parquet", addons)
        atomic_csv(temporary_dir / "event_input_audit.csv", event_input_audit)
        atomic_csv(temporary_dir / "bundle_input_audit.csv", bundle_input_audit)
        conformal_audit = {
            "status": "PASS",
            "rows": conformal_audit_rows,
            "reliability": reliability_audits,
        }
        atomic_json(temporary_dir / "conformal_audit.json", conformal_audit)
        atomic_json(temporary_dir / "regression_audit.json", regression)
        atomic_json(temporary_dir / "source_anchor_audit.json", source_anchor_audit)
        artifact_names = (
            "endpoint_addons.parquet",
            "event_input_audit.csv",
            "bundle_input_audit.csv",
            "conformal_audit.json",
            "regression_audit.json",
            "source_anchor_audit.json",
        )
        manifest: dict[str, Any] = {
            "schema": "TEST_CLARA_GEFCOM_6A_ENDPOINT_UNIT_V1",
            "status": "PASS",
            "generated_at_utc": utc_now(),
            "unit_id": identifier,
            "zone": zone,
            "seed": seed,
            "horizon": horizon,
            "identities": identities,
            "event_count": len(events),
            "coverage_count": int(events["target_coverage"].nunique()),
            "predictor_count": int(events["predictor"].nunique()),
            "addon_action_count": len(ADDON_ACTIONS),
            "legacy_regression_status": regression["status"],
            "legacy_regression_reference_path": regression.get("reference_path"),
            "legacy_regression_reference_sha256": regression.get("reference_sha256"),
            "legacy_regression_reference_unit_manifest_sha256": regression.get(
                "reference_unit_manifest_sha256"
            ),
            "preflight_token_sha256": str(task["preflight_token_sha256"]),
            "preflight_token_id": Path(str(task["preflight_token_path"])).resolve().parent.name,
            "preflight_report_sha256": preflight_token["preflight_report_sha256"],
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        for name in artifact_names:
            manifest[name.replace(".", "_") + "_sha256"] = sha256_file(temporary_dir / name)
        atomic_json(temporary_dir / "manifest.json", manifest)
        ENDPOINT_ROOT.mkdir(parents=True, exist_ok=True)
        temporary_dir.rename(final_dir)
        return manifest
    except Exception as error:
        atomic_json(
            temporary_dir / "exception.json",
            {
                "schema": "TEST_CLARA_GEFCOM_6A_ENDPOINT_EXCEPTION_V1",
                "status": "FAILED_PRESERVED",
                "generated_at_utc": utc_now(),
                "unit_id": identifier,
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def endpoint_status(config: dict[str, Any], identity: str) -> dict[str, Any]:
    identities = endpoint_artifact_identity(config)
    complete: list[str] = []
    incomplete: list[str] = []
    missing: list[str] = []
    tasks = endpoint_tasks(config, identity)
    for task in tasks:
        identifier = endpoint_unit_id(**task)
        path = ENDPOINT_ROOT / identifier
        if not path.exists():
            missing.append(identifier)
        elif _complete_endpoint_manifest(
            path,
            identities,
            config=config,
            zone=str(task["zone"]),
            seed=int(task["seed"]),
            horizon=int(task["horizon"]),
        ) is None:
            incomplete.append(identifier)
        else:
            complete.append(identifier)
    temporary = sorted(path.name for path in ENDPOINT_ROOT.glob(".*.tmp.*")) if ENDPOINT_ROOT.exists() else []
    return {
        "schema": "TEST_CLARA_GEFCOM_6A_ENDPOINT_STATUS_V1",
        "status": (
            "COMPLETE"
            if not missing and not incomplete and not temporary
            else "BLOCKED_TEMPORARY_UNITS_PRESENT"
            if temporary
            else "INCOMPLETE"
        ),
        "generated_at_utc": utc_now(),
        "training_support_identity": identity,
        "expected_unit_count": len(tasks),
        "complete_unit_count": len(complete),
        "missing_unit_count": len(missing),
        "incomplete_unit_count": len(incomplete),
        "preserved_temporary_unit_count": len(temporary),
        "complete_units": complete,
        "missing_units": missing,
        "incomplete_units": incomplete,
        "preserved_temporary_units": temporary,
    }


def seal_endpoint_root(
    config: dict[str, Any],
    identity: str,
    *,
    preflight_token_path: str | None = None,
    preflight_token_sha256: str | None = None,
) -> dict[str, Any]:
    """Atomically seal a complete endpoint cache; selectors must pin this seal."""

    if not preflight_token_path or not preflight_token_sha256:
        raise RuntimeError("端点根封存必须显式提供当次封存 preflight token")
    tasks = endpoint_tasks(config, identity)
    if not tasks:
        raise RuntimeError("端点根封存任务范围为空")
    sealing_token = validate_preflight_gate_token(
        config,
        {
            **tasks[0],
            "preflight_token_path": preflight_token_path,
            "preflight_token_sha256": preflight_token_sha256,
        },
    )
    status = endpoint_status(config, identity)
    if status["status"] != "COMPLETE":
        raise RuntimeError(f"端点根封存前单元状态未完成: {status['status']}")
    identities = endpoint_artifact_identity(config)
    unit_records: list[dict[str, Any]] = []
    total_events = 0
    expected_total_events = 0
    regression_reference_count = 0
    regression_reference_hashes: set[str] = set()
    preflight_token_hashes: set[str] = set()
    preflight_token_ids: set[str] = set()
    for task in tasks:
        unit_id = endpoint_unit_id(**task)
        path = ENDPOINT_ROOT / unit_id
        manifest = _complete_endpoint_manifest(
            path,
            identities,
            config=config,
            zone=str(task["zone"]),
            seed=int(task["seed"]),
            horizon=int(task["horizon"]),
        )
        if manifest is None:
            raise RuntimeError(f"端点根封存时单元身份失配: {unit_id}")
        manifest_sha = sha256_file(path / "manifest.json")
        unit_token_id = str(manifest["preflight_token_id"])
        unit_token_hash = str(manifest["preflight_token_sha256"])
        unit_token_path = CONTROL_ROOT / "preflight_tokens" / unit_token_id / "manifest.json"
        unit_token = validate_preflight_gate_token(
            config,
            {
                **task,
                "preflight_token_path": str(unit_token_path),
                "preflight_token_sha256": unit_token_hash,
            },
        )
        if manifest.get("preflight_report_sha256") != unit_token.get("preflight_report_sha256"):
            raise RuntimeError(f"端点单元 token/report SHA 关联失配: {unit_id}")
        expected_unit_events = expected_endpoint_unit_event_count(
            config,
            zone=str(task["zone"]),
            seed=int(task["seed"]),
            horizon=int(task["horizon"]),
        )
        unit_records.append(
            {
                "unit_id": unit_id,
                "unit_manifest_sha256": manifest_sha,
                "event_count": int(manifest["event_count"]),
                "expected_event_count": expected_unit_events,
                "preflight_token_id": unit_token_id,
                "preflight_token_sha256": unit_token_hash,
                "preflight_report_sha256": str(manifest["preflight_report_sha256"]),
            }
        )
        total_events += int(manifest["event_count"])
        expected_total_events += expected_unit_events
        preflight_token_hashes.add(str(manifest["preflight_token_sha256"]))
        preflight_token_ids.add(unit_token_id)
        if manifest["legacy_regression_status"] == "PASS":
            regression_reference_count += 1
            regression_reference_hashes.add(
                str(manifest["legacy_regression_reference_sha256"])
            )
    expected_units = int(
        config["scope"]["expected_endpoint_unit_count_by_training_identity"][identity]
    )
    expected_regression_units = (
        len(config["scope"]["zones"])
        * len(config["scope"]["seeds"])
        * len(config["endpoint_contract"]["regression_horizons"])
    )
    if len(unit_records) != expected_units or regression_reference_count != expected_regression_units:
        raise RuntimeError("端点根封存的单元或回归参考数量不守恒")
    if total_events != expected_total_events:
        raise RuntimeError("端点根封存的逐单元事件数不守恒")
    if (
        identity == "B_UNIFIED_FIVE_HORIZON_NEW_IDENTITY"
        and total_events != int(config["scope"]["expected_target_event_count_per_method_price"])
    ):
        raise RuntimeError("B 统一五时距端点根事件总数不等于 23,024,760")
    payload = {
        "schema": "TEST_CLARA_GEFCOM_6A_ENDPOINT_ROOT_SEAL_V1",
        "status": "PASS",
        "generated_at_utc": utc_now(),
        "training_support_identity": identity,
        "training_horizons": list(training_horizons(config, identity)),
        "identities": identities,
        "unit_count": len(unit_records),
        "total_event_count": total_events,
        "expected_total_event_count": expected_total_events,
        "total_event_action_count": total_events * len(ADDON_ACTIONS),
        "coverage_unit_count": len(unit_records) * len(config["scope"]["target_coverages"]),
        "predictor_unit_count": len(unit_records) * len(config["scope"]["predictors"]),
        "addon_action_unit_count": len(unit_records) * len(ADDON_ACTIONS),
        "legacy_regression_unit_count": regression_reference_count,
        "legacy_regression_reference_hashes": sorted(regression_reference_hashes),
        "preflight_token_hashes": sorted(preflight_token_hashes),
        "preflight_token_ids": sorted(preflight_token_ids),
        "unit_manifests": sorted(unit_records, key=lambda row: row["unit_id"]),
    }
    root_path = ENDPOINT_ROOT / f"root_manifest__{identity}.json"
    if root_path.exists():
        existing = load_json(root_path)
        comparison_fields = [
            "schema",
            "status",
            "training_support_identity",
            "training_horizons",
            "identities",
            "unit_count",
            "total_event_count",
            "expected_total_event_count",
            "total_event_action_count",
            "coverage_unit_count",
            "predictor_unit_count",
            "addon_action_unit_count",
            "legacy_regression_unit_count",
            "legacy_regression_reference_hashes",
            "preflight_token_hashes",
            "preflight_token_ids",
            "unit_manifests",
        ]
        if any(existing.get(field) != payload.get(field) for field in comparison_fields):
            raise RuntimeError("既有端点根封存身份或守恒字段失配，拒绝覆盖")
        return {**existing, "resumed": True, "root_manifest_sha256": sha256_file(root_path)}
    atomic_json(root_path, payload)
    return {**payload, "root_manifest_sha256": sha256_file(root_path)}


def run_endpoints(*, workers: int, only_unit: str | None) -> dict[str, Any]:
    if not ENDPOINT_BUILD_IMPLEMENTATION_ENABLED:
        raise RuntimeError(
            "build-endpoints 已硬禁用；当前版本只提供双路线 plan 与 fail-closed preflight"
        )
    config = load_endpoint_config()
    preflight_report = preflight(
        require_exact_environment=True,
        write_report=True,
        issue_token=True,
    )
    token = preflight_report["preflight_token"]
    write_endpoint_plan()
    if workers < 1 or workers > int(config["execution"]["maximum_workers"]):
        raise ValueError(f"workers 必须在 1-{int(config['execution']['maximum_workers'])} 之间")
    identity = str(config["training_support_identity"]["chosen_identity"])
    tasks = [
        {
            **task,
            "preflight_token_path": str(token["path"]),
            "preflight_token_sha256": str(token["sha256"]),
        }
        for task in endpoint_tasks(config, identity)
    ]
    if only_unit is not None:
        tasks = [
            task
            for task in tasks
            if endpoint_unit_id(
                str(task["zone"]), int(task["seed"]), int(task["horizon"])
            )
            == only_unit
        ]
        if len(tasks) != 1:
            raise ValueError(f"未知端点单元: {only_unit}")
        workers = 1
    failures: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    started = time.perf_counter()
    if workers == 1:
        for task in tasks:
            try:
                completed.append(run_endpoint_unit(task))
            except Exception as error:
                failures.append(
                    {
                        "task": task,
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
                break
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(run_endpoint_unit, task): task for task in tasks}
            for future in as_completed(future_map):
                task = future_map[future]
                try:
                    completed.append(future.result())
                except Exception as error:
                    failures.append(
                        {
                            "task": task,
                            "error_type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    for pending in future_map:
                        pending.cancel()
                    break
    invocation = {
        "schema": "TEST_CLARA_GEFCOM_6A_ENDPOINT_INVOCATION_V1",
        "status": "FAIL" if failures else "PASS",
        "generated_at_utc": utc_now(),
        "preflight_token_sha256": str(token["sha256"]),
        "preflight_report_sha256": str(
            load_json(Path(str(token["path"])))["preflight_report_sha256"]
        ),
        "workers": workers,
        "requested_unit_count": len(tasks),
        "completed_or_reused_unit_count": len(completed),
        "failure_count": len(failures),
        "elapsed_seconds": float(time.perf_counter() - started),
        "failures": failures,
    }
    invocation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    atomic_json(CONTROL_ROOT / "endpoint_invocations" / f"{invocation_id}.json", invocation)
    if failures:
        raise RuntimeError(f"新增动作端点阶段失败: {failures[0]['message']}")
    status = endpoint_status(config, identity)
    root_seal = (
        seal_endpoint_root(
            config,
            identity,
            preflight_token_path=str(token["path"]),
            preflight_token_sha256=str(token["sha256"]),
        )
        if status["status"] == "COMPLETE"
        else None
    )
    return {"invocation": invocation, "endpoint_status": status, "endpoint_root_seal": root_seal}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight", help="校验冻结矩阵、输入库存与环境")
    preflight_parser.add_argument(
        "--allow-environment-drift-for-development",
        action="store_true",
        help="只允许生成开发期 WARN 报告；端点构建仍强制精确环境",
    )
    preflight_parser.add_argument("--write-report", action="store_true")

    subparsers.add_parser("plan", help="分别生成 A=782 与 B=212 父单元，另附300个逐价子单元")
    subparsers.add_parser(
        "endpoint-plan",
        help="生成不绑定总runner/selector-core SHA的B路线150单元端点计划",
    )
    status_parser = subparsers.add_parser("status", help="读取端点阶段状态，不修改工件")
    status_parser.add_argument(
        "--identity",
        choices=(
            "A_FULL_24_HORIZON_SOURCE_SUPPORT",
            "B_UNIFIED_FIVE_HORIZON_NEW_IDENTITY",
        ),
        default=None,
    )

    endpoint_parser = subparsers.add_parser(
        "build-endpoints",
        help="仅在精确环境、全库哈希、迁移锚和回归锚预检 PASS 后构建端点",
    )
    endpoint_parser.add_argument("--workers", type=int, default=1)
    endpoint_parser.add_argument("--only-unit", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        result = preflight(
            require_exact_environment=not bool(args.allow_environment_drift_for_development),
            write_report=bool(args.write_report),
            issue_token=bool(args.write_report),
        )
    elif args.command == "plan":
        result = write_plan()
    elif args.command == "endpoint-plan":
        result = write_endpoint_plan()
    elif args.command == "status":
        config = load_endpoint_config()
        chosen = args.identity or config["training_support_identity"].get("chosen_identity")
        if chosen is None:
            result = {
                identity: endpoint_status(config, identity)
                for identity in (
                    "A_FULL_24_HORIZON_SOURCE_SUPPORT",
                    "B_UNIFIED_FIVE_HORIZON_NEW_IDENTITY",
                )
            }
        else:
            result = endpoint_status(config, str(chosen))
    elif args.command == "build-endpoints":
        result = run_endpoints(workers=int(args.workers), only_unit=args.only_unit)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
