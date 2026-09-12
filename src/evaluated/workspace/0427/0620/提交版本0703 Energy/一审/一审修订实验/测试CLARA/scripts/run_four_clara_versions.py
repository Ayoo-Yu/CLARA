from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


TEST_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = TEST_ROOT.parent
CONFIG_PATH = TEST_ROOT / "configs" / "test_clara_four_versions_v1.json"
OUTPUT_ROOT = TEST_ROOT / "results_raw" / "four_version_comparison_v1"
PRICE_ROOT = REVISION_ROOT / "08_敏感性分析与消融" / "01_价格敏感性"
PRICE_CONFIG_PATH = PRICE_ROOT / "configs" / "price_sensitivity_protocol_v1.json"
S06_ROOT = REVISION_ROOT / "06_基线实现与训练区调参"
S07_ROOT = REVISION_ROOT / "07_GEFCom完整主实验"
AUTH_CODE = REVISION_ROOT / "_权威代码" / "code"
INDEX_CONTRACT_PATH = S07_ROOT / "configs" / "s07_landscape_index_contract_v1.json"
WIDTH_THRESHOLDS_PATH = (
    S06_ROOT
    / "results_raw"
    / "nested_source_selection_v1"
    / "fold_width_thresholds.parquet"
)
PRICE_CACHE_ROOT = PRICE_ROOT / "results_raw" / "price_decision_cache_v1"
SEALED_PRICE_SUMMARY = (
    PRICE_ROOT
    / "results_verified"
    / "price_sensitivity_v1"
    / "price_comparison.csv"
)

sys.path.insert(0, str(AUTH_CODE))
sys.path.insert(0, str(S07_ROOT / "scripts"))
sys.path.insert(0, str(PRICE_ROOT / "scripts"))

from clara_event_contract import load_frozen_contracts  # noqa: E402
import s07_decision_cache_accelerator as accelerator  # noqa: E402
import build_price_decision_caches as price_builder  # noqa: E402


ACTIONS = tuple(price_builder.ACTIONS)
STATE_FIELDS = list(price_builder.STATE_FIELDS)


def utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs(config: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, specification in config["inputs"].items():
        path = REVISION_ROOT / str(specification["relative_path"])
        observed = sha256_file(path) if path.is_file() else None
        expected = str(specification["sha256"])
        rows.append(
            {
                "input_name": name,
                "path": str(path),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "status": "PASS" if observed == expected else "FAIL",
            }
        )
    failures = [row for row in rows if row["status"] != "PASS"]
    if failures:
        raise RuntimeError(f"冻结输入哈希失配: {failures}")
    return {"status": "PASS", "inputs": rows}


def ensure_disk_gate(config: dict[str, Any]) -> float:
    free_gib = shutil.disk_usage(TEST_ROOT).free / (1024.0**3)
    minimum = float(config["execution"]["minimum_free_disk_gib"])
    if free_gib < minimum:
        raise RuntimeError(
            f"磁盘安全门失败: free={free_gib:.3f} GiB, minimum={minimum:.3f} GiB"
        )
    return float(free_gib)


def parameterized_contracts(contracts: Any, n_min: int, nu: int) -> Any:
    protocol = copy.deepcopy(contracts.protocol)
    support = protocol["support_and_backoff_contract"]
    support["default_n_min"] = int(n_min)
    support["default_nu"] = int(nu)
    return replace(contracts, protocol=protocol)


def subset_zone_arrays(
    arrays: dict[str, np.ndarray], zone_indices: Iterable[int]
) -> dict[str, np.ndarray]:
    selected = np.asarray(list(zone_indices), dtype=np.int64)
    result: dict[str, np.ndarray] = {}
    for name, values in arrays.items():
        if values.ndim == 2:
            result[name] = values[selected, :]
        elif values.ndim == 3:
            result[name] = values[:, selected, :]
        else:
            raise RuntimeError(f"未识别的 Zone 数组维度: {name}/{values.shape}")
    return result


def adaptive_guardrail_thresholds(
    ratio: float, config: dict[str, Any]
) -> dict[str, float]:
    anchors = sorted(
        config["adaptive_guardrail"]["anchors"],
        key=lambda row: float(row["miss_to_capacity_ratio"]),
    )
    current = float(ratio)
    if current <= float(anchors[0]["miss_to_capacity_ratio"]):
        selected = anchors[0]
        return {
            "coverage_shortfall_epsilon": float(
                selected["coverage_shortfall_epsilon"]
            ),
            "tuwr_upper": float(selected["tuwr_upper"]),
            "ard_upper": float(selected["ard_upper"]),
        }
    if current >= float(anchors[-1]["miss_to_capacity_ratio"]):
        selected = anchors[-1]
        return {
            "coverage_shortfall_epsilon": float(
                selected["coverage_shortfall_epsilon"]
            ),
            "tuwr_upper": float(selected["tuwr_upper"]),
            "ard_upper": float(selected["ard_upper"]),
        }
    for left, right in zip(anchors[:-1], anchors[1:]):
        left_ratio = float(left["miss_to_capacity_ratio"])
        right_ratio = float(right["miss_to_capacity_ratio"])
        if left_ratio <= current <= right_ratio:
            fraction = (math.log(current) - math.log(left_ratio)) / (
                math.log(right_ratio) - math.log(left_ratio)
            )
            return {
                name: float(left[name])
                + fraction * (float(right[name]) - float(left[name]))
                for name in (
                    "coverage_shortfall_epsilon",
                    "tuwr_upper",
                    "ard_upper",
                )
            }
    raise RuntimeError(f"价格护栏插值失败: ratio={current}")


def fixed_guardrail_thresholds(config: dict[str, Any]) -> dict[str, float]:
    fixed = config["fixed_guardrail"]
    return {
        "coverage_shortfall_epsilon": float(
            fixed["coverage_shortfall_epsilon"]
        ),
        "tuwr_upper": float(fixed["tuwr_upper"]),
        "ard_upper": float(fixed["ard_upper"]),
    }


def ordered_evidence_matrices(
    evidence: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    action_count = len(ACTIONS)
    if len(evidence) % action_count:
        raise RuntimeError("动作证据行数不能被动作数量整除")
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
        state_codes[:, 0], np.arange(state_count, dtype=np.int64)
    ):
        raise RuntimeError("状态编号不连续")
    expected_actions = np.broadcast_to(
        np.arange(action_count, dtype=np.int64), action_orders.shape
    )
    if not np.array_equal(action_orders, expected_actions):
        raise RuntimeError("动作顺序身份失配")
    return ordered, state_count, action_count


def select_actions(
    evidence: pd.DataFrame,
    thresholds: dict[str, float],
    contracts: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered, state_count, action_count = ordered_evidence_matrices(evidence)
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
    cold = ordered["cold_start_guardrail"].to_numpy(dtype=bool).reshape(
        state_count, action_count
    )[:, 0]
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
            raise RuntimeError(f"并列候选 {name} 可用性不一致")
        applicable = (finite_count == candidate_count) & (candidate_count > 0)
        minimum = np.min(np.where(candidates & finite, values, np.inf), axis=1)
        candidates &= (~applicable[:, None]) | (
            finite & (values <= minimum[:, None] + tolerance)
        )
    if not candidates.any(axis=1).all():
        raise RuntimeError("动作选择产生空候选")
    selected = np.argmax(candidates, axis=1).astype(np.int64)
    return selected, empty, safe


def risk_margin_se_ratio(evidence: pd.DataFrame) -> np.ndarray:
    ordered, state_count, action_count = ordered_evidence_matrices(evidence)
    scores = ordered["risk_score"].to_numpy(dtype=np.float64).reshape(
        state_count, action_count
    )
    standard_errors = ordered["risk_zone_cluster_se"].to_numpy(
        dtype=np.float64
    ).reshape(state_count, action_count)
    order = np.argsort(scores, axis=1, kind="stable")
    rows = np.arange(state_count, dtype=np.int64)
    best = order[:, 0]
    second = order[:, 1]
    gap = scores[rows, second] - scores[rows, best]
    denominator = np.sqrt(
        standard_errors[rows, best] ** 2
        + standard_errors[rows, second] ** 2
    )
    ratio = np.zeros(state_count, dtype=np.float64)
    positive_denominator = denominator > 0
    ratio[positive_denominator] = (
        gap[positive_denominator] / denominator[positive_denominator]
    )
    ratio[(~positive_denominator) & (gap > 0)] = np.inf
    if not np.isfinite(ratio[np.isfinite(ratio)]).all():
        raise RuntimeError("风险排序置信度包含非法数值")
    return ratio


def choose_adaptive_n_min(
    evidence_by_n: dict[int, pd.DataFrame], config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    candidates = sorted(int(value) for value in evidence_by_n)
    threshold = float(
        config["adaptive_support"]["risk_margin_se_ratio_threshold"]
    )
    fallback = int(config["adaptive_support"]["n_min_no_candidate_rule"])
    ratios = {value: risk_margin_se_ratio(evidence_by_n[value]) for value in candidates}
    state_count = len(next(iter(ratios.values())))
    selected = np.full(state_count, -1, dtype=np.int64)
    selected_ratio = np.zeros(state_count, dtype=np.float64)
    for value in candidates:
        current = (selected < 0) & (ratios[value] >= threshold)
        selected[current] = int(value)
        selected_ratio[current] = ratios[value][current]
    remaining = selected < 0
    selected[remaining] = fallback
    selected_ratio[remaining] = ratios[fallback][remaining]
    return selected, selected_ratio, ratios


def stitch_profile(
    profiles: dict[int, pd.DataFrame], selected_n_min: np.ndarray
) -> pd.DataFrame:
    fallback = max(profiles)
    output = profiles[fallback].copy().reset_index(drop=True)
    support_columns = [
        "support_backoff_level",
        "support_backoff_name",
        "selected_group_id",
        "support_n",
        "support_source_zones",
        "bootstrap_supported",
        "cold_start_guardrail",
    ]
    for value, frame in profiles.items():
        current = selected_n_min == int(value)
        if current.any():
            ordered = frame.reset_index(drop=True)
            for column in support_columns:
                output.loc[current, column] = ordered.loc[current, column]
    return output


def choose_adaptive_nu(
    diagnostic_evidence: pd.DataFrame, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    ordered, state_count, action_count = ordered_evidence_matrices(
        diagnostic_evidence
    )
    risk_mean = ordered["risk_mean"].to_numpy(dtype=np.float64).reshape(
        state_count, action_count
    )
    risk_parent = ordered["risk_parent_mean"].to_numpy(dtype=np.float64).reshape(
        state_count, action_count
    )
    standard_error = ordered["risk_zone_cluster_se"].to_numpy(
        dtype=np.float64
    ).reshape(state_count, action_count)
    signal = np.nanmedian(np.abs(risk_mean - risk_parent), axis=1)
    noise = np.nanmedian(standard_error, axis=1)
    floor = float(config["adaptive_support"]["numeric_floor"])
    ratio = signal / np.maximum(noise, floor)
    selected = np.full(state_count, -1, dtype=np.int64)
    for rule in config["adaptive_support"]["nu_rules"]:
        current = (selected < 0) & (ratio >= float(rule["minimum_ratio"]))
        selected[current] = int(rule["nu"])
    if (selected < 0).any():
        raise RuntimeError("自适应 nu 规则未覆盖全部状态")
    return selected, ratio


def stitch_evidence(
    evidence_by_value: dict[int, pd.DataFrame], selected_values: np.ndarray
) -> pd.DataFrame:
    fallback = max(evidence_by_value)
    output = (
        evidence_by_value[fallback]
        .sort_values(["full_state_code", "action_order"], kind="mergesort")
        .reset_index(drop=True)
        .copy()
    )
    action_count = len(ACTIONS)
    shrinkage_columns = [
        "risk_parent_mean",
        "risk_shrunken_mean",
        "risk_score",
    ]
    for value, frame in evidence_by_value.items():
        state_mask = selected_values == int(value)
        if not state_mask.any():
            continue
        row_mask = np.repeat(state_mask, action_count)
        ordered = frame.sort_values(
            ["full_state_code", "action_order"], kind="mergesort"
        ).reset_index(drop=True)
        for column in shrinkage_columns:
            output.loc[row_mask, column] = ordered.loc[row_mask, column]
    return output


def adaptive_support_evidence(
    *,
    states: pd.DataFrame,
    contracts: Any,
    mappings: list[np.ndarray],
    levels: list[dict[str, np.ndarray]],
    profiles: dict[int, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    diagnostic_nu = int(config["adaptive_support"]["diagnostic_nu"])
    evidence_by_n: dict[int, pd.DataFrame] = {}
    for n_min, profile in profiles.items():
        current_contracts = parameterized_contracts(
            contracts, n_min=int(n_min), nu=diagnostic_nu
        )
        evidence_by_n[int(n_min)] = accelerator.compute_cache_evidence(
            profile=profile,
            contracts=current_contracts,
            mappings=mappings,
            levels=levels,
        )
    selected_n_min, selected_margin, _ = choose_adaptive_n_min(
        evidence_by_n, config
    )
    adaptive_profile = stitch_profile(profiles, selected_n_min)
    diagnostic_contracts = parameterized_contracts(
        contracts,
        n_min=int(config["fixed_support"]["n_min"]),
        nu=diagnostic_nu,
    )
    diagnostic_evidence = accelerator.compute_cache_evidence(
        profile=adaptive_profile,
        contracts=diagnostic_contracts,
        mappings=mappings,
        levels=levels,
    )
    selected_nu, selected_nu_ratio = choose_adaptive_nu(
        diagnostic_evidence, config
    )
    evidence_by_nu: dict[int, pd.DataFrame] = {diagnostic_nu: diagnostic_evidence}
    for nu in config["adaptive_support"]["nu_candidates"]:
        current_nu = int(nu)
        if current_nu in evidence_by_nu:
            continue
        current_contracts = parameterized_contracts(
            contracts,
            n_min=int(config["fixed_support"]["n_min"]),
            nu=current_nu,
        )
        evidence_by_nu[current_nu] = accelerator.compute_cache_evidence(
            profile=adaptive_profile,
            contracts=current_contracts,
            mappings=mappings,
            levels=levels,
        )
    adaptive_evidence = stitch_evidence(evidence_by_nu, selected_nu)
    fixed_evidence = evidence_by_n[int(config["fixed_support"]["n_min"])]
    return (
        fixed_evidence,
        adaptive_evidence,
        selected_n_min,
        selected_nu,
        selected_margin,
        selected_nu_ratio,
    )


def heldout_metrics(
    *,
    arrays: dict[str, np.ndarray],
    heldout_zone_index: int,
    selected_actions: np.ndarray,
    states: pd.DataFrame,
    capacity_weight: float,
    miss_weight: float,
) -> dict[str, Any]:
    state_positions = np.arange(len(states), dtype=np.int64)
    event_counts = arrays["event_count"][heldout_zone_index].astype(np.float64)
    guardrail_counts = arrays["guardrail_count"][heldout_zone_index].astype(
        np.float64
    )
    event_count = float(event_counts.sum())
    guardrail_event_count = float(guardrail_counts.sum())
    if event_count <= 0 or guardrail_event_count <= 0:
        raise RuntimeError("留出区事件数量为空")
    errf_sums = (
        float(capacity_weight)
        * arrays["capacity_sum"][:, heldout_zone_index, :]
        + float(miss_weight)
        * arrays["miss_sum"][:, heldout_zone_index, :]
    )
    selected_errf = errf_sums[selected_actions, state_positions]
    selected_covered = arrays["covered_sum"][
        selected_actions, heldout_zone_index, state_positions
    ]
    selected_tuwr = arrays["tuwr_sum"][
        selected_actions, heldout_zone_index, state_positions
    ]
    selected_ard = arrays["ard_sum"][
        selected_actions, heldout_zone_index, state_positions
    ]
    target_coverage = states["target_coverage"].to_numpy(dtype=np.float64)
    target_weighted = float(np.sum(target_coverage * event_counts) / event_count)
    coverage = float(selected_covered.sum() / event_count)
    return {
        "event_count": int(event_count),
        "guardrail_event_count": int(guardrail_event_count),
        "mean_errf": float(selected_errf.sum() / event_count),
        "coverage": coverage,
        "target_coverage_weighted": target_weighted,
        "coverage_gap": float(coverage - target_weighted),
        "tuwr": float(selected_tuwr.sum() / guardrail_event_count),
        "ard": float(selected_ard.sum() / guardrail_event_count),
    }


def fixed_reference_actions(
    outer_zone: str, seed: int, price_id: str, state_count: int
) -> np.ndarray:
    path = (
        PRICE_CACHE_ROOT
        / f"outer_fold={outer_zone}"
        / f"seed={int(seed)}"
        / f"price_id={price_id}"
        / "price_decision_cache.parquet"
    )
    reference = pd.read_parquet(
        path, columns=["full_state_code", "selected_action"]
    )
    states = (
        reference.drop_duplicates("full_state_code")
        .sort_values("full_state_code", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(states) != state_count:
        raise RuntimeError(
            f"固定版本参考状态数量失配: {outer_zone}/seed{seed}/{price_id}"
        )
    action_map = {action: index for index, action in enumerate(ACTIONS)}
    mapped = states["selected_action"].astype(str).map(action_map)
    if mapped.isna().any():
        raise RuntimeError("固定版本参考包含未知动作")
    return mapped.to_numpy(dtype=np.int64)


def build_state_decisions(
    *,
    outer_zone: str,
    seed: int,
    price_id: str,
    price_ratio: float,
    version: str,
    evidence: pd.DataFrame,
    selected: np.ndarray,
    empty: np.ndarray,
    safe: np.ndarray,
    selected_n_min: np.ndarray,
    selected_nu: np.ndarray,
    selected_margin: np.ndarray,
    selected_nu_ratio: np.ndarray,
    thresholds: dict[str, float],
    fixed_selected: np.ndarray,
) -> pd.DataFrame:
    ordered, state_count, action_count = ordered_evidence_matrices(evidence)
    rows = np.arange(state_count, dtype=np.int64)
    positions = rows * action_count + selected
    picked = ordered.iloc[positions].reset_index(drop=True)
    columns = [
        "full_state_code",
        *STATE_FIELDS,
        "action",
        "support_backoff_level",
        "support_backoff_name",
        "support_n",
        "support_source_zones",
        "cold_start_guardrail",
        "risk_mean",
        "risk_parent_mean",
        "risk_shrunken_mean",
        "risk_zone_cluster_se",
        "risk_score",
        "coverage_point",
        "tuwr_point",
        "ard_point",
    ]
    output = picked[columns].copy()
    output = output.rename(columns={"action": "selected_action"})
    output.insert(0, "version", version)
    output.insert(0, "miss_to_capacity_ratio", float(price_ratio))
    output.insert(0, "price_id", price_id)
    output.insert(0, "seed", int(seed))
    output.insert(0, "outer_heldout_zone", outer_zone)
    output["selected_n_min"] = selected_n_min.astype(np.int64)
    output["selected_nu"] = selected_nu.astype(np.int64)
    output["support_margin_se_ratio"] = selected_margin.astype(np.float64)
    output["nu_signal_noise_ratio"] = selected_nu_ratio.astype(np.float64)
    output["coverage_shortfall_epsilon"] = float(
        thresholds["coverage_shortfall_epsilon"]
    )
    output["tuwr_upper"] = float(thresholds["tuwr_upper"])
    output["ard_upper"] = float(thresholds["ard_upper"])
    output["guardrail_empty_fallback"] = empty.astype(bool)
    output["selected_action_guardrail_pass"] = safe[rows, selected].astype(bool)
    output["changed_from_v1"] = selected != fixed_selected
    return output


def unit_root(outer_zone: str, seed: int) -> Path:
    return OUTPUT_ROOT / f"outer_fold={outer_zone}" / f"seed={int(seed)}"


def completed_unit(
    outer_zone: str, seed: int, config_sha256: str
) -> dict[str, Any] | None:
    root = unit_root(outer_zone, seed)
    manifest_path = root / "unit_manifest.json"
    metrics_path = root / "heldout_metrics.csv"
    decisions_path = root / "state_decisions.parquet"
    if not manifest_path.is_file():
        return None
    manifest = load_json(manifest_path)
    valid = (
        manifest.get("status") == "PASS"
        and manifest.get("config_sha256") == config_sha256
        and metrics_path.is_file()
        and decisions_path.is_file()
        and manifest.get("heldout_metrics_sha256") == sha256_file(metrics_path)
        and manifest.get("state_decisions_sha256") == sha256_file(decisions_path)
    )
    if not valid:
        raise RuntimeError(f"已存在单元身份失配: {outer_zone}/seed{seed}")
    return {**manifest, "resume_status": "REUSED"}


def run_unit(task: dict[str, Any]) -> dict[str, Any]:
    outer_zone = str(task["outer_zone"])
    seed = int(task["seed"])
    config = task["config"]
    config_sha256 = str(task["config_sha256"])
    price_config = task["price_config"]
    component_config_sha256 = str(task["component_config_sha256"])
    existing = completed_unit(outer_zone, seed, config_sha256)
    if existing is not None:
        return existing
    root = unit_root(outer_zone, seed)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"未封存单元已存在: {outer_zone}/seed{seed}")
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "checkpoint.json"
    exception_path = root / "exception_ledger.jsonl"
    started = time.perf_counter()
    try:
        ensure_disk_gate(config)
        contracts = load_frozen_contracts()
        expected_protocol = config["causal_event_protocol"]
        if (
            contracts.protocol_id != expected_protocol["protocol_id"]
            or contracts.protocol_version != expected_protocol["protocol_version"]
            or contracts.protocol_sha256 != expected_protocol["sha256"]
        ):
            raise RuntimeError("因果事件协议身份失配")
        index_contract = load_json(INDEX_CONTRACT_PATH)
        base_states, states = accelerator.state_frames(
            index_contract, contracts.protocol
        )
        if len(states) != int(config["scope"]["query_state_count_per_fold_seed"]):
            raise RuntimeError("查询状态数量失配")
        thresholds_frame = pd.read_parquet(WIDTH_THRESHOLDS_PATH)
        q33, q67 = accelerator.threshold_vectors(
            thresholds_frame,
            base_states,
            heldout_zone=outer_zone,
            seed=seed,
        )
        ordered_zones = sorted(str(value) for value in config["scope"]["zones"])
        all_arrays, source_audit = price_builder.aggregate_fold_components(
            zones=ordered_zones,
            seed=seed,
            q33=q33,
            q67=q67,
            full_state_count=len(states),
            config_sha256=component_config_sha256,
        )
        if not source_audit["status"].eq("PASS").all():
            raise RuntimeError("价格成本分量聚合审计失败")
        heldout_index = ordered_zones.index(outer_zone)
        source_indices = [
            index for index, zone in enumerate(ordered_zones) if zone != outer_zone
        ]
        fit_zones = [ordered_zones[index] for index in source_indices]
        if outer_zone in fit_zones or len(fit_zones) != 9:
            raise RuntimeError("外层留出 Zone 被用于拟合")
        fit_arrays = subset_zone_arrays(all_arrays, source_indices)
        source_audit = source_audit.copy()
        source_audit["used_for_fit"] = source_audit["source_zone"].astype(
            str
        ).isin(fit_zones)
        source_audit_path = root / "component_audit.csv"
        source_audit.to_csv(
            source_audit_path, index=False, encoding="utf-8", lineterminator="\n"
        )
        mappings, group_frames = accelerator.factor_level_mappings(states, contracts)
        base_fit_arrays = price_builder.arrays_for_price(fit_arrays, 3.56, 20.0)
        base_levels = accelerator.build_level_aggregates(
            base_fit_arrays, mappings, group_frames
        )
        profiles: dict[int, pd.DataFrame] = {}
        for n_min in config["adaptive_support"]["n_min_candidates"]:
            current_contracts = parameterized_contracts(
                contracts,
                n_min=int(n_min),
                nu=int(config["fixed_support"]["nu"]),
            )
            profiles[int(n_min)] = accelerator.select_support_levels(
                states, current_contracts, mappings, base_levels
            )

        fixed_thresholds = fixed_guardrail_thresholds(config)
        metric_rows: list[dict[str, Any]] = []
        decision_frames: list[pd.DataFrame] = []
        fixed_regression_mismatch_count = 0
        price_grid = [
            row
            for row in price_config["price_grid"]
            if str(row["price_id"]) in set(config["scope"]["price_ids"])
        ]
        if len(price_grid) != len(config["scope"]["price_ids"]):
            raise RuntimeError("价格网格数量失配")
        for price_index, price in enumerate(price_grid):
            price_id = str(price["price_id"])
            price_ratio = float(price["miss_to_capacity_ratio"])
            theta = [float(value) for value in price["theta"]]
            if theta[0] != theta[1] or theta[2] != theta[3]:
                raise RuntimeError(f"价格方向对称合同失配: {price_id}")
            price_fit_arrays = price_builder.arrays_for_price(
                fit_arrays, theta[0], theta[2]
            )
            levels = accelerator.build_level_aggregates(
                price_fit_arrays, mappings, group_frames
            )
            (
                fixed_evidence,
                adaptive_evidence,
                adaptive_n_min,
                adaptive_nu,
                adaptive_margin,
                adaptive_nu_ratio,
            ) = adaptive_support_evidence(
                states=states,
                contracts=contracts,
                mappings=mappings,
                levels=levels,
                profiles=profiles,
                config=config,
            )
            fixed_selected, fixed_empty, fixed_safe = select_actions(
                fixed_evidence, fixed_thresholds, contracts
            )
            reference_selected = fixed_reference_actions(
                outer_zone, seed, price_id, len(states)
            )
            mismatch_count = int((fixed_selected != reference_selected).sum())
            fixed_regression_mismatch_count += mismatch_count
            if mismatch_count:
                raise RuntimeError(
                    f"固定版本动作回归失败: {outer_zone}/seed{seed}/{price_id}, "
                    f"mismatch={mismatch_count}"
                )
            adaptive_thresholds = adaptive_guardrail_thresholds(
                price_ratio, config
            )
            constant_n_min = np.full(
                len(states), int(config["fixed_support"]["n_min"]), dtype=np.int64
            )
            constant_nu = np.full(
                len(states), int(config["fixed_support"]["nu"]), dtype=np.int64
            )
            neutral_diagnostic = np.full(len(states), np.nan, dtype=np.float64)
            version_inputs = {
                "V1_FIXED": (
                    fixed_evidence,
                    fixed_thresholds,
                    constant_n_min,
                    constant_nu,
                    neutral_diagnostic,
                    neutral_diagnostic,
                ),
                "V2_ADAPTIVE_SUPPORT": (
                    adaptive_evidence,
                    fixed_thresholds,
                    adaptive_n_min,
                    adaptive_nu,
                    adaptive_margin,
                    adaptive_nu_ratio,
                ),
                "V3_ADAPTIVE_GUARDRAIL": (
                    fixed_evidence,
                    adaptive_thresholds,
                    constant_n_min,
                    constant_nu,
                    neutral_diagnostic,
                    neutral_diagnostic,
                ),
                "V4_FULLY_ADAPTIVE": (
                    adaptive_evidence,
                    adaptive_thresholds,
                    adaptive_n_min,
                    adaptive_nu,
                    adaptive_margin,
                    adaptive_nu_ratio,
                ),
            }
            for version in config["scope"]["versions"]:
                (
                    evidence,
                    current_thresholds,
                    selected_n_min,
                    selected_nu,
                    selected_margin,
                    selected_nu_ratio,
                ) = version_inputs[str(version)]
                if version == "V1_FIXED":
                    selected, empty, safe = fixed_selected, fixed_empty, fixed_safe
                else:
                    selected, empty, safe = select_actions(
                        evidence, current_thresholds, contracts
                    )
                metrics = heldout_metrics(
                    arrays=all_arrays,
                    heldout_zone_index=heldout_index,
                    selected_actions=selected,
                    states=states,
                    capacity_weight=theta[0],
                    miss_weight=theta[2],
                )
                profile_for_metrics = (
                    profiles[int(config["fixed_support"]["n_min"])]
                    if version in {"V1_FIXED", "V3_ADAPTIVE_GUARDRAIL"}
                    else stitch_profile(profiles, selected_n_min)
                )
                action_counts = np.bincount(
                    selected, minlength=len(ACTIONS)
                ).astype(np.int64)
                row: dict[str, Any] = {
                    "outer_heldout_zone": outer_zone,
                    "seed": int(seed),
                    "price_id": price_id,
                    "miss_to_capacity_ratio": price_ratio,
                    "version": str(version),
                    **metrics,
                    "state_count": len(states),
                    "mean_selected_n_min": float(np.mean(selected_n_min)),
                    "mean_selected_nu": float(np.mean(selected_nu)),
                    "mean_support_backoff_level": float(
                        profile_for_metrics["support_backoff_level"].mean()
                    ),
                    "exact_support_state_fraction": float(
                        profile_for_metrics["support_backoff_level"].eq(0).mean()
                    ),
                    "guardrail_empty_state_fraction": float(empty.mean()),
                    "changed_from_v1_state_count": int(
                        (selected != fixed_selected).sum()
                    ),
                    "coverage_shortfall_epsilon": float(
                        current_thresholds["coverage_shortfall_epsilon"]
                    ),
                    "tuwr_upper": float(current_thresholds["tuwr_upper"]),
                    "ard_upper": float(current_thresholds["ard_upper"]),
                    "heldout_zone_used_in_fit": False,
                }
                for action_index, action in enumerate(ACTIONS):
                    row[f"selected_{action}_state_count"] = int(
                        action_counts[action_index]
                    )
                for n_min in config["adaptive_support"]["n_min_candidates"]:
                    row[f"selected_n_min_{int(n_min)}_state_count"] = int(
                        (selected_n_min == int(n_min)).sum()
                    )
                for nu in config["adaptive_support"]["nu_candidates"]:
                    row[f"selected_nu_{int(nu)}_state_count"] = int(
                        (selected_nu == int(nu)).sum()
                    )
                metric_rows.append(row)
                decision_frames.append(
                    build_state_decisions(
                        outer_zone=outer_zone,
                        seed=seed,
                        price_id=price_id,
                        price_ratio=price_ratio,
                        version=str(version),
                        evidence=evidence,
                        selected=selected,
                        empty=empty,
                        safe=safe,
                        selected_n_min=selected_n_min,
                        selected_nu=selected_nu,
                        selected_margin=selected_margin,
                        selected_nu_ratio=selected_nu_ratio,
                        thresholds=current_thresholds,
                        fixed_selected=fixed_selected,
                    )
                )
            atomic_json(
                checkpoint_path,
                {
                    "stage": "PRICE_COMPLETE",
                    "outer_heldout_zone": outer_zone,
                    "seed": int(seed),
                    "price_id": price_id,
                    "completed_price_count": price_index + 1,
                    "config_sha256": config_sha256,
                    "timestamp_utc": utc_now(),
                },
            )

        metrics_frame = pd.DataFrame(metric_rows).sort_values(
            ["miss_to_capacity_ratio", "version"], kind="mergesort"
        )
        decisions_frame = pd.concat(decision_frames, ignore_index=True).sort_values(
            [
                "miss_to_capacity_ratio",
                "version",
                "full_state_code",
            ],
            kind="mergesort",
        )
        expected_metrics = len(config["scope"]["price_ids"]) * len(
            config["scope"]["versions"]
        )
        expected_decisions = expected_metrics * len(states)
        if len(metrics_frame) != expected_metrics:
            raise RuntimeError("单元指标行数失配")
        if len(decisions_frame) != expected_decisions:
            raise RuntimeError("单元状态决策行数失配")
        metrics_path = root / "heldout_metrics.csv"
        decisions_path = root / "state_decisions.parquet"
        metrics_frame.to_csv(
            metrics_path, index=False, encoding="utf-8", lineterminator="\n"
        )
        decisions_frame.to_parquet(
            decisions_path, index=False, compression="zstd"
        )
        append_jsonl(
            exception_path,
            {
                "record_type": "EXCEPTION_SUMMARY",
                "exception_count": 0,
                "timestamp_utc": utc_now(),
            },
        )
        manifest = {
            "schema": "TEST_CLARA_FOUR_VERSION_UNIT_V1",
            "status": "PASS",
            "generated_at_utc": utc_now(),
            "config_sha256": config_sha256,
            "outer_heldout_zone": outer_zone,
            "seed": int(seed),
            "fit_zones": fit_zones,
            "heldout_zone_used_in_fit": False,
            "price_count": len(price_grid),
            "version_count": len(config["scope"]["versions"]),
            "metric_row_count": len(metrics_frame),
            "state_decision_row_count": len(decisions_frame),
            "fixed_action_regression_mismatch_count": int(
                fixed_regression_mismatch_count
            ),
            "heldout_metrics_sha256": sha256_file(metrics_path),
            "state_decisions_sha256": sha256_file(decisions_path),
            "component_audit_sha256": sha256_file(source_audit_path),
            "elapsed_seconds": time.perf_counter() - started,
            "resume_status": "BUILT",
        }
        atomic_json(root / "unit_manifest.json", manifest)
        atomic_json(
            checkpoint_path,
            {
                "stage": "COMPLETE",
                "outer_heldout_zone": outer_zone,
                "seed": int(seed),
                "config_sha256": config_sha256,
                "unit_manifest_sha256": sha256_file(root / "unit_manifest.json"),
                "timestamp_utc": utc_now(),
            },
        )
        return manifest
    except Exception as error:
        append_jsonl(
            exception_path,
            {
                "record_type": "EXCEPTION",
                "timestamp_utc": utc_now(),
                "outer_heldout_zone": outer_zone,
                "seed": int(seed),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        atomic_json(
            checkpoint_path,
            {
                "stage": "FAILED",
                "outer_heldout_zone": outer_zone,
                "seed": int(seed),
                "config_sha256": config_sha256,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "timestamp_utc": utc_now(),
            },
        )
        raise


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.astype(float).gt(0)
    if not valid.any():
        return math.nan
    return float(
        np.average(
            values.loc[valid].to_numpy(dtype=np.float64),
            weights=weights.loc[valid].to_numpy(dtype=np.float64),
        )
    )


def build_aggregate_metrics(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    zone_rows: list[dict[str, Any]] = []
    zone_keys = [
        "price_id",
        "miss_to_capacity_ratio",
        "version",
        "outer_heldout_zone",
    ]
    for keys, part in metrics.groupby(zone_keys, sort=True, dropna=False):
        row = {name: value for name, value in zip(zone_keys, keys)}
        row["event_count"] = int(part["event_count"].sum())
        row["guardrail_event_count"] = int(part["guardrail_event_count"].sum())
        for metric in (
            "mean_errf",
            "coverage",
            "target_coverage_weighted",
            "coverage_gap",
        ):
            row[metric] = weighted_average(part[metric], part["event_count"])
        for metric in ("tuwr", "ard"):
            row[metric] = weighted_average(
                part[metric], part["guardrail_event_count"]
            )
        for metric in (
            "mean_selected_n_min",
            "mean_selected_nu",
            "mean_support_backoff_level",
            "exact_support_state_fraction",
            "guardrail_empty_state_fraction",
        ):
            row[metric] = float(part[metric].mean())
        row["coverage_shortfall_epsilon"] = float(
            part["coverage_shortfall_epsilon"].iloc[0]
        )
        row["tuwr_upper"] = float(part["tuwr_upper"].iloc[0])
        row["ard_upper"] = float(part["ard_upper"].iloc[0])
        zone_rows.append(row)
    zone_frame = pd.DataFrame(zone_rows)

    overall_rows: list[dict[str, Any]] = []
    overall_keys = ["price_id", "miss_to_capacity_ratio", "version"]
    for keys, part in zone_frame.groupby(overall_keys, sort=True, dropna=False):
        row = {name: value for name, value in zip(overall_keys, keys)}
        row["zone_count"] = int(part["outer_heldout_zone"].nunique())
        row["event_count_repeated_across_seeds"] = int(part["event_count"].sum())
        row["guardrail_event_count_repeated_across_seeds"] = int(
            part["guardrail_event_count"].sum()
        )
        for metric in (
            "mean_errf",
            "coverage",
            "target_coverage_weighted",
            "coverage_gap",
            "tuwr",
            "ard",
            "mean_selected_n_min",
            "mean_selected_nu",
            "mean_support_backoff_level",
            "exact_support_state_fraction",
            "guardrail_empty_state_fraction",
        ):
            row[metric] = float(part[metric].mean())
            row[f"{metric}_zone_se"] = (
                float(part[metric].std(ddof=1) / math.sqrt(len(part)))
                if len(part) > 1
                else math.nan
            )
        row["coverage_shortfall_epsilon"] = float(
            part["coverage_shortfall_epsilon"].iloc[0]
        )
        row["tuwr_upper"] = float(part["tuwr_upper"].iloc[0])
        row["ard_upper"] = float(part["ard_upper"].iloc[0])
        overall_rows.append(row)
    overall = pd.DataFrame(overall_rows)
    fixed = overall[overall["version"].astype(str).eq("V1_FIXED")][
        [
            "price_id",
            "mean_errf",
            "coverage_gap",
            "tuwr",
            "ard",
        ]
    ].rename(
        columns={
            "mean_errf": "v1_mean_errf",
            "coverage_gap": "v1_coverage_gap",
            "tuwr": "v1_tuwr",
            "ard": "v1_ard",
        }
    )
    comparison = overall.merge(fixed, on="price_id", how="left", validate="many_to_one")
    comparison["errf_difference_vs_v1"] = (
        comparison["mean_errf"] - comparison["v1_mean_errf"]
    )
    comparison["relative_errf_improvement_pct_vs_v1"] = 100.0 * (
        comparison["v1_mean_errf"] - comparison["mean_errf"]
    ) / comparison["v1_mean_errf"]
    comparison["coverage_gap_difference_vs_v1"] = (
        comparison["coverage_gap"] - comparison["v1_coverage_gap"]
    )
    comparison["tuwr_difference_vs_v1"] = (
        comparison["tuwr"] - comparison["v1_tuwr"]
    )
    comparison["ard_difference_vs_v1"] = (
        comparison["ard"] - comparison["v1_ard"]
    )
    return zone_frame, comparison.sort_values(
        ["miss_to_capacity_ratio", "version"], kind="mergesort"
    ).reset_index(drop=True)


def baseline_metric_regression(
    comparison: pd.DataFrame, tolerance: float
) -> dict[str, Any]:
    reference = pd.read_csv(SEALED_PRICE_SUMMARY, encoding="utf-8")
    observed = comparison[comparison["version"].astype(str).eq("V1_FIXED")].copy()
    merged = observed.merge(reference, on="price_id", how="inner", validate="one_to_one")
    if len(merged) != 5:
        raise RuntimeError("固定版本封存指标回归价格数量失配")
    fields = {
        "mean_errf": "clara_mean_errf",
        "coverage": "clara_coverage",
        "coverage_gap": "clara_coverage_gap",
        "tuwr": "clara_TUWR",
        "ard": "clara_ARD",
    }
    errors: dict[str, float] = {}
    for current, sealed in fields.items():
        errors[current] = float(
            np.max(
                np.abs(
                    merged[current].to_numpy(dtype=np.float64)
                    - merged[sealed].to_numpy(dtype=np.float64)
                )
            )
        )
    maximum = max(errors.values())
    if maximum > float(tolerance):
        raise RuntimeError(
            f"固定版本指标复算失败: max_error={maximum}, errors={errors}"
        )
    return {
        "status": "PASS",
        "maximum_absolute_error": maximum,
        "tolerance": float(tolerance),
        "field_errors": errors,
    }


def summarize_counts(
    metrics: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parameter_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    keys = ["price_id", "miss_to_capacity_ratio", "version"]
    for group_keys, part in metrics.groupby(keys, sort=True, dropna=False):
        identity = {name: value for name, value in zip(keys, group_keys)}
        state_total = int(part["state_count"].sum())
        for n_min in config["adaptive_support"]["n_min_candidates"]:
            count = int(part[f"selected_n_min_{int(n_min)}_state_count"].sum())
            parameter_rows.append(
                {
                    **identity,
                    "parameter": "n_min",
                    "value": int(n_min),
                    "state_count": count,
                    "state_fraction": float(count / state_total),
                }
            )
        for nu in config["adaptive_support"]["nu_candidates"]:
            count = int(part[f"selected_nu_{int(nu)}_state_count"].sum())
            parameter_rows.append(
                {
                    **identity,
                    "parameter": "nu",
                    "value": int(nu),
                    "state_count": count,
                    "state_fraction": float(count / state_total),
                }
            )
        for action in ACTIONS:
            count = int(part[f"selected_{action}_state_count"].sum())
            action_rows.append(
                {
                    **identity,
                    "action": action,
                    "state_count": count,
                    "state_fraction": float(count / state_total),
                }
            )
    return pd.DataFrame(parameter_rows), pd.DataFrame(action_rows)


def finalize_full_run(
    *,
    config: dict[str, Any],
    config_sha256: str,
    input_audit: dict[str, Any],
    free_disk_before: float,
) -> dict[str, Any]:
    manifests: list[dict[str, Any]] = []
    metric_frames: list[pd.DataFrame] = []
    state_decision_rows = 0
    for outer_zone in config["scope"]["zones"]:
        for seed in config["scope"]["seeds"]:
            manifest = completed_unit(str(outer_zone), int(seed), config_sha256)
            if manifest is None:
                raise RuntimeError(f"全量封存缺少单元: {outer_zone}/seed{seed}")
            manifests.append(manifest)
            root = unit_root(str(outer_zone), int(seed))
            metric_frames.append(
                pd.read_csv(root / "heldout_metrics.csv", encoding="utf-8")
            )
            state_decision_rows += int(manifest["state_decision_row_count"])
    metrics = pd.concat(metric_frames, ignore_index=True)
    if len(metrics) != int(config["scope"]["expected_metric_row_count"]):
        raise RuntimeError("全量指标行数失配")
    if state_decision_rows != int(
        config["scope"]["expected_state_decision_row_count"]
    ):
        raise RuntimeError("全量状态决策行数失配")
    zone_frame, comparison = build_aggregate_metrics(metrics)
    baseline_regression = baseline_metric_regression(
        comparison, float(config["execution"]["numeric_tolerance"])
    )
    parameter_summary, action_summary = summarize_counts(metrics, config)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_ROOT / "all_heldout_metrics.csv"
    zone_path = OUTPUT_ROOT / "zone_balanced_rows.csv"
    comparison_path = OUTPUT_ROOT / "four_version_comparison.csv"
    parameter_path = OUTPUT_ROOT / "parameter_selection_summary.csv"
    action_path = OUTPUT_ROOT / "action_selection_summary.csv"
    baseline_path = OUTPUT_ROOT / "fixed_baseline_regression.json"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8", lineterminator="\n")
    zone_frame.to_csv(zone_path, index=False, encoding="utf-8", lineterminator="\n")
    comparison.to_csv(
        comparison_path, index=False, encoding="utf-8", lineterminator="\n"
    )
    parameter_summary.to_csv(
        parameter_path, index=False, encoding="utf-8", lineterminator="\n"
    )
    action_summary.to_csv(
        action_path, index=False, encoding="utf-8", lineterminator="\n"
    )
    atomic_json(baseline_path, baseline_regression)
    v4 = comparison[comparison["version"].astype(str).eq("V4_FULLY_ADAPTIVE")]
    informative_gate = {
        "prices_with_lower_errf_than_v1": int(
            v4["errf_difference_vs_v1"].lt(0.0).sum()
        ),
        "prices_with_higher_coverage_gap_than_v1": int(
            v4["coverage_gap_difference_vs_v1"].gt(0.0).sum()
        ),
        "prices_with_lower_tuwr_than_v1": int(
            v4["tuwr_difference_vs_v1"].lt(0.0).sum()
        ),
        "prices_with_lower_ard_than_v1": int(
            v4["ard_difference_vs_v1"].lt(0.0).sum()
        ),
        "scientific_review_required": bool(
            v4["errf_difference_vs_v1"].ge(0.0).all()
        ),
    }
    root_manifest = {
        "schema": "TEST_CLARA_FOUR_VERSION_ROOT_V1",
        "status": "PASS",
        "generated_at_utc": utc_now(),
        "config_sha256": config_sha256,
        "runner_sha256": sha256_file(Path(__file__)),
        "input_audit": input_audit,
        "unit_count": len(manifests),
        "metric_row_count": len(metrics),
        "state_decision_row_count": state_decision_rows,
        "fixed_action_regression_mismatch_count": int(
            sum(int(row["fixed_action_regression_mismatch_count"]) for row in manifests)
        ),
        "fixed_baseline_metric_regression": baseline_regression,
        "informative_scientific_gate": informative_gate,
        "artifacts": {
            "all_heldout_metrics": {
                "relative_path": metrics_path.relative_to(OUTPUT_ROOT).as_posix(),
                "sha256": sha256_file(metrics_path),
                "rows": len(metrics),
            },
            "zone_balanced_rows": {
                "relative_path": zone_path.relative_to(OUTPUT_ROOT).as_posix(),
                "sha256": sha256_file(zone_path),
                "rows": len(zone_frame),
            },
            "four_version_comparison": {
                "relative_path": comparison_path.relative_to(OUTPUT_ROOT).as_posix(),
                "sha256": sha256_file(comparison_path),
                "rows": len(comparison),
            },
            "parameter_selection_summary": {
                "relative_path": parameter_path.relative_to(OUTPUT_ROOT).as_posix(),
                "sha256": sha256_file(parameter_path),
                "rows": len(parameter_summary),
            },
            "action_selection_summary": {
                "relative_path": action_path.relative_to(OUTPUT_ROOT).as_posix(),
                "sha256": sha256_file(action_path),
                "rows": len(action_summary),
            },
            "fixed_baseline_regression": {
                "relative_path": baseline_path.relative_to(OUTPUT_ROOT).as_posix(),
                "sha256": sha256_file(baseline_path),
            },
        },
        "unit_manifests": [
            {
                "outer_heldout_zone": row["outer_heldout_zone"],
                "seed": row["seed"],
                "unit_manifest_sha256": sha256_file(
                    unit_root(str(row["outer_heldout_zone"]), int(row["seed"]))
                    / "unit_manifest.json"
                ),
            }
            for row in manifests
        ],
        "free_disk_gib_before": float(free_disk_before),
        "free_disk_gib_after": float(
            shutil.disk_usage(TEST_ROOT).free / (1024.0**3)
        ),
    }
    atomic_json(OUTPUT_ROOT / "root_manifest.json", root_manifest)
    return root_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行四个 CLARA 版本的同协议比较")
    parser.add_argument("--outer-zones", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--skip-finalize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(CONFIG_PATH)
    config_sha256 = sha256_file(CONFIG_PATH)
    price_config = load_json(PRICE_CONFIG_PATH)
    component_config_sha256 = sha256_file(PRICE_CONFIG_PATH)
    input_audit = validate_inputs(config)
    free_disk_before = ensure_disk_gate(config)
    selected_zones = (
        [str(value) for value in args.outer_zones]
        if args.outer_zones
        else [str(value) for value in config["scope"]["zones"]]
    )
    selected_seeds = (
        [int(value) for value in args.seeds]
        if args.seeds is not None and len(args.seeds)
        else [int(value) for value in config["scope"]["seeds"]]
    )
    unknown_zones = sorted(set(selected_zones) - set(config["scope"]["zones"]))
    unknown_seeds = sorted(set(selected_seeds) - set(config["scope"]["seeds"]))
    if unknown_zones or unknown_seeds:
        raise RuntimeError(
            f"请求范围超出冻结协议: zones={unknown_zones}, seeds={unknown_seeds}"
        )
    tasks = [
        {
            "outer_zone": zone,
            "seed": seed,
            "config": config,
            "config_sha256": config_sha256,
            "price_config": price_config,
            "component_config_sha256": component_config_sha256,
        }
        for zone in selected_zones
        for seed in selected_seeds
    ]
    workers = int(
        args.workers
        if args.workers is not None
        else config["execution"]["maximum_workers"]
    )
    workers = max(1, min(workers, len(tasks)))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, Any]] = []
    if workers == 1:
        for task in tasks:
            manifest = run_unit(task)
            manifests.append(manifest)
            print(
                f"完成 {manifest['outer_heldout_zone']}/seed{manifest['seed']} "
                f"状态决策 {manifest['state_decision_row_count']} 行",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_unit, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    manifest = future.result()
                except Exception:
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(
                        f"单元运行失败: {task['outer_zone']}/seed{task['seed']}"
                    ) from future.exception()
                manifests.append(manifest)
                print(
                    f"完成 {manifest['outer_heldout_zone']}/seed{manifest['seed']} "
                    f"状态决策 {manifest['state_decision_row_count']} 行",
                    flush=True,
                )
    complete_scope = (
        set(selected_zones) == set(config["scope"]["zones"])
        and set(selected_seeds) == set(config["scope"]["seeds"])
    )
    if complete_scope and not args.skip_finalize:
        root_manifest = finalize_full_run(
            config=config,
            config_sha256=config_sha256,
            input_audit=input_audit,
            free_disk_before=free_disk_before,
        )
        print(
            json.dumps(
                {
                    "status": root_manifest["status"],
                    "unit_count": root_manifest["unit_count"],
                    "metric_row_count": root_manifest["metric_row_count"],
                    "state_decision_row_count": root_manifest[
                        "state_decision_row_count"
                    ],
                    "informative_scientific_gate": root_manifest[
                        "informative_scientific_gate"
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    else:
        print(
            json.dumps(
                {
                    "status": "PARTIAL_SCOPE_COMPLETE",
                    "unit_count": len(manifests),
                    "finalized": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
