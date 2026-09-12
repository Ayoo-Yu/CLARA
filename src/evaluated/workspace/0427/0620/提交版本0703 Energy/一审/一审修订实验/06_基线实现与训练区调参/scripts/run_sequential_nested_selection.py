from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pandas as pd
import psutil


S06_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = S06_ROOT.parent
AUTH_ROOT = REVISION_ROOT / "_权威代码"
CODE_ROOT = AUTH_ROOT / "code"
RUN_ROOT = S06_ROOT / "results_raw" / "nested_source_selection_v1"
PRECOMPUTE_ROOT = RUN_ROOT / "sequential_precompute_v1"
TIMELINE_ROOT = PRECOMPUTE_ROOT / "zone_timelines"
SELECTION_ROOT = RUN_ROOT / "nested_selection" / "sequential_v1"
LOG_ROOT = RUN_ROOT / "logs"
sys.path.insert(0, str(CODE_ROOT))

from baseline_common import FrozenStateEncoder, build_nested_source_zone_folds  # noqa: E402
from baseline_selection import select_configuration  # noqa: E402
from baseline_sequential_compact import (  # noqa: E402
    evaluate_delayed_hedge_grid,
    evaluate_linucb_grid,
)
from baseline_training import validate_frozen_selector_config  # noqa: E402
from clara_event_contract import (  # noqa: E402
    canonical_json_sha256,
    load_frozen_contracts,
    sha256_file,
)


EXPECTED_CODE_COMMIT = "82ebb8af13c1ae6162b9804d1660665f87269401"
SOURCE_TUNING_CONTRACT_SHA256 = "9a352cba41d7e68911626da950540d1d8506f2f4f63afb72bd7b28cd473c7e14"
SEQUENTIAL_CONTRACT_SHA256 = "36c33a015ff5f10fc8325624a3c5b9d0490772a61218e45e4ab915f67c28c8af"
WIDTH_THRESHOLDS_SHA256 = "ef3ebd2ed7dc4105212ea27fa0d62729c5101416eff8aba2a37c166ceced96ba"
LOSS_SCALES_SHA256 = "343247aafb7d342fa6315801c347d28e45874eb001e9f9586a17b4f13db9bf0c"
PRECOMPUTE_QA_SHA256 = "8374bded385d71a09aab5a780ecdd059717c05e29c3b96c4e486b8bd46892e9d"
COMPACT_EQUIVALENCE_QA_SHA256 = "4ccb7cab6dd2e3e51deebe310c1854a7b6e7ffc99b4632785dc187cc556f31fb"
COMPACT_ENGINE_SHA256 = "c385ac09a8d535be98017fc2d065f52e4709bbe96ea80d1d57cc10a86c300488"
EXPECTED_EVENT_COUNT = 11_070_576
EXPECTED_ISSUE_COUNT = 3506
EXPECTED_HEDGE_CONFIG_COUNT = 5
EXPECTED_LINUCB_CONFIG_COUNT = 12
EXPECTED_INNER_SCORE_COUNT = 17
EXPECTED_INNER_FOLD_COUNT = 90
EXPECTED_ROOT_SCORE_COUNT = EXPECTED_INNER_SCORE_COUNT * EXPECTED_INNER_FOLD_COUNT
EXPECTED_ROOT_SELECTION_COUNT = 20
NUMBA_THREAD_COUNT = 12
MINIMUM_FREE_DISK_GIB = 60.0


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=AUTH_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def zone_number(value: str) -> int:
    text = str(value)
    if text.startswith("zone") and text[4:].isdigit():
        return int(text[4:])
    raise ValueError(f"区域编号无法解析: {text}")


def feature_positions(encoder: FrozenStateEncoder) -> dict[str, dict[Any, int]]:
    return {
        field: {
            value: encoder.feature_names.index(
                f"state__{field}__{encoder._format_value(value)}"
            )
            for value in encoder.vocabularies[field]
        }
        for field in encoder.fields
    }


def build_context_lookup(
    encoder: FrozenStateEncoder,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = feature_positions(encoder)
    fields = ("predictor", "horizon_group", "target_coverage", "ramp_state", "rolling_state")
    sizes = np.asarray([len(encoder.vocabularies[field]) for field in fields], dtype=np.int64)
    offsets = np.asarray([min(positions[field].values()) for field in fields], dtype=np.uint8)
    width_vocabulary = tuple(encoder.vocabularies["raw_width_state"])
    width_positions = np.asarray(
        [positions["raw_width_state"][value] for value in width_vocabulary],
        dtype=np.uint8,
    )
    context_count = int(np.prod(sizes) * len(width_vocabulary))
    active = np.empty((context_count, 7), dtype=np.uint8)
    for context in range(context_count):
        remainder = context
        width_code = remainder % len(width_vocabulary)
        remainder //= len(width_vocabulary)
        decoded = np.empty(len(fields), dtype=np.int64)
        for column in range(len(fields) - 1, -1, -1):
            decoded[column] = remainder % sizes[column]
            remainder //= sizes[column]
        active[context, 0] = 0
        for column in range(len(fields)):
            active[context, column + 1] = offsets[column] + decoded[column]
        active[context, 6] = width_positions[width_code]
    return active, offsets, sizes, width_positions


def base_context_and_coverage(
    base_feature_indices: np.ndarray,
    offsets: np.ndarray,
    sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    codes = base_feature_indices.astype(np.int16) - offsets.astype(np.int16)
    if np.any(codes < 0):
        raise RuntimeError("时间轴基础特征位置低于冻结字段偏移")
    base_context = np.zeros(len(base_feature_indices), dtype=np.uint16)
    for column, size in enumerate(sizes):
        if np.any(codes[:, column] >= int(size)):
            raise RuntimeError("时间轴基础特征位置超出冻结字段词表")
        base_context = (
            base_context.astype(np.uint32) * int(size) + codes[:, column]
        ).astype(np.uint16)
    return base_context, codes[:, 2].astype(np.uint8)


def threshold_vectors(
    *,
    thresholds: pd.DataFrame,
    outer_zone: str,
    encoder: FrozenStateEncoder,
) -> tuple[np.ndarray, np.ndarray]:
    frame = thresholds[thresholds["outer_heldout_zone"].astype(str).eq(outer_zone)]
    predictor_map = {value: index for index, value in enumerate(encoder.vocabularies["predictor"])}
    horizon_map = {value: index for index, value in enumerate(encoder.vocabularies["horizon_group"])}
    coverage_map = {
        float(value): index
        for index, value in enumerate(encoder.vocabularies["target_coverage"])
    }
    predictor_count = len(predictor_map)
    horizon_count = len(horizon_map)
    coverage_count = len(coverage_map)
    group_count = 3 * predictor_count * horizon_count * coverage_count
    q33 = np.full(group_count, np.nan, dtype=np.float64)
    q67 = np.full(group_count, np.nan, dtype=np.float64)
    for row in frame.itertuples(index=False):
        group = (
            ((int(row.seed) * predictor_count + predictor_map[row.predictor]) * horizon_count
            + horizon_map[row.horizon_group])
            * coverage_count
            + coverage_map[float(row.target_coverage)]
        )
        q33[group] = float(row.raw_width_q33)
        q67[group] = float(row.raw_width_q67)
    if not np.isfinite(q33).all() or not np.isfinite(q67).all() or np.any(q33 > q67):
        raise RuntimeError(f"{outer_zone}宽度阈值向量不完整或反转")
    if len(frame) != group_count or frame["heldout_zone_in_threshold"].astype(bool).any():
        raise RuntimeError(f"{outer_zone}宽度阈值行数或持出区门失配")
    return q33, q67


def decision_digest(decisions: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(decisions).tobytes(order="C")).hexdigest()


def grid_contracts(contracts: Any) -> tuple[np.ndarray, list[tuple[float, float]]]:
    entries = {
        str(entry["id"]): entry
        for entry in contracts.baseline_registry["mandatory_contextual_selectors"]
    }
    eta_values = np.asarray(entries["DelayedHedge"]["grid"]["eta"], dtype=np.float64)
    alpha_values = entries["LinUCB"]["grid"]["exploration_alpha"]
    l2_values = entries["LinUCB"]["grid"]["l2_regularization"]
    linucb_grid = [
        (float(alpha), float(l2))
        for alpha, l2 in itertools.product(alpha_values, l2_values)
    ]
    if len(eta_values) != EXPECTED_HEDGE_CONFIG_COUNT or len(linucb_grid) != EXPECTED_LINUCB_CONFIG_COUNT:
        raise RuntimeError("顺序基线冻结网格规模失配")
    for eta in eta_values:
        validate_frozen_selector_config(
            contracts=contracts,
            baseline_id="DelayedHedge",
            config={"eta": float(eta)},
        )
    for alpha, l2 in linucb_grid:
        validate_frozen_selector_config(
            contracts=contracts,
            baseline_id="LinUCB",
            config={"exploration_alpha": alpha, "l2_regularization": l2},
        )
    return eta_values, linucb_grid


def scale_by_excluded_pair(scales: pd.DataFrame) -> dict[frozenset[str], dict[str, Any]]:
    mapping: dict[frozenset[str], dict[str, Any]] = {}
    for row in scales.to_dict(orient="records"):
        key = frozenset((str(row["excluded_zone_a"]), str(row["excluded_zone_b"])))
        if key in mapping or len(key) != 2:
            raise RuntimeError("DelayedHedge留二区尺度键重复或无效")
        mapping[key] = row
    if len(mapping) != 45:
        raise RuntimeError("DelayedHedge留二区尺度数量失配")
    return mapping


def checkpoint_paths(outer_zone: str, validation_zone: str) -> tuple[Path, Path]:
    root = (
        SELECTION_ROOT
        / f"outer_heldout={outer_zone}"
        / f"inner_validation={validation_zone}"
    )
    return root / "validation_scores.parquet", root / "manifest.json"


def verified_checkpoint(
    *,
    outer_zone: str,
    validation_zone: str,
    training_zones: tuple[str, ...],
    timeline_manifest_sha256: str,
) -> pd.DataFrame | None:
    score_path, manifest_path = checkpoint_paths(outer_zone, validation_zone)
    if not score_path.exists() and not manifest_path.exists():
        return None
    if not score_path.exists() or not manifest_path.exists():
        raise RuntimeError(f"顺序内层检查点不完整: {outer_zone}, {validation_zone}")
    manifest = load_json(manifest_path)
    expected = {
        "checkpoint_schema": "S06_SEQUENTIAL_INNER_SELECTION_V1",
        "status": "PASS",
        "code_commit": EXPECTED_CODE_COMMIT,
        "source_tuning_contract_sha256": SOURCE_TUNING_CONTRACT_SHA256,
        "sequential_contract_sha256": SEQUENTIAL_CONTRACT_SHA256,
        "width_thresholds_sha256": WIDTH_THRESHOLDS_SHA256,
        "loss_scales_sha256": LOSS_SCALES_SHA256,
        "precompute_qa_sha256": PRECOMPUTE_QA_SHA256,
        "compact_equivalence_qa_sha256": COMPACT_EQUIVALENCE_QA_SHA256,
        "compact_engine_sha256": COMPACT_ENGINE_SHA256,
        "timeline_manifest_sha256": timeline_manifest_sha256,
        "outer_heldout_zone": outer_zone,
        "inner_validation_zone": validation_zone,
        "training_zones": list(training_zones),
        "score_row_count": EXPECTED_INNER_SCORE_COUNT,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"顺序内层检查点清单身份失配: {outer_zone}, {validation_zone}")
    if sha256_file(score_path) != manifest.get("score_file_sha256"):
        raise RuntimeError(f"顺序内层检查点分数哈希失配: {outer_zone}, {validation_zone}")
    frame = pd.read_parquet(score_path)
    counts = frame.groupby("baseline_id")["configuration_id"].nunique().to_dict()
    if len(frame) != EXPECTED_INNER_SCORE_COUNT or counts != {
        "DelayedHedge": EXPECTED_HEDGE_CONFIG_COUNT,
        "LinUCB": EXPECTED_LINUCB_CONFIG_COUNT,
    }:
        raise RuntimeError(f"顺序内层检查点行数或配置数失配: {outer_zone}, {validation_zone}")
    return frame


def write_checkpoint(
    *,
    scores: pd.DataFrame,
    outer_zone: str,
    validation_zone: str,
    training_zones: tuple[str, ...],
    timeline_manifest_sha256: str,
    loss_scale: float,
    elapsed_seconds: float,
    hedge_seconds: float,
    linucb_seconds: float,
    repeatability_checked: bool,
) -> None:
    score_path, manifest_path = checkpoint_paths(outer_zone, validation_zone)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_score = score_path.with_name("validation_scores.tmp.parquet")
    scores.to_parquet(temporary_score, index=False, compression="zstd")
    os.replace(temporary_score, score_path)
    manifest = {
        "checkpoint_schema": "S06_SEQUENTIAL_INNER_SELECTION_V1",
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": EXPECTED_CODE_COMMIT,
        "source_tuning_contract_sha256": SOURCE_TUNING_CONTRACT_SHA256,
        "sequential_contract_sha256": SEQUENTIAL_CONTRACT_SHA256,
        "width_thresholds_sha256": WIDTH_THRESHOLDS_SHA256,
        "loss_scales_sha256": LOSS_SCALES_SHA256,
        "precompute_qa_sha256": PRECOMPUTE_QA_SHA256,
        "compact_equivalence_qa_sha256": COMPACT_EQUIVALENCE_QA_SHA256,
        "compact_engine_sha256": COMPACT_ENGINE_SHA256,
        "timeline_manifest_sha256": timeline_manifest_sha256,
        "outer_heldout_zone": outer_zone,
        "inner_validation_zone": validation_zone,
        "training_zones": list(training_zones),
        "delayed_hedge_loss_scale": loss_scale,
        "score_row_count": len(scores),
        "score_file_sha256": sha256_file(score_path),
        "hedge_decision_digest_sha256": hashlib.sha256(
            "".join(scores[scores["baseline_id"].eq("DelayedHedge")]["decision_digest_sha256"]).encode("ascii")
        ).hexdigest(),
        "linucb_decision_digest_sha256": hashlib.sha256(
            "".join(scores[scores["baseline_id"].eq("LinUCB")]["decision_digest_sha256"]).encode("ascii")
        ).hexdigest(),
        "event_count": int(scores["event_count"].iloc[0]),
        "issue_batch_count": EXPECTED_ISSUE_COUNT,
        "elapsed_seconds": elapsed_seconds,
        "hedge_seconds": hedge_seconds,
        "linucb_seconds": linucb_seconds,
        "rss_gib": psutil.Process().memory_info().rss / (1024.0**3),
        "numba_thread_count": numba.get_num_threads(),
        "repeatability_checked": repeatability_checked,
        "future_information_violation_count": 0,
        "within_issue_feedback_use_count": 0,
        "outer_heldout_zone_timeline_read_count": 0,
        "outer_heldout_zone_fit_count": 0,
        "outer_heldout_zone_selection_count": 0,
        "configuration_selection_executed": False,
        "performance_comparison_executed": False,
        "method_ranking_executed": False,
        "statistical_inference_executed": False,
        "paper_performance_claim_executed": False,
    }
    temporary_manifest = manifest_path.with_suffix(".tmp.json")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)


def evaluate_pair(
    *,
    contracts: Any,
    encoder: FrozenStateEncoder,
    outer_zone: str,
    validation_zone: str,
    training_zones: tuple[str, ...],
    base_context: np.ndarray,
    coverage_code: np.ndarray,
    context_active: np.ndarray,
    timeline: dict[str, np.ndarray],
    q33: np.ndarray,
    q67: np.ndarray,
    eta_values: np.ndarray,
    linucb_grid: list[tuple[float, float]],
    loss_scale: float,
) -> tuple[pd.DataFrame, float, float]:
    state_group = timeline["state_group"]
    raw_width = timeline["raw_width"]
    width_code = np.where(
        raw_width <= q33[state_group],
        0,
        np.where(raw_width <= q67[state_group], 1, 2),
    ).astype(np.uint8)
    context_ids = (base_context.astype(np.uint32) * 3 + width_code).astype(np.uint16)
    coverage_levels = np.asarray(encoder.vocabularies["target_coverage"], dtype=np.float64)
    coverage_by_state = np.asarray(
        [coverage_levels[index % len(coverage_levels)] for index in range(660)],
        dtype=np.float64,
    )
    hedge_started = time.perf_counter()
    hedge = evaluate_delayed_hedge_grid(
        state_group,
        timeline["issue_offsets"],
        timeline["feedback_order"],
        timeline["matured_ends"],
        timeline["losses"],
        timeline["covered"],
        timeline["tuwr"],
        coverage_by_state,
        eta_values,
        np.full(len(eta_values), loss_scale, dtype=np.float64),
        -1,
    )
    hedge_seconds = time.perf_counter() - hedge_started
    alpha_values = np.asarray([pair[0] for pair in linucb_grid], dtype=np.float64)
    l2_values = np.asarray([pair[1] for pair in linucb_grid], dtype=np.float64)
    linucb_started = time.perf_counter()
    linucb = evaluate_linucb_grid(
        context_ids,
        context_active,
        timeline["seed"],
        coverage_code,
        coverage_levels,
        timeline["issue_offsets"],
        timeline["feedback_order"],
        timeline["matured_ends"],
        timeline["losses"],
        timeline["covered"],
        timeline["tuwr"],
        alpha_values,
        l2_values,
        -1,
    )
    linucb_seconds = time.perf_counter() - linucb_started
    event_count = int(hedge[-1][0])
    if event_count != EXPECTED_EVENT_COUNT or int(linucb[-1][0]) != EXPECTED_EVENT_COUNT:
        raise RuntimeError("顺序网格事件数失配")
    matured_expected = int(timeline["matured_ends"][-1])
    if not np.all(hedge[6] == matured_expected):
        raise RuntimeError("DelayedHedge末次issue前成熟反馈计数失配")
    if not np.all(linucb[6].sum(axis=1) == matured_expected):
        raise RuntimeError("LinUCB末次issue前成熟反馈计数失配")
    if not np.all(hedge[5].sum(axis=1) == event_count) or not np.all(
        linucb[5].sum(axis=1) == event_count
    ):
        raise RuntimeError("顺序网格动作计数未闭合")
    rows: list[dict[str, Any]] = []
    for complexity_rank, eta in enumerate(eta_values):
        config = {"eta": float(eta)}
        selection_config_id = canonical_json_sha256({"baseline_id": "DelayedHedge", **config})
        policy_config_id = canonical_json_sha256(
            {
                "baseline_id": "DelayedHedge",
                **config,
                "loss_scale": float(loss_scale),
                "state_scope": ["theta_id", "seed", "predictor", "horizon_group", "target_coverage"],
            }
        )
        tuwr_count = int(hedge[4][complexity_rank])
        rows.append(
            {
                "outer_heldout_zone": outer_zone,
                "inner_validation_zone": validation_zone,
                "baseline_id": "DelayedHedge",
                "configuration_id": selection_config_id,
                "policy_run_configuration_id": policy_config_id,
                "configuration_json": json.dumps(config, ensure_ascii=False, sort_keys=True),
                "fitted_parameter_json": json.dumps({"loss_scale": float(loss_scale)}, sort_keys=True),
                "complexity_rank": complexity_rank,
                "mean_errf": float(hedge[1][complexity_rank]) / event_count,
                "mean_coverage_gap": float(hedge[2][complexity_rank]) / event_count,
                "mean_tuwr": float(hedge[3][complexity_rank]) / tuwr_count,
                "event_count": event_count,
                "tuwr_observation_count": tuwr_count,
                "decision_digest_sha256": decision_digest(hedge[0][complexity_rank]),
                "action_counts_json": json.dumps(
                    {
                        action: int(hedge[5][complexity_rank, action_index])
                        for action_index, action in enumerate(contracts.actions)
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "fit_zone_count": len(training_zones),
                "causal_validation_feedback_count": event_count * len(contracts.actions),
                "matured_before_last_issue_event_count": matured_expected,
                "final_feedback_drain_event_count": event_count - matured_expected,
                "outer_heldout_zone_timeline_read": False,
                "outer_heldout_zone_fit": False,
                "inner_validation_zone_supervised_fit": False,
                "causal_validation_zone_online_update": True,
                "full_information_feedback": True,
                "selected_action_feedback_only": False,
            }
        )
    for complexity_rank, (alpha, l2) in enumerate(linucb_grid):
        config = {"exploration_alpha": alpha, "l2_regularization": l2}
        config_id = canonical_json_sha256(
            {
                "baseline_id": "LinUCB",
                **config,
                "state_scope": ["theta_id", "seed"],
                "context_dimension": 31,
            }
        )
        tuwr_count = int(linucb[4][complexity_rank])
        rows.append(
            {
                "outer_heldout_zone": outer_zone,
                "inner_validation_zone": validation_zone,
                "baseline_id": "LinUCB",
                "configuration_id": config_id,
                "policy_run_configuration_id": config_id,
                "configuration_json": json.dumps(config, ensure_ascii=False, sort_keys=True),
                "fitted_parameter_json": json.dumps({}, sort_keys=True),
                "complexity_rank": complexity_rank,
                "mean_errf": float(linucb[1][complexity_rank]) / event_count,
                "mean_coverage_gap": float(linucb[2][complexity_rank]) / event_count,
                "mean_tuwr": float(linucb[3][complexity_rank]) / tuwr_count,
                "event_count": event_count,
                "tuwr_observation_count": tuwr_count,
                "decision_digest_sha256": decision_digest(linucb[0][complexity_rank]),
                "action_counts_json": json.dumps(
                    {
                        action: int(linucb[5][complexity_rank, action_index])
                        for action_index, action in enumerate(contracts.actions)
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "fit_zone_count": 0,
                "causal_validation_feedback_count": event_count,
                "matured_before_last_issue_event_count": matured_expected,
                "final_feedback_drain_event_count": event_count - matured_expected,
                "outer_heldout_zone_timeline_read": False,
                "outer_heldout_zone_fit": False,
                "inner_validation_zone_supervised_fit": False,
                "causal_validation_zone_online_update": True,
                "full_information_feedback": False,
                "selected_action_feedback_only": True,
            }
        )
    del hedge, linucb, context_ids, width_code
    scores = pd.DataFrame(rows).sort_values(
        ["baseline_id", "complexity_rank", "configuration_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    numeric = scores[["mean_errf", "mean_coverage_gap", "mean_tuwr"]].to_numpy(dtype=float)
    if len(scores) != EXPECTED_INNER_SCORE_COUNT or not np.isfinite(numeric).all():
        raise RuntimeError("顺序网格内层分数行数或有限值失配")
    return scores, hedge_seconds, linucb_seconds


def frames_reproducible(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    columns = [
        column
        for column in left.columns
        if column not in {"evaluation_seconds", "rss_gib"}
    ]
    return left[columns].equals(right[columns])


def select_outer(
    *,
    contracts: Any,
    outer_zone: str,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for baseline_id in ("DelayedHedge", "LinUCB"):
        baseline_scores = scores[scores["baseline_id"].astype(str).eq(baseline_id)].copy()
        selected = select_configuration(contracts=contracts, validation_scores=baseline_scores)
        matching = baseline_scores[
            baseline_scores["configuration_id"].astype(str).eq(selected.selected_configuration_id)
        ]["configuration_json"].unique()
        if len(matching) != 1:
            raise RuntimeError("顺序基线选中配置正文不唯一")
        rows.append(
            {
                "outer_heldout_zone": outer_zone,
                "baseline_id": baseline_id,
                **selected.as_record(),
                "selected_configuration_json": str(matching[0]),
                "source_validation_zone_count": 9,
                "outer_heldout_zone_timeline_read_count": 0,
                "outer_heldout_zone_fit_count": 0,
                "outer_heldout_zone_selection_count": 0,
            }
        )
    return pd.DataFrame(rows)


def record_exception(stage: str, error: BaseException) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOG_ROOT / "sequential_selection_exception_ledger.csv"
    row = pd.DataFrame(
        [
            {
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "stage": stage,
                "exception_type": type(error).__name__,
                "message": str(error),
                "protocol_drift": False,
                "input_hash_mismatch": False,
                "causal_violation": False,
                "scientific_stop_trigger": False,
            }
        ]
    )
    if path.exists():
        previous = pd.read_csv(path, encoding="utf-8")
        row = pd.concat([previous, row], ignore_index=True)
    row.to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    numba.set_num_threads(min(NUMBA_THREAD_COUNT, numba.config.NUMBA_NUM_THREADS))
    if numba.get_num_threads() != NUMBA_THREAD_COUNT:
        raise RuntimeError(f"Numba线程数失配: {numba.get_num_threads()}")
    if git_output("rev-parse", "HEAD") != EXPECTED_CODE_COMMIT or git_output("status", "--porcelain"):
        raise RuntimeError("顺序全量选择要求批准提交且权威代码工作树干净")
    identity_paths = {
        S06_ROOT / "configs" / "source_tuning_contract_v1.json": SOURCE_TUNING_CONTRACT_SHA256,
        S06_ROOT / "configs" / "sequential_baseline_contract_v1.json": SEQUENTIAL_CONTRACT_SHA256,
        RUN_ROOT / "fold_width_thresholds.parquet": WIDTH_THRESHOLDS_SHA256,
        PRECOMPUTE_ROOT / "delayed_hedge_loss_scales.parquet": LOSS_SCALES_SHA256,
        S06_ROOT / "qa" / "sequential_precompute_v1" / "qa_summary.json": PRECOMPUTE_QA_SHA256,
        S06_ROOT / "qa" / "sequential_compact_equivalence_v1" / "qa_summary.json": COMPACT_EQUIVALENCE_QA_SHA256,
        CODE_ROOT / "baseline_sequential_compact.py": COMPACT_ENGINE_SHA256,
    }
    for path, expected in identity_paths.items():
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"顺序全量选择输入哈希失配: {path.name}, {observed}")
    free_before = shutil.disk_usage(REVISION_ROOT).free / (1024.0**3)
    if free_before < MINIMUM_FREE_DISK_GIB:
        raise RuntimeError(f"顺序全量选择磁盘安全门失败: {free_before:.3f} GiB")

    contracts = load_frozen_contracts()
    encoder = FrozenStateEncoder(contracts)
    zones = sorted(
        (str(zone) for zone in contracts.protocol["datasets"]["gefcom2014"]["zones"]),
        key=zone_number,
    )
    eta_values, linucb_grid = grid_contracts(contracts)
    thresholds = pd.read_parquet(RUN_ROOT / "fold_width_thresholds.parquet")
    threshold_lookup = {
        outer_zone: threshold_vectors(
            thresholds=thresholds,
            outer_zone=outer_zone,
            encoder=encoder,
        )
        for outer_zone in zones
    }
    scales = pd.read_parquet(PRECOMPUTE_ROOT / "delayed_hedge_loss_scales.parquet")
    scale_lookup = scale_by_excluded_pair(scales)
    context_active, offsets, sizes, _ = build_context_lookup(encoder)
    SELECTION_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    exception_path = LOG_ROOT / "sequential_selection_exception_ledger.csv"
    if not exception_path.exists():
        pd.DataFrame(
            columns=[
                "recorded_at_utc",
                "stage",
                "exception_type",
                "message",
                "protocol_drift",
                "input_hash_mismatch",
                "causal_violation",
                "scientific_stop_trigger",
            ]
        ).to_csv(exception_path, index=False, encoding="utf-8")

    resources: list[dict[str, Any]] = []
    completed = 0
    repeatability_done = False
    for validation_zone in zones:
        timeline_root = TIMELINE_ROOT / validation_zone
        timeline_manifest_path = timeline_root / "manifest.json"
        timeline_manifest = load_json(timeline_manifest_path)
        timeline_manifest_sha256 = sha256_file(timeline_manifest_path)
        if timeline_manifest.get("status") != "COMPLETE_VALIDATED":
            raise RuntimeError(f"{validation_zone}时间轴状态未通过")
        timeline = {
            "state_group": np.load(timeline_root / "state_group.npy", mmap_mode="r"),
            "seed": np.load(timeline_root / "seed.npy", mmap_mode="r"),
            "raw_width": np.load(timeline_root / "raw_width.npy", mmap_mode="r"),
            "losses": np.load(timeline_root / "losses.npy", mmap_mode="r"),
            "covered": np.load(timeline_root / "covered.npy", mmap_mode="r"),
            "tuwr": np.load(timeline_root / "tuwr.npy", mmap_mode="r"),
            "issue_offsets": np.load(timeline_root / "issue_offsets.npy", mmap_mode="r"),
            "feedback_order": np.load(timeline_root / "feedback_order.npy", mmap_mode="r"),
            "matured_ends": np.load(timeline_root / "matured_feedback_ends.npy", mmap_mode="r"),
        }
        base_features = np.load(timeline_root / "base_feature_indices.npy", mmap_mode="r")
        base_context, coverage_code = base_context_and_coverage(base_features, offsets, sizes)
        for outer_zone in zones:
            if outer_zone == validation_zone:
                continue
            folds = {
                fold.inner_validation_zone: fold
                for fold in build_nested_source_zone_folds(
                    contracts=contracts,
                    outer_heldout_zone=outer_zone,
                )
            }
            fold = folds[validation_zone]
            training_zones = tuple(str(zone) for zone in fold.inner_training_zones)
            checkpoint = verified_checkpoint(
                outer_zone=outer_zone,
                validation_zone=validation_zone,
                training_zones=training_zones,
                timeline_manifest_sha256=timeline_manifest_sha256,
            )
            pair_started = time.perf_counter()
            if checkpoint is None:
                scale_record = scale_lookup[frozenset((outer_zone, validation_zone))]
                expected_training = set(zones) - {outer_zone, validation_zone}
                if set(json.loads(str(scale_record["training_zones_json"]))) != expected_training:
                    raise RuntimeError("DelayedHedge尺度训练区集合失配")
                loss_scale = float(scale_record["loss_scale"])
                q33, q67 = threshold_lookup[outer_zone]
                scores, hedge_seconds, linucb_seconds = evaluate_pair(
                    contracts=contracts,
                    encoder=encoder,
                    outer_zone=outer_zone,
                    validation_zone=validation_zone,
                    training_zones=training_zones,
                    base_context=base_context,
                    coverage_code=coverage_code,
                    context_active=context_active,
                    timeline=timeline,
                    q33=q33,
                    q67=q67,
                    eta_values=eta_values,
                    linucb_grid=linucb_grid,
                    loss_scale=loss_scale,
                )
                repeatability_checked = False
                if not repeatability_done:
                    repeated, _, _ = evaluate_pair(
                        contracts=contracts,
                        encoder=encoder,
                        outer_zone=outer_zone,
                        validation_zone=validation_zone,
                        training_zones=training_zones,
                        base_context=base_context,
                        coverage_code=coverage_code,
                        context_active=context_active,
                        timeline=timeline,
                        q33=q33,
                        q67=q67,
                        eta_values=eta_values,
                        linucb_grid=linucb_grid,
                        loss_scale=loss_scale,
                    )
                    if not frames_reproducible(scores, repeated):
                        raise RuntimeError("顺序全量首折重复运行逐值不一致")
                    repeatability_checked = True
                    repeatability_done = True
                write_checkpoint(
                    scores=scores,
                    outer_zone=outer_zone,
                    validation_zone=validation_zone,
                    training_zones=training_zones,
                    timeline_manifest_sha256=timeline_manifest_sha256,
                    loss_scale=loss_scale,
                    elapsed_seconds=time.perf_counter() - pair_started,
                    hedge_seconds=hedge_seconds,
                    linucb_seconds=linucb_seconds,
                    repeatability_checked=repeatability_checked,
                )
                status = "COMPUTED"
            else:
                scores = checkpoint
                hedge_seconds = 0.0
                linucb_seconds = 0.0
                status = "REUSED"
            completed += 1
            resource = {
                "outer_heldout_zone": outer_zone,
                "inner_validation_zone": validation_zone,
                "status": status,
                "elapsed_seconds_this_invocation": time.perf_counter() - pair_started,
                "hedge_seconds": hedge_seconds,
                "linucb_seconds": linucb_seconds,
                "rss_gib": psutil.Process().memory_info().rss / (1024.0**3),
                "free_disk_gib": shutil.disk_usage(REVISION_ROOT).free / (1024.0**3),
            }
            resources.append(resource)
            print(
                json.dumps(
                    {
                        "stage": "sequential_nested_selection",
                        "completed_inner_folds": completed,
                        "total_inner_folds": EXPECTED_INNER_FOLD_COUNT,
                        **resource,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        del timeline, base_features, base_context, coverage_code

    if completed != EXPECTED_INNER_FOLD_COUNT:
        raise RuntimeError(f"顺序内层折完成数失配: {completed}")
    outer_manifests: list[dict[str, Any]] = []
    root_scores: list[pd.DataFrame] = []
    root_selections: list[pd.DataFrame] = []
    for outer_zone in zones:
        outer_root = SELECTION_ROOT / f"outer_heldout={outer_zone}"
        score_frames = []
        for validation_zone in zones:
            if validation_zone == outer_zone:
                continue
            score_path, _ = checkpoint_paths(outer_zone, validation_zone)
            score_frames.append(pd.read_parquet(score_path))
        scores = pd.concat(score_frames, ignore_index=True)
        if len(scores) != EXPECTED_INNER_SCORE_COUNT * 9:
            raise RuntimeError(f"{outer_zone}顺序外层分数行数失配")
        selections = select_outer(contracts=contracts, outer_zone=outer_zone, scores=scores)
        scores_path = outer_root / "all_validation_scores.parquet"
        selections_path = outer_root / "selected_configurations.parquet"
        scores.to_parquet(scores_path, index=False, compression="zstd")
        selections.to_parquet(selections_path, index=False, compression="zstd")
        manifest = {
            "manifest_schema": "S06_SEQUENTIAL_OUTER_SELECTION_V1",
            "status": "PASS",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": EXPECTED_CODE_COMMIT,
            "source_tuning_contract_sha256": SOURCE_TUNING_CONTRACT_SHA256,
            "sequential_contract_sha256": SEQUENTIAL_CONTRACT_SHA256,
            "precompute_qa_sha256": PRECOMPUTE_QA_SHA256,
            "outer_heldout_zone": outer_zone,
            "inner_fold_count": 9,
            "score_row_count": len(scores),
            "selection_row_count": len(selections),
            "scores_sha256": sha256_file(scores_path),
            "selections_sha256": sha256_file(selections_path),
            "fallback_selection_count": int(selections["no_feasible_fallback_used"].sum()),
            "outer_heldout_zone_timeline_read_count": 0,
            "outer_heldout_zone_fit_count": 0,
            "outer_heldout_zone_selection_count": 0,
            "configuration_selection_executed": True,
            "performance_comparison_executed": False,
            "method_ranking_executed": False,
            "statistical_inference_executed": False,
            "paper_performance_claim_executed": False,
        }
        (outer_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        outer_manifests.append(manifest)
        root_scores.append(scores)
        root_selections.append(selections)

    all_scores = pd.concat(root_scores, ignore_index=True)
    all_selections = pd.concat(root_selections, ignore_index=True)
    if len(all_scores) != EXPECTED_ROOT_SCORE_COUNT or len(all_selections) != EXPECTED_ROOT_SELECTION_COUNT:
        raise RuntimeError("顺序根级分数或选择行数失配")
    score_output = SELECTION_ROOT / "sequential_validation_scores.parquet"
    selection_output = SELECTION_ROOT / "sequential_selected_configurations.parquet"
    all_scores.to_parquet(score_output, index=False, compression="zstd")
    all_selections.to_parquet(selection_output, index=False, compression="zstd")
    resource_path = LOG_ROOT / "sequential_selection_resource_samples.csv"
    pd.DataFrame(resources).to_csv(resource_path, index=False, encoding="utf-8")
    free_after = shutil.disk_usage(REVISION_ROOT).free / (1024.0**3)
    if free_after < MINIMUM_FREE_DISK_GIB:
        raise RuntimeError(f"顺序全量选择结束磁盘安全门失败: {free_after:.3f} GiB")
    fallback_by_baseline = (
        all_selections.groupby("baseline_id")["no_feasible_fallback_used"].sum().astype(int).to_dict()
    )
    exceptions = pd.read_csv(exception_path, encoding="utf-8")
    summary = {
        "run_id": "S06_SEQUENTIAL_NESTED_SELECTION_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "code_commit": EXPECTED_CODE_COMMIT,
        "source_tuning_contract_sha256": SOURCE_TUNING_CONTRACT_SHA256,
        "sequential_contract_sha256": SEQUENTIAL_CONTRACT_SHA256,
        "width_thresholds_sha256": WIDTH_THRESHOLDS_SHA256,
        "loss_scales_sha256": LOSS_SCALES_SHA256,
        "precompute_qa_sha256": PRECOMPUTE_QA_SHA256,
        "compact_equivalence_qa_sha256": COMPACT_EQUIVALENCE_QA_SHA256,
        "compact_engine_sha256": COMPACT_ENGINE_SHA256,
        "numba_thread_count": numba.get_num_threads(),
        "outer_fold_count": len(zones),
        "inner_fold_count": completed,
        "configuration_counts": {
            "DelayedHedge": EXPECTED_HEDGE_CONFIG_COUNT,
            "LinUCB": EXPECTED_LINUCB_CONFIG_COUNT,
        },
        "validation_score_row_count": len(all_scores),
        "selected_configuration_row_count": len(all_selections),
        "no_feasible_fallback_count_by_baseline": fallback_by_baseline,
        "repeatability_fold_count": sum(
            int(load_json(checkpoint_paths(row["outer_heldout_zone"], row["inner_validation_zone"])[1]).get("repeatability_checked", False))
            for row in resources
        ),
        "exception_ledger_row_count": len(exceptions),
        "outer_heldout_zone_timeline_read_count": 0,
        "outer_heldout_zone_fit_count": 0,
        "outer_heldout_zone_selection_count": 0,
        "validation_scores_sha256": sha256_file(score_output),
        "selected_configurations_sha256": sha256_file(selection_output),
        "resource_samples_sha256": sha256_file(resource_path),
        "exception_ledger_sha256": sha256_file(exception_path),
        "elapsed_seconds": time.perf_counter() - started,
        "free_disk_gib_before": free_before,
        "free_disk_gib_after": free_after,
        "configuration_selection_executed": True,
        "heldout_performance_read": False,
        "performance_comparison_executed": False,
        "method_ranking_executed": False,
        "statistical_inference_executed": False,
        "paper_performance_claim_executed": False,
        "new_scientific_adverse_evidence": False,
        "stop_triggered": False,
    }
    summary_path = SELECTION_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        record_exception("sequential_nested_selection", error)
        raise
