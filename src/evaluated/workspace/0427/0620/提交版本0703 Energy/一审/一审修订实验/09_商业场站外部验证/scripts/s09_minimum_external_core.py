"""S09 最小外部验证的共享计算核心。

本模块只复用已经封存的候选区间、来源选择器和目标适配区。
基础预测器与候选动作定义均不会在这里重新训练或修改。
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from dataclasses import replace
from heapq import heappop, heappush
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


S09_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = Path(__file__).resolve().parents[2]
AUTHORITATIVE_CODE = REVISION_ROOT / "_权威代码" / "code"
sys.path.insert(0, str(AUTHORITATIVE_CODE))

from baseline_compact_training import fit_compact_cart_best_action  # noqa: E402
from baseline_common import FrozenStateEncoder  # noqa: E402
from baseline_conformal import (  # noqa: E402
    conformal_grid,
    run_conformal_configuration,
)
from clara_errf import ErrfTheta  # noqa: E402
from clara_event_contract import FrozenContracts  # noqa: E402
from source_tuning_facts import (  # noqa: E402
    attach_raw_width_states,
    build_source_tuning_facts,
    fit_raw_width_thresholds,
)


ACTIONS = ("Static", "ACI", "AgACI", "EnbPI_RH")
STATE_FIELDS = (
    "predictor",
    "horizon_group",
    "target_coverage",
    "ramp_state",
    "rolling_state",
    "raw_width_state",
)
WIDTH_FIELDS = (
    "predictor",
    "horizon_group",
    "target_coverage",
    "seed",
)
OUTER_FOLDS = tuple(f"zone{index}" for index in range(1, 11))
PRICE_ID = "R05P6179775281"
V4_VERSION = "V4_FULLY_ADAPTIVE"

FULL_REBUILD_ROOT = S09_ROOT / "results_raw" / "full_external_rebuild_v1"
ADAPTATION_ROOT = S09_ROOT / "results_raw" / "adaptation_candidate_rebuild_v1"
SOURCE_SELECTION_ROOT = (
    REVISION_ROOT
    / "06_基线实现与训练区调参"
    / "results_raw"
    / "nested_source_selection_v1"
)
SOURCE_STATE_STATISTICS = SOURCE_SELECTION_ROOT / "state_sufficient_statistics.parquet"
SUPERVISED_SELECTED = (
    SOURCE_SELECTION_ROOT
    / "nested_selection"
    / "supervised_v1"
    / "supervised_selected_configurations.parquet"
)
FOLD_WIDTH_THRESHOLDS = SOURCE_SELECTION_ROOT / "fold_width_thresholds.parquet"
SOURCE_V4_ROOT = (
    REVISION_ROOT
    / "测试CLARA"
    / "results_raw"
    / "four_version_comparison_v1"
)
SOURCE_V4_CONFIG = (
    REVISION_ROOT
    / "测试CLARA"
    / "configs"
    / "test_clara_four_versions_v1.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def bundle_identifier(farm: str, predictor: str, horizon: int, seed: int) -> str:
    return f"{predictor}-H{int(horizon):02d}-{farm}-S{int(seed)}"


def bundle_paths(
    *,
    root: Path,
    farm: str,
    predictor: str,
    horizon: int,
    seed: int,
) -> dict[str, Path]:
    identifier = bundle_identifier(farm, predictor, horizon, seed)
    bundle = root / "bundles" / predictor / identifier
    base = root / "base" / predictor / identifier
    return {
        "identifier": Path(identifier),
        "bundle": bundle,
        "manifest": bundle / "manifest.json",
        "event_core": bundle / "event_core.parquet",
        "candidate_intervals": bundle / "candidate_intervals.parquet",
        "base_manifest": base / "manifest.json",
        "base_predictions": base / "base_predictions.parquet",
    }


def load_fact_bundle(
    *,
    root: Path,
    farm: str,
    predictor: str,
    horizon: int,
    seed: int,
    contracts: FrozenContracts,
    theta: ErrfTheta | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = bundle_paths(
        root=root,
        farm=farm,
        predictor=predictor,
        horizon=horizon,
        seed=seed,
    )
    manifest = load_json(paths["manifest"])
    event_path = paths["event_core"]
    candidate_path = paths["candidate_intervals"]
    if (
        manifest.get("status") != "COMPLETE_VALIDATED"
        or str(manifest.get("event_core_sha256")) != sha256_file(event_path)
        or str(manifest.get("candidate_intervals_sha256"))
        != sha256_file(candidate_path)
        or str(manifest.get("zone")) != str(farm)
        or str(manifest.get("predictor")) != str(predictor)
        or int(manifest.get("horizon")) != int(horizon)
        or int(manifest.get("seed")) != int(seed)
    ):
        raise RuntimeError(f"S09 候选包身份失配: {paths['identifier']}")
    event_core = pd.read_parquet(event_path)
    candidates = pd.read_parquet(candidate_path)
    result = build_source_tuning_facts(
        event_core=event_core,
        candidates=candidates,
        contracts=contracts,
        theta=theta or ErrfTheta(),
    )
    facts = result.facts.copy()
    extra = event_core.loc[
        event_core["label_available_timestamp"].notna(),
        [
            "event_id",
            "target",
            "base_center",
            "nominal_cadence_minutes",
        ],
    ].copy()
    extra["event_id"] = extra["event_id"].astype(str)
    facts = facts.merge(extra, on="event_id", how="left", validate="one_to_one")
    if facts[["target", "base_center", "nominal_cadence_minutes"]].isna().any().any():
        raise RuntimeError(f"S09 完整案例事实缺少评价字段: {paths['identifier']}")
    facts["origin"] = "target" if root == FULL_REBUILD_ROOT else "source"
    facts["bundle_id"] = str(paths["identifier"])
    audit = {
        **result.audit,
        "bundle_id": str(paths["identifier"]),
        "bundle_manifest_sha256": sha256_file(paths["manifest"]),
        "event_core_sha256": sha256_file(event_path),
        "candidate_intervals_sha256": sha256_file(candidate_path),
    }
    return facts, audit


def fit_local_width_thresholds(facts: pd.DataFrame, farm: str) -> pd.DataFrame:
    source = facts.copy()
    source["zone_or_farm"] = str(farm)
    thresholds = fit_raw_width_thresholds(source, source_zones=(str(farm),))
    expected = (
        source.loc[:, list(WIDTH_FIELDS)]
        .drop_duplicates()
        .sort_values(list(WIDTH_FIELDS), kind="mergesort")
    )
    observed = thresholds.loc[:, list(WIDTH_FIELDS)].sort_values(
        list(WIDTH_FIELDS), kind="mergesort"
    )
    if len(expected) != len(observed):
        raise RuntimeError("本地宽度阈值键数量失配")
    return thresholds


def attach_local_width_states(
    facts: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    output = facts.copy()
    output["_row_order"] = np.arange(len(output), dtype=np.int64)
    output = attach_raw_width_states(output, thresholds=thresholds)
    return (
        output.sort_values("_row_order", kind="mergesort")
        .drop(columns="_row_order")
        .reset_index(drop=True)
    )


def load_source_v4_decisions(seed: int) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    expected_config = sha256_file(SOURCE_V4_CONFIG)
    decisions: dict[str, pd.DataFrame] = {}
    audits: list[dict[str, Any]] = []
    for outer_fold in OUTER_FOLDS:
        root = SOURCE_V4_ROOT / f"outer_fold={outer_fold}" / f"seed={int(seed)}"
        path = root / "state_decisions.parquet"
        manifest_path = root / "unit_manifest.json"
        manifest = load_json(manifest_path)
        if (
            manifest.get("status") != "PASS"
            or manifest.get("config_sha256") != expected_config
            or manifest.get("state_decisions_sha256") != sha256_file(path)
            or int(manifest.get("fixed_action_regression_mismatch_count", -1)) != 0
        ):
            raise RuntimeError(f"V4 来源状态决策身份失配: {outer_fold}/seed{seed}")
        frame = pd.read_parquet(path)
        state = frame[
            frame["price_id"].astype(str).eq(PRICE_ID)
            & frame["version"].astype(str).eq(V4_VERSION)
        ][list(STATE_FIELDS) + ["selected_action"]].copy()
        if len(state) != 6600 or state.duplicated(list(STATE_FIELDS)).any():
            raise RuntimeError(f"V4 来源状态决策范围失配: {outer_fold}/seed{seed}")
        decisions[outer_fold] = state
        audits.append(
            {
                "outer_fold": outer_fold,
                "seed": int(seed),
                "manifest_sha256": sha256_file(manifest_path),
                "state_decisions_sha256": sha256_file(path),
            }
        )
    return decisions, audits


def _vote_actions(votes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.column_stack([(votes == action).sum(axis=1) for action in ACTIONS])
    maximum = counts.max(axis=1)
    selected = counts.argmax(axis=1)
    second = np.partition(counts, kth=len(ACTIONS) - 2, axis=1)[:, len(ACTIONS) - 2]
    return np.asarray(ACTIONS, dtype=object)[selected], maximum, maximum - second


def predict_clara_direct(
    facts: pd.DataFrame,
    *,
    seed: int,
    thresholds: pd.DataFrame,
    source_decisions: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, pd.DataFrame]:
    votes: list[np.ndarray] = []
    audit_rows: list[dict[str, Any]] = []
    reference_ids = facts["event_id"].astype(str).to_numpy()
    for outer_fold in OUTER_FOLDS:
        fold_thresholds = thresholds[
            thresholds["outer_heldout_zone"].astype(str).eq(outer_fold)
            & thresholds["seed"].astype(int).eq(int(seed))
        ].copy()
        # Direct 分支必须按每个源折重新分类宽度，移除可能存在的目标域分类列。
        classified = facts.drop(
            columns=["raw_width_state", "raw_width_q33", "raw_width_q67"],
            errors="ignore",
        ).copy()
        classified["_row_order"] = np.arange(len(classified), dtype=np.int64)
        classified = attach_raw_width_states(classified, thresholds=fold_thresholds)
        mapped = classified.merge(
            source_decisions[outer_fold],
            on=list(STATE_FIELDS),
            how="left",
            validate="many_to_one",
            sort=False,
        ).sort_values("_row_order", kind="mergesort")
        if mapped["selected_action"].isna().any():
            raise RuntimeError(f"CLARA Direct 外部状态映射缺失: {outer_fold}")
        if not np.array_equal(mapped["event_id"].astype(str).to_numpy(), reference_ids):
            raise RuntimeError(f"CLARA Direct 外部映射改变事件顺序: {outer_fold}")
        votes.append(mapped["selected_action"].astype(str).to_numpy())
        audit_rows.append(
            {
                "outer_fold": outer_fold,
                "mapped_event_count": len(mapped),
                "threshold_row_count": len(fold_thresholds),
                "missing_action_count": 0,
            }
        )
    selected, maximum, margin = _vote_actions(np.column_stack(votes))
    audit = pd.DataFrame(audit_rows)
    audit["maximum_vote_count_mean"] = float(np.mean(maximum))
    audit["consensus_margin_mean"] = float(np.mean(margin))
    return selected.astype(str), audit


def load_source_cart_models(
    contracts: FrozenContracts,
) -> tuple[dict[str, Any], pd.DataFrame]:
    statistics = pd.read_parquet(SOURCE_STATE_STATISTICS)
    selected = pd.read_parquet(SUPERVISED_SELECTED)
    selected = selected[selected["baseline_id"].astype(str).eq("CARTBestAction")].copy()
    models: dict[str, Any] = {}
    audits: list[dict[str, Any]] = []
    for outer_fold in OUTER_FOLDS:
        row = selected[selected["outer_heldout_zone"].astype(str).eq(outer_fold)]
        if len(row) != 1:
            raise RuntimeError(f"来源 CART 选中配置数量失配: {outer_fold}")
        config = json.loads(str(row.iloc[0]["selected_configuration_json"]))
        config.pop("random_state", None)
        part = statistics[
            statistics["outer_heldout_zone"].astype(str).eq(outer_fold)
        ].copy()
        training_zones = tuple(
            sorted(set(part["source_zone"].astype(str)) - {outer_fold})
        )
        model = fit_compact_cart_best_action(
            contracts=contracts,
            statistics=part,
            config=config,
            training_zones=training_zones,
            outer_heldout_zone=outer_fold,
        )
        models[outer_fold] = model
        audits.append(
            {
                "outer_fold": outer_fold,
                "configuration_id": str(row.iloc[0]["selected_configuration_id"]),
                "configuration_json": json.dumps(config, sort_keys=True),
                "training_zone_count": len(training_zones),
                "fit_zone_count": len(model.fit_zones),
            }
        )
    return models, pd.DataFrame(audits)


def predict_cart_direct(
    facts: pd.DataFrame,
    *,
    seed: int,
    thresholds: pd.DataFrame,
    models: dict[str, Any],
) -> tuple[np.ndarray, pd.DataFrame]:
    votes: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    for outer_fold in OUTER_FOLDS:
        fold_thresholds = thresholds[
            thresholds["outer_heldout_zone"].astype(str).eq(outer_fold)
            & thresholds["seed"].astype(int).eq(int(seed))
        ].copy()
        # Direct 分支使用源折阈值，不能继承目标域宽度分类列。
        raw_facts = facts.drop(
            columns=["raw_width_state", "raw_width_q33", "raw_width_q67"],
            errors="ignore",
        )
        classified = attach_raw_width_states(raw_facts, thresholds=fold_thresholds)
        event_frame = classified[["event_id", *STATE_FIELDS]].copy()
        actions, scores = models[outer_fold].predict_actions(event_frame)
        mapped = event_frame[["event_id"]].copy()
        mapped["selected_action"] = mapped["event_id"].astype(str).map(actions)
        if mapped["selected_action"].isna().any():
            raise RuntimeError(f"CART Direct 预测映射缺失: {outer_fold}")
        votes.append(mapped["selected_action"].astype(str).to_numpy())
        audits.append(
            {
                "outer_fold": outer_fold,
                "event_count": len(mapped),
                "score_min": float(min(scores.values())),
                "score_max": float(max(scores.values())),
            }
        )
    selected, maximum, margin = _vote_actions(np.column_stack(votes))
    audit = pd.DataFrame(audits)
    audit["maximum_vote_count_mean"] = float(np.mean(maximum))
    audit["consensus_margin_mean"] = float(np.mean(margin))
    return selected.astype(str), audit


def compact_statistics_from_facts(
    facts: pd.DataFrame,
    *,
    source_zone: str,
) -> pd.DataFrame:
    frame = facts.reset_index(drop=True).copy()
    errf = np.column_stack(
        [frame[f"{action}__errf"].to_numpy(dtype=float) for action in ACTIONS]
    )
    best = np.argmin(errf, axis=1)
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(list(STATE_FIELDS), sort=True, dropna=False):
        positions = group.index.to_numpy(dtype=np.int64)
        row: dict[str, Any] = {
            "source_zone": str(source_zone),
            **dict(zip(STATE_FIELDS, key)),
            "event_count": len(group),
        }
        for action_index, action in enumerate(ACTIONS):
            tuwr = group[f"{action}__tuwr_indicator"]
            row[f"{action}__errf_sum"] = float(group[f"{action}__errf"].sum())
            row[f"{action}__covered_sum"] = int(group[f"{action}__covered"].sum())
            row[f"{action}__tuwr_sum"] = float(tuwr.sum(skipna=True))
            row[f"{action}__tuwr_count"] = int(tuwr.notna().sum())
            row[f"{action}__best_count"] = int((best[positions] == action_index).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def fit_cart_local(
    facts: pd.DataFrame,
    *,
    farm: str,
    contracts: FrozenContracts,
    config: dict[str, Any],
) -> tuple[Any, pd.DataFrame]:
    statistics = compact_statistics_from_facts(facts, source_zone=farm)
    fitting_config = dict(config)
    fitting_config.pop("random_state", None)
    selector = fit_compact_cart_best_action(
        contracts=contracts,
        statistics=statistics,
        config=fitting_config,
        training_zones=(farm,),
        outer_heldout_zone="__external_final_test__",
    )
    return selector, statistics


def predict_cart_local(facts: pd.DataFrame, selector: Any) -> tuple[np.ndarray, np.ndarray]:
    actions, scores = selector.predict_actions(facts[["event_id", *STATE_FIELDS]].copy())
    event_ids = facts["event_id"].astype(str)
    selected = event_ids.map(actions)
    score = event_ids.map(scores)
    if selected.isna().any() or score.isna().any():
        raise RuntimeError("CART Local 外部事件预测缺失")
    return selected.astype(str).to_numpy(), score.to_numpy(dtype=float)


def parameterized_contracts(
    contracts: FrozenContracts,
    *,
    n_min: int,
    nu: int,
    minimum_source_zones: int = 1,
) -> FrozenContracts:
    protocol = copy.deepcopy(contracts.protocol)
    support = protocol["support_and_backoff_contract"]
    support["default_n_min"] = int(n_min)
    support["default_nu"] = int(nu)
    support["minimum_source_zones"] = int(minimum_source_zones)
    return replace(contracts, protocol=protocol)


def _utc_day(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="raise")
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize(
            "Asia/Shanghai", ambiguous="raise", nonexistent="raise"
        )
    return timestamps.dt.tz_convert("UTC").dt.floor("D")


def _state_universe(facts: pd.DataFrame) -> pd.DataFrame:
    states = facts.loc[:, list(STATE_FIELDS)].drop_duplicates().copy()
    return states.sort_values(list(STATE_FIELDS), kind="mergesort").reset_index(drop=True)


def frozen_state_universe() -> pd.DataFrame:
    path = SOURCE_V4_ROOT / "outer_fold=zone1" / "seed=0" / "state_decisions.parquet"
    frame = pd.read_parquet(path)
    states = frame[
        frame["price_id"].astype(str).eq(PRICE_ID)
        & frame["version"].astype(str).eq(V4_VERSION)
    ][list(STATE_FIELDS)].drop_duplicates()
    states = states.sort_values(list(STATE_FIELDS), kind="mergesort").reset_index(drop=True)
    if len(states) != 6600:
        raise RuntimeError("冻结状态全集数量失配")
    return states


def _group_key_frame(states: pd.DataFrame, fields: Sequence[str]) -> pd.DataFrame:
    if fields:
        return states.loc[:, list(fields)].copy()
    return pd.DataFrame({"_global_key": np.zeros(len(states), dtype=np.int8)})


def _training_level_statistics(
    facts: pd.DataFrame,
    *,
    fields: Sequence[str],
) -> pd.DataFrame:
    frame = facts.copy()
    frame["_utc_day"] = _utc_day(frame["issue_timestamp"])
    group_fields = list(fields)
    if not group_fields:
        frame["_global_key"] = 0
        group_fields = ["_global_key"]
    aggregation: dict[str, tuple[str, str]] = {"event_count": ("event_id", "size")}
    for action in ACTIONS:
        aggregation[f"{action}__errf_mean"] = (f"{action}__errf", "mean")
        aggregation[f"{action}__covered_mean"] = (f"{action}__covered", "mean")
        aggregation[f"{action}__tuwr_mean"] = (f"{action}__tuwr_indicator", "mean")
        aggregation[f"{action}__ard_mean"] = (f"{action}__ard_value", "mean")
    grouped = frame.groupby(group_fields, sort=True, dropna=False).agg(**aggregation).reset_index()
    for action in ACTIONS:
        guardrail = frame[
            frame[f"{action}__tuwr_indicator"].notna()
            & frame[f"{action}__ard_value"].notna()
        ]
        guardrail_coverage = (
            guardrail.groupby(group_fields, sort=True, dropna=False)[
                f"{action}__covered"
            ]
            .mean()
            .rename(f"{action}__covered_guardrail_mean")
            .reset_index()
        )
        grouped = grouped.merge(
            guardrail_coverage,
            on=group_fields,
            how="left",
            validate="one_to_one",
        )
        daily = (
            frame.groupby([*group_fields, "_utc_day"], sort=True, dropna=False)[
                f"{action}__errf"
            ]
            .mean()
            .rename("daily_mean")
            .reset_index()
        )
        daily_stats = (
            daily.groupby(group_fields, sort=True, dropna=False)["daily_mean"]
            .agg(["std", "count"])
            .reset_index()
        )
        daily_stats[f"{action}__risk_day_cluster_se"] = (
            daily_stats["std"] / np.sqrt(daily_stats["count"].clip(lower=1))
        ).fillna(0.0)
        daily_stats = daily_stats[
            [*group_fields, f"{action}__risk_day_cluster_se"]
        ]
        grouped = grouped.merge(
            daily_stats,
            on=group_fields,
            how="left",
            validate="one_to_one",
        )
    return grouped


def _map_level_statistics(
    states: pd.DataFrame,
    statistics: pd.DataFrame,
    *,
    fields: Sequence[str],
) -> pd.DataFrame:
    key_frame = _group_key_frame(states, fields)
    group_fields = list(fields) if fields else ["_global_key"]
    mapped = key_frame.merge(
        statistics,
        on=group_fields,
        how="left",
        validate="many_to_one",
        sort=False,
    )
    mapped["event_count"] = mapped["event_count"].fillna(0).astype(np.int64)
    return mapped


def _selected_profile(
    mapped_levels: list[pd.DataFrame],
    *,
    n_min: int,
) -> np.ndarray:
    selected = np.full(len(mapped_levels[0]), -1, dtype=np.int64)
    for level_index, mapped in enumerate(mapped_levels):
        current = (selected < 0) & (
            mapped["event_count"].to_numpy(dtype=np.int64) >= int(n_min)
        )
        selected[current] = level_index
    if (selected < 0).any():
        raise RuntimeError("CLARA Local 七级回退后仍缺少支持")
    return selected


def _evidence_for_profile(
    *,
    states: pd.DataFrame,
    mapped_levels: list[pd.DataFrame],
    selected_levels: np.ndarray,
    nu: int | np.ndarray,
    beta: float,
) -> dict[str, np.ndarray]:
    state_count = len(selected_levels)
    action_count = len(ACTIONS)
    risk_mean = np.empty((state_count, action_count), dtype=np.float64)
    risk_parent = np.empty((state_count, action_count), dtype=np.float64)
    risk_shrunken = np.empty((state_count, action_count), dtype=np.float64)
    risk_se = np.empty((state_count, action_count), dtype=np.float64)
    coverage = np.empty((state_count, action_count), dtype=np.float64)
    tuwr = np.empty((state_count, action_count), dtype=np.float64)
    ard = np.empty((state_count, action_count), dtype=np.float64)
    nu_values = (
        np.full(state_count, float(nu), dtype=np.float64)
        if np.ndim(nu) == 0
        else np.asarray(nu, dtype=np.float64)
    )
    cold = states["rolling_state"].astype(str).eq("cold_start").to_numpy()
    for state_index in range(state_count):
        selected_level = int(selected_levels[state_index])
        for action_index, action in enumerate(ACTIONS):
            parent = float(mapped_levels[-1].iloc[state_index][f"{action}__errf_mean"])
            current_mean = parent
            current_parent = parent
            current_shrunken = parent
            if selected_level < len(mapped_levels) - 1:
                for level_index in range(len(mapped_levels) - 2, selected_level - 1, -1):
                    row = mapped_levels[level_index].iloc[state_index]
                    current_mean = float(row[f"{action}__errf_mean"])
                    current_n = float(row["event_count"])
                    current_parent = parent
                    denominator = current_n + nu_values[state_index]
                    current_shrunken = (
                        (current_n * current_mean + nu_values[state_index] * parent)
                        / denominator
                        if denominator > 0.0
                        else current_mean
                    )
                    parent = current_shrunken
            selected_row = mapped_levels[selected_level].iloc[state_index]
            risk_mean[state_index, action_index] = current_mean
            risk_parent[state_index, action_index] = current_parent
            risk_shrunken[state_index, action_index] = current_shrunken
            risk_se[state_index, action_index] = float(
                selected_row[f"{action}__risk_day_cluster_se"]
            )
            coverage_column = (
                f"{action}__covered_mean"
                if cold[state_index]
                else f"{action}__covered_guardrail_mean"
            )
            coverage[state_index, action_index] = float(selected_row[coverage_column])
            tuwr[state_index, action_index] = float(
                selected_row[f"{action}__tuwr_mean"]
            )
            ard[state_index, action_index] = float(
                selected_row[f"{action}__ard_mean"]
            )
    return {
        "risk_mean": risk_mean,
        "risk_parent": risk_parent,
        "risk_shrunken": risk_shrunken,
        "risk_se": risk_se,
        "risk_score": risk_shrunken + float(beta) * risk_se,
        "coverage": coverage,
        "tuwr": tuwr,
        "ard": ard,
    }


def _top_two_margin_ratio(evidence: dict[str, np.ndarray]) -> np.ndarray:
    scores = evidence["risk_score"]
    standard_errors = evidence["risk_se"]
    order = np.argsort(scores, axis=1, kind="stable")
    rows = np.arange(len(scores), dtype=np.int64)
    best = order[:, 0]
    second = order[:, 1]
    gap = scores[rows, second] - scores[rows, best]
    denominator = np.sqrt(
        standard_errors[rows, best] ** 2 + standard_errors[rows, second] ** 2
    )
    ratio = np.zeros(len(scores), dtype=np.float64)
    positive = denominator > 0.0
    ratio[positive] = gap[positive] / denominator[positive]
    ratio[(~positive) & (gap > 0.0)] = np.inf
    return ratio


def _select_local_actions(
    *,
    states: pd.DataFrame,
    evidence: dict[str, np.ndarray],
    thresholds: dict[str, float],
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    risk = evidence["risk_score"]
    coverage = evidence["coverage"]
    tuwr = evidence["tuwr"]
    ard = evidence["ard"]
    target = states["target_coverage"].to_numpy(dtype=float)
    cold = states["rolling_state"].astype(str).eq("cold_start").to_numpy()
    safe = coverage >= target[:, None] - float(thresholds["coverage_shortfall_epsilon"])
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
    candidates &= risk <= minimum_risk[:, None] + float(tolerance)
    maximum_coverage = np.max(np.where(candidates, coverage, -np.inf), axis=1)
    candidates &= coverage >= maximum_coverage[:, None] - float(tolerance)
    for values in (tuwr, ard):
        finite = np.isfinite(values)
        count = candidates.sum(axis=1)
        finite_count = (candidates & finite).sum(axis=1)
        mixed = (finite_count > 0) & (finite_count < count)
        if mixed.any():
            raise RuntimeError("CLARA Local 并列候选可靠性可用性不一致")
        applicable = (finite_count == count) & (count > 0)
        minimum = np.min(np.where(candidates & finite, values, np.inf), axis=1)
        candidates &= (~applicable[:, None]) | (
            finite & (values <= minimum[:, None] + float(tolerance))
        )
    if not candidates.any(axis=1).all():
        raise RuntimeError("CLARA Local 动作选择产生空候选")
    selected = np.argmax(candidates, axis=1).astype(np.int64)
    return selected, empty, safe


def fit_local_clara(
    facts: pd.DataFrame,
    *,
    contracts: FrozenContracts,
    protocol: dict[str, Any],
    v4_protocol: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    states = frozen_state_universe()
    support_fields = [tuple(level["fields"]) for level in contracts.support_levels]
    level_statistics = [
        _training_level_statistics(facts, fields=fields) for fields in support_fields
    ]
    mapped_levels = [
        _map_level_statistics(states, statistics, fields=fields)
        for statistics, fields in zip(level_statistics, support_fields)
    ]
    beta = float(contracts.protocol["landscape_contract"]["beta"])
    adaptive = v4_protocol["adaptive_support"]
    diagnostic_nu = int(adaptive["diagnostic_nu"])
    profile_by_n: dict[int, np.ndarray] = {}
    ratio_by_n: dict[int, np.ndarray] = {}
    for n_min in protocol["n_min_candidates"]:
        selected_levels = _selected_profile(mapped_levels, n_min=int(n_min))
        evidence = _evidence_for_profile(
            states=states,
            mapped_levels=mapped_levels,
            selected_levels=selected_levels,
            nu=diagnostic_nu,
            beta=beta,
        )
        profile_by_n[int(n_min)] = selected_levels
        ratio_by_n[int(n_min)] = _top_two_margin_ratio(evidence)
    selected_n = np.full(len(states), -1, dtype=np.int64)
    selected_margin = np.zeros(len(states), dtype=np.float64)
    for n_min in sorted(profile_by_n):
        current = (selected_n < 0) & (
            ratio_by_n[n_min]
            >= float(adaptive["risk_margin_se_ratio_threshold"])
        )
        selected_n[current] = int(n_min)
        selected_margin[current] = ratio_by_n[n_min][current]
    fallback_n = int(adaptive["n_min_no_candidate_rule"])
    unresolved = selected_n < 0
    selected_n[unresolved] = fallback_n
    selected_margin[unresolved] = ratio_by_n[fallback_n][unresolved]
    selected_levels = np.empty(len(states), dtype=np.int64)
    for n_min, levels in profile_by_n.items():
        mask = selected_n == int(n_min)
        selected_levels[mask] = levels[mask]
    diagnostic = _evidence_for_profile(
        states=states,
        mapped_levels=mapped_levels,
        selected_levels=selected_levels,
        nu=diagnostic_nu,
        beta=beta,
    )
    signal = np.nanmedian(
        np.abs(diagnostic["risk_mean"] - diagnostic["risk_parent"]), axis=1
    )
    noise = np.nanmedian(diagnostic["risk_se"], axis=1)
    ratio = signal / np.maximum(noise, float(adaptive["numeric_floor"]))
    selected_nu = np.full(len(states), -1, dtype=np.int64)
    for rule in protocol["nu_rules"]:
        current = (selected_nu < 0) & (ratio >= float(rule["minimum_ratio"]))
        selected_nu[current] = int(rule["nu"])
    if (selected_nu < 0).any():
        raise RuntimeError("CLARA Local 自适应 nu 规则未覆盖全部状态")
    evidence = _evidence_for_profile(
        states=states,
        mapped_levels=mapped_levels,
        selected_levels=selected_levels,
        nu=selected_nu,
        beta=beta,
    )
    selected_index, empty, safe = _select_local_actions(
        states=states,
        evidence=evidence,
        thresholds={
            "coverage_shortfall_epsilon": float(
                protocol["coverage_shortfall_epsilon"]
            ),
            "tuwr_upper": float(protocol["tuwr_upper"]),
            "ard_upper": float(protocol["ard_upper"]),
        },
        tolerance=float(
            contracts.protocol["selection_contract"]["score_tie_tolerance"]
        ),
    )
    output = states.copy()
    output["selected_action"] = np.asarray(ACTIONS, dtype=object)[selected_index]
    output["selected_n_min"] = selected_n
    output["selected_nu"] = selected_nu
    output["support_backoff_level"] = selected_levels
    output["risk_margin_se_ratio"] = selected_margin
    output["nu_signal_noise_ratio"] = ratio
    output["guardrail_empty_fallback"] = empty
    output["selected_action_guardrail_pass"] = safe[
        np.arange(len(states)), selected_index
    ]
    audit = {
        "state_count": len(output),
        "training_event_count": len(facts),
        "n_min_counts": {
            str(value): int((selected_n == int(value)).sum())
            for value in protocol["n_min_candidates"]
        },
        "nu_counts": {
            str(value): int((selected_nu == int(value)).sum())
            for value in protocol["nu_candidates"]
        },
        "guardrail_empty_state_count": int(empty.sum()),
        "support_level_counts": {
            str(value): int((selected_levels == int(value)).sum())
            for value in sorted(set(selected_levels.tolist()))
        },
        "temporal_cluster": "UTC_calendar_day",
        "risk_standard_error": "standard_error_of_nonempty_daily_block_means",
    }
    return output, audit


def predict_local_clara(facts: pd.DataFrame, decisions: pd.DataFrame) -> np.ndarray:
    frame = facts.copy()
    frame["_row_order"] = np.arange(len(frame), dtype=np.int64)
    mapped = frame.merge(
        decisions[list(STATE_FIELDS) + ["selected_action"]],
        on=list(STATE_FIELDS),
        how="left",
        validate="many_to_one",
        sort=False,
    ).sort_values("_row_order", kind="mergesort")
    if mapped["selected_action"].isna().any():
        missing = mapped.loc[mapped["selected_action"].isna(), list(STATE_FIELDS)]
        raise RuntimeError(
            f"CLARA Local 最终状态映射缺失，样例={missing.head(3).to_dict('records')}"
        )
    return mapped["selected_action"].astype(str).to_numpy()


def _state_context_lookup(
    facts: pd.DataFrame,
    contracts: FrozenContracts,
) -> tuple[dict[tuple[Any, ...], np.ndarray], tuple[str, ...]]:
    states = _state_universe(facts)
    states.insert(
        0,
        "event_id",
        [f"state-{index:06d}" for index in range(len(states))],
    )
    encoder = FrozenStateEncoder(contracts)
    encoded = encoder.transform(states)
    features = tuple(encoder.feature_names)
    lookup: dict[tuple[Any, ...], np.ndarray] = {}
    for index, row in states.iterrows():
        key = tuple(row[field] for field in STATE_FIELDS)
        lookup[key] = encoded.loc[index, list(features)].to_numpy(dtype=np.float64)
    return lookup, features


def run_linucb_compact(
    *,
    final_facts: pd.DataFrame,
    contracts: FrozenContracts,
    exploration_alpha: float,
    l2_regularization: float,
    adaptation_facts: pd.DataFrame | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    parts: list[pd.DataFrame] = []
    if adaptation_facts is not None:
        adaptation = adaptation_facts.copy()
        adaptation["_evaluation_row"] = False
        parts.append(adaptation)
    final = final_facts.copy()
    final["_evaluation_row"] = True
    final["_evaluation_order"] = np.arange(len(final), dtype=np.int64)
    parts.append(final)
    frame = pd.concat(parts, ignore_index=True, sort=False)
    frame["issue_timestamp"] = pd.to_datetime(frame["issue_timestamp"], errors="raise")
    frame["label_timestamp"] = pd.to_datetime(frame["label_timestamp"], errors="raise")
    frame["label_available_timestamp"] = pd.to_datetime(
        frame["label_available_timestamp"], errors="raise"
    )
    frame = frame.sort_values(
        ["issue_timestamp", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    context_lookup, feature_names = _state_context_lookup(frame, contracts)
    dimension = len(feature_names)
    inverse = {
        action: np.eye(dimension, dtype=np.float64) / float(l2_regularization)
        for action in ACTIONS
    }
    vector = {action: np.zeros(dimension, dtype=np.float64) for action in ACTIONS}
    updates = {action: 0 for action in ACTIONS}
    pending: list[tuple[int, int, str, int, str, np.ndarray, float]] = []
    selected_output = np.empty(len(final_facts), dtype=object)
    feedback_count = 0
    future_violation_count = 0

    def ns_value(value: Any) -> int:
        return int(pd.Timestamp(value).to_datetime64().astype("datetime64[ns]").astype(np.int64))

    def apply_matured(issue_ns: int) -> None:
        nonlocal feedback_count, future_violation_count
        while pending and pending[0][0] <= issue_ns:
            available_ns, label_ns, _, _, action, context, loss = heappop(pending)
            if label_ns >= issue_ns:
                future_violation_count += 1
                raise RuntimeError("LinUCB 检测到非严格成熟反馈")
            projected = inverse[action] @ context
            denominator = 1.0 + float(context @ projected)
            if denominator <= 0.0 or not np.isfinite(denominator):
                raise RuntimeError("LinUCB 逆矩阵更新分母非法")
            inverse[action] = inverse[action] - np.outer(projected, projected) / denominator
            vector[action] = vector[action] + context * float(loss)
            updates[action] += 1
            feedback_count += 1

    for issue, batch in frame.groupby("issue_timestamp", sort=True, dropna=False):
        issue_ns = ns_value(issue)
        apply_matured(issue_ns)
        batch_records = batch.to_dict(orient="records")
        staged: list[tuple[int, int, str, int, str, np.ndarray, float]] = []
        for row in batch_records:
            state_key = tuple(row[field] for field in STATE_FIELDS)
            context = context_lookup[state_key]
            scores: list[float] = []
            for action in ACTIONS:
                theta = inverse[action] @ vector[action]
                mean_cost = float(theta @ context)
                variance = max(float(context @ inverse[action] @ context), 0.0)
                scores.append(
                    mean_cost - float(exploration_alpha) * math.sqrt(variance)
                )
            minimum = min(scores)
            selected_index = next(
                index
                for index, score in enumerate(scores)
                if score <= minimum + 1e-12
            )
            selected_action = ACTIONS[selected_index]
            if bool(row["_evaluation_row"]):
                selected_output[int(row["_evaluation_order"])] = selected_action
            staged.append(
                (
                    ns_value(row["label_available_timestamp"]),
                    ns_value(row["label_timestamp"]),
                    str(row["event_id"]),
                    selected_index,
                    selected_action,
                    context.copy(),
                    float(row[f"{selected_action}__errf"]),
                )
            )
        for item in staged:
            heappush(pending, item)
    if pd.isna(pd.Series(selected_output)).any():
        raise RuntimeError("LinUCB 最终事件存在缺失决策")
    audit = {
        "event_count": len(frame),
        "adaptation_event_count": 0 if adaptation_facts is None else len(adaptation_facts),
        "final_event_count": len(final_facts),
        "feedback_count_before_terminal_drain": feedback_count,
        "pending_feedback_count": len(pending),
        "final_feedback_drain": False,
        "future_feedback_violation_count": future_violation_count,
        "within_issue_feedback_use_count": 0,
        "action_update_count": updates,
        "context_dimension": dimension,
        "exploration_alpha": float(exploration_alpha),
        "l2_regularization": float(l2_regularization),
    }
    return selected_output.astype(str), audit


def valid_conformal_base(base_predictions: pd.DataFrame) -> pd.DataFrame:
    frame = base_predictions.copy()
    calibration = frame[frame["split"].astype(str).eq("calibration")].copy()
    test = frame[
        frame["split"].astype(str).eq("test")
        & frame["label_available_timestamp"].notna()
    ].copy()
    if calibration.empty or test.empty:
        raise RuntimeError("TunedSingleConformal 校准或测试数据为空")
    return pd.concat([calibration, test], ignore_index=True)


def run_conformal_method(
    *,
    base_predictions: pd.DataFrame,
    contracts: FrozenContracts,
    dataset_id: str,
    farm: str,
    predictor: str,
    seed: int,
    target_coverage: float,
    family: str,
    configuration: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = run_conformal_configuration(
        contracts=contracts,
        base_predictions=valid_conformal_base(base_predictions),
        dataset_id=dataset_id,
        zone_or_farm=farm,
        predictor=predictor,
        seed=int(seed),
        target_coverage=float(target_coverage),
        family=family,
        configuration=configuration,
        drain_final_feedback=False,
    )
    decisions = pd.DataFrame([item.as_record() for item in result.decisions])
    if (
        int(result.audit["future_feedback_violation_count"]) != 0
        or int(result.audit["within_issue_feedback_use_count"]) != 0
    ):
        raise RuntimeError("TunedSingleConformal 因果审计失败")
    return decisions, result.audit


def _metric_arrays_from_endpoints(
    facts: pd.DataFrame,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    theta: ErrfTheta,
) -> dict[str, np.ndarray]:
    target = facts["target"].to_numpy(dtype=np.float64)
    schedule = facts["base_center"].to_numpy(dtype=np.float64)
    cadence = facts["nominal_cadence_minutes"].to_numpy(dtype=np.float64)
    scale = cadence / 60.0
    reserve_up = scale * np.maximum(upper - schedule, 0.0) * float(theta.pi_plus)
    reserve_down = scale * np.maximum(schedule - lower, 0.0) * float(theta.pi_minus)
    miss_upper = scale * np.maximum(target - upper, 0.0) * float(theta.kappa_plus)
    miss_lower = scale * np.maximum(lower - target, 0.0) * float(theta.kappa_minus)
    return {
        "lower": lower,
        "upper": upper,
        "width": upper - lower,
        "covered": (lower <= target) & (target <= upper),
        "reserve": reserve_up + reserve_down,
        "miss": miss_upper + miss_lower,
        "errf": reserve_up + reserve_down + miss_upper + miss_lower,
    }


def endpoint_metrics_for_actions(
    facts: pd.DataFrame,
    selected_actions: Sequence[str],
    *,
    theta: ErrfTheta,
) -> dict[str, np.ndarray]:
    selected = np.asarray(selected_actions, dtype=str)
    if len(selected) != len(facts) or set(selected) - set(ACTIONS):
        raise ValueError("所选动作数量或动作集合失配")
    lower = np.empty(len(facts), dtype=np.float64)
    upper = np.empty(len(facts), dtype=np.float64)
    for action in ACTIONS:
        mask = selected == action
        lower[mask] = facts.loc[mask, f"{action}__candidate_lower"].to_numpy(dtype=float)
        upper[mask] = facts.loc[mask, f"{action}__candidate_upper"].to_numpy(dtype=float)
    return _metric_arrays_from_endpoints(facts, lower, upper, theta=theta)


def endpoint_metrics_for_ensemble(
    facts: pd.DataFrame,
    *,
    theta: ErrfTheta,
) -> dict[str, np.ndarray]:
    lower = np.mean(
        np.column_stack(
            [facts[f"{action}__candidate_lower"].to_numpy(dtype=float) for action in ACTIONS]
        ),
        axis=1,
    )
    upper = np.mean(
        np.column_stack(
            [facts[f"{action}__candidate_upper"].to_numpy(dtype=float) for action in ACTIONS]
        ),
        axis=1,
    )
    return _metric_arrays_from_endpoints(facts, lower, upper, theta=theta)


def endpoint_metrics_for_direct_interval(
    facts: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    theta: ErrfTheta,
) -> dict[str, np.ndarray]:
    mapped = facts[["event_id"]].copy()
    mapped["_row_order"] = np.arange(len(mapped), dtype=np.int64)
    mapped = mapped.merge(
        decisions[["event_id", "selected_lower", "selected_upper"]],
        on="event_id",
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values("_row_order", kind="mergesort")
    if mapped[["selected_lower", "selected_upper"]].isna().any().any():
        raise RuntimeError("直接区间方法无法覆盖完整最终事件")
    return _metric_arrays_from_endpoints(
        facts,
        mapped["selected_lower"].to_numpy(dtype=float),
        mapped["selected_upper"].to_numpy(dtype=float),
        theta=theta,
    )


def rolling_reliability(
    covered: np.ndarray,
    *,
    target_coverage: float,
    cadence_minutes: float,
    window_hours: float = 168.0,
) -> tuple[float, float, float]:
    values = np.asarray(covered, dtype=np.float64)
    window = int(math.ceil(float(window_hours) * 60.0 / float(cadence_minutes)))
    if len(values) < window:
        return math.nan, math.nan, math.nan
    rolling = np.convolve(values, np.ones(window), mode="valid") / float(window)
    gaps = rolling - float(target_coverage)
    tolerance = 1.96 * math.sqrt(
        float(target_coverage) * (1.0 - float(target_coverage)) / float(window)
    )
    return (
        float(np.mean(gaps > tolerance)),
        float(np.mean(gaps < -tolerance)),
        float(np.mean(np.abs(gaps))),
    )


def summarize_method(
    facts: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    *,
    method: str,
    selected_actions: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = facts[
        [
            "event_id",
            "zone_or_farm",
            "predictor",
            "seed",
            "horizon_steps",
            "target_coverage",
            "issue_timestamp",
            "ramp_state",
            "nominal_cadence_minutes",
        ]
    ].copy()
    for name in ("errf", "reserve", "miss", "covered", "width"):
        frame[name] = np.asarray(arrays[name])
    if selected_actions is not None:
        frame["selected_action"] = np.asarray(selected_actions, dtype=str)
    else:
        frame["selected_action"] = pd.NA
    frame["method"] = str(method)
    frame["issue_timestamp"] = pd.to_datetime(frame["issue_timestamp"], errors="raise")
    cell_fields = [
        "zone_or_farm",
        "predictor",
        "seed",
        "horizon_steps",
        "target_coverage",
        "method",
    ]
    metric_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    for regime in ("overall", "ordinary", "ramp"):
        part = frame if regime == "overall" else frame[frame["ramp_state"].astype(str).eq(regime)]
        for key, group in part.groupby(cell_fields, sort=True, dropna=False):
            ordered = group.sort_values(["issue_timestamp", "event_id"], kind="mergesort")
            cadence_values = ordered["nominal_cadence_minutes"].astype(float).unique()
            if len(cadence_values) != 1:
                raise RuntimeError("单元格名义时间步长不唯一")
            towr, tuwr, ard = rolling_reliability(
                ordered["covered"].to_numpy(dtype=bool),
                target_coverage=float(key[4]),
                cadence_minutes=float(cadence_values[0]),
            )
            metric_rows.append(
                {
                    **dict(zip(cell_fields, key)),
                    "regime": regime,
                    "event_count": len(group),
                    "mean_errf": float(group["errf"].mean()),
                    "mean_reserve": float(group["reserve"].mean()),
                    "mean_miss": float(group["miss"].mean()),
                    "empirical_coverage": float(group["covered"].mean()),
                    "coverage_gap": float(group["covered"].mean()) - float(key[4]),
                    "average_width": float(group["width"].mean()),
                    "TOWR": towr,
                    "TUWR": tuwr,
                    "ARD": ard,
                }
            )
            counts = group["selected_action"].value_counts(dropna=False)
            for action, count in counts.items():
                action_rows.append(
                    {
                        **dict(zip(cell_fields, key)),
                        "regime": regime,
                        "selected_action": None if pd.isna(action) else str(action),
                        "event_count": int(count),
                    }
                )
        utc_issue = part["issue_timestamp"]
        if utc_issue.dt.tz is None:
            utc_issue = utc_issue.dt.tz_localize("Asia/Shanghai")
        utc_issue = utc_issue.dt.tz_convert("UTC")
        for block_hours in (24, 168):
            working = part.copy()
            working["block_hours"] = int(block_hours)
            working["block_start_utc"] = utc_issue.dt.floor(f"{int(block_hours)}h")
            grouping = [
                "zone_or_farm",
                "seed",
                "method",
                "block_hours",
                "block_start_utc",
            ]
            aggregated = (
                working.groupby(grouping, sort=True, dropna=False)
                .agg(
                    event_count=("event_id", "size"),
                    errf_sum=("errf", "sum"),
                    reserve_sum=("reserve", "sum"),
                    miss_sum=("miss", "sum"),
                    covered_sum=("covered", "sum"),
                    width_sum=("width", "sum"),
                )
                .reset_index()
            )
            aggregated["regime"] = regime
            block_rows.extend(aggregated.to_dict(orient="records"))
    return pd.DataFrame(metric_rows), pd.DataFrame(action_rows), pd.DataFrame(block_rows)


def select_local_conformal_configuration(
    *,
    stream_summaries: pd.DataFrame,
    coverage_gap_lower: float,
    tuwr_upper: float,
) -> dict[str, Any]:
    required = {
        "configuration_id",
        "family",
        "configuration_json",
        "complexity_rank",
        "event_count",
        "errf_sum",
        "covered_sum",
        "coverage_target_sum",
        "tuwr_sum",
        "tuwr_count",
    }
    missing = sorted(required - set(stream_summaries.columns))
    if missing:
        raise ValueError(f"本地共形选择摘要缺少字段: {missing}")
    aggregate = (
        stream_summaries.groupby(
            ["configuration_id", "family", "configuration_json", "complexity_rank"],
            sort=True,
            as_index=False,
        )[
            [
                "event_count",
                "errf_sum",
                "covered_sum",
                "coverage_target_sum",
                "tuwr_sum",
                "tuwr_count",
            ]
        ]
        .sum()
    )
    aggregate["mean_errf"] = aggregate["errf_sum"] / aggregate["event_count"]
    aggregate["coverage_gap"] = (
        aggregate["covered_sum"] - aggregate["coverage_target_sum"]
    ) / aggregate["event_count"]
    aggregate["mean_tuwr"] = aggregate["tuwr_sum"] / aggregate["tuwr_count"]
    feasible = aggregate[
        aggregate["coverage_gap"].ge(float(coverage_gap_lower))
        & aggregate["mean_tuwr"].le(float(tuwr_upper))
    ].copy()
    if feasible.empty:
        raise RuntimeError("本地共形 19 项配置没有满足冻结可靠性门的候选")
    chosen = feasible.sort_values(
        ["mean_errf", "complexity_rank", "configuration_id"], kind="mergesort"
    ).iloc[0]
    return {
        "configuration_id": str(chosen["configuration_id"]),
        "family": str(chosen["family"]),
        "configuration": json.loads(str(chosen["configuration_json"])),
        "complexity_rank": int(chosen["complexity_rank"]),
        "mean_errf": float(chosen["mean_errf"]),
        "coverage_gap": float(chosen["coverage_gap"]),
        "mean_tuwr": float(chosen["mean_tuwr"]),
        "feasible_configuration_count": len(feasible),
        "candidate_configuration_count": len(aggregate),
    }


def conformal_configuration_records(contracts: FrozenContracts) -> list[dict[str, Any]]:
    return [
        {
            "family": item.family,
            "configuration": item.configuration,
            "configuration_json": json.dumps(item.configuration, sort_keys=True),
            "configuration_id": item.configuration_id,
            "complexity_rank": item.complexity_rank,
        }
        for item in conformal_grid(contracts)
    ]
