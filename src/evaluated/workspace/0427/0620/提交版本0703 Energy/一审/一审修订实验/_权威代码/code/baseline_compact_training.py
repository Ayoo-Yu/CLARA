from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier

from baseline_common import FrozenStateEncoder
from baseline_training import (
    SupervisedActionSelector,
    fit_state_grouped_gradient_boosting,
    validate_frozen_selector_config,
)
from clara_event_contract import FrozenContracts, canonical_json_sha256


COMPACT_STATE_FIELDS = (
    "predictor",
    "horizon_group",
    "target_coverage",
    "ramp_state",
    "rolling_state",
    "raw_width_state",
)


def compact_metric_columns(contracts: FrozenContracts) -> tuple[str, ...]:
    columns: list[str] = ["event_count"]
    for action in contracts.actions:
        columns.extend(
            [
                f"{action}__errf_sum",
                f"{action}__covered_sum",
                f"{action}__tuwr_sum",
                f"{action}__tuwr_count",
                f"{action}__best_count",
            ]
        )
    return tuple(columns)


def validate_compact_state_statistics(
    *,
    contracts: FrozenContracts,
    statistics: pd.DataFrame,
    allowed_zones: Iterable[str],
    outer_heldout_zone: str,
) -> pd.DataFrame:
    required = {"source_zone", *COMPACT_STATE_FIELDS, *compact_metric_columns(contracts)}
    missing = sorted(required - set(statistics.columns))
    if missing:
        raise ValueError(f"压缩状态统计缺少字段: {missing}")
    frame = statistics.loc[:, ["source_zone", *COMPACT_STATE_FIELDS, *compact_metric_columns(contracts)]].copy()
    frame["source_zone"] = frame["source_zone"].astype(str)
    zones = tuple(sorted(str(zone) for zone in allowed_zones))
    if str(outer_heldout_zone) in set(frame["source_zone"]):
        raise ValueError("压缩状态统计包含外层持出区")
    observed_zones = set(frame["source_zone"])
    if observed_zones != set(zones):
        raise ValueError(
            f"压缩状态统计源区集合失配: observed={sorted(observed_zones)}, expected={list(zones)}"
        )
    key = ["source_zone", *COMPACT_STATE_FIELDS]
    if frame[key].duplicated().any():
        raise ValueError("压缩状态统计存在重复源区状态键")
    numeric_columns = compact_metric_columns(contracts)
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    numeric = frame.loc[:, numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("压缩状态统计含非有限指标")
    integer_columns = ["event_count"]
    for action in contracts.actions:
        integer_columns.extend(
            [
                f"{action}__covered_sum",
                f"{action}__tuwr_count",
                f"{action}__best_count",
            ]
        )
    for column in integer_columns:
        values = frame[column].to_numpy(dtype=float)
        if not np.array_equal(values, np.floor(values)):
            raise ValueError(f"压缩状态统计计数字段含非整数: {column}")
        frame[column] = values.astype(np.int64)
    if (frame["event_count"] <= 0).any():
        raise ValueError("压缩状态统计事件计数必须为正")
    event_count = frame["event_count"].to_numpy(dtype=np.int64)
    best_total = np.zeros(len(frame), dtype=np.int64)
    for action in contracts.actions:
        errf_sum = frame[f"{action}__errf_sum"].to_numpy(dtype=float)
        covered_sum = frame[f"{action}__covered_sum"].to_numpy(dtype=np.int64)
        tuwr_sum = frame[f"{action}__tuwr_sum"].to_numpy(dtype=float)
        tuwr_count = frame[f"{action}__tuwr_count"].to_numpy(dtype=np.int64)
        best_count = frame[f"{action}__best_count"].to_numpy(dtype=np.int64)
        if np.any(errf_sum < 0.0):
            raise ValueError("压缩状态统计ERRF总和必须非负")
        if np.any(covered_sum < 0) or np.any(covered_sum > event_count):
            raise ValueError("压缩状态统计covered总和越界")
        if np.any(tuwr_count < 0) or np.any(tuwr_count > event_count):
            raise ValueError("压缩状态统计TUWR计数越界")
        if np.any(tuwr_sum < 0.0) or np.any(tuwr_sum > tuwr_count):
            raise ValueError("压缩状态统计TUWR总和越界")
        if np.any(best_count < 0) or np.any(best_count > event_count):
            raise ValueError("压缩状态统计最优动作计数越界")
        best_total += best_count
    if not np.array_equal(best_total, event_count):
        raise ValueError("压缩状态统计最优动作计数不闭合")
    events = frame.loc[:, list(COMPACT_STATE_FIELDS)].copy()
    events.insert(0, "event_id", [f"compact-validate-{index:06d}" for index in range(len(events))])
    FrozenStateEncoder(contracts).validate(events)
    zone_rank = {zone: index for index, zone in enumerate(zones)}
    frame["_zone_rank"] = frame["source_zone"].map(zone_rank)
    frame = (
        frame.sort_values(["_zone_rank", *COMPACT_STATE_FIELDS], kind="mergesort")
        .drop(columns="_zone_rank")
        .reset_index(drop=True)
    )
    return frame


def collapse_compact_training_states(
    *,
    contracts: FrozenContracts,
    statistics: pd.DataFrame,
    training_zones: Iterable[str],
    outer_heldout_zone: str,
) -> pd.DataFrame:
    validated = validate_compact_state_statistics(
        contracts=contracts,
        statistics=statistics,
        allowed_zones=training_zones,
        outer_heldout_zone=outer_heldout_zone,
    )
    numeric_columns = list(compact_metric_columns(contracts))
    collapsed = (
        validated.groupby(list(COMPACT_STATE_FIELDS), sort=True, as_index=False)[numeric_columns]
        .sum()
        .reset_index(drop=True)
    )
    for column in ["event_count"] + [
        name
        for action in contracts.actions
        for name in (
            f"{action}__covered_sum",
            f"{action}__tuwr_count",
            f"{action}__best_count",
        )
    ]:
        collapsed[column] = collapsed[column].astype(np.int64)
    return collapsed


def _encoded_compact_states(
    *,
    contracts: FrozenContracts,
    collapsed: pd.DataFrame,
) -> tuple[FrozenStateEncoder, pd.DataFrame, np.ndarray]:
    events = collapsed.loc[:, list(COMPACT_STATE_FIELDS)].copy()
    events.insert(0, "event_id", [f"compact-state-{index:06d}" for index in range(len(events))])
    encoder = FrozenStateEncoder(contracts)
    encoded = encoder.transform(events)
    matrix = encoded.loc[:, encoder.feature_names].to_numpy(dtype=np.float64)
    return encoder, events, matrix


def _balanced_class_weight(
    *,
    class_counts: np.ndarray,
) -> np.ndarray:
    positive = class_counts > 0
    if not np.any(positive):
        raise ValueError("压缩CART训练没有任何类别")
    weights = np.zeros_like(class_counts, dtype=np.float64)
    total = float(class_counts.sum())
    class_count = int(positive.sum())
    weights[positive] = total / (class_count * class_counts[positive].astype(np.float64))
    return weights


def _capped_classification_expansion(
    *,
    feature_matrix: np.ndarray,
    class_counts_by_state: np.ndarray,
    class_labels: tuple[str, ...],
    min_samples_leaf: int,
    balanced: bool,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    if min_samples_leaf < 2:
        raise ValueError("压缩CART叶节点样本门必须至少为二")
    state_totals = class_counts_by_state.sum(axis=1).astype(np.int64)
    if np.any(state_totals <= 0):
        raise ValueError("压缩CART状态样本量必须为正")
    class_totals = class_counts_by_state.sum(axis=0).astype(np.int64)
    class_weights = (
        _balanced_class_weight(class_counts=class_totals)
        if balanced
        else np.ones(len(class_labels), dtype=np.float64)
    )
    state_repetitions = np.minimum(state_totals, int(min_samples_leaf)).astype(np.int64)
    expanded_state_indices = np.repeat(np.arange(len(state_totals), dtype=np.int64), state_repetitions)
    labels = np.empty(int(state_repetitions.sum()), dtype=object)
    sample_weights = np.empty(int(state_repetitions.sum()), dtype=np.float64)
    cursor = 0
    for state_index, repetitions in enumerate(state_repetitions):
        counts = class_counts_by_state[state_index].astype(np.int64)
        positive_classes = np.flatnonzero(counts > 0)
        if len(positive_classes) > int(repetitions):
            raise RuntimeError("压缩CART类别数超过状态压缩样本数")
        allocations = np.zeros(len(class_labels), dtype=np.int64)
        allocations[positive_classes] = 1
        allocations[positive_classes[0]] += int(repetitions) - len(positive_classes)
        for class_index in positive_classes:
            allocation = int(allocations[class_index])
            next_cursor = cursor + allocation
            labels[cursor:next_cursor] = class_labels[class_index]
            sample_weights[cursor:next_cursor] = (
                float(counts[class_index]) * float(class_weights[class_index]) / allocation
            )
            cursor = next_cursor
    if cursor != len(labels):
        raise RuntimeError("压缩CART展开游标未闭合")
    unique_sparse = sparse.csr_matrix(feature_matrix, dtype=np.float64)
    expanded_features = unique_sparse[expanded_state_indices]
    return expanded_features, labels, sample_weights


def fit_compact_cart_best_action(
    *,
    contracts: FrozenContracts,
    statistics: pd.DataFrame,
    config: dict[str, Any],
    training_zones: Iterable[str],
    outer_heldout_zone: str,
) -> SupervisedActionSelector:
    validate_frozen_selector_config(
        contracts=contracts,
        baseline_id="CARTBestAction",
        config=config,
    )
    zones = tuple(sorted(str(zone) for zone in training_zones))
    collapsed = collapse_compact_training_states(
        contracts=contracts,
        statistics=statistics,
        training_zones=zones,
        outer_heldout_zone=outer_heldout_zone,
    )
    encoder, _, feature_matrix = _encoded_compact_states(contracts=contracts, collapsed=collapsed)
    class_counts = np.column_stack(
        [collapsed[f"{action}__best_count"].to_numpy(dtype=np.int64) for action in contracts.actions]
    )
    features, labels, sample_weights = _capped_classification_expansion(
        feature_matrix=feature_matrix,
        class_counts_by_state=class_counts,
        class_labels=tuple(contracts.actions),
        min_samples_leaf=int(config["min_samples_leaf"]),
        balanced=config["class_weight"] == "balanced",
    )
    model = DecisionTreeClassifier(
        max_depth=config["max_depth"],
        min_samples_leaf=int(config["min_samples_leaf"]),
        class_weight=None,
        random_state=0,
    )
    model.fit(features, labels, sample_weight=sample_weights)
    normalized_config = {
        "max_depth": config["max_depth"],
        "min_samples_leaf": int(config["min_samples_leaf"]),
        "class_weight": config["class_weight"],
        "random_state": 0,
    }
    return SupervisedActionSelector(
        contracts=contracts,
        baseline_id="CARTBestAction",
        config=normalized_config,
        encoder=encoder,
        model_kind="cart",
        models={"classifier": model},
        fit_zones=zones,
        config_id=canonical_json_sha256({"baseline_id": "CARTBestAction", **normalized_config}),
    )


def fit_compact_per_action_risk(
    *,
    contracts: FrozenContracts,
    statistics: pd.DataFrame,
    baseline_id: str,
    config: dict[str, Any],
    training_zones: Iterable[str],
    outer_heldout_zone: str,
) -> SupervisedActionSelector:
    if baseline_id not in {"RidgePerActionRisk", "GBRPerActionRisk"}:
        raise ValueError(f"未知压缩逐动作风险基线: {baseline_id}")
    validate_frozen_selector_config(
        contracts=contracts,
        baseline_id=baseline_id,
        config=config,
    )
    zones = tuple(sorted(str(zone) for zone in training_zones))
    collapsed = collapse_compact_training_states(
        contracts=contracts,
        statistics=statistics,
        training_zones=zones,
        outer_heldout_zone=outer_heldout_zone,
    )
    encoder, _, feature_matrix = _encoded_compact_states(contracts=contracts, collapsed=collapsed)
    sample_count = collapsed["event_count"].to_numpy(dtype=np.float64)
    models: dict[str, Any] = {}
    if baseline_id == "RidgePerActionRisk":
        normalized_config = {"alpha": float(config["alpha"]), "fit_intercept": False}
    else:
        normalized_config = {
            "n_estimators": int(config["n_estimators"]),
            "learning_rate": float(config["learning_rate"]),
            "max_depth": int(config["max_depth"]),
            "min_samples_leaf": int(config["min_samples_leaf"]),
            "random_state": 0,
        }
    for action in contracts.actions:
        target = collapsed[f"{action}__errf_sum"].to_numpy(dtype=np.float64) / sample_count
        if baseline_id == "RidgePerActionRisk":
            model = Ridge(alpha=normalized_config["alpha"], fit_intercept=False)
            model.fit(feature_matrix, target, sample_weight=sample_count)
        else:
            model = fit_state_grouped_gradient_boosting(
                feature_matrix=feature_matrix,
                target_sum=collapsed[f"{action}__errf_sum"].to_numpy(dtype=np.float64),
                sample_count=sample_count,
                config=normalized_config,
            )
        models[action] = model
    return SupervisedActionSelector(
        contracts=contracts,
        baseline_id=baseline_id,
        config=normalized_config,
        encoder=encoder,
        model_kind="per_action_risk",
        models=models,
        fit_zones=zones,
        config_id=canonical_json_sha256({"baseline_id": baseline_id, **normalized_config}),
    )


def compact_state_events(statistics: pd.DataFrame) -> pd.DataFrame:
    required = set(COMPACT_STATE_FIELDS)
    missing = sorted(required - set(statistics.columns))
    if missing:
        raise ValueError(f"压缩验证状态缺少字段: {missing}")
    events = statistics.loc[:, list(COMPACT_STATE_FIELDS)].copy().reset_index(drop=True)
    events.insert(0, "event_id", [f"compact-score-{index:06d}" for index in range(len(events))])
    return events


def selected_action_metrics_from_compact_statistics(
    *,
    contracts: FrozenContracts,
    statistics: pd.DataFrame,
    selected_actions: Iterable[str],
    inner_validation_zone: str,
) -> dict[str, Any]:
    frame = statistics.reset_index(drop=True)
    actions = np.asarray([str(value) for value in selected_actions], dtype=object)
    if len(actions) != len(frame):
        raise ValueError("压缩验证选中动作数量失配")
    invalid = sorted(set(actions) - set(contracts.actions))
    if invalid:
        raise ValueError(f"压缩验证选中动作非法: {invalid}")
    event_count = frame["event_count"].to_numpy(dtype=np.int64)
    total_events = int(event_count.sum())
    if total_events <= 0:
        raise ValueError("压缩验证事件数必须为正")
    errf_total = 0.0
    covered_total = 0.0
    target_total = float(
        np.dot(event_count.astype(np.float64), frame["target_coverage"].to_numpy(dtype=np.float64))
    )
    tuwr_total = 0.0
    tuwr_count = 0
    for action in contracts.actions:
        mask = actions == action
        if not np.any(mask):
            continue
        errf_total += float(frame.loc[mask, f"{action}__errf_sum"].sum())
        covered_total += float(frame.loc[mask, f"{action}__covered_sum"].sum())
        tuwr_total += float(frame.loc[mask, f"{action}__tuwr_sum"].sum())
        tuwr_count += int(frame.loc[mask, f"{action}__tuwr_count"].sum())
    if tuwr_count <= 0:
        raise ValueError("压缩验证没有非缺失TUWR观测")
    return {
        "inner_validation_zone": str(inner_validation_zone),
        "mean_errf": errf_total / total_events,
        "mean_coverage_gap": (covered_total - target_total) / total_events,
        "mean_tuwr": tuwr_total / tuwr_count,
        "event_count": total_events,
        "tuwr_observation_count": tuwr_count,
    }
