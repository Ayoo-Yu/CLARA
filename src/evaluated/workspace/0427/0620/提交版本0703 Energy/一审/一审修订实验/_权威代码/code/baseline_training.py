from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier

from baseline_common import FrozenStateEncoder, decisions_from_selected_actions
from clara_event_contract import FrozenContracts, canonical_json_sha256


TRAINING_REQUIRED_FIELDS = (
    "event_id",
    "origin",
    "zone_or_farm",
    "issue_timestamp",
    "theta_id",
    "predictor",
    "horizon_group",
    "target_coverage",
    "ramp_state",
    "rolling_state",
    "raw_width_state",
    "action",
    "errf",
    "covered",
)

STATE_GROUPED_GBR_TARGET_DECIMALS = 12


@dataclass(frozen=True)
class ValidatedTrainingObservations:
    events: pd.DataFrame
    observations: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"训练观测缺少字段: {missing}")


def _frozen_grid(contracts: FrozenContracts, baseline_id: str) -> dict[str, tuple[Any, ...]]:
    entries = contracts.baseline_registry["mandatory_contextual_selectors"]
    matches = [entry for entry in entries if entry["id"] == baseline_id]
    if len(matches) != 1:
        raise RuntimeError(f"冻结基线注册表中{baseline_id}定义数量异常")
    return {key: tuple(values) for key, values in matches[0]["grid"].items()}


def validate_frozen_selector_config(
    *,
    contracts: FrozenContracts,
    baseline_id: str,
    config: dict[str, Any],
) -> None:
    grid = _frozen_grid(contracts, baseline_id)
    if set(config) != set(grid):
        raise ValueError(f"{baseline_id}配置字段未严格匹配冻结网格")
    outside = {
        key: value
        for key, value in config.items()
        if value not in grid[key]
    }
    if outside:
        raise ValueError(f"{baseline_id}配置不在冻结网格中: {outside}")


def validate_training_observations(
    *,
    contracts: FrozenContracts,
    observations: pd.DataFrame,
    allowed_zones: Iterable[str] | None = None,
    forbidden_zones: Iterable[str] = (),
) -> ValidatedTrainingObservations:
    _require_columns(observations, TRAINING_REQUIRED_FIELDS)
    frame = observations.copy()
    if frame.loc[:, ["event_id", "action"]].duplicated().any():
        raise ValueError("训练观测存在重复event_id与action")
    frame["event_id"] = frame["event_id"].astype(str)
    frame["zone_or_farm"] = frame["zone_or_farm"].astype(str)
    if set(frame["origin"].astype(str)) != {"source"}:
        raise ValueError("训练观测必须全部来自source")
    forbidden = set(str(zone) for zone in forbidden_zones)
    leaked = sorted(set(frame["zone_or_farm"]) & forbidden)
    if leaked:
        raise ValueError(f"训练观测包含外层持出区: {leaked}")
    if allowed_zones is not None:
        allowed = set(str(zone) for zone in allowed_zones)
        unknown = sorted(set(frame["zone_or_farm"]) - allowed)
        if unknown:
            raise ValueError(f"训练观测包含允许源区之外的区域: {unknown}")
        missing_zones = sorted(allowed - set(frame["zone_or_farm"]))
        if missing_zones:
            raise ValueError(f"训练观测缺少声明的源区: {missing_zones}")
    expected_actions = set(contracts.actions)
    action_errors: list[str] = []
    for event_id, group in frame.groupby("event_id", sort=False):
        actions = list(group["action"].astype(str))
        if len(actions) != len(contracts.actions) or set(actions) != expected_actions:
            action_errors.append(str(event_id))
    if action_errors:
        raise ValueError(f"训练观测四动作不完整: {action_errors[:10]}")
    frame["errf"] = pd.to_numeric(frame["errf"], errors="coerce")
    if not np.isfinite(frame["errf"].to_numpy(dtype=float)).all() or (frame["errf"] < 0.0).any():
        raise ValueError("训练观测ERRF必须为有限非负值")
    frame["covered"] = pd.to_numeric(frame["covered"], errors="coerce")
    if not frame["covered"].isin((0.0, 1.0)).all():
        raise ValueError("训练观测covered必须为零或一")
    state_fields = tuple(contracts.baseline_registry["common_information_contract"]["release_time_state_fields"])
    event_fields = (
        "event_id",
        "zone_or_farm",
        "issue_timestamp",
        "theta_id",
        *state_fields,
    )
    event_consistency = frame.groupby("event_id", sort=False)[list(event_fields[1:])].nunique(dropna=False)
    if (event_consistency > 1).any().any():
        raise ValueError("同一event_id的发布时状态或元数据不一致")
    events = frame.loc[:, event_fields].drop_duplicates("event_id").copy()
    FrozenStateEncoder(contracts).validate(events)
    action_rank = {action: index for index, action in enumerate(contracts.actions)}
    frame["_action_rank"] = frame["action"].map(action_rank)
    frame = (
        frame.sort_values(["event_id", "_action_rank"], kind="mergesort")
        .drop(columns="_action_rank")
        .reset_index(drop=True)
    )
    events = events.sort_values("event_id", kind="mergesort").reset_index(drop=True)
    return ValidatedTrainingObservations(events=events, observations=frame)


def counterfactual_best_actions(
    *,
    contracts: FrozenContracts,
    validated: ValidatedTrainingObservations,
) -> pd.Series:
    action_rank = {action: index for index, action in enumerate(contracts.actions)}
    frame = validated.observations.loc[:, ["event_id", "action", "errf"]].copy()
    frame["action_rank"] = frame["action"].map(action_rank)
    selected = (
        frame.sort_values(["event_id", "errf", "action_rank"], kind="mergesort")
        .groupby("event_id", sort=False)
        .first()["action"]
    )
    return selected.reindex(validated.events["event_id"]).astype(str)


def fit_state_grouped_gradient_boosting(
    *,
    feature_matrix: np.ndarray,
    target_sum: np.ndarray,
    sample_count: np.ndarray,
    config: dict[str, Any],
) -> GradientBoostingRegressor:
    features = np.asarray(feature_matrix, dtype=np.float64)
    sums = np.asarray(target_sum, dtype=np.float64)
    counts = np.asarray(sample_count, dtype=np.float64)
    if features.ndim != 2 or len(features) != len(sums) or len(sums) != len(counts):
        raise ValueError("状态汇总GBR输入尺寸不一致")
    if not np.isfinite(features).all() or not np.isfinite(sums).all() or not np.isfinite(counts).all():
        raise ValueError("状态汇总GBR输入含非有限值")
    if np.any(sums < 0.0) or np.any(counts <= 0.0):
        raise ValueError("状态汇总GBR损失总和或样本计数非法")
    if not np.array_equal(counts, np.floor(counts)):
        raise ValueError("状态汇总GBR样本计数必须为整数")
    unique_features, inverse = np.unique(features, axis=0, return_inverse=True)
    unique_count = np.bincount(inverse, weights=counts, minlength=len(unique_features))
    unique_sum = np.bincount(inverse, weights=sums, minlength=len(unique_features))
    if np.any(unique_count <= 0.0):
        raise RuntimeError("状态汇总GBR出现空状态")
    # 固定精度消除同一状态分批归约时的浮点顺序依赖。
    state_target = np.round(
        unique_sum / unique_count,
        decimals=STATE_GROUPED_GBR_TARGET_DECIMALS,
    )
    minimum_leaf = int(config["min_samples_leaf"])
    total_count = float(unique_count.sum())
    if total_count < 2.0 * minimum_leaf:
        model_minimum_leaf = len(unique_features)
        minimum_weight_fraction = 0.0
    else:
        model_minimum_leaf = 1
        minimum_weight_fraction = np.nextafter(float(minimum_leaf) / total_count, 0.0)
    model = GradientBoostingRegressor(
        n_estimators=int(config["n_estimators"]),
        learning_rate=float(config["learning_rate"]),
        max_depth=int(config["max_depth"]),
        min_samples_leaf=model_minimum_leaf,
        min_weight_fraction_leaf=minimum_weight_fraction,
        random_state=0,
    )
    model.fit(unique_features, state_target, sample_weight=unique_count)
    return model


@dataclass
class SupervisedActionSelector:
    contracts: FrozenContracts
    baseline_id: str
    config: dict[str, Any]
    encoder: FrozenStateEncoder
    model_kind: str
    models: dict[str, Any]
    fit_zones: tuple[str, ...]
    config_id: str

    def predict_actions(self, events: pd.DataFrame) -> tuple[dict[str, str], dict[str, float]]:
        if "event_id" not in events.columns:
            raise ValueError("预测事件缺少event_id")
        ordered_events = events.copy()
        if ordered_events["event_id"].isna().any():
            raise ValueError("预测事件event_id必须非空且唯一")
        ordered_events["event_id"] = ordered_events["event_id"].astype(str)
        if ordered_events["event_id"].duplicated().any():
            raise ValueError("预测事件event_id必须非空且唯一")
        ordered_events = ordered_events.sort_values("event_id", kind="mergesort").reset_index(drop=True)
        encoded = self.encoder.transform(ordered_events)
        feature_frame = encoded.loc[:, self.encoder.feature_names]
        event_ids = encoded["event_id"].astype(str).tolist()
        if self.model_kind == "cart":
            model = self.models["classifier"]
            actions = [str(value) for value in model.predict(feature_frame.to_numpy(dtype=float))]
            probabilities = model.predict_proba(feature_frame.to_numpy(dtype=float))
            class_index = {str(value): index for index, value in enumerate(model.classes_)}
            scores = {
                event_id: float(probabilities[row_index, class_index[action]])
                for row_index, (event_id, action) in enumerate(zip(event_ids, actions))
            }
            return dict(zip(event_ids, actions)), scores
        predictions = {
            action: np.asarray(model.predict(feature_frame.to_numpy(dtype=float)), dtype=float)
            for action, model in self.models.items()
        }
        action_rank = {action: index for index, action in enumerate(self.contracts.actions)}
        selected_actions: dict[str, str] = {}
        scores: dict[str, float] = {}
        for row_index, event_id in enumerate(event_ids):
            action = min(
                self.contracts.actions,
                key=lambda item: (float(predictions[item][row_index]), action_rank[item]),
            )
            selected_actions[event_id] = action
            scores[event_id] = float(predictions[action][row_index])
        return selected_actions, scores

    def predict_decisions(
        self,
        *,
        events: pd.DataFrame,
        candidates: pd.DataFrame,
    ):
        actions, scores = self.predict_actions(events)
        return decisions_from_selected_actions(
            contracts=self.contracts,
            events=events,
            candidates=candidates,
            baseline_id=self.baseline_id,
            selected_actions=actions,
            decision_scores=scores,
            decision_reason=f"{self.baseline_id}:{self.config_id}",
        )


def fit_cart_best_action(
    *,
    contracts: FrozenContracts,
    observations: pd.DataFrame,
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
    validated = validate_training_observations(
        contracts=contracts,
        observations=observations,
        allowed_zones=zones,
        forbidden_zones=(outer_heldout_zone,),
    )
    encoder = FrozenStateEncoder(contracts)
    encoded = encoder.transform(validated.events)
    labels = counterfactual_best_actions(contracts=contracts, validated=validated).to_numpy()
    model = DecisionTreeClassifier(
        max_depth=config["max_depth"],
        min_samples_leaf=int(config["min_samples_leaf"]),
        class_weight=config["class_weight"],
        random_state=0,
    )
    model.fit(encoded.loc[:, encoder.feature_names].to_numpy(dtype=float), labels)
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


def fit_per_action_risk(
    *,
    contracts: FrozenContracts,
    observations: pd.DataFrame,
    baseline_id: str,
    config: dict[str, Any],
    training_zones: Iterable[str],
    outer_heldout_zone: str,
) -> SupervisedActionSelector:
    if baseline_id not in {"RidgePerActionRisk", "GBRPerActionRisk"}:
        raise ValueError(f"未知逐动作风险基线: {baseline_id}")
    validate_frozen_selector_config(
        contracts=contracts,
        baseline_id=baseline_id,
        config=config,
    )
    zones = tuple(sorted(str(zone) for zone in training_zones))
    validated = validate_training_observations(
        contracts=contracts,
        observations=observations,
        allowed_zones=zones,
        forbidden_zones=(outer_heldout_zone,),
    )
    encoder = FrozenStateEncoder(contracts)
    encoded = encoder.transform(validated.events).set_index("event_id")
    models: dict[str, Any] = {}
    normalized_config: dict[str, Any]
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
        action_rows = validated.observations[validated.observations["action"] == action].set_index("event_id")
        ordered = action_rows.reindex(encoded.index)
        if ordered["errf"].isna().any():
            raise RuntimeError(f"动作{action}训练损失缺失")
        if baseline_id == "RidgePerActionRisk":
            model = Ridge(alpha=normalized_config["alpha"], fit_intercept=False)
            model.fit(
                encoded.loc[:, encoder.feature_names].to_numpy(dtype=float),
                ordered["errf"].to_numpy(dtype=float),
            )
        else:
            model = fit_state_grouped_gradient_boosting(
                feature_matrix=encoded.loc[:, encoder.feature_names].to_numpy(dtype=float),
                target_sum=ordered["errf"].to_numpy(dtype=float),
                sample_count=np.ones(len(ordered), dtype=np.float64),
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
