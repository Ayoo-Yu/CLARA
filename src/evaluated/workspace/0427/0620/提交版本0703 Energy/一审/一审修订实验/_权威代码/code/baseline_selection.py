from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from baseline_training import ValidatedTrainingObservations
from clara_event_contract import FrozenContracts


@dataclass(frozen=True)
class ConfigurationSelection:
    selected_configuration_id: str
    feasible_pool_nonempty: bool
    no_feasible_fallback_used: bool
    primary_objective: float
    mean_coverage_gap: float
    mean_tuwr: float
    complexity_rank: int

    def as_record(self) -> dict[str, Any]:
        return {
            "selected_configuration_id": self.selected_configuration_id,
            "feasible_pool_nonempty": self.feasible_pool_nonempty,
            "no_feasible_fallback_used": self.no_feasible_fallback_used,
            "primary_objective": self.primary_objective,
            "mean_coverage_gap": self.mean_coverage_gap,
            "mean_tuwr": self.mean_tuwr,
            "complexity_rank": self.complexity_rank,
        }


def selected_action_zone_metrics(
    *,
    validated: ValidatedTrainingObservations,
    selected_actions: dict[str, str],
) -> pd.DataFrame:
    expected = set(validated.events["event_id"])
    if set(selected_actions) != expected:
        raise ValueError("选中动作映射与训练事件集合不一致")
    frame = validated.observations.copy()
    frame["selected_action"] = frame["event_id"].map(selected_actions)
    selected = frame[frame["action"] == frame["selected_action"]].copy()
    if len(selected) != len(validated.events):
        raise RuntimeError("选中动作损失连接不完整")
    selected["coverage_gap"] = selected["covered"].astype(float) - selected["target_coverage"].astype(float)
    rows: list[dict[str, Any]] = []
    for zone, group in selected.groupby("zone_or_farm", sort=True):
        tuwr_values = pd.to_numeric(group.get("tuwr_indicator"), errors="coerce") if "tuwr_indicator" in group else pd.Series(dtype=float)
        mean_tuwr = float(tuwr_values.dropna().mean()) if not tuwr_values.dropna().empty else float("inf")
        rows.append(
            {
                "inner_validation_zone": str(zone),
                "mean_errf": float(group["errf"].mean()),
                "mean_coverage_gap": float(group["coverage_gap"].mean()),
                "mean_tuwr": mean_tuwr,
                "event_count": len(group),
            }
        )
    return pd.DataFrame(rows)


def select_configuration(
    *,
    contracts: FrozenContracts,
    validation_scores: pd.DataFrame,
) -> ConfigurationSelection:
    required = {
        "configuration_id",
        "inner_validation_zone",
        "mean_errf",
        "mean_coverage_gap",
        "mean_tuwr",
        "complexity_rank",
    }
    missing = sorted(required - set(validation_scores.columns))
    if missing:
        raise ValueError(f"配置验证分数缺少字段: {missing}")
    if validation_scores.loc[:, ["configuration_id", "inner_validation_zone"]].duplicated().any():
        raise ValueError("配置与内层验证区记录重复")
    rows: list[dict[str, Any]] = []
    for configuration_id, group in validation_scores.groupby("configuration_id", sort=True):
        complexity_values = set(int(value) for value in group["complexity_rank"])
        if len(complexity_values) != 1:
            raise ValueError(f"配置{configuration_id}复杂度等级不一致")
        rows.append(
            {
                "configuration_id": str(configuration_id),
                "primary_objective": float(group["mean_errf"].mean()),
                "mean_coverage_gap": float(group["mean_coverage_gap"].mean()),
                "mean_tuwr": float(group["mean_tuwr"].mean()),
                "complexity_rank": next(iter(complexity_values)),
            }
        )
    aggregate = pd.DataFrame(rows)
    if aggregate.empty:
        raise ValueError("配置验证分数为空")
    numeric = aggregate.loc[:, ["primary_objective", "mean_coverage_gap", "mean_tuwr"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("配置验证分数含非有限值")
    constraints = contracts.baseline_registry["inner_validation"]["feasibility_constraints"]
    aggregate["feasible"] = (
        (aggregate["mean_coverage_gap"] >= float(constraints["mean_coverage_gap_minimum"]))
        & (aggregate["mean_tuwr"] <= float(constraints["tuwr_maximum"]))
    )
    feasible = aggregate[aggregate["feasible"]].copy()
    if not feasible.empty:
        selected = feasible.sort_values(
            ["primary_objective", "complexity_rank", "configuration_id"],
            kind="mergesort",
        ).iloc[0]
        fallback = False
    else:
        aggregate["coverage_shortfall"] = np.maximum(-aggregate["mean_coverage_gap"], 0.0)
        selected = aggregate.sort_values(
            ["primary_objective", "coverage_shortfall", "complexity_rank", "configuration_id"],
            kind="mergesort",
        ).iloc[0]
        fallback = True
    return ConfigurationSelection(
        selected_configuration_id=str(selected["configuration_id"]),
        feasible_pool_nonempty=not feasible.empty,
        no_feasible_fallback_used=fallback,
        primary_objective=float(selected["primary_objective"]),
        mean_coverage_gap=float(selected["mean_coverage_gap"]),
        mean_tuwr=float(selected["mean_tuwr"]),
        complexity_rank=int(selected["complexity_rank"]),
    )


def choose_best_fixed_from_source_zones(
    *,
    contracts: FrozenContracts,
    validated: ValidatedTrainingObservations,
) -> ConfigurationSelection:
    baseline_by_action = {
        "Static": "FixedStatic",
        "ACI": "FixedACI",
        "AgACI": "FixedAgACI",
        "EnbPI_RH": "FixedEnbPI_RH",
    }
    score_frames: list[pd.DataFrame] = []
    event_ids = validated.events["event_id"].astype(str).tolist()
    for complexity_rank, action in enumerate(contracts.actions):
        metrics = selected_action_zone_metrics(
            validated=validated,
            selected_actions={event_id: action for event_id in event_ids},
        )
        metrics["configuration_id"] = baseline_by_action[action]
        metrics["complexity_rank"] = complexity_rank
        score_frames.append(metrics)
    return select_configuration(
        contracts=contracts,
        validation_scores=pd.concat(score_frames, ignore_index=True),
    )
