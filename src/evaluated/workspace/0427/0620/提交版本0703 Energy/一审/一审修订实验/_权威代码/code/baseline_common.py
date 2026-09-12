from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from clara_event_contract import FrozenContracts


BASELINE_EVENT_FIELDS = (
    "event_id",
    "zone_or_farm",
    "issue_timestamp",
    "predictor",
    "horizon_group",
    "target_coverage",
    "ramp_state",
    "rolling_state",
    "raw_width_state",
)

BASELINE_CANDIDATE_FIELDS = (
    "event_id",
    "action",
    "candidate_lower",
    "candidate_upper",
)


@dataclass(frozen=True)
class ValidatedBaselineInputs:
    events: pd.DataFrame
    candidates: pd.DataFrame


@dataclass(frozen=True)
class BaselineDecision:
    event_id: str
    baseline_id: str
    selected_action: str | None
    selected_lower: float
    selected_upper: float
    decision_score: float | None
    decision_reason: str

    def as_record(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "baseline_id": self.baseline_id,
            "selected_action": self.selected_action,
            "selected_lower": float(self.selected_lower),
            "selected_upper": float(self.selected_upper),
            "decision_score": None if self.decision_score is None else float(self.decision_score),
            "decision_reason": self.decision_reason,
        }


@dataclass(frozen=True)
class InnerZoneFold:
    outer_heldout_zone: str
    fold_id: str
    inner_validation_zone: str
    inner_training_zones: tuple[str, ...]

    def as_record(self) -> dict[str, Any]:
        return {
            "outer_heldout_zone": self.outer_heldout_zone,
            "fold_id": self.fold_id,
            "inner_validation_zone": self.inner_validation_zone,
            "inner_training_zones": list(self.inner_training_zones),
        }


def _require_columns(frame: pd.DataFrame, required: Iterable[str], table: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{table}缺少字段: {missing}")


def _natural_zone_order(zone: str) -> tuple[str, int | str]:
    prefix = "zone"
    if str(zone).startswith(prefix) and str(zone)[len(prefix) :].isdigit():
        return prefix, int(str(zone)[len(prefix) :])
    return str(zone), str(zone)


def validate_baseline_inputs(
    *,
    contracts: FrozenContracts,
    events: pd.DataFrame,
    candidates: pd.DataFrame,
) -> ValidatedBaselineInputs:
    _require_columns(events, BASELINE_EVENT_FIELDS, "events")
    _require_columns(candidates, BASELINE_CANDIDATE_FIELDS, "candidates")
    event_frame = events.copy()
    candidate_frame = candidates.copy()
    if event_frame["event_id"].isna().any() or event_frame["event_id"].duplicated().any():
        raise ValueError("events的event_id必须非空且唯一")
    event_frame["event_id"] = event_frame["event_id"].astype(str)
    candidate_frame["event_id"] = candidate_frame["event_id"].astype(str)
    if candidate_frame.loc[:, ["event_id", "action"]].duplicated().any():
        raise ValueError("candidates存在重复event_id与action")
    event_ids = set(event_frame["event_id"])
    candidate_event_ids = set(candidate_frame["event_id"])
    if event_ids != candidate_event_ids:
        missing_candidates = sorted(event_ids - candidate_event_ids)
        unknown_candidates = sorted(candidate_event_ids - event_ids)
        raise ValueError(f"事件与候选键不一致: missing={missing_candidates}, unknown={unknown_candidates}")
    expected_actions = set(contracts.actions)
    action_errors: list[str] = []
    for event_id, group in candidate_frame.groupby("event_id", sort=False):
        observed = list(group["action"].astype(str))
        if len(observed) != len(contracts.actions) or set(observed) != expected_actions:
            action_errors.append(str(event_id))
    if action_errors:
        raise ValueError(f"候选动作集合不完整: {action_errors[:10]}")
    numeric = candidate_frame.loc[:, ["candidate_lower", "candidate_upper"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("候选区间包含非有限值")
    candidate_frame["candidate_lower"] = numeric["candidate_lower"].astype(float)
    candidate_frame["candidate_upper"] = numeric["candidate_upper"].astype(float)
    if (candidate_frame["candidate_lower"] > candidate_frame["candidate_upper"]).any():
        raise ValueError("候选区间存在下界高于上界")
    if (candidate_frame["candidate_lower"] < 0.0).any() or (candidate_frame["candidate_upper"] > 1.0).any():
        raise ValueError("候选区间违反0到1共同裁剪合同")
    FrozenStateEncoder(contracts).validate(event_frame)
    action_rank = {action: index for index, action in enumerate(contracts.actions)}
    candidate_frame["_action_rank"] = candidate_frame["action"].map(action_rank)
    if candidate_frame["_action_rank"].isna().any():
        raise ValueError("候选动作不在冻结动作集合中")
    event_frame = event_frame.sort_values("event_id", kind="mergesort").reset_index(drop=True)
    candidate_frame = (
        candidate_frame.sort_values(["event_id", "_action_rank"], kind="mergesort")
        .drop(columns="_action_rank")
        .reset_index(drop=True)
    )
    return ValidatedBaselineInputs(events=event_frame, candidates=candidate_frame)


class FrozenStateEncoder:
    def __init__(self, contracts: FrozenContracts) -> None:
        protocol = contracts.protocol
        self.fields = tuple(contracts.baseline_registry["common_information_contract"]["release_time_state_fields"])
        expected_fields = (
            "predictor",
            "horizon_group",
            "target_coverage",
            "ramp_state",
            "rolling_state",
            "raw_width_state",
        )
        if self.fields != expected_fields:
            raise RuntimeError("基线注册表共同状态字段顺序失配")
        self.vocabularies: dict[str, tuple[Any, ...]] = {
            "predictor": tuple(protocol["predictor_contract"]["predictors"]),
            "horizon_group": tuple(protocol["horizon_group_contract"]["groups"].keys()),
            "target_coverage": tuple(float(value) for value in protocol["coverage_contract"]["target_coverage_levels"]),
            "ramp_state": tuple(protocol["state_contract"]["ramp"]["regimes"]),
            "rolling_state": tuple(protocol["state_contract"]["rolling_reliability"]["states"]),
            "raw_width_state": tuple(protocol["state_contract"]["raw_width"]["states"]),
        }
        names = ["state__intercept"]
        for field in self.fields:
            for value in self.vocabularies[field]:
                names.append(f"state__{field}__{self._format_value(value)}")
        self.feature_names = tuple(names)

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)

    def validate(self, events: pd.DataFrame) -> None:
        _require_columns(events, ("event_id", *self.fields), "events")
        errors: dict[str, list[str]] = {}
        for field in self.fields:
            values = events[field]
            if values.isna().any():
                errors[field] = ["<MISSING>"]
                continue
            if field == "target_coverage":
                observed = set(pd.to_numeric(values, errors="coerce").astype(float))
            else:
                observed = set(values.astype(str))
            allowed = set(self.vocabularies[field])
            unknown = sorted(str(value) for value in observed if value not in allowed)
            if unknown:
                errors[field] = unknown
        if errors:
            raise ValueError(f"状态字段含冻结词表外值: {errors}")

    def transform(self, events: pd.DataFrame) -> pd.DataFrame:
        self.validate(events)
        result = pd.DataFrame({"event_id": events["event_id"].astype(str).to_numpy()})
        result["state__intercept"] = 1.0
        for field in self.fields:
            values = events[field].astype(float) if field == "target_coverage" else events[field].astype(str)
            for value in self.vocabularies[field]:
                name = f"state__{field}__{self._format_value(value)}"
                result[name] = values.eq(value).astype(float).to_numpy()
        return result.loc[:, ["event_id", *self.feature_names]]


def build_nested_source_zone_folds(
    *,
    contracts: FrozenContracts,
    outer_heldout_zone: str,
) -> tuple[InnerZoneFold, ...]:
    zones = tuple(str(zone) for zone in contracts.protocol["datasets"]["gefcom2014"]["zones"])
    heldout = str(outer_heldout_zone)
    if heldout not in zones:
        raise ValueError(f"外层持出区不在冻结GEFCom区域中: {heldout}")
    source_zones = tuple(sorted((zone for zone in zones if zone != heldout), key=_natural_zone_order))
    if len(source_zones) != 9:
        raise RuntimeError("外层折必须恰有九个源区")
    folds: list[InnerZoneFold] = []
    for validation_zone in source_zones:
        training = tuple(zone for zone in source_zones if zone != validation_zone)
        folds.append(
            InnerZoneFold(
                outer_heldout_zone=heldout,
                fold_id=f"outer_{heldout}__inner_{validation_zone}",
                inner_validation_zone=validation_zone,
                inner_training_zones=training,
            )
        )
    return tuple(folds)


def decisions_from_selected_actions(
    *,
    contracts: FrozenContracts,
    events: pd.DataFrame,
    candidates: pd.DataFrame,
    baseline_id: str,
    selected_actions: dict[str, str],
    decision_scores: dict[str, float] | None = None,
    decision_reason: str,
) -> tuple[BaselineDecision, ...]:
    inputs = validate_baseline_inputs(contracts=contracts, events=events, candidates=candidates)
    expected_event_ids = set(inputs.events["event_id"])
    if set(selected_actions) != expected_event_ids:
        raise ValueError("选中动作映射与事件集合不一致")
    invalid_actions = sorted(set(selected_actions.values()) - set(contracts.actions))
    if invalid_actions:
        raise ValueError(f"选中动作不在冻结动作集合中: {invalid_actions}")
    candidate_index = inputs.candidates.set_index(["event_id", "action"])
    if not candidate_index.index.is_unique:
        raise RuntimeError("候选事件与动作索引不唯一")
    decisions: list[BaselineDecision] = []
    for event_id in sorted(expected_event_ids):
        action = selected_actions[event_id]
        row = candidate_index.loc[(event_id, action)]
        score = None if decision_scores is None else decision_scores.get(event_id)
        decisions.append(
            BaselineDecision(
                event_id=event_id,
                baseline_id=str(baseline_id),
                selected_action=action,
                selected_lower=float(row["candidate_lower"]),
                selected_upper=float(row["candidate_upper"]),
                decision_score=None if score is None else float(score),
                decision_reason=str(decision_reason),
            )
        )
    return tuple(decisions)
