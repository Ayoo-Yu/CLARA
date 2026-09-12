from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd

from baseline_common import BaselineDecision, FrozenStateEncoder, validate_baseline_inputs
from baseline_training import validate_frozen_selector_config, validate_training_observations
from clara_event_contract import FrozenContracts, as_datetime, canonical_json_sha256


SEQUENTIAL_REQUIRED_FIELDS = (
    "event_id",
    "origin",
    "zone_or_farm",
    "seed",
    "theta_id",
    "issue_timestamp",
    "label_timestamp",
    "label_available_timestamp",
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


@dataclass(frozen=True)
class ValidatedSequentialObservations:
    events: pd.DataFrame
    observations: pd.DataFrame


@dataclass(frozen=True)
class SequentialRunResult:
    decisions: tuple[BaselineDecision, ...]
    feedback: tuple[dict[str, Any], ...]
    audit: dict[str, Any]


@dataclass(frozen=True)
class _PendingFeedback:
    event_id: str
    issue_timestamp: datetime
    label_timestamp: datetime
    label_available_timestamp: datetime
    event: dict[str, Any]
    selected_action: str
    losses: dict[str, float]
    context: np.ndarray | None


def _require_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"顺序基线观测缺少字段: {missing}")


def validate_sequential_observations(
    *,
    contracts: FrozenContracts,
    observations: pd.DataFrame,
    allowed_zones: Iterable[str],
    allowed_origins: Iterable[str],
    forbidden_zones: Iterable[str] = (),
) -> ValidatedSequentialObservations:
    _require_columns(observations, SEQUENTIAL_REQUIRED_FIELDS)
    frame = observations.copy()
    if frame["event_id"].isna().any() or frame["action"].isna().any():
        raise ValueError("顺序基线event_id与action不得为空")
    frame["event_id"] = frame["event_id"].astype(str)
    frame["action"] = frame["action"].astype(str)
    frame["zone_or_farm"] = frame["zone_or_farm"].astype(str)
    frame["origin"] = frame["origin"].astype(str)
    if frame.loc[:, ["event_id", "action"]].duplicated().any():
        raise ValueError("顺序基线观测存在重复event_id与action")

    allowed_zone_set = {str(zone) for zone in allowed_zones}
    observed_zones = set(frame["zone_or_farm"])
    if len(allowed_zone_set) != 1 or observed_zones != allowed_zone_set:
        raise ValueError("顺序基线每次运行必须精确对应一个允许区域")
    leaked = sorted(observed_zones & {str(zone) for zone in forbidden_zones})
    if leaked:
        raise ValueError(f"顺序基线观测包含禁止区域: {leaked}")
    allowed_origin_set = {str(origin) for origin in allowed_origins}
    unknown_origins = sorted(set(frame["origin"]) - allowed_origin_set)
    if unknown_origins:
        raise ValueError(f"顺序基线观测包含不允许的origin: {unknown_origins}")

    expected_actions = set(contracts.actions)
    invalid_events: list[str] = []
    for event_id, group in frame.groupby("event_id", sort=False):
        actions = list(group["action"])
        if len(actions) != len(contracts.actions) or set(actions) != expected_actions:
            invalid_events.append(str(event_id))
    if invalid_events:
        raise ValueError(f"顺序基线事件四动作不完整: {invalid_events[:10]}")

    frame["errf"] = pd.to_numeric(frame["errf"], errors="coerce")
    if not np.isfinite(frame["errf"].to_numpy(dtype=float)).all() or (frame["errf"] < 0.0).any():
        raise ValueError("顺序基线ERRF必须为有限非负值")
    frame["covered"] = pd.to_numeric(frame["covered"], errors="coerce")
    if not frame["covered"].isin((0.0, 1.0)).all():
        raise ValueError("顺序基线covered必须为零或一")

    state_fields = tuple(contracts.baseline_registry["common_information_contract"]["release_time_state_fields"])
    event_fields = (
        "event_id",
        "origin",
        "zone_or_farm",
        "seed",
        "theta_id",
        "issue_timestamp",
        "label_timestamp",
        "label_available_timestamp",
        *state_fields,
    )
    consistency = frame.groupby("event_id", sort=False)[list(event_fields[1:])].nunique(dropna=False)
    if (consistency > 1).any().any():
        raise ValueError("顺序基线同一事件的元数据或发布时状态不一致")
    events = frame.loc[:, event_fields].drop_duplicates("event_id").copy()
    for column in ("issue_timestamp", "label_timestamp", "label_available_timestamp"):
        events[column] = pd.to_datetime(events[column], errors="raise")
    invalid_issue_label = ~(events["issue_timestamp"] < events["label_timestamp"])
    invalid_label_available = ~(events["label_timestamp"] < events["label_available_timestamp"])
    if invalid_issue_label.any():
        raise ValueError("顺序基线label_timestamp必须晚于issue_timestamp")
    if invalid_label_available.any():
        raise ValueError("顺序基线label_available_timestamp必须严格晚于label_timestamp")
    FrozenStateEncoder(contracts).validate(events)

    action_rank = {action: index for index, action in enumerate(contracts.actions)}
    frame["_action_rank"] = frame["action"].map(action_rank)
    frame = (
        frame.sort_values(["issue_timestamp", "event_id", "_action_rank"], kind="mergesort")
        .drop(columns="_action_rank")
        .reset_index(drop=True)
    )
    events = events.sort_values(["issue_timestamp", "event_id"], kind="mergesort").reset_index(drop=True)
    return ValidatedSequentialObservations(events=events, observations=frame)


def fit_delayed_hedge_loss_scale(
    *,
    contracts: FrozenContracts,
    source_training_observations: pd.DataFrame,
    training_zones: Iterable[str],
    forbidden_zones: Iterable[str],
) -> float:
    zones = tuple(str(zone) for zone in training_zones)
    validated = validate_training_observations(
        contracts=contracts,
        observations=source_training_observations,
        allowed_zones=zones,
        forbidden_zones=forbidden_zones,
    )
    losses = validated.observations["errf"].to_numpy(dtype=float)
    scale = float(np.quantile(losses, 0.95, method="linear"))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("DelayedHedge源区训练损失95分位必须为有限正值")
    return scale


def _select_minimum_with_tolerance(
    *,
    contracts: FrozenContracts,
    scores: dict[str, float],
    tolerance: float = 1e-12,
) -> str:
    minimum = min(scores.values())
    for action in contracts.actions:
        if float(scores[action]) <= float(minimum) + tolerance:
            return action
    raise RuntimeError("最小分数动作选择失败")


def _select_maximum_with_tolerance(
    *,
    contracts: FrozenContracts,
    scores: dict[str, float],
    tolerance: float = 1e-12,
) -> str:
    maximum = max(scores.values())
    for action in contracts.actions:
        if float(scores[action]) >= float(maximum) - tolerance:
            return action
    raise RuntimeError("最大分数动作选择失败")


class _DelayedHedgePolicy:
    baseline_id = "DelayedHedge"
    feedback_scope = "all_four_actions_after_strict_maturity"

    def __init__(
        self,
        *,
        contracts: FrozenContracts,
        eta: float,
        loss_scale: float,
    ) -> None:
        config = {"eta": float(eta)}
        validate_frozen_selector_config(
            contracts=contracts,
            baseline_id=self.baseline_id,
            config=config,
        )
        if not np.isfinite(loss_scale) or float(loss_scale) <= 0.0:
            raise ValueError("DelayedHedge损失尺度必须为有限正值")
        self.contracts = contracts
        self.eta = float(eta)
        self.loss_scale = float(loss_scale)
        self.log_weights: dict[tuple[str, int, str, str, float], np.ndarray] = {}
        self.update_counts: dict[tuple[str, int, str, str, float], int] = {}
        self.config_id = canonical_json_sha256(
            {
                "baseline_id": self.baseline_id,
                "eta": self.eta,
                "loss_scale": self.loss_scale,
                "state_scope": ["theta_id", "seed", "predictor", "horizon_group", "target_coverage"],
            }
        )

    @staticmethod
    def state_key(event: dict[str, Any]) -> tuple[str, int, str, str, float]:
        return (
            str(event["theta_id"]),
            int(event["seed"]),
            str(event["predictor"]),
            str(event["horizon_group"]),
            float(event["target_coverage"]),
        )

    def select(self, event: dict[str, Any], context: np.ndarray | None) -> tuple[str, float]:
        del context
        key = self.state_key(event)
        log_weights = self.log_weights.setdefault(key, np.zeros(len(self.contracts.actions), dtype=float))
        shifted = log_weights - float(np.max(log_weights))
        weights = np.exp(shifted)
        probabilities = weights / float(weights.sum())
        probability_by_action = {
            action: float(probabilities[index])
            for index, action in enumerate(self.contracts.actions)
        }
        action = _select_maximum_with_tolerance(
            contracts=self.contracts,
            scores=probability_by_action,
        )
        return action, probability_by_action[action]

    def apply_feedback(
        self,
        pending: _PendingFeedback,
        application_time: datetime,
    ) -> list[dict[str, Any]]:
        key = self.state_key(pending.event)
        log_weights = self.log_weights.setdefault(key, np.zeros(len(self.contracts.actions), dtype=float))
        before = int(self.update_counts.get(key, 0))
        records: list[dict[str, Any]] = []
        for action_index, action in enumerate(self.contracts.actions):
            raw_loss = float(pending.losses[action])
            normalized_loss = float(np.clip(raw_loss / self.loss_scale, 0.0, 1.0))
            log_weights[action_index] -= self.eta * normalized_loss
            records.append(
                {
                    "event_id": pending.event_id,
                    "baseline_id": self.baseline_id,
                    "selected_action": pending.selected_action,
                    "action_updated": action,
                    "raw_loss": raw_loss,
                    "normalized_loss": normalized_loss,
                    "issue_timestamp": pending.issue_timestamp,
                    "label_timestamp": pending.label_timestamp,
                    "label_available_timestamp": pending.label_available_timestamp,
                    "feedback_application_timestamp": application_time,
                    "eligible_by_strict_time_rule": pending.label_timestamp < application_time,
                    "full_information_feedback": True,
                    "selected_action_feedback_only": False,
                    "update_count_before": before,
                    "update_count_after": before + 1,
                }
            )
        log_weights -= float(np.max(log_weights))
        self.update_counts[key] = before + 1
        return records

    def audit_fields(self) -> dict[str, Any]:
        return {
            "eta": self.eta,
            "loss_scale": self.loss_scale,
            "state_count": len(self.log_weights),
            "state_update_count": int(sum(self.update_counts.values())),
            "full_action_feedback_after_maturity": True,
            "selected_action_feedback_only": False,
        }


class _LinUCBPolicy:
    baseline_id = "LinUCB"
    feedback_scope = "selected_action_only_after_strict_maturity"

    def __init__(
        self,
        *,
        contracts: FrozenContracts,
        exploration_alpha: float,
        l2_regularization: float,
    ) -> None:
        config = {
            "exploration_alpha": float(exploration_alpha),
            "l2_regularization": float(l2_regularization),
        }
        validate_frozen_selector_config(
            contracts=contracts,
            baseline_id=self.baseline_id,
            config=config,
        )
        self.contracts = contracts
        self.exploration_alpha = float(exploration_alpha)
        self.l2_regularization = float(l2_regularization)
        self.encoder = FrozenStateEncoder(contracts)
        self.dimension = len(self.encoder.feature_names)
        self.states: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
        self.config_id = canonical_json_sha256(
            {
                "baseline_id": self.baseline_id,
                **config,
                "state_scope": ["theta_id", "seed"],
                "context_dimension": self.dimension,
            }
        )

    @staticmethod
    def state_key(event: dict[str, Any]) -> tuple[str, int]:
        return str(event["theta_id"]), int(event["seed"])

    def _state(self, event: dict[str, Any]) -> dict[str, dict[str, Any]]:
        key = self.state_key(event)
        if key not in self.states:
            inverse = np.eye(self.dimension, dtype=float) / self.l2_regularization
            self.states[key] = {
                action: {
                    "A_inverse": inverse.copy(),
                    "b": np.zeros(self.dimension, dtype=float),
                    "update_count": 0,
                }
                for action in self.contracts.actions
            }
        return self.states[key]

    def select(self, event: dict[str, Any], context: np.ndarray | None) -> tuple[str, float]:
        if context is None:
            raise ValueError("LinUCB选择缺少上下文")
        state = self._state(event)
        scores: dict[str, float] = {}
        for action in self.contracts.actions:
            arm = state[action]
            inverse = arm["A_inverse"]
            theta = inverse @ arm["b"]
            mean_cost = float(theta @ context)
            variance = max(float(context @ inverse @ context), 0.0)
            scores[action] = mean_cost - self.exploration_alpha * float(np.sqrt(variance))
        action = _select_minimum_with_tolerance(contracts=self.contracts, scores=scores)
        return action, scores[action]

    def apply_feedback(
        self,
        pending: _PendingFeedback,
        application_time: datetime,
    ) -> list[dict[str, Any]]:
        if pending.context is None:
            raise ValueError("LinUCB反馈缺少发布时上下文")
        state = self._state(pending.event)
        arm = state[pending.selected_action]
        context = pending.context
        inverse = arm["A_inverse"]
        projected = inverse @ context
        denominator = 1.0 + float(context @ projected)
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise RuntimeError("LinUCB逆矩阵更新分母无效")
        before = int(arm["update_count"])
        arm["A_inverse"] = inverse - np.outer(projected, projected) / denominator
        raw_loss = float(pending.losses[pending.selected_action])
        arm["b"] = arm["b"] + context * raw_loss
        arm["update_count"] = before + 1
        return [
            {
                "event_id": pending.event_id,
                "baseline_id": self.baseline_id,
                "selected_action": pending.selected_action,
                "action_updated": pending.selected_action,
                "raw_loss": raw_loss,
                "normalized_loss": None,
                "issue_timestamp": pending.issue_timestamp,
                "label_timestamp": pending.label_timestamp,
                "label_available_timestamp": pending.label_available_timestamp,
                "feedback_application_timestamp": application_time,
                "eligible_by_strict_time_rule": pending.label_timestamp < application_time,
                "full_information_feedback": False,
                "selected_action_feedback_only": True,
                "update_count_before": before,
                "update_count_after": before + 1,
            }
        ]

    def audit_fields(self) -> dict[str, Any]:
        action_update_count = {
            action: int(
                sum(int(state[action]["update_count"]) for state in self.states.values())
            )
            for action in self.contracts.actions
        }
        return {
            "exploration_alpha": self.exploration_alpha,
            "l2_regularization": self.l2_regularization,
            "context_dimension": self.dimension,
            "state_count": len(self.states),
            "action_update_count": action_update_count,
            "full_action_feedback_after_maturity": False,
            "selected_action_feedback_only": True,
        }


def _run_policy(
    *,
    contracts: FrozenContracts,
    policy: _DelayedHedgePolicy | _LinUCBPolicy,
    evaluation_observations: pd.DataFrame,
    candidates: pd.DataFrame,
    evaluation_zone: str,
    allowed_origins: Iterable[str],
    forbidden_zones: Iterable[str],
    drain_final_feedback: bool,
) -> SequentialRunResult:
    validated = validate_sequential_observations(
        contracts=contracts,
        observations=evaluation_observations,
        allowed_zones=(evaluation_zone,),
        allowed_origins=allowed_origins,
        forbidden_zones=forbidden_zones,
    )
    baseline_inputs = validate_baseline_inputs(
        contracts=contracts,
        events=validated.events,
        candidates=candidates,
    )
    candidate_index = baseline_inputs.candidates.set_index(["event_id", "action"])
    observation_index = validated.observations.set_index(["event_id", "action"])
    encoder = FrozenStateEncoder(contracts)
    encoded = encoder.transform(validated.events).set_index("event_id")
    event_records = validated.events.to_dict(orient="records")
    pending: list[_PendingFeedback] = []
    feedback: list[dict[str, Any]] = []
    decisions: list[BaselineDecision] = []
    issue_batch_count = 0
    within_issue_feedback_use_count = 0
    future_feedback_violation_count = 0

    def apply_one(item: _PendingFeedback, application_time: datetime) -> None:
        nonlocal future_feedback_violation_count
        if not item.label_timestamp < application_time:
            future_feedback_violation_count += 1
            raise RuntimeError("顺序基线检测到非严格成熟反馈")
        feedback.extend(policy.apply_feedback(item, application_time))

    def apply_matured_before(issue_timestamp: datetime) -> None:
        nonlocal pending
        matured = [
            item
            for item in pending
            if item.label_available_timestamp <= issue_timestamp
            and item.label_timestamp < issue_timestamp
        ]
        matured.sort(
            key=lambda item: (
                item.label_available_timestamp,
                item.label_timestamp,
                item.event_id,
            )
        )
        for item in matured:
            apply_one(item, issue_timestamp)
        matured_ids = {item.event_id for item in matured}
        pending = [item for item in pending if item.event_id not in matured_ids]

    index = 0
    while index < len(event_records):
        issue = as_datetime(event_records[index]["issue_timestamp"])
        batch: list[dict[str, Any]] = []
        while index < len(event_records) and as_datetime(event_records[index]["issue_timestamp"]) == issue:
            batch.append(event_records[index])
            index += 1
        issue_batch_count += 1
        apply_matured_before(issue)
        feedback_count_before_release = len(feedback)
        batch_pending: list[_PendingFeedback] = []
        for event in batch:
            event_id = str(event["event_id"])
            context = (
                encoded.loc[event_id, list(encoder.feature_names)].to_numpy(dtype=float)
                if policy.baseline_id == "LinUCB"
                else None
            )
            selected_action, score = policy.select(event, context)
            candidate = candidate_index.loc[(event_id, selected_action)]
            decisions.append(
                BaselineDecision(
                    event_id=event_id,
                    baseline_id=policy.baseline_id,
                    selected_action=selected_action,
                    selected_lower=float(candidate["candidate_lower"]),
                    selected_upper=float(candidate["candidate_upper"]),
                    decision_score=float(score),
                    decision_reason=f"strict_delayed_feedback:{policy.config_id}",
                )
            )
            losses = {
                action: float(observation_index.loc[(event_id, action), "errf"])
                for action in contracts.actions
            }
            batch_pending.append(
                _PendingFeedback(
                    event_id=event_id,
                    issue_timestamp=issue,
                    label_timestamp=as_datetime(event["label_timestamp"]),
                    label_available_timestamp=as_datetime(event["label_available_timestamp"]),
                    event=event,
                    selected_action=selected_action,
                    losses=losses,
                    context=context,
                )
            )
        if len(feedback) != feedback_count_before_release:
            within_issue_feedback_use_count += 1
            raise RuntimeError("顺序基线同批决策阶段使用了同批反馈")
        pending.extend(batch_pending)

    if drain_final_feedback:
        pending.sort(
            key=lambda item: (
                item.label_available_timestamp,
                item.label_timestamp,
                item.event_id,
            )
        )
        for item in pending:
            apply_one(item, item.label_available_timestamp)
        pending = []

    duplicate_feedback_count = len(feedback) - len(
        {
            (record["event_id"], record["action_updated"])
            for record in feedback
        }
    )
    expected_feedback_per_event = len(contracts.actions) if policy.baseline_id == "DelayedHedge" else 1
    audit = {
        "baseline_id": policy.baseline_id,
        "configuration_id": policy.config_id,
        "evaluation_zone": str(evaluation_zone),
        "event_count": len(validated.events),
        "candidate_count": len(baseline_inputs.candidates),
        "issue_batch_count": issue_batch_count,
        "feedback_count": len(feedback),
        "expected_feedback_per_event": expected_feedback_per_event,
        "within_issue_feedback_use_count": within_issue_feedback_use_count,
        "future_feedback_violation_count": future_feedback_violation_count,
        "duplicate_feedback_count": duplicate_feedback_count,
        "pending_feedback_count": len(pending),
        "drain_final_feedback": bool(drain_final_feedback),
        "strict_delayed_feedback": True,
        "same_issue_batch_shared_snapshot": True,
        "feedback_scope": policy.feedback_scope,
        **policy.audit_fields(),
    }
    ordered_decisions = tuple(sorted(decisions, key=lambda item: item.event_id))
    return SequentialRunResult(
        decisions=ordered_decisions,
        feedback=tuple(feedback),
        audit=audit,
    )


def run_delayed_hedge(
    *,
    contracts: FrozenContracts,
    evaluation_observations: pd.DataFrame,
    candidates: pd.DataFrame,
    evaluation_zone: str,
    allowed_origins: Iterable[str],
    forbidden_zones: Iterable[str],
    eta: float,
    loss_scale: float,
    drain_final_feedback: bool = True,
) -> SequentialRunResult:
    policy = _DelayedHedgePolicy(
        contracts=contracts,
        eta=eta,
        loss_scale=loss_scale,
    )
    return _run_policy(
        contracts=contracts,
        policy=policy,
        evaluation_observations=evaluation_observations,
        candidates=candidates,
        evaluation_zone=evaluation_zone,
        allowed_origins=allowed_origins,
        forbidden_zones=forbidden_zones,
        drain_final_feedback=drain_final_feedback,
    )


def run_linucb(
    *,
    contracts: FrozenContracts,
    evaluation_observations: pd.DataFrame,
    candidates: pd.DataFrame,
    evaluation_zone: str,
    allowed_origins: Iterable[str],
    forbidden_zones: Iterable[str],
    exploration_alpha: float,
    l2_regularization: float,
    drain_final_feedback: bool = True,
) -> SequentialRunResult:
    policy = _LinUCBPolicy(
        contracts=contracts,
        exploration_alpha=exploration_alpha,
        l2_regularization=l2_regularization,
    )
    return _run_policy(
        contracts=contracts,
        policy=policy,
        evaluation_observations=evaluation_observations,
        candidates=candidates,
        evaluation_zone=evaluation_zone,
        allowed_origins=allowed_origins,
        forbidden_zones=forbidden_zones,
        drain_final_feedback=drain_final_feedback,
    )
