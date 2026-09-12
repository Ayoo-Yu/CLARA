"""S09 全局 LinUCB 的等价批量打分加速器。

反馈成熟与矩阵更新仍逐事件执行。每个同起报时刻内的动作分数批量计算，
因为冻结协议禁止同一时刻内使用反馈，所以该变换保持原始决策语义。
"""
from __future__ import annotations

import math
import sys
from heapq import heappop, heappush
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


S09_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = Path(__file__).resolve().parents[2]
AUTHORITATIVE_CODE = REVISION_ROOT / "_权威代码" / "code"
sys.path.insert(0, str(AUTHORITATIVE_CODE))
sys.path.insert(0, str(S09_ROOT / "scripts"))

from clara_event_contract import FrozenContracts  # noqa: E402
import s09_minimum_external_core as core  # noqa: E402


def run_linucb_batch_equivalent(
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
    for column in ("issue_timestamp", "label_timestamp", "label_available_timestamp"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    frame = frame.sort_values(
        ["issue_timestamp", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    context_lookup, feature_names = core._state_context_lookup(frame, contracts)
    dimension = len(feature_names)
    inverse = {
        action: np.eye(dimension, dtype=np.float64) / float(l2_regularization)
        for action in core.ACTIONS
    }
    vector = {
        action: np.zeros(dimension, dtype=np.float64) for action in core.ACTIONS
    }
    updates = {action: 0 for action in core.ACTIONS}
    pending: list[tuple[int, int, str, int, str, np.ndarray, float]] = []
    selected_output = np.empty(len(final_facts), dtype=object)
    feedback_count = 0
    future_violation_count = 0

    def ns_value(value: Any) -> int:
        return int(
            pd.Timestamp(value)
            .to_datetime64()
            .astype("datetime64[ns]")
            .astype(np.int64)
        )

    def apply_matured(issue_ns: int) -> None:
        nonlocal feedback_count, future_violation_count
        while pending and pending[0][0] <= issue_ns:
            available_ns, label_ns, _, _, action, context, loss = heappop(pending)
            if label_ns >= issue_ns:
                future_violation_count += 1
                raise RuntimeError("LinUCB 批量加速器检测到非严格成熟反馈")
            projected = inverse[action] @ context
            denominator = 1.0 + float(context @ projected)
            if denominator <= 0.0 or not np.isfinite(denominator):
                raise RuntimeError("LinUCB 批量加速器逆矩阵更新分母非法")
            inverse[action] = inverse[action] - np.outer(projected, projected) / denominator
            vector[action] = vector[action] + context * float(loss)
            updates[action] += 1
            feedback_count += 1

    for issue, batch in frame.groupby("issue_timestamp", sort=True, dropna=False):
        issue_ns = ns_value(issue)
        apply_matured(issue_ns)
        state_keys = [
            tuple(row)
            for row in batch.loc[:, list(core.STATE_FIELDS)].itertuples(
                index=False, name=None
            )
        ]
        contexts = np.vstack([context_lookup[key] for key in state_keys])
        scores = np.empty((len(batch), len(core.ACTIONS)), dtype=np.float64)
        for action_index, action in enumerate(core.ACTIONS):
            theta = inverse[action] @ vector[action]
            mean_cost = contexts @ theta
            projected = contexts @ inverse[action]
            variance = np.maximum(np.sum(projected * contexts, axis=1), 0.0)
            scores[:, action_index] = mean_cost - float(exploration_alpha) * np.sqrt(
                variance
            )
        minimum = scores.min(axis=1)
        selected_index = (scores <= minimum[:, None] + 1e-12).argmax(axis=1)
        selected_actions = np.asarray(core.ACTIONS, dtype=object)[selected_index]
        evaluation_mask = batch["_evaluation_row"].astype(bool).to_numpy()
        if evaluation_mask.any():
            output_positions = batch.loc[
                evaluation_mask, "_evaluation_order"
            ].to_numpy(dtype=np.int64)
            selected_output[output_positions] = selected_actions[evaluation_mask]
        loss_matrix = batch[
            [f"{action}__errf" for action in core.ACTIONS]
        ].to_numpy(dtype=np.float64)
        selected_losses = loss_matrix[np.arange(len(batch)), selected_index]
        available_values = batch["label_available_timestamp"].tolist()
        label_values = batch["label_timestamp"].tolist()
        event_ids = batch["event_id"].astype(str).tolist()
        for row_index in range(len(batch)):
            heappush(
                pending,
                (
                    ns_value(available_values[row_index]),
                    ns_value(label_values[row_index]),
                    event_ids[row_index],
                    int(selected_index[row_index]),
                    str(selected_actions[row_index]),
                    contexts[row_index].copy(),
                    float(selected_losses[row_index]),
                ),
            )
    if pd.isna(pd.Series(selected_output)).any():
        raise RuntimeError("LinUCB 批量加速器最终事件存在缺失决策")
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
        "batch_scoring": True,
        "feedback_update_order": "identical_eventwise_heap_order",
    }
    return selected_output.astype(str), audit
