from __future__ import annotations

import numpy as np
import pandas as pd

from baseline_common import BaselineDecision, validate_baseline_inputs
from clara_event_contract import FrozenContracts


def select_equal_endpoint_ensemble(
    *,
    contracts: FrozenContracts,
    events: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[BaselineDecision, ...]:
    inputs = validate_baseline_inputs(contracts=contracts, events=events, candidates=candidates)
    grouped = (
        inputs.candidates.groupby("event_id", sort=False)
        .agg(selected_lower=("candidate_lower", "mean"), selected_upper=("candidate_upper", "mean"))
        .reset_index()
        .sort_values("event_id", kind="mergesort")
    )
    grouped["selected_lower"] = np.clip(grouped["selected_lower"].to_numpy(dtype=float), 0.0, 1.0)
    grouped["selected_upper"] = np.clip(grouped["selected_upper"].to_numpy(dtype=float), 0.0, 1.0)
    if (grouped["selected_lower"] > grouped["selected_upper"]).any():
        raise RuntimeError("EqualEndpointEnsemble产生反转区间")
    decisions = tuple(
        BaselineDecision(
            event_id=str(row.event_id),
            baseline_id="EqualEndpointEnsemble",
            selected_action=None,
            selected_lower=float(row.selected_lower),
            selected_upper=float(row.selected_upper),
            decision_score=float(row.selected_upper - row.selected_lower),
            decision_reason="arithmetic_mean_of_four_action_endpoints_then_clip_zero_one",
        )
        for row in grouped.itertuples(index=False)
    )
    if len(decisions) != len(inputs.events):
        raise RuntimeError("EqualEndpointEnsemble输出事件数失配")
    return decisions
