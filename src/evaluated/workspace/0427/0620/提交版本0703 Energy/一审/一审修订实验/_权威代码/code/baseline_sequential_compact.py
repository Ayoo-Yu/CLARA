from __future__ import annotations

import numpy as np
from numba import njit, prange


ACTION_COUNT = 4
CONTEXT_DIMENSION = 31
ACTIVE_FEATURE_COUNT = 7
ACTION_TOLERANCE = 1e-12


@njit(cache=True)
def _softmax_first_maximum(log_weights: np.ndarray) -> int:
    maximum_log_weight = np.max(log_weights)
    probabilities = np.empty(ACTION_COUNT, dtype=np.float64)
    denominator = 0.0
    for action in range(ACTION_COUNT):
        value = np.exp(log_weights[action] - maximum_log_weight)
        probabilities[action] = value
        denominator += value
    maximum_probability = 0.0
    for action in range(ACTION_COUNT):
        probabilities[action] /= denominator
        if probabilities[action] > maximum_probability:
            maximum_probability = probabilities[action]
    for action in range(ACTION_COUNT):
        if probabilities[action] >= maximum_probability - ACTION_TOLERANCE:
            return action
    return 0


@njit(cache=True, parallel=True)
def evaluate_delayed_hedge_grid(
    state_group: np.ndarray,
    issue_offsets: np.ndarray,
    feedback_order: np.ndarray,
    matured_feedback_ends: np.ndarray,
    losses: np.ndarray,
    covered: np.ndarray,
    tuwr: np.ndarray,
    coverage_by_state: np.ndarray,
    etas: np.ndarray,
    loss_scales: np.ndarray,
    issue_limit: int,
) -> tuple[np.ndarray, ...]:
    issue_count = len(issue_offsets) - 1
    if issue_limit > 0 and issue_limit < issue_count:
        issue_count = issue_limit
    event_count = int(issue_offsets[issue_count])
    configuration_count = len(etas)
    decisions = np.zeros((configuration_count, event_count), dtype=np.uint8)
    loss_sums = np.zeros(configuration_count, dtype=np.float64)
    coverage_gap_sums = np.zeros(configuration_count, dtype=np.float64)
    tuwr_sums = np.zeros(configuration_count, dtype=np.float64)
    tuwr_counts = np.zeros(configuration_count, dtype=np.int64)
    action_counts = np.zeros((configuration_count, ACTION_COUNT), dtype=np.int64)
    matured_update_counts = np.zeros(configuration_count, dtype=np.int64)

    for configuration_index in prange(configuration_count):
        eta = float(etas[configuration_index])
        loss_scale = float(loss_scales[configuration_index])
        log_weights = np.zeros((len(coverage_by_state), ACTION_COUNT), dtype=np.float64)
        state_action = np.zeros(len(coverage_by_state), dtype=np.uint8)
        state_stamp = np.full(len(coverage_by_state), -1, dtype=np.int32)
        feedback_pointer = 0
        local_loss_sum = 0.0
        local_gap_sum = 0.0
        local_tuwr_sum = 0.0
        local_tuwr_count = 0
        local_action_counts = np.zeros(ACTION_COUNT, dtype=np.int64)

        for issue_index in range(issue_count):
            matured_end = int(matured_feedback_ends[issue_index])
            while feedback_pointer < matured_end:
                event_index = int(feedback_order[feedback_pointer])
                state = int(state_group[event_index])
                maximum = -np.inf
                for action in range(ACTION_COUNT):
                    normalized_loss = float(losses[event_index, action]) / loss_scale
                    if normalized_loss < 0.0:
                        normalized_loss = 0.0
                    elif normalized_loss > 1.0:
                        normalized_loss = 1.0
                    log_weights[state, action] -= eta * normalized_loss
                    if log_weights[state, action] > maximum:
                        maximum = log_weights[state, action]
                for action in range(ACTION_COUNT):
                    log_weights[state, action] -= maximum
                feedback_pointer += 1

            start = int(issue_offsets[issue_index])
            end = int(issue_offsets[issue_index + 1])
            marker = issue_index
            for event_index in range(start, end):
                state = int(state_group[event_index])
                if state_stamp[state] != marker:
                    state_action[state] = _softmax_first_maximum(log_weights[state])
                    state_stamp[state] = marker
                action = int(state_action[state])
                decisions[configuration_index, event_index] = action
                local_action_counts[action] += 1
                local_loss_sum += float(losses[event_index, action])
                local_gap_sum += float(covered[event_index, action]) - float(coverage_by_state[state])
                tuwr_value = float(tuwr[event_index, action])
                if np.isfinite(tuwr_value):
                    local_tuwr_sum += tuwr_value
                    local_tuwr_count += 1

        loss_sums[configuration_index] = local_loss_sum
        coverage_gap_sums[configuration_index] = local_gap_sum
        tuwr_sums[configuration_index] = local_tuwr_sum
        tuwr_counts[configuration_index] = local_tuwr_count
        action_counts[configuration_index] = local_action_counts
        matured_update_counts[configuration_index] = feedback_pointer

    return (
        decisions,
        loss_sums,
        coverage_gap_sums,
        tuwr_sums,
        tuwr_counts,
        action_counts,
        matured_update_counts,
        np.asarray([event_count], dtype=np.int64),
    )


@njit(cache=True)
def _linucb_first_minimum(
    inverse_by_action: np.ndarray,
    theta_by_action: np.ndarray,
    active_features: np.ndarray,
    exploration_alpha: float,
) -> int:
    scores = np.empty(ACTION_COUNT, dtype=np.float64)
    minimum = np.inf
    for action in range(ACTION_COUNT):
        mean_cost = 0.0
        variance = 0.0
        for left in range(ACTIVE_FEATURE_COUNT):
            left_feature = int(active_features[left])
            mean_cost += theta_by_action[action, left_feature]
            for right in range(ACTIVE_FEATURE_COUNT):
                right_feature = int(active_features[right])
                variance += inverse_by_action[action, left_feature, right_feature]
        if variance < 0.0:
            variance = 0.0
        score = mean_cost - exploration_alpha * np.sqrt(variance)
        scores[action] = score
        if score < minimum:
            minimum = score
    for action in range(ACTION_COUNT):
        if scores[action] <= minimum + ACTION_TOLERANCE:
            return action
    return 0


@njit(cache=True, parallel=True)
def evaluate_linucb_grid(
    context_ids: np.ndarray,
    context_active_features: np.ndarray,
    seed: np.ndarray,
    coverage_code: np.ndarray,
    coverage_levels: np.ndarray,
    issue_offsets: np.ndarray,
    feedback_order: np.ndarray,
    matured_feedback_ends: np.ndarray,
    losses: np.ndarray,
    covered: np.ndarray,
    tuwr: np.ndarray,
    exploration_alphas: np.ndarray,
    l2_regularizations: np.ndarray,
    issue_limit: int,
) -> tuple[np.ndarray, ...]:
    issue_count = len(issue_offsets) - 1
    if issue_limit > 0 and issue_limit < issue_count:
        issue_count = issue_limit
    event_count = int(issue_offsets[issue_count])
    configuration_count = len(exploration_alphas)
    context_count = len(context_active_features)
    decisions = np.zeros((configuration_count, event_count), dtype=np.uint8)
    loss_sums = np.zeros(configuration_count, dtype=np.float64)
    coverage_gap_sums = np.zeros(configuration_count, dtype=np.float64)
    tuwr_sums = np.zeros(configuration_count, dtype=np.float64)
    tuwr_counts = np.zeros(configuration_count, dtype=np.int64)
    action_counts = np.zeros((configuration_count, ACTION_COUNT), dtype=np.int64)
    matured_update_counts = np.zeros((configuration_count, ACTION_COUNT), dtype=np.int64)

    for configuration_index in prange(configuration_count):
        exploration_alpha = float(exploration_alphas[configuration_index])
        l2_regularization = float(l2_regularizations[configuration_index])
        covariance = np.zeros((3, ACTION_COUNT, CONTEXT_DIMENSION, CONTEXT_DIMENSION), dtype=np.float64)
        inverse = np.zeros_like(covariance)
        response = np.zeros((3, ACTION_COUNT, CONTEXT_DIMENSION), dtype=np.float64)
        theta = np.zeros_like(response)
        dirty = np.zeros((3, ACTION_COUNT), dtype=np.uint8)
        update_counts = np.zeros((3, ACTION_COUNT), dtype=np.int64)
        for seed_index in range(3):
            for action in range(ACTION_COUNT):
                for feature in range(CONTEXT_DIMENSION):
                    covariance[seed_index, action, feature, feature] = l2_regularization
                    inverse[seed_index, action, feature, feature] = 1.0 / l2_regularization

        context_stamp = np.full((3, context_count), -1, dtype=np.int32)
        context_action = np.zeros((3, context_count), dtype=np.uint8)
        feedback_pointer = 0
        local_loss_sum = 0.0
        local_gap_sum = 0.0
        local_tuwr_sum = 0.0
        local_tuwr_count = 0
        local_action_counts = np.zeros(ACTION_COUNT, dtype=np.int64)

        for issue_index in range(issue_count):
            matured_end = int(matured_feedback_ends[issue_index])
            while feedback_pointer < matured_end:
                event_index = int(feedback_order[feedback_pointer])
                selected_action = int(decisions[configuration_index, event_index])
                seed_index = int(seed[event_index])
                context = int(context_ids[event_index])
                active = context_active_features[context]
                raw_loss = float(losses[event_index, selected_action])
                for left in range(ACTIVE_FEATURE_COUNT):
                    left_feature = int(active[left])
                    response[seed_index, selected_action, left_feature] += raw_loss
                    for right in range(ACTIVE_FEATURE_COUNT):
                        right_feature = int(active[right])
                        covariance[
                            seed_index,
                            selected_action,
                            left_feature,
                            right_feature,
                        ] += 1.0
                dirty[seed_index, selected_action] = 1
                update_counts[seed_index, selected_action] += 1
                feedback_pointer += 1

            for seed_index in range(3):
                for action in range(ACTION_COUNT):
                    if dirty[seed_index, action] == 1:
                        inverse[seed_index, action] = np.linalg.inv(
                            covariance[seed_index, action]
                        )
                        theta[seed_index, action] = (
                            inverse[seed_index, action] @ response[seed_index, action]
                        )
                        dirty[seed_index, action] = 0

            start = int(issue_offsets[issue_index])
            end = int(issue_offsets[issue_index + 1])
            marker = issue_index
            for event_index in range(start, end):
                seed_index = int(seed[event_index])
                context = int(context_ids[event_index])
                if context_stamp[seed_index, context] != marker:
                    context_action[seed_index, context] = _linucb_first_minimum(
                        inverse[seed_index],
                        theta[seed_index],
                        context_active_features[context],
                        exploration_alpha,
                    )
                    context_stamp[seed_index, context] = marker
                action = int(context_action[seed_index, context])
                decisions[configuration_index, event_index] = action
                local_action_counts[action] += 1
                local_loss_sum += float(losses[event_index, action])
                local_gap_sum += float(covered[event_index, action]) - float(
                    coverage_levels[int(coverage_code[event_index])]
                )
                tuwr_value = float(tuwr[event_index, action])
                if np.isfinite(tuwr_value):
                    local_tuwr_sum += tuwr_value
                    local_tuwr_count += 1

        loss_sums[configuration_index] = local_loss_sum
        coverage_gap_sums[configuration_index] = local_gap_sum
        tuwr_sums[configuration_index] = local_tuwr_sum
        tuwr_counts[configuration_index] = local_tuwr_count
        action_counts[configuration_index] = local_action_counts
        for seed_index in range(3):
            for action in range(ACTION_COUNT):
                matured_update_counts[configuration_index, action] += update_counts[
                    seed_index,
                    action,
                ]

    return (
        decisions,
        loss_sums,
        coverage_gap_sums,
        tuwr_sums,
        tuwr_counts,
        action_counts,
        matured_update_counts,
        np.asarray([event_count], dtype=np.int64),
    )
