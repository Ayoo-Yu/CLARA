from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from baseline_conformal import conformal_grid
from clara_event_contract import FrozenContracts


FAMILY_CODES = {
    "SplitCF": 0,
    "ACI": 1,
    "AgACI": 2,
    "FACI": 3,
    "EnbPI_RH": 4,
    "SAOCP": 5,
    "SPCI": 6,
    "NEX": 7,
    "KOWCPI": 8,
    "WACI": 9,
}
HOUR_NS = np.int64(3_600_000_000_000)


def compact_configuration_arrays(
    contracts: FrozenContracts,
) -> tuple[np.ndarray, np.ndarray]:
    configurations = conformal_grid(contracts)
    family_codes = np.empty(len(configurations), dtype=np.int8)
    parameters = np.zeros((len(configurations), 8), dtype=np.float64)
    for index, item in enumerate(configurations):
        if item.complexity_rank != index or item.family not in FAMILY_CODES:
            raise RuntimeError("TunedSingleConformal紧凑网格顺序或家族失配")
        family_codes[index] = FAMILY_CODES[item.family]
        config = item.configuration
        if item.family == "ACI":
            parameters[index, 0] = float(config["learning_rate"])
        elif item.family == "AgACI":
            rates = [float(value) for value in config["learning_rates"]]
            if len(rates) != 4:
                raise RuntimeError("AgACI紧凑专家数失配")
            parameters[index, :4] = rates
        elif item.family == "FACI":
            parameters[index, 0] = float(config["eta_min"])
            parameters[index, 1] = float(config["eta_max"])
            parameters[index, 2] = float(config["adapt_window_hours"])
        elif item.family == "EnbPI_RH":
            parameters[index, 0] = float(config["history_duration_hours"])
        elif item.family == "SAOCP":
            windows = [float(value) for value in config["window_hours"]]
            if len(windows) != 4:
                raise RuntimeError("SAOCP紧凑专家数失配")
            parameters[index, 0] = float(config["learning_rate"])
            parameters[index, 1:5] = windows
            parameters[index, 5] = float(config["temperature"])
        elif item.family in {"SPCI", "NEX"}:
            parameters[index, 0] = float(config["history_duration_hours"])
            parameters[index, 1] = float(config["decay"])
        elif item.family == "KOWCPI":
            if str(config["kernel"]) != "epanechnikov":
                raise RuntimeError("KOWCPI紧凑核函数失配")
            parameters[index, 0] = float(config["history_duration_hours"])
            parameters[index, 1] = float(config["lag_duration_hours"])
            parameters[index, 2] = float(config["bandwidth"])
        elif item.family == "WACI":
            parameters[index, 0] = float(config["learning_rate"])
            parameters[index, 1] = float(config["width_power"])
            parameters[index, 2] = float(config["adapt_floor"])
            parameters[index, 3] = float(config["adapt_cap"])
    if len(configurations) != 19:
        raise RuntimeError("TunedSingleConformal紧凑配置数失配")
    return family_codes, parameters


def build_feedback_schedule(
    issue_ns: np.ndarray,
    label_ns: np.ndarray,
    available_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    issues = np.asarray(issue_ns, dtype=np.int64)
    labels = np.asarray(label_ns, dtype=np.int64)
    available = np.asarray(available_ns, dtype=np.int64)
    if not (issues.ndim == labels.ndim == available.ndim == 1):
        raise ValueError("反馈日程输入必须为一维")
    if not (len(issues) == len(labels) == len(available)):
        raise ValueError("反馈日程输入长度失配")
    if len(issues) == 0 or np.any(np.diff(issues) <= 0):
        raise ValueError("反馈日程问题时刻必须严格递增")
    if np.any(labels <= issues) or np.any(available <= labels):
        raise ValueError("反馈日程时间顺序失配")
    application_issue = np.full(len(issues), -1, dtype=np.int32)
    application_time = available.copy()
    groups: list[list[int]] = [[] for _ in range(len(issues))]
    final_group: list[int] = []
    for event_index in range(len(issues)):
        first_available = int(np.searchsorted(issues, available[event_index], side="left"))
        first_strict_label = int(np.searchsorted(issues, labels[event_index], side="right"))
        issue_index = max(first_available, first_strict_label)
        if issue_index < len(issues):
            application_issue[event_index] = issue_index
            application_time[event_index] = issues[issue_index]
            groups[issue_index].append(event_index)
        else:
            final_group.append(event_index)
    key = lambda index: (int(available[index]), int(labels[index]), int(index))
    ordered: list[int] = []
    matured_ends = np.zeros(len(issues), dtype=np.int32)
    for issue_index, group in enumerate(groups):
        ordered.extend(sorted(group, key=key))
        matured_ends[issue_index] = len(ordered)
    ordered.extend(sorted(final_group, key=key))
    feedback_order = np.asarray(ordered, dtype=np.int32)
    if len(feedback_order) != len(issues) or len(np.unique(feedback_order)) != len(issues):
        raise RuntimeError("反馈日程没有形成完整排列")
    if np.any(labels[feedback_order[1:]] < labels[feedback_order[:-1]]):
        raise RuntimeError("单流反馈标签顺序未保持单调")
    return feedback_order, matured_ends, application_time


@njit(cache=True)
def _clip_pair(lower: float, upper: float) -> tuple[float, float]:
    clipped_lower = min(max(lower, 0.0), 1.0)
    clipped_upper = min(max(upper, 0.0), 1.0)
    return clipped_lower, clipped_upper


@njit(cache=True)
def _linear_quantile(values: np.ndarray, start: int, end: int, quantile: float) -> float:
    count = end - start
    if count <= 0:
        return 0.0
    ordered = np.sort(values[start:end])
    position = quantile * float(count - 1)
    lower_index = int(np.floor(position))
    upper_index = int(np.ceil(position))
    fraction = position - float(lower_index)
    return float(ordered[lower_index] + fraction * (ordered[upper_index] - ordered[lower_index]))


@njit(cache=True)
def _interp_cdf(
    quantile: float,
    cdf: np.ndarray,
    ordered_values: np.ndarray,
    count: int,
) -> float:
    if count <= 0:
        return 0.0
    if quantile <= cdf[0]:
        return float(ordered_values[0])
    if quantile >= cdf[count - 1]:
        return float(ordered_values[count - 1])
    right = int(np.searchsorted(cdf[:count], quantile, side="left"))
    left = right - 1
    denominator = cdf[right] - cdf[left]
    if denominator <= 0.0:
        return float(ordered_values[right])
    fraction = (quantile - cdf[left]) / denominator
    return float(ordered_values[left] + fraction * (ordered_values[right] - ordered_values[left]))


@njit(cache=True)
def _numpy_pairwise_sum(values: np.ndarray, start: int, count: int) -> float:
    stack_starts = np.empty(64, dtype=np.int64)
    stack_counts = np.empty(64, dtype=np.int64)
    stack_states = np.zeros(64, dtype=np.int8)
    stack_left_results = np.empty(64, dtype=np.float64)
    stack_starts[0] = start
    stack_counts[0] = count
    top = 0
    result = 0.0
    while top >= 0:
        local_start = int(stack_starts[top])
        local_count = int(stack_counts[top])
        state = int(stack_states[top])
        if state == 0 and local_count > 128:
            left_count = local_count // 2
            left_count -= left_count % 8
            stack_states[top] = 1
            top += 1
            stack_starts[top] = local_start
            stack_counts[top] = left_count
            stack_states[top] = 0
            continue
        if state == 1:
            left_count = local_count // 2
            left_count -= left_count % 8
            stack_left_results[top] = result
            stack_states[top] = 2
            top += 1
            stack_starts[top] = local_start + left_count
            stack_counts[top] = local_count - left_count
            stack_states[top] = 0
            continue
        if state == 2:
            result = stack_left_results[top] + result
            top -= 1
            continue

        if local_count < 8:
            result = 0.0
            for index in range(local_count):
                result += values[local_start + index]
        else:
            accumulators = np.empty(8, dtype=np.float64)
            for lane in range(8):
                accumulators[lane] = values[local_start + lane]
            aligned_end = local_count - (local_count % 8)
            position = 8
            while position < aligned_end:
                for lane in range(8):
                    accumulators[lane] += values[local_start + position + lane]
                position += 8
            result = (
                (accumulators[0] + accumulators[1])
                + (accumulators[2] + accumulators[3])
            ) + (
                (accumulators[4] + accumulators[5])
                + (accumulators[6] + accumulators[7])
            )
            while position < local_count:
                result += values[local_start + position]
                position += 1
        top -= 1
    return result


@njit(cache=True)
def _ordered_weighted_cdf(
    values: np.ndarray,
    weights: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    order = np.argsort(values[:count], kind="mergesort")
    ordered_values = np.empty(count, dtype=np.float64)
    ordered_weights = np.empty(count, dtype=np.float64)
    valid = True
    for position in range(count):
        source = int(order[position])
        ordered_values[position] = float(values[source])
        weight = float(weights[source])
        ordered_weights[position] = weight
        if not np.isfinite(weight):
            valid = False
    total = _numpy_pairwise_sum(ordered_weights, 0, count)
    if not valid or total <= 0.0:
        total = float(count)
        for position in range(count):
            ordered_weights[position] = 1.0
    cdf = np.empty(count, dtype=np.float64)
    cumulative = 0.0
    for position in range(count):
        cumulative += ordered_weights[position]
        cdf[position] = cumulative / total
    return ordered_values, cdf


@njit(cache=True)
def _weighted_quantile_history(
    values: np.ndarray,
    labels: np.ndarray,
    start: int,
    end: int,
    issue_ns: np.int64,
    cadence_hours: float,
    decay: float,
    quantile: float,
) -> float:
    count = end - start
    if count <= 0:
        return 0.0
    local_values = np.empty(count, dtype=np.float64)
    local_weights = np.empty(count, dtype=np.float64)
    for position in range(count):
        source = start + position
        age_steps = max(float(issue_ns - labels[source]) / float(HOUR_NS) / cadence_hours, 0.0)
        local_values[position] = values[source]
        local_weights[position] = decay ** age_steps
    ordered_values, cdf = _ordered_weighted_cdf(local_values, local_weights, count)
    return _interp_cdf(quantile, cdf, ordered_values, count)


@njit(cache=True)
def _kowcpi_bounds(
    values: np.ndarray,
    labels: np.ndarray,
    start: int,
    end: int,
    issue_ns: np.int64,
    lag_ns: np.int64,
    bandwidth: float,
    alpha: float,
) -> tuple[float, float]:
    count = end - start
    if count <= 0:
        return 0.0, 0.0
    current_cutoff = issue_ns - lag_ns
    current_start = start
    while current_start < end and labels[current_start] < current_cutoff:
        current_start += 1
    current_context = 0.0
    if current_start < end:
        current_context = float(np.mean(values[current_start:end]))

    prefix = np.empty(count + 1, dtype=np.float64)
    prefix[0] = 0.0
    for position in range(count):
        prefix[position + 1] = prefix[position] + values[start + position]

    candidate_values = np.empty(count, dtype=np.float64)
    candidate_weights = np.empty(count, dtype=np.float64)
    candidate_count = 0
    for position in range(count):
        index = start + position
        historical_cutoff = labels[index] - lag_ns
        historical_start_position = 0
        while (
            historical_start_position < position
            and labels[start + historical_start_position] < historical_cutoff
        ):
            historical_start_position += 1
        historical_count = position - historical_start_position
        if historical_count <= 0:
            continue
        historical_context = (
            prefix[position] - prefix[historical_start_position]
        ) / float(historical_count)
        distance = abs(historical_context - current_context)
        u = distance / max(bandwidth, 1e-12)
        weight = 0.0
        if abs(u) <= 1.0:
            weight = max(0.75 * (1.0 - u * u), 0.0)
        candidate_values[candidate_count] = values[index]
        candidate_weights[candidate_count] = weight
        candidate_count += 1

    if candidate_count == 0:
        for index in range(start, end):
            position = index - start
            candidate_values[position] = values[index]
            candidate_weights[position] = 1.0
        candidate_count = count
    ordered_values, cdf = _ordered_weighted_cdf(
        candidate_values,
        candidate_weights,
        candidate_count,
    )
    best_width = np.inf
    best_lower = 0.0
    best_upper = 0.0
    beta_grid = np.linspace(0.0, alpha, 21)
    for beta_index in range(21):
        beta = float(beta_grid[beta_index])
        lower = _interp_cdf(beta, cdf, ordered_values, candidate_count)
        upper = _interp_cdf(1.0 - alpha + beta, cdf, ordered_values, candidate_count)
        width = upper - lower
        if upper >= lower and width < best_width:
            best_width = width
            best_lower = lower
            best_upper = upper
    return best_lower, best_upper


@njit(cache=True, boundscheck=True)
def _evaluate_grid(
    family_codes: np.ndarray,
    parameters: np.ndarray,
    coverages: np.ndarray,
    calibration_scores: np.ndarray,
    calibration_residuals: np.ndarray,
    calibration_labels: np.ndarray,
    static_margins: np.ndarray,
    reference_widths: np.ndarray,
    test_scores: np.ndarray,
    test_residuals: np.ndarray,
    base_lower: np.ndarray,
    base_upper: np.ndarray,
    base_center: np.ndarray,
    target: np.ndarray,
    issue_ns: np.ndarray,
    label_ns: np.ndarray,
    feedback_order: np.ndarray,
    matured_ends: np.ndarray,
    cadence_hours: float,
) -> tuple[np.ndarray, ...]:
    configuration_count = len(family_codes)
    coverage_count = len(coverages)
    event_count = len(target)
    calibration_count = len(calibration_residuals)
    shape = (configuration_count, coverage_count, event_count)
    lower_output = np.empty(shape, dtype=np.float64)
    upper_output = np.empty(shape, dtype=np.float64)
    raw_lower_output = np.empty(shape, dtype=np.float64)
    raw_upper_output = np.empty(shape, dtype=np.float64)
    state_output = np.empty(shape, dtype=np.float64)

    for task_index in range(configuration_count * coverage_count):
        configuration_index = task_index // coverage_count
        coverage_index = task_index % coverage_count
        family = int(family_codes[configuration_index])
        config = parameters[configuration_index]
        coverage = float(coverages[coverage_index])
        alpha = 1.0 - coverage
        static_margin = float(static_margins[coverage_index])
        reference_width = float(reference_widths[coverage_index])
        offset = static_margin
        offsets = np.full(4, static_margin, dtype=np.float64)
        expert_losses = np.zeros(4, dtype=np.float64)
        payloads = np.zeros((event_count, 4), dtype=np.float64)

        history_values = np.empty(calibration_count + event_count, dtype=np.float64)
        history_labels = np.empty(calibration_count + event_count, dtype=np.int64)
        history_start = 0
        history_end = calibration_count
        if family in (4, 6, 7):
            for index in range(calibration_count):
                history_values[index] = calibration_scores[coverage_index, index]
                history_labels[index] = calibration_labels[index]
        elif family == 8:
            for index in range(calibration_count):
                history_values[index] = calibration_residuals[index]
                history_labels[index] = calibration_labels[index]

        recent_values = np.empty(event_count, dtype=np.float64)
        recent_labels = np.empty(event_count, dtype=np.int64)
        recent_start = 0
        recent_end = 0
        recent_sum = 0.0

        coverage_values = np.empty((event_count, 4), dtype=np.float64)
        coverage_labels = np.empty(event_count, dtype=np.int64)
        coverage_starts = np.zeros(4, dtype=np.int32)
        coverage_end = 0
        coverage_sums = np.zeros(4, dtype=np.float64)

        feedback_pointer = 0
        for event_index in range(event_count):
            matured_end = int(matured_ends[event_index])
            while feedback_pointer < matured_end:
                source = int(feedback_order[feedback_pointer])
                application_time = issue_ns[event_index]
                if family == 1:
                    offset = max(0.0, offset + config[0] * (payloads[source, 0] - alpha))
                elif family == 2:
                    for expert in range(4):
                        rate = config[expert]
                        miss = payloads[source, expert]
                        offsets[expert] = max(0.0, offsets[expert] + rate * (miss - alpha))
                        expert_losses[expert] += miss
                elif family == 3:
                    recent_values[recent_end] = payloads[source, 0]
                    recent_labels[recent_end] = label_ns[source]
                    recent_sum += payloads[source, 0]
                    recent_end += 1
                    cutoff = application_time - np.int64(config[2] * float(HOUR_NS))
                    while recent_start < recent_end and recent_labels[recent_start] < cutoff:
                        recent_sum -= recent_values[recent_start]
                        recent_start += 1
                    miss_rate = recent_sum / float(recent_end - recent_start)
                    eta = config[0] + (config[1] - config[0]) * min(
                        1.0,
                        abs(miss_rate - alpha) / max(alpha, 1e-12),
                    )
                    offset = max(0.0, offset + eta * (payloads[source, 0] - alpha))
                elif family in (4, 6, 7):
                    history_values[history_end] = test_scores[coverage_index, source]
                    history_labels[history_end] = label_ns[source]
                    history_end += 1
                elif family == 5:
                    coverage_labels[coverage_end] = label_ns[source]
                    for expert in range(4):
                        value = 1.0 - payloads[source, expert]
                        coverage_values[coverage_end, expert] = value
                        coverage_sums[expert] += value
                        cutoff = application_time - np.int64(config[1 + expert] * float(HOUR_NS))
                        while (
                            coverage_starts[expert] < coverage_end + 1
                            and coverage_labels[coverage_starts[expert]] < cutoff
                        ):
                            coverage_sums[expert] -= coverage_values[coverage_starts[expert], expert]
                            coverage_starts[expert] += 1
                        scale = np.sqrt(config[1] / config[1 + expert])
                        offsets[expert] = max(
                            0.0,
                            offsets[expert] + config[0] * scale * (payloads[source, expert] - alpha),
                        )
                    coverage_end += 1
                elif family == 8:
                    history_values[history_end] = test_residuals[source]
                    history_labels[history_end] = label_ns[source]
                    history_end += 1
                elif family == 9:
                    offset = max(
                        0.0,
                        offset + config[0] * payloads[source, 1] * (payloads[source, 0] - alpha),
                    )
                feedback_pointer += 1

            base_lo = float(base_lower[coverage_index, event_index])
            base_hi = float(base_upper[coverage_index, event_index])
            center = float(base_center[event_index])
            target_value = float(target[event_index])
            state_value = 0.0
            raw_lower = base_lo
            raw_upper = base_hi

            if family == 0:
                raw_lower = base_lo - static_margin
                raw_upper = base_hi + static_margin
                state_value = static_margin
            elif family == 1:
                raw_lower = base_lo - offset
                raw_upper = base_hi + offset
                state_value = offset
            elif family == 2:
                minimum_loss = np.min(expert_losses)
                weights = np.empty(4, dtype=np.float64)
                denominator = 0.0
                for expert in range(4):
                    weights[expert] = np.exp(-(expert_losses[expert] - minimum_loss))
                    denominator += weights[expert]
                raw_lower = 0.0
                raw_upper = 0.0
                state_value = 0.0
                for expert in range(4):
                    weights[expert] /= denominator
                    raw_lower += weights[expert] * (base_lo - offsets[expert])
                    raw_upper += weights[expert] * (base_hi + offsets[expert])
                    state_value += weights[expert] * offsets[expert]
                    expert_lower, expert_upper = _clip_pair(
                        base_lo - offsets[expert],
                        base_hi + offsets[expert],
                    )
                    payloads[event_index, expert] = (
                        0.0 if expert_lower <= target_value <= expert_upper else 1.0
                    )
            elif family == 3:
                raw_lower = base_lo - offset
                raw_upper = base_hi + offset
                state_value = offset
            elif family == 4:
                cutoff = issue_ns[event_index] - np.int64(config[0] * float(HOUR_NS))
                while history_start < history_end and history_labels[history_start] < cutoff:
                    history_start += 1
                margin = max(
                    0.0,
                    _linear_quantile(history_values, history_start, history_end, coverage),
                )
                raw_lower = base_lo - margin
                raw_upper = base_hi + margin
                state_value = margin
            elif family == 5:
                errors = np.empty(4, dtype=np.float64)
                minimum_error = np.inf
                for expert in range(4):
                    cutoff = issue_ns[event_index] - np.int64(config[1 + expert] * float(HOUR_NS))
                    while (
                        coverage_starts[expert] < coverage_end
                        and coverage_labels[coverage_starts[expert]] < cutoff
                    ):
                        coverage_sums[expert] -= coverage_values[coverage_starts[expert], expert]
                        coverage_starts[expert] += 1
                    count = coverage_end - int(coverage_starts[expert])
                    observed = coverage if count == 0 else coverage_sums[expert] / float(count)
                    errors[expert] = abs(observed - coverage)
                    minimum_error = min(minimum_error, errors[expert])
                weights = np.empty(4, dtype=np.float64)
                denominator = 0.0
                for expert in range(4):
                    weights[expert] = np.exp(
                        -(errors[expert] - minimum_error) / max(config[5], 1e-12)
                    )
                    denominator += weights[expert]
                raw_lower = 0.0
                raw_upper = 0.0
                state_value = 0.0
                for expert in range(4):
                    weights[expert] /= denominator
                    raw_lower += weights[expert] * (base_lo - offsets[expert])
                    raw_upper += weights[expert] * (base_hi + offsets[expert])
                    state_value += weights[expert] * offsets[expert]
                    expert_lower, expert_upper = _clip_pair(
                        base_lo - offsets[expert],
                        base_hi + offsets[expert],
                    )
                    payloads[event_index, expert] = (
                        0.0 if expert_lower <= target_value <= expert_upper else 1.0
                    )
            elif family in (6, 7):
                cutoff = issue_ns[event_index] - np.int64(config[0] * float(HOUR_NS))
                while history_start < history_end and history_labels[history_start] < cutoff:
                    history_start += 1
                margin = max(
                    0.0,
                    _weighted_quantile_history(
                        history_values,
                        history_labels,
                        history_start,
                        history_end,
                        issue_ns[event_index],
                        cadence_hours,
                        config[1],
                        coverage,
                    ),
                )
                raw_lower = base_lo - margin
                raw_upper = base_hi + margin
                state_value = margin
            elif family == 8:
                cutoff = issue_ns[event_index] - np.int64(config[0] * float(HOUR_NS))
                while history_start < history_end and history_labels[history_start] < cutoff:
                    history_start += 1
                q_low, q_high = _kowcpi_bounds(
                    history_values,
                    history_labels,
                    history_start,
                    history_end,
                    issue_ns[event_index],
                    np.int64(config[1] * float(HOUR_NS)),
                    config[2],
                    alpha,
                )
                raw_lower = center + q_low
                raw_upper = center + q_high
                state_value = q_high - q_low
            elif family == 9:
                width = max(base_hi - base_lo, 1e-12)
                multiplier = (max(reference_width, 1e-12) / width) ** config[1]
                multiplier = min(max(multiplier, config[2]), config[3])
                raw_lower = base_lo - offset
                raw_upper = base_hi + offset
                state_value = offset
                payloads[event_index, 1] = multiplier

            lower, upper = _clip_pair(raw_lower, raw_upper)
            if family in (1, 3, 9):
                payloads[event_index, 0] = 0.0 if lower <= target_value <= upper else 1.0
            raw_lower_output[configuration_index, coverage_index, event_index] = raw_lower
            raw_upper_output[configuration_index, coverage_index, event_index] = raw_upper
            lower_output[configuration_index, coverage_index, event_index] = lower
            upper_output[configuration_index, coverage_index, event_index] = upper
            state_output[configuration_index, coverage_index, event_index] = state_value

    return lower_output, upper_output, raw_lower_output, raw_upper_output, state_output


def evaluate_conformal_grid_compact(
    *,
    contracts: FrozenContracts,
    coverages: np.ndarray,
    calibration_target: np.ndarray,
    calibration_lower: np.ndarray,
    calibration_upper: np.ndarray,
    calibration_center: np.ndarray,
    calibration_label_ns: np.ndarray,
    test_target: np.ndarray,
    test_lower: np.ndarray,
    test_upper: np.ndarray,
    test_center: np.ndarray,
    test_issue_ns: np.ndarray,
    test_label_ns: np.ndarray,
    feedback_order: np.ndarray,
    matured_ends: np.ndarray,
    cadence_hours: float,
    configuration_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    coverage_values = np.asarray(coverages, dtype=np.float64)
    calibration_target_values = np.asarray(calibration_target, dtype=np.float64)
    test_target_values = np.asarray(test_target, dtype=np.float64)
    calibration_lower_values = np.asarray(calibration_lower, dtype=np.float64)
    calibration_upper_values = np.asarray(calibration_upper, dtype=np.float64)
    test_lower_values = np.asarray(test_lower, dtype=np.float64)
    test_upper_values = np.asarray(test_upper, dtype=np.float64)
    calibration_center_values = np.asarray(calibration_center, dtype=np.float64)
    test_center_values = np.asarray(test_center, dtype=np.float64)
    if calibration_lower_values.shape != calibration_upper_values.shape:
        raise ValueError("紧凑conformal校准端点形状失配")
    if test_lower_values.shape != test_upper_values.shape:
        raise ValueError("紧凑conformal测试端点形状失配")
    if calibration_lower_values.shape != (len(coverage_values), len(calibration_target_values)):
        raise ValueError("紧凑conformal校准覆盖率形状失配")
    if test_lower_values.shape != (len(coverage_values), len(test_target_values)):
        raise ValueError("紧凑conformal测试覆盖率形状失配")
    arrays = [
        coverage_values,
        calibration_target_values,
        test_target_values,
        calibration_lower_values,
        calibration_upper_values,
        test_lower_values,
        test_upper_values,
        calibration_center_values,
        test_center_values,
    ]
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("紧凑conformal输入含非有限值")
    calibration_scores = np.maximum(
        np.maximum(calibration_lower_values - calibration_target_values[None, :], calibration_target_values[None, :] - calibration_upper_values),
        0.0,
    )
    test_scores = np.maximum(
        np.maximum(test_lower_values - test_target_values[None, :], test_target_values[None, :] - test_upper_values),
        0.0,
    )
    calibration_residuals = calibration_target_values - calibration_center_values
    test_residuals = test_target_values - test_center_values
    static_margins = np.asarray(
        [
            max(0.0, float(np.quantile(calibration_scores[index], coverage, method="linear")))
            for index, coverage in enumerate(coverage_values)
        ],
        dtype=np.float64,
    )
    reference_widths = np.asarray(
        [
            float(np.quantile(calibration_upper_values[index] - calibration_lower_values[index], 0.5, method="linear"))
            for index in range(len(coverage_values))
        ],
        dtype=np.float64,
    )
    family_codes, parameters = compact_configuration_arrays(contracts)
    if configuration_indices is not None:
        indices = np.asarray(configuration_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError("紧凑conformal配置索引必须为非空一维数组")
        if np.any(indices < 0) or np.any(indices >= len(family_codes)):
            raise ValueError("紧凑conformal配置索引越界")
        family_codes = family_codes[indices]
        parameters = parameters[indices]
    return _evaluate_grid(
        family_codes,
        parameters,
        coverage_values,
        calibration_scores,
        calibration_residuals,
        np.asarray(calibration_label_ns, dtype=np.int64),
        static_margins,
        reference_widths,
        test_scores,
        test_residuals,
        test_lower_values,
        test_upper_values,
        test_center_values,
        test_target_values,
        np.asarray(test_issue_ns, dtype=np.int64),
        np.asarray(test_label_ns, dtype=np.int64),
        np.asarray(feedback_order, dtype=np.int32),
        np.asarray(matured_ends, dtype=np.int32),
        float(cadence_hours),
    )


@njit(cache=True)
def summarize_conformal_grid_compact(
    lower: np.ndarray,
    upper: np.ndarray,
    target: np.ndarray,
    center: np.ndarray,
    coverages: np.ndarray,
    label_ns: np.ndarray,
    application_ns: np.ndarray,
    feedback_order: np.ndarray,
    full_history: np.ndarray,
    cadence_minutes: float,
    pi_plus: float,
    pi_minus: float,
    kappa_plus: float,
    kappa_minus: float,
) -> tuple[np.ndarray, ...]:
    configuration_count, coverage_count, event_count = lower.shape
    task_count = configuration_count * coverage_count
    errf_sums = np.zeros((configuration_count, coverage_count), dtype=np.float64)
    covered_sums = np.zeros((configuration_count, coverage_count), dtype=np.int64)
    tuwr_sums = np.zeros((configuration_count, coverage_count), dtype=np.float64)
    tuwr_counts = np.zeros((configuration_count, coverage_count), dtype=np.int64)
    reliability_failures = np.zeros((configuration_count, coverage_count), dtype=np.int64)
    expected_history = int(np.ceil(168.0 * 60.0 / cadence_minutes))
    subwindow = max(1, int(np.ceil(24.0 * 60.0 / cadence_minutes)))
    rolling_window_count = expected_history - subwindow + 1
    cadence_ns = np.int64(round(cadence_minutes * 60.0 * 1e9))
    tolerance_ns = np.int64(60)
    scale = cadence_minutes / 60.0

    for task_index in range(task_count):
        configuration_index = task_index // coverage_count
        coverage_index = task_index % coverage_count
        coverage = float(coverages[coverage_index])
        covered = np.empty(event_count, dtype=np.uint8)
        local_errf = 0.0
        local_covered = 0
        for event_index in range(event_count):
            lo = float(lower[configuration_index, coverage_index, event_index])
            hi = float(upper[configuration_index, coverage_index, event_index])
            target_value = float(target[event_index])
            center_value = float(center[event_index])
            local_errf += scale * (
                max(hi - center_value, 0.0) * pi_plus
                + max(center_value - lo, 0.0) * pi_minus
                + max(target_value - hi, 0.0) * kappa_plus
                + max(lo - target_value, 0.0) * kappa_minus
            )
            value = 1 if lo <= target_value <= hi else 0
            covered[event_index] = value
            local_covered += value

        history_labels = np.empty(event_count, dtype=np.int64)
        history_covered = np.empty(event_count, dtype=np.float64)
        coverage_prefix = np.zeros(event_count + 1, dtype=np.float64)
        under_prefix = np.zeros(event_count + 1, dtype=np.int64)
        history_start = 0
        history_end = 0
        window_count = 0
        local_tuwr_sum = 0.0
        local_tuwr_count = 0
        local_failures = 0
        tau = 1.96 * np.sqrt(coverage * (1.0 - coverage) / float(subwindow))
        threshold = coverage - tau
        for order_position in range(event_count):
            event_index = int(feedback_order[order_position])
            history_labels[history_end] = label_ns[event_index]
            history_covered[history_end] = float(covered[event_index])
            coverage_prefix[history_end + 1] = coverage_prefix[history_end] + history_covered[history_end]
            history_end += 1
            under_prefix[window_count + 1] = under_prefix[window_count]
            if history_end >= subwindow:
                window_sum = coverage_prefix[history_end] - coverage_prefix[history_end - subwindow]
                under_prefix[window_count + 1] = under_prefix[window_count] + int(
                    window_sum / float(subwindow) < threshold
                )
                window_count += 1

            cutoff = application_ns[event_index] - np.int64(168.0 * float(HOUR_NS))
            while history_start < history_end and history_labels[history_start] < cutoff:
                history_start += 1
            if full_history[coverage_index, event_index] == 0:
                continue
            eligible_count = history_end - history_start
            if eligible_count < expected_history:
                local_failures += 1
                continue
            selected_start = history_end - expected_history
            oldest_allowed = cutoff + cadence_ns + tolerance_ns
            if history_labels[selected_start] > oldest_allowed:
                local_failures += 1
                continue
            first_window = window_count - rolling_window_count
            if first_window < 0:
                local_failures += 1
                continue
            under_count = under_prefix[window_count] - under_prefix[first_window]
            local_tuwr_sum += float(under_count) / float(rolling_window_count)
            local_tuwr_count += 1

        errf_sums[configuration_index, coverage_index] = local_errf
        covered_sums[configuration_index, coverage_index] = local_covered
        tuwr_sums[configuration_index, coverage_index] = local_tuwr_sum
        tuwr_counts[configuration_index, coverage_index] = local_tuwr_count
        reliability_failures[configuration_index, coverage_index] = local_failures
    return errf_sums, covered_sums, tuwr_sums, tuwr_counts, reliability_failures
