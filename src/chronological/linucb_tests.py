"""Equivalence and future-feedback tests for the corrected LinUCB runner."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import numba
from linucb_forward import (Timeline, _schedule, run_grid, CONTEXT_FEATURES,
                           REFERENCE_RATIO, CONFIGURATIONS)


def synthetic_timeline(issues: int, per_issue: int, seed: int = 71) -> Timeline:
    rng = np.random.default_rng(seed)
    n = issues * per_issue
    issue = np.repeat(np.arange(issues, dtype=np.int64) * 3600_000_000_000, per_issue)
    horizons = np.take([1, 3, 6, 12, 24], np.arange(n) % 5)
    label = issue + horizons * 3600_000_000_000
    available = label + 3600_000_000_000
    context = rng.integers(0, 6600, n, dtype=np.uint16)
    cap = rng.uniform(.04, .4, (n, 6))
    miss = rng.exponential(.05, (n, 6)) * (rng.random((n, 6)) < .2)
    hit = (miss == 0).astype(np.uint8)
    tuwr = rng.uniform(0, .6, (n, 6))
    offsets, feedback_order, matured = _schedule(issue, label, available)
    return Timeline(issue, label, available, context, rng.integers(0, 11, n, dtype=np.uint8),
                    cap, miss, hit, tuwr, offsets, feedback_order, matured,
                    np.zeros(n, dtype=np.uint8), np.arange(n, dtype=np.int32),
                    (n // 11,), "synthetic", 0, "source")


def reference_eventwise(timeline: Timeline, alpha: float, l2: float, ratio: float):
    inverse = np.repeat((np.eye(31) / l2)[None], 6, axis=0)
    response = np.zeros((6, 31))
    decisions = np.zeros(timeline.size, dtype=np.uint8)
    contexts = np.zeros((6600, 31))
    contexts[np.arange(6600)[:, None], CONTEXT_FEATURES] = 1
    losses = 3.56 * (timeline.capacity + ratio * timeline.exceedance)
    pointer = 0
    for step in range(len(timeline.issue_offsets) - 1):
        while pointer < timeline.matured_ends[step]:
            event = timeline.feedback_order[pointer]
            action = decisions[event]
            x = contexts[timeline.context_ids[event]]
            projection = inverse[action] @ x
            inverse[action] -= np.outer(projection, projection) / (1 + x @ projection)
            response[action] += x * losses[event, action]
            pointer += 1
        begin, end = timeline.issue_offsets[step:step + 2]
        X = contexts[timeline.context_ids[begin:end]]
        scores = np.empty((end - begin, 6))
        for action in range(6):
            theta = inverse[action] @ response[action]
            scores[:, action] = X @ theta - alpha * np.sqrt(np.maximum(np.sum((X @ inverse[action]) * X, axis=1), 0))
        decisions[begin:end] = (scores <= scores.min(axis=1, keepdims=True) + 1e-12).argmax(axis=1)
    return decisions


def test_equivalence_and_causality():
    timeline = synthetic_timeline(50, 44)
    configurations = ((.1, .1), (.5, 1.), (2., 10.))
    actual = run_grid(timeline, configurations)
    for i, (alpha, l2) in enumerate(configurations):
        expected = reference_eventwise(timeline, alpha, l2, REFERENCE_RATIO)
        assert np.array_equal(actual["choices"][i], expected), (i, np.flatnonzero(actual["choices"][i] != expected)[:10])
    # Outcomes unavailable by a cutoff cannot influence decisions through that cutoff.
    cutoff = 30 * 3600_000_000_000
    old = timeline.exceedance.copy()
    timeline.exceedance[timeline.available_ns > cutoff] += 10
    altered = run_grid(timeline, configurations)
    mask = timeline.issue_ns <= cutoff
    assert np.array_equal(actual["choices"][:, mask], altered["choices"][:, mask])
    timeline.exceedance = old
    # The source/target protocol uses bandit feedback: changing losses of
    # unselected actions cannot alter this configuration's decision sequence.
    one = run_grid(timeline, (configurations[0],))
    unselected = np.ones_like(timeline.exceedance, dtype=bool)
    unselected[np.arange(timeline.size), one["choices"][0]] = False
    timeline.exceedance = old.copy()
    timeline.exceedance[unselected] += 100.0
    perturbed = run_grid(timeline, (configurations[0],))
    assert np.array_equal(one["choices"], perturbed["choices"])
    timeline.exceedance = old
    assert actual["audit"]["feedback_count_before_last_issue"] == int(timeline.matured_ends[-1])
    return {"eventwise_reference_equal": True, "future_outcome_perturbation_equal": True,
            "unselected_outcome_perturbation_equal": True,
            "tested_configurations": len(configurations), "event_count": timeline.size}


def test_real_stream():
    import forward_common as common
    import clara_forward as clara
    import linucb_forward as linucb
    stream = common.load_stream("zone1", 0, "GBR", 1, "source", 15)
    clara._verify_source_times(stream)
    n = len(stream["issue_ns"])
    # The state-code test does not fit a model; fixed edges exercise all bins.
    q = np.broadcast_to(np.array([.1, .4]), (4, 5, 11, 2)).copy()
    codes = linucb._state_codes(stream, 1, 0, q)
    assert np.array_equal(codes, clara.state_codes(stream, "GBR", 1, q))
    cap, miss, hit = common.endpoint_components(stream["lower"], stream["upper"],
                                                stream["center"][None, None, :],
                                                stream["y"][None, None, :])
    issue = np.tile(stream["issue_ns"], 11)
    order = np.argsort(issue, kind="stable")
    issue = issue[order]
    label = np.tile(stream["label_ns"], 11)[order]
    available = np.tile(stream["available_ns"], 11)[order]
    offsets, feedback_order, matured = _schedule(issue, label, available)
    timeline = Timeline(issue, label, available, codes.ravel()[order],
                        np.repeat(np.arange(11, dtype=np.uint8), n)[order],
                        cap.reshape(6, -1).T[order], miss.reshape(6, -1).T[order],
                        hit.reshape(6, -1).T.astype(np.uint8)[order],
                        stream["candidate_tuwr"].reshape(6, -1).T[order],
                        offsets, feedback_order, matured,
                        np.zeros(11 * n, dtype=np.uint8), np.arange(11 * n)[order],
                        (n,), "zone1", 0, "source")
    configs = ((.1, .1), (.5, 1.), (2., 10.))
    run = run_grid(timeline, configs)
    for i, (alpha, l2) in enumerate(configs):
        expected = reference_eventwise(timeline, alpha, l2, REFERENCE_RATIO)
        assert np.array_equal(run["choices"][i], expected), (i, np.flatnonzero(run["choices"][i] != expected)[:10])
    return {"real_source_state_codes_equal": True, "real_source_decisions_equal": True,
            "event_count": timeline.size, "source_feedback_precedes_test": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--issues", type=int, default=1750)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    numba.set_num_threads(args.threads)
    report = test_equivalence_and_causality()
    print(json.dumps(report), flush=True)
    if args.real:
        report["real_source"] = test_real_stream()
        print(json.dumps(report["real_source"]), flush=True)
    if args.benchmark:
        timeline = synthetic_timeline(args.issues, 220)
        started = perf_counter()
        result = run_grid(timeline)
        report["benchmark"] = {"events": timeline.size, "configurations": len(CONFIGURATIONS),
                               "threads": args.threads, "seconds": perf_counter() - started}
        print(json.dumps(report["benchmark"]), flush=True)
    (Path.cwd() / "outputs" / "linucb_test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    (Path.cwd()/"outputs").mkdir(exist_ok=True)
    main()
