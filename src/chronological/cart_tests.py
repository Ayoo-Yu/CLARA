"""Checks for corrected CART data boundaries and event-weighted evaluation."""
from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier

import cart_forward as cart
from clara_forward import make_states


def test_compact_metrics_and_time_boundary():
    rng = np.random.default_rng(9147)
    states = make_states()
    n = 501
    y = rng.random(n)
    center = np.clip(y + rng.normal(0, .17, n), 0, 1)
    half_widths = np.linspace(.05, .39, 6)[:, None, None]
    lower = np.maximum(center[None, None, :] - half_widths * cart.COVERAGES[None, :, None], 0)
    upper = np.minimum(center[None, None, :] + half_widths * cart.COVERAGES[None, :, None], 1)
    # Include an exact eventwise tie: Static must be preferred to TSC.
    lower[4] = lower[0]
    upper[4] = upper[0]
    issue = np.arange(n, dtype=np.int64) * 3_600_000_000_000
    tuwr = rng.random((6, 11, n))
    tuwr[:, :, :168] = np.nan
    stream = dict(lower=lower, upper=upper, y=y, center=center,
                  issue_ns=issue, label_ns=issue + 3_600_000_000_000,
                  available_ns=issue + 7_200_000_000_000, candidate_tuwr=tuwr)
    codes = np.empty((11, n), dtype=np.int64)
    for c in range(11):
        codes[c] = ((c * 2 + rng.integers(0, 2, n)) * 5 + rng.integers(0, 5, n)) * 3 + rng.integers(0, 3, n)
    statistics = cart.CartStatistics.empty(len(states), ("zone2",))
    statistics.add_stream(stream, codes, cutoff_ns=int(stream["available_ns"].max()) + 1)
    statistics.validate()
    assert statistics.best_count[:, 4].sum() == 0
    choices = rng.integers(0, 6, len(states))
    metrics = cart.selected_metrics(statistics, choices, states)
    chosen = choices[codes]
    cv = np.arange(11)[:, None]
    t = np.arange(n)[None, :]
    low = lower[chosen, cv, t]
    up = upper[chosen, cv, t]
    cap = np.maximum(up - center, 0) + np.maximum(center - low, 0)
    miss = np.maximum(y - up, 0) + np.maximum(low - y, 0)
    expected = {
        "mean_errf": float((3.56 * cap + 20 * miss).mean()),
        "mean_coverage_gap": float(((low <= y) & (y <= up)).mean() - cart.COVERAGES.mean()),
        "mean_tuwr": float(np.nanmean(tuwr[chosen, cv, t])),
    }
    for name, value in expected.items():
        np.testing.assert_allclose(metrics[name], value, rtol=2e-14, atol=2e-14)
    # Eq. (3) admits feedback available exactly at the cutoff, provided the
    # label itself and issue both precede it.
    at_cutoff = cart.CartStatistics.empty(len(states), ("zone2",))
    at_cutoff.add_stream(stream, codes, cutoff_ns=int(stream["available_ns"].max()))
    assert at_cutoff.event_rows == codes.size
    for cutoff in (int(stream["available_ns"].max()) - 1, int(stream["label_ns"].max())):
        try:
            cart.CartStatistics.empty(len(states), ("zone2",)).add_stream(stream, codes, cutoff_ns=cutoff)
        except ValueError as exc:
            assert "availability <= cutoff" in str(exc)
        else:
            raise AssertionError("Unavailable feedback or a label at the cutoff was accepted")
    print("PASS: eventwise validation, ties, inclusive availability and strictly earlier labels")



def test_capped_cart_against_eventwise_fit():
    rng = np.random.default_rng(2148)
    # Unique one-hot states provide a transparent fully expanded reference.
    feature_matrix = np.eye(32, dtype=float)
    counts = rng.integers(1, 32, (32, 6))
    counts[np.arange(32), np.arange(32) % 6] += 250
    state_ids = []
    full_labels = []
    for row in range(len(counts)):
        for action, label in enumerate(cart.ACTIONS):
            state_ids.extend([row] * int(counts[row, action]))
            full_labels.extend([label] * int(counts[row, action]))
    full_matrix = feature_matrix[np.asarray(state_ids)]
    checked = 0
    for config in cart.configuration_grid():
        policy = cart.fit_counts(counts, feature_matrix=feature_matrix, config=config, fit_zones=("zone1",))
        reference = DecisionTreeClassifier(max_depth=config["max_depth"], min_samples_leaf=config["min_samples_leaf"],
                                           class_weight=config["class_weight"], random_state=0)
        reference.fit(full_matrix, np.asarray(full_labels))
        np.testing.assert_array_equal(policy.classifier.predict(feature_matrix), reference.predict(feature_matrix))
        np.testing.assert_allclose(policy.classifier.predict_proba(feature_matrix), reference.predict_proba(feature_matrix), atol=1e-11, rtol=1e-11)
        checked += 1
    print(f"PASS: original compact CART agrees with full event expansion for all {checked} configurations")


def test_shared_training_matrix():
    rng = np.random.default_rng(721)
    states = make_states()
    x = cart.encode_states(states)
    training = cart.CartStatistics.empty(6600, tuple(f"zone{i}" for i in range(3, 11)))
    validation = cart.CartStatistics.empty(6600, ("zone2",))
    positions = rng.choice(6600, 110, replace=False)
    for stat in (training, validation):
        stat.best_count[positions] = rng.integers(1, 150, (110, 6))
        stat.event_count = stat.best_count.sum(axis=1)
        stat.event_rows = int(stat.event_count.sum())
        stat.errf_sum[positions] = rng.random((110, 6)) * stat.event_count[positions, None]
        stat.covered_sum[positions] = np.floor(rng.random((110, 6)) * stat.event_count[positions, None])
        stat.tuwr_count[positions] = stat.event_count[positions, None]
        stat.tuwr_sum[positions] = rng.random((110, 6)) * stat.event_count[positions, None]
    scores = cart.tune_fold(training, validation, states=states, outer_zone="zone1", validation_zone="zone2")
    for rank, config in enumerate(cart.configuration_grid()):
        policy = cart.fit_counts(training.best_count, feature_matrix=x, config=config, fit_zones=training.source_zones)
        metrics = cart.selected_metrics(validation, policy.state_actions, states)
        for name in ("mean_errf", "mean_coverage_gap", "mean_tuwr"):
            np.testing.assert_equal(scores.iloc[rank][name], metrics[name])
    print("PASS: all30 shared-matrix grid fits produce identical validation metrics to independent original fits")


def test_real_source_matches_clara_label_counts():
    from forward_common import dataset_path, load_stream
    from clara_forward import state_codes, new_arrays, add_stream_statistics
    path = dataset_path("zone1", 0, "GBR", 1, "source")
    if not path.exists():
        print("SKIP: generated source pilot is not yet available")
        return
    stream = load_stream("zone1", 0, "GBR", 1, "source", 0)
    edges = np.tile([.1, .3], (4, 5, 11, 1))
    codes = state_codes(stream, "GBR", 1, edges)
    statistics = cart.CartStatistics.empty(6600, ("zone1",))
    statistics.add_stream(stream, codes, cutoff_ns=int(stream["source_cutoff_ns"]))
    arrays = new_arrays(1)
    best_count = np.zeros((5, 6, 1, 6600), dtype=np.int64)
    add_stream_statistics(arrays, best_count, 0, stream, codes)
    np.testing.assert_array_equal(statistics.best_count, best_count[2, :, 0].T)
    np.testing.assert_array_equal(statistics.covered_sum, arrays["covered_sum"][:, 0].T)
    np.testing.assert_array_equal(statistics.tuwr_sum, arrays["tuwr_sum"][:, 0].T)
    print(f"PASS: {statistics.event_rows} generated source events match CLARA's exact CART labels, coverage and TUWR sums")


if __name__ == "__main__":
    test_compact_metrics_and_time_boundary()
    test_capped_cart_against_eventwise_fit()
    test_shared_training_matrix()
    test_real_source_matches_clara_label_counts()
