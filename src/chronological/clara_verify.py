"""Mathematical regression and time-gate checks for the corrected fitter."""
from pathlib import Path
import ast
import json
import time
import numpy as np
import pandas as pd
import clara_forward as c


def synthetic_stream(zone_index=0):
    rng = np.random.default_rng(821 + zone_index)
    n = 700
    step = 3_600_000_000_000
    issue = np.arange(n, dtype=np.int64) * step + np.datetime64("2013-06-01", "ns").astype(np.int64)
    y = rng.beta(2, 3, n)
    center = np.clip(y + rng.normal(0, .18, n), -.1, 1.1)
    half = c.COVERAGES[:, None] * (.12 + rng.uniform(0, .25, (11, n)))
    lower = np.clip(center[None, None, :] - half[None] * np.arange(.85, 1.15, .05)[:, None, None], 0, 1)
    upper = np.clip(center[None, None, :] + half[None] * np.arange(.85, 1.15, .05)[:, None, None], 0, 1)
    rolling = rng.integers(1, 5, (11, n))
    rolling[:, :168] = 0
    tuwr, ard = rng.uniform(0, .6, (6, 11, n)), rng.uniform(0, .2, (6, 11, n))
    tuwr[:, :, :168], ard[:, :, :168] = np.nan, np.nan
    return dict(issue_ns=issue, label_ns=issue+step, available_ns=issue+2*step,
                source_cutoff_ns=issue[-1]+3*step, y=y, center=center, raw_width=2*half,
                ramp=(rng.random(n)>.7).astype(int), rolling_state=rolling,
                lower=lower, upper=upper, candidate_tuwr=tuwr, candidate_ard=ard)


def legacy_frame(stream, zone, states, code, predictor="GBR", horizon=1, seed=0):
    n = len(stream["y"])
    frame = states.iloc[code.reshape(-1)][list(c.STATE_FIELDS)].reset_index(drop=True).copy()
    frame["event_id"] = [f"{zone}-{i}" for i in range(len(frame))]
    frame["zone_or_farm"] = zone
    frame["predictor"] = predictor
    frame["horizon_steps"] = horizon
    frame["seed"] = seed
    frame["issue_timestamp"] = pd.to_datetime(np.tile(stream["issue_ns"], 11))
    frame["target_after_maturity"] = np.tile(stream["y"], 11)
    frame["schedule_proxy"] = np.tile(stream["center"], 11)
    for a, name in enumerate(c.ACTIONS):
        lo, up = stream["lower"][a].reshape(-1), stream["upper"][a].reshape(-1)
        frame[name+"__candidate_lower"] = lo
        frame[name+"__candidate_upper"] = up
        frame[name+"__covered"] = (lo <= frame.target_after_maturity) & (frame.target_after_maturity <= up)
        frame[name+"__tuwr_indicator"] = stream["candidate_tuwr"][a].reshape(-1)
        frame[name+"__ard_value"] = stream["candidate_ard"][a].reshape(-1)
    return frame


def verify_accumulator():
    states = c.make_states()
    legacy = c.original.SixActionStatsBuilder(states=states, prices=c.PRICES,
        heldout_zone="zone10", source_zones=("zone1", "zone2", "zone3"), seed=0,
        horizons=(1,), predictors=("GBR",), coverages=tuple(c.COVERAGES), require_causal_source_audit=False)
    arrays, best = c.new_arrays(3), np.zeros((5, 6, 3, c.STATE_COUNT), dtype=np.int64)
    edges = np.broadcast_to(np.array([.1, .3]), (4, 5, 11, 2)).copy()
    for zi, zone in enumerate(("zone1", "zone2", "zone3")):
        stream = synthetic_stream(zi)
        code = c.state_codes(stream, "GBR", 1, edges)
        c.add_stream_statistics(arrays, best, zi, stream, code)
        legacy.add_stream(legacy_frame(stream, zone, states, code), source_zone=zone,
                          predictor="GBR", horizon=1, seed=0)
    old = legacy.finalize()
    errors = {}
    for key, value in arrays.items():
        np.testing.assert_array_equal(value, old.arrays[key])
        errors[key] = 0.
    np.testing.assert_array_equal(best, old.cart_best_count)
    return old, edges, {"all_sufficient_statistics_bitwise_equal": True,
                         "cart_five_price_labels_bitwise_equal": True, "tested_events": int(arrays["event_count"].sum())}


def verify_guardrail():
    path = c.ROOT / "analysis/guardrail_redesign_20260908/policies.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keep = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("choose", "select")]
    namespace = {"np": np}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), namespace)
    rng = np.random.default_rng(119)
    states = c.make_states()
    con, config = c.contracts_and_config()
    e = {"risk_score": rng.uniform(0, 2, (c.STATE_COUNT, 6)),
         "coverage": rng.uniform(0, 1, (c.STATE_COUNT, 6)),
         "tuwr": rng.uniform(0, .6, (c.STATE_COUNT, 6)),
         "ard": rng.uniform(0, .3, (c.STATE_COUNT, 6))}
    cold = states.rolling_state.eq("cold_start").to_numpy()
    e["tuwr"][cold], e["ard"][cold] = np.nan, np.nan
    # Equal-score rows exercise coverage/ARD/fixed-order ties, not only argmin.
    e["risk_score"][::3] = 1.
    e["coverage"][::11] = .95
    for price in c.PRICES:
        thresholds = c.original.adaptive_v4.adaptive_guardrail_thresholds(price.miss_to_capacity_ratio, config)
        selected, audit = c.directional_repair(states, e, thresholds)
        expected, previous = namespace["select"](states, e, thresholds, "DIRECTIONAL_REPAIR", 1.)
        np.testing.assert_array_equal(selected, expected)
        np.testing.assert_array_equal(audit["guardrail_empty_fallback"], previous["initial_empty"])
        np.testing.assert_array_equal(audit["selected_max_excess"], previous["selected_max_excess"])
    return {"current_directional_repair_all_five_prices_exact": True, "states_per_price": c.STATE_COUNT}


def verify_time_gate():
    stream = synthetic_stream()
    c._verify_source_times(stream)
    changed = dict(stream)
    changed["available_ns"] = stream["available_ns"].copy()
    changed["available_ns"][-1] = stream["source_cutoff_ns"]
    c._verify_source_times(changed)
    changed["available_ns"][-1] = stream["source_cutoff_ns"] + 1
    try:
        c._verify_source_times(changed)
    except ValueError:
        return {"feedback_available_at_cutoff_accepted_with_earlier_label": True,
                "feedback_after_cutoff_rejected": True}
    raise AssertionError("A future source outcome was accepted")


def verify_actual_available():
    import forward_common as f
    checks = []
    for zone, seed, predictor, horizon in (("zone1",0,"GBR",1), ("zone3",1,"MLP",24), ("zone5",2,"QRLSTM",6)):
        if not f.dataset_path(zone, seed, predictor, horizon, "source").is_file():
            continue
        d = f.load_stream(zone, seed, predictor, horizon, "source", 18)
        c._verify_source_times(d)
        states = c.make_states()
        edges = np.zeros((4,5,11,2))
        edges[c.PREDICTORS.index(predictor),c.HORIZONS.index(horizon)] = np.quantile(d["raw_width"], [.33,.67], axis=1).T
        codes = c.state_codes(d, predictor, horizon, edges)
        arrays, best = c.new_arrays(1), np.zeros((5,6,1,c.STATE_COUNT), dtype=np.int64)
        c.add_stream_statistics(arrays, best, 0, d, codes)
        legacy = c.original.SixActionStatsBuilder(states=states, prices=c.PRICES,
            heldout_zone="zone10", source_zones=(zone,), seed=seed, horizons=(horizon,),
            predictors=(predictor,), coverages=tuple(c.COVERAGES), require_causal_source_audit=False)
        legacy.add_stream(legacy_frame(d,zone,states,codes,predictor,horizon,seed), source_zone=zone,
                          predictor=predictor,horizon=horizon,seed=seed)
        old = legacy.finalize()
        for key, value in arrays.items():
            np.testing.assert_array_equal(value, old.arrays[key])
        np.testing.assert_array_equal(best, old.cart_best_count)
        checks.append(dict(zone=zone,seed=seed,predictor=predictor,horizon=horizon,
                           event_count=int(arrays["event_count"].sum()), all_statistics_bitwise_equal=True))
    return checks


if __name__ == "__main__":
    began = time.perf_counter()
    stats, edges, stats_audit = verify_accumulator()
    guard_audit, time_audit = verify_guardrail(), verify_time_gate()
    actual_audit = verify_actual_available()
    bundle = c.SourceBundle(stats, edges, stats.audit)
    model = c.fit_policies(bundle)
    assert model.selected_actions.shape == (5, 6600)
    audit = dict(status="PASS", synthetic_structural_test=True, accumulator=stats_audit,
                 guardrail=guard_audit, timing=time_audit, fit_finite=True,
                 actual_source_stream_checks=actual_audit, seconds=time.perf_counter()-began)
    (c.HERE/"clara_verification.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit), flush=True)
