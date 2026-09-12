"""Replay the evaluated six-candidate joint-update algorithm on an anonymous archive.

All loss, historical reliability, support, shrinkage and selection calculations
call the evaluated functions. This wrapper only reconstructs their input schema.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
from _bootstrap import REPOSITORY
import trial
from engine import add_rows, evidence, run_batches
from clara_errf import ErrfTheta
from s09_minimum_external_core import _metric_arrays_from_endpoints

PUBLIC_ACTIONS = ["Static", "ACI", "AgACI", "EnbPI-RH", "TSC", "EEE"]
INTERNAL_ACTIONS = list(trial.A)
TICK_NS = 900_000_000_000


def load_stream(path, split, price):
    frame = pd.read_parquet(path / f"base_{split}.parquet")
    tsc = pd.read_parquet(path / f"{price['tsc_configuration']}_{split}.parquet")
    frame = frame.merge(tsc, on="row_id", how="left", validate="one_to_one")
    # Public row IDs retain the original within-stream order; sorting used in
    # reliability() is chronological within each target coverage.
    frame = frame.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    for column, tick in [("issue_timestamp", "issue_tick"), ("label_timestamp", "label_tick"), ("label_available_timestamp", "available_tick")]:
        frame[column] = pd.to_datetime(frame[tick].to_numpy(np.int64) * TICK_NS, unit="ns")
    frame["event_id"] = frame.row_id.astype(str)
    frame["nominal_cadence_minutes"] = 15.0
    theta = ErrfTheta(pi_plus=price["capacity_weight"], pi_minus=price["capacity_weight"], kappa_plus=price["exceedance_weight"], kappa_minus=price["exceedance_weight"])
    metric_values = []
    for public, internal in zip(PUBLIC_ACTIONS, INTERNAL_ACTIONS):
        lower_col = public + "__lower" if public != "TSC" else "lower"
        upper_col = public + "__upper" if public != "TSC" else "upper"
        values = _metric_arrays_from_endpoints(frame, frame[lower_col].to_numpy(float), frame[upper_col].to_numpy(float), theta=theta)
        metric_values.append(values)
        frame[internal + "__covered"] = values["covered"]
        # These optional archived columns are absent from the compact public
        # input; the evaluated routine recomputes them from candidate coverage.
        frame[internal + "__tuwr_indicator"] = np.nan
        frame[internal + "__ard_value"] = np.nan
    historical, _ = trial.reliability(frame)
    losses = np.column_stack([v["errf"] for v in metric_values])
    covered = np.column_stack([v["covered"] for v in metric_values])
    facts = np.stack([losses, covered, historical[:, :, 0], historical[:, :, 1]], axis=2)
    meta = frame[["replay_order", "state_id", "issue_tick", "label_tick", "available_tick", "utc_day_group", "target_coverage"]].copy()
    return meta, facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="One FarmA/FarmB seed directory from the candidate archive")
    parser.add_argument("--price", default="rho_reference", choices=["rho_1", "rho_2", "rho_reference", "rho_10", "rho_20"])
    parser.add_argument("--max-test-days", type=float, help="Optional prefix validation; all adaptation records are retained")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    manifest = json.loads((args.archive / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["actions"] == PUBLIC_ACTIONS
    config = json.loads((args.archive / "price_configs.json").read_text(encoding="utf-8"))
    price = next(p for p in config["prices"] if p["price_id"] == args.price)
    states = pd.read_parquet(args.archive / "state_map.parquet").sort_values("state_id").reset_index(drop=True)
    np.testing.assert_array_equal(states.state_id, np.arange(len(states)))
    with np.load(args.archive / "policies" / f"{args.price}.npz") as p:
        policy = {key: p[key].copy() for key in p.files}
    parts = {"adaptation": [], "test": []}
    values = {"adaptation": [], "test": []}
    for predictor in manifest["predictors"]:
        for horizon in manifest["horizon_steps"]:
            path = args.archive / "streams" / f"{predictor}_h{horizon:02d}"
            for split in parts:
                meta, facts = load_stream(path, split, price)
                parts[split].append(meta)
                values[split].append(facts)
            print(f"Loaded {predictor}, horizon {horizon}", flush=True)
    loaded = {}
    for split in parts:
        meta = pd.concat(parts[split], ignore_index=True)
        order = np.argsort(meta.replay_order.to_numpy(), kind="stable")
        meta = meta.iloc[order].reset_index(drop=True)
        np.testing.assert_array_equal(meta.replay_order, np.arange(len(meta)))
        loaded[split] = meta, np.concatenate(values[split])[order]
        assert len(meta) == manifest["counts"][split]
    del parts, values
    am, af = loaded["adaptation"]
    fm, ff = loaded["test"]
    issue = fm.issue_tick.to_numpy(np.int64)
    label = fm.label_tick.to_numpy(np.int64)
    available = fm.available_tick.to_numpy(np.int64)
    assert am.available_tick.max() <= issue.min() and am.label_tick.max() < issue.min()
    assert np.all(np.diff(issue) >= 0)
    aday = am.utc_day_group.to_numpy(np.int64)
    fday = fm.utc_day_group.to_numpy(np.int64)
    mapping, stats = trial.layout(states, int(max(aday.max(), fday.max()) + 1))
    nmin = policy["n_min"].astype(np.int64)
    nu = policy["nu"].astype(float)
    frozen = policy["frozen_evidence"]
    cold = states.rolling_state.eq("cold_start").to_numpy()
    coverage = states.target_coverage.to_numpy(float)
    add_rows(am.state_id.to_numpy(np.int64), aday, af, 0, len(am), mapping, *stats)
    initial, levels = evidence(np.arange(len(states)), mapping, nmin, nu, cold, stats[0], stats[1], stats[2], stats[5], stats[3], stats[4])
    np.testing.assert_array_equal(levels, policy["initial_support_level"])
    np.testing.assert_allclose(initial, frozen, atol=3e-10, rtol=0, equal_nan=True)
    initial_error = float(np.nanmax(np.abs(initial - frozen)))
    print(f"Initial adaptation evidence verified: max error {initial_error:.3g}", flush=True)
    feedback_order = np.lexsort((label, available)).astype(np.int64)
    stop = len(fm) if args.max_test_days is None else int(np.searchsorted(issue, issue.min() + args.max_test_days * 96, side="left"))
    selected, predictions, trace, feedback_count = run_batches(issue, label, available, fm.state_id.to_numpy(np.int64), fday, ff, feedback_order, mapping, nmin, nu, cold, coverage, frozen, float(policy["coverage_shortfall"]), float(policy["tuwr_limit"]), stats, stop)
    with np.load(args.archive / "choices" / f"{args.price}.npz") as saved:
        np.testing.assert_array_equal(selected, saved["selected"][:stop, :4])
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "replayed_choices.npz", selected=selected, trace=trace)
    result = {"status": "PASS", "farm": manifest["farm_id"], "seed": manifest["seed"], "price": args.price,
              "adaptation_records": len(am), "test_records_replayed": stop, "test_records_total": len(fm),
              "all_test_records_replayed": stop == len(fm), "update_controls": manifest["choice_methods"][:4],
              "initial_evidence_max_error": initial_error, "matured_feedback_records": int(feedback_count),
              "exact_selected_actions": True, "seconds": time.monotonic() - started,
              "scope": "Original loss/reliability/support/update/selection functions on anonymized candidate intervals; base predictors are not retrained."}
    (args.output / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
