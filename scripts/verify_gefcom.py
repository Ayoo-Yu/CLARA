"""Recompute all released GEFCom state choices and the nine-method summaries.

This runs the evaluated selection function on fitted candidate evidence. It does
not retrain the forecasters or infer new historical risk estimates.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from _bootstrap import REPOSITORY
from policies import select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPOSITORY / "results/gefcom")
    parser.add_argument("--policy-root", type=Path, help="Extracted full policy evidence; otherwise discover assets or use the bundled single-fold fixture")
    parser.add_argument("--output", type=Path, default=REPOSITORY / "results/recomputed/gefcom")
    args = parser.parse_args()
    locations = [REPOSITORY / "assets/gefcom_policy_evidence_v1", REPOSITORY.parent / "assets/gefcom_policy_evidence_v1", args.input]
    policy_root = args.policy_root or next(p for p in locations if (p / "policy_index.json").is_file())
    manifest = json.loads((policy_root / "policy_index.json").read_text(encoding="utf-8"))
    checked = 0
    for record in manifest["records"]:
        directory = policy_root / "policies" / record["unit"]
        states = pd.read_parquet(directory / "states.parquet")
        with np.load(directory / (record["price_id"] + ".npz")) as archive:
            evidence = {key: archive[key] for key in ("risk_score", "coverage", "tuwr", "ard")}
            selected, _ = select(states, evidence, record["thresholds"], "DIRECTIONAL_REPAIR", 1.0)
            np.testing.assert_array_equal(selected, archive["selected"])
        checked += len(states)
    frame = pd.read_parquet(args.input / "condition_metrics.parquet")
    keys = ["zone_or_farm", "price_id", "predictor", "horizon_steps", "target_coverage"]
    assert frame.groupby(keys, observed=True).size().eq(9).all()
    assert frame.method.nunique() == 9
    np.testing.assert_allclose(frame.ERRF, frame.capacity + frame.exceedance, atol=2e-12, rtol=0)
    # The archived CART/LinUCB summaries do not contain gap decomposition.
    have_gaps = frame[["ARD", "under_gap", "over_gap"]].notna().all(axis=1)
    np.testing.assert_allclose(frame.loc[have_gaps, "ARD"], frame.loc[have_gaps, "under_gap"] + frame.loc[have_gaps, "over_gap"], atol=2e-12, rtol=0)
    # The exact aggregation and average-tie ranking used by the experiment.
    frame["mean_rank"] = frame.groupby(keys, observed=True).ERRF.rank(method="average")
    frame["relative_excess_pct"] = 100 * (frame.ERRF / frame.groupby(keys, observed=True).ERRF.transform("min") - 1)
    metrics = ["ERRF", "capacity", "exceedance", "coverage", "width", "TUWR", "TOWR", "ARD", "under_gap", "over_gap", "mean_rank", "relative_excess_pct"]
    zones = frame.groupby(["zone_or_farm", "method"], observed=True)[metrics].mean()
    overall = zones.groupby("method", observed=True).mean().sort_values("ERRF")
    by_price = frame.groupby(["price_id", "zone_or_farm", "method"], observed=True)[metrics].mean().groupby(["price_id", "method"], observed=True).mean()
    np.testing.assert_allclose(overall.loc["CLARA", "ERRF"], 2.8452, atol=0.00005, rtol=0)
    np.testing.assert_allclose(overall.loc["CLARA", "mean_rank"], 2.66, atol=0.005, rtol=0)
    np.testing.assert_allclose(overall.loc["CLARA", "relative_excess_pct"], 1.65, atol=0.005, rtol=0)
    args.output.mkdir(parents=True, exist_ok=True)
    overall.to_csv(args.output / "overall.csv")
    by_price.to_csv(args.output / "by_price.csv")
    result = {"status": "PASS", "fitted_policies": len(manifest["records"]), "state_choices_reproduced": checked,
              "forecasting_conditions": len(frame) // 9, "methods": 9,
              "conditions_with_gap_decomposition": int(have_gaps.sum()),
              "CLARA": overall.loc["CLARA"].to_dict(),
              "scope": "Original selector on fitted evidence; numerical summaries from released condition-level results; no forecast-model retraining."}
    (args.output / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
