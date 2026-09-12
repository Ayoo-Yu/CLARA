"""Run the evaluated forecasting and six-candidate calibration code for one stream.

Input is the processed hourly GEFCom CSV defined in the data contract. No input
data are downloaded. The wrapper adapts provenance for source archives without
Git metadata; it does not modify training or calibration functions.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from _bootstrap import REPOSITORY
import run_base_predictor as base
import run_calibration as calibration
from clara_event_contract import load_frozen_contracts
from baseline_conformal import run_conformal_configuration
from run_gefcom_six_action_full_v1 import TSC_FROZEN_FAMILY, TSC_FROZEN_CONFIGURATION


def snapshot_identity(_):
    data = (REPOSITORY / "configs/source_provenance.json").read_bytes()
    return "source-snapshot-sha256:" + hashlib.sha256(data).hexdigest()


def snapshot_status(_):
    return "source-archive; Git worktree status not asserted"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--zone", required=True)
    parser.add_argument("--predictor", choices=["Ridge", "GBR", "MLP", "QRLSTM"], required=True)
    parser.add_argument("--horizon", type=int, choices=[1, 3, 6, 12, 24], required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--coverage", type=float, nargs="+", default=[.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contracts = load_frozen_contracts()
    if not set(args.coverage).issubset(contracts.coverages):
        raise ValueError("Coverage must be from the evaluated eleven-level grid")
    # The unchanged writer records a source-snapshot identity instead of
    # pretending that an unpacked archive has an original Git commit.
    base.git_commit_for_path = snapshot_identity
    base.git_status_porcelain = snapshot_status
    cfg = dict(protocol_id=contracts.protocol_id, protocol_version=contracts.protocol_version,
               protocol_sha256=contracts.protocol_sha256, predictor=args.predictor, zone=args.zone,
               seed=args.seed, horizon=args.horizon, data_path=str(args.data.resolve()),
               dataset_id="gefcom2014", split={"train_ratio":.6,"calibration_ratio":.2},
               results_dir=str(args.output / "base"), require_clean_code=False)
    result = base.run_base_prediction(cfg)
    candidate_cfg = cfg | dict(baseline_registry_sha256=contracts.baseline_sha256,
                              output_contract_sha256=contracts.output_sha256,
                              base_predictions_path=str(args.output / "base/base_predictions.parquet"))
    prepared = calibration.prepare_candidate_base(candidate_cfg, code_status=snapshot_status(None), code_commit=snapshot_identity(None))
    outputs = []
    for c in args.coverage:
        parts = []
        event_identity = None
        for method in ["SplitCF", "ACI", "AgACI", "EnbPI"]:
            part = calibration.build_candidate_frames(candidate_cfg | dict(method=method, alpha=round(1-c,10)), prepared, event_identity)
            event_identity = part["event_identity"]
            parts.append(part["candidate"])
        four = pd.concat(parts, ignore_index=True)
        # The evaluated added-candidate runner receives only test issues with
        # a known feedback-availability time; retain the common event set.
        last_test_issue = prepared["predictions"].loc[prepared["predictions"].split.eq("test"), "issue_timestamp"].max()
        common_base = prepared["predictions"].loc[prepared["predictions"].label_available_timestamp.le(last_test_issue)].copy()
        common_issues = set(common_base.loc[common_base.split.eq("test"), "issue_timestamp"])
        four = four.loc[four.issue_timestamp.isin(common_issues)].copy()
        metadata = parts[0][["event_id", "issue_timestamp", "label_timestamp", "label_available_timestamp", "target_coverage", "target", "base_center"]].copy()
        columns = ["event_id", "action", "candidate_lower", "candidate_upper"]
        tuned = run_conformal_configuration(contracts=contracts, base_predictions=common_base,
                   dataset_id="gefcom2014", zone_or_farm=args.zone, predictor=args.predictor, seed=args.seed,
                   target_coverage=c, family=TSC_FROZEN_FAMILY, configuration=TSC_FROZEN_CONFIGURATION,
                   drain_final_feedback=True).candidates
        tuned["action"] = "TunedSingleConformal"
        # Same arithmetic endpoint mean and final [0,1] clipping as the
        # evaluated baseline_ensemble.select_equal_endpoint_ensemble function.
        ensemble = four.groupby("event_id", sort=False)[["candidate_lower", "candidate_upper"]].mean().clip(0,1).reset_index()
        ensemble["action"] = "EqualEndpointEnsemble"
        combined = pd.concat([four[columns], tuned[columns], ensemble[columns]], ignore_index=True)
        assert combined.groupby("event_id").size().eq(6).all()
        combined = combined.merge(metadata, on="event_id", how="left", validate="many_to_one")
        assert combined.issue_timestamp.notna().all()
        outputs.append(combined)
        print(f"Completed coverage {c}", flush=True)
    final = pd.concat(outputs, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    final.to_parquet(args.output / "six_candidates.parquet", index=False, compression="zstd")
    receipt = {"status":"PASS", "base_forecast_rows":result["n_output"], "candidate_rows":len(final),
               "actions":final.action.drop_duplicates().tolist(),"coverage_levels":args.coverage,
               "TSC_family":TSC_FROZEN_FAMILY,"TSC_configuration":TSC_FROZEN_CONFIGURATION,
               "scope":"One processed GEFCom stream; original forecaster and base/TSC calibration; no selector fitting."}
    (args.output / "run_summary.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
