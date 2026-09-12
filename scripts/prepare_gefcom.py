"""Recreate the study's GEFCom2014 inputs from the official wind archive.

The input is GEFCom2014-W_V2.zip obtained separately from the competition
authors. No competition data are included with this script.

The retained rows, target interpolation and weather/calendar features match
the archived study inputs. Output rows are chronological, and the lagged
target is recomputed in that order. This removes an obsolete, lexically
ordered lag column in the archived CSVs. The evaluated forecasting code also
sorts timestamps and recomputes this column before training and prediction.

The historical row set excludes 2012-01-01 10:00 (the first lexical timestamp
in the historical preparation) and the final six hours without target data.
These omissions are retained to preserve the evaluated train/test splits.
Interior target gaps are linearly interpolated in chronological order,
matching the archived inputs. This is a reconstruction of those inputs, not
a general recommendation for preprocessing a new operational data set.
The interpolation uses later observed endpoints; numerical reproduction
must not be interpreted as verification that every feature was available
at its forecast issue time.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


ARCHIVE_MEMBERS = {
    "history": "Wind/Task 15/Task15_W_Zone1_10.zip",
    "weather": "Wind/Task 15/TaskExpVars15_W_Zone1_10.zip",
    "solution": "Wind/Solution to Task 15/solution15_W.csv",
    "instructions": "Wind/Instructions.txt",
}
FEATURES = [
    "feature_u10", "feature_v10", "feature_u100", "feature_v100",
    "feature_ws10", "feature_ws100", "feature_wd10", "feature_wd100",
    "feature_target_lag1", "feature_hour_sin", "feature_hour_cos",
    "feature_month_sin", "feature_month_cos",
]
HORIZONS = (1, 3, 6, 12, 24)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct_zone(
    history_zip: ZipFile,
    weather_zip: ZipFile,
    solution: pd.DataFrame,
    zone: int,
) -> tuple[pd.DataFrame, dict]:
    history = pd.read_csv(
        history_zip.open(f"Task15_W_Zone1_10/Task15_W_Zone{zone}.csv")
    )
    weather = pd.read_csv(
        weather_zip.open(f"TaskExpVars15_W_Zone1_10/TaskExpVars15_W_Zone{zone}.csv")
    )
    continuation = weather.merge(
        solution.loc[solution["ZONEID"].eq(zone)],
        on=["ZONEID", "TIMESTAMP"],
        validate="one_to_one",
        how="left",
    )
    raw = pd.concat([history, continuation], ignore_index=True)
    raw["parsed_timestamp"] = pd.to_datetime(
        raw["TIMESTAMP"], format="%Y%m%d %H:%M", errors="raise"
    )
    raw = raw.sort_values("parsed_timestamp").reset_index(drop=True)
    if raw["parsed_timestamp"].duplicated().any():
        raise ValueError(f"Zone {zone}: duplicate timestamps in official data")
    if not raw["ZONEID"].eq(zone).all():
        raise ValueError(f"Zone {zone}: unexpected zone identifier")
    if raw[["U10", "V10", "U100", "V100"]].isna().any().any():
        raise ValueError(f"Zone {zone}: unhandled missing weather values")

    originally_missing = raw["TARGETVAR"].isna()
    observed_anchor_position = pd.Series(
        np.where(originally_missing, np.nan, np.arange(len(raw))),
        index=raw.index,
    )
    next_observed_position = observed_anchor_position.bfill()
    raw["TARGETVAR"] = raw["TARGETVAR"].interpolate(limit_area="inside")
    interpolated = originally_missing & raw["TARGETVAR"].notna()
    trailing_unavailable = raw["TARGETVAR"].isna()
    lexical_first = raw.loc[raw["TIMESTAMP"].idxmin(), "parsed_timestamp"]
    if lexical_first != pd.Timestamp("2012-01-01 10:00:00"):
        raise ValueError(f"Zone {zone}: unexpected archive time range")
    keep = raw["TARGETVAR"].notna() & raw["parsed_timestamp"].ne(lexical_first)
    retained = raw.loc[keep].reset_index(drop=True)
    retained_raw_positions = np.flatnonzero(keep.to_numpy())
    ts = retained["parsed_timestamp"]
    out = pd.DataFrame({
        "timestamp": ts.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "target": retained["TARGETVAR"],
        "capacity": 1.0,
        "site_id": f"gefcom2014_zone{zone}",
        "zone_id": zone,
    })
    for name in ("U10", "V10", "U100", "V100"):
        out[f"feature_{name.lower()}"] = retained[name]
    for height in (10, 100):
        out[f"feature_ws{height}"] = np.sqrt(
            retained[f"U{height}"] ** 2 + retained[f"V{height}"] ** 2
        )
    for height in (10, 100):
        out[f"feature_wd{height}"] = (
            np.degrees(np.arctan2(retained[f"U{height}"], retained[f"V{height}"]))
            + 360
        ) % 360
    out["feature_target_lag1"] = retained["TARGETVAR"].shift(1)
    out["feature_hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    out["feature_hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    out["feature_month_sin"] = np.sin(2 * np.pi * (ts.dt.month - 1) / 12)
    out["feature_month_cos"] = np.cos(2 * np.pi * (ts.dt.month - 1) / 12)
    details = {
        "zone": zone,
        "official_rows": len(raw),
        "retained_rows": len(out),
        "columns": list(out.columns),
        "interior_targets_interpolated": int(interpolated.sum()),
        "unavailable_terminal_targets_excluded": int(trailing_unavailable.sum()),
        "historical_row_omission": lexical_first.isoformat(),
        "first_timestamp": ts.iloc[0].isoformat(),
        "last_timestamp": ts.iloc[-1].isoformat(),
        "output_order": "chronological",
        "lag_definition": "target in the preceding retained chronological row",
        "first_lag_is_missing": True,
    }
    interpolation_audit = []
    for source_row, raw_row in enumerate(retained_raw_positions):
        if not interpolated.iloc[raw_row]:
            continue
        right_raw_row = int(next_observed_position.iloc[raw_row])
        right_timestamp = raw["parsed_timestamp"].iloc[right_raw_row]
        lag_issue_row = source_row + 1
        lag_issue_exists = lag_issue_row < len(retained)
        lag_issue_time = (
            retained["parsed_timestamp"].iloc[lag_issue_row]
            if lag_issue_exists else pd.NaT
        )
        interpolation_audit.append({
            "source_row_index": source_row,
            "official_row_index": int(raw_row),
            "right_observed_anchor_official_row_index": right_raw_row,
            "lag_issue_source_row_index": lag_issue_row if lag_issue_exists else None,
            "right_anchor_not_strictly_before_lag_issue": (
                bool(right_timestamp >= lag_issue_time) if lag_issue_exists else None
            ),
            "right_anchor_hours_after_lag_issue": (
                float((right_timestamp - lag_issue_time) / pd.Timedelta(hours=1))
                if lag_issue_exists else None
            ),
        })
    details["interpolated_target_audit"] = {
        "row_index_base": 0,
        "events": interpolation_audit,
        "interpolated_targets_used_as_lag_with_unavailable_right_anchor": sum(
            event["right_anchor_not_strictly_before_lag_issue"] is True
            for event in interpolation_audit
        ),
        "interpretation": (
            "Chronological lag recomputation removes the archived string-order bug. "
            "It does not remove the look-ahead introduced by interpolating missing "
            "targets using a later observed endpoint. Counts below distinguish "
            "interpolated evaluation labels from lag inputs using that endpoint."
        ),
    }
    return out, details


def compare_frames(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    if list(left.columns) != list(right.columns) or left.shape != right.shape:
        raise AssertionError("Frame shape or column order differs")
    per_column = {}
    for name in left.columns:
        x, y = left[name], right[name]
        if pd.api.types.is_numeric_dtype(x.dtype):
            xa, ya = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            exact = np.array_equal(xa, ya, equal_nan=True)
            if not np.array_equal(np.isnan(xa), np.isnan(ya)):
                raise AssertionError(f"Missing-value locations differ: {name}")
            valid = ~np.isnan(xa)
            maximum = float(np.max(np.abs(xa[valid] - ya[valid]))) if valid.any() else 0.0
            if not np.allclose(xa, ya, rtol=0, atol=1e-12, equal_nan=True):
                raise AssertionError(f"Numeric values differ: {name}, maximum={maximum}")
            per_column[name] = {"exact": bool(exact), "max_abs_difference": maximum}
        else:
            exact = x.reset_index(drop=True).equals(y.reset_index(drop=True))
            if not exact:
                raise AssertionError(f"Non-numeric values differ: {name}")
            per_column[name] = {"exact": True}
    return {
        "rows": len(left), "columns": len(left.columns),
        "all_columns_exact": all(value["exact"] for value in per_column.values()),
        "max_abs_difference": max(
            (value.get("max_abs_difference", 0.0) for value in per_column.values()),
            default=0.0,
        ),
        "per_column": per_column,
    }


def verify_against_reference(output: Path, reference: Path, details: dict) -> dict:
    # Load the actual evaluated preprocessing; this does not train any model.
    import _bootstrap  # noqa: F401
    from common import prepare_multihorizon, select_feature_columns
    from run_base_predictor import _build_feature_context, _load_source

    new = _load_source(output)
    old = _load_source(reference)
    without_lag = [c for c in new.columns if c != "feature_target_lag1"]
    base_check = compare_frames(new[without_lag], old[without_lag])
    old_lag = old["feature_target_lag1"].to_numpy()
    corrected_lag = old["target"].shift(1).to_numpy()
    differing = ~np.isclose(old_lag, corrected_lag, rtol=0, atol=1e-12, equal_nan=True)
    original_order = pd.read_csv(reference)
    original_ts = pd.to_datetime(original_order["timestamp"])
    preceding_row_ts = original_ts.shift(1)
    future_legacy_sources = int(preceding_row_ts.gt(original_ts).sum())
    feature_columns = select_feature_columns(new)
    context_check = compare_frames(
        _build_feature_context(new, feature_columns),
        _build_feature_context(old, feature_columns),
    )
    prepared_checks = {}
    interpolated_rows = {
        event["source_row_index"]
        for event in details["interpolated_target_audit"]["events"]
    }
    unavailable_anchor_issue_rows = {
        event["lag_issue_source_row_index"]
        for event in details["interpolated_target_audit"]["events"]
        if event["right_anchor_not_strictly_before_lag_issue"] is True
    }
    audit_by_horizon = {}
    for horizon in HORIZONS:
        new_prepared = prepare_multihorizon(new, horizon, 0.6, 0.2)
        old_prepared = prepare_multihorizon(old, horizon, 0.6, 0.2)
        check = compare_frames(new_prepared, old_prepared)
        if new_prepared.attrs != old_prepared.attrs:
            raise AssertionError(f"Horizon {horizon}: split metadata differ")
        check["split_metadata_exact"] = True
        check["split_counts"] = {
            str(k): int(v) for k, v in new_prepared["split"].value_counts().items()
        }
        prepared_checks[str(horizon)] = check
        split_audit = {}
        for split_name in ("train", "calibration", "test"):
            group = new_prepared.loc[new_prepared["split"].eq(split_name)]
            split_audit[split_name] = {
                "rows": len(group),
                "interpolated_target_labels": int(group["label_row_index"].isin(interpolated_rows).sum()),
                "lag_rows_using_unavailable_interpolation_anchor": int(group["issue_row_index"].isin(unavailable_anchor_issue_rows).sum()),
            }
        audit_by_horizon[str(horizon)] = split_audit
    return {
        "reference_sha256": sha256(reference),
        "all_original_columns_except_legacy_lag": base_check,
        "legacy_lag_values_changed": int(differing.sum()),
        "legacy_lag_previous_rows_with_later_timestamps": future_legacy_sources,
        "prediction_feature_context": context_check,
        "prepared_matrices_by_horizon_hours": prepared_checks,
        "interpolation_effect_by_horizon_and_split": audit_by_horizon,
        "numeric_comparison_atol": 1e-12,
        "numeric_comparison_rtol": 0,
        "verification_scope": (
            "All retained source rows and columns; original training-label, feature, "
            "time-alignment and split matrices at five horizons; complete original "
            "prediction feature context. No model retraining in this check."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Official GEFCom2014-W_V2.zip")
    parser.add_argument("--output", type=Path, required=True, help="Directory for ten processed CSV files")
    parser.add_argument("--zone", type=int, choices=range(1, 11), action="append", help="Optional zones; default all ten")
    parser.add_argument("--reference-dir", type=Path, help="Optional archived processed CSVs for full matrix comparison")
    parser.add_argument("--report", type=Path, help="Optional JSON verification report path")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    zones = sorted(set(args.zone or range(1, 11)))
    report = {
        "status": "PASS",
        "status_scope": "Data reconstruction and optional numeric-equivalence checks only",
        "archive_filename": args.archive.name,
        "archive_sha256": sha256(args.archive),
        "source_members": ARCHIVE_MEMBERS,
        "target_scale": "Already divided by each wind farm's nominal capacity in the official release; no additional normalization",
        "wind_direction_convention": "degrees(arctan2(U, V)) + 360, modulo 360; retained for compatibility",
        "licensing_observation": "Wind/Instructions.txt describes the variables but states no redistribution license",
        "reference_comparison_requested": args.reference_dir is not None,
        "zones": [],
    }
    with ZipFile(args.archive) as outer:
        with ZipFile(io.BytesIO(outer.read(ARCHIVE_MEMBERS["history"]))) as history:
            with ZipFile(io.BytesIO(outer.read(ARCHIVE_MEMBERS["weather"]))) as weather:
                solution = pd.read_csv(outer.open(ARCHIVE_MEMBERS["solution"]))
                for zone in zones:
                    frame, details = reconstruct_zone(history, weather, solution, zone)
                    output = args.output / f"gefcom2014_zone{zone}_processed.csv"
                    frame.to_csv(output, index=False)
                    details["output_filename"] = output.name
                    details["output_sha256"] = sha256(output)
                    if args.reference_dir is not None:
                        reference = args.reference_dir / output.name
                        details["reference_verification"] = verify_against_reference(output, reference, details)
                    report["zones"].append(details)
                    print(f"Zone {zone}: {len(frame):,} rows prepared; chronological lag recomputed")
    report_path = args.report or args.output / "conversion_report.json"
    report["interpolation_causality"] = {
        "assessment": "Retrospective interpolation retained from the evaluated inputs",
        "interpolated_targets": sum(z["interior_targets_interpolated"] for z in report["zones"]),
        "lag_rows_with_right_anchor_not_before_issue": sum(
            z["interpolated_target_audit"]["interpolated_targets_used_as_lag_with_unavailable_right_anchor"]
            for z in report["zones"]
        ),
        "causal_availability_verified": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: {len(zones)} zones; report {report_path.name}")
    print("Scope: numeric reproduction. Retrospective target interpolation is retained and audited.")


if __name__ == "__main__":
    main()
