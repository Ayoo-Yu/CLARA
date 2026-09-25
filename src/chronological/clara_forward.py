"""CLARA fitting on genuinely pre-test source feedback.

This module reuses the existing seven-level risk-estimation mathematics, but
constructs its own data provenance from a cutoff-limited test-period prefix.
No old source facts, source-fitted width thresholds, or fitted policies are read.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from forward_common import ROOT
EXP = ROOT / "workspace/0427/0620/提交版本0703 Energy/一审/一审修订实验"
SCRIPTS = EXP / "测试CLARA/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import gefcom_six_action_selector_core_v1 as original

PREDICTORS = ("Ridge", "GBR", "MLP", "QRLSTM")
HORIZONS = (1, 3, 6, 12, 24)
HORIZON_GROUPS = ("H01", "H02_H03", "H04_H06", "H07_H12", "H13_H24")
COVERAGES = np.array((.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99))
ROLLING_STATES = ("cold_start", "undercoverage_pressure", "overcoverage_pressure", "volatile", "stable")
RAMP_STATES = ("ordinary", "ramp")
WIDTH_STATES = ("narrow", "medium", "wide")
ACTIONS = tuple(original.SIX_ACTIONS)
PRICE_IDS = tuple(row[0] for row in original.FORMAL_PRICE_ROWS)
PRICES = tuple(original.formal_price_spec(name) for name in PRICE_IDS)
STATE_FIELDS = tuple(original.STATE_FIELDS)
STATE_COUNT = 4 * 5 * 11 * 2 * 5 * 3
SCHEMA = "GEFCOM_CHRONO_PREFIX28_FROZEN_CLARA_V1"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def make_states() -> pd.DataFrame:
    rows = []
    for p, hg, c, r, g, w in itertools.product(PREDICTORS, HORIZON_GROUPS, COVERAGES,
                                               RAMP_STATES, ROLLING_STATES, WIDTH_STATES):
        rows.append((len(rows), p, hg, float(c), r, g, w))
    result = pd.DataFrame(rows, columns=("full_state_code", *STATE_FIELDS))
    assert len(result) == STATE_COUNT
    return result


def contracts_and_config():
    contracts = original.six_action_contracts(original.load_frozen_contracts())
    width_names = tuple(contracts.protocol["state_contract"]["raw_width"]["states"])
    if width_names != WIDTH_STATES:
        raise ValueError(f"Unexpected width vocabulary: {width_names}")
    old = json.loads(original.FOUR_VERSION_CONFIG_PATH.read_text(encoding="utf-8"))
    # These are algorithm parameters, not the old experiment's provenance.
    config = {key: old[key] for key in ("fixed_support", "adaptive_support", "adaptive_guardrail")}
    return contracts, config


def fit_width_edges(source_records: Iterable[tuple[str, str, int, Mapping[str, Any]]]) -> np.ndarray:
    """Fit per-predictor/horizon/coverage width quantiles from supplied sources.

    Call separately for each seed and outer/inner training set. Records are
    (zone, predictor, horizon, stream); only raw_width and timestamps are read.
    """
    widths: dict[tuple[int, int], list[np.ndarray]] = {}
    seen = set()
    for zone, predictor, horizon, stream in source_records:
        key = (str(zone), str(predictor), int(horizon))
        if key in seen:
            raise ValueError(f"Repeated width-fit stream: {key}")
        seen.add(key)
        _verify_source_times(stream)
        values = np.asarray(stream["raw_width"], dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != len(COVERAGES) or not np.isfinite(values).all():
            raise ValueError(f"Invalid raw widths: {key}/{values.shape}")
        widths.setdefault((PREDICTORS.index(predictor), HORIZONS.index(horizon)), []).append(values)
    output = np.full((4, 5, 11, 2), np.nan)
    for (p, h), parts in widths.items():
        joined = np.concatenate(parts, axis=1)
        output[p, h] = np.quantile(joined, (.33, .67), axis=1, method="linear").T
    if not np.isfinite(output).all():
        missing = np.argwhere(~np.isfinite(output).all(axis=(2, 3)))
        raise ValueError(f"Width-fit axis incomplete: {missing.tolist()}")
    return output


def _load(loader: Callable, zone: str, seed: int, p: str, h: int,
          split: str, tsc_index: int | None) -> Mapping[str, Any]:
    return loader(zone, seed, p, h, split, 0 if tsc_index is None else tsc_index)


def fit_width_edges_from_loader(source_zones: Sequence[str], seed: int,
                                load_stream: Callable, tsc_index: int | None = None) -> np.ndarray:
    records = ((z, p, h, _load(load_stream, z, seed, p, h, "source", tsc_index))
               for z in source_zones for p in PREDICTORS for h in HORIZONS)
    return fit_width_edges(records)


def state_codes(stream: Mapping[str, Any], predictor: str, horizon: int,
                width_edges: np.ndarray) -> np.ndarray:
    """Map coverage-by-time inputs to the shared 6,600 named states."""
    p, h = PREDICTORS.index(predictor), HORIZONS.index(horizon)
    widths = np.asarray(stream["raw_width"], dtype=np.float64)
    g = np.asarray(stream["rolling_state"], dtype=np.int64)
    r = np.asarray(stream["ramp"], dtype=np.int64)
    if widths.ndim != 2 or widths.shape[0] != 11 or g.shape != widths.shape or r.shape != (widths.shape[1],):
        raise ValueError("State input dimensions differ")
    if np.any((g < 0) | (g >= 5)) or np.any((r < 0) | (r >= 2)):
        raise ValueError("State category outside vocabulary")
    q = np.asarray(width_edges, dtype=np.float64)[p, h]
    w = np.where(widths <= q[:, 0, None], 0, np.where(widths <= q[:, 1, None], 1, 2))
    c = np.arange(11)[:, None]
    return (((((p * 5 + h) * 11 + c) * 2 + r[None, :]) * 5 + g) * 3 + w).astype(np.int64)


def _verify_source_times(stream: Mapping[str, Any], expected_cutoff_ns: int | None = None) -> dict:
    issue = np.asarray(stream["issue_ns"], dtype=np.int64)
    label = np.asarray(stream["label_ns"], dtype=np.int64)
    avail = np.asarray(stream["available_ns"], dtype=np.int64)
    cutoff = int(stream["source_cutoff_ns"] if expected_cutoff_ns is None else expected_cutoff_ns)
    if len(issue) == 0 or issue.shape != label.shape or issue.shape != avail.shape:
        raise ValueError("Empty or malformed source timestamps")
    if not (np.all(issue < label) and np.all(label < avail) and np.all(avail <= cutoff)
            and np.all(label < cutoff)):
        raise ValueError("Source labels are not strictly available before the held-out test cutoff")
    if not np.all(np.diff(issue) > 0):
        raise ValueError("Source issue times must increase strictly")
    if "predictor_train_available_max_ns" in stream:
        if int(stream["predictor_train_available_max_ns"]) >= int(issue.min()):
            raise ValueError("A source issue precedes completion of base-forecaster training")
    return {"issue_min_ns": int(issue.min()), "issue_max_ns": int(issue.max()),
            "label_max_ns": int(label.max()), "available_max_ns": int(avail.max()),
            "test_cutoff_ns": cutoff, "prediction_count": len(issue),
            "all_source_feedback_precedes_test": True}


def new_arrays(zone_count: int) -> dict[str, np.ndarray]:
    arrays = {}
    for key in ("event_count", "guardrail_count"):
        arrays[key] = np.zeros((zone_count, STATE_COUNT), dtype=np.int64)
    for prefix in ("", "guardrail_"):
        arrays[prefix + "issue_min_us"] = np.full((zone_count, STATE_COUNT), np.iinfo(np.int64).max, dtype=np.int64)
        arrays[prefix + "issue_max_us"] = np.full((zone_count, STATE_COUNT), np.iinfo(np.int64).min, dtype=np.int64)
    for key in ("capacity_sum", "miss_sum", "covered_sum", "guardrail_covered_sum", "tuwr_sum", "ard_sum"):
        arrays[key] = np.zeros((6, zone_count, STATE_COUNT), dtype=np.float64)
    return arrays


def stream_components(stream: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.asarray(stream["lower"], dtype=np.float64)
    upper = np.asarray(stream["upper"], dtype=np.float64)
    y = np.asarray(stream["y"], dtype=np.float64)[None, None, :]
    center = np.asarray(stream["center"], dtype=np.float64)[None, None, :]
    if lower.shape != upper.shape or lower.shape != (6, 11, y.shape[-1]):
        raise ValueError("Candidate endpoint dimensions differ")
    if not (np.isfinite(lower).all() and np.isfinite(upper).all()) or np.any(lower > upper):
        raise ValueError("Nonfinite or reversed candidate endpoints")
    # Keep the original unclipped point forecast in the capacity valuation.
    capacity = np.maximum(upper - center, 0) + np.maximum(center - lower, 0)
    miss = np.maximum(y - upper, 0) + np.maximum(lower - y, 0)
    covered = (lower <= y) & (y <= upper)
    return capacity, miss, covered


def add_stream_statistics(arrays: dict[str, np.ndarray], cart_best_count: np.ndarray,
                          zone_index: int, stream: Mapping[str, Any], codes: np.ndarray) -> None:
    capacity, miss, covered = stream_components(stream)
    code = codes.reshape(-1)
    issue = np.tile(np.asarray(stream["issue_ns"], dtype=np.int64) // 1000, 11)
    warm = np.asarray(stream["rolling_state"]).reshape(-1) != 0
    tuwr = np.asarray(stream["candidate_tuwr"], dtype=np.float64).reshape(6, -1)
    ard = np.asarray(stream["candidate_ard"], dtype=np.float64).reshape(6, -1)
    for values in (tuwr, ard):
        if not (np.array_equal(np.isnan(values), np.broadcast_to(~warm, values.shape))
                and np.isfinite(values[:, warm]).all()
                and np.all((values[:, warm] >= 0) & (values[:, warm] <= 1))):
            raise ValueError("Historical candidate TUWR/ARD must be finite exactly in non-cold states")
    arrays["event_count"][zone_index] += np.bincount(code, minlength=STATE_COUNT)
    np.minimum.at(arrays["issue_min_us"][zone_index], code, issue)
    np.maximum.at(arrays["issue_max_us"][zone_index], code, issue)
    wc, wi = code[warm], issue[warm]
    arrays["guardrail_count"][zone_index] += np.bincount(wc, minlength=STATE_COUNT)
    np.minimum.at(arrays["guardrail_issue_min_us"][zone_index], wc, wi)
    np.maximum.at(arrays["guardrail_issue_max_us"][zone_index], wc, wi)
    for action in range(6):
        for name, values in (("capacity_sum", capacity), ("miss_sum", miss), ("covered_sum", covered)):
            arrays[name][action, zone_index] += np.bincount(code, weights=values[action].reshape(-1), minlength=STATE_COUNT)
        for name, values in (("guardrail_covered_sum", covered.reshape(6, -1)),
                             ("tuwr_sum", tuwr), ("ard_sum", ard)):
            arrays[name][action, zone_index] += np.bincount(wc, weights=values[action, warm], minlength=STATE_COUNT)
    for pi, price in enumerate(PRICES):
        selected = np.argmin(price.capacity_weight * capacity + price.miss_weight * miss, axis=0).reshape(-1)
        for action in range(6):
            cart_best_count[pi, action, zone_index] += np.bincount(code[selected == action], minlength=STATE_COUNT)


@dataclass
class SourceBundle:
    stats: Any
    width_edges: np.ndarray
    audit: dict[str, Any]


def build_source_stats(heldout_zone: str, seed: int, tsc_index: int,
                       load_stream: Callable, *, source_zones: Sequence[str] | None = None,
                       cutoff_ns: int | None = None, width_edges: np.ndarray | None = None) -> SourceBundle:
    sources = tuple(source_zones or (f"zone{z}" for z in range(1, 11) if f"zone{z}" != heldout_zone))
    if heldout_zone in sources or len(set(sources)) != len(sources):
        raise ValueError("Held-out zone or repeated zone in source set")
    if width_edges is None:
        width_edges = fit_width_edges_from_loader(sources, seed, load_stream, tsc_index)
    arrays = new_arrays(len(sources))
    best = np.zeros((5, 6, len(sources), STATE_COUNT), dtype=np.int64)
    audit_rows = []
    for zi, zone in enumerate(sources):
        for predictor in PREDICTORS:
            for horizon in HORIZONS:
                stream = _load(load_stream, zone, seed, predictor, horizon, "source", tsc_index)
                temporal = _verify_source_times(stream, cutoff_ns)
                codes = state_codes(stream, predictor, horizon, width_edges)
                add_stream_statistics(arrays, best, zi, stream, codes)
                audit_rows.append(dict(source_zone=zone, predictor=predictor, horizon=horizon,
                                       seed=seed, tsc_configuration_index=int(tsc_index), **temporal))
    np.testing.assert_array_equal(best.sum(axis=1), np.broadcast_to(arrays["event_count"], (5, *arrays["event_count"].shape)))
    states = make_states()
    audit = dict(schema=SCHEMA, heldout_zone=heldout_zone, source_zones=list(sources), seed=seed,
                 tsc_configuration_index=int(tsc_index), source_split="Original test prefix; mature feedback available by 2013-09-05 00:00 UTC",
                 heldout_zone_used_in_fit=False, source_event_count=int(arrays["event_count"].sum()),
                 all_source_feedback_precedes_test=True, stream_count=len(audit_rows),
                 source_cutoff_ns=sorted({row["test_cutoff_ns"] for row in audit_rows}),
                 source_available_max_ns=max(row["available_max_ns"] for row in audit_rows),
                 width_quantiles=[.33, .67], width_fit_zones=list(sources),
                 width_edges_sha256=hashlib.sha256(width_edges.tobytes()).hexdigest(),
                 rolling_state_vocabulary=list(ROLLING_STATES), status="PASS")
    identities, checksum = original._statistics_array_content_identity(arrays, best)
    audit["sufficient_statistics_array_identities"] = identities
    audit["sufficient_statistics_content_sha256"] = checksum
    audit["state_universe_content_sha256"] = original._state_universe_content_sha256(states)
    audit["identity_sha256"] = _digest(audit)
    stats = original.SixActionSufficientStats(schema=SCHEMA, actions=ACTIONS, prices=PRICES,
        heldout_zone=heldout_zone, source_zones=sources, seed=int(seed), horizons=HORIZONS,
        predictors=PREDICTORS, coverages=tuple(COVERAGES), states=states, arrays=arrays,
        cart_best_count=best, stream_audit=pd.DataFrame(audit_rows), audit=audit)
    return SourceBundle(stats, width_edges, audit)


def choose_by_score(evidence: Mapping[str, np.ndarray], eligible: np.ndarray,
                    tolerance: float = 1e-12) -> np.ndarray:
    candidates = eligible.copy()
    if not candidates.any(axis=1).all():
        raise ValueError("Empty final choice set")
    score = evidence["risk_score"]
    best = np.min(np.where(candidates, score, np.inf), axis=1)
    candidates &= score <= best[:, None] + tolerance
    cov = np.where(np.isfinite(evidence["coverage"]), evidence["coverage"], -np.inf)
    best = np.max(np.where(candidates, cov, -np.inf), axis=1)
    candidates &= cov >= best[:, None] - tolerance
    for values in (evidence["tuwr"], evidence["ard"]):
        finite = np.isfinite(values)
        count, valid_count = candidates.sum(axis=1), (candidates & finite).sum(axis=1)
        if ((valid_count > 0) & (valid_count < count)).any():
            raise ValueError("Mixed missing reliability values in a tied choice set")
        applies = (valid_count == count) & (count > 0)
        best = np.min(np.where(candidates & finite, values, np.inf), axis=1)
        candidates &= (~applies[:, None]) | (finite & (values <= best[:, None] + tolerance))
    if not candidates.any(axis=1).all():
        raise ValueError("Tie breaking removed every candidate")
    return candidates.argmax(axis=1)


def directional_repair(states: pd.DataFrame, evidence: Mapping[str, np.ndarray],
                       thresholds: Mapping[str, float]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Exact current policy: coverage + TUWR, least maximum violation fallback."""
    eps = float(thresholds["coverage_shortfall_epsilon"])
    tau = float(thresholds["tuwr_upper"])
    target = states.target_coverage.to_numpy(float)[:, None]
    warm = states.rolling_state.ne("cold_start").to_numpy()[:, None]
    cov = evidence["coverage"] >= target - eps
    tuwr = (~warm) | (np.isfinite(evidence["tuwr"]) & (evidence["tuwr"] <= tau))
    admissible = cov & tuwr
    empty = ~admissible.any(axis=1)
    excess_cov = np.where(np.isfinite(evidence["coverage"]),
                          np.maximum(0, target - eps - evidence["coverage"]) / eps,
                          np.inf)
    excess_tuwr = np.where(warm, np.maximum(0, evidence["tuwr"] - tau) / tau, 0)
    excess_tuwr = np.where(warm & ~np.isfinite(evidence["tuwr"]), np.inf, excess_tuwr)
    violation = np.maximum(excess_cov, excess_tuwr)
    eligible = admissible.copy()
    minimum = violation.min(axis=1)
    eligible[empty] = violation[empty] <= minimum[empty, None] + 1e-12
    selected = choose_by_score(evidence, eligible)
    rows = np.arange(len(states))
    if not admissible[rows, selected][~empty].all():
        raise AssertionError("Selected an inadmissible candidate despite a nonempty admissible set")
    np.testing.assert_allclose(violation[rows, selected][empty], minimum[empty], rtol=0, atol=1e-12)
    return selected, {"guardrail_empty_fallback": empty,
                      "selected_action_guardrail_pass": admissible[rows, selected],
                      "selected_max_excess": violation[rows, selected],
                      "eligible": eligible, "admissible": admissible}


@dataclass
class FittedPolicies:
    bundle: SourceBundle
    decisions: pd.DataFrame
    evidence: pd.DataFrame
    selected_actions: np.ndarray
    audit: dict[str, Any]


def fit_policies(bundle: SourceBundle) -> FittedPolicies:
    contracts, config = contracts_and_config()
    prepared = original.prepare_clara_fit(bundle.stats, contracts=contracts,
                                          v4_config=config, require_formal_identity=False)
    decisions, evidence, choices, audits = [], [], [], []
    for price in PRICES:
        fit = original.fit_price_conditioned_clara(prepared, price=price, v4_config=config)
        ev = fit.action_evidence.sort_values(["full_state_code", "action_order"], kind="mergesort").copy()
        matrices = {key: ev[column].to_numpy(float).reshape(STATE_COUNT, 6)
                    for key, column in (("risk_score", "risk_score"),
                                        ("coverage", "coverage_point"),
                                        ("tuwr", "tuwr_point"), ("ard", "ard_point"))}
        selected, guard = directional_repair(bundle.stats.states, matrices, fit.audit["adaptive_guardrail_thresholds"])
        dec = fit.decisions.copy()
        dec["selected_action"] = np.asarray(ACTIONS)[selected]
        dec["selected_action_index"] = selected
        for key in ("guardrail_empty_fallback", "selected_action_guardrail_pass", "selected_max_excess"):
            dec[key] = guard[key]
        ev["guardrail_pass"] = guard["admissible"].reshape(-1)
        ev["eligible_after_fallback"] = guard["eligible"].reshape(-1)
        decisions.append(dec)
        evidence.append(ev)
        choices.append(selected)
        audits.append(dict(price=price.as_dict(), thresholds=fit.audit["adaptive_guardrail_thresholds"],
                           n_min_counts=fit.audit["n_min_counts"], nu_counts=fit.audit["nu_counts"],
                           empty_state_count=int(guard["guardrail_empty_fallback"].sum()),
                           selected_action_counts={a: int((selected == i).sum()) for i, a in enumerate(ACTIONS)}))
    audit = dict(schema=SCHEMA, status="PASS", algorithm="DIRECTIONAL_REPAIR",
                 source_identity_sha256=bundle.audit["identity_sha256"],
                 future_source_feedback_used=False, per_price=audits,
                 reused_mathematics="seven-level recursive shrinkage, zone-cluster standard errors, adaptive n_min and nu",
                 algorithm_configuration=config,
                 original_risk_module_sha256=hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest())
    return FittedPolicies(bundle, pd.concat(decisions, ignore_index=True),
                          pd.concat(evidence, ignore_index=True), np.stack(choices), audit)


def replay_stream(model: FittedPolicies, stream: Mapping[str, Any],
                   predictor: str, horizon: int) -> np.ndarray:
    """Return action indices [five prices, eleven coverages, prediction times]."""
    codes = state_codes(stream, predictor, horizon, model.bundle.width_edges)
    return model.selected_actions[:, codes]


def save_bundle(bundle: SourceBundle, directory: Path | str) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "source_statistics.npz", **bundle.stats.arrays,
                        cart_best_count=bundle.stats.cart_best_count, width_edges=bundle.width_edges)
    bundle.stats.stream_audit.to_csv(path / "source_time_audit.csv", index=False)
    (path / "source_audit.json").write_text(json.dumps(bundle.audit, ensure_ascii=False, indent=2), encoding="utf-8")


def load_bundle(directory: Path | str) -> SourceBundle:
    path = Path(directory)
    audit = json.loads((path / "source_audit.json").read_text(encoding="utf-8"))
    with np.load(path / "source_statistics.npz") as stored:
        best = stored["cart_best_count"].copy()
        edges = stored["width_edges"].copy()
        arrays = {name: stored[name].copy() for name in stored.files if name not in ("cart_best_count", "width_edges")}
    states = make_states()
    _, checksum = original._statistics_array_content_identity(arrays, best)
    if checksum != audit["sufficient_statistics_content_sha256"]:
        raise ValueError("Saved sufficient statistics checksum differs")
    stats = original.SixActionSufficientStats(schema=SCHEMA, actions=ACTIONS, prices=PRICES,
        heldout_zone=audit["heldout_zone"], source_zones=tuple(audit["source_zones"]), seed=int(audit["seed"]),
        horizons=HORIZONS, predictors=PREDICTORS, coverages=tuple(COVERAGES), states=states, arrays=arrays,
        cart_best_count=best, stream_audit=pd.read_csv(path / "source_time_audit.csv"), audit=audit)
    return SourceBundle(stats, edges, audit)


def save_policies(model: FittedPolicies, directory: Path | str) -> None:
    path = Path(directory)
    save_bundle(model.bundle, path)
    model.decisions.to_parquet(path / "clara_decisions.parquet", index=False)
    model.evidence.to_parquet(path / "clara_action_evidence.parquet", index=False)
    np.save(path / "clara_selected_actions.npy", model.selected_actions)
    (path / "clara_audit.json").write_text(json.dumps(model.audit, ensure_ascii=False, indent=2), encoding="utf-8")
