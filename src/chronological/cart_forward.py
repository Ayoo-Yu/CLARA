"""CART selection using source-zone feedback available before target testing.

The original 30-configuration grid, compact classification expansion, class
balancing, deterministic CART fit, and source-validation selection rule are
preserved.  Data and hyperparameters are rebuilt from the 28-day source preparation
period; no archived source labels or selected configurations are imported.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import itertools
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable, Sequence

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from forward_common import ROOT, EXP as EXPERIMENTS, AUTH
if str(AUTH) not in sys.path:
    sys.path.insert(0, str(AUTH))

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from baseline_common import FrozenStateEncoder
from baseline_compact_training import _capped_classification_expansion
from baseline_selection import select_configuration
from clara_event_contract import canonical_json_sha256, load_frozen_contracts

ACTIONS = (
    "Static", "ACI", "AgACI", "EnbPI_RH", "TunedSingleConformal", "EqualEndpointEnsemble"
)
ZONES = tuple(f"zone{i}" for i in range(1, 11))
SEEDS = (0, 1, 2)
PREDICTORS = ("Ridge", "GBR", "MLP", "QRLSTM")
HORIZONS = (1, 3, 6, 12, 24)
COVERAGES = np.asarray((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99))
STATE_FIELDS = ("predictor", "horizon_group", "target_coverage", "ramp_state", "rolling_state", "raw_width_state")
CAPACITY_WEIGHT = 3.56
REFERENCE_RATIO = 20.0 / 3.56


@lru_cache(maxsize=1)
def contracts():
    return load_frozen_contracts()


def configuration_grid() -> list[dict[str, Any]]:
    registry = contracts().baseline_registry["mandatory_contextual_selectors"]
    grid = next(row["grid"] for row in registry if row["id"] == "CARTBestAction")
    configurations = [
        {"max_depth": depth, "min_samples_leaf": int(leaf), "class_weight": weight}
        for depth, leaf, weight in itertools.product(
            grid["max_depth"], grid["min_samples_leaf"], grid["class_weight"]
        )
    ]
    if len(configurations) != 30:
        raise ValueError("The original CART grid must contain 30 configurations")
    return configurations


def configuration_id(config: dict[str, Any]) -> str:
    return canonical_json_sha256({"baseline_id": "CARTBestAction", **config, "random_state": 0})


def encode_states(states: pd.DataFrame) -> np.ndarray:
    events = states.loc[:, STATE_FIELDS].copy()
    events.insert(0, "event_id", [f"forward-cart-state-{i}" for i in range(len(states))])
    encoder = FrozenStateEncoder(contracts())
    encoded = encoder.transform(events)
    return encoded.loc[:, encoder.feature_names].to_numpy(dtype=np.float64)


@dataclass
class CartStatistics:
    """Event-weighted quantities on the common complete state vocabulary."""

    event_count: np.ndarray
    best_count: np.ndarray
    errf_sum: np.ndarray
    covered_sum: np.ndarray
    tuwr_sum: np.ndarray
    tuwr_count: np.ndarray
    source_zones: tuple[str, ...]
    event_rows: int = 0
    latest_available_ns: int = -1

    @classmethod
    def empty(cls, state_count: int, source_zones: Sequence[str]) -> "CartStatistics":
        shape = (state_count, len(ACTIONS))
        return cls(
            event_count=np.zeros(state_count, np.int64),
            best_count=np.zeros(shape, np.int64),
            errf_sum=np.zeros(shape),
            covered_sum=np.zeros(shape),
            tuwr_sum=np.zeros(shape),
            tuwr_count=np.zeros(shape, np.int64),
            source_zones=tuple(source_zones),
        )

    def validate(self) -> None:
        if not np.array_equal(self.event_count, self.best_count.sum(axis=1)):
            raise ValueError("Eventwise CART best-action counts do not close")
        if self.event_rows != int(self.event_count.sum()):
            raise ValueError("CART event count does not close")
        if not all(np.isfinite(x).all() for x in (self.errf_sum, self.covered_sum, self.tuwr_sum)):
            raise ValueError("CART statistics contain non-finite values")
        if np.any(self.tuwr_count > self.event_count[:, None]):
            raise ValueError("CART TUWR counts exceed event counts")

    def add_stream(
        self,
        stream: dict[str, Any],
        codes: np.ndarray,
        *,
        price_ratio: float = REFERENCE_RATIO,
        cutoff_ns: int | None = None,
    ) -> None:
        lower = np.asarray(stream["lower"], dtype=np.float64)
        upper = np.asarray(stream["upper"], dtype=np.float64)
        y = np.asarray(stream["y"], dtype=np.float64)
        center = np.asarray(stream["center"], dtype=np.float64)
        if lower.shape != upper.shape or lower.shape != (6, len(COVERAGES), len(y)):
            raise ValueError(f"Unexpected six-candidate shape: {lower.shape}")
        if codes.shape != lower.shape[1:]:
            raise ValueError("CART state codes do not match candidate events")
        issue_ns = np.asarray(stream["issue_ns"], dtype=np.int64)
        label_ns = np.asarray(stream["label_ns"], dtype=np.int64)
        available_ns = np.asarray(stream["available_ns"], dtype=np.int64)
        if not np.all((issue_ns < label_ns) & (label_ns < available_ns)):
            raise ValueError("CART source stream violates strict feedback ordering")
        if cutoff_ns is not None and np.any((issue_ns >= int(cutoff_ns)) | (label_ns >= int(cutoff_ns)) | (available_ns > int(cutoff_ns))):
            raise ValueError("CART source violates issue < cutoff, label < cutoff, availability <= cutoff")
        self.latest_available_ns = max(self.latest_available_ns, int(available_ns.max()))
        flat_codes = codes.reshape(-1)
        state_count = len(self.event_count)
        self.event_count += np.bincount(flat_codes, minlength=state_count)
        self.event_rows += len(flat_codes)
        capacity = np.maximum(upper - center[None, None, :], 0.0) + np.maximum(center[None, None, :] - lower, 0.0)
        misses = np.maximum(y[None, None, :] - upper, 0.0) + np.maximum(lower - y[None, None, :], 0.0)
        # Keep the original multiplication/addition order: algebraic
        # refactoring can change exact near-tie labels at floating precision.
        loss = CAPACITY_WEIGHT * capacity + (CAPACITY_WEIGHT * float(price_ratio)) * misses
        covered = (lower <= y[None, None, :]) & (y[None, None, :] <= upper)
        # Preserve the original six-action event-label tie rule exactly.
        best = np.argmin(loss, axis=0).reshape(-1)
        tuwr = np.asarray(stream["candidate_tuwr"], dtype=np.float64)
        if tuwr.shape != lower.shape:
            raise ValueError("CART candidate TUWR does not match candidate events")
        for action in range(len(ACTIONS)):
            self.best_count[:, action] += np.bincount(flat_codes[best == action], minlength=state_count)
            self.errf_sum[:, action] += np.bincount(flat_codes, weights=loss[action].reshape(-1), minlength=state_count)
            self.covered_sum[:, action] += np.bincount(flat_codes, weights=covered[action].reshape(-1), minlength=state_count)
            values = tuwr[action].reshape(-1)
            valid = np.isfinite(values)
            self.tuwr_sum[:, action] += np.bincount(flat_codes[valid], weights=values[valid], minlength=state_count)
            self.tuwr_count[:, action] += np.bincount(flat_codes[valid], minlength=state_count)


@dataclass
class CartPolicy:
    configuration: dict[str, Any]
    classifier: DecisionTreeClassifier
    state_actions: np.ndarray
    fit_zones: tuple[str, ...]
    fit_event_count: int

    def predict_codes(self, codes: np.ndarray) -> np.ndarray:
        return self.state_actions[np.asarray(codes, dtype=np.int64)]


def fit_counts(
    best_count: np.ndarray,
    *,
    feature_matrix: np.ndarray,
    config: dict[str, Any],
    fit_zones: Sequence[str],
) -> CartPolicy:
    """Use the exact original compact supervised CART fitting procedure."""
    counts = np.asarray(best_count, dtype=np.int64)
    if counts.shape != (len(feature_matrix), 6) or np.any(counts < 0):
        raise ValueError("Invalid six-action CART training counts")
    observed = counts.sum(axis=1) > 0
    features, labels, weights = _capped_classification_expansion(
        feature_matrix=feature_matrix[observed],
        class_counts_by_state=counts[observed],
        class_labels=ACTIONS,
        min_samples_leaf=int(config["min_samples_leaf"]),
        balanced=config["class_weight"] == "balanced",
    )
    classifier = DecisionTreeClassifier(
        max_depth=config["max_depth"],
        min_samples_leaf=int(config["min_samples_leaf"]),
        class_weight=None,
        random_state=0,
    )
    classifier.fit(features, labels, sample_weight=weights)
    labels_by_state = classifier.predict(feature_matrix)
    action_index = {action: i for i, action in enumerate(ACTIONS)}
    state_actions = np.asarray([action_index[str(x)] for x in labels_by_state], dtype=np.int8)
    return CartPolicy(dict(config), classifier, state_actions, tuple(fit_zones), int(counts.sum()))


def selected_metrics(statistics: CartStatistics, state_actions: np.ndarray, states: pd.DataFrame) -> dict[str, Any]:
    statistics.validate()
    actions = np.asarray(state_actions, dtype=np.int64)
    rows = np.arange(len(statistics.event_count))
    if actions.shape != rows.shape or np.any((actions < 0) | (actions >= 6)):
        raise ValueError("Invalid validation action indices")
    event_count = int(statistics.event_count.sum())
    tuwr_count = int(statistics.tuwr_count[rows, actions].sum())
    if event_count <= 0 or tuwr_count <= 0:
        raise ValueError("Validation has no events or complete TUWR observations")
    target_total = float(np.dot(statistics.event_count, states["target_coverage"].to_numpy(dtype=float)))
    return {
        "mean_errf": float(statistics.errf_sum[rows, actions].sum() / event_count),
        "mean_coverage_gap": float((statistics.covered_sum[rows, actions].sum() - target_total) / event_count),
        "mean_tuwr": float(statistics.tuwr_sum[rows, actions].sum() / tuwr_count),
        "event_count": event_count,
        "tuwr_observation_count": tuwr_count,
    }


def tune_fold(
    training: CartStatistics,
    validation: CartStatistics,
    *,
    states: pd.DataFrame,
    outer_zone: str,
    validation_zone: str,
) -> pd.DataFrame:
    """Evaluate all original configurations on one independent source zone."""
    training.validate()
    validation.validate()
    expected_fit = set(ZONES) - {outer_zone, validation_zone}
    if set(training.source_zones) != expected_fit or validation.source_zones != (validation_zone,):
        raise ValueError("CART inner fold must fit eight zones and validate the ninth")
    feature_matrix = encode_states(states)
    rows: list[dict[str, Any]] = []
    grid = configuration_grid()
    observed = training.best_count.sum(axis=1) > 0
    action_index = {action: i for i, action in enumerate(ACTIONS)}
    # The five depths share exactly the same expanded sample matrix for each
    # leaf/class-weight pair. Reuse this matrix without changing its values or
    # sample weights; sklearn otherwise repeats the same CSR->CSC conversion.
    groups: dict[tuple[int, Any], list[tuple[int, dict[str, Any]]]] = {}
    for rank, config in enumerate(grid):
        groups.setdefault((config["min_samples_leaf"], config["class_weight"]), []).append((rank, config))
    for (leaf_size, class_weight), group in groups.items():
        features, labels, weights = _capped_classification_expansion(
            feature_matrix=feature_matrix[observed],
            class_counts_by_state=training.best_count[observed],
            class_labels=ACTIONS,
            min_samples_leaf=int(leaf_size),
            balanced=class_weight == "balanced",
        )
        features = features.astype(np.float32).tocsc()
        features.sort_indices()
        for rank, config in group:
            started = time.perf_counter()
            classifier = DecisionTreeClassifier(max_depth=config["max_depth"],
                                                min_samples_leaf=int(leaf_size),
                                                class_weight=None, random_state=0)
            classifier.fit(features, labels, sample_weight=weights)
            actions = np.asarray([action_index[str(x)] for x in classifier.predict(feature_matrix)], dtype=np.int8)
            rows.append({
                "outer_heldout_zone": outer_zone,
                "inner_validation_zone": validation_zone,
                "configuration_id": configuration_id(config),
                "configuration_json": json.dumps(config, sort_keys=True),
                "complexity_rank": rank,
                **selected_metrics(validation, actions, states),
                "fit_event_count": training.event_rows,
                "fit_zone_count": len(training.source_zones),
                "fit_seconds": time.perf_counter() - started,
                "latest_training_feedback_ns": training.latest_available_ns,
                "latest_validation_feedback_ns": validation.latest_available_ns,
            })
    return pd.DataFrame(rows).sort_values("complexity_rank").reset_index(drop=True)


def select_outer_configuration(scores: pd.DataFrame, outer_zone: str) -> dict[str, Any]:
    required_zones = set(ZONES) - {outer_zone}
    if set(scores["inner_validation_zone"]) != required_zones:
        raise ValueError("CART outer selection requires all nine source validation folds")
    if set(scores["outer_heldout_zone"]) != {outer_zone} or len(scores) != 270:
        raise ValueError("Unexpected CART outer validation identity or count")
    decision = select_configuration(contracts=contracts(), validation_scores=scores)
    selected = scores.loc[scores.configuration_id == decision.selected_configuration_id].iloc[0]
    return {
        "outer_heldout_zone": outer_zone,
        "config": json.loads(selected.configuration_json),
        **decision.as_record(),
        "source_zones": sorted(required_zones),
        "inner_fold_count": 9,
        "configurations_per_fold": 30,
        "calibration_action_count": 6,
        "forecast_horizons": list(HORIZONS),
        "seeds_pooled": list(SEEDS),
        "price_ratio": REFERENCE_RATIO,
    }


def fit_from_clara_stats(stats: Any, config: dict[str, Any], price_index: int) -> CartPolicy:
    """Fit the final price-specific CART from newly built CLARA source counts."""
    feature_matrix = encode_states(stats.states)
    counts = np.asarray(stats.cart_best_count[price_index]).sum(axis=1).T
    event_counts = np.asarray(stats.arrays["event_count"]).sum(axis=0)
    if not np.array_equal(counts.sum(axis=1), event_counts):
        raise ValueError("Final CART labels do not match corrected source event counts")
    fit_zones = getattr(stats, "source_zones", ())
    return fit_counts(counts, feature_matrix=feature_matrix, config=config, fit_zones=fit_zones)


def run_outer_selection(
    outer_zone: str,
    *,
    load_stream: Callable[..., dict[str, Any]],
    select_tsc: Callable[[Sequence[str]], int],
    cutoff_ns: int,
    output_dir: str | Path,
    data_identity: str,
    source_split: str = "source",
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run/resume all nine inner folds, using only the pre-test source period.

    ``select_tsc(excluded_zones)`` must select a conformal configuration from
    the remaining zones.  Both this choice and width quantiles are rebuilt
    without either the outer target or the inner validation zone.
    ``data_identity`` identifies the corrected stream artifacts and generation
    code; it is recorded and checked before checkpoint reuse.
    """
    from clara_forward import fit_width_edges, make_states, state_codes

    if outer_zone not in ZONES or not data_identity:
        raise ValueError("CART requires a valid outer zone and corrected-data identity")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    states = make_states()
    code_identity = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    score_frames = []
    for validation_zone in ZONES:
        if validation_zone == outer_zone:
            continue
        training_zones = tuple(z for z in ZONES if z not in {outer_zone, validation_zone})
        tsc_index = int(select_tsc((outer_zone, validation_zone)))
        prefix = output_dir / f"cart_{outer_zone}_validate_{validation_zone}"
        score_path = prefix.with_suffix(".parquet")
        manifest_path = prefix.with_suffix(".json")
        identity = {
            "protocol": "GEFCOM_LAST20_PREFIX28D_CART_NESTED_V1",
            "outer_zone": outer_zone,
            "validation_zone": validation_zone,
            "training_zones": list(training_zones),
            "tsc_index": tsc_index,
            "cutoff_ns": int(cutoff_ns),
            "data_identity": str(data_identity),
            "code_sha256": code_identity,
            "source_split": source_split,
        }
        if score_path.exists() and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("identity") == identity and manifest.get("score_sha256") == hashlib.sha256(score_path.read_bytes()).hexdigest():
                frame = pd.read_parquet(score_path)
                if len(frame) == 30:
                    score_frames.append(frame)
                    if progress:
                        progress({"event": "cart_inner_resumed", "outer": outer_zone, "validation": validation_zone})
                    continue
        started = time.perf_counter()
        training = CartStatistics.empty(len(states), training_zones)
        validation = CartStatistics.empty(len(states), (validation_zone,))
        width_hashes = {}
        for seed in SEEDS:
            records = []
            validation_records = []
            for zone in (*training_zones, validation_zone):
                for predictor, horizon in itertools.product(PREDICTORS, HORIZONS):
                    stream = load_stream(zone, seed, predictor, horizon, source_split, tsc_index)
                    record = (zone, predictor, horizon, stream)
                    if zone == validation_zone:
                        validation_records.append(record)
                    else:
                        records.append(record)
            # As in the final per-seed evaluation, fit width boundaries only
            # using this seed's eight training zones, then pool label counts.
            width_edges = fit_width_edges(records)
            width_hashes[str(seed)] = hashlib.sha256(np.asarray(width_edges, dtype=np.float64).tobytes()).hexdigest()
            for target, batch in ((training, records), (validation, validation_records)):
                for zone, predictor, horizon, stream in batch:
                    codes = state_codes(stream, predictor, horizon, width_edges)
                    target.add_stream(stream, codes, cutoff_ns=cutoff_ns)
            del records, validation_records, stream, record, batch
        frame = tune_fold(training, validation, states=states, outer_zone=outer_zone, validation_zone=validation_zone)
        frame["tsc_index"] = tsc_index
        score_frames.append(frame)
        frame.to_parquet(score_path, index=False)
        manifest = {
            "identity": identity,
            "score_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
            "width_edges_sha256_by_seed": width_hashes,
            "fit_events": training.event_rows,
            "validation_events": validation.event_rows,
            "latest_training_feedback_ns": training.latest_available_ns,
            "latest_validation_feedback_ns": validation.latest_available_ns,
            "elapsed_seconds": time.perf_counter() - started,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
        if progress:
            progress({"event": "cart_inner_completed", "outer": outer_zone, "validation": validation_zone, "seconds": manifest["elapsed_seconds"]})
    scores = pd.concat(score_frames, ignore_index=True)
    selection = select_outer_configuration(scores, outer_zone)
    scores.to_parquet(output_dir / f"cart_{outer_zone}_all_scores.parquet", index=False)
    (output_dir / f"cart_{outer_zone}_selection.json").write_text(
        json.dumps(selection, indent=2, default=_json_default), encoding="utf-8"
    )
    return selection


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def run_outer(outer_zone: str) -> dict[str, Any]:
    """Data-bound runner; one source-file read feeds all pending inner folds."""
    from forward_common import RUN, dataset_path, load_all_configs, MAIN_CONFIG_INDICES, select_tsc
    from clara_forward import make_states, state_codes
    from linucb_forward import cached_width_edges

    if outer_zone not in ZONES:
        raise ValueError(outer_zone)
    if not (RUN / "DATA_READY.json").is_file():
        raise RuntimeError("Corrected source generation and validation are not complete")
    output_dir = RUN / "cart_results" / outer_zone
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_hash = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).resolve().parent / 'forward_common.py', Path(__file__).resolve().parent / 'prepare_chrono.py', RUN / "tsc_zone_scores.parquet", RUN / "DATA_READY.json"):
        identity_hash.update(path.read_bytes())
    source_zones = tuple(z for z in ZONES if z != outer_zone)
    for z, seed, predictor, horizon in itertools.product(source_zones, SEEDS, PREDICTORS, HORIZONS):
        path = dataset_path(z, seed, predictor, horizon, "source")
        if not path.is_file() or not path.with_suffix(".json").is_file():
            raise FileNotFoundError(f"Corrected source stream not complete: {path}")
        identity_hash.update(path.with_suffix(".json").read_bytes())
    data_identity = identity_hash.hexdigest()
    states = make_states()
    score_frames = []
    folds = {}
    for validation_zone in source_zones:
        tsc_index = int(select_tsc((outer_zone, validation_zone)))
        training_zones = tuple(z for z in source_zones if z != validation_zone)
        prefix = output_dir / f"cart_{outer_zone}_validate_{validation_zone}"
        score_path, manifest_path = prefix.with_suffix(".parquet"), prefix.with_suffix(".json")
        identity = {
            "protocol": "GEFCOM_LAST20_PREFIX28D_CART_NESTED_V1",
            "outer_zone": outer_zone,
            "validation_zone": validation_zone,
            "training_zones": list(training_zones),
            "tsc_index": tsc_index,
            "data_identity": data_identity,
        }
        if score_path.exists() and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("identity") == identity and manifest.get("score_sha256") == hashlib.sha256(score_path.read_bytes()).hexdigest():
                frame = pd.read_parquet(score_path)
                if len(frame) == 30:
                    score_frames.append(frame)
                    print(json.dumps({"event": "cart_inner_resumed", "outer": outer_zone, "validation": validation_zone}), flush=True)
                    continue
        folds[validation_zone] = dict(
            tsc_index=tsc_index,
            training=CartStatistics.empty(len(states), training_zones),
            validation=CartStatistics.empty(len(states), (validation_zone,)),
            identity=identity, score_path=score_path, manifest_path=manifest_path,
            width_hashes={},
        )
    started = time.perf_counter()
    if folds:
        for seed in SEEDS:
            edges = {v: cached_width_edges((outer_zone, v), seed) for v in folds}
            for v, width_edges in edges.items():
                folds[v]["width_hashes"][str(seed)] = hashlib.sha256(width_edges.tobytes()).hexdigest()
            for zone in source_zones:
                for predictor, horizon in itertools.product(PREDICTORS, HORIZONS):
                    all_configs = load_all_configs(zone, seed, predictor, horizon, "source")
                    candidate_cache = {}
                    for validation_zone, fold in folds.items():
                        tsc_index = fold["tsc_index"]
                        if tsc_index not in candidate_cache:
                            inds = list(MAIN_CONFIG_INDICES) + [tsc_index, 19]
                            stream = dict(all_configs)
                            for key in ("lower", "upper", "candidate_tuwr", "candidate_ard"):
                                stream[key] = all_configs[key][inds]
                            candidate_cache[tsc_index] = stream
                        stream = candidate_cache[tsc_index]
                        codes = state_codes(stream, predictor, horizon, edges[validation_zone])
                        target = fold["validation"] if zone == validation_zone else fold["training"]
                        target.add_stream(stream, codes, cutoff_ns=int(stream["source_cutoff_ns"]))
            print(json.dumps({"event": "cart_source_seed_completed", "outer": outer_zone,
                              "seed": seed, "pending_inner_folds": len(folds),
                              "elapsed_seconds": time.perf_counter() - started}), flush=True)
        for validation_zone, fold in folds.items():
            fit_started = time.perf_counter()
            frame = tune_fold(fold["training"], fold["validation"], states=states,
                              outer_zone=outer_zone, validation_zone=validation_zone)
            frame["tsc_index"] = fold["tsc_index"]
            frame.to_parquet(fold["score_path"], index=False)
            manifest = {
                "identity": fold["identity"],
                "score_sha256": hashlib.sha256(fold["score_path"].read_bytes()).hexdigest(),
                "width_edges_sha256_by_seed": fold["width_hashes"],
                "fit_events": fold["training"].event_rows,
                "validation_events": fold["validation"].event_rows,
                "latest_training_feedback_ns": fold["training"].latest_available_ns,
                "latest_validation_feedback_ns": fold["validation"].latest_available_ns,
                "fit_seconds": time.perf_counter() - fit_started,
            }
            fold["manifest_path"].write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
            score_frames.append(frame)
            print(json.dumps({"event": "cart_inner_completed", "outer": outer_zone,
                              "validation": validation_zone, "seconds": manifest["fit_seconds"]}), flush=True)
    scores = pd.concat(score_frames, ignore_index=True)
    selection = select_outer_configuration(scores, outer_zone)
    scores.to_parquet(output_dir / f"cart_{outer_zone}_all_scores.parquet", index=False)
    selection["data_identity"] = data_identity
    selection["elapsed_seconds"] = time.perf_counter() - started
    (output_dir / f"cart_{outer_zone}_selection.json").write_text(json.dumps(selection, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"event": "cart_outer_completed", **selection}, default=_json_default), flush=True)
    return selection


def replay_final(
    outer_zone: str,
    seed: int,
    bundle_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Refit five price-specific trees and select each target candidate."""
    import joblib
    from forward_common import RUN, PRICES, load_stream, select_tsc
    from clara_forward import load_bundle, state_codes

    if not (RUN / "DATA_READY.json").is_file():
        raise RuntimeError("Corrected data are not ready for target replay")
    selection_path = RUN / "cart_results" / outer_zone / f"cart_{outer_zone}_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    bundle_dir = Path(bundle_dir)
    bundle = load_bundle(bundle_dir)
    stats = bundle.stats
    if stats.heldout_zone != outer_zone or stats.seed != int(seed):
        raise ValueError("CART source bundle outer zone/seed does not match")
    if set(stats.source_zones) != set(ZONES) - {outer_zone}:
        raise ValueError("CART final fit must use exactly the other nine zones")
    tsc_index = int(select_tsc((outer_zone,)))
    if int(bundle.audit["tsc_configuration_index"]) != tsc_index:
        raise ValueError("CART bundle and newly selected TSC do not match")
    output_dir = Path(output_dir or RUN / "cart_results" / outer_zone / f"target_seed{seed}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    policies = [fit_from_clara_stats(stats, selection["config"], i) for i in range(len(PRICES))]
    state_actions = np.stack([p.state_actions for p in policies])
    np.save(output_dir / "state_actions.npy", state_actions)
    joblib.dump([p.classifier for p in policies], output_dir / "cart_classifiers.joblib", compress=3)
    features = encode_states(stats.states)
    evidence = []
    for price_index, policy in enumerate(policies):
        frame = stats.states.copy()
        frame["price_ratio"] = float(PRICES[price_index])
        frame["action_index"] = policy.state_actions
        frame["selected_action"] = [ACTIONS[i] for i in policy.state_actions]
        leaf_ids = policy.classifier.apply(features)
        frame["leaf_id"] = leaf_ids
        frame["leaf_weighted_training_samples"] = policy.classifier.tree_.weighted_n_node_samples[leaf_ids]
        probs = policy.classifier.predict_proba(features)
        for action in ACTIONS:
            frame[f"class_probability__{action}"] = 0.0
        for i, action in enumerate(policy.classifier.classes_):
            frame[f"class_probability__{action}"] = probs[:, i]
        evidence.append(frame)
    pd.concat(evidence, ignore_index=True).to_parquet(output_dir / "cart_state_evidence.parquet", index=False)
    decisions = {}
    target_rows = []
    for predictor, horizon in itertools.product(PREDICTORS, HORIZONS):
        stream = load_stream(outer_zone, seed, predictor, horizon, "target", tsc_index)
        codes = state_codes(stream, predictor, horizon, bundle.width_edges)
        if int(bundle.audit["source_available_max_ns"]) > int(np.min(stream["issue_ns"])):
            raise ValueError("CART fitted source feedback reaches the target evaluation period")
        decisions[f"{predictor}__H{horizon:02d}"] = state_actions[:, codes]
        target_rows.append({
            "predictor": predictor, "horizon": horizon,
            "issue_count": len(stream["issue_ns"]),
            "coverage_event_count": int(codes.size),
            "first_issue_ns": int(np.min(stream["issue_ns"])),
            "last_issue_ns": int(np.max(stream["issue_ns"])),
        })
    np.savez_compressed(output_dir / "decisions.npz", **decisions)
    audit = {
        "schema": "GEFCOM_CHRONOLOGICAL_FROZEN_CART_TARGET_V1", "outer_zone": outer_zone, "seed": int(seed),
        "source_zones": list(stats.source_zones), "source_event_count": int(stats.arrays["event_count"].sum()),
        "source_available_max_ns": int(bundle.audit["source_available_max_ns"]),
        "source_statistics_sha256": hashlib.sha256((bundle_dir / "source_statistics.npz").read_bytes()).hexdigest(),
        "source_audit_sha256": hashlib.sha256((bundle_dir / "source_audit.json").read_bytes()).hexdigest(),
        "hyperparameter_selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "configuration": selection["config"], "tsc_configuration_index": tsc_index,
        "price_ratios": PRICES.tolist(), "target_streams": target_rows,
        "decisions_sha256": hashlib.sha256((output_dir / "decisions.npz").read_bytes()).hexdigest(),
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"event": "cart_target_completed", "outer": outer_zone, "seed": seed,
                      "runtime_seconds": audit["runtime_seconds"]}), flush=True)
    return audit


def wait_for_source_bundle(outer_zone: str, seed: int, timeout_seconds: float = 5400) -> Path:
    from forward_common import RUN
    directory = RUN / "clara" / outer_zone / f"seed{seed}"
    started = time.perf_counter()
    next_notice = 0.0
    while not (directory / "COMPLETE.json").is_file():
        elapsed = time.perf_counter() - started
        if elapsed > timeout_seconds:
            raise TimeoutError(f"Corrected CLARA source bundle is still unavailable: {directory}")
        if elapsed >= next_notice:
            print(json.dumps({"event": "cart_waiting_for_source_bundle", "outer": outer_zone,
                              "seed": seed, "elapsed_seconds": elapsed}), flush=True)
            next_notice = elapsed + 60
        time.sleep(5)
    return directory


def replay_outer_final(outer_zone: str) -> list[dict[str, Any]]:
    audits = []
    for seed in SEEDS:
        bundle = wait_for_source_bundle(outer_zone, seed)
        audits.append(replay_final(outer_zone, seed, bundle))
    return audits


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--show-grid", action="store_true")
    choice.add_argument("--outer", choices=ZONES)
    choice.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mode", choices=("tune", "final"), default="tune")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    if args.show_grid:
        print(json.dumps(configuration_grid(), indent=2, default=_json_default))
    elif args.outer:
        if args.mode == "final":
            if args.seed is None:
                if args.bundle is not None:
                    parser.error("An explicit --bundle requires --seed")
                replay_outer_final(args.outer)
            else:
                bundle = args.bundle or wait_for_source_bundle(args.outer, args.seed)
                replay_final(args.outer, args.seed, bundle)
        else:
            run_outer(args.outer)
    elif args.all:
        if not 1 <= args.workers <= 4:
            parser.error("--workers must be between 1 and 4")
        run = run_outer if args.mode == "tune" else replay_outer_final
        if args.workers == 1:
            for zone in ZONES:
                run(zone)
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(run, z): z for z in ZONES}
                for future in as_completed(futures):
                    future.result()
    else:
        parser.error("Select --outer or --all to run corrected source validation")


if __name__ == "__main__":
    _main()
