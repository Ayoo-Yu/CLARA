"""Six-candidate LinUCB with source-period tuning and causal target replay.

Each zone/seed/price starts from the ridge prior. Only the action selected at an
earlier issue time contributes feedback, and only after its availability time.
All horizons share a model. The evaluator never initializes parameters from
source outcomes; source data determine hyperparameters and state width edges.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
from numba import njit, prange

PREDICTORS = ("Ridge", "GBR", "MLP", "QRLSTM")
HORIZONS = (1, 3, 6, 12, 24)
COVERAGES = np.array((.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99))
ACTIONS = ("Static", "ACI", "AgACI", "EnbPI_RH", "TunedSingleConformal", "EqualEndpointEnsemble")
REFERENCE_RATIO = 20.0 / 3.56
PRICE_RATIOS = (1.0, 2.0, REFERENCE_RATIO, 10.0, 20.0)
CONFIGURATIONS = tuple(itertools.product((.1, .5, 1.0, 2.0), (.1, 1.0, 10.0)))
STATE_SHAPE = (4, 5, 11, 2, 5, 3)
CONTEXT_DIMENSION = 31


def context_features() -> np.ndarray:
    """Indices of intercept and six active one-hot features in each state."""
    codes = np.array(np.unravel_index(np.arange(6600), STATE_SHAPE)).T
    offsets = np.array((1, 5, 10, 21, 23, 28))
    return np.ascontiguousarray(np.column_stack((np.zeros(6600, dtype=int), codes + offsets)), dtype=np.uint8)


CONTEXT_FEATURES = context_features()


@dataclass
class Timeline:
    issue_ns: np.ndarray
    label_ns: np.ndarray
    available_ns: np.ndarray
    context_ids: np.ndarray
    coverage_code: np.ndarray
    capacity: np.ndarray
    exceedance: np.ndarray
    covered: np.ndarray
    candidate_tuwr: np.ndarray
    issue_offsets: np.ndarray
    feedback_order: np.ndarray
    matured_ends: np.ndarray
    stream_code: np.ndarray
    stream_position: np.ndarray
    stream_sizes: tuple[int, ...]
    zone: str
    seed: int
    split: str

    @property
    def size(self) -> int:
        return len(self.context_ids)

    def unpack(self, choices: np.ndarray) -> dict[tuple[str, int], np.ndarray]:
        """Restore event choices to predictor/horizon arrays of shape [11,n]."""
        choices = np.asarray(choices)
        if choices.shape[-1] != self.size:
            raise ValueError("Decision count differs from the timeline")
        result = {}
        for code, n in enumerate(self.stream_sizes):
            mask = self.stream_code == code
            block = np.empty((*choices.shape[:-1], 11 * n), dtype=choices.dtype)
            block[..., self.stream_position[mask]] = choices[..., mask]
            result[(PREDICTORS[code // 5], HORIZONS[code % 5])] = block.reshape(*choices.shape[:-1], 11, n)
        return result

    def audit(self) -> dict[str, Any]:
        return {
            "zone": self.zone, "seed": self.seed, "split": self.split,
            "event_count": self.size,
            "issue_count": len(self.issue_offsets) - 1,
            "first_issue": str(np.datetime64(int(self.issue_ns[0]), "ns")),
            "last_issue": str(np.datetime64(int(self.issue_ns[-1]), "ns")),
            "latest_label_available": str(np.datetime64(int(self.available_ns.max()), "ns")),
            "feedback_count_before_last_issue": int(self.matured_ends[-1]),
            "pending_feedback_at_last_issue": self.size - int(self.matured_ends[-1]),
            "starts_from_ridge_prior": True,
            "source_parameter_initialization": False,
            "selected_action_feedback_only": True,
            "global_horizon_timeline": True,
            "within_issue_feedback": False,
            "terminal_feedback_drain": False,
        }


def _schedule(issue: np.ndarray, label: np.ndarray, available: np.ndarray):
    if len(issue) == 0 or np.any(issue[1:] < issue[:-1]):
        raise ValueError("Timeline must contain sorted issue times")
    if np.any(issue >= label) or np.any(label >= available):
        raise ValueError("Required ordering is issue < label < availability")
    # A row never sees an outcome from its own issue batch. Outcomes enter only
    # before a later batch, sorted by their availability, label, and row order.
    feedback_order = np.lexsort((np.arange(len(issue)), label, available)).astype(np.int64)
    starts = np.r_[0, np.flatnonzero(issue[1:] != issue[:-1]) + 1]
    offsets = np.r_[starts, len(issue)].astype(np.int64)
    matured = np.searchsorted(available[feedback_order], issue[starts], side="right").astype(np.int64)
    return offsets, feedback_order, matured


def _state_codes(stream: Mapping[str, Any], p: int, h: int, edges: np.ndarray) -> np.ndarray:
    width = np.asarray(stream["raw_width"], dtype=np.float64)
    n = width.shape[1]
    if width.shape != (11, n):
        raise ValueError("Raw widths must have shape [11,n]")
    ramp = np.asarray(stream["ramp"], dtype=np.int64)
    rolling = np.asarray(stream["rolling_state"], dtype=np.int64)
    if ramp.shape != (n,) or rolling.shape != (11, n):
        raise ValueError("Incorrect state shape")
    if np.any((ramp < 0) | (ramp > 1)) or np.any((rolling < 0) | (rolling > 4)):
        raise ValueError("Unknown ramp or rolling-state code")
    width_code = np.where(width <= edges[p, h, :, 0, None], 0,
                          np.where(width <= edges[p, h, :, 1, None], 1, 2))
    c = np.arange(11)[:, None]
    return (((((p * 5 + h) * 11 + c) * 2 + ramp) * 5 + rolling) * 3 + width_code).astype(np.uint16)


def build_timeline(
    zone: str, seed: int, split: str, tsc_index: int, width_edges: np.ndarray,
    *, loader: Callable[..., Mapping[str, Any]] | None = None,
) -> Timeline:
    """Load one zone/seed, preserving the common candidate-bank event set.

    ``width_edges`` has axes predictor,horizon,coverage,quantile and must have
    been fitted on the relevant eight (inner) or nine (outer) source zones.
    """
    if loader is None:
        from forward_common import load_stream
        loader = load_stream
    edges = np.asarray(width_edges, dtype=np.float64)
    if edges.shape != (4, 5, 11, 2) or not np.isfinite(edges).all():
        raise ValueError("Expected finite width edges [4,5,11,2]")
    if np.any(edges[..., 0] > edges[..., 1]):
        raise ValueError("Width quantiles are reversed")
    parts: dict[str, list[np.ndarray]] = {key: [] for key in (
        "issue", "label", "available", "context", "coverage", "capacity",
        "exceedance", "covered", "tuwr", "stream", "position")}
    sizes = []
    for p, predictor in enumerate(PREDICTORS):
        for h, horizon in enumerate(HORIZONS):
            stream = loader(zone, int(seed), predictor, int(horizon), split, int(tsc_index))
            if split in ("source", "fit"):
                from clara_forward import _verify_source_times
                _verify_source_times(stream)
            issue = np.asarray(stream["issue_ns"], dtype=np.int64)
            label = np.asarray(stream["label_ns"], dtype=np.int64)
            available = np.asarray(stream["available_ns"], dtype=np.int64)
            n = len(issue)
            lo = np.asarray(stream["lower"], dtype=np.float64)
            up = np.asarray(stream["upper"], dtype=np.float64)
            if (lo.shape != (6, 11, n) or up.shape != lo.shape
                    or not np.isfinite(lo).all() or not np.isfinite(up).all() or np.any(lo > up)):
                raise ValueError("Expected six ordered candidate intervals [6,11,n]")
            y = np.asarray(stream["y"], dtype=np.float64)[None, None, :]
            center = np.asarray(stream["center"], dtype=np.float64)[None, None, :]
            if not np.isfinite(y).all() or not np.isfinite(center).all():
                raise ValueError("Targets and forecast centers must be finite")
            cap = np.maximum(center - lo, 0) + np.maximum(up - center, 0)
            miss = np.maximum(lo - y, 0) + np.maximum(y - up, 0)
            hit = ((lo <= y) & (y <= up)).astype(np.uint8)
            tuwr = np.asarray(stream["candidate_tuwr"], dtype=np.float64)
            if tuwr.shape != lo.shape:
                raise ValueError("Candidate TUWR shape differs from candidates")
            parts["issue"].append(np.tile(issue, 11))
            parts["label"].append(np.tile(label, 11))
            parts["available"].append(np.tile(available, 11))
            parts["context"].append(_state_codes(stream, p, h, edges).ravel())
            parts["coverage"].append(np.repeat(np.arange(11, dtype=np.uint8), n))
            parts["capacity"].append(cap.reshape(6, -1).T)
            parts["exceedance"].append(miss.reshape(6, -1).T)
            parts["covered"].append(hit.reshape(6, -1).T)
            parts["tuwr"].append(tuwr.reshape(6, -1).T)
            parts["stream"].append(np.full(11 * n, p * 5 + h, dtype=np.uint8))
            parts["position"].append(np.arange(11 * n, dtype=np.int32))
            sizes.append(n)
    data = {key: np.concatenate(values) for key, values in parts.items()}
    # Only within-issue order is arbitrary; all decisions share one pre-batch
    # model, and accumulated sufficient statistics are independent of that order.
    order = np.argsort(data["issue"], kind="stable")
    data = {key: np.ascontiguousarray(value[order]) for key, value in data.items()}
    offsets, feedback_order, matured = _schedule(data["issue"], data["label"], data["available"])
    return Timeline(data["issue"], data["label"], data["available"],
                    data["context"], data["coverage"], data["capacity"],
                    data["exceedance"], data["covered"], data["tuwr"],
                    offsets, feedback_order, matured, data["stream"], data["position"],
                    tuple(sizes), str(zone), int(seed), str(split))


@njit(cache=True)
def _choose(inverse, theta, active, alpha):
    scores = np.empty(6, dtype=np.float64)
    minimum = np.inf
    for action in range(6):
        mean = 0.0
        variance = 0.0
        for left in range(7):
            lf = active[left]
            mean += theta[action, lf]
            for right in range(7):
                variance += inverse[action, lf, active[right]]
        scores[action] = mean - alpha * np.sqrt(max(variance, 0.0))
        minimum = min(minimum, scores[action])
    for action in range(6):
        if scores[action] <= minimum + 1e-12:
            return action
    return 0


@njit(cache=True, parallel=True)
def _run_grid(context_ids, active_lookup, coverage_code, issue_offsets,
              feedback_order, matured_ends, capacity, exceedance, covered,
              candidate_tuwr, alphas, l2s, ratios):
    count = len(context_ids)
    configs = len(alphas)
    decisions = np.zeros((configs, count), dtype=np.uint8)
    # loss sum, coverage gap sum, candidate TUWR sum, candidate TUWR count
    totals = np.zeros((configs, 4), dtype=np.float64)
    updates = np.zeros((configs, 6), dtype=np.int64)
    for config in prange(configs):
        covariance = np.zeros((6, 31, 31), dtype=np.float64)
        inverse = np.zeros((6, 31, 31), dtype=np.float64)
        response = np.zeros((6, 31), dtype=np.float64)
        theta = np.zeros((6, 31), dtype=np.float64)
        for action in range(6):
            for feature in range(31):
                covariance[action, feature, feature] = l2s[config]
                inverse[action, feature, feature] = 1.0 / l2s[config]
        stamp = np.full(6600, -1, dtype=np.int32)
        cached = np.zeros(6600, dtype=np.uint8)
        dirty = np.zeros(6, dtype=np.uint8)
        pointer = 0
        for issue_index in range(len(issue_offsets) - 1):
            matured_end = matured_ends[issue_index]
            while pointer < matured_end:
                event = feedback_order[pointer]
                action = decisions[config, event]
                active = active_lookup[context_ids[event]]
                loss = 3.56 * (capacity[event, action] + ratios[config] * exceedance[event, action])
                for left in range(7):
                    lf = active[left]
                    response[action, lf] += loss
                    for right in range(7):
                        covariance[action, lf, active[right]] += 1.0
                dirty[action] = 1
                updates[config, action] += 1
                pointer += 1
            # The existing compact LinUCB implementation accumulates A and b,
            # then solves once before the next issue batch. This preserves the
            # selected-feedback linear model and avoids eventwise matrix inverses.
            for action in range(6):
                if dirty[action]:
                    inverse[action] = np.linalg.inv(covariance[action])
                    theta[action] = inverse[action] @ response[action]
                    dirty[action] = 0
            for event in range(issue_offsets[issue_index], issue_offsets[issue_index + 1]):
                context = context_ids[event]
                if stamp[context] != issue_index:
                    cached[context] = _choose(inverse, theta, active_lookup[context], alphas[config])
                    stamp[context] = issue_index
                action = cached[context]
                decisions[config, event] = action
                totals[config, 0] += 3.56 * (capacity[event, action] + ratios[config] * exceedance[event, action])
                totals[config, 1] += covered[event, action] - COVERAGES[coverage_code[event]]
                tv = candidate_tuwr[event, action]
                if np.isfinite(tv):
                    totals[config, 2] += tv
                    totals[config, 3] += 1.0
    return decisions, totals, updates


def run_grid(
    timeline: Timeline,
    configurations: Sequence[tuple[float, float]] = CONFIGURATIONS,
    price_ratios: Sequence[float] = (REFERENCE_RATIO,),
) -> dict[str, Any]:
    """Replay the Cartesian product of configurations and prices from fresh priors."""
    run_specs = [(float(a), float(l2), float(r)) for r in price_ratios for a, l2 in configurations]
    params = np.asarray(run_specs, dtype=np.float64)
    if len(params) == 0 or not np.isfinite(params).all() or np.any(params[:, 0] < 0) or np.any(params[:, 1:] <= 0):
        raise ValueError("Invalid LinUCB hyperparameters or prices")
    started = perf_counter()
    choices, totals, updates = _run_grid(
        timeline.context_ids, CONTEXT_FEATURES, timeline.coverage_code,
        timeline.issue_offsets, timeline.feedback_order, timeline.matured_ends,
        timeline.capacity, timeline.exceedance, timeline.covered,
        timeline.candidate_tuwr, np.ascontiguousarray(params[:, 0]),
        np.ascontiguousarray(params[:, 1]), np.ascontiguousarray(params[:, 2]))
    seconds = perf_counter() - started
    if not np.all(updates.sum(axis=1) == timeline.matured_ends[-1]):
        raise RuntimeError("LinUCB feedback counts do not match matured outcomes")
    rows = []
    for i, (alpha, l2, ratio) in enumerate(run_specs):
        rows.append({
            "configuration_id": f"alpha={alpha:g}__l2={l2:g}",
            "exploration_alpha": alpha, "l2_regularization": l2,
            "price_ratio": ratio, "complexity_rank": list(configurations).index((alpha, l2)),
            "mean_errf": float(totals[i, 0] / timeline.size),
            "mean_coverage_gap": float(totals[i, 1] / timeline.size),
            "mean_tuwr": float(totals[i, 2] / totals[i, 3]) if totals[i, 3] else float("nan"),
            "event_count": timeline.size, "tuwr_observation_count": int(totals[i, 3]),
            "errf_sum": float(totals[i, 0]), "coverage_gap_sum": float(totals[i, 1]),
            "tuwr_sum": float(totals[i, 2]),
            "decision_sha256": hashlib.sha256(choices[i].tobytes()).hexdigest(),
            "action_counts": np.bincount(choices[i], minlength=6).tolist(),
            "action_update_counts": updates[i].tolist(),
        })
    return {"choices": choices, "scores": pd.DataFrame(rows),
            "audit": {**timeline.audit(), "run_count": len(run_specs), "runtime_seconds": seconds}}


def run_inner_fold(
    outer_zone: str, validation_zone: str, tsc_index: int,
    width_edges_by_seed: Mapping[int, np.ndarray], *,
    loader: Callable[..., Mapping[str, Any]] | None = None,
    split: str = "source", output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Replay all 12 configurations on one inner zone, pooling the three seeds."""
    if outer_zone == validation_zone:
        raise ValueError("Outer and inner validation zones must differ")
    frames = []
    audits = []
    for seed in (0, 1, 2):
        timeline = build_timeline(validation_zone, seed, split, tsc_index,
                                  width_edges_by_seed[seed], loader=loader)
        run = run_grid(timeline)
        score = run["scores"].copy()
        score["seed"] = seed
        frames.append(score)
        audits.append(run["audit"])
    raw = pd.concat(frames, ignore_index=True)
    keys = ["configuration_id", "exploration_alpha", "l2_regularization", "price_ratio", "complexity_rank"]
    sums = ["errf_sum", "coverage_gap_sum", "tuwr_sum", "event_count", "tuwr_observation_count"]
    scores = raw.groupby(keys, sort=False, as_index=False)[sums].sum()
    scores["mean_errf"] = scores.errf_sum / scores.event_count
    scores["mean_coverage_gap"] = scores.coverage_gap_sum / scores.event_count
    scores["mean_tuwr"] = scores.tuwr_sum / scores.tuwr_observation_count
    scores["outer_heldout_zone"] = outer_zone
    scores["inner_validation_zone"] = validation_zone
    scores["tsc_index"] = int(tsc_index)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        scores.to_parquet(out / "validation_scores.parquet", index=False)
        raw.to_parquet(out / "validation_seed_scores.parquet", index=False)
        (out / "audit.json").write_text(json.dumps({"timelines": audits}, indent=2), encoding="utf-8")
    return scores


def replay_final(
    zone: str, seed: int, tsc_index: int, width_edges: np.ndarray,
    exploration_alpha: float, l2_regularization: float, *,
    loader: Callable[..., Mapping[str, Any]] | None = None,
    split: str = "target", output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the five prices with source-selected hyperparameters and candidate bank."""
    timeline = build_timeline(zone, seed, split, tsc_index, width_edges, loader=loader)
    result = run_grid(timeline, ((float(exploration_alpha), float(l2_regularization)),), PRICE_RATIOS)
    result["stream_choices"] = timeline.unpack(result["choices"])
    result["evaluation_metrics"] = evaluate_decisions(timeline, result["choices"], PRICE_RATIOS)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "decisions.npz", **{
            f"{p}__H{h:02d}": array for (p, h), array in result["stream_choices"].items()})
        result["scores"].to_parquet(out / "replay_scores.parquet", index=False)
        result["evaluation_metrics"].to_parquet(out / "evaluation_metrics.parquet", index=False)
        (out / "audit.json").write_text(json.dumps({**result["audit"], "tsc_index": int(tsc_index)}, indent=2), encoding="utf-8")
    return result


def evaluate_decisions(timeline: Timeline, choices: np.ndarray,
                       ratios: Sequence[float]) -> pd.DataFrame:
    """Final metrics from each selected interval sequence, with 168-point windows.

    Candidate historical TUWR used during source-prefix hyperparameter selection is deliberately
    not used here. Seeds are retained separately for later weighted pooling.
    """
    if choices.shape != (len(ratios), timeline.size):
        raise ValueError("Expected one target decision path per price")
    rows = []
    for code, n in enumerate(timeline.stream_sizes):
        ids = np.flatnonzero(timeline.stream_code == code)
        ids = ids[np.argsort(timeline.stream_position[ids])]
        assert len(ids) == 11 * n
        predictor, horizon = PREDICTORS[code // 5], HORIZONS[code % 5]
        for price_index, ratio in enumerate(ratios):
            selected = choices[price_index, ids]
            cap = timeline.capacity[ids, selected].reshape(11, n)
            miss = timeline.exceedance[ids, selected].reshape(11, n)
            hit = timeline.covered[ids, selected].reshape(11, n)
            prefix = np.pad(np.cumsum(hit, axis=1, dtype=np.int64), ((0, 0), (1, 0)))
            rolling = (prefix[:, 168:] - prefix[:, :-168]) / 168.0
            gap = rolling - COVERAGES[:, None]
            tolerance = 1.96 * np.sqrt(COVERAGES * (1 - COVERAGES) / 168)
            for c, coverage in enumerate(COVERAGES):
                window_count = rolling.shape[1]
                under = int(np.count_nonzero(gap[c] < -tolerance[c]))
                over = int(np.count_nonzero(gap[c] > tolerance[c]))
                errf_sum = float(3.56 * (cap[c].sum() + ratio * miss[c].sum()))
                rows.append(dict(zone=timeline.zone, seed=timeline.seed, predictor=predictor,
                                 horizon=horizon, target_coverage=float(coverage), price_ratio=float(ratio),
                                 method="LinUCB", n=n, window_count=window_count,
                                 mean_errf=errf_sum / n, mean_capacity_cost=float(3.56 * cap[c].mean()),
                                 mean_exceedance_cost=float(3.56 * ratio * miss[c].mean()),
                                 coverage=float(hit[c].mean()), tuwr=under / window_count,
                                 towr=over / window_count, ard=float(np.abs(gap[c]).mean()),
                                 errf_sum=errf_sum, covered_sum=int(hit[c].sum()),
                                 tuwr_count=under, towr_count=over,
                                 absolute_deviation_sum=float(np.abs(gap[c]).sum())))
    return pd.DataFrame(rows)


def cached_width_edges(excluded_zones: Sequence[str], seed: int) -> np.ndarray:
    """Source-only width edges; endpoint matrices need not be decompressed."""
    from forward_common import RUN, ZONES, dataset_path
    from clara_forward import fit_width_edges
    excluded = tuple(sorted(set(excluded_zones)))
    sources = tuple(z for z in ZONES if z not in excluded)
    if not excluded or len(excluded) not in (1, 2) or len(sources) + len(excluded) != 10:
        raise ValueError("Width fitting requires one outer or two nested exclusions")
    out = RUN / "width_edges" / ("exclude_" + "_".join(excluded) + f"__seed{seed}.npz")
    files = [dataset_path(z, seed, p, h, "source") for z in sources for p in PREDICTORS for h in HORIZONS]
    identity = hashlib.sha256(json.dumps([(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]).encode()).hexdigest()
    if out.exists():
        with np.load(out, allow_pickle=False) as f:
            if str(f["source_identity"]) == identity:
                return f["width_edges"]
    def records():
        for z in sources:
            for p in PREDICTORS:
                for h in HORIZONS:
                    with np.load(dataset_path(z, seed, p, h, "source"), allow_pickle=False) as f:
                        keys = ("raw_width", "issue_ns", "label_ns", "available_ns", "source_cutoff_ns")
                        d = {k: f[k] for k in keys}
                        if "predictor_train_available_max_ns" in f:
                            d["predictor_train_available_max_ns"] = f["predictor_train_available_max_ns"]
                    yield z, p, h, d
    edges = fit_width_edges(records())
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.stem + f"__pid{os.getpid()}.npz")
    np.savez_compressed(temporary, width_edges=edges, source_identity=identity,
                        excluded_zones=np.asarray(excluded), source_zones=np.asarray(sources))
    os.replace(temporary, out)
    return edges


def tune_outer(outer_zone: str) -> dict[str, Any]:
    """Run nine pre-test source validation folds and retain the original selector."""
    from forward_common import RUN, ZONES, CONTRACTS, select_tsc, tsc_selection_details
    from baseline_selection import select_configuration
    if outer_zone not in ZONES:
        raise ValueError("Unknown outer held-out zone")
    root = RUN / "linucb" / outer_zone
    identity = hashlib.sha256((RUN / "DATA_READY.json").read_bytes()
                              + Path(__file__).read_bytes()
                              + (Path(__file__).resolve().parent / 'clara_forward.py').read_bytes()
                              + (Path(__file__).resolve().parent / 'forward_common.py').read_bytes()).hexdigest()
    frames = []
    for validation_zone in ZONES:
        if validation_zone == outer_zone:
            continue
        out = root / "inner" / validation_zone
        score_path = out / "validation_scores.parquet"
        identity_path = out / "experiment_identity.json"
        tsc_index = select_tsc((outer_zone, validation_zone))
        reusable = (score_path.exists() and identity_path.exists()
                    and json.loads(identity_path.read_text(encoding="utf-8")).get("identity") == identity)
        if reusable:
            scores = pd.read_parquet(score_path)
            if len(scores) != 12 or set(scores.tsc_index) != {tsc_index}:
                raise RuntimeError("Existing inner fold has incompatible TSC selection")
        else:
            edges = {s: cached_width_edges((outer_zone, validation_zone), s) for s in (0, 1, 2)}
            scores = run_inner_fold(outer_zone, validation_zone, tsc_index, edges, output_dir=out)
            (out / "tsc_selection.json").write_text(json.dumps(tsc_selection_details((outer_zone, validation_zone)), indent=2), encoding="utf-8")
            identity_path.write_text(json.dumps({"identity": identity,
                                                "outer_heldout_zone": outer_zone,
                                                "inner_validation_zone": validation_zone,
                                                "tsc_index": int(tsc_index)}, indent=2), encoding="utf-8")
        frames.append(scores)
        print(f"LinUCB {outer_zone}: inner {validation_zone} complete", flush=True)
    all_scores = pd.concat(frames, ignore_index=True)
    chosen = select_configuration(contracts=CONTRACTS, validation_scores=all_scores).as_record()
    row = all_scores[all_scores.configuration_id == chosen["selected_configuration_id"]].iloc[0]
    chosen.update(exploration_alpha=float(row.exploration_alpha),
                  l2_regularization=float(row.l2_regularization),
                  outer_heldout_zone=outer_zone, validation_zone_count=9,
                  candidate_count=6, horizon_count=5,
                  source_split="first 28 days of original final 20%, feedback strictly before cutoff", target_outcomes_used_for_tuning=False,
                  starts_from_ridge_prior=True, experiment_identity=identity)
    root.mkdir(parents=True, exist_ok=True)
    all_scores.to_parquet(root / "validation_scores.parquet", index=False)
    (root / "selected_configuration.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    return chosen


def _run_outer_process(arguments: tuple[str, str, int | None, int]) -> dict[str, Any]:
    mode, outer_zone, requested_seed, threads = arguments
    import numba
    numba.set_num_threads(threads)
    from forward_common import RUN, select_tsc
    started = perf_counter()
    if mode in ("tune", "outer"):
        selected = tune_outer(outer_zone)
        print(json.dumps(selected), flush=True)
    if mode in ("final", "outer"):
        path = RUN / "linucb" / outer_zone / "selected_configuration.json"
        selected = json.loads(path.read_text(encoding="utf-8"))
        for seed in ((requested_seed,) if requested_seed is not None else (0, 1, 2)):
            output = RUN / "linucb" / outer_zone / f"target_seed{seed}"
            result = replay_final(outer_zone, seed, select_tsc((outer_zone,)),
                                  cached_width_edges((outer_zone,), seed),
                                  selected["exploration_alpha"], selected["l2_regularization"],
                                  output_dir=output)
            print(f"LinUCB {outer_zone}: final seed {seed} complete ({result['audit']['runtime_seconds']:.2f} s)", flush=True)
    return {"outer_zone": outer_zone, "mode": mode, "status": "complete",
            "seconds": perf_counter() - started, "configuration": selected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("tune", "final", "outer"), default="outer")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--outer")
    scope.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    from forward_common import RUN, ZONES
    if not (RUN / "DATA_READY.json").exists():
        raise RuntimeError("DATA_READY.json is required before full LinUCB tuning or replay")
    if args.workers < 1 or args.threads < 1:
        raise ValueError("Worker and thread counts must be positive")
    if args.all:
        completed = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_outer_process, (args.mode, z, args.seed, args.threads)): z for z in ZONES}
            for future in as_completed(futures):
                row = future.result()
                completed.append(row)
                print(json.dumps({"completed_outer_count": len(completed), **row}), flush=True)
                output = RUN / "linucb" / "batch_progress.json"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(completed, indent=2), encoding="utf-8")
        (RUN / "linucb" / "BATCH_COMPLETE.json").write_text(json.dumps({
            "outer_count": len(completed), "mode": args.mode,
            "target_seed_count": 0 if args.mode == "tune" else (10 if args.seed is not None else 30),
            "price_count": 5, "workers": args.workers, "numba_threads_per_worker": args.threads,
            "completed": completed,
        }, indent=2), encoding="utf-8")
    else:
        print(json.dumps(_run_outer_process((args.mode, args.outer, args.seed, args.threads))), flush=True)


if __name__ == "__main__":
    main()
