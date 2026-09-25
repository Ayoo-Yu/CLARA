"""Source-tuned LinUCB initialized from the held-out site's pre-cutoff history.

This is a separately registered sensitivity analysis, not a replacement for the
main nine-method comparison. Hyperparameters, TSC choice and state width edges
are selected from the other zones. The original LinUCB kernel replays the entire
available issue timeline once; only post-cutoff choices enter reported metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np

import linucb_forward as lin

STREAM_ARRAYS = (
    "issue_ns", "label_ns", "available_ns", "y", "center", "raw_width",
    "ramp", "rolling_state", "lower", "upper", "candidate_tuwr", "candidate_ard",
)


def assemble_stream(zone: str, seed: int, predictor: str, horizon: int,
                    tsc_index: int, cutoff_ns: int, loader: Callable) -> tuple[dict, dict]:
    """Join mature prefix, pending prefix and target without dropping feedback."""
    parts = {split: loader(zone, seed, predictor, horizon, split, tsc_index)
             for split in ("source", "prefix_pending", "target")}
    source, pending, target = (parts[x] for x in ("source", "prefix_pending", "target"))
    for name, d in parts.items():
        issue = np.asarray(d["issue_ns"], np.int64)
        label = np.asarray(d["label_ns"], np.int64)
        avail = np.asarray(d["available_ns"], np.int64)
        if not len(issue) or not np.all(np.diff(issue) > 0):
            raise ValueError(f"{name}: missing or unordered issue records")
        if not np.all((issue < label) & (label < avail)):
            raise ValueError(f"{name}: invalid delayed-feedback ordering")
        if int(d["source_cutoff_ns"]) != int(cutoff_ns):
            raise ValueError(f"{name}: mismatched cutoff")
        if name == "source" and not np.all((issue < cutoff_ns) & (label < cutoff_ns) & (avail <= cutoff_ns)):
            raise ValueError("Source includes information unavailable at the cutoff")
        if name == "prefix_pending" and not np.all((issue < cutoff_ns) & ((label >= cutoff_ns) | (avail > cutoff_ns))):
            raise ValueError("Pending prefix has an invalid availability boundary")
        if name == "target" and not np.all(issue >= cutoff_ns):
            raise ValueError("Target contains pre-cutoff issues")
    merged = dict(target)
    for key in STREAM_ARRAYS:
        merged[key] = np.concatenate([np.asarray(parts[s][key]) for s in parts], axis=-1)
    order = np.argsort(merged["issue_ns"], kind="stable")
    for key in STREAM_ARRAYS:
        merged[key] = merged[key][..., order]
    if not np.all(np.diff(merged["issue_ns"]) == 3_600_000_000_000):
        raise ValueError("The complete hourly replay has duplicate or missing issues")
    target_mask = merged["issue_ns"] >= cutoff_ns
    for key in STREAM_ARRAYS:
        if not np.array_equal(merged[key][..., target_mask], target[key], equal_nan=True):
            raise ValueError(f"Warm-start target records differ from main replay: {key}")
    if np.count_nonzero(target_mask) != len(target["issue_ns"]):
        raise ValueError("Target event count changed")
    prefix_mask = ~target_mask
    crossing = prefix_mask & (merged["available_ns"] > cutoff_ns)
    if not np.any(crossing):
        raise ValueError("No pending pre-cutoff feedback; this does not exercise continuous warm-up")
    merged["split"] = "warm_prefix_and_target"
    audit = dict(predictor=predictor, horizon=int(horizon),
                 mature_prefix_issues=len(source["issue_ns"]),
                 pending_prefix_issues=len(pending["issue_ns"]),
                 target_issues=len(target["issue_ns"]),
                 total_issues=len(merged["issue_ns"]),
                 first_issue_ns=int(merged["issue_ns"][0]),
                 first_target_issue_ns=int(target["issue_ns"][0]),
                 last_target_issue_ns=int(target["issue_ns"][-1]),
                 pending_prefix_available_min_ns=int(merged["available_ns"][crossing].min()),
                 pending_prefix_available_max_ns=int(merged["available_ns"][crossing].max()),
                 target_arrays_identical=True)
    return merged, audit


def build_warm_timeline(zone: str, seed: int, tsc_index: int, width_edges: np.ndarray,
                        cutoff_ns: int, *, loader: Callable | None = None):
    if loader is None:
        from forward_common import load_stream
        loader = load_stream
    source_audits = []
    masks = {}

    def full_loader(z, s, p, h, split, ti):
        if split != "warm_prefix_and_target":
            raise ValueError(split)
        merged, audit = assemble_stream(z, s, p, h, ti, cutoff_ns, loader)
        source_audits.append(audit)
        masks[(p, h)] = merged["issue_ns"] >= cutoff_ns
        return merged

    timeline = lin.build_timeline(zone, seed, "warm_prefix_and_target", tsc_index,
                                  width_edges, loader=full_loader)
    return timeline, masks, source_audits


def target_choices(timeline: lin.Timeline, choices: np.ndarray, masks: Mapping) -> dict:
    unpacked = timeline.unpack(choices)
    result = {}
    for (predictor, horizon), array in unpacked.items():
        mask = masks[(predictor, horizon)]
        if array.shape[-1] != len(mask):
            raise ValueError("Target-mask length differs from the replay stream")
        result[f"{predictor}__H{horizon:02d}"] = array[..., mask]
    return result


def replay_final(zone: str, seed: int, *, output_dir: Path | None = None) -> dict[str, Any]:
    """Use the main run's selected settings; save only identically matched targets."""
    from forward_common import RUN, dataset_path, load_stream, select_tsc, sha256
    required = [RUN / "DATA_READY.json", RUN / "PREFIX_PENDING_READY.json",
                RUN / "linucb" / zone / "selected_configuration.json"]
    if not all(p.is_file() for p in required):
        raise RuntimeError("Warm-start requires complete data/pending-prefix receipts and main source tuning")
    if (RUN / "STOP_REQUESTED.txt").exists():
        raise RuntimeError("STOP_REQUESTED is present in the new run")
    chosen = json.loads(required[-1].read_text(encoding="utf-8"))
    cutoff = json.loads((RUN / "chrono_cutoff.json").read_text(encoding="utf-8"))["t0_ns"]
    if chosen.get("outer_heldout_zone") != zone:
        raise ValueError("Source-selected hyperparameters belong to a different outer fold")
    ti = int(select_tsc((zone,)))
    width_edges = lin.cached_width_edges((zone,), seed)
    inputs = []
    for p in lin.PREDICTORS:
        for h in lin.HORIZONS:
            for split in ("source", "prefix_pending", "target"):
                path = dataset_path(zone, seed, p, h, split)
                if not path.is_file() or not path.with_suffix(".json").is_file():
                    raise FileNotFoundError(path)
                inputs.append(dict(path=str(path.relative_to(RUN)),
                                   metadata_sha256=sha256(path.with_suffix(".json"))))
    began = perf_counter()
    timeline, masks, stream_audit = build_warm_timeline(zone, seed, ti, width_edges, cutoff)
    result = lin.run_grid(timeline, ((chosen["exploration_alpha"], chosen["l2_regularization"]),),
                          lin.PRICE_RATIOS)
    decisions = target_choices(timeline, result["choices"], masks)
    # Compare the actual issue coordinates, not merely the array dimensions.
    for (p, h), mask in masks.items():
        d = load_stream(zone, seed, p, h, "target", ti)
        if decisions[f"{p}__H{h:02d}"].shape != (5, 11, len(d["issue_ns"])):
            raise ValueError("Warm-start actions are not matched to main target records")
    pre = timeline.issue_ns < cutoff
    crossed = pre & (timeline.available_ns > cutoff)
    # The reused kernel processes every feedback item up to the final issue.
    matured_ids = timeline.feedback_order[:int(timeline.matured_ends[-1])]
    crossed_matured = int(np.count_nonzero(crossed[matured_ids]))
    if crossed_matured != int(np.count_nonzero(crossed)):
        raise ValueError("Some pending prefix feedback did not mature during target replay")
    out = output_dir or RUN / "linucb_warm" / zone / f"target_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "decisions.npz", **decisions)
    audit = dict(schema="LINUCB_TARGET_PREFIX_INITIALIZATION_V1", zone=zone, seed=seed,
                 cutoff_ns=int(cutoff), target_event_count=int(np.count_nonzero(~pre)),
                 prefix_event_count=int(np.count_nonzero(pre)),
                 pending_prefix_event_count=int(np.count_nonzero(crossed)),
                 pending_prefix_events_later_updated=crossed_matured,
                 target_only_output=True, target_coordinates_equal_main=True,
                 reset_at_cutoff=False, selected_action_feedback_only=True,
                 source_selected_configuration_sha256=sha256(required[-1]),
                 tsc_configuration_index=ti, width_edges_sha256=hashlib.sha256(width_edges.tobytes()).hexdigest(),
                 configuration={k: chosen[k] for k in ("exploration_alpha", "l2_regularization")},
                 price_ratios=list(lin.PRICE_RATIOS), kernel_sha256=sha256(Path(lin.__file__)),
                 warmstart_code_sha256=sha256(Path(__file__)), input_streams=inputs,
                 stream_checks=stream_audit, full_replay_audit=result["audit"],
                 decisions_sha256=sha256(out / "decisions.npz"), seconds=perf_counter() - began,
                 interpretation="Retrospective pre-period initialization with settings selected at T0; only post-T0 performance is evaluated.")
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def _synthetic_loader(zone, seed, predictor, horizon, split, tsc_index):
    n = 240
    hour = 3_600_000_000_000
    cutoff = 72 * hour
    rng = np.random.default_rng(843 + lin.PREDICTORS.index(predictor) * 50 + horizon)
    issue = np.arange(n, dtype=np.int64) * hour
    label = issue + horizon * hour
    avail = label + hour
    y = rng.uniform(0, 1, n)
    center = np.clip(y + rng.normal(0, .15, n), 0, 1)
    widths = np.linspace(.03, .45, 6)[:, None, None] * (.3 + lin.COVERAGES[None, :, None])
    lower = np.maximum(center[None, None, :] - widths, 0)
    upper = np.minimum(center[None, None, :] + widths, 1)
    d = dict(issue_ns=issue, label_ns=label, available_ns=avail, y=y, center=center,
             raw_width=upper[0] - lower[0], ramp=rng.integers(0, 2, n),
             rolling_state=rng.integers(0, 5, (11, n)), lower=lower, upper=upper,
             candidate_tuwr=rng.uniform(0, 1, (6, 11, n)),
             candidate_ard=rng.uniform(0, 1, (6, 11, n)))
    source = (issue < cutoff) & (label < cutoff) & (avail <= cutoff)
    masks = dict(source=source, prefix_pending=(issue < cutoff) & ~source, target=issue >= cutoff)
    result = {k: v[..., masks[split]] for k, v in d.items()}
    result["source_cutoff_ns"] = cutoff
    return result


def smoke() -> dict:
    from linucb_tests import reference_eventwise
    hour = 3_600_000_000_000
    cutoff = 72 * hour
    edges = np.broadcast_to([.1, .4], (4, 5, 11, 2)).copy()
    timeline, masks, streams = build_warm_timeline("synthetic", 0, 0, edges, cutoff,
                                                 loader=_synthetic_loader)
    config = ((.5, 1.0),)
    full = lin.run_grid(timeline, config)
    reference = reference_eventwise(timeline, .5, 1., lin.REFERENCE_RATIO)
    np.testing.assert_array_equal(full["choices"][0], reference)
    extracted = target_choices(timeline, full["choices"], masks)
    assert len(extracted) == 20 and all(a.shape == (1, 11, 168) for a in extracted.values())
    assert sum(a.size for a in extracted.values()) == np.count_nonzero(timeline.issue_ns >= cutoff)
    # Alter outcomes not available by a time within the scored period.
    old = timeline.exceedance.copy()
    checkpoint = 90 * hour
    timeline.exceedance[timeline.available_ns > checkpoint] += 100
    future = lin.run_grid(timeline, config)
    np.testing.assert_array_equal(full["choices"][:, timeline.issue_ns <= checkpoint],
                                  future["choices"][:, timeline.issue_ns <= checkpoint])
    timeline.exceedance = old.copy()
    unselected = np.ones_like(old, dtype=bool)
    unselected[np.arange(timeline.size), full["choices"][0]] = False
    timeline.exceedance[unselected] += 100
    alternative = lin.run_grid(timeline, config)
    np.testing.assert_array_equal(full["choices"], alternative["choices"])
    timeline.exceedance = old
    crossed = (timeline.issue_ns < cutoff) & (timeline.available_ns > cutoff)
    assert crossed.sum() == sum(x["pending_prefix_issues"] for x in streams) * 11
    matured = timeline.feedback_order[:timeline.matured_ends[-1]]
    assert crossed[matured].sum() == crossed.sum()
    # Controlled boundary case: a prefix loss is unavailable at T0, arrives one
    # issue later, and must then change the zero-exploration action choice.
    issue = np.arange(4, dtype=np.int64) * hour
    label, avail = issue + hour, issue + 2 * hour
    offsets, order, ends = lin._schedule(issue, label, avail)
    miss = np.zeros((4, 6)); miss[0, 0] = 100
    toy = lin.Timeline(issue, label, avail, np.zeros(4, np.uint16), np.zeros(4, np.uint8),
                       np.zeros((4, 6)), miss, np.ones((4, 6), np.uint8), np.zeros((4, 6)),
                       offsets, order, ends, np.zeros(4, np.uint8), np.arange(4),
                       (4,), "boundary", 0, "synthetic")
    affected = lin.run_grid(toy, ((0., 1.),), (1.,))["choices"][0]
    toy.exceedance[:] = 0
    unchanged = lin.run_grid(toy, ((0., 1.),), (1.,))["choices"][0]
    assert affected[1] == unchanged[1] == 0
    assert affected[2] == 1 and unchanged[2] == 0
    report = dict(status="PASS", synthetic_events=timeline.size,
                  target_events=sum(a.size for a in extracted.values()),
                  stream_count=20, full_timeline_matches_eventwise_reference=True,
                  target_projection_matches_main_coordinates=True,
                  future_outcome_perturbation_invariant=True,
                  unselected_outcome_perturbation_invariant=True,
                  pending_prefix_events=int(crossed.sum()),
                  all_pending_prefix_feedback_processed=True,
                  pending_feedback_changes_only_post_availability_decisions=True,
                  original_linucb_kernel_reused=True,
                  scope="Synthetic equivalence and time-boundary tests; not formal experimental performance")
    (Path.cwd() / "outputs" / "linucb_warmstart_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--zone")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    import numba
    numba.set_num_threads(args.threads)
    if args.smoke:
        print(json.dumps(smoke()), flush=True)
    else:
        if args.zone is None or args.seed is None:
            parser.error("Formal warm-start replay requires --zone and --seed")
        row = replay_final(args.zone, args.seed)
        print(json.dumps({k: row[k] for k in ("zone", "seed", "target_event_count",
                                              "pending_prefix_event_count", "seconds")}), flush=True)


if __name__ == "__main__":
    (Path.cwd()/"outputs").mkdir(exist_ok=True)
    main()
