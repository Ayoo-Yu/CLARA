"""Independent compact-statistic QA for the formal GEFCom six-action run.

This module intentionally does not import the production selector runner or call
its aggregation helpers.  It reconstructs the frozen estimands from deep-loaded
replay children and compares them with the sealed aggregate outputs.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA = "TEST_CLARA_GEFCOM_6A_INDEPENDENT_QA_MODULE_V1"
ZONES = tuple(f"zone{index}" for index in range(1, 11))
SEEDS = (0, 1, 2)
HORIZONS = (1, 3, 6, 12, 24)
PREDICTORS = ("Ridge", "GBR", "MLP", "QRLSTM")
COVERAGES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
ACTIONS = (
    "Static",
    "ACI",
    "AgACI",
    "EnbPI_RH",
    "TunedSingleConformal",
    "EqualEndpointEnsemble",
)
METHODS = (
    "CLARA_6A",
    "CART_6A",
    "LinUCB_6A",
    "TunedSingleConformal",
    "EqualEndpointEnsemble",
    "FixedStatic",
    "FixedACI",
    "FixedAgACI",
    "FixedEnbPI_RH",
)
PRICE_IDS = ("R01", "R02", "R05P6179775281", "R10", "R20")
POLICY_MODES = (
    "RESELECT_EACH_RATIO",
    "FIXED_MAIN_RATIO_R05P6179775281",
)
MAIN_PRICE_ID = "R05P6179775281"
RESULT_SCOPE_FIELDS = (
    "policy_mode",
    "evaluation_price_id",
    "policy_price_id",
)
CELL_FIELDS = (
    "zone_or_farm",
    "predictor",
    "horizon_steps",
    "seed",
    "target_coverage",
    "ramp_state",
)
DIAGNOSTIC_FIELDS = (
    "zone_or_farm",
    "predictor",
    "horizon_steps",
    "seed",
    "target_coverage",
    "regime",
)
DIAGNOSTIC_ADDITIVE = (
    "event_count",
    "errf_sum",
    "reserve_up_sum",
    "reserve_down_sum",
    "miss_upper_sum",
    "miss_lower_sum",
    "covered_sum",
    "coverage_target_sum",
    "coverage_gap_sum",
    "width_sum",
    "interval_score_sum",
    "rolling_window_count",
    "towr_exceedance_count",
    "tuwr_exceedance_count",
    "rolling_absolute_deviation_sum",
    "support_applicable_count",
    "support_backoff_count",
    "guardrail_applicable_count",
    "guardrail_any_exclusion_count",
    "guardrail_excluded_action_count",
    "guardrail_empty_count",
    "selected_action_guardrail_fail_count",
)
EVENT_SUM_COLUMNS = (
    "event_count",
    "errf_sum",
    "reserve_up_sum",
    "reserve_down_sum",
    "miss_upper_sum",
    "miss_lower_sum",
    "covered_sum",
    "coverage_target_sum",
    "coverage_gap_sum",
    "width_sum",
    "interval_score_sum",
)


def module_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _is_sha(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _ratio(top: Sequence[Any], bottom: Sequence[Any]) -> np.ndarray:
    numerator = np.asarray(top, dtype=float)
    denominator = np.asarray(bottom, dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(numerator), np.nan, dtype=float),
        where=denominator > 0.0,
    )


def _means(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    count = result["event_count"].to_numpy(dtype=float)
    if np.any(count <= 0.0):
        raise RuntimeError("independent QA found non-positive event_count")
    mapping = {
        "mean_errf": "errf_sum",
        "mean_reserve_up": "reserve_up_sum",
        "mean_reserve_down": "reserve_down_sum",
        "mean_miss_upper": "miss_upper_sum",
        "mean_miss_lower": "miss_lower_sum",
        "empirical_coverage": "covered_sum",
        "mean_coverage_target": "coverage_target_sum",
        "coverage_gap": "coverage_gap_sum",
        "mean_width": "width_sum",
        "mean_interval_score": "interval_score_sum",
    }
    if not set(mapping.values()).issubset(result):
        raise RuntimeError("independent QA missing additive event metrics")
    for target, source in mapping.items():
        result[target] = result[source].to_numpy(dtype=float) / count
    return result


def _diagnostic_rates(frame: pd.DataFrame) -> pd.DataFrame:
    result = _means(frame)
    result["TOWR"] = _ratio(result["towr_exceedance_count"], result["rolling_window_count"])
    result["TUWR"] = _ratio(result["tuwr_exceedance_count"], result["rolling_window_count"])
    result["ARD"] = _ratio(result["rolling_absolute_deviation_sum"], result["rolling_window_count"])
    result["support_backoff_rate"] = _ratio(result["support_backoff_count"], result["support_applicable_count"])
    result["guardrail_any_exclusion_rate"] = _ratio(result["guardrail_any_exclusion_count"], result["guardrail_applicable_count"])
    result["guardrail_action_exclusion_rate"] = _ratio(
        result["guardrail_excluded_action_count"],
        result["guardrail_applicable_count"].to_numpy(dtype=float) * len(ACTIONS),
    )
    result["guardrail_empty_rate"] = _ratio(result["guardrail_empty_count"], result["guardrail_applicable_count"])
    result["selected_action_guardrail_fail_rate"] = _ratio(
        result["selected_action_guardrail_fail_count"], result["guardrail_applicable_count"]
    )
    return result


def _expected_policy_price(policy_mode: str, evaluation_price_id: str) -> str:
    if policy_mode == POLICY_MODES[0]:
        return str(evaluation_price_id)
    if policy_mode == POLICY_MODES[1]:
        return MAIN_PRICE_ID
    raise RuntimeError(f"independent QA unknown policy mode: {policy_mode}")


def _assert_exact_multiset(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    drop_columns: Sequence[str],
    label: str,
) -> None:
    missing = sorted(
        set(drop_columns) - (set(left.columns) & set(right.columns))
    )
    if missing:
        raise RuntimeError(f"independent QA {label} missing identity columns: {missing}")
    left_value = left.drop(columns=list(drop_columns))
    right_value = right.drop(columns=list(drop_columns))
    if list(left_value.columns) != list(right_value.columns):
        raise RuntimeError(f"independent QA {label} columns differ")
    order = list(left_value.columns)
    try:
        pd.testing.assert_frame_equal(
            left_value.sort_values(order, kind="mergesort").reset_index(drop=True),
            right_value.sort_values(order, kind="mergesort").reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError as error:
        raise RuntimeError(f"independent QA {label} exact algebra failed: {error}") from error


def _cross_mode_algebra(raw: Mapping[str, pd.DataFrame]) -> None:
    deterministic = {
        "TunedSingleConformal",
        "EqualEndpointEnsemble",
        "FixedStatic",
        "FixedACI",
        "FixedAgACI",
        "FixedEnbPI_RH",
    }
    compact_names = (
        "cell_metrics",
        "action_counts",
        "paired_blocks",
        "diagnostic_metrics",
        "clara_state_counts",
        "event_conservation",
    )
    for name in compact_names:
        frame = raw[name]
        if not set(RESULT_SCOPE_FIELDS).issubset(frame.columns):
            raise RuntimeError(f"independent QA {name} lacks dual-mode axes")
        main = frame[
            frame["evaluation_price_id"].astype(str).eq(MAIN_PRICE_ID)
        ]
        reselected = main[
            main["policy_mode"].astype(str).eq(POLICY_MODES[0])
        ]
        fixed = main[main["policy_mode"].astype(str).eq(POLICY_MODES[1])]
        _assert_exact_multiset(
            reselected,
            fixed,
            drop_columns=("policy_mode",),
            label=f"R05 cross-mode {name}",
        )
        if "method" in frame:
            for evaluation_price_id in PRICE_IDS:
                current = frame[
                    frame["evaluation_price_id"].astype(str).eq(
                        evaluation_price_id
                    )
                    & frame["method"].astype(str).isin(deterministic)
                ]
                reselected = current[
                    current["policy_mode"].astype(str).eq(POLICY_MODES[0])
                ]
                fixed = current[
                    current["policy_mode"].astype(str).eq(POLICY_MODES[1])
                ]
                _assert_exact_multiset(
                    reselected,
                    fixed,
                    drop_columns=("policy_mode", "policy_price_id"),
                    label=f"deterministic cross-mode {name} {evaluation_price_id}",
                )
    fixed_actions = raw["action_counts"][
        raw["action_counts"]["policy_mode"].astype(str).eq(POLICY_MODES[1])
    ]
    reference = fixed_actions[
        fixed_actions["evaluation_price_id"].astype(str).eq(MAIN_PRICE_ID)
    ]
    for evaluation_price_id in PRICE_IDS:
        current = fixed_actions[
            fixed_actions["evaluation_price_id"].astype(str).eq(
                evaluation_price_id
            )
        ]
        _assert_exact_multiset(
            reference,
            current,
            drop_columns=RESULT_SCOPE_FIELDS,
            label=f"fixed-scan action counts {evaluation_price_id}",
        )


def _cross_mode_parent_children(children: Sequence[Any]) -> None:
    """Independently validate one exact ten-child replay parent in-place."""

    if len(children) != 10:
        raise RuntimeError("independent QA parent batch must contain exactly 10 children")
    compact_names = (
        "cell_metrics",
        "action_counts",
        "paired_blocks",
        "diagnostic_metrics",
        "clara_state_counts",
        "event_conservation",
    )
    deterministic = {
        "TunedSingleConformal",
        "EqualEndpointEnsemble",
        "FixedStatic",
        "FixedACI",
        "FixedAgACI",
        "FixedEnbPI_RH",
    }
    by_key: dict[tuple[str, str], Any] = {}
    parent_axes: set[tuple[str, int]] = set()
    for child in children:
        manifest = dict(child.manifest)
        zone = str(manifest.get("evaluation_zone", ""))
        seed = int(manifest.get("seed", -1))
        mode = str(manifest.get("policy_mode", ""))
        evaluation_price_id = str(manifest.get("evaluation_price_id", ""))
        policy_price_id = str(manifest.get("policy_price_id", ""))
        expected_policy_price_id = _expected_policy_price(
            mode, evaluation_price_id
        )
        key = (mode, evaluation_price_id)
        if (
            zone not in ZONES
            or seed not in SEEDS
            or mode not in POLICY_MODES
            or evaluation_price_id not in PRICE_IDS
            or policy_price_id != expected_policy_price_id
            or key in by_key
        ):
            raise RuntimeError(
                "independent QA replay parent child axes/crosswire failure"
            )
        parent_axes.add((zone, seed))
        for name in compact_names:
            frame = getattr(child, name)
            if not set(RESULT_SCOPE_FIELDS).issubset(frame.columns):
                raise RuntimeError(
                    f"independent QA replay child {name} lacks dual-mode scope"
                )
            observed_scope = set(
                zip(
                    frame["policy_mode"].astype(str),
                    frame["evaluation_price_id"].astype(str),
                    frame["policy_price_id"].astype(str),
                )
            )
            if observed_scope != {
                (mode, evaluation_price_id, expected_policy_price_id)
            }:
                raise RuntimeError(
                    f"independent QA replay child {name} internal scope crosswire"
                )
        by_key[key] = child
    expected_keys = {
        (mode, evaluation_price_id)
        for mode in POLICY_MODES
        for evaluation_price_id in PRICE_IDS
    }
    if len(parent_axes) != 1 or set(by_key) != expected_keys:
        raise RuntimeError(
            "independent QA parent batch is not one exact 2-mode x 5-price parent"
        )
    r05_reselected = by_key[(POLICY_MODES[0], MAIN_PRICE_ID)]
    r05_fixed = by_key[(POLICY_MODES[1], MAIN_PRICE_ID)]
    for name in compact_names:
        _assert_exact_multiset(
            getattr(r05_reselected, name),
            getattr(r05_fixed, name),
            drop_columns=("policy_mode",),
            label=f"streamed parent R05 cross-mode {name}",
        )
    for evaluation_price_id in PRICE_IDS:
        reselected = by_key[(POLICY_MODES[0], evaluation_price_id)]
        fixed = by_key[(POLICY_MODES[1], evaluation_price_id)]
        for name in compact_names:
            left = getattr(reselected, name)
            right = getattr(fixed, name)
            if "method" in left.columns:
                left = left[left["method"].astype(str).isin(deterministic)]
                right = right[right["method"].astype(str).isin(deterministic)]
            _assert_exact_multiset(
                left,
                right,
                drop_columns=("policy_mode", "policy_price_id"),
                label=(
                    "streamed parent deterministic cross-mode "
                    f"{evaluation_price_id}/{name}"
                ),
            )
    reference_actions = r05_fixed.action_counts
    for evaluation_price_id in PRICE_IDS:
        _assert_exact_multiset(
            reference_actions,
            by_key[(POLICY_MODES[1], evaluation_price_id)].action_counts,
            drop_columns=RESULT_SCOPE_FIELDS,
            label=f"streamed parent fixed-scan actions {evaluation_price_id}",
        )


def _entropy(actions: pd.DataFrame, *, include_zone: bool) -> pd.DataFrame:
    base = [*RESULT_SCOPE_FIELDS, "method"] + (
        ["zone_or_farm"] if include_zone else []
    )
    overall = (
        actions.groupby([*base, "selected_action"], sort=True, as_index=False)["action_count"]
        .sum()
        .assign(regime="overall")
    )
    strata = (
        actions.groupby([*base, "ramp_state", "selected_action"], sort=True, as_index=False)["action_count"]
        .sum()
        .rename(columns={"ramp_state": "regime"})
    )
    frame = pd.concat([overall, strata], ignore_index=True, sort=False)
    keys = [*base, "regime"]
    totals = frame.groupby(keys, sort=True)["action_count"].transform("sum")
    probability = frame["action_count"].to_numpy(dtype=float) / totals.to_numpy(dtype=float)
    frame["term"] = np.where(probability > 0.0, -probability * np.log(probability), 0.0)
    result = (
        frame.groupby(keys, sort=True, as_index=False)
        .agg(action_count=("action_count", "sum"), action_entropy_nats=("term", "sum"))
        .reset_index(drop=True)
    )
    result["action_entropy_normalized"] = result["action_entropy_nats"] / np.log(len(ACTIONS))
    return result


def _collapse(children: Iterable[Any]) -> dict[str, pd.DataFrame]:
    compact_names = (
        "cell_metrics",
        "action_counts",
        "diagnostic_metrics",
        "clara_state_counts",
        "event_conservation",
    )
    parts = {name: [] for name in (*compact_names, "paired_blocks")}
    index_rows: list[dict[str, Any]] = []
    parent_batch: list[Any] = []
    child_count = 0
    for child in children:
        child_count += 1
        parent_batch.append(child)
        manifest = dict(child.manifest)
        for name in compact_names:
            parts[name].append(getattr(child, name).copy())
        paired_zone = (
            child.paired_blocks.groupby(
                [*RESULT_SCOPE_FIELDS, "zone_or_farm", "method"],
                sort=True,
                as_index=False,
            )[["event_count", "errf_sum"]]
            .sum()
            .reset_index(drop=True)
        )
        parts["paired_blocks"].append(paired_zone)
        manifest_sha = str(manifest.get("_manifest_sha256", ""))
        manifest_path = str(manifest.get("_manifest_relative_path", ""))
        if not _is_sha(manifest_sha) or not manifest_path:
            raise RuntimeError("independent QA child lineage missing")
        index_rows.append(
            {
                "evaluation_zone": str(manifest["evaluation_zone"]),
                "seed": int(manifest["seed"]),
                "policy_mode": str(manifest["policy_mode"]),
                "evaluation_price_id": str(manifest["evaluation_price_id"]),
                "policy_price_id": str(manifest["policy_price_id"]),
                "manifest_relative_path": manifest_path,
                "manifest_sha256": manifest_sha,
            }
        )
        if len(parent_batch) == 10:
            _cross_mode_parent_children(parent_batch)
            parent_batch.clear()
            del child
        elif len(parent_batch) > 10:
            raise RuntimeError("independent QA child stream lost parent boundary")
    if parent_batch:
        raise RuntimeError("independent QA child stream ended inside a parent")
    if child_count != 300 or any(len(values) != 300 for values in parts.values()):
        raise RuntimeError("independent QA requires exactly 300 replay children")
    raw = {name: pd.concat(values, ignore_index=True, sort=False) for name, values in parts.items()}
    cell_numeric = [
        column
        for column in raw["cell_metrics"].columns
        if column not in {*RESULT_SCOPE_FIELDS, "method", *CELL_FIELDS}
    ]
    raw["cell_metrics"] = raw["cell_metrics"].groupby(
        [*RESULT_SCOPE_FIELDS, "method", *CELL_FIELDS], sort=True, as_index=False
    )[cell_numeric].sum().reset_index(drop=True)
    raw["action_counts"] = raw["action_counts"].groupby(
        [*RESULT_SCOPE_FIELDS, "method", *CELL_FIELDS, "selected_action"],
        sort=True,
        as_index=False,
    )["action_count"].sum().reset_index(drop=True)
    diagnostic_keys = [*RESULT_SCOPE_FIELDS, "method", *DIAGNOSTIC_FIELDS]
    if raw["diagnostic_metrics"].duplicated(diagnostic_keys).any():
        raise RuntimeError("independent QA diagnostic key duplicate")
    raw["diagnostic_metrics"] = raw["diagnostic_metrics"].sort_values(diagnostic_keys, kind="mergesort").reset_index(drop=True)
    state_keys = [column for column in raw["clara_state_counts"].columns if column != "event_count"]
    raw["clara_state_counts"] = raw["clara_state_counts"].groupby(
        state_keys, sort=True, dropna=False, as_index=False
    )["event_count"].sum().reset_index(drop=True)
    conservation = raw["event_conservation"]
    raw["event_conservation"] = conservation.groupby(
        [*RESULT_SCOPE_FIELDS, "method"], sort=True, as_index=False
    ).agg(
        event_count=("event_count", "sum"),
        hash_sum_u64=("hash_sum_u64", lambda values: int(sum(int(value) for value in values) % (1 << 64))),
        hash_xor_u64=("hash_xor_u64", lambda values: int(np.bitwise_xor.reduce(np.asarray(values, dtype=np.uint64)))),
        hash_square_sum_u64=("hash_square_sum_u64", lambda values: int(sum(int(value) for value in values) % (1 << 64))),
    ).reset_index(drop=True)
    raw["paired_blocks"] = raw["paired_blocks"].groupby(
        [*RESULT_SCOPE_FIELDS, "zone_or_farm", "method"],
        sort=True,
        as_index=False,
    )[["event_count", "errf_sum"]].sum().reset_index(drop=True)
    raw["compact_artifact_index"] = pd.DataFrame(index_rows).sort_values(
        ["evaluation_zone", "seed", "policy_mode", "evaluation_price_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    expected_index = {
        (zone, seed, mode, price_id, _expected_policy_price(mode, price_id))
        for zone in ZONES
        for seed in SEEDS
        for mode in POLICY_MODES
        for price_id in PRICE_IDS
    }
    observed_index = set(
        zip(
            raw["compact_artifact_index"]["evaluation_zone"].astype(str),
            raw["compact_artifact_index"]["seed"].astype(int),
            raw["compact_artifact_index"]["policy_mode"].astype(str),
            raw["compact_artifact_index"]["evaluation_price_id"].astype(str),
            raw["compact_artifact_index"]["policy_price_id"].astype(str),
        )
    )
    if observed_index != expected_index:
        raise RuntimeError("independent QA replay child dual-mode axes incomplete")
    for name, frame in raw.items():
        if name == "compact_artifact_index":
            continue
        observed_scope = set(
            zip(
                frame["policy_mode"].astype(str),
                frame["evaluation_price_id"].astype(str),
                frame["policy_price_id"].astype(str),
            )
        )
        expected_scope = {
            (mode, price_id, _expected_policy_price(mode, price_id))
            for mode in POLICY_MODES
            for price_id in PRICE_IDS
        }
        if observed_scope != expected_scope:
            raise RuntimeError(f"independent QA {name} dual-mode scope incomplete")
    return raw


def _overall(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    zone_raw = cells.groupby(
        [*RESULT_SCOPE_FIELDS, "method", "zone_or_farm"],
        sort=True,
        as_index=False,
    )[list(EVENT_SUM_COLUMNS)].sum().reset_index(drop=True)
    zone = _means(zone_raw)
    mean_columns = [
        "mean_errf",
        "mean_reserve_up",
        "mean_reserve_down",
        "mean_miss_upper",
        "mean_miss_lower",
        "empirical_coverage",
        "mean_coverage_target",
        "coverage_gap",
        "mean_width",
        "mean_interval_score",
    ]
    balanced = zone.groupby(
        [*RESULT_SCOPE_FIELDS, "method"], sort=True, as_index=False
    )[mean_columns].mean()
    counts = zone.groupby([*RESULT_SCOPE_FIELDS, "method"], sort=True)[
        "zone_or_farm"
    ].nunique()
    if len(counts) != 90 or not counts.eq(10).all():
        raise RuntimeError("independent QA overall lacks exact ten zones")
    totals = zone_raw.groupby(
        [*RESULT_SCOPE_FIELDS, "method"], sort=True, as_index=False
    )["event_count"].sum()
    balanced = balanced.merge(
        totals, on=[*RESULT_SCOPE_FIELDS, "method"], validate="one_to_one"
    )
    balanced["zone_count"] = 10
    balanced["aggregation_identity"] = "EVENT_MEAN_WITHIN_ZONE_THEN_EQUAL_WEIGHT_TEN_ZONES"
    balanced["errf_rank_within_price"] = balanced.groupby(
        ["policy_mode", "evaluation_price_id"]
    )["mean_errf"].rank(method="min", ascending=True).astype(int)
    balanced = balanced.sort_values(
        ["policy_mode", "evaluation_price_id", "errf_rank_within_price", "method"],
        kind="mergesort",
    ).reset_index(drop=True)
    weighted = _means(
        cells.groupby(
            [*RESULT_SCOPE_FIELDS, "method"], sort=True, as_index=False
        )[list(EVENT_SUM_COLUMNS)].sum()
    )
    weighted["aggregation_identity"] = "EVENT_WEIGHTED_ACROSS_ZONES_DESCRIPTIVE_ONLY"
    return balanced, weighted, zone


def _assert_paired_zone_conservation(
    cells: pd.DataFrame, paired_zone: pd.DataFrame
) -> None:
    """Close streamed paired sufficient totals against an independent cell sum."""

    keys = [*RESULT_SCOPE_FIELDS, "zone_or_farm", "method"]
    from_cells = (
        cells.groupby(keys, sort=True, as_index=False)[["event_count", "errf_sum"]]
        .sum()
        .sort_values(keys, kind="mergesort")
        .reset_index(drop=True)
    )
    from_paired = (
        paired_zone.groupby(keys, sort=True, as_index=False)[
            ["event_count", "errf_sum"]
        ]
        .sum()
        .sort_values(keys, kind="mergesort")
        .reset_index(drop=True)
    )
    if (
        len(from_cells) != len(from_paired)
        or not from_cells[keys].equals(from_paired[keys])
        or not np.array_equal(
            from_cells["event_count"].to_numpy(dtype=np.int64),
            from_paired["event_count"].to_numpy(dtype=np.int64),
        )
        or not np.allclose(
            from_cells["errf_sum"].to_numpy(dtype=float),
            from_paired["errf_sum"].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-10,
        )
    ):
        raise RuntimeError(
            "independent QA streamed paired totals do not close against cell metrics"
        )


def _design_zone(cells: pd.DataFrame) -> pd.DataFrame:
    keys = [
        *RESULT_SCOPE_FIELDS,
        "method",
        "zone_or_farm",
        "seed",
        "predictor",
        "horizon_steps",
        "target_coverage",
    ]
    design = cells.groupby(keys, sort=True, as_index=False)[["event_count", "errf_sum"]].sum()
    counts = design.groupby(
        [*RESULT_SCOPE_FIELDS, "method", "zone_or_farm"], sort=True
    ).size()
    if (
        len(counts) != 2 * 5 * 9 * 10
        or not counts.eq(660).all()
        or set(design["seed"].astype(int)) != set(SEEDS)
        or set(design["predictor"].astype(str)) != set(PREDICTORS)
        or set(design["horizon_steps"].astype(int)) != set(HORIZONS)
        or set(design["target_coverage"].astype(float)) != set(COVERAGES)
    ):
        raise RuntimeError("independent QA exact 660 design cells not closed")
    design["cell_mean"] = design["errf_sum"].to_numpy(dtype=float) / design["event_count"].to_numpy(dtype=float)
    result = design.groupby(
        [*RESULT_SCOPE_FIELDS, "method", "zone_or_farm"],
        sort=True,
        as_index=False,
    ).agg(
        zone_mean_errf=("cell_mean", "mean"),
        design_cell_count=("cell_mean", "size"),
        source_event_count=("event_count", "sum"),
    )
    result["aggregation_identity"] = "EVENT_MEAN_WITHIN_660_DESIGN_CELLS_THEN_EQUAL_CELL_MEAN_WITHIN_ZONE"
    return result.reset_index(drop=True)


def _paired(zone: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparators = tuple(method for method in METHODS if method != "CLARA_6A")
    rng = np.random.default_rng(2026082802)
    differences_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    for policy_mode in POLICY_MODES:
        for evaluation_price_id in PRICE_IDS:
            policy_price_id = (
                evaluation_price_id
                if policy_mode == POLICY_MODES[0]
                else MAIN_PRICE_ID
            )
            current = zone[
                zone["policy_mode"].astype(str).eq(policy_mode)
                & zone["evaluation_price_id"].astype(str).eq(
                    evaluation_price_id
                )
                & zone["policy_price_id"].astype(str).eq(policy_price_id)
            ]
            pivot = current.pivot(
                index="zone_or_farm", columns="method", values="zone_mean_errf"
            ).reindex(index=ZONES, columns=METHODS)
            if pivot.isna().any().any():
                raise RuntimeError("independent QA paired matrix incomplete")
            for comparator in comparators:
                clara = pivot["CLARA_6A"].to_numpy(dtype=float)
                baseline = pivot[comparator].to_numpy(dtype=float)
                delta = clara - baseline
                for zone_id, clara_value, baseline_value, difference in zip(
                    ZONES, clara, baseline, delta
                ):
                    differences_rows.append(
                        {
                            "policy_mode": policy_mode,
                            "evaluation_price_id": evaluation_price_id,
                            "policy_price_id": policy_price_id,
                            "comparator": comparator,
                            "zone_or_farm": zone_id,
                            "clara_mean_errf": float(clara_value),
                            "comparator_mean_errf": float(baseline_value),
                            "paired_difference_clara_minus_comparator": float(
                                difference
                            ),
                        }
                    )
                observed = float(delta.mean())
                sample = rng.integers(0, 10, size=(5000, 10))
                bootstrap = delta[sample].mean(axis=1)
                signs = np.where(
                    ((np.arange(1024)[:, None] >> np.arange(10)) & 1) == 1,
                    1.0,
                    -1.0,
                )
                null = (signs * delta[None, :]).mean(axis=1)
                extreme = int((np.abs(null) >= abs(observed) - 1e-15).sum())
                baseline_mean = float(baseline.mean())
                inference_rows.append(
                    {
                        "policy_mode": policy_mode,
                        "evaluation_price_id": evaluation_price_id,
                        "policy_price_id": policy_price_id,
                        "comparator": comparator,
                        "zone_count": 10,
                        "mean_paired_difference_clara_minus_comparator": observed,
                        "relative_difference_percent": (
                            np.nan
                            if baseline_mean == 0.0
                            else 100.0 * observed / baseline_mean
                        ),
                        "zone_bootstrap_ci95_lower": float(
                            np.quantile(bootstrap, 0.025)
                        ),
                        "zone_bootstrap_ci95_upper": float(
                            np.quantile(bootstrap, 0.975)
                        ),
                        "exact_sign_flip_extreme_count": extreme,
                        "exact_sign_flip_configuration_count": 1024,
                        "exact_sign_flip_p_value": extreme / 1024.0,
                        "bootstrap_replicates": 5000,
                        "bootstrap_seed": 2026082802,
                    }
                )
    inference = pd.DataFrame(inference_rows)
    inference["holm_adjusted_p_value"] = np.nan
    inference["holm_reject_0p05"] = False
    for _, indices in inference.groupby(
        ["policy_mode", "evaluation_price_id"], sort=False
    ).groups.items():
        order = sorted(indices, key=lambda index: inference.at[index, "exact_sign_flip_p_value"])
        running = 0.0
        for rank, index in enumerate(order):
            adjusted = min(1.0, max(running, (len(order) - rank) * float(inference.at[index, "exact_sign_flip_p_value"])))
            running = adjusted
            inference.at[index, "holm_adjusted_p_value"] = adjusted
            inference.at[index, "holm_reject_0p05"] = adjusted <= 0.05
    if len(inference) != 80:
        raise RuntimeError("independent QA paired inference必须精确80行")
    inference = inference.sort_values(
        [
            "policy_mode",
            "evaluation_price_id",
            "holm_adjusted_p_value",
            "comparator",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    return pd.DataFrame(differences_rows), inference


def _secondary(diagnostics: pd.DataFrame, actions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = [*RESULT_SCOPE_FIELDS, "method", "zone_or_farm", "regime"]
    zone_raw = diagnostics.groupby(keys, sort=True, as_index=False)[list(DIAGNOSTIC_ADDITIVE)].sum()
    counts = zone_raw.groupby(
        [*RESULT_SCOPE_FIELDS, "method", "regime"], sort=True
    )["zone_or_farm"].nunique()
    if len(counts) != 270 or not counts.eq(10).all():
        raise RuntimeError("independent QA diagnostic zones incomplete")
    zone = _diagnostic_rates(zone_raw).merge(
        _entropy(actions, include_zone=True), on=keys, how="left", validate="one_to_one"
    )
    event_means = [
        "mean_errf", "mean_reserve_up", "mean_reserve_down", "mean_miss_upper",
        "mean_miss_lower", "empirical_coverage", "mean_coverage_target", "coverage_gap",
        "mean_width", "mean_interval_score",
    ]
    identity = [*RESULT_SCOPE_FIELDS, "method", "regime"]
    equal = zone.groupby(identity, sort=True, as_index=False)[event_means].mean()
    raw = zone_raw.groupby(identity, sort=True, as_index=False)[list(DIAGNOSTIC_ADDITIVE)].sum()
    pooled = _diagnostic_rates(raw)
    entropy = _entropy(actions, include_zone=False)
    rates = [
        "TOWR", "TUWR", "ARD", "support_backoff_rate", "guardrail_any_exclusion_rate",
        "guardrail_action_exclusion_rate", "guardrail_empty_rate", "selected_action_guardrail_fail_rate",
    ]
    secondary = equal.merge(pooled[[*identity, *DIAGNOSTIC_ADDITIVE, *rates]], on=identity, validate="one_to_one").merge(entropy, on=identity, validate="one_to_one")
    secondary["zone_count"] = 10
    secondary["event_metric_aggregation_identity"] = "EVENT_MEAN_WITHIN_ZONE_THEN_EQUAL_WEIGHT_TEN_ZONES"
    secondary["rolling_and_guardrail_aggregation_identity"] = "RATIO_OF_FROZEN_INTEGER_SUFFICIENT_COUNTS_ACROSS_TEN_ZONES"
    descriptive = pooled.merge(entropy, on=identity, validate="one_to_one")
    descriptive["aggregation_identity"] = "EVENT_WEIGHTED_ACROSS_ZONES_DESCRIPTIVE_ONLY"
    clara = secondary[secondary["method"].astype(str).eq("CLARA_6A")].reset_index(drop=True)
    return zone, secondary, descriptive, clara


def independent_recompute(children: Iterable[Any]) -> dict[str, pd.DataFrame]:
    raw = _collapse(children)
    _cross_mode_algebra(raw)
    cells = raw["cell_metrics"]
    actions = raw["action_counts"]
    diagnostics = raw["diagnostic_metrics"]
    _assert_paired_zone_conservation(cells, raw["paired_blocks"])
    overall, weighted, zone = _overall(cells)
    design = _design_zone(cells)
    paired_zone, inference = _paired(design)
    stratified = _means(
        cells.groupby(
            [
                *RESULT_SCOPE_FIELDS,
                "method",
                "zone_or_farm",
                "predictor",
                "horizon_steps",
                "target_coverage",
            ],
            sort=True,
            as_index=False,
        )[list(EVENT_SUM_COLUMNS)].sum()
    )
    action_share = pd.concat(
        [
            actions.groupby(
                [*RESULT_SCOPE_FIELDS, "method", "selected_action"],
                sort=True,
                as_index=False,
            )["action_count"].sum().assign(regime="overall"),
            actions.groupby(
                [*RESULT_SCOPE_FIELDS, "method", "ramp_state", "selected_action"],
                sort=True,
                as_index=False,
            )["action_count"].sum().rename(columns={"ramp_state": "regime"}),
        ],
        ignore_index=True,
        sort=False,
    ).sort_values(
        [*RESULT_SCOPE_FIELDS, "method", "regime", "selected_action"],
        kind="mergesort",
    ).reset_index(drop=True)
    totals = action_share.groupby(
        [*RESULT_SCOPE_FIELDS, "method", "regime"], sort=True
    )["action_count"].transform("sum")
    action_share["action_share"] = action_share["action_count"].to_numpy(dtype=float) / totals.to_numpy(dtype=float)
    diagnostic_zone, secondary, descriptive, clara = _secondary(diagnostics, actions)
    return {
        "compact_artifact_index": raw["compact_artifact_index"],
        "cell_metrics": cells,
        "action_counts": actions,
        "diagnostic_metrics": diagnostics,
        "clara_state_counts": raw["clara_state_counts"],
        "event_conservation": raw["event_conservation"],
        "overall_ranking": overall,
        "event_weighted_overall_descriptive": weighted,
        "zone_metric_means": zone,
        "zone_design_equal_metrics": design,
        "stratified_metrics": stratified,
        "action_share": action_share,
        "diagnostic_zone_metrics": diagnostic_zone,
        "secondary_zone_balanced": secondary,
        "secondary_event_weighted_descriptive": descriptive,
        "clara_support_guardrail_summary": clara,
        "paired_zone_differences": paired_zone,
        "paired_inference": inference,
    }


def _sorted(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    return frame.sort_values(list(keys), kind="mergesort").reset_index(drop=True)


def _compare(
    expected: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    keys: Sequence[str],
    label: str,
    subset_columns: bool = False,
) -> None:
    if subset_columns:
        missing = sorted(set(expected.columns) - set(observed.columns))
        if missing:
            raise RuntimeError(f"independent QA {label} missing columns: {missing}")
        observed = observed.loc[:, list(expected.columns)]
    elif list(expected.columns) != list(observed.columns):
        raise RuntimeError(f"independent QA {label} exact columns differ")
    try:
        pd.testing.assert_frame_equal(
            _sorted(expected, keys),
            _sorted(observed, keys),
            check_dtype=True,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as error:
        raise RuntimeError(f"independent QA {label} mismatch: {error}") from error


def validate_aggregate_outputs(
    children: Iterable[Any],
    stored_outputs: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Independently recompute every sealed compact layer and return PASS checks."""

    recomputed = independent_recompute(children)
    if set(stored_outputs) != set(recomputed):
        raise RuntimeError(
            "independent QA aggregate output set differs: "
            f"missing={sorted(set(recomputed)-set(stored_outputs))}, "
            f"extra={sorted(set(stored_outputs)-set(recomputed))}"
        )
    keys = {
        "compact_artifact_index": (
            "evaluation_zone",
            "seed",
            "policy_mode",
            "evaluation_price_id",
        ),
        "cell_metrics": (*RESULT_SCOPE_FIELDS, "method", *CELL_FIELDS),
        "action_counts": (
            *RESULT_SCOPE_FIELDS,
            "method",
            *CELL_FIELDS,
            "selected_action",
        ),
        "diagnostic_metrics": (
            *RESULT_SCOPE_FIELDS,
            "method",
            *DIAGNOSTIC_FIELDS,
        ),
        "clara_state_counts": tuple(
            column for column in recomputed["clara_state_counts"].columns if column != "event_count"
        ),
        "event_conservation": (*RESULT_SCOPE_FIELDS, "method"),
        "overall_ranking": (*RESULT_SCOPE_FIELDS, "method"),
        "event_weighted_overall_descriptive": (*RESULT_SCOPE_FIELDS, "method"),
        "zone_metric_means": (*RESULT_SCOPE_FIELDS, "method", "zone_or_farm"),
        "zone_design_equal_metrics": (
            *RESULT_SCOPE_FIELDS,
            "method",
            "zone_or_farm",
        ),
        "stratified_metrics": (
            *RESULT_SCOPE_FIELDS,
            "method",
            "zone_or_farm",
            "predictor",
            "horizon_steps",
            "target_coverage",
        ),
        "action_share": (
            *RESULT_SCOPE_FIELDS,
            "method",
            "regime",
            "selected_action",
        ),
        "diagnostic_zone_metrics": (
            *RESULT_SCOPE_FIELDS,
            "method",
            "zone_or_farm",
            "regime",
        ),
        "secondary_zone_balanced": (*RESULT_SCOPE_FIELDS, "method", "regime"),
        "secondary_event_weighted_descriptive": (
            *RESULT_SCOPE_FIELDS,
            "method",
            "regime",
        ),
        "clara_support_guardrail_summary": (
            *RESULT_SCOPE_FIELDS,
            "method",
            "regime",
        ),
        "paired_zone_differences": (
            *RESULT_SCOPE_FIELDS,
            "comparator",
            "zone_or_farm",
        ),
        "paired_inference": (*RESULT_SCOPE_FIELDS, "comparator"),
    }
    checks: list[dict[str, str]] = []
    for name, expected in recomputed.items():
        _compare(
            expected,
            stored_outputs[name],
            keys=keys[name],
            label=name,
            subset_columns=name == "compact_artifact_index",
        )
        checks.append(
            {
                "check_id": f"INDEPENDENT_RECOMPUTE_{name.upper()}",
                "status": "PASS",
                "detail": f"rows={len(expected)}",
            }
        )
    conservation = recomputed["event_conservation"]
    if len(conservation) != 90 or not conservation["event_count"].eq(23_024_760).all():
        raise RuntimeError("independent QA event conservation failed")
    checks.append({"check_id": "INDEPENDENT_GLOBAL_EVENT_CONSERVATION", "status": "PASS", "detail": "90 method-mode-evaluation-price rows x 23,024,760"})
    action_totals = recomputed["action_counts"].groupby(
        [*RESULT_SCOPE_FIELDS, "method"], sort=True
    )["action_count"].sum()
    if len(action_totals) != 90 or not action_totals.eq(23_024_760).all():
        raise RuntimeError("independent QA action conservation failed")
    checks.append({"check_id": "INDEPENDENT_GLOBAL_ACTION_CONSERVATION", "status": "PASS", "detail": "90 method-mode-evaluation-price rows x 23,024,760"})
    diagnostic = recomputed["diagnostic_metrics"]
    overall_count = diagnostic[
        diagnostic["regime"].astype(str).eq("overall")
    ].groupby([*RESULT_SCOPE_FIELDS, "method"])["event_count"].sum()
    if len(overall_count) != 90 or not overall_count.eq(23_024_760).all():
        raise RuntimeError("independent QA diagnostic conservation failed")
    checks.append({"check_id": "INDEPENDENT_SELECTED_ENDPOINT_DIAGNOSTICS", "status": "PASS", "detail": "rolling/support/guardrail sufficient counts closed"})
    return pd.DataFrame(checks)


__all__ = [
    "SCHEMA",
    "module_sha256",
    "independent_recompute",
    "validate_aggregate_outputs",
]
