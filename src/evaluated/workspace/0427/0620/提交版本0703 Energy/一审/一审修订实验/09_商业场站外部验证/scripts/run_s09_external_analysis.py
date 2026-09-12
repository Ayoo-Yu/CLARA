from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


S09_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = S09_ROOT / "results_raw" / "minimum_external_evaluation_v1"
RAW_QA = S09_ROOT / "qa" / "minimum_external_evaluation_v1" / "qa_summary.json"
CONFIG_PATH = S09_ROOT / "configs" / "s09_minimum_external_evaluation_v1.json"
EXTERNAL_PROTOCOL_PATH = S09_ROOT / "configs" / "s09_external_validation_v1.json"
OUTPUT_ROOT = S09_ROOT / "results_verified" / "minimum_external_analysis_v1"
TEMP_ROOT = OUTPUT_ROOT.with_name(f"{OUTPUT_ROOT.name}.tmp")

EVENT_METRICS = [
    "mean_errf",
    "mean_reserve",
    "mean_miss",
    "empirical_coverage",
    "coverage_gap",
    "average_width",
]
RELIABILITY_METRICS = ["TOWR", "TUWR", "ARD"]
WEIGHTINGS = [
    "time_sample_weighted",
    "equal_farm_weighted",
    "capacity_weighted",
]
CLARA_METHODS = ["CLARA_V4_Direct", "CLARA_V4_Local"]
FIXED_ACTIONS = ["Static", "ACI", "AgACI", "EnbPI_RH"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    denominator = float(weights.sum())
    if denominator <= 0:
        return float("nan")
    return float(np.dot(values.to_numpy(float), weights.to_numpy(float)) / denominator)


def build_farm_summary(cell_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["zone_or_farm", "method", "regime"]
    for key, group in cell_metrics.groupby(keys, sort=True):
        row: dict[str, Any] = dict(zip(keys, key, strict=True))
        row["event_count"] = int(group["event_count"].sum())
        row["reliability_event_count"] = int(group["reliability_event_count"].sum())
        for metric in EVENT_METRICS:
            row[metric] = weighted_average(group[metric], group["event_count"])
        for metric in RELIABILITY_METRICS:
            row[metric] = weighted_average(
                group[metric], group["reliability_event_count"]
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys, kind="mergesort").reset_index(drop=True)


def build_weighted_summary(
    farm_summary: pd.DataFrame,
    capacities: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method, regime), group in farm_summary.groupby(
        ["method", "regime"], sort=True
    ):
        for weighting in WEIGHTINGS:
            row: dict[str, Any] = {
                "weighting": weighting,
                "method": method,
                "regime": regime,
                "farm_count": int(group["zone_or_farm"].nunique()),
                "event_count": int(group["event_count"].sum()),
                "reliability_event_count": int(
                    group["reliability_event_count"].sum()
                ),
            }
            if weighting == "time_sample_weighted":
                event_weights = group["event_count"].astype(float)
                reliability_weights = group["reliability_event_count"].astype(float)
            elif weighting == "equal_farm_weighted":
                event_weights = pd.Series(np.ones(len(group)), index=group.index)
                reliability_weights = event_weights
            else:
                capacity_values = group["zone_or_farm"].map(capacities).astype(float)
                event_weights = capacity_values
                reliability_weights = capacity_values
            for metric in EVENT_METRICS:
                row[metric] = weighted_average(group[metric], event_weights)
            for metric in RELIABILITY_METRICS:
                row[metric] = weighted_average(group[metric], reliability_weights)
            rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["weighting", "regime", "mean_errf", "method"], kind="mergesort")
        .reset_index(drop=True)
    )


def build_rankings(weighted_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (weighting, regime), group in weighted_summary.groupby(
        ["weighting", "regime"], sort=True
    ):
        ordered = group.sort_values(["mean_errf", "method"], kind="mergesort")
        for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
            rows.append(
                {
                    "weighting": weighting,
                    "regime": regime,
                    "rank": rank,
                    "method": row["method"],
                    "mean_errf": float(row["mean_errf"]),
                    "coverage_gap": float(row["coverage_gap"]),
                    "TUWR": float(row["TUWR"]),
                    "ARD": float(row["ARD"]),
                }
            )
    return pd.DataFrame(rows)


def build_action_mix(action_counts: pd.DataFrame) -> pd.DataFrame:
    overall = action_counts.loc[action_counts["regime"].eq("overall")].copy()
    grouped = (
        overall.groupby(
            ["zone_or_farm", "method", "selected_action"],
            as_index=False,
            sort=True,
        )["event_count"]
        .sum()
        .rename(columns={"event_count": "selected_event_count"})
    )
    grouped["method_event_count"] = grouped.groupby(
        ["zone_or_farm", "method"]
    )["selected_event_count"].transform("sum")
    grouped["selection_share"] = (
        grouped["selected_event_count"] / grouped["method_event_count"]
    )
    return grouped


def deterministic_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join([str(base_seed), *[str(part) for part in parts]])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def paired_bootstrap(
    block_statistics: pd.DataFrame,
    *,
    methods: list[str],
    capacities: dict[str, float],
    replicates: int,
    base_seed: int,
) -> pd.DataFrame:
    output_rows: list[dict[str, Any]] = []
    for block_hours in sorted(block_statistics["block_hours"].unique()):
        for regime in sorted(block_statistics["regime"].unique()):
            subset = block_statistics.loc[
                block_statistics["block_hours"].eq(block_hours)
                & block_statistics["regime"].eq(regime)
            ]
            clustered = (
                subset.groupby(
                    ["zone_or_farm", "block_start_utc", "method"],
                    as_index=False,
                    sort=True,
                )[["event_count", "errf_sum"]]
                .sum()
            )
            farm_point_numerators: dict[str, dict[str, float]] = {}
            farm_point_denominators: dict[str, float] = {}
            farm_boot_numerators: dict[str, dict[str, np.ndarray]] = {}
            farm_boot_denominators: dict[str, np.ndarray] = {}
            for farm in sorted(capacities):
                farm_data = clustered.loc[clustered["zone_or_farm"].eq(farm)]
                numerator_pivot = farm_data.pivot(
                    index="block_start_utc", columns="method", values="errf_sum"
                ).sort_index()
                count_pivot = farm_data.pivot(
                    index="block_start_utc", columns="method", values="event_count"
                ).sort_index()
                if (
                    numerator_pivot.empty
                    or numerator_pivot.isna().any().any()
                    or count_pivot.isna().any().any()
                    or set(numerator_pivot.columns) != set(methods)
                ):
                    raise RuntimeError(
                        f"S09 bootstrap 配对块不完整: {block_hours}, {regime}, {farm}"
                    )
                numerator_pivot = numerator_pivot.reindex(columns=methods)
                count_pivot = count_pivot.reindex(columns=methods)
                count_values = count_pivot.to_numpy(float)
                if not np.allclose(count_values, count_values[:, [0]], atol=0, rtol=0):
                    raise RuntimeError(
                        f"S09 bootstrap 方法间事件数不守恒: {block_hours}, {regime}, {farm}"
                    )
                common_counts = count_values[:, 0]
                numerator_values = numerator_pivot.to_numpy(float)
                rng = np.random.default_rng(
                    deterministic_seed(base_seed, block_hours, regime, farm)
                )
                sample_indices = rng.integers(
                    0,
                    len(numerator_pivot),
                    size=(replicates, len(numerator_pivot)),
                )
                sampled_denominator = common_counts[sample_indices].sum(axis=1)
                farm_point_denominators[farm] = float(common_counts.sum())
                farm_boot_denominators[farm] = sampled_denominator
                farm_point_numerators[farm] = {}
                farm_boot_numerators[farm] = {}
                for method_index, method in enumerate(methods):
                    values = numerator_values[:, method_index]
                    farm_point_numerators[farm][method] = float(values.sum())
                    farm_boot_numerators[farm][method] = values[sample_indices].sum(axis=1)

            point_by_weighting: dict[str, dict[str, float]] = {}
            boot_by_weighting: dict[str, dict[str, np.ndarray]] = {}
            for weighting in WEIGHTINGS:
                point_by_weighting[weighting] = {}
                boot_by_weighting[weighting] = {}
                for method in methods:
                    farm_points = {
                        farm: farm_point_numerators[farm][method]
                        / farm_point_denominators[farm]
                        for farm in capacities
                    }
                    farm_boot = {
                        farm: farm_boot_numerators[farm][method]
                        / farm_boot_denominators[farm]
                        for farm in capacities
                    }
                    if weighting == "time_sample_weighted":
                        point = sum(
                            farm_point_numerators[farm][method] for farm in capacities
                        ) / sum(farm_point_denominators.values())
                        boot = sum(
                            farm_boot_numerators[farm][method] for farm in capacities
                        ) / sum(farm_boot_denominators.values())
                    elif weighting == "equal_farm_weighted":
                        point = float(np.mean(list(farm_points.values())))
                        boot = np.mean(np.stack(list(farm_boot.values())), axis=0)
                    else:
                        capacity_total = float(sum(capacities.values()))
                        point = sum(
                            farm_points[farm] * capacities[farm] for farm in capacities
                        ) / capacity_total
                        boot = sum(
                            farm_boot[farm] * capacities[farm] for farm in capacities
                        ) / capacity_total
                    point_by_weighting[weighting][method] = float(point)
                    boot_by_weighting[weighting][method] = np.asarray(boot, dtype=float)

            for weighting in WEIGHTINGS:
                for clara_method in CLARA_METHODS:
                    for comparator in methods:
                        if comparator == clara_method:
                            continue
                        point_difference = (
                            point_by_weighting[weighting][clara_method]
                            - point_by_weighting[weighting][comparator]
                        )
                        bootstrap_difference = (
                            boot_by_weighting[weighting][clara_method]
                            - boot_by_weighting[weighting][comparator]
                        )
                        probability_better = float(np.mean(bootstrap_difference < 0))
                        output_rows.append(
                            {
                                "block_hours": int(block_hours),
                                "regime": regime,
                                "weighting": weighting,
                                "clara_method": clara_method,
                                "comparator": comparator,
                                "mean_errf_difference": float(point_difference),
                                "ci95_lower": float(
                                    np.quantile(bootstrap_difference, 0.025)
                                ),
                                "ci95_upper": float(
                                    np.quantile(bootstrap_difference, 0.975)
                                ),
                                "probability_clara_better": probability_better,
                                "paired_sign_p_two_sided": float(
                                    min(1.0, 2.0 * min(probability_better, 1.0 - probability_better))
                                ),
                                "bootstrap_replicates": int(replicates),
                                "bootstrap_seed": int(base_seed),
                            }
                        )
    return pd.DataFrame(output_rows)


def build_adverse_gate(weighted_summary: pd.DataFrame) -> dict[str, Any]:
    primary = weighted_summary.loc[
        weighted_summary["weighting"].eq("equal_farm_weighted")
        & weighted_summary["regime"].eq("overall")
    ].set_index("method")
    ramp = weighted_summary.loc[
        weighted_summary["weighting"].eq("equal_farm_weighted")
        & weighted_summary["regime"].eq("ramp")
    ].set_index("method")
    best_fixed = min(FIXED_ACTIONS, key=lambda method: float(primary.loc[method, "mean_errf"]))
    relative_excess = {
        method: float(primary.loc[method, "mean_errf"])
        / float(primary.loc[best_fixed, "mean_errf"])
        - 1.0
        for method in CLARA_METHODS
    }
    tuwr_improvement = {
        method: bool(float(primary.loc[method, "TUWR"]) < float(primary.loc[best_fixed, "TUWR"]))
        for method in CLARA_METHODS
    }
    ramp_improvement = {
        method: bool(
            float(ramp.loc[method, "mean_errf"])
            < float(ramp.loc[best_fixed, "mean_errf"])
        )
        for method in CLARA_METHODS
    }
    triggered = (
        all(value > 0.10 for value in relative_excess.values())
        and not any(tuwr_improvement.values())
        and not any(ramp_improvement.values())
    )
    return {
        "schema": "S09_EXTERNAL_ADVERSE_RESULT_GATE_V1",
        "status": "TRIGGERED" if triggered else "PASS",
        "aggregation": "event_weighted_within_farm_then_equal_mean_across_two_farms",
        "best_fixed_action": best_fixed,
        "best_fixed_overall_mean_errf": float(primary.loc[best_fixed, "mean_errf"]),
        "best_fixed_overall_tuwr": float(primary.loc[best_fixed, "TUWR"]),
        "best_fixed_ramp_mean_errf": float(ramp.loc[best_fixed, "mean_errf"]),
        "clara_relative_errf_excess_vs_best_fixed": relative_excess,
        "clara_tuwr_improvement_vs_best_fixed": tuwr_improvement,
        "clara_ramp_errf_improvement_vs_best_fixed": ramp_improvement,
        "adverse_result_gate_triggered": bool(triggered),
    }


def build_overview(
    weighted_summary: pd.DataFrame,
    rankings: pd.DataFrame,
    adverse_gate: dict[str, Any],
) -> dict[str, Any]:
    top_methods: list[dict[str, Any]] = []
    for (weighting, regime), group in rankings.groupby(
        ["weighting", "regime"], sort=True
    ):
        first = group.sort_values("rank").iloc[0]
        top_methods.append(
            {
                "weighting": weighting,
                "regime": regime,
                "method": first["method"],
                "mean_errf": float(first["mean_errf"]),
            }
        )
    clara_rows = rankings.loc[rankings["method"].isin(CLARA_METHODS)].copy()
    return {
        "schema": "S09_EXTERNAL_ANALYSIS_OVERVIEW_V1",
        "top_methods": top_methods,
        "clara_rankings": clara_rows.to_dict(orient="records"),
        "adverse_result_gate": adverse_gate,
        "weighted_summary_row_count": int(len(weighted_summary)),
    }


def build_markdown(
    rankings: pd.DataFrame,
    weighted_summary: pd.DataFrame,
    adverse_gate: dict[str, Any],
) -> str:
    lines = [
        "# S09 商业场站外部验证结果摘要",
        "",
        "## 技术状态",
        "",
        "六个正式评价单元已完成，独立技术 QA 已通过。下列统计只在 QA 通过后生成。",
        "",
        "## 三种加权口径下的 ERRF 第一名",
        "",
        "| 加权口径 | 工况 | 方法 | mean ERRF |",
        "|---|---|---|---:|",
    ]
    for (weighting, regime), group in rankings.groupby(
        ["weighting", "regime"], sort=True
    ):
        row = group.sort_values("rank").iloc[0]
        lines.append(
            f"| {weighting} | {regime} | {row['method']} | {float(row['mean_errf']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## CLARA 排名",
            "",
            "| 加权口径 | 工况 | 方法 | 排名 | mean ERRF | TUWR |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    clara = rankings.loc[rankings["method"].isin(CLARA_METHODS)]
    summary_index = weighted_summary.set_index(["weighting", "regime", "method"])
    for row in clara.sort_values(["weighting", "regime", "method"]).itertuples():
        tuwr = float(summary_index.loc[(row.weighting, row.regime, row.method), "TUWR"])
        lines.append(
            f"| {row.weighting} | {row.regime} | {row.method} | {int(row.rank)} | {float(row.mean_errf):.6f} | {tuwr:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 不利结果安全门",
            "",
            f"安全门状态：{adverse_gate['status']}。",
            f"最佳固定动作：{adverse_gate['best_fixed_action']}。",
            "",
            "完整数值、动作构成和 24 小时及 168 小时配对块 bootstrap 结果见同目录 CSV 文件。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if OUTPUT_ROOT.exists() or TEMP_ROOT.exists():
        raise RuntimeError(f"S09 分析输出目录已存在，拒绝覆盖: {OUTPUT_ROOT}")
    raw_manifest = load_json(RAW_ROOT / "root_manifest.json")
    raw_qa = load_json(RAW_QA)
    config = load_json(CONFIG_PATH)
    external_protocol = load_json(EXTERNAL_PROTOCOL_PATH)
    if raw_manifest.get("status") != "COMPLETE_PENDING_INDEPENDENT_QA":
        raise RuntimeError("S09 正式评价根清单状态异常")
    if raw_qa.get("status") != "PASS" or int(raw_qa.get("failed_count", -1)) != 0:
        raise RuntimeError("S09 独立技术 QA 未通过，禁止读取性能")
    if str(raw_qa.get("root_manifest_sha256")) != sha256_file(
        RAW_ROOT / "root_manifest.json"
    ):
        raise RuntimeError("S09 正式评价根清单与 QA 身份失配")

    methods = list(config["scope"]["methods"])
    capacities = {
        farm: float(details["capacity_mw"])
        for farm, details in external_protocol["data"]["farms"].items()
    }
    cell_metrics = pd.read_parquet(RAW_ROOT / "cell_metrics.parquet")
    action_counts = pd.read_parquet(RAW_ROOT / "action_counts.parquet")
    block_statistics = pd.read_parquet(RAW_ROOT / "paired_block_statistics.parquet")
    if set(cell_metrics["method"].unique()) != set(methods):
        raise RuntimeError("S09 正式评价方法集合失配")

    farm_summary = build_farm_summary(cell_metrics)
    weighted_summary = build_weighted_summary(farm_summary, capacities)
    rankings = build_rankings(weighted_summary)
    action_mix = build_action_mix(action_counts)
    bootstrap = paired_bootstrap(
        block_statistics,
        methods=methods,
        capacities=capacities,
        replicates=int(config["reporting"]["bootstrap_replicates"]),
        base_seed=int(config["reporting"]["bootstrap_seed"]),
    )
    adverse_gate = build_adverse_gate(weighted_summary)
    overview = build_overview(weighted_summary, rankings, adverse_gate)

    TEMP_ROOT.mkdir(parents=True, exist_ok=False)
    farm_summary.to_csv(TEMP_ROOT / "farm_method_summary.csv", index=False, encoding="utf-8")
    weighted_summary.to_csv(
        TEMP_ROOT / "weighted_method_summary.csv", index=False, encoding="utf-8"
    )
    rankings.to_csv(TEMP_ROOT / "method_rankings.csv", index=False, encoding="utf-8")
    action_mix.to_csv(TEMP_ROOT / "action_mix.csv", index=False, encoding="utf-8")
    bootstrap.to_csv(
        TEMP_ROOT / "paired_block_bootstrap.csv", index=False, encoding="utf-8"
    )
    write_json(TEMP_ROOT / "adverse_result_gate.json", adverse_gate)
    write_json(TEMP_ROOT / "analysis_overview.json", overview)
    (TEMP_ROOT / "S09_EXTERNAL_RESULTS_SUMMARY.md").write_text(
        build_markdown(rankings, weighted_summary, adverse_gate), encoding="utf-8"
    )
    artifact_names = [
        "farm_method_summary.csv",
        "weighted_method_summary.csv",
        "method_rankings.csv",
        "action_mix.csv",
        "paired_block_bootstrap.csv",
        "adverse_result_gate.json",
        "analysis_overview.json",
        "S09_EXTERNAL_RESULTS_SUMMARY.md",
    ]
    manifest = {
        "schema": "S09_MINIMUM_EXTERNAL_ANALYSIS_ROOT_MANIFEST_V1",
        "status": "COMPLETE_PENDING_INDEPENDENT_QA",
        "input_root_manifest_sha256": sha256_file(RAW_ROOT / "root_manifest.json"),
        "input_qa_summary_sha256": sha256_file(RAW_QA),
        "config_sha256": sha256_file(CONFIG_PATH),
        "external_protocol_sha256": sha256_file(EXTERNAL_PROTOCOL_PATH),
        "method_count": len(methods),
        "farm_summary_row_count": int(len(farm_summary)),
        "weighted_summary_row_count": int(len(weighted_summary)),
        "ranking_row_count": int(len(rankings)),
        "action_mix_row_count": int(len(action_mix)),
        "bootstrap_row_count": int(len(bootstrap)),
        "bootstrap_replicates": int(config["reporting"]["bootstrap_replicates"]),
        "bootstrap_seed": int(config["reporting"]["bootstrap_seed"]),
        "adverse_result_gate_triggered": bool(
            adverse_gate["adverse_result_gate_triggered"]
        ),
        "artifacts": {
            name: sha256_file(TEMP_ROOT / name) for name in artifact_names
        },
    }
    write_json(TEMP_ROOT / "root_manifest.json", manifest)
    TEMP_ROOT.replace(OUTPUT_ROOT)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
