from __future__ import annotations

import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil


S06_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = S06_ROOT.parent
AUTH_ROOT = REVISION_ROOT / "_权威代码"
CODE_ROOT = AUTH_ROOT / "code"
RUN_ROOT = S06_ROOT / "results_raw" / "nested_source_selection_v1"
SELECTION_ROOT = RUN_ROOT / "nested_selection" / "supervised_v1"
sys.path.insert(0, str(CODE_ROOT))

from baseline_common import build_nested_source_zone_folds  # noqa: E402
from baseline_compact_training import (  # noqa: E402
    compact_state_events,
    fit_compact_cart_best_action,
    fit_compact_per_action_risk,
    selected_action_metrics_from_compact_statistics,
)
from baseline_selection import select_configuration  # noqa: E402
from clara_event_contract import load_frozen_contracts, sha256_file  # noqa: E402


EXPECTED_CODE_COMMIT = "4d76f25c84176ba1e232acf3a22d467672e1984c"
SOURCE_TUNING_CONTRACT_SHA256 = "9a352cba41d7e68911626da950540d1d8506f2f4f63afb72bd7b28cd473c7e14"
STATE_STATISTICS_SHA256 = "8e083819b430f2723f4186997d9527b57bfd9cc59ccec7b114fffeacd1e939fb"
SUPERVISED_GATE_SHA256 = "d482c3f35a5a984786a56bdc075b2dcf4fdb8fd51e0f2b0143359907b954d352"
WORKER_COUNT = 6
EXPECTED_CONFIG_COUNTS = {
    "BestFixedChosenInSourceZones": 4,
    "CARTBestAction": 30,
    "RidgePerActionRisk": 5,
    "GBRPerActionRisk": 16,
}
EXPECTED_INNER_SCORE_ROWS = sum(EXPECTED_CONFIG_COUNTS.values())
EXPECTED_OUTER_SCORE_ROWS = EXPECTED_INNER_SCORE_ROWS * 9
EXPECTED_ROOT_SCORE_ROWS = EXPECTED_OUTER_SCORE_ROWS * 10
EXPECTED_ROOT_SELECTION_ROWS = len(EXPECTED_CONFIG_COUNTS) * 10
MINIMUM_FREE_DISK_GIB = 60.0


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=AUTH_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def zone_number(zone: str) -> int:
    text = str(zone)
    if text.startswith("zone") and text[4:].isdigit():
        return int(text[4:])
    raise ValueError(f"区域编号无法解析: {text}")


def configuration_grid(contracts: Any) -> dict[str, list[dict[str, Any]]]:
    entries = {
        str(entry["id"]): entry
        for entry in contracts.baseline_registry["mandatory_contextual_selectors"]
    }
    cart = entries["CARTBestAction"]["grid"]
    ridge = entries["RidgePerActionRisk"]["grid"]
    gbr = entries["GBRPerActionRisk"]["grid"]
    grids = {
        "CARTBestAction": [
            {"max_depth": depth, "min_samples_leaf": leaf, "class_weight": weight}
            for depth, leaf, weight in itertools.product(
                cart["max_depth"],
                cart["min_samples_leaf"],
                cart["class_weight"],
            )
        ],
        "RidgePerActionRisk": [{"alpha": alpha} for alpha in ridge["alpha"]],
        "GBRPerActionRisk": [
            {
                "n_estimators": estimators,
                "learning_rate": rate,
                "max_depth": depth,
                "min_samples_leaf": leaf,
            }
            for estimators, rate, depth, leaf in itertools.product(
                gbr["n_estimators"],
                gbr["learning_rate"],
                gbr["max_depth"],
                gbr["min_samples_leaf"],
            )
        ],
    }
    observed = {key: len(values) for key, values in grids.items()}
    if observed != {key: EXPECTED_CONFIG_COUNTS[key] for key in grids}:
        raise RuntimeError(f"监督基线冻结网格规模失配: {observed}")
    return grids


def score_selected_actions(
    *,
    contracts: Any,
    validation_statistics: pd.DataFrame,
    selected_actions: list[str],
    validation_zone: str,
) -> dict[str, Any]:
    metrics = selected_action_metrics_from_compact_statistics(
        contracts=contracts,
        statistics=validation_statistics,
        selected_actions=selected_actions,
        inner_validation_zone=validation_zone,
    )
    numeric = [metrics["mean_errf"], metrics["mean_coverage_gap"], metrics["mean_tuwr"]]
    if not np.isfinite(np.asarray(numeric, dtype=float)).all():
        raise RuntimeError("监督基线内层验证分数含非有限值")
    return metrics


def fixed_action_rows(
    *,
    contracts: Any,
    validation_statistics: pd.DataFrame,
    outer_heldout_zone: str,
    validation_zone: str,
) -> list[dict[str, Any]]:
    mapping = {
        "Static": "FixedStatic",
        "ACI": "FixedACI",
        "AgACI": "FixedAgACI",
        "EnbPI_RH": "FixedEnbPI_RH",
    }
    rows: list[dict[str, Any]] = []
    for complexity_rank, action in enumerate(contracts.actions):
        metrics = score_selected_actions(
            contracts=contracts,
            validation_statistics=validation_statistics,
            selected_actions=[action] * len(validation_statistics),
            validation_zone=validation_zone,
        )
        rows.append(
            {
                "outer_heldout_zone": outer_heldout_zone,
                "inner_validation_zone": validation_zone,
                "baseline_id": "BestFixedChosenInSourceZones",
                "configuration_id": mapping[action],
                "configuration_json": json.dumps({"action": action}, ensure_ascii=False, sort_keys=True),
                "complexity_rank": complexity_rank,
                **metrics,
                "fit_seconds": 0.0,
                "predict_seconds": 0.0,
                "fit_zone_count": 0,
                "outer_heldout_zone_fit": False,
                "inner_validation_zone_fit": False,
            }
        )
    return rows


def fitted_configuration_row(
    *,
    contracts: Any,
    training_statistics: pd.DataFrame,
    validation_statistics: pd.DataFrame,
    baseline_id: str,
    config: dict[str, Any],
    complexity_rank: int,
    outer_heldout_zone: str,
    validation_zone: str,
    training_zones: tuple[str, ...],
) -> dict[str, Any]:
    fit_started = time.perf_counter()
    if baseline_id == "CARTBestAction":
        selector = fit_compact_cart_best_action(
            contracts=contracts,
            statistics=training_statistics,
            config=config,
            training_zones=training_zones,
            outer_heldout_zone=outer_heldout_zone,
        )
    else:
        selector = fit_compact_per_action_risk(
            contracts=contracts,
            statistics=training_statistics,
            baseline_id=baseline_id,
            config=config,
            training_zones=training_zones,
            outer_heldout_zone=outer_heldout_zone,
        )
    fit_seconds = time.perf_counter() - fit_started
    if selector.fit_zones != tuple(sorted(training_zones)):
        raise RuntimeError("监督基线拟合源区身份失配")
    if outer_heldout_zone in set(selector.fit_zones) or validation_zone in set(selector.fit_zones):
        raise RuntimeError("监督基线拟合包含外层持出区或内层验证区")
    validation_events = compact_state_events(validation_statistics)
    predict_started = time.perf_counter()
    action_map, _ = selector.predict_actions(validation_events)
    selected_actions = [action_map[event_id] for event_id in validation_events["event_id"].astype(str)]
    predict_seconds = time.perf_counter() - predict_started
    metrics = score_selected_actions(
        contracts=contracts,
        validation_statistics=validation_statistics,
        selected_actions=selected_actions,
        validation_zone=validation_zone,
    )
    return {
        "outer_heldout_zone": outer_heldout_zone,
        "inner_validation_zone": validation_zone,
        "baseline_id": baseline_id,
        "configuration_id": selector.config_id,
        "configuration_json": json.dumps(selector.config, ensure_ascii=False, sort_keys=True),
        "complexity_rank": complexity_rank,
        **metrics,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "fit_zone_count": len(selector.fit_zones),
        "outer_heldout_zone_fit": False,
        "inner_validation_zone_fit": False,
    }


def inner_paths(outer_zone: str, validation_zone: str) -> tuple[Path, Path]:
    root = SELECTION_ROOT / f"outer_heldout={outer_zone}" / f"inner_validation={validation_zone}"
    return root / "validation_scores.parquet", root / "manifest.json"


def verified_inner_checkpoint(
    *,
    outer_zone: str,
    validation_zone: str,
    training_zones: tuple[str, ...],
) -> pd.DataFrame | None:
    score_path, manifest_path = inner_paths(outer_zone, validation_zone)
    if not score_path.exists() or not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "checkpoint_schema": "S06_SUPERVISED_INNER_SELECTION_V1",
            "status": "PASS",
            "code_commit": EXPECTED_CODE_COMMIT,
            "state_statistics_sha256": STATE_STATISTICS_SHA256,
            "supervised_gate_sha256": SUPERVISED_GATE_SHA256,
            "outer_heldout_zone": outer_zone,
            "inner_validation_zone": validation_zone,
            "training_zones": list(training_zones),
            "score_row_count": EXPECTED_INNER_SCORE_ROWS,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            return None
        if sha256_file(score_path) != str(manifest.get("score_file_sha256")):
            return None
        frame = pd.read_parquet(score_path)
        if len(frame) != EXPECTED_INNER_SCORE_ROWS:
            return None
        return frame
    except Exception:
        return None


def write_inner_checkpoint(
    *,
    scores: pd.DataFrame,
    outer_zone: str,
    validation_zone: str,
    training_zones: tuple[str, ...],
) -> None:
    score_path, manifest_path = inner_paths(outer_zone, validation_zone)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_score_path = score_path.with_name("validation_scores.tmp.parquet")
    scores.to_parquet(temporary_score_path, index=False, compression="zstd")
    os.replace(temporary_score_path, score_path)
    manifest = {
        "checkpoint_schema": "S06_SUPERVISED_INNER_SELECTION_V1",
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": EXPECTED_CODE_COMMIT,
        "source_tuning_contract_sha256": SOURCE_TUNING_CONTRACT_SHA256,
        "state_statistics_sha256": STATE_STATISTICS_SHA256,
        "supervised_gate_sha256": SUPERVISED_GATE_SHA256,
        "outer_heldout_zone": outer_zone,
        "inner_validation_zone": validation_zone,
        "training_zones": list(training_zones),
        "score_row_count": len(scores),
        "score_file_sha256": sha256_file(score_path),
        "outer_heldout_zone_fit_count": int(scores["outer_heldout_zone_fit"].sum()),
        "inner_validation_zone_fit_count": int(scores["inner_validation_zone_fit"].sum()),
        "configuration_selection_executed": False,
        "performance_comparison_executed": False,
        "method_ranking_executed": False,
        "statistical_inference_executed": False,
        "paper_performance_claim_executed": False,
    }
    temporary_manifest_path = manifest_path.with_suffix(".tmp.json")
    temporary_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest_path, manifest_path)


def compute_inner_scores(
    *,
    contracts: Any,
    grids: dict[str, list[dict[str, Any]]],
    outer_statistics: pd.DataFrame,
    outer_zone: str,
    validation_zone: str,
    training_zones: tuple[str, ...],
) -> pd.DataFrame:
    training = outer_statistics[outer_statistics["source_zone"].astype(str).isin(training_zones)].copy()
    validation = outer_statistics[
        outer_statistics["source_zone"].astype(str).eq(validation_zone)
    ].reset_index(drop=True)
    if set(training["source_zone"].astype(str)) != set(training_zones):
        raise RuntimeError("监督基线内层训练源区集合失配")
    if set(validation["source_zone"].astype(str)) != {validation_zone}:
        raise RuntimeError("监督基线内层验证源区集合失配")
    rows = fixed_action_rows(
        contracts=contracts,
        validation_statistics=validation,
        outer_heldout_zone=outer_zone,
        validation_zone=validation_zone,
    )
    for baseline_id in ("CARTBestAction", "RidgePerActionRisk", "GBRPerActionRisk"):
        for complexity_rank, config in enumerate(grids[baseline_id]):
            rows.append(
                fitted_configuration_row(
                    contracts=contracts,
                    training_statistics=training,
                    validation_statistics=validation,
                    baseline_id=baseline_id,
                    config=config,
                    complexity_rank=complexity_rank,
                    outer_heldout_zone=outer_zone,
                    validation_zone=validation_zone,
                    training_zones=training_zones,
                )
            )
    scores = pd.DataFrame(rows)
    if len(scores) != EXPECTED_INNER_SCORE_ROWS:
        raise RuntimeError("监督基线内层验证分数行数失配")
    observed_counts = scores.groupby("baseline_id")["configuration_id"].nunique().to_dict()
    if observed_counts != EXPECTED_CONFIG_COUNTS:
        raise RuntimeError(f"监督基线内层配置数量失配: {observed_counts}")
    numeric = scores.loc[:, ["mean_errf", "mean_coverage_gap", "mean_tuwr"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("监督基线内层验证分数含非有限值")
    if scores.loc[:, ["baseline_id", "configuration_id"]].duplicated().any():
        raise RuntimeError("监督基线内层验证配置键重复")
    return scores.sort_values(["baseline_id", "complexity_rank", "configuration_id"], kind="mergesort").reset_index(drop=True)


def select_outer_configurations(
    *,
    contracts: Any,
    scores: pd.DataFrame,
    outer_zone: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for baseline_id in EXPECTED_CONFIG_COUNTS:
        baseline_scores = scores[scores["baseline_id"].astype(str).eq(baseline_id)].copy()
        selected = select_configuration(
            contracts=contracts,
            validation_scores=baseline_scores,
        )
        matching_config = baseline_scores[
            baseline_scores["configuration_id"].astype(str).eq(selected.selected_configuration_id)
        ]["configuration_json"].unique()
        if len(matching_config) != 1:
            raise RuntimeError("监督基线选中配置正文不唯一")
        rows.append(
            {
                "outer_heldout_zone": outer_zone,
                "baseline_id": baseline_id,
                **selected.as_record(),
                "selected_configuration_json": str(matching_config[0]),
                "source_validation_zone_count": 9,
                "outer_heldout_zone_fit_count": 0,
                "outer_heldout_zone_selection_count": 0,
            }
        )
    return pd.DataFrame(rows)


def process_outer_fold(outer_zone: str) -> dict[str, Any]:
    started = time.perf_counter()
    process = psutil.Process()
    contracts = load_frozen_contracts()
    grids = configuration_grid(contracts)
    statistics = pd.read_parquet(RUN_ROOT / "state_sufficient_statistics.parquet")
    outer_statistics = statistics[
        statistics["outer_heldout_zone"].astype(str).eq(outer_zone)
    ].copy()
    if outer_zone in set(outer_statistics["source_zone"].astype(str)):
        raise RuntimeError(f"监督基线外层折状态统计含持出区: {outer_zone}")
    folds = build_nested_source_zone_folds(contracts=contracts, outer_heldout_zone=outer_zone)
    score_frames: list[pd.DataFrame] = []
    recomputed_inner_folds = 0
    reused_inner_folds = 0
    for fold_index, fold in enumerate(folds, start=1):
        checkpoint = verified_inner_checkpoint(
            outer_zone=outer_zone,
            validation_zone=fold.inner_validation_zone,
            training_zones=fold.inner_training_zones,
        )
        was_reused = checkpoint is not None
        if checkpoint is None:
            checkpoint = compute_inner_scores(
                contracts=contracts,
                grids=grids,
                outer_statistics=outer_statistics,
                outer_zone=outer_zone,
                validation_zone=fold.inner_validation_zone,
                training_zones=fold.inner_training_zones,
            )
            write_inner_checkpoint(
                scores=checkpoint,
                outer_zone=outer_zone,
                validation_zone=fold.inner_validation_zone,
                training_zones=fold.inner_training_zones,
            )
            recomputed_inner_folds += 1
        else:
            reused_inner_folds += 1
        score_frames.append(checkpoint)
        print(
            json.dumps(
                {
                    "stage": "supervised_nested_selection",
                    "outer_heldout_zone": outer_zone,
                    "completed_inner_folds": fold_index,
                    "total_inner_folds": len(folds),
                    "reused": was_reused,
                    "pid": os.getpid(),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    scores = pd.concat(score_frames, ignore_index=True)
    if len(scores) != EXPECTED_OUTER_SCORE_ROWS:
        raise RuntimeError(f"监督基线外层折分数行数失配: {outer_zone}")
    selections = select_outer_configurations(
        contracts=contracts,
        scores=scores,
        outer_zone=outer_zone,
    )
    if len(selections) != len(EXPECTED_CONFIG_COUNTS):
        raise RuntimeError(f"监督基线外层折选中配置数量失配: {outer_zone}")
    outer_root = SELECTION_ROOT / f"outer_heldout={outer_zone}"
    scores_path = outer_root / "all_validation_scores.parquet"
    selections_path = outer_root / "selected_configurations.parquet"
    scores.to_parquet(scores_path, index=False, compression="zstd")
    selections.to_parquet(selections_path, index=False, compression="zstd")
    manifest = {
        "manifest_schema": "S06_SUPERVISED_OUTER_SELECTION_V1",
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": EXPECTED_CODE_COMMIT,
        "state_statistics_sha256": STATE_STATISTICS_SHA256,
        "supervised_gate_sha256": SUPERVISED_GATE_SHA256,
        "outer_heldout_zone": outer_zone,
        "inner_fold_count": len(folds),
        "recomputed_inner_fold_count": recomputed_inner_folds,
        "reused_inner_fold_count": reused_inner_folds,
        "score_row_count": len(scores),
        "selection_row_count": len(selections),
        "scores_sha256": sha256_file(scores_path),
        "selections_sha256": sha256_file(selections_path),
        "fallback_selection_count": int(selections["no_feasible_fallback_used"].sum()),
        "rss_gib": process.memory_info().rss / (1024.0**3),
        "elapsed_seconds": time.perf_counter() - started,
        "outer_heldout_zone_fit_count": int(selections["outer_heldout_zone_fit_count"].sum()),
        "outer_heldout_zone_selection_count": int(selections["outer_heldout_zone_selection_count"].sum()),
        "configuration_selection_executed": True,
        "performance_comparison_executed": False,
        "method_ranking_executed": False,
        "statistical_inference_executed": False,
        "paper_performance_claim_executed": False,
    }
    manifest_path = outer_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def record_exception(error: Exception) -> None:
    path = RUN_ROOT / "logs" / "supervised_selection_exception_ledger.csv"
    row = pd.DataFrame(
        [
            {
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "code_commit": EXPECTED_CODE_COMMIT,
            }
        ]
    )
    if path.exists():
        row = pd.concat([pd.read_csv(path, keep_default_na=False), row], ignore_index=True)
    row.to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    if git_output("rev-parse", "HEAD") != EXPECTED_CODE_COMMIT:
        raise RuntimeError("监督基线全量嵌套选择提交身份失配")
    if git_output("status", "--porcelain"):
        raise RuntimeError("监督基线全量嵌套选择要求权威代码工作树干净")
    contract_path = S06_ROOT / "configs" / "source_tuning_contract_v1.json"
    if sha256_file(contract_path) != SOURCE_TUNING_CONTRACT_SHA256:
        raise RuntimeError("监督基线全量嵌套选择合同SHA256失配")
    statistics_path = RUN_ROOT / "state_sufficient_statistics.parquet"
    if sha256_file(statistics_path) != STATE_STATISTICS_SHA256:
        raise RuntimeError("监督基线全量嵌套选择状态统计SHA256失配")
    gate_path = S06_ROOT / "qa" / "compact_supervised_training_gate_v1" / "qa_summary.json"
    if sha256_file(gate_path) != SUPERVISED_GATE_SHA256:
        raise RuntimeError("监督基线全量嵌套选择准入总门SHA256失配")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or gate.get("code_commit") != EXPECTED_CODE_COMMIT:
        raise RuntimeError("监督基线全量嵌套选择准入总门状态失配")
    if shutil.disk_usage(REVISION_ROOT).free / (1024.0**3) < MINIMUM_FREE_DISK_GIB:
        raise RuntimeError("监督基线全量嵌套选择磁盘安全门失败")
    contracts = load_frozen_contracts()
    zones = tuple(str(zone) for zone in contracts.protocol["datasets"]["gefcom2014"]["zones"])
    SELECTION_ROOT.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=WORKER_COUNT) as executor:
        future_by_zone = {
            executor.submit(process_outer_fold, zone): zone
            for zone in zones
        }
        for future in as_completed(future_by_zone):
            zone = future_by_zone[future]
            manifest = future.result()
            manifests.append(manifest)
            print(
                json.dumps(
                    {
                        "stage": "supervised_outer_fold_complete",
                        "outer_heldout_zone": zone,
                        "completed_outer_folds": len(manifests),
                        "total_outer_folds": len(zones),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    manifests.sort(key=lambda item: zone_number(str(item["outer_heldout_zone"])))
    score_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    for manifest in manifests:
        outer_root = SELECTION_ROOT / f"outer_heldout={manifest['outer_heldout_zone']}"
        scores_path = outer_root / "all_validation_scores.parquet"
        selections_path = outer_root / "selected_configurations.parquet"
        if sha256_file(scores_path) != manifest["scores_sha256"]:
            raise RuntimeError("监督基线外层分数文件SHA256失配")
        if sha256_file(selections_path) != manifest["selections_sha256"]:
            raise RuntimeError("监督基线外层选择文件SHA256失配")
        score_frames.append(pd.read_parquet(scores_path))
        selection_frames.append(pd.read_parquet(selections_path))
    all_scores = pd.concat(score_frames, ignore_index=True)
    all_selections = pd.concat(selection_frames, ignore_index=True)
    if len(all_scores) != EXPECTED_ROOT_SCORE_ROWS:
        raise RuntimeError("监督基线根级分数行数失配")
    if len(all_selections) != EXPECTED_ROOT_SELECTION_ROWS:
        raise RuntimeError("监督基线根级选择行数失配")
    score_counts = all_scores.groupby("baseline_id").size().to_dict()
    expected_score_counts = {
        baseline_id: count * 9 * 10
        for baseline_id, count in EXPECTED_CONFIG_COUNTS.items()
    }
    if score_counts != expected_score_counts:
        raise RuntimeError(f"监督基线根级分数分布失配: {score_counts}")
    if all_scores.groupby(["outer_heldout_zone", "inner_validation_zone"])["event_count"].nunique().max() != 1:
        raise RuntimeError("监督基线同一内层验证区配置事件数不一致")
    if all_scores["outer_heldout_zone_fit"].any() or all_scores["inner_validation_zone_fit"].any():
        raise RuntimeError("监督基线根级分数含持出区拟合")
    numeric = all_scores.loc[:, ["mean_errf", "mean_coverage_gap", "mean_tuwr"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("监督基线根级分数含非有限值")
    score_output_path = SELECTION_ROOT / "supervised_validation_scores.parquet"
    selection_output_path = SELECTION_ROOT / "supervised_selected_configurations.parquet"
    all_scores.to_parquet(score_output_path, index=False, compression="zstd")
    all_selections.to_parquet(selection_output_path, index=False, compression="zstd")
    resources = pd.DataFrame(
        [
            {
                "outer_heldout_zone": manifest["outer_heldout_zone"],
                "elapsed_seconds": manifest["elapsed_seconds"],
                "rss_gib": manifest["rss_gib"],
                "recomputed_inner_fold_count": manifest["recomputed_inner_fold_count"],
                "reused_inner_fold_count": manifest["reused_inner_fold_count"],
            }
            for manifest in manifests
        ]
    )
    resource_path = RUN_ROOT / "logs" / "supervised_selection_resource_samples.csv"
    resources.to_csv(resource_path, index=False, encoding="utf-8")
    fallback_by_baseline = (
        all_selections.groupby("baseline_id")["no_feasible_fallback_used"].sum().astype(int).to_dict()
    )
    summary = {
        "run_id": "S06_SUPERVISED_NESTED_SELECTION_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "code_commit": EXPECTED_CODE_COMMIT,
        "source_tuning_contract_sha256": SOURCE_TUNING_CONTRACT_SHA256,
        "state_statistics_sha256": STATE_STATISTICS_SHA256,
        "supervised_gate_sha256": SUPERVISED_GATE_SHA256,
        "worker_count": WORKER_COUNT,
        "outer_fold_count": len(zones),
        "inner_fold_count": 90,
        "configuration_counts": EXPECTED_CONFIG_COUNTS,
        "validation_score_row_count": len(all_scores),
        "selected_configuration_row_count": len(all_selections),
        "no_feasible_fallback_count_by_baseline": fallback_by_baseline,
        "outer_heldout_zone_fit_count": int(all_selections["outer_heldout_zone_fit_count"].sum()),
        "outer_heldout_zone_selection_count": int(all_selections["outer_heldout_zone_selection_count"].sum()),
        "validation_scores_sha256": sha256_file(score_output_path),
        "selected_configurations_sha256": sha256_file(selection_output_path),
        "resource_samples_sha256": sha256_file(resource_path),
        "elapsed_seconds": time.perf_counter() - started,
        "free_disk_gib_after": shutil.disk_usage(REVISION_ROOT).free / (1024.0**3),
        "configuration_selection_executed": True,
        "heldout_performance_read": False,
        "performance_comparison_executed": False,
        "method_ranking_executed": False,
        "statistical_inference_executed": False,
        "paper_performance_claim_executed": False,
        "new_scientific_adverse_evidence": False,
        "stop_triggered": False,
    }
    summary_path = SELECTION_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        record_exception(error)
        raise
