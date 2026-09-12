from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
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
sys.path.insert(0, str(CODE_ROOT))

from clara_event_contract import load_frozen_contracts, sha256_file  # noqa: E402


EXPECTED_CODE_COMMIT = "980fad5174840883318b15a225957f70df8928f6"
SOURCE_TUNING_CONTRACT_SHA256 = "9a352cba41d7e68911626da950540d1d8506f2f4f63afb72bd7b28cd473c7e14"
SOURCE_FACT_ROOT_MANIFEST_SHA256 = "30a0f86d6f775c7c32a3654117807dfa9acabc8187c2bdefeb8440e1466ef9e7"
EXPECTED_STREAM_COUNT = 2880
EXPECTED_COMPLETE_EVENT_COUNT = 110705760
EXPECTED_THRESHOLD_ROW_COUNT = 6600
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


def main() -> None:
    started = time.perf_counter()
    if git_output("rev-parse", "HEAD") != EXPECTED_CODE_COMMIT:
        raise RuntimeError("外层折宽度阈值提交身份失配")
    if git_output("status", "--porcelain"):
        raise RuntimeError("外层折宽度阈值要求权威代码工作树干净")
    contract_path = S06_ROOT / "configs" / "source_tuning_contract_v1.json"
    if sha256_file(contract_path) != SOURCE_TUNING_CONTRACT_SHA256:
        raise RuntimeError("源区嵌套选择合同SHA256失配")
    root_manifest_path = RUN_ROOT / "source_fact_root_manifest.json"
    if sha256_file(root_manifest_path) != SOURCE_FACT_ROOT_MANIFEST_SHA256:
        raise RuntimeError("全量源域事实根清单SHA256失配")
    if shutil.disk_usage(REVISION_ROOT).free / (1024.0**3) < MINIMUM_FREE_DISK_GIB:
        raise RuntimeError("外层折宽度阈值磁盘安全门失败")

    contracts = load_frozen_contracts()
    zones = tuple(str(zone) for zone in contracts.protocol["datasets"]["gefcom2014"]["zones"])
    predictors = tuple(str(value) for value in contracts.protocol["predictor_contract"]["predictors"])
    horizon_groups = tuple(str(value) for value in contracts.protocol["horizon_group_contract"]["groups"])
    coverages = tuple(float(value) for value in contracts.coverages)
    seeds = (0, 1, 2)
    ledger = pd.read_csv(RUN_ROOT / "checkpoints" / "source_fact_ledger.csv", keep_default_na=False)
    completed = ledger[ledger["status"].astype(str) == "COMPLETED"].copy()
    if len(completed) != EXPECTED_STREAM_COUNT:
        raise RuntimeError("外层折宽度阈值输入流数量失配")
    completed["_zone_number"] = completed["zone"].map(zone_number)
    completed = completed.sort_values(
        ["predictor", "_zone_number", "horizon", "seed"],
        kind="mergesort",
    ).drop(columns="_zone_number")

    arrays: dict[tuple[str, str, str, float, int], list[np.ndarray]] = {}
    loaded_event_count = 0
    for index, row in enumerate(completed.itertuples(index=False), start=1):
        frame = pd.read_parquet(
            Path(str(row.facts_file)),
            columns=["target_coverage", "horizon_group", "raw_width_value"],
        )
        if len(frame) != int(row.complete_case_event_count):
            raise RuntimeError(f"宽度阈值输入事实行数失配: {row.stream_id}")
        groups = set(frame["horizon_group"].astype(str))
        if len(groups) != 1:
            raise RuntimeError(f"逐流时长组不唯一: {row.stream_id}")
        horizon_group = next(iter(groups))
        for coverage, part in frame.groupby("target_coverage", sort=True):
            values = part["raw_width_value"].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all() or np.any(values < 0.0):
                raise RuntimeError(f"宽度阈值输入含非法值: {row.stream_id}")
            key = (
                str(row.zone),
                str(row.predictor),
                horizon_group,
                float(coverage),
                int(row.seed),
            )
            arrays.setdefault(key, []).append(values)
        loaded_event_count += len(frame)
        if index % 240 == 0:
            print(
                json.dumps(
                    {
                        "stage": "load_width_arrays",
                        "completed": index,
                        "total": len(completed),
                        "loaded_event_count": loaded_event_count,
                        "rss_gib": psutil.Process().memory_info().rss / (1024.0**3),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if loaded_event_count != EXPECTED_COMPLETE_EVENT_COUNT:
        raise RuntimeError("宽度阈值完整案例事件总数失配")

    consolidated: dict[tuple[str, str, str, float, int], np.ndarray] = {}
    for key, chunks in arrays.items():
        consolidated[key] = np.concatenate(chunks)
    del arrays

    rows: list[dict[str, Any]] = []
    for heldout_index, heldout_zone in enumerate(zones, start=1):
        source_zones = tuple(zone for zone in zones if zone != heldout_zone)
        if len(source_zones) != 9:
            raise RuntimeError("每个外层折必须恰有九个源区")
        for predictor in predictors:
            for horizon_group in horizon_groups:
                for coverage in coverages:
                    for seed in seeds:
                        chunks = [
                            consolidated[(zone, predictor, horizon_group, coverage, seed)]
                            for zone in source_zones
                        ]
                        values = np.concatenate(chunks)
                        q33, q67 = np.quantile(values, [0.33, 0.67], method="linear")
                        rows.append(
                            {
                                "outer_heldout_zone": heldout_zone,
                                "predictor": predictor,
                                "horizon_group": horizon_group,
                                "target_coverage": coverage,
                                "seed": seed,
                                "raw_width_q33": float(q33),
                                "raw_width_q67": float(q67),
                                "threshold_event_count": len(values),
                                "threshold_source_zone_count": len(source_zones),
                                "heldout_zone_in_threshold": False,
                            }
                        )
        print(
            json.dumps(
                {
                    "stage": "fit_outer_fold_thresholds",
                    "completed_outer_folds": heldout_index,
                    "total_outer_folds": len(zones),
                    "threshold_rows": len(rows),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    thresholds = pd.DataFrame(rows).sort_values(
        ["outer_heldout_zone", "predictor", "horizon_group", "target_coverage", "seed"],
        key=lambda series: series.map(zone_number) if series.name == "outer_heldout_zone" else series,
        kind="mergesort",
    ).reset_index(drop=True)
    if len(thresholds) != EXPECTED_THRESHOLD_ROW_COUNT:
        raise RuntimeError("外层折宽度阈值行数失配")
    if thresholds.loc[:, ["outer_heldout_zone", "predictor", "horizon_group", "target_coverage", "seed"]].duplicated().any():
        raise RuntimeError("外层折宽度阈值键重复")
    numeric = thresholds.loc[:, ["raw_width_q33", "raw_width_q67"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (thresholds["raw_width_q33"] < 0.0).any():
        raise RuntimeError("外层折宽度阈值含非法数值")
    if (thresholds["raw_width_q33"] > thresholds["raw_width_q67"]).any():
        raise RuntimeError("外层折宽度分位数次序错误")
    if not thresholds["threshold_source_zone_count"].eq(9).all():
        raise RuntimeError("外层折宽度阈值源区数量失配")
    if thresholds["heldout_zone_in_threshold"].any():
        raise RuntimeError("外层持出区进入宽度阈值")

    output_path = RUN_ROOT / "fold_width_thresholds.parquet"
    thresholds.to_parquet(output_path, index=False, compression="zstd")
    elapsed = time.perf_counter() - started
    summary = {
        "run_id": "S06_OUTER_FOLD_WIDTH_THRESHOLDS_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "code_commit": EXPECTED_CODE_COMMIT,
        "source_tuning_contract_sha256": SOURCE_TUNING_CONTRACT_SHA256,
        "source_fact_root_manifest_sha256": SOURCE_FACT_ROOT_MANIFEST_SHA256,
        "input_stream_count": len(completed),
        "input_complete_case_event_count": loaded_event_count,
        "outer_fold_count": len(zones),
        "threshold_row_count": len(thresholds),
        "threshold_rows_per_outer_fold": int(len(thresholds) / len(zones)),
        "source_zone_count_per_outer_fold_values": sorted(
            int(value) for value in thresholds["threshold_source_zone_count"].unique()
        ),
        "heldout_zone_in_threshold_count": int(thresholds["heldout_zone_in_threshold"].sum()),
        "nonfinite_threshold_count": int((~np.isfinite(numeric)).sum()),
        "inverted_threshold_count": int((thresholds["raw_width_q33"] > thresholds["raw_width_q67"]).sum()),
        "threshold_file": str(output_path),
        "threshold_file_sha256": sha256_file(output_path),
        "elapsed_seconds": elapsed,
        "peak_rss_gib_after_consolidation": psutil.Process().memory_info().rss / (1024.0**3),
        "free_disk_gib_after": shutil.disk_usage(REVISION_ROOT).free / (1024.0**3),
        "heldout_performance_read": False,
        "performance_comparison_executed": False,
        "method_ranking_executed": False,
        "statistical_inference_executed": False,
        "paper_performance_claim_executed": False,
        "new_scientific_adverse_evidence": False,
        "stop_triggered": False,
    }
    summary_path = RUN_ROOT / "fold_width_thresholds_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
