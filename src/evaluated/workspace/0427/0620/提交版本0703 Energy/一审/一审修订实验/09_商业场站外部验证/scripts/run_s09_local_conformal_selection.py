"""使用外部适配区选择两个场站的本地共形配置。"""
from __future__ import annotations

import json
import math
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


S09_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = Path(__file__).resolve().parents[2]
AUTHORITATIVE_CODE = REVISION_ROOT / "_权威代码" / "code"
sys.path.insert(0, str(AUTHORITATIVE_CODE))
sys.path.insert(0, str(S09_ROOT / "scripts"))

from clara_event_contract import load_frozen_contracts  # noqa: E402
from clara_errf import ErrfTheta  # noqa: E402
import s09_minimum_external_core as core  # noqa: E402


CONFIG_PATH = S09_ROOT / "configs" / "s09_minimum_external_evaluation_v1.json"
GATE_PATH = S09_ROOT / "configs" / "s09_minimum_external_full_run_gate_v1.json"
RUNNER_PATH = Path(__file__).resolve()
CORE_PATH = S09_ROOT / "scripts" / "s09_minimum_external_core.py"
RESULT_ROOT = S09_ROOT / "results_raw" / "local_conformal_selection_v1"
UNIT_ROOT = RESULT_ROOT / "units"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def unit_id(farm: str, predictor: str, horizon: int) -> str:
    return f"{farm}-{predictor}-H{int(horizon):02d}-S0"


def reliability_counts(
    covered: np.ndarray,
    *,
    target_coverage: float,
    cadence_minutes: float,
    window_hours: float = 168.0,
) -> tuple[int, int]:
    values = np.asarray(covered, dtype=np.float64)
    window = int(math.ceil(float(window_hours) * 60.0 / float(cadence_minutes)))
    if len(values) < window:
        return 0, 0
    rolling = np.convolve(values, np.ones(window), mode="valid") / float(window)
    tolerance = 1.96 * math.sqrt(
        float(target_coverage) * (1.0 - float(target_coverage)) / float(window)
    )
    return int((rolling < float(target_coverage) - tolerance).sum()), len(rolling)


def completed_unit(path: Path, config_sha256: str, core_sha256: str) -> bool:
    manifest_path = path / "manifest.json"
    summary_path = path / "stream_summaries.parquet"
    if not manifest_path.exists() or not summary_path.exists():
        return False
    manifest = core.load_json(manifest_path)
    return (
        manifest.get("status") == "PASS"
        and manifest.get("config_sha256") == config_sha256
        and manifest.get("core_sha256") == core_sha256
        and manifest.get("stream_summaries_sha256") == core.sha256_file(summary_path)
        and int(manifest.get("summary_row_count", -1)) == 19 * 11
    )


def run_unit(task: tuple[str, str, int, str, str]) -> dict[str, Any]:
    farm, predictor, horizon, config_sha256, core_sha256 = task
    identifier = unit_id(farm, predictor, horizon)
    output = UNIT_ROOT / identifier
    if completed_unit(output, config_sha256, core_sha256):
        return {"unit_id": identifier, "status": "REUSED"}
    if output.exists():
        raise RuntimeError(f"发现未封存的本地共形选择单元，拒绝覆盖: {output}")
    output.mkdir(parents=True, exist_ok=False)
    started_at = utc_now()
    try:
        config = core.load_json(CONFIG_PATH)
        contracts = load_frozen_contracts()
        theta = ErrfTheta()
        facts, bundle_audit = core.load_fact_bundle(
            root=core.ADAPTATION_ROOT,
            farm=farm,
            predictor=predictor,
            horizon=horizon,
            seed=0,
            contracts=contracts,
            theta=theta,
        )
        paths = core.bundle_paths(
            root=core.ADAPTATION_ROOT,
            farm=farm,
            predictor=predictor,
            horizon=horizon,
            seed=0,
        )
        base_predictions = pd.read_parquet(paths["base_predictions"])
        bundle_manifest = core.load_json(paths["manifest"])
        records = core.conformal_configuration_records(contracts)
        rows: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        for record in records:
            for target_coverage in config["scope"]["target_coverages"]:
                coverage_facts = facts[
                    facts["target_coverage"].astype(float).eq(float(target_coverage))
                ].copy()
                coverage_facts = coverage_facts.sort_values(
                    ["issue_timestamp", "event_id"], kind="mergesort"
                ).reset_index(drop=True)
                decisions, audit = core.run_conformal_method(
                    base_predictions=base_predictions,
                    contracts=contracts,
                    dataset_id=str(bundle_manifest["dataset_id"]),
                    farm=farm,
                    predictor=predictor,
                    seed=0,
                    target_coverage=float(target_coverage),
                    family=str(record["family"]),
                    configuration=dict(record["configuration"]),
                )
                arrays = core.endpoint_metrics_for_direct_interval(
                    coverage_facts,
                    decisions,
                    theta=theta,
                )
                cadence_values = coverage_facts[
                    "nominal_cadence_minutes"
                ].astype(float).unique()
                if len(cadence_values) != 1:
                    raise RuntimeError(f"适配单元名义时间步长不唯一: {identifier}")
                tuwr_sum, tuwr_count = reliability_counts(
                    arrays["covered"],
                    target_coverage=float(target_coverage),
                    cadence_minutes=float(cadence_values[0]),
                )
                rows.append(
                    {
                        "farm": farm,
                        "predictor": predictor,
                        "horizon_steps": int(horizon),
                        "seed": 0,
                        "target_coverage": float(target_coverage),
                        "configuration_id": str(record["configuration_id"]),
                        "family": str(record["family"]),
                        "configuration_json": str(record["configuration_json"]),
                        "complexity_rank": int(record["complexity_rank"]),
                        "event_count": len(coverage_facts),
                        "errf_sum": float(np.asarray(arrays["errf"]).sum()),
                        "covered_sum": int(np.asarray(arrays["covered"]).sum()),
                        "coverage_target_sum": float(target_coverage)
                        * len(coverage_facts),
                        "tuwr_sum": int(tuwr_sum),
                        "tuwr_count": int(tuwr_count),
                    }
                )
                audit_rows.append(
                    {
                        "configuration_id": str(record["configuration_id"]),
                        "target_coverage": float(target_coverage),
                        "test_event_count": int(audit["test_event_count"]),
                        "future_feedback_violation_count": int(
                            audit["future_feedback_violation_count"]
                        ),
                        "within_issue_feedback_use_count": int(
                            audit["within_issue_feedback_use_count"]
                        ),
                        "invalid_interval_count": int(audit["invalid_interval_count"]),
                        "out_of_bounds_count": int(audit["out_of_bounds_count"]),
                        "final_feedback_drain": bool(audit["drain_final_feedback"]),
                    }
                )
        summaries = pd.DataFrame(rows)
        audits = pd.DataFrame(audit_rows)
        if len(summaries) != 19 * 11 or len(audits) != 19 * 11:
            raise RuntimeError(f"本地共形选择单元行数失配: {identifier}")
        if (
            audits["future_feedback_violation_count"].astype(int).ne(0).any()
            or audits["within_issue_feedback_use_count"].astype(int).ne(0).any()
            or audits["invalid_interval_count"].astype(int).ne(0).any()
            or audits["out_of_bounds_count"].astype(int).ne(0).any()
            or audits["final_feedback_drain"].astype(bool).any()
        ):
            raise RuntimeError(f"本地共形选择单元因果或区间审计失败: {identifier}")
        core.write_parquet_atomic(summaries, output / "stream_summaries.parquet")
        core.write_parquet_atomic(audits, output / "causal_audits.parquet")
        manifest = {
            "schema": "S09_LOCAL_CONFORMAL_SELECTION_UNIT_V1",
            "status": "PASS",
            "unit_id": identifier,
            "farm": farm,
            "predictor": predictor,
            "horizon_steps": int(horizon),
            "seed": 0,
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "config_sha256": config_sha256,
            "core_sha256": core_sha256,
            "runner_sha256": core.sha256_file(RUNNER_PATH),
            "adaptation_bundle_manifest_sha256": core.sha256_file(paths["manifest"]),
            "adaptation_base_predictions_sha256": core.sha256_file(
                paths["base_predictions"]
            ),
            "adaptation_fact_content_sha256": bundle_audit["fact_content_sha256"],
            "configuration_count": 19,
            "coverage_count": 11,
            "summary_row_count": len(summaries),
            "causal_audit_row_count": len(audits),
            "stream_summaries_sha256": core.sha256_file(
                output / "stream_summaries.parquet"
            ),
            "causal_audits_sha256": core.sha256_file(output / "causal_audits.parquet"),
            "final_test_performance_read": False,
        }
        core.write_json_atomic(manifest, output / "manifest.json")
        return {"unit_id": identifier, "status": "COMPLETED"}
    except Exception as exc:
        core.write_json_atomic(
            {
                "schema": "S09_LOCAL_CONFORMAL_SELECTION_EXCEPTION_V1",
                "status": "FAILED_PRESERVED",
                "unit_id": identifier,
                "failed_at_utc": utc_now(),
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "final_test_performance_read": False,
            },
            output / "exception.json",
        )
        raise


def main() -> int:
    config = core.load_json(CONFIG_PATH)
    gate = core.load_json(GATE_PATH)
    config_sha256 = core.sha256_file(CONFIG_PATH)
    core_sha256 = core.sha256_file(CORE_PATH)
    if gate.get("status") != "AUTHORIZED_FOR_FULL_TECHNICAL_EXECUTION":
        raise RuntimeError("S09 全量运行安全门未授权")
    if gate["minimum_protocol"]["sha256"] != config_sha256:
        raise RuntimeError("S09 全量运行安全门与最小协议身份失配")
    if bool(config.get("performance_read_allowed")):
        raise RuntimeError("适配选择阶段禁止开放最终性能读取")
    if shutil.disk_usage(S09_ROOT.drive + "\\").free < 30 * 1024**3:
        raise RuntimeError("S09 本地共形选择磁盘安全门失败")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    UNIT_ROOT.mkdir(parents=True, exist_ok=True)
    if (RESULT_ROOT / "root_manifest.json").exists():
        raise RuntimeError("S09 本地共形选择根清单已存在，拒绝重复运行")

    tasks = [
        (farm, predictor, int(horizon), config_sha256, core_sha256)
        for farm in config["scope"]["farms"]
        for predictor in config["scope"]["predictors"]
        for horizon in config["scope"]["horizon_steps"]
    ]
    completed = 0
    with ProcessPoolExecutor(max_workers=4) as executor:
        future_map = {executor.submit(run_unit, task): task for task in tasks}
        for future in as_completed(future_map):
            result = future.result()
            completed += 1
            print(
                json.dumps(
                    {"progress": f"{completed}/{len(tasks)}", **result},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    unit_summaries: list[pd.DataFrame] = []
    unit_records: list[dict[str, Any]] = []
    for task in tasks:
        farm, predictor, horizon, _, _ = task
        output = UNIT_ROOT / unit_id(farm, predictor, horizon)
        manifest = core.load_json(output / "manifest.json")
        if not completed_unit(output, config_sha256, core_sha256):
            raise RuntimeError(f"本地共形选择单元封存失败: {output.name}")
        unit_summaries.append(pd.read_parquet(output / "stream_summaries.parquet"))
        unit_records.append(
            {
                "unit_id": output.name,
                "manifest_sha256": core.sha256_file(output / "manifest.json"),
                "stream_summaries_sha256": manifest["stream_summaries_sha256"],
                "causal_audits_sha256": manifest["causal_audits_sha256"],
            }
        )
    summaries = pd.concat(unit_summaries, ignore_index=True)
    selected: dict[str, Any] = {}
    for farm in config["scope"]["farms"]:
        chosen = core.select_local_conformal_configuration(
            stream_summaries=summaries[summaries["farm"].astype(str).eq(str(farm))],
            coverage_gap_lower=float(
                config["tuned_single_conformal"]["local_feasibility"][
                    "coverage_gap_lower"
                ]
            ),
            tuwr_upper=float(
                config["tuned_single_conformal"]["local_feasibility"][
                    "tuwr_upper"
                ]
            ),
        )
        selected[str(farm)] = chosen
    core.write_parquet_atomic(summaries, RESULT_ROOT / "stream_summaries.parquet")
    core.write_json_atomic(selected, RESULT_ROOT / "selected_configurations.json")
    root_manifest = {
        "schema": "S09_LOCAL_CONFORMAL_SELECTION_ROOT_MANIFEST_V1",
        "status": "COMPLETE_PENDING_INDEPENDENT_QA",
        "completed_at_utc": utc_now(),
        "config_sha256": config_sha256,
        "gate_sha256": core.sha256_file(GATE_PATH),
        "core_sha256": core_sha256,
        "runner_sha256": core.sha256_file(RUNNER_PATH),
        "unit_count": len(tasks),
        "summary_row_count": len(summaries),
        "selected_farm_count": len(selected),
        "stream_summaries_sha256": core.sha256_file(
            RESULT_ROOT / "stream_summaries.parquet"
        ),
        "selected_configurations_sha256": core.sha256_file(
            RESULT_ROOT / "selected_configurations.json"
        ),
        "units": unit_records,
        "final_test_performance_read": False,
    }
    core.write_json_atomic(root_manifest, RESULT_ROOT / "root_manifest.json")
    print(json.dumps(root_manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
