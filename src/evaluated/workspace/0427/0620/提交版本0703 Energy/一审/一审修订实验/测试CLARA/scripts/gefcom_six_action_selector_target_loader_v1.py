"""Fail-closed target-stream loader for the formal GEFCom six-action replay.

The source-fit loader intentionally rejects the outer held-out zone.  This module
implements the complementary target-only path: it derives every path from the
frozen axes, verifies the completed endpoint root seal and its leaf chain, applies
the same strict-maturity filter, and constructs the held-out-fold raw-width state.
It never exposes a non-formal/path-override mode.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping as ABCMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

import gefcom_six_action_selector_core_v1 as core


SCHEMA = "TEST_CLARA_GEFCOM_6A_CAUSAL_TARGET_STREAM_V1"
_TARGET_CAPABILITY = object()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require_sha(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RuntimeError(f"{label}不是小写SHA256")
    return text


@dataclass(frozen=True)
class VerifiedCausalTargetAudit(ABCMapping[str, Any]):
    """Unforgeable in-process capability produced only by this formal loader."""

    _payload: str
    _capability: object

    def _decoded(self) -> dict[str, Any]:
        return json.loads(self._payload)

    def __getitem__(self, key: str) -> Any:
        return self._decoded()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._decoded())

    def __len__(self) -> int:
        return len(self._decoded())

    def verified_by_formal_loader(self) -> bool:
        return self._capability is _TARGET_CAPABILITY


def formal_target_paths(
    *, evaluation_zone: str, predictor: str, horizon: int, seed: int
) -> dict[str, Path]:
    if (
        str(evaluation_zone) not in core.FORMAL_ZONES
        or str(predictor) not in core.FORMAL_PREDICTORS
        or int(horizon) not in core.FORMAL_HORIZONS
        or int(seed) not in range(3)
    ):
        raise RuntimeError("target loader轴越出冻结正式范围")
    return core.formal_causal_source_paths(
        source_zone=str(evaluation_zone),
        predictor=str(predictor),
        horizon=int(horizon),
        seed=int(seed),
    )


def target_feedback_release_audit(
    facts: pd.DataFrame, feedback: pd.DataFrame
) -> dict[str, int | bool]:
    """Audit delayed-feedback eligibility without filtering target decisions.

    Every target event remains in ``facts``.  Eligible four-action rows only define
    feedback that a later issue-time coordinator may consume.
    """

    core._require_columns(facts, ("event_id",), "target评价事件")
    core._require_columns(
        feedback,
        ("event_id", "action", "eligible_by_strict_time_rule"),
        "target延迟反馈",
    )
    core._validate_event_ids(facts, "target评价事件")
    target_ids = facts["event_id"].astype(str)
    trace = feedback.copy()
    trace["event_id"] = trace["event_id"].astype(str)
    trace["action"] = trace["action"].astype(str)
    eligible = trace[trace["eligible_by_strict_time_rule"].astype(bool)].copy()
    if eligible[["event_id", "action"]].duplicated().any():
        raise RuntimeError("target严格成熟反馈键重复")
    action_sets = eligible.groupby("event_id", sort=False)["action"].agg(
        lambda values: tuple(sorted(set(values)))
    )
    expected = tuple(sorted(core.SIX_ACTIONS[:4]))
    if not action_sets.map(lambda values: values == expected).all():
        raise RuntimeError("target严格成熟反馈存在非四动作完整事件")
    mature_ids = set(action_sets.index.astype(str)) & set(target_ids)
    if len(eligible[eligible["event_id"].isin(mature_ids)]) != len(mature_ids) * 4:
        raise RuntimeError("target严格成熟四动作反馈不守恒")
    pending = len(facts) - len(mature_ids)
    if pending < 0:
        raise RuntimeError("target pending反馈计数非法")
    return {
        "target_event_count": len(facts),
        "historically_releasable_feedback_event_count": len(mature_ids),
        "terminal_or_not_yet_releasable_event_count": pending,
        "target_event_filter_applied": False,
        "strict_maturity_used_for_feedback_release_only": True,
    }


def assert_target_unit_conservation(
    predictor_frames: Mapping[str, pd.DataFrame],
    *,
    expected_event_count: int,
) -> dict[str, Any]:
    """Close one zone×seed×horizon target unit against its endpoint root row."""

    if set(predictor_frames) != set(core.FORMAL_PREDICTORS):
        raise RuntimeError("target unit必须精确包含四预测器")
    frames: list[pd.DataFrame] = []
    cold_count = 0
    for predictor in core.FORMAL_PREDICTORS:
        frame = predictor_frames[predictor]
        core._require_columns(
            frame,
            ("event_id", "predictor", "rolling_state", "target_coverage"),
            "target unit predictor frame",
        )
        if set(frame["predictor"].astype(str)) != {predictor}:
            raise RuntimeError("target unit predictor key/content crosswire")
        if set(frame["target_coverage"].astype(float)) != set(core.FORMAL_COVERAGES):
            raise RuntimeError("target unit predictor十一覆盖率不完整")
        core._validate_event_ids(frame, f"target unit/{predictor}")
        cold_count += int(frame["rolling_state"].astype(str).eq("cold_start").sum())
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    core._validate_event_ids(combined, "target unit四预测器合并")
    if len(combined) != int(expected_event_count):
        raise RuntimeError(
            f"target unit事件守恒失败: observed={len(combined)}, "
            f"expected={int(expected_event_count)}"
        )
    if cold_count <= 0:
        raise RuntimeError("target unit错误删除全部cold_start评价事件")
    return {
        "status": "PASS",
        "event_count": len(combined),
        "cold_start_event_count": cold_count,
        "event_signature": core.event_id_signature(
            combined["event_id"], include_canonical_sha256=True
        ),
    }


def _validate_source_anchor(
    *,
    source_anchor_path: Path,
    addon_manifest: Mapping[str, Any],
    root_manifest: Mapping[str, Any],
    stream_id: str,
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    if _sha256(source_anchor_path) != str(
        addon_manifest.get("source_anchor_audit_json_sha256")
    ):
        raise RuntimeError("target endpoint source-anchor SHA链断裂")
    anchor = _load_json(source_anchor_path)
    records = anchor.get("records")
    migration_sha = str(
        root_manifest["identities"]["endpoint_dependency_hashes"][
            "migration_full_sha256_manifest"
        ]
    )
    if (
        anchor.get("schema")
        != "TEST_CLARA_GEFCOM_6A_ENDPOINT_SOURCE_ANCHOR_AUDIT_V1"
        or anchor.get("status") != "PASS"
        or str(anchor.get("migration_manifest_sha256")) != migration_sha
        or not isinstance(records, list)
        or int(anchor.get("checked_file_count", -1)) != 32
        or len(records) != 32
        or len({str(row.get("relative_path")) for row in records}) != 32
    ):
        raise RuntimeError("target endpoint source-anchor未完整闭合")
    current = [row for row in records if str(row.get("stream_id")) == stream_id]
    if len(current) != 8:
        raise RuntimeError("target当前流迁移anchor记录必须精确为8")
    normalized = [
        {**row, "_path": str(row.get("relative_path", "")).replace("\\", "/")}
        for row in current
    ]
    anchored = {
        "source_fact": paths["source_fact"],
        "source_fact_manifest": paths["source_fact_manifest"],
        "bundle_manifest": paths["bundle_manifest"],
        "event_core": paths["event_core"],
        "candidate_intervals": paths["candidate_intervals"],
        "feedback_trace": paths["feedback_trace"],
    }
    for name, path in anchored.items():
        is_source = name.startswith("source_fact")
        matches = [
            row
            for row in normalized
            if row["_path"].endswith("/" + path.name)
            and (("/source_facts/" in row["_path"]) == is_source)
            and (("/bundles/" in row["_path"]) == (not is_source))
        ]
        if len(matches) != 1 or str(matches[0].get("sha256")) != _sha256(path):
            raise RuntimeError(f"target迁移anchor与当前文件失配: {name}")
    return {
        "source_anchor_audit_sha256": _sha256(source_anchor_path),
        "migration_manifest_sha256": migration_sha,
    }


def load_causal_target_stream(
    *,
    evaluation_zone: str,
    predictor: str,
    horizon: int,
    seed: int,
    expected_endpoint_root_sha256: str,
) -> tuple[pd.DataFrame, VerifiedCausalTargetAudit]:
    """Load one official target stream with endpoint/root/leaf/TOCTOU gates."""

    zone = str(evaluation_zone)
    predictor_id = str(predictor)
    horizon_id = int(horizon)
    seed_id = int(seed)
    root_sha = _require_sha(expected_endpoint_root_sha256, "endpoint root")
    paths = formal_target_paths(
        evaluation_zone=zone,
        predictor=predictor_id,
        horizon=horizon_id,
        seed=seed_id,
    )
    bundle_dir = paths["candidate_bundle_dir"]
    fact_path = paths["source_fact"]
    addon_path = paths["endpoint_addon"]
    root_path = paths["endpoint_root_manifest"]
    threshold_path = paths["width_thresholds"]
    required_paths = {
        "source_fact": fact_path,
        "source_fact_manifest": fact_path.parent / "manifest.json",
        "bundle_manifest": bundle_dir / "manifest.json",
        "event_core": bundle_dir / "event_core.parquet",
        "candidate_intervals": bundle_dir / "candidate_intervals.parquet",
        "feedback_trace": bundle_dir / "feedback_trace.parquet",
        "endpoint_addon": addon_path,
        "endpoint_addon_manifest": addon_path.parent / "manifest.json",
        "endpoint_source_anchor": addon_path.parent / "source_anchor_audit.json",
        "endpoint_root_manifest": root_path,
        "width_thresholds": threshold_path,
    }
    missing = {name: str(path) for name, path in required_paths.items() if not path.is_file()}
    if missing:
        raise FileNotFoundError(f"target正式输入缺失: {missing}")

    root_manifest, root_units = core.validate_endpoint_root_seal(
        root_path, expected_endpoint_root_sha256=root_sha
    )
    unit_id = f"{zone}__seed{seed_id}__H{horizon_id:02d}"
    if unit_id not in root_units:
        raise RuntimeError("target endpoint root缺当前unit")
    fact_manifest = _load_json(required_paths["source_fact_manifest"])
    bundle_manifest = _load_json(required_paths["bundle_manifest"])
    addon_manifest = _load_json(required_paths["endpoint_addon_manifest"])
    core._assert_endpoint_unit_root_identity(
        addon_manifest,
        root_manifest,
        source_zone=zone,
        horizon=horizon_id,
        seed=seed_id,
    )
    if str(root_units[unit_id].get("unit_manifest_sha256")) != _sha256(
        required_paths["endpoint_addon_manifest"]
    ):
        raise RuntimeError("target endpoint unit manifest未被root精确pin")
    expected_identity = {
        "predictor": predictor_id,
        "zone": zone,
        "horizon": horizon_id,
        "seed": seed_id,
    }
    for manifest, label in ((fact_manifest, "S06"), (bundle_manifest, "S03")):
        if manifest.get("status") != "COMPLETE_VALIDATED":
            raise RuntimeError(f"target {label} manifest未通过")
        for field, expected in expected_identity.items():
            observed = manifest.get(field)
            observed = int(observed) if field in {"horizon", "seed"} else str(observed)
            if observed != expected:
                raise RuntimeError(f"target {label} manifest轴失配: {field}")
    if str(fact_manifest.get("input_bundle_manifest_sha256")) != _sha256(
        required_paths["bundle_manifest"]
    ):
        raise RuntimeError("target S06→S03 bundle身份链断裂")
    dependency_hashes = addon_manifest["identities"]["endpoint_dependency_hashes"]
    if str(dependency_hashes.get("fold_width_thresholds")) != _sha256(threshold_path):
        raise RuntimeError("target宽度阈值与endpoint冻结依赖失配")
    stream_id = f"{predictor_id}-H{horizon_id:02d}-{zone}-S{seed_id}"
    anchor_identity = _validate_source_anchor(
        source_anchor_path=required_paths["endpoint_source_anchor"],
        addon_manifest=addon_manifest,
        root_manifest=root_manifest,
        stream_id=stream_id,
        paths=required_paths,
    )
    expected_leaf_sha = {
        "source_fact": str(fact_manifest["facts_sha256"]),
        "event_core": str(bundle_manifest["event_core_sha256"]),
        "candidate_intervals": str(bundle_manifest["candidate_intervals_sha256"]),
        "feedback_trace": str(bundle_manifest["feedback_trace_sha256"]),
        "endpoint_addon": str(addon_manifest["endpoint_addons_parquet_sha256"]),
        "width_thresholds": str(dependency_hashes["fold_width_thresholds"]),
    }
    before_sha = {name: _sha256(required_paths[name]) for name in expected_leaf_sha}
    if before_sha != expected_leaf_sha:
        mismatch = {
            name: {"expected": expected_leaf_sha[name], "observed": before_sha[name]}
            for name in expected_leaf_sha
            if expected_leaf_sha[name] != before_sha[name]
        }
        raise RuntimeError(f"target leaf SHA失配: {mismatch}")

    facts = pd.read_parquet(fact_path)
    core._validate_event_ids(facts, "target S06源事件")
    facts["event_id"] = facts["event_id"].astype(str)
    raw_event_count = len(facts)
    if raw_event_count != int(fact_manifest["complete_case_event_count"]):
        raise RuntimeError("target S06行数与manifest失配")
    target_membership = pd.read_parquet(
        addon_path,
        columns=[
            "event_id",
            "zone_or_farm",
            "predictor",
            "seed",
            "horizon_steps",
        ],
    )
    target_membership["event_id"] = target_membership["event_id"].astype(str)
    target_membership = target_membership[
        target_membership["zone_or_farm"].astype(str).eq(zone)
        & target_membership["predictor"].astype(str).eq(predictor_id)
        & target_membership["seed"].astype(int).eq(seed_id)
        & target_membership["horizon_steps"].astype(int).eq(horizon_id)
    ].copy()
    if target_membership.empty or target_membership["event_id"].duplicated().any():
        raise RuntimeError("target endpoint root成员集为空或重复")
    endpoint_target_ids = set(target_membership["event_id"])
    facts = facts[facts["event_id"].isin(endpoint_target_ids)].copy()
    if len(facts) != len(endpoint_target_ids):
        raise RuntimeError("target S06未完整覆盖endpoint root成员集")
    feedback = pd.read_parquet(
        required_paths["feedback_trace"],
        columns=["event_id", "action", "eligible_by_strict_time_rule"],
    )
    release_audit = target_feedback_release_audit(facts, feedback)
    target_event_ids = set(facts["event_id"])

    event_core = pd.read_parquet(
        required_paths["event_core"], columns=["event_id", "target", "base_center"]
    )
    event_core["event_id"] = event_core["event_id"].astype(str)
    event_core = event_core[event_core["event_id"].isin(target_event_ids)].copy()
    if len(event_core) != len(facts) or event_core["event_id"].duplicated().any():
        raise RuntimeError("target成熟event_core映射不完整")
    facts = facts.merge(
        event_core.rename(
            columns={"target": "target_after_maturity", "base_center": "schedule_proxy"}
        ),
        on="event_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    thresholds = pd.read_parquet(threshold_path)
    threshold_keys = ["predictor", "horizon_group", "target_coverage", "seed"]
    selected_thresholds = core.validated_fold_width_thresholds(
        thresholds,
        facts,
        outer_heldout_zone=zone,
        predictor=predictor_id,
        seed=seed_id,
    )
    facts = facts.merge(
        selected_thresholds,
        on=threshold_keys,
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if facts[["raw_width_q33", "raw_width_q67"]].isna().any().any():
        raise RuntimeError("target持出折宽度阈值映射不完整")
    if facts["heldout_zone_in_threshold"].astype(bool).any():
        raise RuntimeError("target宽度阈值混入持出区")
    raw_width = facts["raw_width_value"].to_numpy(dtype=np.float64)
    facts["raw_width_state"] = np.where(
        raw_width <= facts["raw_width_q33"].to_numpy(dtype=np.float64),
        "narrow",
        np.where(
            raw_width <= facts["raw_width_q67"].to_numpy(dtype=np.float64),
            "medium",
            "wide",
        ),
    )

    addon_columns = [
        "event_id",
        "zone_or_farm",
        "predictor",
        "seed",
        "horizon_steps",
        "target_coverage",
        *(
            f"{action}__{field}"
            for action in core.SIX_ACTIONS[4:]
            for field in (
                "candidate_lower",
                "candidate_upper",
                "covered",
                "tuwr_indicator",
                "ard_value",
            )
        ),
    ]
    addons = pd.read_parquet(addon_path, columns=addon_columns)
    addons["event_id"] = addons["event_id"].astype(str)
    addons = addons[
        addons["event_id"].isin(target_event_ids)
        & addons["zone_or_farm"].astype(str).eq(zone)
        & addons["predictor"].astype(str).eq(predictor_id)
        & addons["seed"].astype(int).eq(seed_id)
        & addons["horizon_steps"].astype(int).eq(horizon_id)
    ].copy()
    if len(addons) != len(facts) or addons["event_id"].duplicated().any():
        raise RuntimeError("target新增动作端点与成熟事件集失配")
    facts = facts.merge(
        addons[["event_id", *[column for column in addon_columns if "__" in column]]],
        on="event_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    required = {
        "event_id",
        "zone_or_farm",
        "predictor",
        "seed",
        "horizon_steps",
        "target_coverage",
        "issue_timestamp",
        "label_timestamp",
        "label_available_timestamp",
        "target_after_maturity",
        "schedule_proxy",
        *core.STATE_FIELDS,
    }
    for action in core.SIX_ACTIONS:
        required.update(
            {
                f"{action}__candidate_lower",
                f"{action}__candidate_upper",
                f"{action}__covered",
                f"{action}__tuwr_indicator",
                f"{action}__ard_value",
            }
        )
    core._require_columns(facts, required, "target六动作正式事件")
    if set(facts["target_coverage"].astype(float)) != set(core.FORMAL_COVERAGES):
        raise RuntimeError("target十一覆盖率不完整")
    if set(facts["zone_or_farm"].astype(str)) != {zone}:
        raise RuntimeError("target事件混入其他区")
    core._validate_event_ids(facts, "target最终事件")
    after_sha = {name: _sha256(required_paths[name]) for name in expected_leaf_sha}
    if after_sha != before_sha or _sha256(root_path) != root_sha:
        raise RuntimeError("target loader检测到读取期间输入TOCTOU漂移")
    signature = core.event_id_signature(
        facts["event_id"], include_canonical_sha256=True
    )
    audit = {
        "schema": SCHEMA,
        "status": "PASS",
        "evaluation_zone": zone,
        "predictor": predictor_id,
        "horizon": horizon_id,
        "seed": seed_id,
        "partition_id": f"{zone}__seed{seed_id}__H{horizon_id:02d}__{predictor_id}",
        "raw_s06_event_count": raw_event_count,
        "endpoint_root_target_event_count": len(facts),
        "s06_events_outside_endpoint_root_count": raw_event_count - len(facts),
        "cold_start_event_count": int(
            facts["rolling_state"].astype(str).eq("cold_start").sum()
        ),
        **release_audit,
        "strict_mature_action_count": 4,
        "heldout_zone_used_only_as_target": True,
        "raw_width_threshold_outer_fold": zone,
        "endpoint_root_manifest_sha256": root_sha,
        "endpoint_unit_manifest_sha256": _sha256(
            required_paths["endpoint_addon_manifest"]
        ),
        "endpoint_unit_expected_event_count": int(root_units[unit_id]["event_count"]),
        "event_signature": signature,
        "leaf_sha256": before_sha,
        "leaf_hashes_verified": True,
        "toctou_recheck_passed": True,
        **anchor_identity,
    }
    payload = json.dumps(
        audit, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return facts.reset_index(drop=True), VerifiedCausalTargetAudit(
        _payload=payload, _capability=_TARGET_CAPABILITY
    )


__all__ = [
    "SCHEMA",
    "VerifiedCausalTargetAudit",
    "formal_target_paths",
    "target_feedback_release_audit",
    "assert_target_unit_conservation",
    "load_causal_target_stream",
]
