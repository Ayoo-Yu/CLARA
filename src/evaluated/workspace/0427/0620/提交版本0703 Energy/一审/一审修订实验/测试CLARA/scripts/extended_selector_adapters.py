"""扩展动作库下的 CART 与 LinUCB 兼容层。

该模块只允许替换候选动作列表。状态编码、CART 超参数、LinUCB 更新顺序、
严格延迟反馈和损失定义均复用 S09 的冻结实现。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


TEST_ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = TEST_ROOT.parent
S09_ROOT = REVISION_ROOT / "09_商业场站外部验证"
AUTHORITATIVE_CODE = REVISION_ROOT / "_权威代码" / "code"
sys.path.insert(0, str(AUTHORITATIVE_CODE))
sys.path.insert(0, str(S09_ROOT / "scripts"))

from baseline_compact_training import (  # noqa: E402
    COMPACT_STATE_FIELDS,
    _capped_classification_expansion,
    _encoded_compact_states,
)
from baseline_training import validate_frozen_selector_config  # noqa: E402
from clara_event_contract import FrozenContracts  # noqa: E402
from s09_linucb_batch_accelerator import (  # noqa: E402
    run_linucb_batch_equivalent,
)
import s09_minimum_external_core as core  # noqa: E402


@dataclass
class ExtendedCartSelector:
    """使用冻结状态编码器和可变动作列表的 CART 选择器。"""

    encoder: Any
    classifier: DecisionTreeClassifier
    actions: tuple[str, ...]
    configuration: dict[str, Any]

    def predict_actions(
        self, events: pd.DataFrame
    ) -> tuple[dict[str, str], dict[str, float]]:
        required = {"event_id", *COMPACT_STATE_FIELDS}
        missing = sorted(required - set(events.columns))
        if missing:
            raise ValueError(f"扩展 CART 预测事件缺少字段: {missing}")
        ordered = events.loc[:, ["event_id", *COMPACT_STATE_FIELDS]].copy()
        ordered["event_id"] = ordered["event_id"].astype(str)
        if ordered["event_id"].isna().any() or ordered["event_id"].duplicated().any():
            raise ValueError("扩展 CART 预测事件标识必须非空且唯一")
        ordered = ordered.sort_values("event_id", kind="mergesort").reset_index(drop=True)
        encoded = self.encoder.transform(ordered)
        feature_matrix = encoded.loc[:, self.encoder.feature_names].to_numpy(
            dtype=np.float64
        )
        selected = self.classifier.predict(feature_matrix).astype(str)
        invalid = sorted(set(selected) - set(self.actions))
        if invalid:
            raise RuntimeError(f"扩展 CART 产生候选库外动作: {invalid}")
        probabilities = self.classifier.predict_proba(feature_matrix)
        class_index = {
            str(value): index for index, value in enumerate(self.classifier.classes_)
        }
        event_ids = encoded["event_id"].astype(str).tolist()
        scores = {
            event_id: float(probabilities[row_index, class_index[action]])
            for row_index, (event_id, action) in enumerate(zip(event_ids, selected))
        }
        return dict(zip(event_ids, selected)), scores


def _validated_actions(actions: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(action) for action in actions)
    if len(normalized) < 2 or len(set(normalized)) != len(normalized):
        raise ValueError("扩展选择器动作列表必须至少含两个唯一动作")
    return normalized


def compact_statistics_from_facts_actions(
    facts: pd.DataFrame,
    *,
    source_zone: str,
    actions: Sequence[str],
) -> pd.DataFrame:
    """按冻结状态生成任意动作列表的 CART 充分统计量。"""

    action_order = _validated_actions(actions)
    required = set(COMPACT_STATE_FIELDS)
    for action in action_order:
        required.update(
            {
                f"{action}__errf",
                f"{action}__covered",
                f"{action}__tuwr_indicator",
            }
        )
    missing = sorted(required - set(facts.columns))
    if missing:
        raise ValueError(f"扩展 CART 训练事实缺少字段: {missing}")
    frame = facts.reset_index(drop=True).copy()
    errf = np.column_stack(
        [frame[f"{action}__errf"].to_numpy(dtype=np.float64) for action in action_order]
    )
    if not np.isfinite(errf).all() or np.any(errf < 0.0):
        raise ValueError("扩展 CART 训练事实包含非法 ERRF")
    best = np.argmin(errf, axis=1)
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(
        list(COMPACT_STATE_FIELDS), sort=True, dropna=False
    ):
        positions = group.index.to_numpy(dtype=np.int64)
        row: dict[str, Any] = {
            "source_zone": str(source_zone),
            **dict(zip(COMPACT_STATE_FIELDS, key)),
            "event_count": len(group),
        }
        for action_index, action in enumerate(action_order):
            tuwr = group[f"{action}__tuwr_indicator"]
            row[f"{action}__errf_sum"] = float(group[f"{action}__errf"].sum())
            row[f"{action}__covered_sum"] = int(
                group[f"{action}__covered"].sum()
            )
            row[f"{action}__tuwr_sum"] = float(tuwr.sum(skipna=True))
            row[f"{action}__tuwr_count"] = int(tuwr.notna().sum())
            row[f"{action}__best_count"] = int(
                (best[positions] == action_index).sum()
            )
        rows.append(row)
    if not rows:
        raise ValueError("扩展 CART 训练事实为空")
    return pd.DataFrame(rows)


def fit_cart_local_actions(
    facts: pd.DataFrame,
    *,
    farm: str,
    contracts: FrozenContracts,
    config: dict[str, Any],
    actions: Sequence[str],
) -> tuple[ExtendedCartSelector, pd.DataFrame, dict[str, Any]]:
    """用原 CART 算法在指定动作列表上训练本地选择器。"""

    action_order = _validated_actions(actions)
    fitting_config = dict(config)
    random_state = int(fitting_config.pop("random_state", 0))
    if random_state != 0:
        raise ValueError("扩展 CART 必须保持冻结随机种子零")
    validate_frozen_selector_config(
        contracts=contracts,
        baseline_id="CARTBestAction",
        config=fitting_config,
    )
    statistics = compact_statistics_from_facts_actions(
        facts,
        source_zone=farm,
        actions=action_order,
    )
    metric_columns = ["event_count"]
    for action in action_order:
        metric_columns.extend(
            [
                f"{action}__errf_sum",
                f"{action}__covered_sum",
                f"{action}__tuwr_sum",
                f"{action}__tuwr_count",
                f"{action}__best_count",
            ]
        )
    collapsed = (
        statistics.groupby(
            list(COMPACT_STATE_FIELDS), sort=True, as_index=False
        )[metric_columns]
        .sum()
        .reset_index(drop=True)
    )
    best_total = np.zeros(len(collapsed), dtype=np.int64)
    for action in action_order:
        counts = collapsed[f"{action}__best_count"].to_numpy(dtype=np.int64)
        if np.any(counts < 0):
            raise ValueError("扩展 CART 最优动作计数为负")
        best_total += counts
    event_count = collapsed["event_count"].to_numpy(dtype=np.int64)
    if not np.array_equal(best_total, event_count):
        raise RuntimeError("扩展 CART 最优动作计数不闭合")
    encoder, _, feature_matrix = _encoded_compact_states(
        contracts=contracts,
        collapsed=collapsed,
    )
    class_counts = np.column_stack(
        [
            collapsed[f"{action}__best_count"].to_numpy(dtype=np.int64)
            for action in action_order
        ]
    )
    features, labels, sample_weights = _capped_classification_expansion(
        feature_matrix=feature_matrix,
        class_counts_by_state=class_counts,
        class_labels=action_order,
        min_samples_leaf=int(fitting_config["min_samples_leaf"]),
        balanced=fitting_config["class_weight"] == "balanced",
    )
    classifier = DecisionTreeClassifier(
        max_depth=fitting_config["max_depth"],
        min_samples_leaf=int(fitting_config["min_samples_leaf"]),
        class_weight=None,
        random_state=0,
    )
    classifier.fit(features, labels, sample_weight=sample_weights)
    selector = ExtendedCartSelector(
        encoder=encoder,
        classifier=classifier,
        actions=action_order,
        configuration={
            "max_depth": fitting_config["max_depth"],
            "min_samples_leaf": int(fitting_config["min_samples_leaf"]),
            "class_weight": fitting_config["class_weight"],
            "random_state": 0,
        },
    )
    audit = {
        "actions": list(action_order),
        "state_statistics_row_count": len(statistics),
        "collapsed_state_count": len(collapsed),
        "expanded_training_row_count": int(features.shape[0]),
        "feature_count": int(features.shape[1]),
        "observed_classes": [str(value) for value in classifier.classes_],
        "tree_depth": int(classifier.get_depth()),
        "tree_leaf_count": int(classifier.get_n_leaves()),
        "configuration": selector.configuration,
    }
    return selector, statistics, audit


def predict_cart_local_actions(
    facts: pd.DataFrame,
    selector: ExtendedCartSelector,
) -> tuple[np.ndarray, np.ndarray]:
    """按输入事件顺序返回扩展 CART 的动作和置信分数。"""

    actions, scores = selector.predict_actions(
        facts.loc[:, ["event_id", *COMPACT_STATE_FIELDS]].copy()
    )
    event_ids = facts["event_id"].astype(str)
    selected = event_ids.map(actions)
    score = event_ids.map(scores)
    if selected.isna().any() or score.isna().any():
        raise RuntimeError("扩展 CART 外部事件预测缺失")
    return selected.astype(str).to_numpy(), score.to_numpy(dtype=np.float64)


def run_linucb_local_actions(
    *,
    final_facts: pd.DataFrame,
    adaptation_facts: pd.DataFrame,
    contracts: FrozenContracts,
    exploration_alpha: float,
    l2_regularization: float,
    actions: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """在指定动作列表上调用 S09 等价批量 LinUCB。"""

    action_order = _validated_actions(actions)
    original_actions = core.ACTIONS
    try:
        core.ACTIONS = action_order
        selected, audit = run_linucb_batch_equivalent(
            final_facts=final_facts,
            adaptation_facts=adaptation_facts,
            contracts=contracts,
            exploration_alpha=float(exploration_alpha),
            l2_regularization=float(l2_regularization),
        )
    finally:
        core.ACTIONS = original_actions
    invalid = sorted(set(selected) - set(action_order))
    if invalid:
        raise RuntimeError(f"扩展 LinUCB 产生候选库外动作: {invalid}")
    audit = dict(audit)
    audit["actions"] = list(action_order)
    return selected, audit
