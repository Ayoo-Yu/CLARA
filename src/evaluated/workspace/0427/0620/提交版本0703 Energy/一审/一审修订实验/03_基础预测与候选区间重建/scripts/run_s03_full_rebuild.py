from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


步骤目录 = Path(__file__).resolve().parents[1]
修订根目录 = 步骤目录.parent
权威仓库 = 修订根目录 / "_权威代码"
代码目录 = 权威仓库 / "code"
sys.path.insert(0, str(代码目录))

from common import (  # noqa: E402
    canonical_json_sha256,
    load_json,
    save_json,
    sha256_file,
)
from run_base_grid import _cache_mismatches  # noqa: E402
from run_base_predictor import run_base_prediction  # noqa: E402
from run_candidate_bundle import (  # noqa: E402
    run_candidate_bundle,
    validate_candidate_bundle_cache,
)


class 致命流错误(RuntimeError):
    """表示继续运行会破坏协议身份或科学有效性的错误。"""


@dataclass(frozen=True)
class 流定义:
    predictor: str
    zone: str
    horizon: int
    seed: int

    @property
    def stream_id(self) -> str:
        return (
            f"{self.predictor}-H{self.horizon:02d}-"
            f"{self.zone}-S{self.seed}"
        )


def 当前时间() -> str:
    return datetime.now(timezone.utc).isoformat()


def 当前提交() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=权威仓库,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def 工作树状态() -> str:
    return subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=权威仓库,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def 文件总字节(目录: Path) -> int:
    return sum(路径.stat().st_size for 路径 in 目录.rglob("*") if 路径.is_file())


def 设置线程限制(线程数: int) -> None:
    值 = str(int(线程数))
    for 名称 in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[名称] = 值


def 限制深度学习线程(线程数: int) -> None:
    try:
        import torch

        torch.set_num_threads(int(线程数))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except ImportError:
        pass


def 路径位于(路径: Path, 根目录: Path) -> bool:
    try:
        路径.resolve().relative_to(根目录.resolve())
        return True
    except ValueError:
        return False


def 隔离不完整目录(运行目录: Path, 结果根目录: Path, 阶段: str) -> str | None:
    if not 运行目录.exists() or not any(运行目录.iterdir()):
        return None
    if not 路径位于(运行目录, 结果根目录) or 运行目录.resolve() == 结果根目录.resolve():
        raise 致命流错误(f"拒绝移动结果根目录外的不完整输出: {运行目录}")
    隔离根目录 = 结果根目录 / "quarantine" / 阶段
    隔离根目录.mkdir(parents=True, exist_ok=True)
    目标 = 隔离根目录 / (
        f"{运行目录.name}_{datetime.now().strftime('%Y%m%dT%H%M%S')}_"
        f"{uuid.uuid4().hex[:8]}"
    )
    if not 路径位于(目标, 结果根目录):
        raise 致命流错误(f"隔离目标超出结果根目录: {目标}")
    shutil.move(str(运行目录), str(目标))
    return str(目标)


def 基础目录(配置: dict[str, Any], 流: 流定义) -> Path:
    return Path(配置["results_root"]) / "base" / 流.predictor / 流.stream_id


def 候选目录(配置: dict[str, Any], 流: 流定义) -> Path:
    return Path(配置["results_root"]) / "bundles" / 流.predictor / 流.stream_id


def 基础配置(配置: dict[str, Any], 流: 流定义) -> dict[str, Any]:
    return {
        "protocol_id": 配置["protocol_id"],
        "protocol_version": 配置["protocol_version"],
        "protocol_sha256": 配置["protocol_sha256"],
        "dataset_id": 配置["dataset_id"],
        "predictor": 流.predictor,
        "zone": 流.zone,
        "seed": int(流.seed),
        "horizon": int(流.horizon),
        "data_path": 配置["data_paths"][流.zone],
        "split": 配置["split"],
        "save_splits": 配置["save_splits"],
        "require_clean_code": True,
        "results_dir": str(基础目录(配置, 流)),
    }


def 候选配置(配置: dict[str, Any], 流: 流定义) -> dict[str, Any]:
    return {
        "protocol_id": 配置["protocol_id"],
        "protocol_version": 配置["protocol_version"],
        "protocol_sha256": 配置["protocol_sha256"],
        "baseline_registry_sha256": 配置["baseline_registry_sha256"],
        "output_contract_sha256": 配置["output_contract_sha256"],
        "dataset_id": 配置["dataset_id"],
        "predictor": 流.predictor,
        "zone": 流.zone,
        "seed": int(流.seed),
        "horizon": int(流.horizon),
        "base_predictions_path": str(
            基础目录(配置, 流) / "base_predictions.parquet"
        ),
        "target_coverages": 配置["target_coverages"],
        "methods": 配置["methods"],
        "results_dir": str(候选目录(配置, 流)),
        "compute_performance_metrics": False,
        "require_clean_code": True,
    }


def 提取裁剪诊断(清单: dict[str, Any]) -> list[dict[str, Any]]:
    候选 = pd.read_parquet(
        清单["candidate_file"],
        columns=(
            "target_coverage",
            "action",
            "candidate_clip_applied",
            "preclip_lower",
            "preclip_upper",
        ),
    )
    行集合: list[dict[str, Any]] = []
    for (覆盖率, 动作), 分组 in 候选.groupby(
        ["target_coverage", "action"], sort=True
    ):
        下越界 = pd.to_numeric(分组["preclip_lower"], errors="coerce") < 0.0
        上越界 = pd.to_numeric(分组["preclip_upper"], errors="coerce") > 1.0
        已裁剪 = 分组["candidate_clip_applied"].astype(bool)
        行集合.append(
            {
                "target_coverage": float(覆盖率),
                "action": str(动作),
                "rows": int(len(分组)),
                "clipped_rows": int(已裁剪.sum()),
                "clip_rate": float(已裁剪.mean()),
                "preclip_lower_below_zero_rows": int(下越界.sum()),
                "preclip_upper_above_one_rows": int(上越界.sum()),
            }
        )
    return 行集合


def 运行基础流(载荷: dict[str, Any]) -> dict[str, Any]:
    配置 = 载荷["config"]
    流 = 流定义(**载荷["stream"])
    设置线程限制(int(配置["runtime"]["worker_thread_limit"]))
    限制深度学习线程(int(配置["runtime"]["worker_thread_limit"]))
    结果根目录 = Path(配置["results_root"])
    运行配置 = 基础配置(配置, 流)
    运行目录 = Path(运行配置["results_dir"])
    清单路径 = 运行目录 / "manifest.json"
    开始 = time.perf_counter()
    隔离目录 = None

    if 清单路径.exists():
        差异 = _cache_mismatches(清单路径, 运行配置)
        if 差异:
            raise 致命流错误(f"基础缓存身份核验失败: {流.stream_id}; {差异}")
        清单 = load_json(清单路径)
        状态 = "SKIPPED_VALID_CACHE"
    else:
        隔离目录 = 隔离不完整目录(运行目录, 结果根目录, "base")
        运行目录.mkdir(parents=True, exist_ok=True)
        save_json(运行配置, 运行目录 / "run_config.json")
        清单 = run_base_prediction(运行配置)
        差异 = _cache_mismatches(清单路径, 运行配置)
        if 差异:
            raise 致命流错误(f"新基础输出核验失败: {流.stream_id}; {差异}")
        状态 = "COMPLETED"

    重排 = 清单["quantile_rearrangement"]
    return {
        "stage": "base",
        "stream_id": 流.stream_id,
        **asdict(流),
        "status": 状态,
        "elapsed_seconds": time.perf_counter() - 开始,
        "output_bytes": 文件总字节(运行目录),
        "manifest_sha256": sha256_file(清单路径),
        "output_sha256": 清单["output_sha256"],
        "output_rows": int(清单["n_output"]),
        "pre_rearrangement_crossing_rows": int(
            重排["rows_with_crossing_before"]
        ),
        "pre_rearrangement_crossing_rate": float(
            重排["crossing_row_rate_before"]
        ),
        "mean_absolute_adjustment": float(
            重排["mean_absolute_adjustment"]
        ),
        "max_absolute_adjustment": float(
            重排["max_absolute_adjustment"]
        ),
        "post_rearrangement_crossing_rows": int(
            重排["rows_with_crossing_after"]
        ),
        "quarantined_partial_dir": 隔离目录,
        "performance_comparison_performed": False,
        "completed_utc": 当前时间(),
    }


def 运行候选流(载荷: dict[str, Any]) -> dict[str, Any]:
    配置 = 载荷["config"]
    流 = 流定义(**载荷["stream"])
    设置线程限制(int(配置["runtime"]["worker_thread_limit"]))
    限制深度学习线程(int(配置["runtime"]["worker_thread_limit"]))
    结果根目录 = Path(配置["results_root"])
    运行配置 = 候选配置(配置, 流)
    基础清单路径 = Path(运行配置["base_predictions_path"]).parent / "manifest.json"
    if not 基础清单路径.exists():
        raise RuntimeError(f"候选流缺少基础依赖: {流.stream_id}")
    基础差异 = _cache_mismatches(基础清单路径, 基础配置(配置, 流))
    if 基础差异:
        raise 致命流错误(f"候选流基础依赖身份失败: {流.stream_id}; {基础差异}")

    运行目录 = Path(运行配置["results_dir"])
    清单路径 = 运行目录 / "manifest.json"
    开始 = time.perf_counter()
    隔离目录 = None
    if 清单路径.exists():
        差异 = validate_candidate_bundle_cache(清单路径, 运行配置)
        if 差异:
            raise 致命流错误(f"候选缓存身份核验失败: {流.stream_id}; {差异}")
        清单 = load_json(清单路径)
        状态 = "SKIPPED_VALID_CACHE"
    else:
        隔离目录 = 隔离不完整目录(运行目录, 结果根目录, "bundle")
        运行目录.parent.mkdir(parents=True, exist_ok=True)
        清单 = run_candidate_bundle(运行配置)
        差异 = validate_candidate_bundle_cache(清单路径, 运行配置)
        if 差异:
            raise 致命流错误(f"新候选包核验失败: {流.stream_id}; {差异}")
        状态 = "COMPLETED"

    验证 = 清单["validations"]
    致命计数 = sum(
        int(验证[字段])
        for 字段 in (
            "duplicate_event_ids",
            "duplicate_event_action_rows",
            "incomplete_action_events",
            "nonfinite_candidate_rows",
            "reversed_interval_rows",
            "out_of_bounds_rows",
            "strict_feedback_violations",
        )
    )
    if 致命计数:
        raise 致命流错误(f"候选包科学有效性失败: {流.stream_id}; {验证}")
    return {
        "stage": "bundle",
        "stream_id": 流.stream_id,
        **asdict(流),
        "status": 状态,
        "elapsed_seconds": time.perf_counter() - 开始,
        "output_bytes": 文件总字节(运行目录),
        "manifest_sha256": sha256_file(清单路径),
        "event_rows": int(清单["event_rows"]),
        "candidate_rows": int(清单["candidate_rows"]),
        "feedback_rows": int(清单["feedback_rows"]),
        "validation_failure_count": 致命计数,
        "clip_diagnostics": 提取裁剪诊断(清单),
        "quarantined_partial_dir": 隔离目录,
        "performance_comparison_performed": False,
        "completed_utc": 当前时间(),
    }


def 追加JSONL(路径: Path, 记录: dict[str, Any]) -> None:
    路径.parent.mkdir(parents=True, exist_ok=True)
    with 路径.open("a", encoding="utf-8", newline="") as 文件:
        文件.write(json.dumps(记录, ensure_ascii=False, separators=(",", ":")))
        文件.write("\n")


def 写入表格(路径: Path, 行集合: list[dict[str, Any]]) -> None:
    if not 行集合:
        return
    字段 = sorted({键 for 行 in 行集合 for 键 in 行 if 键 != "clip_diagnostics"})
    with 路径.open("w", encoding="utf-8", newline="") as 文件:
        写入器 = csv.DictWriter(文件, fieldnames=字段, extrasaction="ignore")
        写入器.writeheader()
        写入器.writerows(行集合)


def 读取JSONL(路径: Path) -> list[dict[str, Any]]:
    if not 路径.exists():
        return []
    结果: list[dict[str, Any]] = []
    with 路径.open("r", encoding="utf-8") as 文件:
        for 行号, 行 in enumerate(文件, start=1):
            文本 = 行.strip()
            if not 文本:
                continue
            try:
                结果.append(json.loads(文本))
            except json.JSONDecodeError as 异常:
                raise RuntimeError(f"JSONL第{行号}行损坏: {路径}") from 异常
    return 结果


def 资源监控(停止事件: threading.Event, 结果根目录: Path, 输出路径: Path) -> None:
    输出路径.parent.mkdir(parents=True, exist_ok=True)
    新文件 = not 输出路径.exists() or 输出路径.stat().st_size == 0
    try:
        import psutil
    except ImportError:
        psutil = None
    with 输出路径.open("a", encoding="utf-8", newline="") as 文件:
        字段 = (
            "timestamp_utc",
            "disk_free_bytes",
            "memory_available_bytes",
            "memory_percent",
            "child_processes",
            "child_rss_bytes",
        )
        写入器 = csv.DictWriter(文件, fieldnames=字段)
        if 新文件:
            写入器.writeheader()
        while True:
            磁盘 = shutil.disk_usage(结果根目录.anchor or 结果根目录)
            内存可用 = None
            内存比例 = None
            子进程数 = None
            子进程内存 = None
            if psutil is not None:
                内存 = psutil.virtual_memory()
                内存可用 = int(内存.available)
                内存比例 = float(内存.percent)
                子进程 = psutil.Process().children(recursive=True)
                子进程数 = len(子进程)
                子进程内存 = 0
                for 进程 in 子进程:
                    try:
                        子进程内存 += int(进程.memory_info().rss)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            写入器.writerow(
                {
                    "timestamp_utc": 当前时间(),
                    "disk_free_bytes": int(磁盘.free),
                    "memory_available_bytes": 内存可用,
                    "memory_percent": 内存比例,
                    "child_processes": 子进程数,
                    "child_rss_bytes": 子进程内存,
                }
            )
            文件.flush()
            if 停止事件.wait(2.0):
                break


def 构造全部流(配置: dict[str, Any]) -> list[流定义]:
    return [
        流定义(预测器, 区域, int(时域), int(种子))
        for 预测器 in 配置["predictors"]
        for 区域 in 配置["zones"]
        for 时域 in 配置["horizons"]
        for 种子 in 配置["seeds"]
    ]


def 应用选择(
    全部流: list[流定义],
    选择路径: str | None,
) -> tuple[list[流定义], dict[str, Any] | None]:
    if not 选择路径:
        return 全部流, None
    选择 = load_json(选择路径)
    请求 = {流定义(**项目) for 项目 in 选择["streams"]}
    全部集合 = set(全部流)
    未知 = 请求 - 全部集合
    if 未知:
        raise ValueError(f"选择文件包含网格外的流: {sorted(流.stream_id for 流 in 未知)}")
    return [流 for 流 in 全部流 if 流 in 请求], 选择


def 核验启动门(配置: dict[str, Any]) -> None:
    if 当前提交() != 配置["expected_code_commit"]:
        raise 致命流错误("权威代码提交与全量配置不一致")
    if 工作树状态():
        raise 致命流错误("权威代码工作树不干净")
    if 配置.get("compute_performance_metrics"):
        raise 致命流错误("S03运行不得执行性能比较")
    if set(配置["methods"]) != {"SplitCF", "ACI", "AgACI", "EnbPI"}:
        raise 致命流错误("候选动作集合与冻结协议不一致")
    for 区域 in 配置["zones"]:
        路径 = Path(配置["data_paths"][区域])
        if not 路径.exists():
            raise 致命流错误(f"输入文件缺失: {区域}")
        if sha256_file(路径).upper() != 配置["data_sha256"][区域].upper():
            raise 致命流错误(f"输入SHA256不一致: {区域}")


def 核验或建立根清单(
    配置: dict[str, Any],
    源配置路径: Path,
    选择: dict[str, Any] | None,
    流集合: list[流定义],
) -> None:
    结果根目录 = Path(配置["results_root"])
    结果根目录.mkdir(parents=True, exist_ok=True)
    清单路径 = 结果根目录 / "run_root_manifest.json"
    身份 = {
        "manifest_schema": "S03_FULL_REBUILD_ROOT_V1",
        "run_id": 配置["run_id"],
        "effective_config_sha256": canonical_json_sha256(配置),
        "source_config_sha256": sha256_file(源配置路径),
        "code_commit": 当前提交(),
        "selection_sha256": canonical_json_sha256(选择) if 选择 else None,
        "selected_streams": len(流集合),
        "performance_comparison_performed": False,
    }
    if 清单路径.exists():
        现有 = load_json(清单路径)
        差异 = [键 for 键, 值 in 身份.items() if 现有.get(键) != 值]
        if 差异:
            raise 致命流错误(f"结果根清单身份不一致: {差异}")
        return
    身份["created_utc"] = 当前时间()
    save_json(配置, 结果根目录 / "effective_run_config.json")
    if 选择:
        save_json(选择, 结果根目录 / "selection_snapshot.json")
    save_json(身份, 清单路径)


def 执行阶段(
    阶段: str,
    配置: dict[str, Any],
    流集合: list[流定义],
    工作函数: Callable[[dict[str, Any]], dict[str, Any]],
    最大并发: int,
) -> dict[str, Any]:
    结果根目录 = Path(配置["results_root"])
    检查点路径 = 结果根目录 / "checkpoints" / f"{阶段}_events.jsonl"
    错误路径 = 结果根目录 / "logs" / "error_ledger.jsonl"
    最大重试 = int(配置["runtime"]["max_retries_per_stream"])
    最低磁盘字节 = int(
        float(配置["runtime"]["minimum_free_disk_gib"]) * 1024 ** 3
    )
    系统失败阈值 = int(配置["runtime"]["systemic_failure_threshold"])
    队列 = deque((流, 1) for 流 in 流集合)
    记录集合: list[dict[str, Any]] = []
    终止失败: list[dict[str, Any]] = []
    开始 = time.perf_counter()
    已处理 = 0
    上次汇报 = 开始
    上下文 = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=最大并发, mp_context=上下文) as 执行器:
        运行中: dict[Any, tuple[流定义, int]] = {}
        while 队列 or 运行中:
            while 队列 and len(运行中) < 最大并发:
                磁盘 = shutil.disk_usage(结果根目录.anchor or 结果根目录)
                if 磁盘.free < 最低磁盘字节:
                    raise 致命流错误(
                        f"磁盘可用空间低于{最低磁盘字节 / 1024 ** 3:.2f} GiB安全值"
                    )
                流, 尝试 = 队列.popleft()
                载荷 = {"config": 配置, "stream": asdict(流)}
                任务 = 执行器.submit(工作函数, 载荷)
                运行中[任务] = (流, 尝试)

            if not 运行中:
                continue
            完成集合, _ = wait(
                tuple(运行中), timeout=5.0, return_when=FIRST_COMPLETED
            )
            if not 完成集合:
                现在 = time.perf_counter()
                if 现在 - 上次汇报 >= 60.0:
                    print(
                        f"[{阶段}] completed={已处理}/{len(流集合)} "
                        f"inflight={len(运行中)} queued={len(队列)}",
                        flush=True,
                    )
                    上次汇报 = 现在
                continue

            for 任务 in 完成集合:
                流, 尝试 = 运行中.pop(任务)
                try:
                    记录 = 任务.result()
                    记录["attempt"] = 尝试
                    记录集合.append(记录)
                    追加JSONL(检查点路径, 记录)
                    已处理 += 1
                except BaseException as 异常:
                    致命 = isinstance(异常, 致命流错误)
                    错误记录 = {
                        "timestamp_utc": 当前时间(),
                        "stage": 阶段,
                        "stream_id": 流.stream_id,
                        **asdict(流),
                        "attempt": 尝试,
                        "fatal": 致命,
                        "error_type": type(异常).__name__,
                        "error_message": str(异常),
                        "traceback": "".join(
                            traceback.format_exception(type(异常), 异常, 异常.__traceback__)
                        ),
                    }
                    追加JSONL(错误路径, 错误记录)
                    if 致命:
                        for 其他任务 in 运行中:
                            其他任务.cancel()
                        raise 致命流错误(
                            f"{阶段}流{流.stream_id}触发致命错误: {异常}"
                        ) from 异常
                    if 尝试 <= 最大重试:
                        队列.append((流, 尝试 + 1))
                    else:
                        终止失败.append(错误记录)
                        已处理 += 1
                        if len(终止失败) >= 系统失败阈值:
                            for 其他任务 in 运行中:
                                其他任务.cancel()
                            raise 致命流错误(
                                f"{阶段}累计{len(终止失败)}条终止失败，达到系统阈值"
                            )

                现在 = time.perf_counter()
                if 已处理 % 10 == 0 or 现在 - 上次汇报 >= 60.0:
                    print(
                        f"[{阶段}] completed={已处理}/{len(流集合)} "
                        f"terminal_failures={len(终止失败)} "
                        f"inflight={len(运行中)} queued={len(队列)}",
                        flush=True,
                    )
                    上次汇报 = 现在

    历史记录 = 读取JSONL(检查点路径)
    最新记录: dict[str, dict[str, Any]] = {}
    for 记录 in 历史记录:
        最新记录[记录["stream_id"]] = 记录
    当前最新 = [最新记录[流.stream_id] for 流 in 流集合 if 流.stream_id in 最新记录]
    写入表格(结果根目录 / f"{阶段}_stream_ledger.csv", 当前最新)

    if 阶段 == "bundle":
        裁剪行: list[dict[str, Any]] = []
        for 记录 in 当前最新:
            for 诊断 in 记录.get("clip_diagnostics", []):
                裁剪行.append(
                    {
                        "stream_id": 记录["stream_id"],
                        "predictor": 记录["predictor"],
                        "zone": 记录["zone"],
                        "horizon": 记录["horizon"],
                        "seed": 记录["seed"],
                        **诊断,
                    }
                )
        写入表格(结果根目录 / "candidate_clipping_diagnostics.csv", 裁剪行)

    汇总 = {
        "stage": 阶段,
        "status": "PASS" if not 终止失败 else "INCOMPLETE",
        "selected_streams": len(流集合),
        "records_this_invocation": len(记录集合),
        "completed_this_invocation": sum(
            记录["status"] == "COMPLETED" for 记录 in 记录集合
        ),
        "validated_cache_skips_this_invocation": sum(
            记录["status"] == "SKIPPED_VALID_CACHE" for 记录 in 记录集合
        ),
        "terminal_failures": len(终止失败),
        "elapsed_seconds": time.perf_counter() - 开始,
        "runner_script_sha256": sha256_file(Path(__file__)),
        "performance_comparison_performed": False,
    }
    save_json(汇总, 结果根目录 / f"{阶段}_stage_summary.json")
    追加JSONL(结果根目录 / "logs" / "stage_invocations.jsonl", 汇总)
    return 汇总


def 主函数() -> int:
    解析器 = argparse.ArgumentParser()
    解析器.add_argument("--config", required=True)
    解析器.add_argument("--stage", choices=("base", "bundle", "all"), default="all")
    解析器.add_argument("--selection")
    解析器.add_argument("--results-root-override")
    解析器.add_argument("--base-workers", type=int)
    解析器.add_argument("--bundle-workers", type=int)
    参数 = 解析器.parse_args()

    源配置路径 = Path(参数.config)
    配置 = load_json(源配置路径)
    if 参数.results_root_override:
        配置["results_root"] = str(Path(参数.results_root_override))
    if 参数.base_workers is not None:
        配置["runtime"]["base_max_workers"] = int(参数.base_workers)
    if 参数.bundle_workers is not None:
        配置["runtime"]["bundle_max_workers"] = int(参数.bundle_workers)
    设置线程限制(int(配置["runtime"]["worker_thread_limit"]))

    核验启动门(配置)
    全部流 = 构造全部流(配置)
    流集合, 选择 = 应用选择(全部流, 参数.selection)
    if not 流集合:
        raise ValueError("运行选择为空")
    核验或建立根清单(配置, 源配置路径, 选择, 流集合)
    结果根目录 = Path(配置["results_root"])
    停止事件 = threading.Event()
    监控线程 = threading.Thread(
        target=资源监控,
        args=(停止事件, 结果根目录, 结果根目录 / "logs" / "resource_samples.csv"),
        daemon=True,
    )
    监控线程.start()
    总开始 = time.perf_counter()
    阶段汇总: dict[str, Any] = {}
    try:
        if 参数.stage in {"base", "all"}:
            阶段汇总["base"] = 执行阶段(
                "base",
                配置,
                流集合,
                运行基础流,
                int(配置["runtime"]["base_max_workers"]),
            )
            if 阶段汇总["base"]["status"] != "PASS":
                raise 致命流错误("基础阶段存在终止失败，候选阶段不启动")
        if 参数.stage in {"bundle", "all"}:
            阶段汇总["bundle"] = 执行阶段(
                "bundle",
                配置,
                流集合,
                运行候选流,
                int(配置["runtime"]["bundle_max_workers"]),
            )
        总状态 = (
            "PASS"
            if all(汇总["status"] == "PASS" for 汇总 in 阶段汇总.values())
            else "INCOMPLETE"
        )
    finally:
        停止事件.set()
        监控线程.join(timeout=10.0)

    总汇总 = {
        "run_id": 配置["run_id"],
        "status": 总状态,
        "stage_requested": 参数.stage,
        "selected_streams": len(流集合),
        "code_commit": 当前提交(),
        "effective_config_sha256": canonical_json_sha256(配置),
        "selection_sha256": canonical_json_sha256(选择) if 选择 else None,
        "stage_summaries": 阶段汇总,
        "elapsed_seconds": time.perf_counter() - 总开始,
        "runner_script_sha256": sha256_file(Path(__file__)),
        "performance_comparison_performed": False,
        "completed_utc": 当前时间(),
    }
    save_json(总汇总, 结果根目录 / "runner_summary.json")
    追加JSONL(结果根目录 / "logs" / "runner_invocations.jsonl", 总汇总)
    print(json.dumps(总汇总, ensure_ascii=False, indent=2), flush=True)
    return 0 if 总状态 == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(主函数())
