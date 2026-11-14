"""JSONL 读写与中间产物缓存。

JSONL 而非 CSV / Parquet
----------------------------------------------
* 相对 CSV：多肽样本带嵌套字段（靶点信息、属性快照、奖励分解），
  而且序列里可能含逗号；CSV 要转义半天还不如直接上 JSON。
* 相对 Parquet：JSONL 可流式追加。数据构建流程（下载 -> 清洗 -> 构造语料）
  里每一步都可能只处理增量数据，能追加比能列存更重要；
  而且出问题时可以直接 `head -1` 看一条样本，调试成本低。
* 体积不是问题：10 万条多肽样本大概几十 MB，压缩后更小。

原子写
------
所有写操作都是"先写 .tmp 再 rename"。训练任务动辄跑几小时，
如果在写文件的中途被杀（OOM、抢占、手动 kill），留下一个半截的语料文件
会让下一次运行读到坏数据，而且报错位置离真正的原因很远。
rename 在 POSIX/Windows 上都是原子的，这个习惯能省掉大量排查时间。
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TypeVar

import orjson

from pepdesign.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


# --------------------------------------------------------------------------
# JSONL
# --------------------------------------------------------------------------
def read_jsonl(path: str | Path) -> list[dict]:
    """读取整个 JSONL 文件。样本量大时改用 iter_jsonl 流式读。"""
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """流式逐行读取。

    空行会跳过，损坏的行会告警并跳过而不是抛异常：
    一条坏行不应该让整个 10 万条语料的构建任务失败，
    但也不能静默吞掉——所以计数并打印，最后统计时能看到"跳过了 N 行"。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"数据文件不存在: {p}")

    skipped = 0
    with open(p, "rb") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                skipped += 1
                if skipped <= 5:  # 只打前 5 条，避免坏文件刷屏
                    logger.warning("跳过损坏行 %s:%d (%s)", p.name, lineno, exc)
                continue
            yield obj
    if skipped:
        logger.warning("%s 共跳过 %d 行损坏数据", p.name, skipped)


def write_jsonl(path: str | Path, records: Iterable[dict], *, append: bool = False) -> int:
    """写入 JSONL，返回写入条数。默认原子覆盖写。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if append:
        count = 0
        with open(p, "ab") as f:
            for rec in records:
                f.write(orjson.dumps(rec, option=orjson.OPT_APPEND_NEWLINE))
                count += 1
        return count

    tmp = p.with_suffix(p.suffix + ".tmp")
    count = 0
    with open(tmp, "wb") as f:
        for rec in records:
            f.write(orjson.dumps(rec, option=orjson.OPT_APPEND_NEWLINE))
            count += 1
        f.flush()
        os.fsync(f.fileno())  # 落盘再 rename，断电也不会留下空文件
    os.replace(tmp, p)
    return count


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------
def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "rb") as f:
        return orjson.loads(f.read())


def write_json(path: str | Path, obj: Any, *, indent: bool = True) -> Path:
    """写 JSON。indent 用于需要人读的报表（实验指标、评估报告）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    option = orjson.OPT_INDENT_2 if indent else 0
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(orjson.dumps(obj, option=option | orjson.OPT_NON_STR_KEYS))
    os.replace(tmp, p)
    return p


# --------------------------------------------------------------------------
# 缓存
# --------------------------------------------------------------------------
class JsonlCache:
    """按 key 缓存"昂贵计算"的结果。

    典型用途：奖励函数里对预测器（Tm 预测、亲和力预测）的调用。
    GRPO 每个 step 要对 8 × batch 条生成序列打分，其中很多序列
    （尤其是 SFT 后模型输出同质化时）会重复出现，缓存能显著省时间。

    限制容量并用简单的 FIFO 淘汰，避免训练几天后内存爆掉——
    一个不设上限的缓存是 OOM 的经典原因。
    """

    def __init__(self, max_size: int = 100_000) -> None:
        self.max_size = max_size
        self._data: dict[str, Any] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        if key in self._data:
            self.hits += 1
            return self._data[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        if len(self._data) >= self.max_size:
            # FIFO：dict 在 Python 3.7+ 保持插入顺序，pop 第一个即可
            self._data.pop(next(iter(self._data)), None)
        self._data[key] = value

    def clear(self) -> None:
        self._data.clear()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict[str, float | int]:
        return {
            "size": len(self._data),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
        }


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_lines(path: str | Path) -> int:
    """统计文件行数。

    语料规模是数据报表里的基本指标，但 read_jsonl 会把整个文件加载进内存
    （几十万条样本时几百 MB），只想知道条数时用这个流式实现。
    """
    p = Path(path)
    if not p.exists():
        return 0
    with open(p, "rb") as f:
        return sum(1 for _ in f)


def human_size(num_bytes: int) -> str:
    """把字节数变成人读的字符串，用在数据报表里。"""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"
