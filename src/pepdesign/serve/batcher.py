"""批量调度：把并发请求攒成批次后送入模型。

本模块不依赖 torch / asyncio / fastapi，只包含成批策略，可在任意环境做纯逻辑测试。
实现的是静态批：批次成员在发车时固定，整批等待最慢的一条生成完成；
连续批由 vLLM 后端提供，不在本模块范围内。

需同时满足两个上限：`max_num_seqs`（序列数）与 `max_num_batched_tokens`
（token 预算）；超时窗口 `batch_timeout_ms` 保障单条请求的延迟下界。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class PendingRequest(Generic[T]):
    """排队中的一条请求。

    `n_tokens` 是 prompt 分词后的长度，不是字符数。
    """

    request_id: str
    payload: T
    n_tokens: int
    enqueued_at: float = field(default_factory=time.monotonic)


@dataclass
class BatchPlan:
    """一次成批的结果。

    `oversized` 是单条即超过 token 预算的请求，需直接报错而非继续排队。
    """

    batch: list[PendingRequest]
    remainder: list[PendingRequest]
    oversized: list[PendingRequest]

    @property
    def n_tokens(self) -> int:
        # 空批为正常路径（超时分支），返回 0 而不报错。
        return sum(r.n_tokens for r in self.batch) if self.batch else 0

    def __len__(self) -> int:
        return len(self.batch)


def should_dispatch(
    pending: list[PendingRequest],
    *,
    max_seqs: int,
    max_batched_tokens: int,
    now: float | None = None,
    timeout_ms: int = 50,
) -> bool:
    """判断当前是否应发车。

    满足任一条件即发车：请求数达到 `max_seqs`、token 总数达到
    `max_batched_tokens`、或队首等待超过 `timeout_ms`。
    `now` 由调用方传入，便于测试构造等待时间。
    """
    if max_seqs < 1:
        raise ValueError(f"max_seqs 必须 >= 1，收到 {max_seqs}")
    if max_batched_tokens < 1:
        raise ValueError(f"max_batched_tokens 必须 >= 1，收到 {max_batched_tokens}")
    if not pending:
        return False

    if now is None:
        now = time.monotonic()
    if len(pending) >= max_seqs:
        return True
    if sum(r.n_tokens for r in pending) >= max_batched_tokens:
        return True
    waited_ms = (now - pending[0].enqueued_at) * 1000.0
    return waited_ms >= max(0, int(timeout_ms))


def form_batch(
    pending: list[PendingRequest],
    *,
    max_seqs: int,
    max_batched_tokens: int,
) -> BatchPlan:
    """从队首开始取请求，直到撞上序列数或 token 预算上限。

    撞到预算上限时停在原地而不跳过，以保持 FIFO。单条自身超过预算的请求
    进入 `oversized`，由调用方直接报错。本函数不判断发车时机。
    """
    if max_seqs < 1:
        raise ValueError(f"max_seqs 必须 >= 1，收到 {max_seqs}")
    if max_batched_tokens < 1:
        raise ValueError(f"max_batched_tokens 必须 >= 1，收到 {max_batched_tokens}")
    if not pending:
        return BatchPlan(batch=[], remainder=[], oversized=[])

    oversized = [r for r in pending if r.n_tokens > max_batched_tokens]
    queueable = [r for r in pending if r.n_tokens <= max_batched_tokens]

    batch: list[PendingRequest] = []
    used = 0
    cut = len(queueable)
    for i, req in enumerate(queueable):
        if len(batch) >= max_seqs or used + req.n_tokens > max_batched_tokens:
            cut = i
            break
        batch.append(req)
        used += req.n_tokens

    return BatchPlan(batch=batch, remainder=queueable[cut:], oversized=oversized)


def bench_batching(
    single_ms: float,
    *,
    batch_size: int,
    n_requests: int,
) -> dict:
    """估算静态批相对单条串行的吞吐提升倍数。

    `single_ms` 为单条请求端到端耗时（毫秒）。使用保守经验模型估算，
    返回值为估计值而非实测。
    """
    if single_ms <= 0:
        raise ValueError(f"single_ms 必须为正，收到 {single_ms}")
    if batch_size < 1 or n_requests < 1:
        raise ValueError("batch_size 与 n_requests 都必须 >= 1")

    serial_ms = single_ms * n_requests
    batch_ms = single_ms * (1 + 0.15 * (batch_size - 1))
    n_batches = -(-n_requests // batch_size)
    batched_ms = batch_ms * n_batches

    return {
        "estimated": True,
        "single_ms": single_ms,
        "batch_size": batch_size,
        "n_requests": n_requests,
        "serial_ms": round(serial_ms, 1),
        "batched_ms": round(batched_ms, 1),
        "speedup": round(serial_ms / batched_ms, 2) if batched_ms > 0 else 0.0,
        "note": "估算值（经验模型），不是实测。实测需要加载模型，见 bench_live。",
    }
