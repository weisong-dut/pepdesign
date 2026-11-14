"""成批策略与接口契约的纯逻辑测试，不依赖 torch 与 fastapi。

覆盖三条边界：`max_num_seqs`、`max_num_batched_tokens` 与超时窗口。
写错它们不会报错，只会让吞吐退回单条串行或在长 prompt 下 OOM。
HTTP 路由与模型加载见 `test_serve_http.py`。
"""

from __future__ import annotations

import time

import pytest

from pepdesign.serve.batcher import (
    PendingRequest,
    bench_batching,
    form_batch,
    should_dispatch,
)


def _req(rid: str, n_tokens: int, *, enqueued_at: float | None = None) -> PendingRequest:
    """造一条排队请求；`enqueued_at` 默认取当前单调时钟。

    写死 0.0 会让"刚刚入队"被算成已等待数十万秒，超时断言因此全部失真。
    """
    return PendingRequest(
        request_id=rid,
        payload=rid,
        n_tokens=n_tokens,
        enqueued_at=time.monotonic() if enqueued_at is None else enqueued_at,
    )


# ---------------------------------------------------------------------------
# should_dispatch：什么时候发车
# ---------------------------------------------------------------------------
def test_empty_queue_never_dispatches():
    assert should_dispatch([], max_seqs=4, max_batched_tokens=100) is False


def test_dispatches_when_queue_reaches_max_seqs():
    pending = [_req(f"r{i}", 10) for i in range(4)]
    assert should_dispatch(pending, max_seqs=4, max_batched_tokens=9999) is True
    # 差一条就不发（前提是还没超时、token 也没到顶）
    assert should_dispatch(pending[:3], max_seqs=4, max_batched_tokens=9999,
                           timeout_ms=99999) is False


def test_dispatches_when_token_budget_is_reached():
    """token 预算到顶必须发车；只按条数成批会在长 prompt 下 OOM。"""
    pending = [_req("a", 400), _req("b", 400)]
    # 条数远没到上限，但 token 预算已经到顶
    assert should_dispatch(pending, max_seqs=32, max_batched_tokens=800,
                           timeout_ms=99999) is True


def test_dispatches_when_head_has_waited_too_long():
    """队首超时必须发车；只按"攒满"发车会让低负载下的延迟反而更高。"""
    pending = [_req("r0", 10, enqueued_at=100.0)]
    # 等了 30ms，阈值 50ms → 不发
    assert should_dispatch(pending, max_seqs=32, max_batched_tokens=9999,
                           now=100.030, timeout_ms=50) is False
    # 等了 60ms → 发
    assert should_dispatch(pending, max_seqs=32, max_batched_tokens=9999,
                           now=100.060, timeout_ms=50) is True


def test_timeout_looks_at_the_head_not_the_whole_queue():
    """只看队首的等待时间；这样等太久的请求一定被发走，不会被后来者饿死。"""
    pending = [
        _req("old", 10, enqueued_at=0.0),
        _req("new", 10, enqueued_at=999.0),
    ]
    assert should_dispatch(pending, max_seqs=32, max_batched_tokens=9999,
                           now=999.5, timeout_ms=50) is True


def test_should_dispatch_validates_limits():
    with pytest.raises(ValueError, match="max_seqs"):
        should_dispatch([_req("a", 1)], max_seqs=0, max_batched_tokens=10)
    with pytest.raises(ValueError, match="max_batched_tokens"):
        should_dispatch([_req("a", 1)], max_seqs=1, max_batched_tokens=0)


# ---------------------------------------------------------------------------
# form_batch：这一批取哪些
# ---------------------------------------------------------------------------
def test_form_batch_respects_max_seqs():
    pending = [_req(f"r{i}", 10) for i in range(10)]
    plan = form_batch(pending, max_seqs=3, max_batched_tokens=99999)
    assert [r.request_id for r in plan.batch] == ["r0", "r1", "r2"]
    assert [r.request_id for r in plan.remainder] == [f"r{i}" for i in range(3, 10)]
    assert plan.oversized == []


def test_form_batch_stops_at_token_budget_without_skipping():
    """撞到预算上限时停在原地，不跳过它去拿后面更短的请求。

    跳过会破坏 FIFO，让长 prompt 被后来的短请求无限插队。
    """
    # 三条各自都装得下，只是前两条装不进同一批；超预算的请求会进 oversized，
    # 到不了这条分支。
    pending = [_req("a", 30), _req("b", 30), _req("c", 10)]
    plan = form_batch(pending, max_seqs=32, max_batched_tokens=50)

    # a 进批；b 放不下（30+30 > 50），停在这里
    assert [r.request_id for r in plan.batch] == ["a"]
    # c 虽然装得下（30+10 = 40 ≤ 50），也不能被提前取走——否则 b 被插队了
    assert [r.request_id for r in plan.remainder] == ["b", "c"]
    assert plan.oversized == []


def test_form_batch_reports_oversized_instead_of_queueing_forever():
    """单条自身就超预算的请求进 `oversized`；留在队里会永远装不下并拖住队列。"""
    pending = [_req("ok", 10), _req("huge", 100000), _req("ok2", 10)]
    plan = form_batch(pending, max_seqs=32, max_batched_tokens=100)

    assert [r.request_id for r in plan.batch] == ["ok", "ok2"]
    assert [r.request_id for r in plan.oversized] == ["huge"]
    assert plan.remainder == []


def test_form_batch_preserves_fifo_order():
    pending = [_req(f"r{i}", 10) for i in range(6)]
    plan = form_batch(pending, max_seqs=2, max_batched_tokens=99999)
    assert [r.request_id for r in plan.batch] == ["r0", "r1"]
    assert [r.request_id for r in plan.remainder] == ["r2", "r3", "r4", "r5"]


def test_form_batch_empty_queue():
    plan = form_batch([], max_seqs=4, max_batched_tokens=100)
    assert plan.batch == [] and plan.remainder == [] and plan.oversized == []
    assert plan.n_tokens == 0
    assert len(plan) == 0


def test_form_batch_validates_limits():
    with pytest.raises(ValueError, match="max_seqs"):
        form_batch([_req("a", 1)], max_seqs=0, max_batched_tokens=10)
    with pytest.raises(ValueError, match="max_batched_tokens"):
        form_batch([_req("a", 1)], max_seqs=1, max_batched_tokens=0)


def test_form_batch_exact_fit_is_included():
    """刚好等于预算的请求要进批（边界是 `>` 不是 `>=`），否则会白白空出一块预算。"""
    plan = form_batch([_req("a", 100)], max_seqs=4, max_batched_tokens=100)
    assert [r.request_id for r in plan.batch] == ["a"]
    assert plan.n_tokens == 100


# ---------------------------------------------------------------------------
# 吞吐估算
# ---------------------------------------------------------------------------
def test_bench_batching_is_marked_as_estimated():
    """估算值必须自带标记，不能让人误当成实测。"""
    out = bench_batching(100.0, batch_size=8, n_requests=32)
    assert out["estimated"] is True
    assert "估算" in out["note"]
    # 32 条 / 批 8 = 4 批；每批 100 × (1 + 0.15×7) = 205ms → 820ms
    assert out["n_requests"] == 32
    assert out["batched_ms"] < out["serial_ms"]
    assert out["speedup"] > 1


def test_bench_batching_rejects_nonsense():
    with pytest.raises(ValueError, match="必须为正"):
        bench_batching(0.0, batch_size=4, n_requests=10)
    with pytest.raises(ValueError, match="必须"):
        bench_batching(10.0, batch_size=0, n_requests=10)
