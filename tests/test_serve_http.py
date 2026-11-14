"""推理服务的 HTTP 层与批量调度测试。

注入一个假引擎，只替换模型前向，其余走真实的 HTTP 路由、成批逻辑与
错误映射，不需要 GPU 与 transformers。需要 fastapi / httpx 时整文件 skip。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="推理服务需要 infer extra（fastapi/uvicorn）")
pytest.importorskip("httpx", reason="HTTP 测试需要 httpx")

from fastapi.testclient import TestClient

from pepdesign.serve.app import _State, _worker
from pepdesign.serve.batcher import PendingRequest
from pepdesign.serve.engine import GenOutput
from pepdesign.utils.config import InferConfig


class StubEngine:
    """假引擎：不加载任何权重，返回带 "stub" 标记的占位文本。"""

    def __init__(self, cfg: InferConfig) -> None:
        self.cfg = cfg
        self.tokenizer = None
        self._adapter = cfg.adapter_path
        self.calls: list[list[str]] = []          # 每次前向收到的 prompt 列表
        self.batch_sizes: list[int] = []

    @property
    def is_ready(self) -> bool:
        return True

    def load(self) -> None:
        pass

    def generate_batch(self, prompts: list[str], params) -> list[GenOutput]:
        self.calls.append(list(prompts))
        self.batch_sizes.append(len(prompts))
        return [
            GenOutput(
                texts=[f"[stub] {p[:12]}" for _ in range(params.n)],
                n_prompt_tokens=max(1, len(p) // 4),
                n_completion_tokens=[3] * params.n,
            )
            for p in prompts
        ]

    def list_adapters(self) -> list[str]:
        return ["outputs/sft/final"]

    def set_adapter(self, path: str | None) -> None:
        self._adapter = path

    def health(self) -> dict:
        return {
            "status": "ok", "backend": "transformers", "model": self.cfg.base_model,
            "adapter": self._adapter, "device": "cpu", "dtype": "float32",
            "max_model_len": self.cfg.max_model_len, "detail": "",
        }


def _make_client(**overrides: Any):
    """造一个注入假引擎的 app 并返回 TestClient（with 进入时才会跑 lifespan）。"""
    from pepdesign.serve.app import create_app

    cfg = InferConfig(**{"adapter_path": None, **overrides})
    engine = StubEngine(cfg)
    app = create_app(cfg, engine=engine)  # type: ignore[arg-type]
    return app, engine, TestClient(app)


# ---------------------------------------------------------------------------
# 基本路由
# ---------------------------------------------------------------------------
def test_health_reports_device_and_adapter():
    """`/health` 必须报出 device / dtype / adapter。

    客户端据这些字段选择超时；挂错 adapter 时输出会看似正常但内容不对。
    """
    _, _, client = _make_client()
    with client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["backend"] == "transformers"
    assert body["device"] == "cpu"
    assert body["dtype"] == "float32"
    assert "adapter" in body


def test_list_models_returns_openai_shape():
    _, _, client = _make_client()
    with client:
        body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert body["data"] and body["data"][0]["object"] == "model"


def test_completions_returns_openai_shape():
    _, _, client = _make_client(max_tokens=16)
    with client:
        resp = client.post("/v1/completions", json={"prompt": "design a peptide"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"].startswith("[stub]")
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )


def test_completions_batch_prompt_returns_one_choice_each():
    """`prompt` 传列表时每条 prompt 至少对应一个 choice，否则结果无法对应到请求。"""
    _, _, client = _make_client()
    with client:
        resp = client.post("/v1/completions", json={"prompt": ["a", "bb", "ccc"]})
    assert resp.status_code == 200
    assert len(resp.json()["choices"]) == 3


def test_echo_prepends_the_prompt():
    _, _, client = _make_client()
    with client:
        body = client.post(
            "/v1/completions", json={"prompt": "ACDEF", "echo": True}
        ).json()
    assert body["choices"][0]["text"].startswith("ACDEF")


def test_n_produces_n_choices():
    _, _, client = _make_client()
    with client:
        body = client.post("/v1/completions", json={"prompt": "x", "n": 3}).json()
    assert len(body["choices"]) == 3
    assert [c["index"] for c in body["choices"]] == [0, 1, 2]


# ---------------------------------------------------------------------------
# 错误映射
# ---------------------------------------------------------------------------
def test_unsupported_param_returns_400_with_openai_error_body():
    """未实现的参数返回 400 而不是 500：该改的是请求，不是服务。"""
    _, _, client = _make_client()
    with client:
        resp = client.post("/v1/completions", json={"prompt": "x", "stream": True})
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert "stream" in err["message"]


def test_empty_prompt_returns_422_from_pydantic():
    """结构层面的错误由 pydantic 拦下（422），与业务层面的 400 区分。"""
    _, _, client = _make_client()
    with client:
        resp = client.post("/v1/completions", json={"prompt": "   "})
    assert resp.status_code == 422


def test_chat_requires_chat_template():
    """假引擎没有 tokenizer，/v1/chat/completions 要给出可操作的报错。"""
    _, _, client = _make_client()
    with client:
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 400
    assert "chat_template" in resp.json()["error"]["message"]


def test_oversized_prompt_is_rejected_fast_not_queued_forever():
    """单条超过 token 预算的 prompt 要立刻报错，否则客户端会一直挂到超时。"""
    _, _, client = _make_client(max_num_batched_tokens=8)
    with client:
        resp = client.post("/v1/completions", json={"prompt": "x" * 400})
    assert resp.status_code == 400
    assert "预算" in resp.json()["error"]["message"]


# ---------------------------------------------------------------------------
# 批量调度
# ---------------------------------------------------------------------------
def test_worker_actually_batches_concurrent_requests():
    """并发进来的 5 条请求要由一次前向处理掉，而不是 5 次。

    直接驱动 `_worker`，检查假引擎记录的每次前向 prompt 条数。
    """
    async def main():
        cfg = InferConfig(
            adapter_path=None,
            max_num_seqs=8,
            max_num_batched_tokens=8192,
            batch_timeout_ms=50,
        )
        engine = StubEngine(cfg)
        state = _State(cfg=cfg, engine=engine)
        task = asyncio.create_task(_worker(state))
        try:
            # 5 条几乎同时入队。每条自己造一个 future，模拟 app 的 _enqueue。
            futures = []
            for i in range(5):
                fut = asyncio.get_running_loop().create_future()
                futures.append(fut)
                await state.queue.put(
                    PendingRequest(
                        request_id=f"r{i}",
                        payload=type("J", (), {
                            "prompts": [f"prompt-{i}"], "params": _params(),
                            "future": fut,
                        })(),
                        n_tokens=10,
                        enqueued_at=time.monotonic(),
                    )
                )
            results = await asyncio.wait_for(asyncio.gather(*futures), timeout=5)
        finally:
            task.cancel()
        return engine.batch_sizes, results, state

    batch_sizes, results, state = asyncio.run(main())
    assert batch_sizes == [5], (
        f"5 条并发请求应当只触发 1 次前向，实际触发了 {len(batch_sizes)} 次"
        f"（每次的条数：{batch_sizes}）——成批没有生效，吞吐会退回单条串行"
    )
    assert len(results) == 5
    assert state.n_batches == 1 and state.n_requests == 5


def test_worker_stops_at_token_budget():
    """token 预算到顶就发车；只按条数成批会在长 prompt 下 OOM。"""
    async def main():
        cfg = InferConfig(
            adapter_path=None,
            max_num_seqs=32,
            max_num_batched_tokens=25,   # 只装得下 2 条 10-token 的
            batch_timeout_ms=20,
        )
        engine = StubEngine(cfg)
        state = _State(cfg=cfg, engine=engine)
        task = asyncio.create_task(_worker(state))
        try:
            futures = []
            for i in range(4):
                fut = asyncio.get_running_loop().create_future()
                futures.append(fut)
                await state.queue.put(PendingRequest(
                    request_id=f"r{i}",
                    payload=type("J", (), {
                        "prompts": [f"p{i}"], "params": _params(), "future": fut,
                    })(),
                    n_tokens=10, enqueued_at=time.monotonic(),
                ))
            await asyncio.wait_for(asyncio.gather(*futures), timeout=5)
        finally:
            task.cancel()
        return engine.batch_sizes

    sizes = asyncio.run(main())
    assert sizes == [2, 2], f"应当按 token 预算切成两批各 2 条，实际 {sizes}"


def _params():
    from pepdesign.serve.engine import GenParams

    return GenParams(max_tokens=8, temperature=0.7, top_p=0.9, n=1, stop=[], echo=False)
