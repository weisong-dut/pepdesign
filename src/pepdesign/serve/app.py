"""推理服务的 HTTP 层：OpenAI 兼容接口与批量调度。

启动：``python scripts/serve.py --config configs/serve.yaml``。
请求入队后由单个 worker 协程按 batch_size、token 预算或 batch_timeout_ms 攒批；
成批逻辑位于 `batcher.py`。

`model.generate` 为同步阻塞调用，一律走线程池，事件循环只负责排队与回填。
单 worker 为有意设计：多 worker 会各自攒小批，反而降低利用率。

未实现的接口参数一律显式拒绝（400），不静默忽略。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from pepdesign.serve.batcher import PendingRequest, form_batch, should_dispatch
from pepdesign.serve.engine import GenParams, InferenceEngine, ModelNotReadyError
from pepdesign.serve.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ErrorBody,
    ErrorResponse,
    HealthResponse,
    ModelCard,
    ModelListResponse,
    UnsupportedParamError,
    Usage,
)
from pepdesign.utils.config import InferConfig, LoRAConfig
from pepdesign.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class _Job:
    """一个请求的业务内容。`future` 用于把结果异步交回 HTTP 处理函数。

    入队的是包着它的 `PendingRequest`，不是本对象。
    """

    prompts: list[str]
    params: GenParams
    future: asyncio.Future | None = None
    request_id: str = field(default_factory=lambda: f"req-{uuid.uuid4().hex[:12]}")
    chat_mode: bool = False
    # prompt 分词后的总长度，用于计算成批时的 token 预算。
    n_tokens: int = 0


@dataclass
class _State:
    """进程级单例状态。用 dataclass 便于测试构造独立实例。"""

    cfg: InferConfig
    engine: InferenceEngine
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    # 上一轮攒批剩余的请求，必须先于新请求处理，避免长 prompt 被后来者插队。
    carry: list[PendingRequest] = field(default_factory=list)
    worker: asyncio.Task | None = None
    started_at: float = 0.0
    n_batches: int = 0
    n_requests: int = 0


# ---------------------------------------------------------------------------
# 批量 worker
# ---------------------------------------------------------------------------
async def _worker(state: _State) -> None:
    """攒批 → 前向 → 回填。每进程一个，运行在事件循环中。"""
    loop = asyncio.get_running_loop()
    cfg = state.cfg
    # 超时换算成秒且至少留 1ms：`wait_for(timeout=0)` 会立即超时，
    # 配置为 0 时退化为忙等。
    timeout_s = max(cfg.batch_timeout_ms, 1) / 1000.0

    while True:
        try:
            # 1) 取出待处理请求，carry 优先（其中的请求上一轮已等待过）。
            had_carry = bool(state.carry)
            pending = list(state.carry)
            state.carry = []
            if not pending:
                first = await state.queue.get()
                if first is None:  # 停机信号
                    return
                pending.append(first)

            if had_carry:
                # carry 分支不再等待（已等过一轮），但需非阻塞取走队列中已就绪的请求，
                # 否则会退化成每个 carry 单独发一批，降低吞吐。
                while len(pending) < cfg.max_num_seqs and not state.queue.empty():
                    try:
                        item = state.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if item is None:
                        return
                    pending.append(item)
            else:
                # 新起一批：在超时窗口内继续收，直到该发车
                deadline = loop.time() + timeout_s
                while True:
                    now = time.monotonic()
                    if should_dispatch(
                        pending,
                        max_seqs=cfg.max_num_seqs,
                        max_batched_tokens=cfg.max_num_batched_tokens,
                        now=now,
                        timeout_ms=cfg.batch_timeout_ms,
                    ):
                        break
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(state.queue.get(), timeout=remaining)
                    except asyncio.TimeoutError:
                        break
                    if item is None:
                        return
                    pending.append(item)

            # 3) 成批
            plan = form_batch(
                pending,
                max_seqs=cfg.max_num_seqs,
                max_batched_tokens=cfg.max_num_batched_tokens,
            )
            for req in plan.oversized:
                job: _Job = req.payload
                if not job.future.done():
                    job.future.set_exception(
                        ValueError(
                            f"prompt 自身长度 {req.n_tokens} token 超过了单批预算 "
                            f"max_num_batched_tokens={cfg.max_num_batched_tokens}。"
                            f"它无论排多久都装不下，所以直接返回错误而不是让它一直等。"
                            f"请调大 max_num_batched_tokens，或缩短这条 prompt。"
                        )
                    )
            state.carry = list(plan.remainder)

            if not plan.batch:
                continue

            state.n_batches += 1
            state.n_requests += len(plan.batch)
            logger.info(
                "发车：%d 条请求，合计 %d token（队列剩余 %d）",
                len(plan.batch), plan.n_tokens, len(state.carry),
            )

            # 4)+5) 前向 + 回填。两者必须在同一个 try 内：只要一批已经成形，
            # 其中每条 future 都必须有归宿（result 或 exception），否则客户端会挂到超时。
            try:
                outputs = await loop.run_in_executor(
                    None, _run_batch, state.engine, plan.batch
                )
                # 条数对不上说明下标对应关系已不可靠，宁可报错也不能发错结果。
                for req, out in zip(plan.batch, outputs, strict=True):
                    job = req.payload
                    if not job.future.done():
                        job.future.set_result(out)
            except Exception as exc:  # 一批失败时将异常发给该批全部请求
                for req in plan.batch:
                    job = req.payload
                    if job.future is not None and not job.future.done():
                        job.future.set_exception(exc)
                logger.exception("批次执行失败，已把异常回填给 %d 条请求", len(plan.batch))

        except asyncio.CancelledError:
            return
        except Exception:  # worker 不能退出，否则服务会静默停止响应
            logger.exception("批量 worker 出现未预期异常，继续处理后续请求")
            await asyncio.sleep(0.05)


def _run_batch(engine: InferenceEngine, batch: list[PendingRequest]):
    """在工作线程里执行一个批次。

    批内所有请求共用第一份的采样参数；参数不一致时发出警告。
    """
    jobs: list[_Job] = [r.payload for r in batch]
    params = jobs[0].params
    if any(j.params != params for j in jobs[1:]):
        logger.warning(
            "同一批里的采样参数不一致（共 %d 条），本批将统一使用第一条的参数"
            "（temperature=%.2f, top_p=%.2f, max_tokens=%d）。"
            "本实现没有按参数分组，需要精确控制请把请求拆开发。",
            len(jobs), params.temperature, params.top_p, params.max_tokens,
        )

    # 把批内 prompt 摊平后一次前向：n>1 的请求展开成多条，生成后按序切回。
    flat: list[str] = []
    spans: list[tuple[int, int]] = []
    for j in jobs:
        start = len(flat)
        flat.extend(j.prompts)
        spans.append((start, len(flat)))

    outs = engine.generate_batch(flat, params)

    # 每个 job 返回其自身 prompt 的 GenOutput 列表；usage 依赖其中的
    # n_prompt_tokens / n_completion_tokens。
    return [list(outs[lo:hi]) for (lo, hi) in spans]


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
def create_app(
    cfg: InferConfig,
    *,
    lora_cfg: LoRAConfig | None = None,
    engine: InferenceEngine | None = None,
):
    """构造 FastAPI 应用。

    `fastapi` 在函数内导入，缺失时不影响 `batcher` / `schemas` 的纯逻辑使用。
    `engine` 可注入，供 stub 与测试传入不加载权重的引擎。
    """
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "推理服务需要 fastapi / uvicorn，当前环境没有装：\n"
            "  pip install -e '.[infer]'\n"
            "（只需要接口契约和成批逻辑时，可以不装——"
            "`pepdesign.serve.schemas` 与 `pepdesign.serve.batcher` 都不依赖它们。）"
        ) from exc

    state = _State(
        cfg=cfg,
        engine=engine if engine is not None else InferenceEngine(cfg, lora_cfg=lora_cfg),
    )

    @asynccontextmanager
    async def lifespan(app) -> AsyncIterator[None]:
        state.started_at = time.time()
        # 3B 模型加载是分钟级的，放到线程池执行，避免启动期间不接受连接。
        state.worker = asyncio.create_task(_worker(state))
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, state.engine.load)
            logger.info("模型加载完成，服务就绪")
        except Exception:
            logger.exception("模型加载失败——/health 会报 error，请检查日志")
        yield
        if state.worker:
            state.worker.cancel()

    app = FastAPI(
        title="pepdesign 多肽设计推理服务",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    def _error(status: int, message: str, *, type_: str, code: str | None = None):
        return JSONResponse(
            status_code=status,
            content=ErrorResponse(
                error=ErrorBody(message=message, type=type_, code=code)
            ).model_dump(),
        )

    @app.exception_handler(UnsupportedParamError)
    async def _unsupported(request: Request, exc: UnsupportedParamError):
        return _error(400, str(exc), type_="invalid_request_error",
                      code="unsupported_parameter")

    @app.exception_handler(ModelNotReadyError)
    async def _not_ready(request: Request, exc: ModelNotReadyError):
        # 503 表示"稍后再试"；映射成 500 会让客户端放弃重试。
        return _error(503, str(exc), type_="service_unavailable")

    @app.exception_handler(ValueError)
    async def _bad_request(request: Request, exc: ValueError):
        return _error(400, str(exc), type_="invalid_request_error")

    # ------------------------------------------------------------------
    def _resolve(req_max: int | None, req_temp: float | None, req_top_p: float | None,
                 n: int, stop, echo: bool) -> GenParams:
        """把请求参数与配置默认值合成确定的 GenParams。

        使用 `is not None` 判断：`temperature=0`（贪心解码）是合法值，
        不能用 `or` 回退到默认值。
        """
        return GenParams(
            max_tokens=req_max if req_max is not None else cfg.max_tokens,
            temperature=req_temp if req_temp is not None else cfg.temperature,
            top_p=req_top_p if req_top_p is not None else cfg.top_p,
            n=n,
            stop=([stop] if isinstance(stop, str) else list(stop or [])),
            echo=echo,
        )

    async def _enqueue(job: _Job):
        """把作业入队并等待结果。

        入队的是 `PendingRequest` 而非 `_Job`，队列元素类型必须与下游一致。
        """
        loop = asyncio.get_running_loop()
        job.future = loop.create_future()
        await state.queue.put(
            PendingRequest(
                request_id=job.request_id,
                payload=job,
                n_tokens=job.n_tokens or 1,
                enqueued_at=time.monotonic(),
            )
        )
        return await asyncio.wait_for(job.future, timeout=cfg.request_timeout_s)

    # ------------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(**state.engine.health())

    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models():
        name = cfg.merged_model_path or cfg.base_model
        return ModelListResponse(data=[ModelCard(id=name)])

    @app.get("/admin/stats")
    async def stats():
        """运维统计。

        `n_requests / n_batches` 即平均批大小，长期为 1 表示攒批未生效。
        """
        return {
            "uptime_s": round(time.time() - state.started_at, 1) if state.started_at else 0.0,
            "n_batches": state.n_batches,
            "n_requests": state.n_requests,
            "avg_batch_size": round(state.n_requests / state.n_batches, 2)
                              if state.n_batches else 0.0,
            "queue_depth": state.queue.qsize(),
            "carry": len(state.carry),
        }

    @app.get("/admin/adapters")
    async def list_adapters():
        return {"current": state.engine._adapter, "available": state.engine.list_adapters()}

    @app.post("/admin/adapters")
    async def switch_adapter(body: dict):
        path = body.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("请求体需要 {\"path\": \"...\"}，且 path 不能为空")
        state.engine.set_adapter(path)
        return {"ok": True, "current": path}

    # ------------------------------------------------------------------
    @app.post("/v1/completions", response_model=CompletionResponse)
    async def completions(req: CompletionRequest):
        req.reject_unsupported()
        if not state.engine.is_ready:
            raise ModelNotReadyError("模型尚未加载完成，请稍后重试。")
        prompts = [req.prompt] if isinstance(req.prompt, str) else list(req.prompt)
        params = _resolve(req.max_tokens, req.temperature, req.top_p,
                          req.n, req.stop, req.echo)
        job = _Job(prompts=prompts, params=params, future=None,
                   n_tokens=sum(len(p) for p in prompts))
        outs = await _enqueue(job)

        choices = []
        n_prompt = 0
        n_completion = 0
        for i, out in enumerate(outs):
            n_prompt += out.n_prompt_tokens
            for k, text in enumerate(out.texts):
                n_completion += out.n_completion_tokens[k]
                # echo 会把 prompt 回显在 text 前，符合 OpenAI 语义。
                choices.append(CompletionChoice(
                    text=(prompts[i] + text) if req.echo else text,
                    index=len(choices),
                    finish_reason="length",
                ))
        return CompletionResponse(
            model=cfg.merged_model_path or cfg.base_model,
            choices=choices,
            usage=Usage(prompt_tokens=n_prompt, completion_tokens=n_completion,
                        total_tokens=n_prompt + n_completion),
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(req: ChatCompletionRequest):
        req.reject_unsupported()
        if not state.engine.is_ready:
            raise ModelNotReadyError("模型尚未加载完成，请稍后重试。")
        tok = state.engine.tokenizer
        if tok is None or not getattr(tok, "chat_template", None):
            raise ValueError(
                "tokenizer 没有 chat_template，无法渲染 chat 请求。"
                "请改用 /v1/completions 并自行拼好 prompt。"
            )
        prompt = tok.apply_chat_template(
            [m.model_dump() for m in req.messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        params = _resolve(req.max_tokens, req.temperature, req.top_p,
                          req.n, req.stop, echo=False)
        job = _Job(prompts=[prompt], params=params, future=None,
                   chat_mode=True, n_tokens=len(prompt))
        outs = await _enqueue(job)

        choices = []
        n_completion = 0
        for out in outs:
            for k, text in enumerate(out.texts):
                n_completion += out.n_completion_tokens[k]
                choices.append(ChatCompletionChoice(
                    index=len(choices),
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason="length",
                ))
        return ChatCompletionResponse(
            model=cfg.merged_model_path or cfg.base_model,
            choices=choices,
            usage=Usage(prompt_tokens=outs[0].n_prompt_tokens if outs else 0,
                        completion_tokens=n_completion,
                        total_tokens=(outs[0].n_prompt_tokens if outs else 0) + n_completion),
        )

    return app
