"""推理服务的请求 / 响应契约（OpenAI 兼容）。

本模块只依赖 pydantic，不导入 torch / transformers / fastapi，便于纯逻辑测试。
字段名与 OpenAI 协议保持一致（/v1/completions、/v1/chat/completions、
/v1/models）；/health 为自定义存活探针。

已实现参数严格校验，未实现参数显式拒绝（400），其余未知字段忽略。
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# 公共
# ---------------------------------------------------------------------------
class _Strict(BaseModel):
    """请求基类：未知字段一律忽略（extra="ignore"）。

    OpenAI 客户端常附带未定义字段；已实现参数严格校验，未实现参数显式拒绝。
    """

    model_config = ConfigDict(extra="ignore")


#: 本服务已实现的采样参数，供校验与文档引用。
SUPPORTED_SAMPLING: frozenset[str] = frozenset(
    {"temperature", "top_p", "max_tokens", "n", "stop", "echo"}
)

#: OpenAI 协议中已定义但本服务未实现的参数，传入时返回 400。
UNSUPPORTED_PARAMS: dict[str, str] = {
    "stream": "本服务只提供非流式响应。需要流式请在上层用 SSE 自行包装。"
              "（这与后端无关：vLLM 后端同样只返回完整结果，"
              "流式属于 HTTP 层的改造，见 InferConfig.backend 的说明。）",
    "logprobs": "逐 token 对数概率没有实现：本服务不返回 token 级别的分数。",
    "best_of": "best_of 没有实现；需要多采样请在请求里用 n，然后在客户端筛。",
    "tools": "工具调用（function calling）没有实现。",
    "function_call": "工具调用没有实现。",
    "response_format": "结构化输出（JSON mode）没有实现。",
    "seed": "seed 没有实现——本服务不保证同一个 seed 的可复现性。",
}


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ErrorBody(BaseModel):
    message: str
    type: str = "invalid_request_error"
    code: str | None = None


class ErrorResponse(BaseModel):
    """OpenAI 错误体。结构必须保持一致：客户端按 `error.message` 取值。"""

    error: ErrorBody


# ---------------------------------------------------------------------------
# /v1/completions
# ---------------------------------------------------------------------------
class CompletionRequest(_Strict):
    model: str = ""
    prompt: str | list[str]
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=32)
    stop: str | list[str] | None = None
    echo: bool = False
    # 显式声明的未实现参数，用于给出明确错误而不是被 extra="ignore" 忽略。
    stream: bool | None = None
    logprobs: int | None = None
    best_of: int | None = None
    seed: int | None = None

    @field_validator("prompt")
    @classmethod
    def _prompt_not_empty(cls, v: str | list[str]) -> str | list[str]:
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("prompt 不能为空字符串")
            return v
        if not v:
            raise ValueError("prompt 列表不能为空")
        if any(not p.strip() for p in v):
            raise ValueError("prompt 列表里有空字符串——批量请求里混进空串，"
                             "会得到一个和前一条对不上的下标")
        return v

    def reject_unsupported(self) -> None:
        """校验未实现参数并抛出 UnsupportedParamError。

        `stream=False` 不拒绝，只有 `stream=True` 才报错。
        """
        if self.stream:
            raise UnsupportedParamError("stream", UNSUPPORTED_PARAMS["stream"])
        for name in ("logprobs", "best_of", "seed"):
            if getattr(self, name) is not None:
                raise UnsupportedParamError(name, UNSUPPORTED_PARAMS[name])


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: None = None
    finish_reason: Literal["stop", "length"] = "stop"


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[CompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(_Strict):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=32)
    stop: str | list[str] | None = None
    stream: bool | None = None
    logprobs: int | None = None
    seed: int | None = None

    @field_validator("messages")
    @classmethod
    def _messages_not_empty(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        if not v:
            raise ValueError("messages 不能为空")
        # 最后一条必须是 user：以 assistant 结尾会让模板续写上一条回复而非回答新问题。
        if v[-1].role != "user":
            raise ValueError(
                f"messages 的最后一条必须是 user，收到 {v[-1].role!r}。"
                f"以 assistant 结尾等于让模型续写自己上一句话，"
                f"而不是回答新问题。"
            )
        return v

    def reject_unsupported(self) -> None:
        if self.stream:
            raise UnsupportedParamError("stream", UNSUPPORTED_PARAMS["stream"])
        for name in ("logprobs", "seed"):
            if getattr(self, name) is not None:
                raise UnsupportedParamError(name, UNSUPPORTED_PARAMS[name])


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# ---------------------------------------------------------------------------
# /v1/models 与 /health
# ---------------------------------------------------------------------------
class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "pepdesign"


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class HealthResponse(BaseModel):
    """存活探针，非 OpenAI 标准。

    额外报告 `backend` / `adapter` / `device` / `dtype`：挂错 adapter 或
    设备精度不同会直接影响输出正确性与超时设置。
    """

    status: Literal["ok", "loading", "error"] = "ok"
    backend: str = "transformers"
    model: str = ""
    adapter: str | None = None
    device: str = "cpu"
    dtype: str = "float32"
    max_model_len: int = 0
    detail: str = ""


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------
class UnsupportedParamError(ValueError):
    """请求包含本服务未实现的参数。单独类型便于映射为 HTTP 400 而非 500。"""

    def __init__(self, param: str, reason: str) -> None:
        self.param = param
        self.reason = reason
        super().__init__(f"参数 {param!r} 未实现：{reason}")
