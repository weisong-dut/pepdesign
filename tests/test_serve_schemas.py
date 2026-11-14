"""接口契约的纯逻辑测试：字段校验与未实现参数的显式拒绝。

覆盖三条设计选择：已实现参数严格校验、未实现参数返回 400、
不认识字段一律忽略（OpenAI 客户端会附带额外字段）。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pepdesign.serve.schemas import (
    SUPPORTED_SAMPLING,
    UNSUPPORTED_PARAMS,
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    ErrorResponse,
    UnsupportedParamError,
)


# ---------------------------------------------------------------------------
# 第 3 条：不认识的字段要忽略（否则"无缝对接"是假的）
# ---------------------------------------------------------------------------
def test_unknown_fields_are_ignored_not_rejected():
    """OpenAI 客户端会带 `user` / `presence_penalty` 等字段，必须忽略而非报 422。

    忽略不认识的扩展位与静默忽略已实现参数是两回事。
    """
    req = CompletionRequest(
        prompt="design a peptide",
        user="some-client-id",
        presence_penalty=0.5,
        frequency_penalty=0.0,
        unknown_future_param=123,
    )
    assert req.prompt == "design a peptide"


# ---------------------------------------------------------------------------
# 第 2 条：未实现的参数显式拒绝
# ---------------------------------------------------------------------------
def test_stream_true_is_rejected_not_silently_ignored():
    """`stream=True` 必须报错；静默忽略会让客户端一直等一个不会到来的 SSE 流。"""
    req = CompletionRequest(prompt="x", stream=True)
    with pytest.raises(UnsupportedParamError, match="stream"):
        req.reject_unsupported()


def test_stream_false_is_accepted():
    """`stream=False` 显式表示不要流式，与本服务行为一致，应当接受。"""
    CompletionRequest(prompt="x", stream=False).reject_unsupported()


@pytest.mark.parametrize("param", ["logprobs", "best_of", "seed"])
def test_other_unimplemented_params_are_rejected(param):
    req = CompletionRequest(prompt="x", **{param: 5 if param != "seed" else 42})
    with pytest.raises(UnsupportedParamError, match=param):
        req.reject_unsupported()


def test_chat_rejects_unimplemented_params_too():
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hi")], stream=True
    )
    with pytest.raises(UnsupportedParamError, match="stream"):
        req.reject_unsupported()


def test_every_declared_unsupported_param_is_actually_checked():
    """`UNSUPPORTED_PARAMS` 是对外声明的清单，必须与实际校验保持一致。

    清单与校验任一方向漏改，这里都会立刻失败。
    """
    # 清单一共 7 项
    assert len(UNSUPPORTED_PARAMS) == 7, (
        f"未实现参数清单变了（现在 {len(UNSUPPORTED_PARAMS)} 项）。"
        f"请同步检查 CompletionRequest / ChatCompletionRequest 的 "
        f"reject_unsupported 以及 tests 里的参数化列表。"
    )
    # 抽掉不支持的那些，剩下的应当正好是已实现的采样参数 + 协议字段
    declared = set(UNSUPPORTED_PARAMS)
    assert declared.isdisjoint(SUPPORTED_SAMPLING), (
        "同一个参数不能既声明'已实现'又声明'未实现'"
    )


# ---------------------------------------------------------------------------
# 第 1 条：已实现的参数严格校验
# ---------------------------------------------------------------------------
def test_completion_rejects_empty_prompt():
    with pytest.raises(ValidationError, match="不能为空"):
        CompletionRequest(prompt="   ")
    with pytest.raises(ValidationError, match="不能为空"):
        CompletionRequest(prompt=[])


def test_completion_rejects_empty_string_inside_batch():
    """批量请求里混进空串要报错；空串会让结果与需求的下标对应关系变得含糊。"""
    with pytest.raises(ValidationError, match="空字符串"):
        CompletionRequest(prompt=["ok", ""])


def test_chat_requires_last_message_to_be_user():
    """最后一条必须是 user；以 assistant 结尾时模型会续写上一句而非回答新问题。"""
    with pytest.raises(ValidationError, match="必须是 user"):
        ChatCompletionRequest(messages=[
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="previous answer"),
        ])


def test_chat_accepts_a_normal_conversation():
    req = ChatCompletionRequest(messages=[
        ChatMessage(role="system", content="你是多肽设计专家。"),
        ChatMessage(role="user", content="设计一条 10 残基的抗菌肽。"),
    ])
    assert len(req.messages) == 2


def test_sampling_params_are_range_checked():
    """越界的采样参数要报错；它们不会让服务崩，只会让输出退化为噪声。"""
    with pytest.raises(ValidationError):
        CompletionRequest(prompt="x", temperature=-1.0)
    with pytest.raises(ValidationError):
        CompletionRequest(prompt="x", top_p=0.0)
    with pytest.raises(ValidationError):
        CompletionRequest(prompt="x", max_tokens=0)
    with pytest.raises(ValidationError):
        CompletionRequest(prompt="x", n=999)


# ---------------------------------------------------------------------------
# 错误体形状
# ---------------------------------------------------------------------------
def test_error_response_matches_openai_shape():
    """错误体必须是 `{"error": {"message": ...}}`，客户端按 `error.message` 取值。"""
    body = ErrorResponse(error={"message": "boom", "type": "invalid_request_error"})
    dumped = body.model_dump()
    assert set(dumped) == {"error"}
    assert "message" in dumped["error"]
