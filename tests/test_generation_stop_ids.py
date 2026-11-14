"""`generation_stop_ids` 的纯逻辑测试，不依赖 torch 与真实 tokenizer。

正常路径由冒烟测试用真实 tokenizer 覆盖；这里覆盖降级路径：
模板缺失终止符、前缀不一致、探针渲染失败时，函数必须退回 `{pad, eos}`
并留下警告。每种降级都会让 rollout 跑满 `max_completion_length` 且不报错，
因此警告本身也一并断言。
"""

from __future__ import annotations

import pytest

from pepdesign.models.loader import generation_stop_ids

#: Llama-3 形状的假模板：assistant 内容之后追加 `<|eot_id|>`。
#: id 只需彼此不同，不必对应某个真实基座。
BOS, EOS, EOT, USER_HDR, ASST_HDR, X, Y = 0, 1, 99, 10, 11, 50, 51


class _FakeTokenizer:
    """只实现 `generation_stop_ids` 真正用到的那几个接口。"""

    def __init__(
        self,
        *,
        append_terminator: bool = True,
        prefix_stable: bool = True,
        terminator_token: str = "<|eot_id|>",
        raise_on_render: bool = False,
    ):
        self.pad_token_id = EOS
        self.eos_token_id = EOS
        self.all_special_ids = [BOS, EOS, EOT]
        self._append = append_terminator
        self._prefix_stable = prefix_stable
        self._terminator_token = terminator_token
        self._raise = raise_on_render

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=False, return_dict=False
    ):
        if self._raise:
            raise RuntimeError("模板炸了")
        ids = [BOS, USER_HDR, X, EOT, ASST_HDR]
        has_assistant = any(m["role"] == "assistant" for m in messages)
        if has_assistant:
            ids += [Y]
            if self._append:
                ids += [EOT]
            if not self._prefix_stable:
                # 模拟"模板含随调用变化的成分"：前缀断言必须失败。
                ids = [BOS, *ids]
        return ids

    def encode(self, text, add_special_tokens=False):
        return {"x": [X], "y": [Y]}[text]

    def convert_ids_to_tokens(self, tid):
        return {EOS: "<|end_of_text|>", EOT: self._terminator_token}.get(tid, "?")


def test_picks_up_the_terminator_the_template_appends():
    """正常路径：模板追加的终止符（不是 eos）要进停止集。"""
    stop = generation_stop_ids(_FakeTokenizer())
    assert stop == {EOS, EOT}


def test_falls_back_when_the_template_appends_nothing(caplog):
    """模板不追加终止符时不能把内容最后一个 token 当成停止符。

    否则所有 completion 都会在内容末字符处被截断；长度检查负责拦住它。
    """
    import logging

    with caplog.at_level(logging.WARNING):
        stop = generation_stop_ids(_FakeTokenizer(append_terminator=False))
    assert stop == {EOS}
    assert any("与单独编码的回复内容" in r.message for r in caplog.records)


def test_falls_back_when_the_terminator_is_not_a_control_token(caplog):
    """形状检查：非 `<|...|>` 形状的 token 一律不认。"""
    import logging

    with caplog.at_level(logging.WARNING):
        stop = generation_stop_ids(_FakeTokenizer(terminator_token="."))
    assert stop == {EOS}
    assert any("不像对话控制 token" in r.message for r in caplog.records)


def test_falls_back_when_the_template_prefix_is_unstable(caplog):
    """前缀不一致时无法定位尾巴，必须降级而不是猜。"""
    import logging

    with caplog.at_level(logging.WARNING):
        stop = generation_stop_ids(_FakeTokenizer(prefix_stable=False))
    assert stop == {EOS}
    assert any("前缀不一致" in r.message for r in caplog.records)


def test_falls_back_when_rendering_raises(caplog):
    """模板渲染抛异常不该让训练起不来，但要出声。"""
    import logging

    with caplog.at_level(logging.WARNING):
        stop = generation_stop_ids(_FakeTokenizer(raise_on_render=True))
    assert stop == {EOS}
    assert any("无法用探针对话渲染" in r.message for r in caplog.records)


def test_never_returns_an_empty_set():
    """停止集为空会让掩码找不到任何终止点。兜底必须保住 eos/pad。"""
    stop = generation_stop_ids(_FakeTokenizer(append_terminator=False))
    assert stop, "停止集不能是空的"


@pytest.mark.parametrize("tid", [None, 0])
def test_handles_missing_pad_or_eos(tid):
    """pad/eos 缺失时不该把 None 塞进集合里（`None in {int}` 是类型污染）。"""

    class _Partial(_FakeTokenizer):
        def __init__(self):
            super().__init__()

    t = _Partial()
    t.pad_token_id = tid
    stop = generation_stop_ids(t)
    assert None not in stop
    assert all(isinstance(i, int) for i in stop)
