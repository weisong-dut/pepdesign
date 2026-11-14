"""掩码 / 截断 / padding 的纯逻辑测试。

覆盖"训练照跑、loss 照降、模型学不会"的失败模式：掩码差位、方向反、
ids 与 mask 分开补齐。断言写出期望的 labels，而不是只要求不崩。
本文件不 import torch。
"""

from __future__ import annotations

import pytest

from pepdesign.train.masking import (
    ELLIPSIS_TOKEN_ID,
    apply_mask_to_labels,
    build_completion_mask,
    build_rollout_masks,
    completion_end,
    completion_token_count,
    has_generation_markers,
    pad_batch,
    pad_left,
    resolve_ellipsis,
    shift_mask_for_labels,
    summarize_lengths,
    truncate_prompt,
)

# ---------------------------------------------------------------------------
# build_completion_mask
# ---------------------------------------------------------------------------


def test_completion_mask_marks_only_answer():
    mask = build_completion_mask(prompt_len=3, total_len=6)
    assert mask == [0, 0, 0, 1, 1, 1]
    assert completion_token_count(mask) == 3


def test_completion_mask_all_prompt_and_empty():
    # 边界：整条都是 prompt（没有任何答案 token）。
    # 这不是假想情况——生成提前遇到 eos 时就会产生它。
    assert build_completion_mask(4, 4) == [0, 0, 0, 0]
    assert build_completion_mask(0, 0) == []
    # 边界：prompt 为空（理论上不该出现，但出现了也不该静默出错）
    assert build_completion_mask(0, 2) == [1, 1]


def test_completion_mask_rejects_length_inversion():
    """prompt_len > total_len 必须报错，不能静默截断成全 1 掩码。"""
    with pytest.raises(ValueError, match="不能大于"):
        build_completion_mask(prompt_len=5, total_len=3)
    with pytest.raises(ValueError, match="不能为负"):
        build_completion_mask(prompt_len=-1, total_len=3)


# ---------------------------------------------------------------------------
# shift：off-by-one 的防线
# ---------------------------------------------------------------------------


def test_shift_mask_moves_left_by_one():
    """`out[i] == mask[i+1]`：因果 LM 中 `logits[t]` 预测 `input_ids[t+1]`，位移必须向左。"""
    mask = [0, 0, 0, 1, 1, 1]
    shifted = shift_mask_for_labels(mask)
    assert shifted == [0, 0, 1, 1, 1, 0]
    for i in range(len(mask)):
        expected = mask[i + 1] if i + 1 < len(mask) else 0
        assert shifted[i] == expected, f"位置 {i} 的位移不对"


def test_shift_mask_edge_cases():
    assert shift_mask_for_labels([]) == []
    # 整条都是答案：左移后首位变成 1（第一个答案 token 由 prompt 的
    # 最后一个 token 位置来预测），末位永远是 0（那一位没有标签可比）。
    assert shift_mask_for_labels([1, 1, 1]) == [1, 1, 0]
    # 整条都是 prompt：左移后全 0
    assert shift_mask_for_labels([0, 0, 0]) == [0, 0, 0]


# ---------------------------------------------------------------------------
# labels：与 input_ids 的对齐
# ---------------------------------------------------------------------------


def test_labels_align_with_same_position():
    """保留位置满足 `labels[i] == input_ids[i]`；HF 的因果 LM 会自行移位对齐。"""
    ids = [10, 11, 12, 13, 14, 15]
    mask = [0, 0, 0, 1, 1, 1]
    labels = apply_mask_to_labels(ids, mask)

    assert labels == [-100, -100, -100, 13, 14, 15]
    for i, lab in enumerate(labels):
        if lab != -100:
            assert lab == ids[i]

    supervised = [i for i, lab in enumerate(labels) if lab != -100]
    assert supervised == [3, 4, 5]


def test_labels_all_completion_and_all_prompt():
    # 整条都是答案：labels 与 ids 完全一致
    assert apply_mask_to_labels([2, 3], [1, 1]) == [2, 3]
    # 整条都是 prompt：一个监督信号都没有
    assert apply_mask_to_labels([2, 3, 4], [0, 0, 0]) == [-100, -100, -100]
    assert apply_mask_to_labels([], []) == []


def test_labels_length_mismatch_rejected():
    with pytest.raises(ValueError, match="长度不一致"):
        apply_mask_to_labels([1, 2, 3], [0, 1])


# ---------------------------------------------------------------------------
# padding：ids 和 mask 必须一起补
# ---------------------------------------------------------------------------


def test_pad_batch_pads_ids_attention_and_mask_together():
    """掩码必须与 ids 用同样的宽度补齐；长度不一致时 loss 会算在错的 token 上。"""
    out = pad_batch(
        [[1, 2, 3], [4, 5]],
        pad_token_id=0,
        token_masks=[[0, 1, 1], [0, 1]],
    )
    assert out["input_ids"] == [[1, 2, 3], [4, 5, 0]]
    assert out["attention_mask"] == [[1, 1, 1], [1, 1, 0]]
    assert out["token_mask"] == [[0, 1, 1], [0, 1, 0]]

    # 三个列表的宽度必须一致，否则下游的 zip 会静默截断
    widths = {len(row) for key in out for row in out[key]}
    assert widths == {3}


def test_pad_batch_padding_never_becomes_supervised():
    """补齐位置必须既不在 attention 里、也不在掩码里，否则 pad 会参与训练。"""
    out = pad_batch([[1, 2, 3], [4, 5]], pad_token_id=0, token_masks=[[0, 1, 1], [0, 1]])
    for ids, attn, mask in zip(
        out["input_ids"], out["attention_mask"], out["token_mask"], strict=True
    ):
        for i in range(len(ids)):
            if attn[i] == 0:
                assert mask[i] == 0
            if mask[i] == 1:
                assert attn[i] == 1


def test_pad_batch_without_masks_and_empty():
    out = pad_batch([[1, 2], [3]], pad_token_id=9)
    assert out["input_ids"] == [[1, 2], [3, 9]]
    assert out["token_mask"] == []          # 没给掩码就不返回掩码

    empty = pad_batch([], pad_token_id=0)
    assert empty == {"input_ids": [], "attention_mask": [], "token_mask": []}


def test_pad_batch_max_length_truncates_both():
    out = pad_batch(
        [[1, 2, 3, 4], [5, 6]],
        pad_token_id=0,
        max_length=2,
        token_masks=[[0, 1, 1, 1], [0, 1]],
    )
    assert out["input_ids"] == [[1, 2], [5, 6]]
    assert out["token_mask"] == [[0, 1], [0, 1]]
    assert out["attention_mask"] == [[1, 1], [1, 1]]


def test_pad_batch_rejects_mask_length_mismatch():
    with pytest.raises(ValueError, match="不一致"):
        pad_batch([[1, 2, 3]], pad_token_id=0, token_masks=[[0, 1]])


# ---------------------------------------------------------------------------
# 截断：从中间砍，保住 system prompt 和尾部约束
# ---------------------------------------------------------------------------


def test_truncate_keeps_prefix_and_suffix():
    ids = list(range(10))
    out = truncate_prompt(ids, max_prompt_length=5, keep_prefix=2)
    assert len(out) == 5
    assert out[:2] == [0, 1]                 # system prompt 前缀保住
    assert out[2] == ELLIPSIS_TOKEN_ID       # 中间用省略标记占一位
    assert out[3:] == [8, 9]                 # 尾部的具体约束保住


def test_truncate_noop_when_within_budget():
    ids = [1, 2, 3]
    assert truncate_prompt(ids, 5) == [1, 2, 3]
    assert truncate_prompt(ids, 3) == [1, 2, 3]


def test_truncate_replaces_ellipsis_placeholder():
    """占位符必须换成真实 token id，硬编码数字会在换 tokenizer 后变成普通词。"""
    out = resolve_ellipsis([1, ELLIPSIS_TOKEN_ID, 2], unk_token_id=999)
    assert out == [1, 999, 2]
    assert resolve_ellipsis([1, 2], 999) == [1, 2]


# ---------------------------------------------------------------------------
# 长度统计与 chat template 预检
# ---------------------------------------------------------------------------


def test_summarize_lengths_flags_truncation():
    """预检的意义：`max_seq_length` 配小时截断不会报错，模型只会学到残缺答案。"""
    report = summarize_lengths([10, 20, 30, 100], max_length=50)
    assert report["count"] == 4
    assert report["truncated"] == 1
    assert report["truncated_ratio"] == 0.25
    assert report["should_warn"] is True
    assert report["max_len"] == 100

    ok = summarize_lengths([10, 20, 30], max_length=50)
    assert ok["truncated"] == 0
    assert ok["should_warn"] is False

    assert summarize_lengths([], 50)["count"] == 0


def test_has_generation_markers():
    """官方 Qwen / Llama-3 模板都没有 {% generation %}，因此必须在构造 trainer 前预检。"""
    assert has_generation_markers("{% generation %}x{% endgeneration %}") is True
    assert has_generation_markers("没有标记的模板") is False
    assert has_generation_markers(None) is False
    # 只有一半也算缺——只写开始标记不会有任何效果
    assert has_generation_markers("{% generation %}x") is False


# ---------------------------------------------------------------------------
# build_rollout_masks —— GRPO 采样结果的掩码
#
# 按"token 值 == pad_token_id"判 padding 会把停止 token 一起遮掉（pad 就是 eos），
# 导致整条 completion 掩码全为 0、梯度恒为 0。
# ---------------------------------------------------------------------------
#: 基座 tokenizer 的 pad 与 eos 相同；这里只需同一个 id，具体数值不重要。
PAD = EOS = 151645


def test_completion_end_includes_the_stop_token():
    """终止 token 计入 completion，其后的填充不计；否则模型学不会停止。"""
    row = [1, 2, 3, EOS, PAD, PAD]
    assert completion_end(row, gen_start=3, stop_ids={EOS}) == 4   # 含 EOS
    assert completion_end(row, gen_start=3, stop_ids={PAD, EOS}) == 4
    # 跑满 max_new_tokens、没有终止 token 时，整段都算
    assert completion_end([1, 2, 3, 4, 5], gen_start=3, stop_ids={EOS}) == 5


def test_rollout_masks_exclude_left_padding_but_keep_stop_token():
    """左 padding 不进 attention；停止 token 进 attention 也进 completion。"""
    # 输入宽度 5：第一条 prompt 真实长度 5（无 padding），第二条只有 3
    full_ids = [
        [1, 2, 3, 4, 5, 6, 7, EOS, PAD, PAD],   # 无左侧 padding
        [PAD, PAD, 3, 4, 5, 6, EOS, PAD, PAD, PAD],  # 左侧 2 列 padding
    ]
    att, comp = build_rollout_masks(
        full_ids, gen_start=5, prompt_lens=[5, 3], group_size=1, stop_ids={PAD, EOS}
    )

    assert att[0] == [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    assert comp[0] == [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]

    # 左侧两列是 padding，但第 6 列（EOS）是真实 token，必须是 1
    assert att[1] == [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
    assert comp[1] == [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]


def test_rollout_masks_are_not_all_zero_when_generation_stops_immediately():
    """回归测试：序列一生成就停时 completion 掩码不能全为 0。"""
    full_ids = [[1, 2, 3, EOS, PAD, PAD]]
    _att, comp = build_rollout_masks(
        full_ids, gen_start=3, prompt_lens=[3], group_size=1, stop_ids={PAD, EOS}
    )
    assert comp[0] == [0, 0, 0, 1, 0, 0]
    assert completion_token_count(comp[0]) == 1


def test_rollout_masks_expand_prompt_major():
    """行 i 属于第 `i // group_size` 个 prompt（prompt-major），与 `generate` 的顺序一致。"""
    # 2 个 prompt、G=2：行序应是 [p0, p0, p1, p1]
    full_ids = [
        [1, 2, 3, 4, EOS, PAD],          # p0，无左 padding
        [1, 2, 3, 4, EOS, PAD],          # p0
        [PAD, 2, 3, 4, EOS, PAD],        # p1，左侧 1 列 padding
        [PAD, 2, 3, 4, 5, EOS],          # p1
    ]
    att, _ = build_rollout_masks(
        full_ids, gen_start=4, prompt_lens=[4, 3], group_size=2, stop_ids={PAD, EOS}
    )
    assert att[0][0] == 1 and att[1][0] == 1     # p0 的两条都没有左 padding
    assert att[2][0] == 0 and att[3][0] == 0     # p1 的两条都有


def test_rollout_masks_reject_row_count_mismatch():
    """行数与 prompt 数 × G 对不上时必须报错，不能将错就错地算下去。"""
    with pytest.raises(ValueError, match="group_size"):
        build_rollout_masks(
            [[1, 2, 3]], gen_start=3, prompt_lens=[3, 3], group_size=2, stop_ids={PAD}
        )


def test_rollout_masks_reject_more_prompt_tokens_than_the_input_width():
    """prompt 比输入还长时 attention 掩码来源就是错的，必须报错。"""
    with pytest.raises(ValueError, match="超过了输入宽度"):
        build_rollout_masks(
            [[1, 2, 3, 4]], gen_start=3, prompt_lens=[5], group_size=1, stop_ids={PAD}
        )


def test_rollout_masks_lines_up_with_the_shifted_logits():
    """掩码左移一位后，1 的位置恰好是"该预测答案的 logits"。"""
    full_ids = [[1, 2, 3, 4, EOS, PAD]]
    _, comp = build_rollout_masks(
        full_ids, gen_start=3, prompt_lens=[3], group_size=1, stop_ids={PAD, EOS}
    )
    shifted = shift_mask_for_labels(comp[0])
    # completion 覆盖位置 3（内容 `4`）和 4（EOS）；左移后位置 2/3 的 logits
    # 分别预测它们，位置 4 预测的是填充，所以为 0。
    assert shifted == [0, 0, 1, 1, 0, 0]
    supervised = [i for i, s in enumerate(shifted) if s]
    assert [full_ids[0][i + 1] for i in supervised] == [4, EOS]


# ---------------------------------------------------------------------------
# 截断的预算越界（回归锁）
# ---------------------------------------------------------------------------


def test_truncate_never_exceeds_the_budget_even_when_prefix_eats_it_all():
    """`keep_prefix >= max_prompt_length` 时结果不能超过预算。

    超长一位会让左 padding 宽度算错、掩码整体错位且不报错。
    """
    ids = list(range(10))
    for keep in (3, 4, 5, 6, 9):
        out = truncate_prompt(ids, max_prompt_length=5, keep_prefix=keep)
        assert len(out) <= 5, f"keep_prefix={keep} 时返回了 {len(out)} 个 token"

    # 前缀把预算吃光时，结果就是前缀本身
    assert truncate_prompt(ids, 5, keep_prefix=5) == [0, 1, 2, 3, 4]
    assert truncate_prompt(ids, 5, keep_prefix=99) == [0, 1, 2, 3, 4]


def test_truncate_uses_the_ellipsis_slot_when_only_one_token_is_left():
    """`suffix_budget == 1` 时那一位是省略标记；否则模型会把截断处当成正常上下文。"""
    out = truncate_prompt(list(range(10)), max_prompt_length=4, keep_prefix=3)
    assert out == [0, 1, 2, ELLIPSIS_TOKEN_ID]


# ---------------------------------------------------------------------------
# pad_left：GRPO 的生成要求
# ---------------------------------------------------------------------------


def test_pad_left_pads_on_the_left_and_reports_width():
    """左补齐并返回宽度；左 padding 让所有行的 completion 都从第 `width` 列开始。"""
    out = pad_left([[1, 2, 3], [4]], pad_token_id=0)
    assert out["width"] == 3
    assert out["input_ids"] == [[1, 2, 3], [0, 0, 4]]
    assert out["attention_mask"] == [[1, 1, 1], [0, 0, 1]]


def test_pad_left_edge_cases():
    assert pad_left([], pad_token_id=0) == {"input_ids": [], "attention_mask": [], "width": 0}
    # 只有一条：宽度就是它自己的长度，一个 pad 都不加
    out = pad_left([[7, 8]], pad_token_id=0)
    assert out["input_ids"] == [[7, 8]] and out["width"] == 2
    # 等长：同样不加 pad
    out = pad_left([[1], [2]], pad_token_id=0)
    assert out["attention_mask"] == [[1], [1]]


def test_pad_left_then_rollout_masks_stay_consistent_after_truncation():
    """截断、左 padding 与掩码三者的口径必须一致。

    `gen_start` 来自 padding 后宽度，`prompt_lens` 必须来自截断后长度。
    """
    prompts = [
        list(range(10)),        # 会被截到 6
        [100, 101, 102],        # 不需要截断
    ]
    truncated = [truncate_prompt(p, max_prompt_length=6, keep_prefix=2) for p in prompts]
    batched = pad_left(truncated, pad_token_id=0)

    gen_start = batched["width"]
    prompt_lens = [len(t) for t in truncated]
    assert gen_start == max(prompt_lens) == 6

    # 每条 prompt 后面接 2 个生成出来的 token
    full_ids = [[*row, 900 + i, 901 + i] for i, row in enumerate(batched["input_ids"])]
    attention, completion = build_rollout_masks(
        full_ids, gen_start=gen_start, prompt_lens=prompt_lens,
        group_size=1, stop_ids={0},
    )

    for i, n in enumerate(prompt_lens):
        left_pad = gen_start - n
        assert attention[i][:left_pad] == [0] * left_pad, "左侧 padding 没有被标成 0"
        assert attention[i][left_pad] == 1, "真实 prompt 的第一个 token 必须是可见的"
        # completion 覆盖 [gen_start, end)
        assert completion[i][:gen_start] == [0] * gen_start
        assert completion[i][gen_start] == 1
        assert sum(completion[i]) >= 1


# ---------------------------------------------------------------------------
# 省略标记的真实 token id 兜底
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """只带 token id 属性的假 tokenizer——本项目基座的真实情况是没有 unk。"""

    def __init__(self, **ids):
        self.unk_token_id = ids.get("unk_token_id")
        self.pad_token_id = ids.get("pad_token_id")
        self.eos_token_id = ids.get("eos_token_id")


def test_ellipsis_token_id_falls_back_when_unk_is_missing():
    """基座 tokenizer 的 `unk_token` 为 None，兜底链不能假设它存在。"""
    from pepdesign.train.grpo import ellipsis_token_id

    assert ellipsis_token_id(_FakeTokenizer(unk_token_id=7)) == 7
    assert ellipsis_token_id(_FakeTokenizer(pad_token_id=151643)) == 151643
    assert ellipsis_token_id(_FakeTokenizer(eos_token_id=151645)) == 151645
    assert ellipsis_token_id(
        _FakeTokenizer(unk_token_id=None, pad_token_id=5, eos_token_id=6)
    ) == 5, "unk 缺失时要退到 pad"
    with pytest.raises(ValueError, match="unk/pad/eos"):
        ellipsis_token_id(_FakeTokenizer())
