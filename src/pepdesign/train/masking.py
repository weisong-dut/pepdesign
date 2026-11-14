"""掩码、截断与 padding：训练循环中对齐最敏感的一层。

本模块不 import torch，全部在 `list[int]` 上运算，便于逐条断言。三个关键约束：
损失位置比 token 位置左移一位（`logits[t]` 预测 `input_ids[t+1]`）；被截断的
答案不得计入 loss；prompt 截断不能丢弃 system prompt 中的输出格式契约。

纯函数版本由 `tests/test_train_masking.py` 覆盖。
"""

from __future__ import annotations

from collections.abc import Sequence

# ---------------------------------------------------------------------------
# 完成部分（completion）的掩码
# ---------------------------------------------------------------------------

def build_completion_mask(prompt_len: int, total_len: int) -> list[int]:
    """返回长度 `total_len` 的掩码：答案 token 为 1，prompt token 为 0。

    `mask[i] == 1` 表示 `input_ids[i]` 属于答案；作用于 loss 前须经
    `shift_mask_for_labels` 左移一位。
    """
    if prompt_len < 0 or total_len < 0:
        raise ValueError(f"长度不能为负: prompt_len={prompt_len}, total_len={total_len}")
    if prompt_len > total_len:
        raise ValueError(
            f"prompt_len ({prompt_len}) 不能大于 total_len ({total_len})。"
            f"这通常说明 tokenize 时把 prompt 和 completion 分别处理、"
            f"拼接顺序弄反了。"
        )
    return [0] * prompt_len + [1] * (total_len - prompt_len)


def shift_mask_for_labels(mask: Sequence[int]) -> list[int]:
    """把 token 位置掩码转为 logits 位置掩码：整体左移一位，首位补 0。

    `out[i] == mask[i + 1]`；末位恒为 0，因其预测的 token 不在 `input_ids` 内。
    """
    if not mask:
        return []
    return [int(m) for m in mask[1:]] + [0]


def apply_mask_to_labels(
    input_ids: Sequence[int],
    token_mask: Sequence[int],
    *,
    ignore_index: int = -100,
) -> list[int]:
    """按 token 掩码生成 labels：答案位置保留同一位置的 token，其余填 `ignore_index`。

    此处不能移位：HuggingFace 因果 LM 会在内部对 labels 做一次对齐。契约是
    `labels[i] == input_ids[i]`（当 `token_mask[i] == 1`）。
    """
    if len(input_ids) != len(token_mask):
        raise ValueError(
            f"input_ids 与 token_mask 长度不一致: {len(input_ids)} vs {len(token_mask)}"
        )
    return [
        int(input_ids[i]) if token_mask[i] else ignore_index
        for i in range(len(input_ids))
    ]


def completion_token_count(token_mask: Sequence[int]) -> int:
    """答案 token 数，用于监控掩码是否被全部吃掉。"""
    return sum(1 for m in token_mask if m)


# ---------------------------------------------------------------------------
# rollout 张量的掩码（GRPO 用）
# ---------------------------------------------------------------------------
def completion_end(row: Sequence[int], gen_start: int, stop_ids) -> int:
    """返回一条采样序列中 completion 的右开上界，排除生成后的填充。

    停止 token 计入上界（`j + 1`）；`stop_ids` 须包含 eos、pad 及模板的回合终止符。
    """
    for j in range(gen_start, len(row)):
        if row[j] in stop_ids:
            return j + 1
    return len(row)          # 跑满了 max_new_tokens，没有终止 token


def build_rollout_masks(
    full_ids: Sequence[Sequence[int]],
    *,
    gen_start: int,
    prompt_lens: Sequence[int],
    group_size: int,
    stop_ids,
) -> tuple[list[list[int]], list[list[int]]]:
    """由采样序列计算 `(attention_mask, completion_mask)`。

    输入输出均为纯 list。左 padding 由 `gen_start - prompt_lens[i]` 按位置确定，
    右侧填充由 `completion_end` 确定；停止 token 在两个掩码中均为 1。
    行与 prompt 的对应关系为 prompt-major：第 i 行属于第 `i // group_size` 个 prompt。
    """
    if len(prompt_lens) * group_size != len(full_ids):
        raise ValueError(
            f"prompt 数 ({len(prompt_lens)}) × group_size ({group_size}) "
            f"!= 序列行数 ({len(full_ids)})。生成的行序与 prompt 对不上，"
            f"掩码会串到别的 prompt 上。"
        )

    attention_rows: list[list[int]] = []
    completion_rows: list[list[int]] = []

    for i, row in enumerate(full_ids):
        n = len(row)
        if gen_start > n:
            raise ValueError(
                f"第 {i} 行只有 {n} 个 token，但 completion 从第 {gen_start} 列开始，"
                f"说明 generate 一个 token 都没生成。"
            )
        left_pad = gen_start - int(prompt_lens[i // group_size])
        if left_pad < 0:
            raise ValueError(
                f"第 {i} 行的 prompt 长度 {prompt_lens[i // group_size]} "
                f"超过了输入宽度 {gen_start}，说明喂给 generate 的注意力掩码不对。"
            )
        end = completion_end(row, gen_start, stop_ids)

        # 左侧补齐与右侧填充都不进 attention；停止 token 是真实输出，计入。
        attention_rows.append([0] * left_pad + [1] * (end - left_pad) + [0] * (n - end))
        completion_rows.append(
            [0] * gen_start + [1] * (end - gen_start) + [0] * (n - end)
        )

    return attention_rows, completion_rows


# ---------------------------------------------------------------------------
# prompt 截断
# ---------------------------------------------------------------------------
def truncate_prompt(
    token_ids: Sequence[int],
    max_prompt_length: int,
    *,
    keep_prefix: int = 0,
) -> list[int]:
    """把过长 prompt 压到 `max_prompt_length` 以内，从中间截断。

    保留开头 `keep_prefix` 个 token（system prompt）和结尾的约束描述，中间替换为
    一个省略占位。`keep_prefix=0` 时仅保留结尾。
    """
    n = len(token_ids)
    if n <= max_prompt_length:
        return list(token_ids)
    if max_prompt_length <= 0:
        return []

    prefix = max(0, min(keep_prefix, max_prompt_length))
    suffix_budget = max_prompt_length - prefix

    # `suffix_budget == 0` 时前缀吃满预算，只保留前缀，结果长度恰为 `max_prompt_length`；
    # 否则会超出上限并导致左 padding 宽度与掩码整体错位。
    if suffix_budget <= 0:
        return list(token_ids[:prefix])

    # 省略标记占一位；`suffix_budget == 1` 时它就是全部后缀预算。
    if suffix_budget == 1:
        return [*token_ids[:prefix], ELLIPSIS_TOKEN_ID]

    suffix = list(token_ids[-(suffix_budget - 1):])
    return [*token_ids[:prefix], ELLIPSIS_TOKEN_ID, *suffix]


#: 截断时插入的占位 token id；-1 表示由调用方替换为真实的 unk/省略 token，
#: 避免硬编码 id 在更换 tokenizer 后变成普通词。
ELLIPSIS_TOKEN_ID: int = -1


def resolve_ellipsis(ids: Sequence[int], unk_token_id: int) -> list[int]:
    """把 `ELLIPSIS_TOKEN_ID` 占位替换为 tokenizer 中可用的真实 id。

    与 `truncate_prompt` 分离，使截断逻辑可在无 tokenizer 时测试。
    """
    return [unk_token_id if i == ELLIPSIS_TOKEN_ID else i for i in ids]


# ---------------------------------------------------------------------------
# padding 与长度统计
# ---------------------------------------------------------------------------
def pad_batch(
    sequences: Sequence[Sequence[int]],
    pad_token_id: int,
    *,
    max_length: int | None = None,
    token_masks: Sequence[Sequence[int]] | None = None,
) -> dict:
    """把变长 id 序列右侧补齐成矩形，返回 `input_ids` / `attention_mask` / `token_mask`。

    `attention_mask` 显式生成；传入 `token_masks` 时按相同宽度补 0。
    """
    if not sequences:
        return {"input_ids": [], "attention_mask": [], "token_mask": []}

    longest = max(len(s) for s in sequences)
    width = longest if max_length is None else min(longest, max_length)
    input_ids, attention, masks = [], [], []
    for i, s in enumerate(sequences):
        row = list(s[:width])
        valid = len(row)
        row += [pad_token_id] * (width - valid)
        input_ids.append(row)
        attention.append([1] * valid + [0] * (width - valid))

        if token_masks is not None:
            if len(token_masks[i]) != len(s):
                raise ValueError(
                    f"第 {i} 条序列长度 {len(s)} 与掩码长度 {len(token_masks[i])} 不一致"
                )
            m = list(token_masks[i][:width])
            masks.append(m + [0] * (width - len(m)))
    return {"input_ids": input_ids, "attention_mask": attention, "token_mask": masks}


def pad_left(sequences: Sequence[Sequence[int]], pad_token_id: int) -> dict:
    """把变长 id 序列左侧补齐成矩形，返回 `input_ids` / `attention_mask` / `width`。

    GRPO 生成必须左 padding：所有行的真实 token 右对齐，completion 从固定的
    `width` 列开始；`width` 即 `sample_completions` 中的 `gen_start`。
    """
    if not sequences:
        return {"input_ids": [], "attention_mask": [], "width": 0}

    width = max(len(s) for s in sequences)
    input_ids, attention = [], []
    for s in sequences:
        row = list(s)
        n_pad = width - len(row)
        input_ids.append([pad_token_id] * n_pad + row)
        attention.append([0] * n_pad + [1] * len(row))
    return {"input_ids": input_ids, "attention_mask": attention, "width": width}


def summarize_lengths(
    lengths: Sequence[int], max_length: int, *, warn_ratio: float = 0.05
) -> dict:
    """统计一批样本的长度分布与被截断比例，用于训练前预检。

    `warn_ratio` 为 `should_warn` 的阈值。
    """
    n = len(lengths)
    if n == 0:
        return {"count": 0, "max_length": max_length, "truncated": 0, "truncated_ratio": 0.0}

    over = [x for x in lengths if x > max_length]
    ratio = len(over) / n
    return {
        "count": n,
        "max_length": max_length,
        "min_len": min(lengths),
        "max_len": max(lengths),
        "mean_len": round(sum(lengths) / n, 1),
        "p95_len": sorted(lengths)[min(n - 1, int(n * 0.95))],
        "truncated": len(over),
        "truncated_ratio": round(ratio, 4),
        # 超过阈值就值得打警告，具体阈值由调用方定（不同任务不一样）
        "should_warn": ratio > warn_ratio,
    }


def has_generation_markers(chat_template: str | None) -> bool:
    """检查 chat template 是否包含 `{% generation %}` / `{% endgeneration %}` 标记。

    trl 的 `assistant_only_loss` 依赖该标记，官方模板默认没有。
    """
    if not chat_template:
        return False
    return "{% generation %}" in chat_template and "{% endgeneration %}" in chat_template
