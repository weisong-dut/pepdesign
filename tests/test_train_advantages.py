"""GRPO 数值层：KL 估计、裁剪区间、损失分母、优势广播。

断言均为可用计算器核对的具体数值；这些位置的错误通常不报错，
只会让训练朝错误方向进行。本文件不 import torch，torch 实现与纯函数
版本的一致性由 CPU 冒烟测试对比。
"""

from __future__ import annotations

import math

import pytest

from pepdesign.train.advantages import (
    advantage_mode,
    broadcast_advantages,
    clip_bounds,
    clipped_surrogate,
    kl_penalty,
    loss_denominator,
    masked_mean,
    normalize_advantages,
    policy_ratio,
)

# ---------------------------------------------------------------------------
# KL：k3 估计子
# ---------------------------------------------------------------------------


def test_kl_is_zero_when_policy_equals_reference():
    """π_θ == π_ref 时 KL 精确为 0；否则训练一开始就会有恒定偏置。"""
    logps = [-0.5, -1.2, -2.0, -0.3]
    kl = kl_penalty(logps, logps)
    assert kl == [0.0, 0.0, 0.0, 0.0]


def test_kl_is_always_nonnegative():
    """逐 token 恒非负；朴素差值 `logπ_θ − logπ_ref` 无下界，会产生负 KL 日志。"""
    pairs = [
        (-0.1, -3.0),   # 策略比参考差很多
        (-3.0, -0.1),   # 策略比参考好很多
        (-1.0, -1.0),   # 相同
        (-0.5, -0.5001),
        (-0.5, -0.4999),
    ]
    for lp, rp in pairs:
        (k,) = kl_penalty([lp], [rp])
        assert k >= 0.0, f"k3 在 ({lp}, {rp}) 上为负: {k}"


def test_kl_matches_hand_computed_values():
    """手算核对：d = logπ_ref − logπ_θ，k3 = exp(d) − d − 1。

    方向必须是参考减当前；用不对称的一组数把方向钉死。
    """
    policy = [-0.5, -2.0]
    ref = [-1.5, -1.0]      # d = -1.0 和 +1.0，一正一负，方向写反必然被发现

    kl = kl_penalty(policy, ref)

    for got, (lp, rp) in zip(kl, zip(policy, ref, strict=True), strict=True):
        d = rp - lp
        assert got == pytest.approx(math.exp(d) - d - 1.0, abs=1e-12)

    # d = -1: exp(-1) + 1 - 1 = 0.3679；d = +1: exp(1) - 1 - 1 = 0.7183
    assert kl[0] == pytest.approx(0.36787944, abs=1e-6)
    assert kl[1] == pytest.approx(0.71828183, abs=1e-6)


def test_kl_clamps_extreme_values():
    """极端 d 被截断到 ±30，避免 exp 溢出成 inf。

    截断的是 exp 的自变量而非整个惩罚项；方向保留，偏离越远惩罚越大。
    """
    (far,) = kl_penalty([0.0], [-1000.0])       # d = -1000 → 夹到 -30
    (near,) = kl_penalty([-1000.0], [0.0])      # d = +1000 → 夹到 +30
    assert math.isfinite(far) and math.isfinite(near)
    assert far == pytest.approx(math.exp(-30) + 30 - 1, rel=1e-9)
    assert near == pytest.approx(math.exp(30) - 30 - 1, rel=1e-6)
    # 两者都远大于 1，都还是"很大的惩罚"
    assert far > 20 and near > 20


def test_kl_rejects_length_mismatch():
    with pytest.raises(ValueError, match="长度不一致"):
        kl_penalty([-1.0, -2.0], [-1.0])


# ---------------------------------------------------------------------------
# 重要性比
# ---------------------------------------------------------------------------


def test_policy_ratio_is_one_at_start_and_direction_correct():
    """同策略时比值恒为 1；new > old 时比值 > 1。"""
    assert policy_ratio([-1.0, -2.0], [-1.0, -2.0]) == [1.0, 1.0]
    # new 的概率更大（对数概率更接近 0）→ 比值 > 1
    assert policy_ratio([-0.5], [-1.0])[0] == pytest.approx(math.e ** 0.5)
    assert policy_ratio([-1.5], [-1.0])[0] < 1.0


def test_policy_ratio_does_not_overflow():
    """必须先做差再取指数：`exp(new)/exp(old)` 在极端值下会得到 0/0。"""
    assert policy_ratio([-1000.0], [-1001.0])[0] == pytest.approx(math.e, rel=1e-9)


# ---------------------------------------------------------------------------
# 裁剪
# ---------------------------------------------------------------------------


def test_clip_bounds_symmetric_except_dapo():
    """默认对称 1±ε；只有 dapo 放宽上界，用于缓解熵塌缩。"""
    assert clip_bounds("grpo", 0.2) == (0.8, 1.2)
    assert clip_bounds("dr_grpo", 0.2) == (0.8, 1.2)
    assert clip_bounds("bnpo", 0.2) == (0.8, 1.2)
    assert clip_bounds("dapo", 0.2, clip_higher=0.28) == (0.8, 1.28)

    with pytest.raises(ValueError, match="epsilon"):
        clip_bounds("grpo", 0.0)


def test_clipped_surrogate_is_pessimistic_bound():
    """`min(未裁剪, 裁剪)` 是悲观下界，优势为正和为负时都要验证。"""
    lo, hi = 0.8, 1.2

    # 优势为正、比值超出上界：取裁剪后的值（比未裁剪的小）
    unclipped, chosen = clipped_surrogate(2.0, advantage=1.0, low=lo, high=hi)
    assert unclipped == 2.0
    assert chosen == 1.2

    # 优势为负、比值超出上界：未裁剪项 = 2.0 × (-1) = -2，裁剪项 = 1.2 × (-1) = -1.2
    # min(-2, -1.2) = -2，也就是限制下跌（不让它跌得更狠）
    unclipped, chosen = clipped_surrogate(2.0, advantage=-1.0, low=lo, high=hi)
    assert unclipped == -2.0
    assert chosen == -2.0

    # 比值落在区间内：两项相同，没有被裁
    unclipped, chosen = clipped_surrogate(1.0, advantage=1.0, low=lo, high=hi)
    assert unclipped == chosen == 1.0


def test_pessimistic_min_clips_only_one_direction():
    """`min(未裁剪, 裁剪)` 只在上界一侧裁剪，下界一侧不裁。

    因此 `clip_fraction` 定义为比值落在 [lo, hi] 之外的 token 占比。
    """
    lo, hi = 0.8, 1.2

    # 上界外：被裁
    unclipped, chosen = clipped_surrogate(2.0, 1.0, lo, hi)
    assert chosen != unclipped and chosen == hi

    # 下界外：min 取的是未裁剪项，所以"梯度被裁"这件事没有发生
    unclipped, chosen = clipped_surrogate(0.5, 1.0, lo, hi)
    assert chosen == unclipped == 0.5

    # 区间内：两项相同
    unclipped, chosen = clipped_surrogate(1.0, 1.0, lo, hi)
    assert unclipped == chosen == 1.0


# ---------------------------------------------------------------------------
# 优势口径
# ---------------------------------------------------------------------------


def test_advantage_mode_scale_rewards_wins():
    """`scale_rewards=False` 一律钉死为 none，两个开关同时生效时优先级必须明确。"""
    assert advantage_mode("std", scale_rewards=True) == "std"
    assert advantage_mode("mean", scale_rewards=True) == "mean"
    assert advantage_mode("none", scale_rewards=True) == "none"
    assert advantage_mode("std", scale_rewards=False) == "none"
    assert advantage_mode("mean", scale_rewards=False) == "none"

    with pytest.raises(ValueError, match="advantage_normalization"):
        advantage_mode("variance", scale_rewards=True)


def test_normalize_advantages_passthrough_and_scale():
    adv = [1.0, -1.0, 2.0, -2.0]
    # std / none 原样返回：std 已经标准化过，none 就是不做缩放
    assert normalize_advantages(adv, "std") == adv
    assert normalize_advantages(adv, "none") == adv
    # 空列表不炸
    assert normalize_advantages([], "mean") == []

    scaled = normalize_advantages(adv, "mean")
    rms = math.sqrt(sum(a * a for a in adv) / len(adv))
    for got, a in zip(scaled, adv, strict=True):
        assert got == pytest.approx(a / rms)

    # 全零输入（退化组）必须返回全零，不能除以 0
    assert normalize_advantages([0.0, 0.0], "mean") == [0.0, 0.0]

    with pytest.raises(ValueError, match="未知的优势口径"):
        normalize_advantages(adv, "rms")


# ---------------------------------------------------------------------------
# 损失分母：长度偏置的来源
# ---------------------------------------------------------------------------


def test_loss_denominator_four_modes():
    """四种口径的分母逐个核对。

    `grpo` 用样本数，`bnpo`/`dapo` 用有效 token 数，
    `dr_grpo` 用样本数 × 固定长度。
    """
    mask = [[1, 1, 0, 0], [1, 1, 1, 1]]
    max_len = 4

    # grpo：分母是样本数（每条序列内部已经各自除过自己的 token 数）
    assert loss_denominator("grpo", token_mask=mask, max_completion_length=max_len) == 2.0
    # bnpo / dapo：有效 token 总数 = 2 + 4 = 6
    assert loss_denominator("bnpo", token_mask=mask, max_completion_length=max_len) == 6.0
    assert loss_denominator("dapo", token_mask=mask, max_completion_length=max_len) == 6.0
    # dr_grpo：样本数 × 固定长度 = 2 × 4 = 8，与有效 token 数无关
    assert loss_denominator("dr_grpo", token_mask=mask, max_completion_length=max_len) == 8.0

    with pytest.raises(ValueError, match="未知的 loss_type"):
        loss_denominator("ppo", token_mask=mask, max_completion_length=max_len)


def test_loss_denominator_dr_grpo_is_independent_of_valid_tokens():
    """dr_grpo 的分母是常数，这是它作为默认值的理由：loss 曲线跨 batch 可比。"""
    short = [[1, 1, 0, 0]]
    long = [[1, 1, 1, 1]]
    d1 = loss_denominator("dr_grpo", token_mask=short, max_completion_length=4)
    d2 = loss_denominator("dr_grpo", token_mask=long, max_completion_length=4)
    assert d1 == d2 == 4.0
    # bnpo 则会随有效 token 数变化
    assert loss_denominator("bnpo", token_mask=short, max_completion_length=4) == 2.0
    assert loss_denominator("bnpo", token_mask=long, max_completion_length=4) == 4.0


def test_loss_denominator_empty_batch():
    assert loss_denominator("grpo", token_mask=[], max_completion_length=4) == 1.0
    # 全被遮掉时不能除以 0
    assert loss_denominator("bnpo", token_mask=[[0, 0]], max_completion_length=2) == 1.0


# ---------------------------------------------------------------------------
# masked_mean / broadcast
# ---------------------------------------------------------------------------


def test_masked_mean_ignores_masked_positions():
    """分母是有效位置数，不是 len(values)；否则 padding 会造成隐蔽的长度偏置。"""
    assert masked_mean([1.0, 2.0, 3.0, 4.0], [1, 1, 0, 0]) == 1.5
    # 被遮掉的位置上的值再离谱也不影响结果
    assert masked_mean([1.0, 2.0, 1e9, -1e9], [1, 1, 0, 0]) == 1.5
    # 全部遮掉返回 0 而不是 nan：答案被截断干净是可能的，不该让整个 step 挂掉
    assert masked_mean([5.0, 6.0], [0, 0]) == 0.0
    assert masked_mean([], []) == 0.0

    with pytest.raises(ValueError, match="长度不一致"):
        masked_mean([1.0, 2.0], [1])


def test_broadcast_advantages_uses_matching_mask():
    """优势铺开成 token 级，遮掉的位置补 0，避免 padding 参与损失。"""
    out = broadcast_advantages([1.5, -0.5], [[1, 1, 0], [1, 0, 0]])
    assert out == [[1.5, 1.5, 0.0], [-0.5, 0.0, 0.0]]

    with pytest.raises(ValueError, match="不一致"):
        broadcast_advantages([1.5, 2.0], [[1, 1]])
