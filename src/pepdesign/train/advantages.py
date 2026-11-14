"""GRPO 的数值层：KL 估计、裁剪区间、损失归一化与优势广播。

本模块不 import torch，全部为纯 Python 运算，便于手算核对与无 GPU 测试。

`GRPOConfig.advantage_normalization` 支持三种口径：`std` 为标准 GRPO；
`mean` 按优势 RMS 缩放，近似 `(r-mean)/|mean|`；`none` 只去均值。
无效序列不参与均值/标准差计算，但在返回列表中占位为 0.0。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

#: 判定"分母退化"的阈值。低于它时该组没有可用离散度，返回全零优势。
_DENOM_EPS: float = 1e-6

#: DAPO 的默认上界放宽系数。放松上界、保留下界的裁剪可缓解熵塌缩，
#: 让优势为正的 token 继续上调。
DEFAULT_CLIP_HIGHER: float = 0.28


# ---------------------------------------------------------------------------
# KL 估计
# ---------------------------------------------------------------------------
def kl_penalty(
    policy_logps: Sequence[float], ref_logps: Sequence[float]
) -> list[float]:
    """逐 token 的 KL 惩罚，使用 Schulman 的 k3 估计子 `exp(d) - d - 1`。

    其中 `d = logπ_ref − logπ_θ`。k3 恒非负、恒有梯度，可直接作为惩罚项；
    指数在 [-30, 30] 内截断以防溢出。入参为逐 token 对数概率，遮掉的位置
    由调用方先填 0，本函数不处理掩码。
    """
    if len(policy_logps) != len(ref_logps):
        raise ValueError(
            f"policy/ref 对数概率长度不一致: {len(policy_logps)} vs {len(ref_logps)}。"
            f"这通常说明生成时的 batch 顺序和重算 logprob 时的顺序没对齐。"
        )
    out: list[float] = []
    for lp, rp in zip(policy_logps, ref_logps, strict=True):
        d = rp - lp
        # exp 溢出保护：d 过大时 exp(d) 会变成 inf，一个 inf 就能让整批 loss 变成 nan。
        # 截断到 30 即视为 KL 已大到无法区分。
        d = max(-30.0, min(30.0, d))
        out.append(math.exp(d) - d - 1.0)
    return out


def policy_ratio(new_logps: Sequence[float], old_logps: Sequence[float]) -> list[float]:
    """重要性采样比 `exp(logπ_new − logπ_old)`。

    必须先做差再取指数：分别取指数会溢出或下溢，且浮点误差更大。
    """
    if len(new_logps) != len(old_logps):
        raise ValueError(
            f"new/old 对数概率长度不一致: {len(new_logps)} vs {len(old_logps)}"
        )
    return [math.exp(n - o) for n, o in zip(new_logps, old_logps, strict=True)]


# ---------------------------------------------------------------------------
# 裁剪区间
# ---------------------------------------------------------------------------
def clip_bounds(
    loss_type: str, epsilon: float = 0.2, clip_higher: float = DEFAULT_CLIP_HIGHER
) -> tuple[float, float]:
    """返回 PPO/GRPO 重要性比裁剪的 `(下界, 上界)`。

    四种 `loss_type` 中仅 `dapo` 使用非对称裁剪，其余为对称的 `1±ε`。
    """
    if epsilon <= 0:
        raise ValueError(f"clip 的 epsilon 必须为正，收到 {epsilon}")
    if loss_type == "dapo":
        return 1.0 - epsilon, 1.0 + clip_higher
    return 1.0 - epsilon, 1.0 + epsilon


def clipped_surrogate(
    ratio: float, advantage: float, low: float, high: float
) -> tuple[float, float]:
    """单个 token 的裁剪替代目标，返回 `(未裁剪项, 裁剪项)`。

    目标为 `min(ratio·A, clip(ratio, low, high)·A)`，即悲观下界。拆成两项是
    为了计算 `clip_fraction = mean(未裁剪 != 裁剪)`。本纯函数仅用于日志与
    无 GPU 测试；反传使用 `train/grpo.py` 中的 torch 实现，两者数值由
    冒烟测试断言一致。
    """
    unclipped = ratio * advantage
    clipped = max(low, min(high, ratio)) * advantage
    return unclipped, min(unclipped, clipped)


# ---------------------------------------------------------------------------
# 损失归一化
# ---------------------------------------------------------------------------
def advantage_mode(advantage_normalization: str, scale_rewards: bool = True) -> str:
    """按 `scale_rewards` 与 `advantage_normalization` 决定最终优势口径。

    `scale_rewards=False` 一律映射为 `none`；配置本身为 `none` 时不产生警告。
    """
    if not scale_rewards:
        return "none"
    if advantage_normalization not in ("std", "mean", "none"):
        raise ValueError(
            f"未知的 advantage_normalization: {advantage_normalization!r}，"
            f"可选 std / mean / none"
        )
    return advantage_normalization


def normalize_advantages(advantages: Sequence[float], mode: str) -> list[float]:
    """按口径缩放组内优势；入参须为已去均值的优势。

    `std` 与 `none` 原样返回；`mean` 除以优势的 RMS，作为 `(r-mean)/|mean|`
    的近似。RMS 低于 `_DENOM_EPS` 时返回全零。
    """
    if mode == "std" or mode == "none" or not advantages:
        return list(advantages)

    if mode != "mean":
        raise ValueError(f"未知的优势口径: {mode!r}，可选 std / mean / none")

    rms = math.sqrt(sum(a * a for a in advantages) / len(advantages))
    if rms < _DENOM_EPS:
        return [0.0] * len(advantages)
    return [a / rms for a in advantages]


# ---------------------------------------------------------------------------
# 逐 token 聚合
# ---------------------------------------------------------------------------
def masked_mean(values: Sequence[float], mask: Sequence[int]) -> float:
    """只在 mask 为 1 的位置求平均。

    分母为有效位置数而非 `len(values)`，避免 padding 引入长度偏置；
    全部被遮掉时返回 0.0，不抛异常。
    """
    if len(values) != len(mask):
        raise ValueError(f"values 与 mask 长度不一致: {len(values)} vs {len(mask)}")
    total, count = 0.0, 0
    for v, m in zip(values, mask, strict=True):
        if m:
            total += v
            count += 1
    return total / count if count else 0.0


def loss_denominator(
    loss_type: str,
    *,
    token_mask: Sequence[Sequence[int]],
    max_completion_length: int,
) -> float:
    """损失函数的分母，口径由 `loss_type` 决定。

    `token_mask` 为整个 batch 的掩码（外层样本、内层 token）。`grpo` 返回样本数
    （调用方须先做序列内平均）；`bnpo` / `dapo` 返回有效 token 总数；
    `dr_grpo` 返回 `样本数 × max_completion_length` 这一固定常数。
    """
    n_samples = len(token_mask)
    if n_samples == 0:
        return 1.0

    if loss_type == "grpo":
        return float(n_samples)
    if loss_type in ("bnpo", "dapo"):
        return float(sum(sum(1 for m in row if m) for row in token_mask)) or 1.0
    if loss_type == "dr_grpo":
        return float(n_samples * max(max_completion_length, 1))

    raise ValueError(
        f"未知的 loss_type: {loss_type!r}，可选 grpo / dr_grpo / bnpo / dapo"
    )


def broadcast_advantages(
    advantages: Sequence[float], token_mask: Sequence[Sequence[int]]
) -> list[list[float]]:
    """把序列级优势铺开成逐 token 优势，遮掉的位置填 0.0。

    掩码必须与 advantages 的顺序一致（prompt-major）；长度不匹配直接报错。
    """
    if len(advantages) != len(token_mask):
        raise ValueError(
            f"优势条数 ({len(advantages)}) 与掩码行数 ({len(token_mask)}) 不一致。"
            f"常见原因：采样顺序（prompt-major 还是 sample-major）与打分顺序不一致。"
        )
    return [
        [float(a) if m else 0.0 for m in row]
        for a, row in zip(advantages, token_mask, strict=True)
    ]
