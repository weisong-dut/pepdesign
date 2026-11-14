"""GRPO 组内优势 `A_i = (r_i - mean(r)) / std(r)`。

覆盖三条不报错的失败模式：std≈0 时得到 nan/inf、样本被静默丢弃、
无效序列混入组内基线。断言校验具体数值，而非只要求不崩溃。
"""

from __future__ import annotations

import math
import random

import pytest

from pepdesign.bio.aa import STANDARD_AA
from pepdesign.rewards.base import RewardBreakdown
from pepdesign.rewards.composite import ADVANTAGE_STD_EPS, CompositeScorer
from pepdesign.utils.config import RewardConfig


def breakdown(total: float, *, invalid: str | None = None) -> RewardBreakdown:
    """手搓一条打分结果，用于精确控制组内的奖励分布。"""
    return RewardBreakdown(
        sequence="KWKLFKKIEK", total=total, dimensions={}, penalties={},
        invalid_reason=invalid,
    )


def random_group(n: int, seed: int, lo: int = 8, hi: int = 18) -> list[str]:
    """确定性随机多肽组；固定 seed 避免偶发 flaky。"""
    rng = random.Random(seed)
    return [
        "".join(rng.choice(STANDARD_AA) for _ in range(rng.randint(lo, hi)))
        for _ in range(n)
    ]


@pytest.fixture(scope="module")
def scorer() -> CompositeScorer:
    return CompositeScorer(RewardConfig())


class TestStandardization:
    def test_advantages_are_standardized(self):
        """z-score 的两个定义性质：均值 0、标准差 1。"""
        adv = CompositeScorer().group_advantages([breakdown(t) for t in (1.0, 2.0, 3.0, 4.0)])
        assert sum(adv) / len(adv) == pytest.approx(0.0, abs=1e-6)
        assert math.sqrt(sum(a * a for a in adv) / len(adv)) == pytest.approx(1.0, abs=1e-6)

    def test_order_is_preserved(self):
        """优势必须与奖励同序——排序反了等于在惩罚好序列。"""
        adv = CompositeScorer().group_advantages([breakdown(t) for t in (0.1, 0.5, 0.9)])
        assert adv == sorted(adv)

    def test_better_reward_gives_positive_advantage(self):
        adv = CompositeScorer().group_advantages([breakdown(t) for t in (0.1, 0.5, 0.9)])
        assert adv[0] < 0 < adv[-1]
        assert adv[-1] > 0

    def test_uses_population_variance_not_sample_variance(self):
        """分母是 N 而不是 N-1：组内候选即全体，G=8 时用样本标准差会让优势偏大约 7%。"""
        adv = CompositeScorer().group_advantages([breakdown(t) for t in (1.0, 2.0, 3.0, 4.0)])
        assert math.sqrt(sum(a * a for a in adv) / len(adv)) == pytest.approx(1.0, abs=1e-6)

    def test_advantages_are_rounded_not_raw_floats(self):
        """优势会被写进日志和 JSONL，无限小数既没必要也不稳定。"""
        adv = CompositeScorer().group_advantages([breakdown(t) for t in (0.1, 0.5, 0.9)])
        assert all(a == round(a, 6) for a in adv)


class TestDegenerateGroups:
    """退化情形：三种都会毁掉梯度，必须显式挡住。"""

    def test_identical_rewards_return_all_zeros_not_nan(self):
        """std≈0 时返回全零而不是 nan/inf；后者会一步污染整个 batch 的梯度。"""
        adv = CompositeScorer().group_advantages([breakdown(0.42)] * 8)
        assert adv == [0.0] * 8
        assert not any(math.isnan(a) or math.isinf(a) for a in adv)

    def test_near_identical_rewards_also_collapse(self):
        """浮点噪声级（1e-9）的差异同样按退化处理；EPS 兜住 round 后的尾差。"""
        assert ADVANTAGE_STD_EPS > 0
        adv = CompositeScorer().group_advantages(
            [breakdown(0.4), breakdown(0.4 + 1e-9), breakdown(0.4)]
        )
        assert adv == [0.0, 0.0, 0.0]

    def test_all_invalid_returns_all_zeros(self):
        """全是无效序列时组内没有任何有效信号，返回全零。"""
        scores = [breakdown(-1.0, invalid="parse_failed") for _ in range(8)]
        assert CompositeScorer().group_advantages(scores) == [0.0] * 8

    def test_empty_group_returns_empty_list(self):
        assert CompositeScorer().group_advantages([]) == []

    def test_single_sample_group_is_degenerate(self):
        """G=1 时 std 必然是 0，不能变成 0/0。"""
        adv = CompositeScorer().group_advantages([breakdown(0.7)])
        assert adv == [0.0]

    def test_all_advantages_finite_for_zero_rewards(self):
        """组内奖励全为 0 也是标准退化情形，返回全零。"""
        adv = CompositeScorer().group_advantages([breakdown(0.0)] * 8)
        assert adv == [0.0] * 8


class TestInvalidExclusion:
    def test_invalid_entries_do_not_enter_mean_or_std(self):
        """无效序列占位为 0，但不参与 mean/std；否则会系统性拉低基线。"""
        scores = [
            breakdown(1.0),
            breakdown(2.0),
            breakdown(3.0),
            breakdown(-1.0, invalid="parse_failed"),
        ]
        adv = CompositeScorer().group_advantages(scores)

        valid = [1.0, 2.0, 3.0]
        mean = sum(valid) / 3
        std = math.sqrt(sum((v - mean) ** 2 for v in valid) / 3)
        assert adv[:3] == [round((v - mean) / std, 6) for v in valid]
        assert adv[3] == 0.0

    def test_invalid_entries_keep_their_slots(self):
        """返回列表必须与输入等长且逐位对应；短一个会让上层优势全体错位。"""
        scores = [breakdown(t, invalid=("bad" if i % 2 else None))
                  for i, t in enumerate([0.1, 0.9, 0.4, 0.7])]
        adv = CompositeScorer().group_advantages(scores)
        assert len(adv) == len(scores)
        assert adv[1] == adv[3] == 0.0

    def test_one_valid_sample_among_invalids_is_degenerate(self):
        """只有一个有效样本 → 组内 std=0 → 全零，而不是让那一条拿到巨大优势。"""
        scores = [breakdown(0.5)] + [breakdown(-1.0, invalid="x") for _ in range(3)]
        assert CompositeScorer().group_advantages(scores) == [0.0] * 4


class TestEndToEndWithRealScorer:
    """用真实打分器跑完整链路，验证训练是否会产生非零优势。"""

    def test_random_group_produces_nonzero_advantages(self):
        """回归测试：8 条随机多肽的组内优势必须有非零标准差。

        曾因 `score_group` 的 zip 截断只返回 1 条分数，导致优势全零。
        """
        group = random_group(8, seed=0)
        scores = CompositeScorer().score_group(group, [{"task": "de_novo"}])

        assert len(scores) == 8, "组内样本被丢了——检查 score_group 的 meta 广播"
        totals = [s.total for s in scores]
        mean = sum(totals) / len(totals)
        std = math.sqrt(sum((t - mean) ** 2 for t in totals) / len(totals))
        assert std > 0.01, f"组内奖励几乎没有区别（std={std:.6f}），优势会退化成全零"

        adv = CompositeScorer().group_advantages(scores)
        assert any(a != 0.0 for a in adv), "优势全零 = 这一步没有梯度"

    def test_advantages_standardized_across_seeds(self):
        c = CompositeScorer()
        for seed in range(4):
            scores = c.score_group(random_group(8, seed=seed), [{"task": "de_novo"}])
            adv = c.group_advantages(scores)
            assert len(adv) == 8
            assert all(math.isfinite(a) for a in adv), f"seed={seed} 出现了非有限值"
            assert sum(adv) / len(adv) == pytest.approx(0.0, abs=1e-4)
            assert math.sqrt(sum(a * a for a in adv) / len(adv)) == pytest.approx(1.0, abs=1e-3)

    def test_no_nan_anywhere_in_the_pipeline(self):
        """一个 nan 就能毁掉整批梯度——奖励链路上每一环都要挡住它。"""
        c = CompositeScorer()
        group = random_group(8, seed=2)
        for s in c.score_group(group, [{"task": "de_novo"}]):
            assert math.isfinite(s.total)
            for name, d in s.dimensions.items():
                assert math.isfinite(d.value), f"{name} 给出了非有限值"

    def test_invalid_sequence_keeps_slot_but_leaves_the_baseline(self):
        """端到端回归：含非法字符的样本被排除出 mean/std，但仍占住自己的位置。"""
        group = random_group(7, seed=11)
        group.insert(3, "KWK!LFKK")  # 掺一条含非法字符的
        scores = CompositeScorer().score_group(group, [{"task": "de_novo"}])

        assert len(scores) == 8, "无效样本也必须占住自己的位置"
        assert scores[3].invalid_reason is not None, "无效样本必须被标出来"

        adv = CompositeScorer().group_advantages(scores)
        assert adv[3] == 0.0, "无效样本自己不该拿到优势"
        # 均值/标准差只由剩下 7 条有效样本决定——手算核对，确保它没被算进去
        valid = [s.total for i, s in enumerate(scores) if i != 3]
        mean = sum(valid) / len(valid)
        std = math.sqrt(sum((v - mean) ** 2 for v in valid) / len(valid))
        expected = [round((v - mean) / std, 6) for v in valid]
        got = [a for i, a in enumerate(adv) if i != 3]
        assert got == pytest.approx(expected)

    def test_group_of_identical_sequences_is_safely_degenerate(self, scorer):
        """模型收敛到组内同一条序列时，优势全零但不崩，这是训练后期的正常状态。"""
        scores = scorer.score_group(["KWKLFKKIEKVK"] * 8, [{"task": "de_novo"}])
        assert [s.total for s in scores] == [scores[0].total] * 8
        assert scorer.group_advantages(scores) == [0.0] * 8


# ---------------------------------------------------------------------------
# advantage_normalization 的三种口径（GRPOConfig.advantage_normalization）
# ---------------------------------------------------------------------------
class TestAdvantageModes:
    """`group_advantages(mode=...)` 的三种口径。

    三种口径共用"无效序列占位为 0"和"退化组返回全零"两条逻辑，
    因此放在一个函数里按 mode 分流，避免多份实现漂移。
    """

    def test_std_is_the_default_and_matches_hand_computation(self, scorer):
        scores = [breakdown(t) for t in (0.9, 0.1)]
        # 默认不传 mode，行为必须和显式传 "std" 完全一致
        assert scorer.group_advantages(scores) == scorer.group_advantages(scores, mode="std")

        # 均值 0.5、总体标准差 0.4 → ±1.0
        assert scorer.group_advantages(scores, mode="std") == [1.0, -1.0]

    def test_mean_divides_by_absolute_mean(self, scorer):
        """`(r - mean) / |mean|`，不是除以 std；两种口径的结果必须不同。"""
        scores = [breakdown(t) for t in (0.9, 0.1)]
        assert scorer.group_advantages(scores, mode="mean") == [0.8, -0.8]
        # 和 std 口径确实不同——相同就说明 mode 没被用上
        assert scorer.group_advantages(scores, mode="mean") != scorer.group_advantages(
            scores, mode="std"
        )

    def test_mean_uses_absolute_value_so_sign_is_preserved(self, scorer):
        """分母取 `|mean|`：均值本身的正负不该翻转优势符号。

        写成 `(r - mean) / mean` 时，一组全为负的奖励会让所有优势符号反转。
        """
        scores = [breakdown(t) for t in (-0.1, -0.9)]   # 均值 -0.5，分高的在前
        adv = scorer.group_advantages(scores, mode="mean")
        assert adv[0] > 0 > adv[1], "奖励高的一条必须拿到正优势"
        assert adv == [0.8, -0.8]

    def test_none_only_subtracts_the_mean(self, scorer):
        """只去均值、不做任何缩放。"""
        scores = [breakdown(t) for t in (0.9, 0.1)]
        assert scorer.group_advantages(scores, mode="none") == [0.4, -0.4]

    def test_all_modes_keep_invalid_slots_and_share_the_baseline(self, scorer):
        """三种口径下，无效序列都占位为 0 且不参与基线。"""
        scores = [breakdown(0.9), breakdown(0.0, invalid="parse_failed:x"), breakdown(0.1)]
        for mode in ("std", "mean", "none"):
            adv = scorer.group_advantages(scores, mode=mode)
            assert adv[1] == 0.0, f"{mode} 口径下无效序列没有占位"
            assert adv[0] > 0 > adv[2], f"{mode} 口径下有效序列的优势方向不对"

    def test_all_modes_return_zeros_for_degenerate_group(self, scorer):
        """组内奖励完全相同 → 三种口径都返回全零、都不是 nan。"""
        scores = [breakdown(0.42)] * 4
        for mode in ("std", "mean", "none"):
            adv = scorer.group_advantages(scores, mode=mode)
            assert adv == [0.0] * 4, f"{mode} 口径下退化组没有返回全零"
            assert all(a == a for a in adv), "出现了 nan"

    def test_mean_mode_degrades_when_the_mean_is_zero(self, scorer):
        """`mean` 口径特有的退化：均值为 0 时 `(r-mean)/|mean|` 会发散。"""
        scores = [breakdown(0.5), breakdown(-0.5)]
        assert scorer.group_advantages(scores, mode="mean") == [0.0, 0.0]
        # std 口径下同一组数据是正常的（标准差 0.5）
        assert scorer.group_advantages(scores, mode="std") == [1.0, -1.0]

    def test_all_invalid_group_returns_zeros_in_every_mode(self, scorer):
        scores = [breakdown(0.0, invalid="empty_sequence")] * 3
        for mode in ("std", "mean", "none"):
            assert scorer.group_advantages(scores, mode=mode) == [0.0] * 3

    def test_unknown_mode_raises_instead_of_falling_back(self, scorer):
        """未知口径必须报错，不能静默退回 std；否则配置拼错会长期不可见。"""
        with pytest.raises(ValueError, match="未知的优势口径"):
            scorer.group_advantages([breakdown(0.5), breakdown(0.1)], mode="variance")

    def test_empty_group(self, scorer):
        assert scorer.group_advantages([], mode="std") == []
