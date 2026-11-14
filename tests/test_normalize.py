"""归一化函数的数值行为。

奖励层唯一一个对每条样本都要算、且结果直接进 loss 的量就是归一化，
所以它是数值风险最集中的地方：一个 inf 或 nan 能毁掉整批梯度。
这组测试锁死三件事——单调性、不动点、极端值不炸。
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from pepdesign.rewards.base import linear_clip, ratio_score, sigmoid_normalize


class TestSigmoidNormalize:
    def test_center_is_half(self):
        """value == center 时必须精确返回 0.5。

        这个不动点不是巧合，是被文档明确承诺过的语义
        （"基准序列得分 0.5"），报告里会直接引用它，所以要锁住。
        """
        assert sigmoid_normalize(120.0, center=120.0, temperature=70.0) == pytest.approx(0.5)
        assert sigmoid_normalize(0.0, center=0.0, temperature=1.0) == pytest.approx(0.5)

    def test_monotonic_increasing(self):
        """必须严格单调递增，否则"更好的序列拿到更低的奖励"。"""
        vals = [sigmoid_normalize(v, center=50.0, temperature=10.0)
                for v in range(-50, 151, 5)]
        assert all(b > a for a, b in pairwise(vals))

    def test_output_in_unit_interval(self):
        for v in (-1e6, -1000, -1, 0, 1, 1000, 1e6):
            s = sigmoid_normalize(v, center=0.0, temperature=1.0)
            assert 0.0 <= s <= 1.0
            assert math.isfinite(s)

    def test_extreme_values_do_not_overflow(self):
        """exp(1000) 会溢出成 inf → nan。实现里对指数做了 ±60 截断，
        这里验证截断真的生效：极端输入只应得到 0 或 1，不能是 nan/inf。
        """
        for v in (1e9, -1e9, 1e300, -1e300, float("inf") * -1):
            for center in (0.0, 1e9, -1e9):
                s = sigmoid_normalize(v, center=center, temperature=0.12)
                assert math.isfinite(s), f"v={v} center={center} 产生了非有限值"
                assert 0.0 <= s <= 1.0

    def test_smaller_temperature_is_steeper(self):
        """temperature 越小越陡——这条方向在 config.py 的注释里曾经写反过，
        所以用数值把它钉死，避免以后再被"修正"回去。

        在 center 的右侧取一点：温度小的应该离 0.5 更远（更接近 1）。
        """
        v, c = 10.0, 0.0
        steep = sigmoid_normalize(v, center=c, temperature=1.0)
        flat = sigmoid_normalize(v, center=c, temperature=50.0)
        assert steep > flat
        assert steep > 0.5 and flat > 0.5

    def test_non_positive_temperature_raises(self):
        """temperature=0 会除零。必须报错，不能静默产生 inf。"""
        for t in (0.0, -1.0):
            with pytest.raises(ValueError):
                sigmoid_normalize(1.0, center=0.0, temperature=t)

    def test_symmetry_around_center(self):
        """以 center 为对称中心：f(c+d) + f(c-d) == 1。"""
        for d in (1.0, 5.0, 20.0):
            a = sigmoid_normalize(50 + d, center=50.0, temperature=12.0)
            b = sigmoid_normalize(50 - d, center=50.0, temperature=12.0)
            assert a + b == pytest.approx(1.0)


class TestLinearClip:
    def test_clips_both_ends(self):
        assert linear_clip(5.0, 0.0, 1.0) == 1.0
        assert linear_clip(-5.0, 0.0, 1.0) == 0.0

    def test_passes_through_in_range(self):
        assert linear_clip(0.42, 0.0, 1.0) == 0.42
        assert linear_clip(0.0, -1.0, 1.0) == 0.0

    def test_boundaries_inclusive(self):
        assert linear_clip(0.0, 0.0, 1.0) == 0.0
        assert linear_clip(1.0, 0.0, 1.0) == 1.0


class TestRatioScore:
    def test_baseline_is_half(self):
        """value == baseline → 0.5。这个不动点让"基准序列得 0.5"可直接写进文档。"""
        assert ratio_score(45.0, baseline=45.0) == pytest.approx(0.5)

    def test_higher_is_better_direction(self):
        assert ratio_score(60.0, baseline=45.0) > 0.5
        assert ratio_score(30.0, baseline=45.0) < 0.5

    def test_lower_is_better_flips_direction(self):
        """酶切位点数这类"越少越好"的量，方向必须能翻过来。"""
        assert ratio_score(1.0, baseline=5.0, higher_is_better=False) > 0.5
        assert ratio_score(9.0, baseline=5.0, higher_is_better=False) < 0.5

    def test_range(self):
        for v in (-1e6, 0.0, 45.0, 1e6):
            assert 0.0 <= ratio_score(v, baseline=45.0) <= 1.0

    def test_scale_decouples_from_baseline(self):
        """scale 与 baseline 解耦是本项目修过的一个真实问题。

        同一个点，在同一个 baseline 下，换 scale 只应改变陡度；
        而 baseline 变了、scale 明确给出时，陡度不应该跟着变。
        旧实现把 scale 默认成 |baseline|，于是"把基准从 45 调到 60"
        会顺带把曲线斜率改掉 4/3 倍，两次实验的奖励尺度不可比。
        """
        # 同 baseline 同偏移，显式 scale 不同 → 得分不同（陡度确实在起作用）
        near = ratio_score(55.0, baseline=45.0, scale=10.0)
        far = ratio_score(55.0, baseline=45.0, scale=100.0)
        assert near > far

        # baseline 变了但显式 scale 不变 → 偏移量相同则得分相同
        a = ratio_score(55.0, baseline=45.0, scale=10.0)
        b = ratio_score(65.0, baseline=55.0, scale=10.0)
        assert a == pytest.approx(b)

    def test_baseline_zero_does_not_divide_by_zero(self):
        """baseline 为 0 时 |baseline| 会让 scale 变成 0 → 除零。
        实现里有 max(s, 1e-6) 兜底，这里验证它生效。"""
        s = ratio_score(5.0, baseline=0.0)
        assert math.isfinite(s)
        assert s > 0.5
