"""酶解稳定性预测器：把 `bio/cleavage` 的规则层包装成预测器接口。

统一接口后，奖励函数无需关心位点由规则还是模型得到，也便于替换为深度学习
位点预测器。输出为加权酶切位点数（物理量），归一化留给奖励层。
"""

from __future__ import annotations

from pepdesign.bio.cleavage import PROTEASE_WEIGHTS, analyze
from pepdesign.rewards.predictors.base import BasePredictor, PredictionResult


class ProteaseSitesPredictor(BasePredictor):
    """血清蛋白酶酶切位点预测器（规则层）。"""

    name = "protease_rule"

    def __init__(self, *, modifications: tuple[str, ...] = (), **kwargs) -> None:
        super().__init__(**kwargs)
        # 末端修饰会屏蔽外切酶位点，必须进入缓存 key。
        self.modifications = tuple(modifications)

    def _cache_key(self, sequence: str) -> str:
        return f"{sequence}|{','.join(self.modifications)}"

    def _predict_one(self, sequence: str) -> PredictionResult:
        report = analyze(sequence, modifications=self.modifications)

        # 规则层确定但只覆盖 6 种常见血清蛋白酶，置信度不打满。
        coverage_factor = 0.85
        length_factor = min(1.0, len(sequence) / 6.0)

        return PredictionResult(
            value=round(report.weighted_site_count, 4),
            confidence=round(coverage_factor * length_factor, 4),
            detail={
                "raw_site_count": report.raw_site_count,
                "sites_by_protease": report.sites_by_protease,
                "dominant_protease": report.dominant_protease,
                "modifications": list(self.modifications),
            },
            predictor=self.name,
        )


class ProteaseHalfLifePredictor(BasePredictor):
    """血清半衰期估算（小时）。

    经验关系 `t½ ≈ t_base · exp(-k · weighted_sites)`，多个独立切割位点
    相当于并行的降解通道。参数为数量级估计，未标定、置信度低，
    仅用于提供量级参考。
    """

    name = "serum_half_life_estimate"
    is_calibrated = False

    #: 无酶切位点时的参考半衰期（小时）
    BASE_HALF_LIFE_H: float = 24.0
    #: 每个加权位点带来的衰减速率
    DECAY_K: float = 0.55

    def __init__(self, *, modifications: tuple[str, ...] = (), **kwargs) -> None:
        super().__init__(**kwargs)
        self.modifications = tuple(modifications)

    def _cache_key(self, sequence: str) -> str:
        return f"{sequence}|{','.join(self.modifications)}"

    def _predict_one(self, sequence: str) -> PredictionResult:
        import math

        report = analyze(sequence, modifications=self.modifications)
        half_life = self.BASE_HALF_LIFE_H * math.exp(
            -self.DECAY_K * report.weighted_site_count
        )
        return PredictionResult(
            value=round(half_life, 4),
            # 数量级估算，置信度刻意给低
            confidence=0.3,
            detail={"weighted_sites": round(report.weighted_site_count, 3)},
            predictor=self.name,
        )


def protease_weight_table() -> dict[str, float]:
    """暴露权重表，供评估报告解释"为什么某个位点被赋予了更高权重"。"""
    return dict(PROTEASE_WEIGHTS)
