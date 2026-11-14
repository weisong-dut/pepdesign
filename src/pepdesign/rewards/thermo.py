"""维度一：热稳定性奖励（基于 Tm 预测）。

两段归一化：`ratio_score` 把 Tm 相对基准的差值映射为语义分（0.5 = 与基准持平，
scale 以 ℃ 为单位决定陡度），再用 `sigmoid_normalize` 围绕 0.5 做对比度增强，
把组内密集的分数拉开，保证 GRPO 优势信号强度。

`scale` 与 `temperature` 独立传入，避免更换基准时顺带改变曲线斜率。

长度小于 `TM_EXTRAPOLATION_LENGTH` 的序列不适用 Tm 概念，会混入基于
`aliphatic_index` 的代理分，并在 detail 中标记 `length_extrapolation=True`。
"""

from __future__ import annotations

from pepdesign.bio.sequence import compute_properties
from pepdesign.rewards.base import (
    DimensionScore,
    DimensionScorer,
    ratio_score,
    sigmoid_normalize,
)
from pepdesign.rewards.predictors.base import BasePredictor

# 长度低于此值时，Tm 预测属于外推，需要混入代理分
TM_EXTRAPOLATION_LENGTH: int = 12
# 外推时的混合权重：真实 Tm 预测占 40%，序列层面的稳定性代理占 60%
EXTRAPOLATION_MIX: float = 0.40

# 代理分 sigmoid 的参考点与温度。AI 值域 [0, 390]，多肽实测样本约 [0, 293]，
# 中位量级 100-120；取 center=120、temperature=70 可避免多数序列饱和到 1.0。
PROXY_CENTER: float = 120.0
PROXY_TEMPERATURE: float = 70.0


class ThermoScorer(DimensionScorer):
    """热稳定性维度打分器。"""

    name = "thermo"

    def __init__(
        self,
        predictor: BasePredictor,
        *,
        baseline_tm: float = 45.0,
        temperature: float = 0.12,
        score_scale: float = 45.0,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        self.predictor = predictor
        self.baseline_tm = baseline_tm
        self.temperature = temperature
        # 与 baseline_tm 解耦的独立陡度参数
        self.score_scale = score_scale

    # ------------------------------------------------------------------
    def score_one(self, sequence: str, meta: dict) -> DimensionScore:
        pred = self.predictor.predict_one(sequence)
        tm = pred.value

        # 第 1 段：物理量 → 语义分，score_scale 与 baseline_tm 解耦。
        base_score = ratio_score(
            tm,
            self.baseline_tm,
            higher_is_better=True,
            scale=self.score_scale,
        )

        length = len(sequence)
        extrapolating = length < TM_EXTRAPOLATION_LENGTH
        if extrapolating:
            # 短肽：Tm 预测不可靠，混入序列层面的稳定性代理
            props = compute_properties(sequence)
            proxy = sigmoid_normalize(
                props.aliphatic_index,
                center=PROXY_CENTER,
                temperature=PROXY_TEMPERATURE,
            )
            value = EXTRAPOLATION_MIX * base_score + (1 - EXTRAPOLATION_MIX) * proxy
            # 外推场景下整体置信度打折
            confidence = pred.confidence * 0.6
        else:
            value = base_score
            confidence = pred.confidence

        # 第 2 段：围绕 0.5 做对比度增强，GRPO 只依赖组内相对大小。
        # sigmoid 以 0.5 对称，与基准持平的序列增强前后都等于 0.5。
        final = sigmoid_normalize(value, center=0.5, temperature=self.temperature)

        return DimensionScore(
            dimension=self.name,
            value=round(final, 4),
            raw=round(tm, 3),
            confidence=round(confidence, 4),
            predictor=pred.predictor,
            detail={
                **pred.detail,
                "baseline_tm": self.baseline_tm,
                "score_scale": self.score_scale,
                "delta_tm": round(tm - self.baseline_tm, 3),
                "length_extrapolation": extrapolating,
                # 保留增强前的分数，报告里可以还原出"到底比基准强多少"
                "score_before_enhance": round(value, 4),
            },
        )
