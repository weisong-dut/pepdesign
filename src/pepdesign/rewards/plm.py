"""维度五：蛋白质语言模型（ESM-2）的序列天然度奖励。

前四维均由 `bio/` 规则引擎产出，校准集也来自同一引擎，构成闭环；
ESM-2 的分数独立于本项目代码，可作为外部审计信号。

该维默认权重为 0：分数照常计算并记录，但不参与加权求和，避免被优化后
失去裁判意义。显式调高权重后成为第五个优化维度。

预测器已输出 [0,1]，本层只做 clip 并透传原始平均 log 概率，不叠加 sigmoid。
"""

from __future__ import annotations

from pepdesign.rewards.base import DimensionScore, DimensionScorer, linear_clip
from pepdesign.rewards.predictors.base import BasePredictor


class PlmScorer(DimensionScorer):
    """序列天然度维度打分器（ESM-2 掩码边缘概率）。

    构造时不加载权重；预测器首次打分时惰性加载，失败时降级为 confidence=0。
    """

    name = "plm"

    def __init__(
        self,
        predictor: BasePredictor,
        *,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        self.predictor = predictor

    # ------------------------------------------------------------------
    def score_one(self, sequence: str, meta: dict) -> DimensionScore:
        pred = self.predictor.predict_one(sequence)
        detail = dict(pred.detail or {})

        # mean_logprob 是有绝对意义的原始量（nat/残基），提到 detail 顶层供报告引用。
        if "mean_logprob" in detail:
            detail["mean_logprob_nat_per_residue"] = detail.pop("mean_logprob")

        # 预测不可用时显式标记，避免审计报告出现看似正常的 0.5。
        if pred.confidence <= 0.0:
            detail["unavailable"] = True
            detail.setdefault("error", detail.get("note", "PLM 分数不可用"))

        return DimensionScore(
            dimension=self.name,
            value=round(linear_clip(pred.value, 0.0, 1.0), 4),
            raw=round(pred.value, 4),
            confidence=round(max(0.0, min(1.0, pred.confidence)), 4),
            predictor=pred.predictor or self.predictor.name,
            detail=detail,
        )
