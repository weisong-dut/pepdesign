"""维度四：结构合理性奖励。

`StructureReasonablenessPredictor` 已输出 [0,1] 综合分，内部按固定权重合成
四个子指标，并在连续疏水片段过长、净电荷极端、聚集倾向过高或无标准残基时
把总分上限压到 0.3。再次 sigmoid 会抹掉这一否决语义，因此本模块只做
clip 和 detail 透传。

detail 中的 `vetoed` / `veto_reason` 是报告解释低分的关键信息。
"""

from __future__ import annotations

from pepdesign.rewards.base import DimensionScore, DimensionScorer, linear_clip
from pepdesign.rewards.predictors.base import BasePredictor


class StructureScorer(DimensionScorer):
    """结构合理性维度打分器。"""

    name = "structure"

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

        # 一票否决信息提升到 detail 顶层
        vetoes = detail.get("vetoes") or []
        if vetoes:
            detail["vetoed"] = True
            detail["veto_reason"] = "；".join(str(v) for v in vetoes)
        else:
            detail["vetoed"] = False

        return DimensionScore(
            dimension=self.name,
            value=round(linear_clip(pred.value, 0.0, 1.0), 4),
            raw=round(pred.value, 4),
            confidence=round(max(0.0, min(1.0, pred.confidence)), 4),
            predictor=pred.predictor or self.predictor.name,
            detail=detail,
        )
