"""维度三：靶点亲和力奖励。

优先使用相对口径：`predict_with_reference` 返回候选相对基准序列的保持度
∈ [0, 1.5]，数值天然适合 GRPO 的组内比较。基准取自 `meta["base_sequence"]`
（用户输入的序列），不读取 `reference_sequence`——后者在部分数据构造路径中
填入目标序列，会造成奖励泄漏。

`meta` 缺少 `base_sequence` 时退回绝对口径：用 `sigmoid_normalize` 压到 (0,1)，
置信度乘 `ABSOLUTE_CONFIDENCE_DISCOUNT`，并在 detail 中标记 `used_absolute`，
使口径变化可见。
"""

from __future__ import annotations

from pepdesign.rewards.base import (
    DimensionScore,
    DimensionScorer,
    linear_clip,
    sigmoid_normalize,
)
from pepdesign.rewards.predictors.base import BasePredictor

# detail 中标记打分口径的键名，测试与报告依赖它
USED_ABSOLUTE_KEY = "used_absolute"
# 相对口径的上限：ratio 可超过 1（候选可优于参考）
RELATIVE_SCORE_MAX = 1.5
# 退化到绝对口径时的置信度折扣
ABSOLUTE_CONFIDENCE_DISCOUNT = 0.6


class AffinityScorer(DimensionScorer):
    """靶点亲和力维度打分器。"""

    name = "affinity"

    def __init__(
        self,
        predictor: BasePredictor,
        *,
        baseline_affinity: float = 0.5,
        temperature: float = 0.15,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        self.predictor = predictor
        self.baseline_affinity = baseline_affinity
        self.temperature = temperature

    # ------------------------------------------------------------------
    def score_one(self, sequence: str, meta: dict) -> DimensionScore:
        # 基准是用户输入的序列；不读 reference_sequence，避免奖励泄漏。
        reference = meta.get("base_sequence") or ""

        if reference and hasattr(self.predictor, "predict_with_reference"):
            pred = self.predictor.predict_with_reference(sequence, reference)
            # 相对分已以 1.0 表示"与参考持平"，只做 clip，不再叠加 sigmoid。
            value = linear_clip(pred.value, 0.0, RELATIVE_SCORE_MAX)
            used_absolute = False
            confidence = pred.confidence
        else:
            pred = self.predictor.predict_one(sequence)
            # 绝对分是"倾向分"，量纲不适合直接进加权求和，压到 (0,1)
            value = sigmoid_normalize(
                pred.value, center=self.baseline_affinity,
                temperature=self.temperature,
            )
            used_absolute = True
            confidence = pred.confidence * ABSOLUTE_CONFIDENCE_DISCOUNT

        return DimensionScore(
            dimension=self.name,
            value=round(value, 4),
            # raw 保留预测器原始输出，便于报告溯源
            raw=round(pred.value, 4),
            confidence=round(max(0.0, min(1.0, confidence)), 4),
            predictor=pred.predictor or self.predictor.name,
            detail={
                **(pred.detail or {}),
                USED_ABSOLUTE_KEY: used_absolute,
                # 记录分母来源，报告中可看出基准序列
                "baseline_sequence": reference or None,
                # 无基准序列时写明退化原因
                "fallback_reason": (
                    None if reference else "meta 中缺少 base_sequence，退回绝对口径"
                ),
                "scoring_scope": "absolute" if used_absolute else "relative",
                # 相对口径下 raw 与 value 同值，显式说明避免误读
                "raw_is_normalized": used_absolute,
            },
        )
