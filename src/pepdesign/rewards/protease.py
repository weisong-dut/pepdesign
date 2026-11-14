"""维度二：酶解稳定性奖励。

`bio/cleavage.stability_score()` 已返回 [0,1] 的稳定性分，且内部对加权位点
密度做了饱和截断，因此本模块直接透传，不再叠加 sigmoid。统一发生在输出
量纲上，而不是每个维度都套一层归一化。

打分基于位点密度（加权位点数 / 序列长度），回答"相对于自身长度有多容易
被切"；detail 同时给出 `n_sites` 与 `weighted_site_count`，便于报告区分
位点数量与位点权重。
"""

from __future__ import annotations

from pepdesign.bio.cleavage import analyze, stability_score
from pepdesign.rewards.base import DimensionScore, DimensionScorer
from pepdesign.rewards.predictors.base import BasePredictor

# half_life 可选注入，仅用于报告展示，不参与打分：半衰期由位点数指数映射而来，
# 噪声大且信息与位点数高度重合。
_HALF_LIFE_NOTE = (
    "半衰期由位点数经指数关系推算（t½ ≈ t_base·exp(-k·N)），"
    "数量级参考，不参与打分。"
)


class ProteaseScorer(DimensionScorer):
    """酶解稳定性维度打分器。"""

    name = "protease"

    def __init__(
        self,
        predictor: BasePredictor | None = None,
        *,
        half_life_predictor: BasePredictor | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        # predictor 仅用于报告溯源；不传也能打分（stability_score 是纯函数）
        self.predictor = predictor
        self.half_life_predictor = half_life_predictor

    # ------------------------------------------------------------------
    def score_one(self, sequence: str, meta: dict) -> DimensionScore:
        # 末端修饰改变酶切行为（乙酰化/酰胺化封端），必须随预测传入。
        mods = tuple(meta.get("modifications") or ())

        report = analyze(sequence, modifications=mods)
        value = stability_score(sequence, modifications=mods)

        detail: dict = {
            "n_sites": report.raw_site_count,
            "weighted_site_count": round(report.weighted_site_count, 3),
            "sites_by_protease": dict(report.sites_by_protease),
            "dominant_protease": report.dominant_protease,
            "modifications_applied": list(mods),
            # 透传打分函数，报告里可以直接回答"这个分是谁算的"
            "score_source": "bio.cleavage.stability_score",
        }

        # 半衰期只做展示，失败不影响打分
        if self.half_life_predictor is not None:
            try:
                hl = self.half_life_predictor.predict_one(sequence)
                detail["half_life_h"] = round(hl.value, 3)
                detail["half_life_confidence"] = hl.confidence
                detail["half_life_note"] = _HALF_LIFE_NOTE
            except Exception as exc:  # noqa: BLE001 - 展示项，不应影响奖励
                detail["half_life_error"] = str(exc)

        return DimensionScore(
            dimension=self.name,
            value=round(value, 4),
            raw=float(report.raw_site_count),
            # 置信度取标准残基占比，模糊残基使位点判定不可信。
            confidence=round(_sequence_coverage(sequence), 4),
            predictor=self.predictor.name if self.predictor else "cleavage_rules",
            detail=detail,
        )


def _sequence_coverage(sequence: str) -> float:
    """标准残基占比，作为酶切预测的置信度代理。"""
    from pepdesign.bio.sequence import coverage

    return coverage(sequence)
