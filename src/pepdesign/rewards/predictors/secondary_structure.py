"""结构合理性预测：二级结构置信度、疏水平衡、电荷分布与组成自然度。

该维度与另外三个维度形成制衡：重复疏水残基、极端电荷、异常组成等
会在这里得到低分，避免其他维度的奖励漏洞被利用。

四个子指标按固定权重合成；连续疏水片段过长、净电荷极端、聚集倾向过高
或无标准残基时触发否决，总分上限压到 0.3。
"""

from __future__ import annotations

import math
from typing import ClassVar

from pepdesign.bio.aa import STANDARD_AA
from pepdesign.bio.sequence import (
    CHARGE_PROMPT_LIMIT,
    CHARGE_VETO_LIMIT,
    compute_properties,
    standard_only,
)
from pepdesign.rewards.predictors.base import BasePredictor, PredictionResult

# ---------------------------------------------------------------------------
# 天然多肽的氨基酸组成本底
# ---------------------------------------------------------------------------
# 蛋白质组经验组成（约取自 UniProt 统计），作为组成自然度的参照分布。
# 顺序与 STANDARD_AA 一致，便于按位比较。
NATURAL_COMPOSITION: dict[str, float] = {
    "A": 0.0825, "C": 0.0137, "D": 0.0545, "E": 0.0675, "F": 0.0386,
    "G": 0.0707, "H": 0.0227, "I": 0.0591, "K": 0.0580, "L": 0.0965,
    "M": 0.0242, "N": 0.0406, "P": 0.0470, "Q": 0.0393, "R": 0.0553,
    "S": 0.0667, "T": 0.0535, "V": 0.0687, "W": 0.0110, "Y": 0.0292,
}

# 疏水平衡的理想区间：过低太亲水，过高太疏水（溶解度差、易聚集）。
GRAVY_IDEAL_RANGE: tuple[float, float] = (-1.2, 0.2)
# 净电荷合理范围，取自 bio/sequence.py 的统一阈值。
CHARGE_LIMIT: float = CHARGE_PROMPT_LIMIT


def ss_confidence(seq: str, props=None) -> float:
    """二级结构置信度 ∈ [0, 1]。

    取三个倾向值中最高值与次高值之差并归一化：差值越大，结构偏好越明确。
    只看最强的一对竞争者，能识别"两种构象倾向接近相等"的失败设计。
    """
    props = props if props is not None else compute_properties(seq)
    values = sorted(
        [props.helix_propensity, props.sheet_propensity, props.coil_propensity]
    )
    top, second = values[-1], values[-2]
    gap = top - second
    # 倾向值约在 0.5-1.7；0.35 为经验标尺。
    return round(max(0.0, min(1.0, gap / 0.35)), 4)


def hydrophobic_balance(seq: str, props=None) -> float:
    """疏水平衡得分 ∈ [0, 1]，1 = GRAVY 落在理想区间中心。

    理想区间内给满分，偏离越远扣分越多，超出 2 个单位得 0。
    """
    props = props if props is not None else compute_properties(seq)
    g = props.gravy
    lo, hi = GRAVY_IDEAL_RANGE
    if lo <= g <= hi:
        return 1.0
    distance = (lo - g) if g < lo else (g - hi)
    return round(max(0.0, 1.0 - distance / 2.0), 4)


def charge_distribution(seq: str, props=None) -> float:
    """电荷分布合理性 ∈ [0, 1]。

    净电荷超过 `CHARGE_LIMIT` 后线性扣分；电荷过度聚集
    （`charge_asymmetry` > 0.6）再追加惩罚，同时避免误伤抗菌肽常见的 N 端正电头。
    """
    props = props if props is not None else compute_properties(seq)
    q = abs(props.net_charge_ph74)

    charge_score = (
        1.0 if q <= CHARGE_LIMIT else max(0.0, 1.0 - (q - CHARGE_LIMIT) / 4.0)
    )

    asym = props.charge_asymmetry
    cluster_penalty = max(0.0, (asym - 0.6) / 0.4) * 0.3

    return round(max(0.0, charge_score - cluster_penalty), 4)


def composition_naturalness(seq: str) -> float:
    """组成自然度 ∈ [0, 1]，用与天然组成本底的余弦相似度度量。

    余弦只关心组成向量的方向，对长度不敏感，可跨长度比较。
    """
    s = standard_only(seq)
    if not s:
        return 0.0
    n = len(s)
    counts = {aa: s.count(aa) / n for aa in STANDARD_AA}
    natural = NATURAL_COMPOSITION

    dot = sum(counts.get(aa, 0.0) * natural[aa] for aa in STANDARD_AA)
    norm_seq = math.sqrt(sum(v * v for v in counts.values()))
    norm_nat = math.sqrt(sum(v * v for v in natural.values()))
    if norm_seq < 1e-9 or norm_nat < 1e-9:
        return 0.0
    cosine = dot / (norm_seq * norm_nat)

    # 余弦在实际序列上的取值区间大约 [0.6, 1.0]，线性映射到 [0,1]
    return round(max(0.0, min(1.0, (cosine - 0.6) / 0.4)), 4)


class StructureReasonablenessPredictor(BasePredictor):
    """结构合理性预测器：输出 ∈ [0, 1] 的综合分。"""

    name = "structure_propensity"

    # 四个子指标的权重：结构置信度最高，组成自然度次之，疏水平衡与电荷分布用于细化。
    # ClassVar 使该表被所有实例共享。
    SUB_WEIGHTS: ClassVar[dict[str, float]] = {
        "ss_confidence": 0.35,
        "composition_natural": 0.25,
        "hydrophobic_balance": 0.25,
        "charge_distribution": 0.15,
    }

    def _predict_one(self, sequence: str) -> PredictionResult:
        # props 只算一次并传给各子指标，避免重复计算。
        props = compute_properties(sequence)

        subs = {
            "ss_confidence": ss_confidence(sequence, props),
            "hydrophobic_balance": hydrophobic_balance(sequence, props),
            "charge_distribution": charge_distribution(sequence, props),
            "composition_natural": composition_naturalness(sequence),
        }
        total = sum(subs[k] * self.SUB_WEIGHTS[k] for k in subs)

        # 触发任一否决项时总分上限压到 0.3，与正常序列拉开明显差距。
        vetoes: list[str] = []
        if props.max_hydrophobic_run >= 6:
            vetoes.append(f"连续疏水片段过长({props.max_hydrophobic_run})")
        if abs(props.net_charge_ph74) >= CHARGE_VETO_LIMIT:
            vetoes.append(f"净电荷极端({props.net_charge_ph74:.1f})")
        if props.aggregation_propensity >= 0.8:
            vetoes.append(f"聚集倾向过高({props.aggregation_propensity:.2f})")
        if not standard_only(sequence):
            vetoes.append("无任何标准氨基酸残基")

        if vetoes:
            total = min(total, 0.3)

        return PredictionResult(
            value=round(total, 4),
            confidence=round(0.5 * props.coverage, 4),
            detail={
                "sub_scores": subs,
                "vetoes": vetoes,
                "propensities": {
                    "helix": props.helix_propensity,
                    "sheet": props.sheet_propensity,
                    "coil": props.coil_propensity,
                },
            },
            predictor=self.name,
        )
