"""奖励层：四维训练奖励 + 一维独立审计。

热稳定性、酶解稳定性、靶点亲和力、结构合理性四维由 `CompositeScorer` 聚合为
单一标量，供 GRPO 计算组内优势。序列天然度（`PlmScorer`，ESM-2 掩码边缘概率）
默认权重为 0，只计算并记录，作为独立于规则引擎的审计信号；显式调高权重后
才成为第五个优化维度。

各维度的原始量纲不同，本层统一其输出量纲；归一化手段按维度各自选择，
不在聚合层叠加第二次映射。本包顶层不 import torch/transformers，
可选后端由 `predictors.factory` 惰性加载。
"""

from pepdesign.rewards.affinity import AffinityScorer
from pepdesign.rewards.base import (
    DimensionScore,
    DimensionScorer,
    PenaltyConfig,
    RewardBreakdown,
    compute_penalties,
    linear_clip,
    ratio_score,
    sequence_invalid_reason,
    sigmoid_normalize,
)
from pepdesign.rewards.composite import (
    UNCALIBRATED_DISCOUNT,
    CompositeScorer,
    dynamic_weights,
    sequence_identity,
    similarity_penalty,
)
from pepdesign.rewards.plm import PlmScorer
from pepdesign.rewards.protease import ProteaseScorer
from pepdesign.rewards.structure import StructureScorer
from pepdesign.rewards.thermo import ThermoScorer

# 列表按语义分层（数据结构 → 各维度 → 聚合），不采用 RUF022 要求的类型分组排序，
# 以免 score 维度被打散到字母序中。
__all__ = [  # noqa: RUF022
    # 数据结构与基础设施
    "DimensionScore",
    "DimensionScorer",
    "PenaltyConfig",
    "RewardBreakdown",
    "compute_penalties",
    "linear_clip",
    "ratio_score",
    "sequence_invalid_reason",
    "sigmoid_normalize",
    # 四个训练维度
    "ThermoScorer",
    "ProteaseScorer",
    "AffinityScorer",
    "StructureScorer",
    # 第五维：独立审计（默认权重 0）
    "PlmScorer",
    # 聚合
    "CompositeScorer",
    "dynamic_weights",
    "sequence_identity",
    "similarity_penalty",
    "UNCALIBRATED_DISCOUNT",
]
