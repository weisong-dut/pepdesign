"""奖励预测器集合。

统一接口见 `base.BasePredictor`。实现包括 Tm 线性预测器、酶切位点规则预测器、
血清半衰期估算、基于特征或 ESM 嵌入的亲和力代理、结构合理性综合分、
ESM-2 掩码边缘概率（第五维）以及 plm 关闭时的空预测器。

用 `build_predictor(kind, backend)` 统一构造，或用五个 `build_xxx_predictor`。
`THERMO_BACKENDS` 等常量列出各类后端的合法取值。

本包顶层不 import torch/transformers，可选后端由 `factory._lazy` 在构造时导入。
"""

from pepdesign.rewards.predictors.base import BasePredictor, PredictionResult
from pepdesign.rewards.predictors.factory import (
    AFFINITY_BACKENDS,
    DEFAULT_BACKENDS,
    PLM_BACKENDS,
    PREDICTOR_KINDS,
    PROTEASE_BACKENDS,
    STRUCTURE_BACKENDS,
    THERMO_BACKENDS,
    build_affinity_predictor,
    build_plm_predictor,
    build_predictor,
    build_protease_predictor,
    build_structure_predictor,
    build_thermo_predictor,
)

__all__ = [
    "AFFINITY_BACKENDS",
    "DEFAULT_BACKENDS",
    "PLM_BACKENDS",
    "PREDICTOR_KINDS",
    "PROTEASE_BACKENDS",
    "STRUCTURE_BACKENDS",
    "THERMO_BACKENDS",
    "BasePredictor",
    "PredictionResult",
    "build_affinity_predictor",
    "build_plm_predictor",
    "build_predictor",
    "build_protease_predictor",
    "build_structure_predictor",
    "build_thermo_predictor",
]
