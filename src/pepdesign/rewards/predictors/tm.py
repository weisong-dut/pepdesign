"""热稳定性预测（解链温度 Tm）。

不使用端到端模型的原因
----------------------------------
真实的 Tm 预测模型（如 ESM + 回归头、ThermoNet、各种 GNN）确实更准，
但在当前场景下有三个现实约束：

1. 奖励函数要在 CPU 上、每条 rollout 都能算。GRPO 一个 step 要打
   group_size × batch_size 次分（典型 8×16=128 次），如果每次都要一次 GPU 前向，
   采样和打分就串行成了瓶颈，训练吞吐会掉一半以上。
2. 必须可解释。研发人员问"为什么这条序列的稳定性分比那条高"，
   "模型说的"是没法接受的答案。
3. 数据量不够训一个可靠的 Tm 回归模型。真正带实验 Tm 的多肽数据
   在公开库里只有几千条量级，训练深层模型会严重过拟合。

所以本模块实现的是一个透明的线性模型：
    Tm_hat = b0 + Σ w_i · f_i(sequence)
其中特征 f_i 都是有文献依据的、与螺旋肽热稳定性明确相关的物理量。
默认系数是按文献报道的效应方向与量级设定的先验值；
一旦拿到内部实验 Tm 数据，调用 `fit()` 用岭回归重新标定即可，
`is_calibrated` 会自动变成 True，奖励聚合时就会按标定过的权重对待它。

这是"先上可解释的基线，再用实验数据迭代"的标准工程路径。

特征为什么选这几个
------------------
对螺旋型短肽而言，Tm 主要由螺旋含量和螺旋的稳定性决定，具体：
  * helix_propensity —— 螺旋倾向高的残基越多，螺旋越稳定；
  * pro_gly_ratio    —— Pro/Gly 是螺旋破坏者，显著降低 Tm；
  * aliphatic_index  —— 脂肪族侧链堆积，经典的蛋白质热稳定性指标；
  * |net_charge|     —— 净电荷过多会产生螺旋偶极与侧链排斥，降低螺旋稳定性；
  * salt_bridges     —— i/i+4 的 K-E、R-D 盐桥是短肽稳定螺旋最有效的手段；
  * amphipathicity   —— 两亲性螺旋的疏水面埋藏带来额外的稳定化能；
  * length           —— 螺旋越长协同性越强（但收益递减，故用 log）；
  * aggregation      —— 自聚集会虚高表观 Tm（测不准），单独作为警示项。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from pepdesign.bio.sequence import (
    hydrophobic_moment,
    net_charge,
    standard_only,
)
from pepdesign.rewards.predictors.base import BasePredictor, PredictionResult

# ---------------------------------------------------------------------------
# 先验系数（未标定）
# ---------------------------------------------------------------------------
# 这些是先验值，不是拟合出来的，所以 is_calibrated=False。
# 数值设定依据：各特征先归一化到大致 [0,1] 或小量级，
# 系数大小反映"该因素对 Tm 影响的相对重要性"，
# 然后整体缩放到人可读的摄氏度量级。
PRIOR_COEFFICIENTS: dict[str, float] = {
    "intercept": 28.0,          # 基线：一段普通短肽的典型 Tm
    "helix_propensity": 22.0,   # 螺旋倾向从 0.9 到 1.2，约带来 +7℃
    "pro_gly_ratio": -18.0,     # 破坏螺旋，比例每增加 10% 约 -1.8℃
    # AI 的值域是 [0, 390]（见 bio/sequence.py 的量纲说明），所以
    # 0.010 × 100 = 1.0，等价于"AI 每 +100 约 +1℃"这个文献量级。
    # 注意：这里用的是 AI 的原始值，不是百分数也不是归一化值——
    # 如果哪天 AI 的量纲变了，这个系数必须同步改——曾经因为 AI 被多乘了
    # 100，单项贡献直接盖过其余 7 个特征，把线性模型变成了 AI 的换算式。
    # tests/test_bio_sequence.py 里有一条断言把 AI 的取值范围钉死在 [0,390]。
    "aliphatic_index": 0.010,
    "abs_charge": -0.85,        # 净电荷每多 1 个单位约 -0.85℃
    "salt_bridges": 2.2,        # 每个 i/i+4 盐桥约 +2.2℃（文献报道范围 1.5-3）
    "amphipathicity": 12.0,     # 疏水矩每 +0.1 约 +1.2℃
    "log_length": 3.5,          # 协同性项
    "aggregation_penalty": -4.0,  # 聚集风险作为负向修正（表观 Tm 不可信）
}

# 特征名顺序固定，保证 fit 出来的系数向量和预测时的特征向量严格对齐。
# 这个"顺序即契约"的设计如果被破坏（比如有人往中间插了一个特征名），
# 标定过的系数就会全部错位——所以下面 fit() 里会做一次显式校验。
FEATURE_NAMES: tuple[str, ...] = tuple(
    k for k in PRIOR_COEFFICIENTS if k != "intercept"
)


# ---------------------------------------------------------------------------
# 特征工程
# ---------------------------------------------------------------------------
def count_salt_bridges(seq: str, offset: int = 4) -> int:
    """统计 i/i+4 位置的盐桥（K-E、R-D、K-D、R-E）。

    为什么是 i/i+4：α-螺旋每圈 3.6 个残基，所以序列上相隔 4 的残基
    在空间上位于同一侧且相距约一个螺旋周期（5-6 Å），
    正好是侧链可以形成盐桥的距离。这是短肽稳定螺旋最常用的改造策略，
    也正因如此它必须是 Tm 预测的特征之一。
    """
    s = standard_only(seq)
    pairs = {("K", "E"), ("E", "K"), ("R", "D"), ("D", "R"), ("K", "D"), ("D", "K"),
             ("R", "E"), ("E", "R")}
    count = 0
    for i in range(len(s) - offset):
        if (s[i], s[i + offset]) in pairs:
            count += 1
    return count


def extract_features(seq: str) -> dict[str, float]:
    """抽取 Tm 预测用的特征向量。

    与 bio/sequence.py 共用同一套口径——这一点不能妥协：
    如果数据分析和奖励函数各算一套，做特征分析时得到的结论
    就不能用来解释奖励行为，整个调参过程会失去依据。

    量纲约定（每条特征都必须是"幅度可控的小量"，否则线性模型的
    某个系数会独占整个输出）：
        helix_propensity / pro_gly_ratio / aggregation_penalty   ∈ [0, 1]
        amphipathicity（疏水矩）                                  ∈ [0, ~1.5]
        abs_charge（净电荷绝对值）                                ∈ [0, ~15]
        salt_bridges（i/i+4 计数）                                ∈ [0, ~N/4]
        aliphatic_index                                          ∈ [0, 390]
        log_length                                                ∈ [0, ~5.3]
    新增特征前先检查它落在哪个量级——把一个大到几百的量直接塞进来，
    等于给这一维一个隐式的巨大系数。
    """
    from pepdesign.bio.sequence import compute_properties

    s = standard_only(seq)
    if not s:
        return dict.fromkeys(FEATURE_NAMES, 0.0)

    props = compute_properties(seq)
    n = len(s)
    pro_gly = sum(1 for c in s if c in "PG") / n

    return {
        "helix_propensity": props.helix_propensity,
        "pro_gly_ratio": pro_gly,
        "aliphatic_index": props.aliphatic_index,
        "abs_charge": abs(net_charge(s, 7.4)),
        "salt_bridges": float(count_salt_bridges(s)),
        "amphipathicity": hydrophobic_moment(s, angle_deg=100.0),
        "log_length": math.log(max(n, 1)),
        "aggregation_penalty": props.aggregation_propensity,
    }


def features_to_vector(features: dict[str, float]) -> np.ndarray:
    """按 FEATURE_NAMES 的固定顺序把特征字典转成向量。"""
    return np.array([features.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float64)


# ---------------------------------------------------------------------------
# 预测器
# ---------------------------------------------------------------------------
class TmPredictor(BasePredictor):
    """解链温度 Tm 预测器（线性模型 + 可标定）。"""

    name = "tm_linear"

    def __init__(self, coefficients: dict[str, float] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.coefficients = dict(coefficients or PRIOR_COEFFICIENTS)
        # 只有当系数是从实验数据拟合出来时才认为已标定
        self.is_calibrated = bool(coefficients)

    # ------------------------------------------------------------------
    def _predict_one(self, sequence: str) -> PredictionResult:
        features = extract_features(sequence)
        vector = features_to_vector(features)
        weights = np.array(
            [self.coefficients.get(name, 0.0) for name in FEATURE_NAMES],
            dtype=np.float64,
        )
        tm = float(self.coefficients.get("intercept", 0.0) + vector @ weights)

        # 置信度：序列越短、模糊残基越多，预测越不可信
        from pepdesign.bio.sequence import coverage

        cov = coverage(sequence)
        length_factor = min(1.0, len(sequence) / 8.0)   # <8 残基时显著降信
        confidence = round(max(0.1, cov * length_factor), 4)

        return PredictionResult(
            value=round(tm, 3),
            confidence=confidence,
            detail={
                "features": {k: round(v, 4) for k, v in features.items()},
                "contributions": {
                    name: round(
                        float(features.get(name, 0.0) * self.coefficients.get(name, 0.0)), 3
                    )
                    for name in FEATURE_NAMES
                },
            },
            predictor=self.name,
        )

    # ------------------------------------------------------------------
    def fit(
        self,
        sequences: Sequence[str],
        tm_values: Sequence[float],
        *,
        ridge_alpha: float = 1.0,
    ) -> dict[str, float]:
        """用实验 Tm 数据标定系数（岭回归）。

        为什么用岭回归而不是普通最小二乘：
          1. 特征之间有共线性（helix_propensity 和 aliphatic_index 高度相关），
             OLS 在这种情况下的系数会非常不稳定，换一批数据符号都可能翻转；
          2. 实验 Tm 数据量通常只有几百到几千条，而特征有 8 个，
             过拟合风险实实在在，L2 正则是必要的。
        alpha 默认 1.0 是很轻的正则；数据量越少应该调得越大。

        标准化很重要：特征的量纲差别巨大（aliphatic_index 可以到 390，
        amphipathicity 只有 0.5），不标准化的话 L2 惩罚会几乎只作用在
        小量纲的特征上，等价于对不同特征施加了不同的正则强度，
        结果完全不可解释。
        """
        X = np.stack([features_to_vector(extract_features(s)) for s in sequences])
        y = np.asarray(tm_values, dtype=np.float64)

        # 特征标准化
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma < 1e-8] = 1.0     # 常数特征不能除零
        Xs = (X - mu) / sigma

        # 岭回归闭式解: w = (XᵀX + αI)⁻¹ Xᵀy
        n_features = Xs.shape[1]
        A = Xs.T @ Xs + ridge_alpha * np.eye(n_features)
        w_std = np.linalg.solve(A, Xs.T @ y)

        # 把标准化空间里的系数换回原始空间：
        #   y = w_std · (x - mu)/sigma = (w_std/sigma)·x - (w_std/sigma)·mu
        #   所以原始空间权重 = w_std / sigma，截距相应调整
        w_raw = w_std / sigma
        intercept = float(y.mean() - (w_raw * mu).sum())

        self.coefficients = {"intercept": intercept}
        self.coefficients.update(
            {name: float(w) for name, w in zip(FEATURE_NAMES, w_raw, strict=True)}
        )
        self.is_calibrated = True
        self._cache.clear()   # 系数变了，旧缓存全部失效

        # 训练集上的拟合质量，用于判断标定是否可信
        pred = X @ w_raw + intercept
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "n_samples": len(sequences),
            "r2": round(r2, 4),
            "rmse": round(float(np.sqrt(ss_res / max(len(y), 1))), 3),
            "coefficients": {k: round(v, 5) for k, v in self.coefficients.items()},
        }
