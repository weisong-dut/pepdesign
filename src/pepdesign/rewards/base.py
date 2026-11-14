"""奖励函数的基础设施：归一化、数据结构、通用惩罚项。

四个维度的原始量纲差异很大（Tm 为 ℃、酶切位点为计数、亲和力为倾向分），
不归一化直接加权求和会让权重失去意义。`sigmoid_normalize` 提供处处可导的
(0,1) 映射，temperature 控制曲线陡度：过陡会退化为硬阈值，过平则组内优势趋近 0。
线性截断用于物理范围明确的量。

各维度的输出量纲由本层与聚合层共同保证，不在单个维度上叠加多层归一化。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise


# ---------------------------------------------------------------------------
# 归一化函数
# ---------------------------------------------------------------------------
def sigmoid_normalize(value: float, center: float, temperature: float) -> float:
    """把任意实数映射到 (0, 1)，value=center 时得 0.5：
    1 / (1 + exp(-(value - center) / temperature))。

    指数输入截断到 ±60，避免 exp 溢出成 inf 后传播为 nan。
    temperature 越小，曲线在 center 附近越陡。
    """
    if temperature <= 0:
        raise ValueError(f"temperature 必须为正，收到 {temperature}")
    z = (value - center) / temperature
    z = max(-60.0, min(60.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def linear_clip(value: float, lo: float, hi: float) -> float:
    """线性截断到 [lo, hi]。用于已知物理范围明确的量。"""
    return max(lo, min(hi, value))


def ratio_score(
    value: float,
    baseline: float,
    *,
    higher_is_better: bool = True,
    scale: float | None = None,
) -> float:
    """相对基准的比例得分，映射到 (0,1)：0.5 + 0.5 * tanh(delta / scale)。

    用差值而非比值，因为 Tm 有绝对零点偏移。baseline 决定零点位置，
    scale 决定曲线陡度，两者需独立传入；scale=None 时退回 |baseline| 以兼容旧调用。
    """
    delta = (value - baseline) if higher_is_better else (baseline - value)
    s = abs(baseline) if scale is None else abs(scale)
    return 0.5 + 0.5 * math.tanh(delta / max(s, 1e-6))


# ---------------------------------------------------------------------------
# 结果数据结构
# ---------------------------------------------------------------------------
@dataclass
class DimensionScore:
    """单个奖励维度的打分结果。

    `value` 是归一化后用于加权的分数，`raw` 是物理量（Tm、位点数等），
    两者分开保存，使报告既能给出分数又能给出可解释的原始值。
    """

    dimension: str
    value: float
    raw: float
    confidence: float = 1.0
    weight: float = 0.0
    predictor: str = ""
    detail: dict = field(default_factory=dict)

    @property
    def weighted(self) -> float:
        return self.value * self.weight

    def as_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "value": round(self.value, 4),
            "raw": round(self.raw, 4),
            "confidence": round(self.confidence, 4),
            "weight": round(self.weight, 4),
            "weighted": round(self.weighted, 4),
            "predictor": self.predictor,
            "detail": self.detail,
        }


@dataclass
class RewardBreakdown:
    """一条序列的完整奖励分解，用于报告与日志的可解释性输出。"""

    sequence: str
    total: float
    dimensions: dict[str, DimensionScore] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)
    # 解析失败等硬性问题的标记
    invalid_reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "total": round(self.total, 4),
            "dimensions": {k: v.as_dict() for k, v in self.dimensions.items()},
            "penalties": {k: round(v, 4) for k, v in self.penalties.items()},
            "invalid_reason": self.invalid_reason,
        }

    def explain(self) -> str:
        """生成一行人类可读的解释，用于日志与报告。"""
        if self.invalid_reason:
            return f"[无效] {self.invalid_reason}"
        parts = [
            f"{k}={v.value:.3f}(raw {v.raw:.2f})"
            for k, v in sorted(
                self.dimensions.items(), key=lambda kv: -kv[1].weighted
            )
        ]
        pen = (
            " 惩罚:" + ",".join(f"{k}-{v:.2f}" for k, v in self.penalties.items())
            if self.penalties
            else ""
        )
        return f"总分 {self.total:+.4f} | " + " ".join(parts) + pen


# ---------------------------------------------------------------------------
# 通用惩罚项
# ---------------------------------------------------------------------------
@dataclass
class PenaltyConfig:
    """软约束惩罚的强度配置。"""

    # 聚集倾向：只优化 Tm 会得到一堆不溶的序列
    aggregation: float = 0.30
    # 连续疏水片段：影响血浆蛋白结合与半衰期
    hydrophobic_run: float = 0.20
    # 极端净电荷：破坏静电匹配与溶解性
    charge: float = 0.15
    # 低复杂度：单一残基占比过高（均聚物、poly-Q 等）。
    # 缺少这一项时，T20/N20/Q20/H20 这类序列会拿到高于真实活性肽的奖励。
    low_complexity: float = 0.80
    # 同残基连续片段：任何残基连续超过 5 个
    homopolymer_run: float = 0.40
    # 低 Shannon 熵：针对 QHQH / QTQT 这类两残基交替序列。
    # 阈值按长度分档，只命中不到 1% 的真实肽。
    low_entropy: float = 0.90
    # 非法序列：生成了解析不出或含非法字符的东西
    invalid: float = 0.50


def sequence_invalid_reason(sequence: str) -> str | None:
    """序列不可用时返回原因，可用时返回 None。

    判据复用 `bio/sequence.validate`，并额外拒绝 X/B/Z/U/O。
    全项目只有这一个判定入口，避免惩罚项与 `invalid_reason` 不一致。
    """
    from pepdesign.bio.sequence import validate

    ok, reason = validate(sequence)
    if not ok:
        return reason or "invalid_sequence"

    # 奖励只接受 20 种标准氨基酸。validate 对 X/B/Z/U/O 留了容错，
    # 但生成任务若放行这些字符，会形成可被利用的奖励漏洞。
    standard = frozenset("ACDEFGHIKLMNPQRSTVWY")
    if any(aa not in standard for aa in sequence.upper()):
        return "nonstandard_residue"

    return None


def _longest_homopolymer_run(sequence: str) -> int:
    """最长同残基连续片段。空序列返回 0。"""
    if not sequence:
        return 0
    best = run = 1
    for prev, cur in pairwise(sequence):
        run = run + 1 if cur == prev else 1
        best = max(best, run)
    return best


def compute_penalties(
    sequence: str, cfg: PenaltyConfig, *, props=None, meta: dict | None = None
) -> dict[str, float]:
    """计算所有软约束惩罚（均为正数，在总分中扣除）。

    惩罚强度用连续函数而非布尔开关，使模型在越过阈值前就能收到信号。
    非法序列只返回 `invalid` 一项，由上层将其排除出优势计算。
    """
    from pepdesign.bio.sequence import (
        CHARGE_SOFT_LIMIT,
        compute_properties,
    )

    penalties: dict[str, float] = {}

    # 非法序列优先返回，不再计算其余性质（空串会得到一堆无意义的 0）。
    if sequence_invalid_reason(sequence) is not None:
        penalties["invalid"] = round(cfg.invalid, 4)
        return penalties

    props = props or compute_properties(sequence)

    if props.aggregation_propensity > 0.4:
        excess = min(1.0, (props.aggregation_propensity - 0.4) / 0.4)
        penalties["aggregation"] = round(excess * cfg.aggregation, 4)

    run = props.max_hydrophobic_run
    if run >= 5:
        excess = min(1.0, (run - 4) / 4.0)
        penalties["hydrophobic_run"] = round(excess * cfg.hydrophobic_run, 4)

    # 软惩罚阈值低于 prompt 中的硬要求，为模型留出提前量。
    q = abs(props.net_charge_ph74)
    if q > CHARGE_SOFT_LIMIT:
        excess = min(1.0, (q - CHARGE_SOFT_LIMIT) / 6.0)
        penalties["charge"] = round(excess * cfg.charge, 4)

    # ---- 低复杂度 / 均聚物 ----
    # 阈值 0.45：真实肽最大残基占比的 p99 为 0.42，起罚不会误伤。
    if sequence:
        max_fraction = max(sequence.count(aa) for aa in set(sequence)) / len(sequence)
        if max_fraction > 0.45:
            excess = min(1.0, (max_fraction - 0.45) / 0.55)
            value = round(excess * cfg.low_complexity, 4)
            if value > 0:
                penalties["low_complexity"] = value

        run = _longest_homopolymer_run(sequence)
        if run > 5:
            excess = min(1.0, (run - 5) / 5.0)
            value = round(excess * cfg.homopolymer_run, 4)
            if value > 0:
                penalties["homopolymer_run"] = value

        # 两残基交替序列的最大占比仅 0.5，low_complexity 覆盖不到，用
        # Shannon 熵补充。短肽阈值 1.5 bit，长肽 2.0 bit；真实短肽最低熵约 1.84。
        if len(sequence) >= 5:
            threshold = 1.5 if len(sequence) < 10 else 2.0
            counts = [sequence.count(aa) for aa in set(sequence)]
            entropy = -sum(
                (c / len(sequence)) * math.log2(c / len(sequence)) for c in counts
            )
            if entropy < threshold:
                excess = min(1.0, (threshold - entropy) / threshold)
                value = round(excess * cfg.low_entropy, 4)
                if value > 0:
                    penalties["low_entropy"] = value

    # ---- 任务约束 ----
    # 长度/电荷范围等约束来自 meta，惩罚随违反幅度增长。
    meta = meta or {}
    length_range = meta.get("length_range")
    if length_range and len(length_range) >= 2:
        lo, hi = int(length_range[0]), int(length_range[1])
        if len(sequence) < lo or len(sequence) > hi:
            distance = (lo - len(sequence)) if len(sequence) < lo else (len(sequence) - hi)
            penalties["length"] = round(min(0.60, 0.20 * distance), 4)

    if meta.get("allow_length_change") is False:
        base = str(meta.get("base_sequence") or "")
        if base and len(sequence) != len(base):
            penalties["length"] = max(penalties.get("length", 0.0), 0.30)

    charge_range = meta.get("charge_range")
    if charge_range and len(charge_range) >= 2:
        q_lo, q_hi = float(charge_range[0]), float(charge_range[1])
        charge_now = props.net_charge_ph74
        if charge_now < q_lo or charge_now > q_hi:
            distance = (q_lo - charge_now) if charge_now < q_lo else (charge_now - q_hi)
            penalties["charge_target"] = round(min(0.40, 0.10 * distance), 4)

    return penalties


# ---------------------------------------------------------------------------
# 维度打分的公共骨架
# ---------------------------------------------------------------------------
class DimensionScorer:
    """一个奖励维度的打分器。

    子类实现 `score_one(sequence, meta)` 返回 DimensionScore，
    基类提供统一的单条与批量接口。
    """

    #: 维度名（thermo / protease / affinity / structure）
    name: str = "dimension"

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._call_count = 0

    def score(self, sequence: str, meta: dict | None = None) -> DimensionScore:
        self._call_count += 1
        return self.score_one(sequence, meta or {})

    def score_one(self, sequence: str, meta: dict) -> DimensionScore:  # pragma: no cover
        raise NotImplementedError

    def score_batch(
        self, sequences: Sequence[str], metas: Sequence[dict] | None = None
    ) -> list[DimensionScore]:
        """批量打分。默认朴素循环；有批量能力的子类应覆写。"""
        if metas is None:
            metas = [{} for _ in sequences]
        # strict=True：数量不符时立即报错，避免 zip 静默截断导致奖励错位。
        return [
            self.score_one(seq, meta) for seq, meta in zip(sequences, metas, strict=True)
        ]

    def stats(self) -> dict:
        return {"dimension": self.name, "calls": self._call_count}
