"""四维奖励的聚合层：把各维度分数合成 GRPO 使用的单一标量。

    w      = dynamic_weights(meta["task"], meta["objectives"], cfg=reward_cfg)
    w_eff  = w.normalized()[i] × confidence_i × calibration_discount_i
    total  = Σ (w_eff / Σw_eff) × value_i − Σ penalties
    total  = clip(total, clip_min, clip_max)

有效权重会重新归一化，避免低置信度维度把总分整体压低；置信度与标定状态
分别打折，前者逐条变化，后者对维度恒定。权重来源优先级为：
样本 objectives > 配置中显式的 preset/weights > TASK_PRESETS[task] > balanced。

组内奖励相同（std≈0）时 `group_advantages` 返回全零而非 nan。
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

from pepdesign.rewards.affinity import AffinityScorer
from pepdesign.rewards.base import (
    DimensionScore,
    DimensionScorer,
    PenaltyConfig,
    RewardBreakdown,
    compute_penalties,
    linear_clip,
    sequence_invalid_reason,
)
from pepdesign.rewards.predictors import (
    build_affinity_predictor,
    build_plm_predictor,
    build_protease_predictor,
    build_structure_predictor,
    build_thermo_predictor,
)
from pepdesign.rewards.protease import ProteaseScorer
from pepdesign.rewards.structure import StructureScorer
from pepdesign.rewards.thermo import ThermoScorer
from pepdesign.utils.config import REWARD_PRESETS, RewardConfig, RewardWeights

# ---------------------------------------------------------------------------
# 可调常量
# ---------------------------------------------------------------------------
# 未标定预测器的权重折扣。过轻会让标定失去意义，过重会让未标定维度接近失效。
UNCALIBRATED_DISCOUNT: float = 0.70

# 相似度约束：同一性低于 0.7 时连续扣分。stabilize 任务通常只改 1-3 个残基，
# 0.7 卡在"正常改造"与"改得面目全非"之间。
SIMILARITY_FLOOR: float = 0.70
SIMILARITY_PENALTY_STRENGTH: float = 0.30

# 环化约束惩罚的强度（constraint_penalty 本身输出 [0,1]）
CYCLIC_PENALTY_STRENGTH: float = 0.30

# 优势计算里判定"组内奖励是否退化"的阈值
ADVANTAGE_STD_EPS: float = 1e-6

# 任务名 -> 权重预设。task 取值来自 data/build_grpo.py 的 meta。
TASK_PRESETS: dict[str, str] = {
    # 稳定性优化：主攻稳定性，亲和力由相似度约束守护而非权重。
    "stabilize": "stability_first",
    # 从头设计：没有基准序列，四个维度都要照顾
    "de_novo": "balanced",
    # 多目标协同：权重从 meta["objectives"] 现算，见 _weights_from_objectives
    "multi_objective": "balanced",
}

# 中文目标名 -> 英文维度名。prompt 与数据侧使用中文，RewardWeights 字段是英文；
# 映射缺失会静默退回默认权重，因此名称改动必须同步此表。
OBJECTIVE_ALIASES: dict[str, str] = {
    "热稳定性": "thermo",
    "热稳定": "thermo",
    "thermo": "thermo",
    "酶解稳定性": "protease",
    "酶解稳定": "protease",
    "protease": "protease",
    "靶点亲和力": "affinity",
    "亲和力": "affinity",
    "affinity": "affinity",
    # "成药性"指结构合理性（溶解度、聚集、电荷分布），不是 plm 维度。
    "成药性": "structure",
    "结构合理性": "structure",
    "structure": "structure",
    # 第五维别名。该维默认权重为 0，仅在配置中调高 plm 权重并启用 esm 后端时生效。
    "天然度": "plm",
    "序列天然度": "plm",
    "plm": "plm",
}


# ---------------------------------------------------------------------------
# 动态权重
# ---------------------------------------------------------------------------
def _weights_from_objectives(objectives: dict[str, float]) -> RewardWeights | None:
    """把 prompt 里的中文目标权重翻译成 RewardWeights。无法识别时返回 None。"""
    mapped: dict[str, float] = {}
    for name, weight in objectives.items():
        key = OBJECTIVE_ALIASES.get(str(name).strip())
        if key is None:
            continue
        # 同一维度多个别名时取较大值，避免脏数据把权重变小。
        mapped[key] = max(mapped.get(key, 0.0), float(weight))
    if not mapped:
        return None
    # 全零视为未提供：normalized() 会对权重和 <= 0 抛错，而这里处于训练循环内。
    if sum(mapped.values()) <= 0:
        return None
    # 未出现的维度补 0 而非默认值。新增维度字段时必须同步这里，
    # 否则会拿到类默认值；`test_objective_mapping_covers_every_dimension` 覆盖此约束。
    return RewardWeights(
        thermo=mapped.get("thermo", 0.0),
        protease=mapped.get("protease", 0.0),
        affinity=mapped.get("affinity", 0.0),
        structure=mapped.get("structure", 0.0),
        plm=mapped.get("plm", 0.0),
    )


def dynamic_weights(
    task: str, objectives: dict[str, float] | None = None, *, cfg: RewardConfig | None = None
) -> RewardWeights:
    """按任务类型（和可选的目标权重）给出四维权重。

    优先级从高到低：multi_objective 的 objectives > 配置中显式的 preset/weights
    > `TASK_PRESETS[task]` > balanced。objectives 会原样写入 prompt，配置不能覆盖。
    未知 task 退回 balanced 而不报错。
    """
    if task == "multi_objective" and objectives:
        weights = _weights_from_objectives(objectives)
        if weights is not None:
            return weights
        # objectives 不可用属于配置错误，必须告警；静默使用 balanced 会让错配不可见。
        # 需区分"名称无法识别"与"权重之和为 0"，两者排查方向不同。
        if not any(OBJECTIVE_ALIASES.get(str(k).strip()) for k in objectives):
            detail = "没有任何名称能映射到已知维度"
        else:
            detail = "权重之和为 0"
        warnings.warn(
            f"multi_objective 任务的 objectives 不可用（{detail}）: "
            f"{list(objectives)}；已退回 balanced 权重。"
            f"可用名称见 composite.OBJECTIVE_ALIASES。",
            RuntimeWarning,
            stacklevel=2,
        )

    # 第 2 级：配置中显式写出的 preset/weights 优先于任务预设。
    if cfg is not None:
        configured = cfg.weights_from_config()
        if configured is not None:
            return configured

    preset = TASK_PRESETS.get(task, "balanced")
    return RewardWeights(**REWARD_PRESETS[preset])


# ---------------------------------------------------------------------------
# 相似度约束
# ---------------------------------------------------------------------------
def sequence_identity(a: str, b: str) -> float:
    """两条序列按位置比较的同一性 ∈ [0, 1]。

    用位置对应而非集合交集，避免组成相同但顺序不同的序列被判为高同一性。
    """
    from pepdesign.bio.sequence import canonicalize

    sa, sb = canonicalize(a), canonicalize(b)
    if not sa or not sb:
        return 0.0
    n = min(len(sa), len(sb))
    matched = sum(1 for i in range(n) if sa[i] == sb[i])
    # 分母取较长的那条：长度变化本身也是一种"改远了"，应该被算进代价
    return matched / max(len(sa), len(sb))


def similarity_penalty(base: str, candidate: str) -> float:
    """改得离基准序列太远时的连续扣分。

    同一性从 0.7 降到 0 的过程中，扣分从 0 平滑升至 0.30。
    """
    if not base:
        return 0.0
    identity = sequence_identity(base, candidate)
    if identity >= SIMILARITY_FLOOR:
        return 0.0
    gap = (SIMILARITY_FLOOR - identity) / SIMILARITY_FLOOR
    return round(min(1.0, gap) * SIMILARITY_PENALTY_STRENGTH, 4)


# ---------------------------------------------------------------------------
# 聚合器
# ---------------------------------------------------------------------------
class CompositeScorer:
    """四维奖励聚合器。GRPO 训练中每条 rollout 调用 `score()`。

    默认按配置 backend 构造打分器，也允许注入自定义 scorer 以便测试。
    """

    def __init__(
        self,
        cfg: RewardConfig | None = None,
        *,
        scorers: Sequence[DimensionScorer] | None = None,
        penalty_cfg: PenaltyConfig | None = None,
    ) -> None:
        self.cfg = cfg or RewardConfig()
        self.penalty_cfg = penalty_cfg or PenaltyConfig(
            aggregation=self.cfg.aggregation_penalty,
            hydrophobic_run=self.cfg.hydrophobic_run_penalty,
            charge=self.cfg.charge_penalty,
            invalid=self.cfg.invalid_sequence_penalty,
        )
        self.scorers: dict[str, DimensionScorer] = (
            {s.name: s for s in scorers}
            if scorers is not None
            else self._build_default_scorers()
        )

    # ------------------------------------------------------------------
    def _build_default_scorers(self) -> dict[str, DimensionScorer]:
        """按配置的 backend 构造各维度打分器。

        前四维始终存在；第五维仅在 `plm_backend != "off"` 时加入，
        保证默认配置下奖励与未引入该维时完全一致。plm 权重为 0 时只记录不参与求和。
        """
        cfg = self.cfg
        scorers: dict[str, DimensionScorer] = {
            "thermo": ThermoScorer(
                build_thermo_predictor(cfg.thermo_backend),
                baseline_tm=cfg.baseline_tm,
                temperature=cfg.thermo_temperature,
                score_scale=cfg.thermo_score_scale,
            ),
            "protease": ProteaseScorer(build_protease_predictor(cfg.protease_backend)),
            "affinity": AffinityScorer(
                build_affinity_predictor(cfg.affinity_backend),
                baseline_affinity=cfg.baseline_affinity,
                temperature=cfg.affinity_temperature,
            ),
            "structure": StructureScorer(
                build_structure_predictor(cfg.structure_backend)
            ),
        }
        if cfg.plm_backend != "off":
            from pepdesign.rewards.plm import PlmScorer

            scorers["plm"] = PlmScorer(build_plm_predictor(cfg.plm_backend))
        return scorers

    # ------------------------------------------------------------------
    def score(self, sequence: str, meta: dict | None = None) -> RewardBreakdown:
        """给一条序列打完整的分，返回可解释的四维分解。"""
        meta = dict(meta or {})

        # 无效输入没有分数，返回带 invalid_reason 的 breakdown 供上层排除出优势计算。
        # 无效来源有两个：调用方写入 meta 的，以及序列内容本身不合法。
        invalid_reason = meta.get("invalid_reason")
        if not sequence:
            invalid_reason = invalid_reason or "empty_sequence"
        elif not invalid_reason:
            # 混合大小写的 D-型残基归一化后不可分辨；输出只允许 20 种标准氨基酸。
            if "noncanonical_residues" in (meta.get("modifications") or ()):
                invalid_reason = "noncanonical_residues"
            else:
                invalid_reason = sequence_invalid_reason(sequence)

        if invalid_reason:
            return RewardBreakdown(
                sequence=sequence or "",
                total=self.cfg.clip_min,
                dimensions={},
                penalties={"invalid": round(self.penalty_cfg.invalid, 4)},
                invalid_reason=invalid_reason,
            )

        dim_scores = self._score_dimensions(sequence, meta)

        # ---- 有效权重：置信度 + 标定状态 ----
        # cfg=self.cfg 是配置权重生效的入口。
        w = dynamic_weights(
            meta.get("task", ""), meta.get("objectives"), cfg=self.cfg
        ).normalized()
        weights = w.model_dump()

        eff: dict[str, float] = {}
        for name, score in dim_scores.items():
            base_w = weights.get(name, 0.0)
            discount = 1.0
            if not self._is_calibrated(name):
                discount *= UNCALIBRATED_DISCOUNT
            eff[name] = base_w * max(0.0, min(1.0, score.confidence)) * discount

        total_eff = sum(eff.values())
        # 全部有效权重为 0 时退化为等权平均，避免恒零奖励导致组内 std 归零。
        if total_eff <= 0:
            n = max(len(dim_scores), 1)
            eff = dict.fromkeys(dim_scores, 1.0 / n)
            total_eff = 1.0

        for name, score in dim_scores.items():
            score.weight = round(eff[name] / total_eff, 4)

        # ---- 加权求和 ----
        weighted_sum = sum(
            (eff[name] / total_eff) * score.value for name, score in dim_scores.items()
        )

        # ---- 软约束惩罚 ----
        # 非法序列已提前返回，此处只计算软约束惩罚。
        penalties = compute_penalties(sequence, self.penalty_cfg, meta=meta)

        sim_pen = similarity_penalty(meta.get("base_sequence", ""), sequence)
        if sim_pen > 0:
            penalties["similarity"] = sim_pen

        cyclic_type = meta.get("cyclic_type")
        if cyclic_type and cyclic_type != "none":
            from pepdesign.bio.cyclization import constraint_penalty

            try:
                cp = constraint_penalty(sequence, cyclic_type)
            except KeyError:
                # cyclic_type 来自数据而非配置，拼错的值会走到这里。单条脏数据
                # 不应让整个 step 崩溃，因此按满额惩罚处理并告警。
                cp = 1.0
                warnings.warn(
                    "遇到无法识别的 cyclic_type，已按满额环化惩罚处理；"
                    "合法取值为 head_to_tail/disulfide/staple/lactam/none。"
                    "请检查数据集的 cyclic_type 字段。",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if cp > 0:
                penalties["cyclic"] = round(cp * CYCLIC_PENALTY_STRENGTH, 4)

        total = weighted_sum - sum(penalties.values())
        total = linear_clip(total, self.cfg.clip_min, self.cfg.clip_max)

        return RewardBreakdown(
            sequence=sequence,
            total=round(total, 4),
            dimensions=dim_scores,
            penalties={k: round(v, 4) for k, v in penalties.items()},
            invalid_reason=None,
        )

    # ------------------------------------------------------------------
    def _score_dimensions(
        self, sequence: str, meta: dict
    ) -> dict[str, DimensionScore]:
        """依次运行各维度打分。

        单维度异常被捕获并记为中性分 0.5，避免一条样本让整个训练 step 失败。
        """
        out: dict[str, DimensionScore] = {}
        for name, scorer in self.scorers.items():
            if not scorer.enabled:
                continue
            try:
                out[name] = scorer.score(sequence, meta)
            except Exception as exc:  # noqa: BLE001 - 见上面 docstring
                out[name] = DimensionScore(
                    dimension=name,
                    value=0.5,
                    raw=0.0,
                    confidence=0.0,
                    predictor="<failed>",
                    detail={"error": f"{type(exc).__name__}: {exc}"},
                )
        return out

    def _is_calibrated(self, dimension: str) -> bool:
        scorer = self.scorers.get(dimension)
        predictor = getattr(scorer, "predictor", None)
        if predictor is None:
            # 纯规则维度没有预测器，视为已就绪。
            return True
        return bool(getattr(predictor, "is_calibrated", False))

    # ------------------------------------------------------------------
    def score_group(
        self,
        sequences: Sequence[str],
        metas: Sequence[dict] | None = None,
    ) -> list[RewardBreakdown]:
        """给一个 GRPO 组打分。

        组内序列共享同一份 meta，因此允许只传一条并广播到全组；
        条数不一致必须报错，否则 zip 静默截断会让整组优势归零。
        """
        n = len(sequences)
        meta_list = [{} for _ in range(n)] if metas is None else list(metas)

        if len(meta_list) == 1 and n != 1:
            meta_list = meta_list * n
        elif len(meta_list) != n:
            raise ValueError(
                f"metas 与 sequences 数量不一致: {len(meta_list)} vs {n}。"
                f"给单条 meta 会自动广播到全组，其余情况必须逐条对应。"
            )

        # 长度已校验，strict=True 作为校验逻辑被破坏时的兜底。
        return [self.score(s, m) for s, m in zip(sequences, meta_list, strict=True)]

    # ------------------------------------------------------------------
    def group_advantages(
        self, scores: Sequence[RewardBreakdown], mode: str = "std"
    ) -> list[float]:
        """GRPO 的组内优势：(r_i - mean) / std。

        `mode` 支持 `"std"`、`"mean"`（除以 |mean|）和 `"none"`（只去均值）。
        无效序列不参与 mean/std 计算，但在返回列表中占位 0.0；
        组内奖励退化（std≈0 或 mean≈0）时返回全零而非 nan。
        """
        if mode not in ("std", "mean", "none"):
            raise ValueError(
                f"未知的优势口径: {mode!r}，可选 std / mean / none"
            )

        n = len(scores)
        if n == 0:
            return []

        valid_idx = [i for i, s in enumerate(scores) if s.invalid_reason is None]
        if not valid_idx:
            return [0.0] * n

        rewards = [scores[i].total for i in valid_idx]
        mean = sum(rewards) / len(rewards)
        var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        std = math.sqrt(var)

        out = [0.0] * n

        # 各口径的退化情形不同，统一返回全零，语义为"这一步没有可学的信息"。
        if mode == "std":
            if std < ADVANTAGE_STD_EPS:
                return out
            denom = std
        elif mode == "mean":
            if abs(mean) < ADVANTAGE_STD_EPS:
                return out
            denom = abs(mean)
        else:
            denom = 1.0

        for i in valid_idx:
            out[i] = (scores[i].total - mean) / denom
        return [round(a, 6) for a in out]

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """诊断快照。

        报告打分会实际使用的权重：`weight_source` 为 `"config"` 表示配置显式指定，
        否则按 `TASK_PRESETS` 取。`task_presets` 同时给出各任务预设供对照。
        """
        configured = self.cfg.weights_from_config()
        return {
            "scorers": {k: v.stats() for k, v in self.scorers.items()},
            "weight_source": "config" if configured is not None else "task_preset",
            "config": {
                "preset": self.cfg.preset,
                # 配置未显式指定时为 None，不伪装成生效值。
                "weights": (
                    configured.normalized().model_dump() if configured is not None else None
                ),
            },
            "task_presets": {
                task: RewardWeights(**REWARD_PRESETS[preset]).normalized().model_dump()
                for task, preset in TASK_PRESETS.items()
            },
            "note": (
                "task=multi_objective 的样本，权重来自该条 prompt 里写死的 "
                "meta['objectives']（两者必须一致，见 templates.multi_objective_prompt），"
                "上面这张表对它不适用。"
            ),
        }
