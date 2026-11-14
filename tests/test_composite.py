"""四维奖励聚合层。

`composite.py` 把四个维度的分数合成为 GRPO 使用的单一标量。聚合错误不会
崩溃，而表现为"训练在跑、loss 在降、指标不动"，因此测试重点在聚合代数
（权重分配、折扣、惩罚叠加、裁剪）。

使用固定值假打分器 `FixedScorer`，不接真预测器；惩罚在纯代数测试中
被显式关掉（`PenaltyConfig` 全零），使 `total` 精确等于加权和。
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import ClassVar

import pytest

from pepdesign.rewards.base import (
    DimensionScore,
    DimensionScorer,
    PenaltyConfig,
    compute_penalties,
)
from pepdesign.rewards.composite import (
    CYCLIC_PENALTY_STRENGTH,
    SIMILARITY_FLOOR,
    SIMILARITY_PENALTY_STRENGTH,
    TASK_PRESETS,
    UNCALIBRATED_DISCOUNT,
    CompositeScorer,
    _weights_from_objectives,
    dynamic_weights,
    sequence_identity,
    similarity_penalty,
)
from pepdesign.utils.config import REWARD_PRESETS, RewardConfig, RewardWeights

# 一条干净的多肽：不触发聚集 / 疏水段 / 极端电荷中的任何一条，
# 于是"惩罚项为空"这件事在测试里是可控的。
CLEAN_SEQ = "KWKLFKKIEK"

NO_PENALTIES = PenaltyConfig(
    aggregation=0.0,
    hydrophobic_run=0.0,
    charge=0.0,
    low_complexity=0.0,
    homopolymer_run=0.0,
    low_entropy=0.0,
    invalid=0.0,
)

# 相似度测试的基准肽与定点突变体：逐位替换使同一性恰好为 (len-k)/len。
BASE10 = "ACDEFGHIKL"


def mutate(base: str, k: int) -> str:
    """把 base 的前 k 位各换成一个不同的残基，同一性恰好是 (len-k)/len。"""
    out = list(base)
    for i in range(k):
        out[i] = "W" if base[i] != "W" else "A"
    return "".join(out)


class FixedScorer(DimensionScorer):
    """恒返回固定值的打分器。

    `predictor.is_calibrated` 是标定折扣的读取来源，缺失会被当成"已标定"。
    """

    def __init__(
        self,
        name: str,
        value: float = 0.5,
        *,
        confidence: float = 1.0,
        raw: float = 0.0,
        is_calibrated: bool = True,
        detail: dict | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        self.name = name
        self.fixed_value = value
        self.fixed_confidence = confidence
        self.fixed_raw = raw
        self.fixed_detail = detail or {}
        self.predictor = SimpleNamespace(name=f"{name}_fake", is_calibrated=is_calibrated)

    def score_one(self, sequence: str, meta: dict) -> DimensionScore:
        return DimensionScore(
            dimension=self.name,
            value=self.fixed_value,
            raw=self.fixed_raw,
            confidence=self.fixed_confidence,
            predictor=self.predictor.name,
            detail=dict(self.fixed_detail),
        )


def build(values: dict[str, float], **kwargs) -> CompositeScorer:
    """用指定的四维取值造一个聚合器，惩罚默认全关。"""
    scorers = [FixedScorer(name, v) for name, v in values.items()]
    return CompositeScorer(
        RewardConfig(),
        scorers=scorers,
        penalty_cfg=kwargs.pop("penalty_cfg", NO_PENALTIES),
        **kwargs,
    )


def build_with(cfg: RewardConfig, values: dict[str, float]) -> CompositeScorer:
    """同 `build`，但奖励配置由调用方提供；`build` 使用默认配置作为对照组。"""
    return CompositeScorer(
        cfg,
        scorers=[FixedScorer(name, v) for name, v in values.items()],
        penalty_cfg=NO_PENALTIES,
    )


# ---------------------------------------------------------------------------
# 动态权重
# ---------------------------------------------------------------------------
class TestDynamicWeights:
    def test_stabilize_uses_stability_first(self):
        """稳定性优化任务的权重向 thermo/protease 倾斜。"""
        w = dynamic_weights("stabilize")
        assert w.thermo == REWARD_PRESETS["stability_first"]["thermo"]
        assert w.protease == REWARD_PRESETS["stability_first"]["protease"]
        assert w.thermo > w.affinity

    def test_de_novo_uses_balanced(self):
        assert dynamic_weights("de_novo").model_dump() == REWARD_PRESETS["balanced"]

    def test_multi_objective_reads_objectives(self):
        w = dynamic_weights("multi_objective", {"热稳定性": 0.6, "靶点亲和力": 0.4})
        assert w.thermo == pytest.approx(0.6)
        assert w.affinity == pytest.approx(0.4)
        # 没列出来的维度补 0，而不是偷偷给默认份额
        assert w.protease == 0.0 and w.structure == 0.0

    def test_unlisted_dimension_gets_zero_not_default(self):
        """未列出的维度权重为 0，不补默认值。"""
        w = dynamic_weights("multi_objective", {"热稳定性": 1.0})
        assert w.model_dump() == {
            "thermo": 1.0, "protease": 0.0, "affinity": 0.0,
            "structure": 0.0, "plm": 0.0,
        }

    def test_developability_maps_to_structure_not_to_plm(self):
        """`成药性` 映射到 structure 维而非 plm 维；两者衡量的是不同性质。"""
        w = dynamic_weights("multi_objective", {"成药性": 0.3, "热稳定性": 0.7})
        assert w.structure == pytest.approx(0.3)
        assert w.thermo == pytest.approx(0.7)
        assert w.plm == 0.0, "成药性不该落到 plm 维上"

    def test_objective_mapping_covers_every_dimension(self):
        """`_weights_from_objectives` 必须覆盖 `RewardWeights` 的每个字段，未列出维度显式补 0。"""
        from pepdesign.utils.config import RewardWeights

        w = _weights_from_objectives({"热稳定性": 1.0})
        assert w is not None
        # 未列出的维度必须全部为 0
        for field in RewardWeights.model_fields:
            if field == "thermo":
                assert getattr(w, field) == pytest.approx(1.0)
            else:
                assert getattr(w, field) == 0.0, (
                    f"{field} 没有被 _weights_from_objectives 显式补 0，"
                    f"它拿到了类默认值 {getattr(RewardWeights(), field)}"
                )

    def test_reward_presets_cover_every_dimension(self):
        """每个预设必须写全所有维度，含 `plm`；缺项会退回类默认值。"""
        from pepdesign.utils.config import REWARD_PRESETS, RewardWeights

        expected = set(RewardWeights.model_fields)
        for name, preset in REWARD_PRESETS.items():
            assert set(preset) == expected, (
                f"预设 {name!r} 的维度集合与 RewardWeights 不一致："
                f"缺 {sorted(expected - set(preset))}，"
                f"多 {sorted(set(preset) - expected)}"
            )

    def test_unknown_objective_names_fall_back_with_a_warning(self):
        """无法映射的目标名退回 balanced，并发出 RuntimeWarning。"""
        with pytest.warns(RuntimeWarning, match="没有任何名称能映射到已知维度"):
            w = dynamic_weights("multi_objective", {"whatever": 0.5})
        assert w.model_dump() == REWARD_PRESETS["balanced"]

    def test_all_zero_objectives_falls_back_instead_of_raising(self):
        """全零 objectives 是脏数据，退回 balanced 而非抛错。"""
        with pytest.warns(RuntimeWarning, match="权重之和为 0"):
            w = dynamic_weights("multi_objective", {"热稳定性": 0.0})
        assert w.model_dump() == REWARD_PRESETS["balanced"]

    def test_negative_objectives_falls_back(self):
        with pytest.warns(RuntimeWarning):
            w = dynamic_weights("multi_objective", {"热稳定性": -1.0})
        assert w.model_dump() == REWARD_PRESETS["balanced"]

    def test_duplicate_alias_takes_the_larger_value(self):
        """同一维度由多个别名指定时取较大值。"""
        w = dynamic_weights("multi_objective", {"热稳定性": 0.3, "thermo": 0.5})
        assert w.thermo == pytest.approx(0.5)

    def test_unknown_task_falls_back_to_balanced_without_error(self):
        """未知任务名退回 balanced。"""
        assert dynamic_weights("some_future_task").model_dump() == REWARD_PRESETS["balanced"]

    def test_empty_task_falls_back_to_balanced(self):
        assert dynamic_weights("").model_dump() == REWARD_PRESETS["balanced"]

    def test_multi_objective_without_objectives_uses_balanced(self):
        assert dynamic_weights("multi_objective").model_dump() == REWARD_PRESETS["balanced"]

    def test_every_task_preset_name_exists(self):
        """TASK_PRESETS 中的预设名必须都存在于 REWARD_PRESETS。"""
        for task, preset in TASK_PRESETS.items():
            assert preset in REWARD_PRESETS, f"{task} -> {preset} 不存在"

    def test_weights_normalize_to_one(self):
        for task in TASK_PRESETS:
            weights = dynamic_weights(task).normalized().model_dump()
            assert sum(weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 配置里的权重（曾经的"配了不生效"）
# ---------------------------------------------------------------------------
class TestWeightsFromConfig:
    """`RewardConfig.weights_from_config()` 区分"显式配置"与"保持默认"。

    `weights` 的类默认值与 `preset="balanced"` 的权重数值相同，仅凭数值无法区分。
    """

    def test_default_config_reports_nothing_explicit(self):
        assert RewardConfig().weights_from_config() is None

    def test_empty_reward_weights_object_is_not_explicit(self):
        """`weights=RewardWeights()` 视为未显式指定。"""
        assert RewardConfig(weights=RewardWeights()).weights_from_config() is None

    def test_preset_alone_is_enough_to_be_explicit(self):
        w = RewardConfig(preset="activity_first").weights_from_config()
        assert w is not None
        assert w.affinity == pytest.approx(REWARD_PRESETS["activity_first"]["affinity"])

    def test_partial_weights_keep_the_preset_for_the_rest(self):
        """只写 thermo 时，其余维度保留预设份额。"""
        w = RewardConfig(
            preset="activity_first", weights=RewardWeights(thermo=0.8)
        ).weights_from_config()
        assert w is not None
        assert w.thermo == pytest.approx(0.8)
        assert w.affinity == pytest.approx(REWARD_PRESETS["activity_first"]["affinity"])
        assert w.protease == pytest.approx(REWARD_PRESETS["activity_first"]["protease"])

    def test_weights_alone_default_to_balanced_as_the_base(self):
        """未写 preset 时以 balanced 为底。"""
        w = RewardConfig(weights=RewardWeights(thermo=0.9)).weights_from_config()
        assert w is not None
        assert w.affinity == pytest.approx(REWARD_PRESETS["balanced"]["affinity"])


class TestConfigWeightsTakeEffect:
    """配置中的权重必须真正改变打分。

    `score()` 曾只读模块级预设、不读 `cfg`，而 `stats()` 报告的是 cfg 的值。
    每条用例同时断言总分变化与手算结果一致。
    """

    # 取值刻意在维度之间拉开差距：权重变了总分才一定跟着变。
    # 如果四个维度取值相同，换任何权重总分都不动，测试会假绿。
    VALUES: ClassVar[dict[str, float]] = {
        "thermo": 0.9,
        "protease": 0.1,
        "affinity": 0.9,
        "structure": 0.1,
    }

    def test_unset_config_leaves_task_presets_in_charge(self):
        """对照组：配置未显式指定时按 TASK_PRESETS 计权。"""
        b = build_with(RewardConfig(), self.VALUES).score(CLEAN_SEQ, {"task": "stabilize"})
        # stability_first: thermo .45 protease .35 affinity .15 structure .05
        expected = 0.45 * 0.9 + 0.35 * 0.1 + 0.15 * 0.9 + 0.05 * 0.1  # = 0.59
        assert b.total == pytest.approx(expected, abs=1e-4)

    def test_explicit_preset_overrides_the_task_preset(self):
        """显式选 activity_first → stabilize 任务也按 activity_first 算。"""
        b = build_with(RewardConfig(preset="activity_first"), self.VALUES).score(
            CLEAN_SEQ, {"task": "stabilize"}
        )
        # activity_first: thermo .15 protease .15 affinity .55 structure .15
        expected = 0.15 * 0.9 + 0.15 * 0.1 + 0.55 * 0.9 + 0.15 * 0.1  # = 0.66
        assert b.total == pytest.approx(expected, abs=1e-4)

    def test_explicit_weights_change_the_score(self):
        """只给 thermo 权重时总分退化为该维取值。"""
        cfg = RewardConfig(
            weights=RewardWeights(thermo=1.0, protease=0.0, affinity=0.0, structure=0.0)
        )
        b = build_with(cfg, self.VALUES).score(CLEAN_SEQ, {"task": "stabilize"})
        assert b.total == pytest.approx(0.9, abs=1e-4)
        assert b.dimensions["thermo"].weight == pytest.approx(1.0, abs=1e-3)
        assert b.dimensions["affinity"].weight == pytest.approx(0.0, abs=1e-3)

    def test_objectives_still_beat_the_config(self):
        """multi_objective 的 objectives 优先于配置。"""
        b = build_with(RewardConfig(preset="activity_first"), self.VALUES).score(
            CLEAN_SEQ, {"task": "multi_objective", "objectives": {"热稳定性": 1.0}}
        )
        assert b.dimensions["thermo"].weight == pytest.approx(1.0, abs=1e-3)
        assert b.dimensions["affinity"].weight == pytest.approx(0.0, abs=1e-3)

    def test_configured_weights_apply_without_a_task(self):
        """meta 中没有 task 时配置同样生效。"""
        cfg = RewardConfig(preset="stability_first")
        b = build_with(cfg, self.VALUES).score(CLEAN_SEQ)
        assert b.dimensions["thermo"].weight == pytest.approx(0.45, abs=1e-3)
        assert b.dimensions["affinity"].weight == pytest.approx(0.15, abs=1e-3)


# ---------------------------------------------------------------------------
# 加权求和
# ---------------------------------------------------------------------------
class TestWeightedSum:
    VALUES: ClassVar[dict[str, float]] = {
        "thermo": 0.8,
        "protease": 0.6,
        "affinity": 0.4,
        "structure": 0.2,
    }

    def test_matches_hand_computed_weighted_average(self):
        """balanced 预设下，总分 = 0.30·0.8 + 0.25·0.6 + 0.30·0.4 + 0.15·0.2。"""
        b = build(self.VALUES).score(CLEAN_SEQ)
        expected = 0.30 * 0.8 + 0.25 * 0.6 + 0.30 * 0.4 + 0.15 * 0.2  # = 0.54
        assert b.total == pytest.approx(expected, abs=1e-4)

    def test_no_penalties_when_disabled(self):
        b = build(self.VALUES).score(CLEAN_SEQ)
        assert b.penalties == {}

    def test_dimension_weights_are_the_normalized_preset(self):
        """`score.weight` 落在每个维度上，报告可以直接展示"这一维占多少"。"""
        b = build(self.VALUES).score(CLEAN_SEQ)
        assert b.dimensions["thermo"].weight == pytest.approx(0.30, abs=1e-4)
        assert b.dimensions["protease"].weight == pytest.approx(0.25, abs=1e-4)
        assert b.dimensions["affinity"].weight == pytest.approx(0.30, abs=1e-4)
        assert b.dimensions["structure"].weight == pytest.approx(0.15, abs=1e-4)

    def test_dimension_weights_sum_to_one(self):
        b = build(self.VALUES).score(CLEAN_SEQ)
        assert sum(d.weight for d in b.dimensions.values()) == pytest.approx(1.0, abs=1e-3)

    def test_total_is_a_weighted_average_not_a_sum(self):
        """总分必须落在各维度取值的 [min, max] 内；权重未归一化会使奖励尺度漂移。"""
        b = build(self.VALUES).score(CLEAN_SEQ)
        assert min(self.VALUES.values()) <= b.total <= max(self.VALUES.values())

    def test_stabilize_task_reweights_toward_thermo(self):
        """同一个序列，换任务 → 权重变 → 总分变。"""
        b = build(self.VALUES).score(CLEAN_SEQ, {"task": "stabilize"})
        # stability_first: thermo .45 protease .35 affinity .15 structure .05
        expected = 0.45 * 0.8 + 0.35 * 0.6 + 0.15 * 0.4 + 0.05 * 0.2  # = 0.62
        assert b.total == pytest.approx(expected, abs=1e-4)

    def test_only_scorers_that_ran_get_weight(self):
        """只注册部分维度时，权重需在它们之间重新归一化。"""
        b = build({"thermo": 0.8, "affinity": 0.2}).score(CLEAN_SEQ)
        assert sum(d.weight for d in b.dimensions.values()) == pytest.approx(1.0, abs=1e-3)
        assert b.total == pytest.approx(0.5, abs=1e-3)  # 0.30/0.60 与 0.30/0.60 各半

    def test_disabled_scorer_is_skipped_entirely(self):
        scorers = [
            FixedScorer("thermo", 0.8),
            FixedScorer("protease", 0.0, enabled=False),
        ]
        b = CompositeScorer(RewardConfig(), scorers=scorers, penalty_cfg=NO_PENALTIES)
        bd = b.score(CLEAN_SEQ)
        assert "protease" not in bd.dimensions
        assert bd.total == pytest.approx(0.8, abs=1e-4)  # 只剩 thermo，独占权重

    def test_breakdown_is_explainable(self):
        b = build(self.VALUES).score(CLEAN_SEQ)
        assert b.sequence == CLEAN_SEQ
        assert b.invalid_reason is None
        text = b.explain()
        assert "总分" in text and "thermo" in text


# ---------------------------------------------------------------------------
# 置信度与标定折扣
# ---------------------------------------------------------------------------
class TestConfidenceAndCalibration:
    def test_uncalibrated_dimension_loses_share(self):
        """未标定预测器的有效权重乘以 0.7，份额相应下降。"""
        scorers = [
            FixedScorer("thermo", 0.5, is_calibrated=False),
            FixedScorer("protease", 0.5, is_calibrated=True),
            FixedScorer("affinity", 0.5, is_calibrated=True),
            FixedScorer("structure", 0.5, is_calibrated=True),
        ]
        c = CompositeScorer(RewardConfig(), scorers=scorers, penalty_cfg=NO_PENALTIES)
        b = c.score(CLEAN_SEQ)
        # eff_thermo = 0.30 × 0.7 = 0.21；其余合计 0.70；总和 0.91
        assert b.dimensions["thermo"].weight == pytest.approx(0.21 / 0.91, abs=1e-4)
        # 标定过的维度份额等比上升
        assert b.dimensions["protease"].weight > 0.25

    def test_low_confidence_loses_share(self):
        """置信度 0.1 → 有效权重 0.03；重新归一化后 thermo 份额约为 0.03/0.28。"""
        scorers = [
            FixedScorer("thermo", 0.5, confidence=0.1),
            FixedScorer("protease", 0.5, confidence=1.0),
        ]
        c = CompositeScorer(RewardConfig(), scorers=scorers, penalty_cfg=NO_PENALTIES)
        b = c.score(CLEAN_SEQ)
        assert b.dimensions["thermo"].weight == pytest.approx(0.03 / 0.28, abs=1e-4)
        assert b.dimensions["protease"].weight == pytest.approx(0.25 / 0.28, abs=1e-4)
        # 与不折扣时的 0.545 / 0.455 相比，thermo 的份额显著下降
        assert b.dimensions["thermo"].weight < 0.30 / (0.30 + 0.25)

    def test_discount_does_not_shrink_total_magnitude(self):
        """折扣只改变份额，不改变量级；四维全为 0.5 时总分仍为 0.5。"""
        scorers = [
            FixedScorer("thermo", 0.5, confidence=0.1, is_calibrated=False),
            FixedScorer("protease", 0.5, confidence=1.0),
            FixedScorer("affinity", 0.5, confidence=0.3, is_calibrated=False),
            FixedScorer("structure", 0.5, confidence=1.0),
        ]
        c = CompositeScorer(RewardConfig(), scorers=scorers, penalty_cfg=NO_PENALTIES)
        assert c.score(CLEAN_SEQ).total == pytest.approx(0.5, abs=1e-4)

    def test_scorer_without_predictor_counts_as_calibrated(self):
        """规则型维度没有"标定"概念，不应打折。"""

        class NoPredictor(DimensionScorer):
            name = "protease"

            def score_one(self, sequence, meta):
                return DimensionScore(dimension="protease", value=1.0, raw=1.0)

        c = CompositeScorer(
            RewardConfig(), scorers=[NoPredictor()], penalty_cfg=NO_PENALTIES
        )
        assert c._is_calibrated("protease") is True

    def test_all_zero_confidence_falls_back_to_equal_weights(self):
        """全部置信度为 0 时退化为等权平均；总分恒为 0 会使 GRPO 组内 std 归零。"""
        scorers = [
            FixedScorer("thermo", 0.9, confidence=0.0),
            FixedScorer("protease", 0.1, confidence=0.0),
        ]
        c = CompositeScorer(RewardConfig(), scorers=scorers, penalty_cfg=NO_PENALTIES)
        b = c.score(CLEAN_SEQ)
        assert b.total == pytest.approx(0.5, abs=1e-4)  # 等权平均，不是 0
        # 中性分（0.5）而不是 0——0 会把模型往错误方向推
        assert b.dimensions["thermo"].weight == pytest.approx(0.5, abs=1e-3)

    def test_uncalibrated_discount_constant_is_not_a_no_op(self):
        """折扣必须小于 1，否则标定在奖励中不生效。"""
        assert 0.0 < UNCALIBRATED_DISCOUNT < 1.0


# ---------------------------------------------------------------------------
# 惩罚与裁剪
# ---------------------------------------------------------------------------
class TestPenalties:
    def test_penalties_are_subtracted_from_weighted_sum(self):
        # 用部分疏水序列而非全 L：全 L 会同时触发 low_complexity 等，总分被裁到 clip_min。
        seq = "LLLLLACDEFGH"
        scorers = [FixedScorer("thermo", 0.8), FixedScorer("protease", 0.8)]
        c = CompositeScorer(RewardConfig(), scorers=scorers)
        b = c.score(seq)
        assert b.penalties, "聚亮氨酸应触发至少一条软约束惩罚"
        weighted_sum = 0.8  # 两维同值，权重怎么分都是 0.8
        assert b.total == pytest.approx(weighted_sum - sum(b.penalties.values()), abs=1e-3)

    def test_invalid_penalty_is_applied_by_compute_penalties(self):
        """`PenaltyConfig.invalid` 必须被 `compute_penalties` 读取并应用。"""
        cfg = PenaltyConfig()
        pen = compute_penalties("KWK!LF", cfg)  # 非法字符
        assert pen == {"invalid": cfg.invalid}

    def test_empty_sequence_is_invalid_not_just_low_scoring(self):
        """解析阶段判死的输入没有分数，而不是分数低。"""
        b = build({"thermo": 0.9}).score("")
        assert b.invalid_reason == "empty_sequence"
        assert b.dimensions == {}
        assert b.total == pytest.approx(RewardConfig().clip_min)

    def test_meta_invalid_reason_propagates(self):
        b = build({"thermo": 0.9}).score("KWKLFKKIEK", {"invalid_reason": "parse_failed"})
        assert b.invalid_reason == "parse_failed"
        assert b.dimensions == {}

    def test_content_invalid_sequence_also_gets_a_reason(self):
        """回归测试：内容非法也要带 invalid_reason，否则会被当作正常样本进入组内统计。"""
        b = build({"thermo": 0.9}).score("KWK!LFKK")  # 含非氨基酸字符
        assert b.invalid_reason is not None
        assert "illegal_char" in b.invalid_reason
        # 与空序列同一处理：不打分，直接给下限分 + 明确的原因
        assert b.dimensions == {}
        assert b.total == pytest.approx(RewardConfig().clip_min)

    def test_too_long_sequence_is_invalid(self):
        """长度越界由 `validate` 判定。"""
        b = build({"thermo": 0.9}).score("A" * 250)
        assert b.invalid_reason == "too_long"

    def test_total_clipped_at_max(self):
        """打分器给出越界值时，总分被裁到 clip_max。"""
        b = build({"thermo": 1.5}).score(CLEAN_SEQ)
        assert b.total == pytest.approx(RewardConfig().clip_max)

    def test_total_clipped_at_min(self):
        """惩罚使总分低于 clip_min 时被裁住。"""
        scorers = [FixedScorer("thermo", 0.0)]
        cfg = PenaltyConfig(
            aggregation=5.0, hydrophobic_run=0.0, charge=0.0, invalid=0.0
        )
        c = CompositeScorer(RewardConfig(), scorers=scorers, penalty_cfg=cfg)
        b = c.score("LLLLLLLLLLLL")
        assert b.penalties  # 确实触发了惩罚，否则这条测试测的就不是裁剪
        assert b.total == pytest.approx(RewardConfig().clip_min)

    def test_penalty_values_are_positive_and_never_add(self):
        """惩罚为正数并从总分中减去；符号搞反会使模型主动触发惩罚。"""
        b = CompositeScorer(RewardConfig(), scorers=[FixedScorer("thermo", 0.9)]).score(
            "LLLLLLLLLLLL"
        )
        assert all(v > 0 for v in b.penalties.values())


class TestSimilarityConstraint:
    def test_identity_is_positional_not_set_based(self):
        """同一性按位置计算：`ACDEFG` 与 `GFEDCA` 的同一性为 0。"""
        assert sequence_identity("ACDEFG", "GFEDCA") == 0.0
        assert sequence_identity("ACDEFG", "ACDEFG") == 1.0

    def test_identity_penalizes_length_change(self):
        """分母取较长的那条，长度变化也计入差异。"""
        assert sequence_identity("ACDEFG", "ACDEF") == pytest.approx(5 / 6)

    def test_identity_with_empty_sequence_is_zero(self):
        assert sequence_identity("", "ACDEFG") == 0.0
        assert sequence_identity("ACDEFG", "") == 0.0

    def test_no_penalty_above_the_floor(self):
        assert similarity_penalty("ACDEFG", "ACDEFG") == 0.0
        # 10 残基改 3 个 = 0.7 同一性，正好卡在线上，不扣分
        assert similarity_penalty(BASE10, mutate(BASE10, 3)) == 0.0

    def test_penalty_grows_continuously_as_identity_drops(self):
        """扣分随同一性下降连续增长，而非布尔开关。"""
        penalties = [similarity_penalty(BASE10, mutate(BASE10, k)) for k in (3, 4, 5, 7, 9)]
        assert penalties[0] == 0.0, "恰好卡在阈值上不该扣分"
        assert penalties == sorted(penalties)
        assert penalties == sorted(set(penalties)), "每一档都要有区分度，不能压平"
        assert penalties[-1] > penalties[0]

    def test_penalty_is_capped_at_strength(self):
        """扣分上限为 SIMILARITY_PENALTY_STRENGTH。"""
        assert similarity_penalty("ACDEFGHIKL", "WWWWWWWWWW") <= SIMILARITY_PENALTY_STRENGTH

    def test_no_base_sequence_means_no_constraint(self):
        """没有基准序列时相似度约束自动失效。"""
        assert similarity_penalty("", "ACDEFGHIKL") == 0.0

    def test_floor_constant_is_sane(self):
        assert 0.0 < SIMILARITY_FLOOR < 1.0

    def test_meta_base_sequence_triggers_penalty_in_score(self):
        b = build({"thermo": 0.8}).score(
            "WWWWWWWWWW", {"base_sequence": "ACDEFGHIKL"}
        )
        assert "similarity" in b.penalties
        assert b.total < 0.8

    def test_identical_to_base_gets_no_similarity_penalty(self):
        b = build({"thermo": 0.8}).score(
            "ACDEFGHIKL", {"base_sequence": "ACDEFGHIKL"}
        )
        assert "similarity" not in b.penalties


class TestCyclicConstraint:
    def test_no_cyclic_type_means_no_constraint(self):
        b = build({"thermo": 0.8}).score(CLEAN_SEQ)
        assert "cyclic" not in b.penalties

    def test_explicit_none_means_no_constraint(self):
        """`cyclic_type: "none"` 是合法取值，不构成环化约束。"""
        b = build({"thermo": 0.8}).score(CLEAN_SEQ, {"cyclic_type": "none"})
        assert "cyclic" not in b.penalties

    def test_cyclic_type_adds_a_penalty_key(self):
        b = build({"thermo": 0.8}).score(CLEAN_SEQ, {"cyclic_type": "head_to_tail"})
        # 满足约束时罚 0，所以键可能不出现；出现了就必须是正的、有上限的
        if "cyclic" in b.penalties:
            assert 0.0 < b.penalties["cyclic"] <= CYCLIC_PENALTY_STRENGTH

    def test_unknown_cyclic_type_warns_and_applies_full_penalty(self):
        """未知环化方式按满额惩罚处理并告警，不中断 training step。"""
        with pytest.warns(RuntimeWarning, match="cyclic_type"):
            b = build({"thermo": 0.8}).score(CLEAN_SEQ, {"cyclic_type": "bogus_cycle"})
        assert b.penalties["cyclic"] == pytest.approx(CYCLIC_PENALTY_STRENGTH)


# ---------------------------------------------------------------------------
# 打分器异常隔离
# ---------------------------------------------------------------------------
class TestScorerFailureIsolation:
    def test_failing_dimension_gets_neutral_score_and_records_error(self):
        """单个维度抛异常时返回中性分并记录错误，不影响其他维度。"""

        class Exploding(DimensionScorer):
            name = "thermo"

            def score_one(self, sequence, meta):
                raise ValueError("模拟预测器故障")

        c = CompositeScorer(
            RewardConfig(),
            scorers=[Exploding(), FixedScorer("protease", 0.9)],
            penalty_cfg=NO_PENALTIES,
        )
        b = c.score(CLEAN_SEQ)
        failed = b.dimensions["thermo"]
        assert failed.value == 0.5
        assert failed.confidence == 0.0
        assert "ValueError" in failed.detail["error"]
        # 另一个维度照常打分
        assert b.dimensions["protease"].value == 0.9

    def test_failed_dimension_gets_no_effective_weight(self):
        """失败维度置信度为 0，因此不参与总分归一化。"""
        class Exploding(DimensionScorer):
            name = "thermo"

            def score_one(self, sequence, meta):
                raise RuntimeError("boom")

        c = CompositeScorer(
            RewardConfig(),
            scorers=[Exploding(), FixedScorer("protease", 1.0)],
            penalty_cfg=NO_PENALTIES,
        )
        b = c.score(CLEAN_SEQ)
        assert b.total == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 批量接口
# ---------------------------------------------------------------------------
class TestScoreGroup:
    def test_single_meta_broadcasts_to_the_whole_group(self):
        """组内序列共享 meta 时，单条 meta 需广播到全组。"""
        group = [f"KWKLFKKIEK{v}K" for v in "ACDEFG"]
        scores = build({"thermo": 0.5}).score_group(group, [{"task": "stabilize"}])
        assert len(scores) == len(group)

    def test_none_meta_gives_one_result_per_sequence(self):
        group = [f"KWKLFKKIEK{v}K" for v in "ACDEFG"]
        assert len(build({"thermo": 0.5}).score_group(group)) == len(group)

    def test_matching_metas_are_used_per_sequence(self):
        group = ["KWKLFKKIEK", "ACDEFGHIKL"]
        metas = [{"task": "stabilize"}, {"task": "de_novo"}]
        scores = build({"thermo": 0.7}).score_group(group, metas)
        assert len(scores) == 2

    def test_mismatched_meta_count_raises(self):
        """回归测试：序列与 meta 数量不一致必须报错；`zip` 会静默截断并使组内 std 为 0。"""
        group = [f"KWKLFKKIEK{v}K" for v in "ACDEFG"]
        with pytest.raises(ValueError, match="数量不一致"):
            build({"thermo": 0.5}).score_group(group, [{}, {}])

    def test_empty_group_returns_empty(self):
        assert build({"thermo": 0.5}).score_group([]) == []

    def test_group_of_one_with_one_meta_is_fine(self):
        """组大小为 1 时不能被广播规则误伤。"""
        scores = build({"thermo": 0.5}).score_group(["KWKLFKKIEK"], [{"task": "de_novo"}])
        assert len(scores) == 1

    def test_result_order_matches_input_order(self):
        """分数必须与输入逐位对应。"""
        group = ["KWKLFKKIEK", "ACDEFGHIKL", "RRWQWRRRRW"]
        metas = [{"task": "stabilize"}, {"task": "de_novo"}, {"task": "multi_objective"}]
        scores = build({"thermo": 0.5}).score_group(group, metas)
        assert [s.sequence for s in scores] == group


class TestStats:
    """`stats()` 必须报告打分会使用的权重，而非配置里写的值。"""

    def test_reports_task_preset_source_when_config_is_silent(self):
        s = build({"thermo": 0.5, "protease": 0.5}).stats()
        assert s["weight_source"] == "task_preset"
        # 配置未显式指定 → 这里必须是 None，不能用 fallback 值伪装成已配置。
        assert s["config"]["weights"] is None
        assert set(s["scorers"]) == {"thermo", "protease"}

    def test_task_presets_are_always_listed(self):
        """无论来源如何，逐任务的生效权重都要报出来。"""
        s = build({"thermo": 0.5}).stats()
        assert set(s["task_presets"]) == set(TASK_PRESETS)
        for task, weights in s["task_presets"].items():
            assert sum(weights.values()) == pytest.approx(1.0), task

    def test_reports_config_source_when_explicitly_set(self):
        cfg = RewardConfig(preset="activity_first")
        s = CompositeScorer(
            cfg,
            scorers=[FixedScorer("thermo", 0.5)],
            penalty_cfg=NO_PENALTIES,
        ).stats()
        assert s["weight_source"] == "config"
        assert s["config"]["preset"] == "activity_first"
        assert s["config"]["weights"]["affinity"] == pytest.approx(
            REWARD_PRESETS["activity_first"]["affinity"]
        )
        assert sum(s["config"]["weights"].values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 端到端：真实预测器
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def scorer() -> CompositeScorer:
    """真预测器的聚合器；模块级共享以避免重复构造并复用预测器缓存。"""
    return CompositeScorer(RewardConfig())


class TestEndToEndWithRealPredictors:
    """用真预测器跑一小批，确认四维链路真的能一起工作。"""

    def test_four_dimensions_all_present(self, scorer):
        b = scorer.score("KWKLFKKIEKVK")
        assert set(b.dimensions) == {"thermo", "protease", "affinity", "structure"}

    def test_scores_are_finite_and_in_unit_interval(self, scorer):
        for seq in ["KWKLFKKIEKVK", "GIGAVLKVLTTGLPALISWIKRKRQQ",
                    "RRWQWR", "AILAILAILAIL", "ACDEFGHIKLMNPQRSTVWY"]:
            b = scorer.score(seq)
            assert math.isfinite(b.total), seq
            for name, d in b.dimensions.items():
                assert 0.0 <= d.value <= 1.0, f"{seq} 的 {name} = {d.value}"

    def test_total_within_clip_bounds(self, scorer):
        cfg = RewardConfig()
        for seq in ["KWKLFKKIEKVK", "LLLLLLLLLLLL", "RRWQWR"]:
            assert cfg.clip_min <= scorer.score(seq).total <= cfg.clip_max

    def test_different_peptides_get_different_totals(self, scorer):
        """奖励必须有区分度；全同分意味着 GRPO 组内优势恒为 0。"""
        seqs = ["KWKLFKKIEKVK", "AILAILAILAIL", "RRWQWR", "GPGPGPGPGP"]
        totals = {scorer.score(s).total for s in seqs}
        assert len(totals) >= 3

    def test_predictors_are_reused_across_calls(self, scorer):
        """打分器实例在构造时建立一次，预测器缓存不应被逐条重建。"""
        a = scorer.score("KWKLFKKIEKVK")
        b = scorer.score("KWKLFKKIEKVK")
        assert a.total == b.total

    def test_batch_matches_single_calls(self, scorer):
        """批量与逐条的结果必须一致，否则训练和评估口径不一致。"""
        seqs = ["KWKLFKKIEKVK", "RRWQWR"]
        group = scorer.score_group(seqs, [{"task": "de_novo"}])
        assert [s.total for s in group] == [scorer.score(s).total for s in seqs]
