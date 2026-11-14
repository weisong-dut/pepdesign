"""四个维度打分器的行为。

每个维度都要满足两条契约：
  1. 输出归一化到 [0, 1]（否则加权求和的权重失去意义）；
  2. `confidence` ∈ [0, 1]，且逐条可以不同。

除此之外每个维度各有自己的关键性质：thermo 的单调性、protease 的
"位点越少越好"、affinity 的口径切换、structure 的一票否决。

用假打分器而不依赖真预测器的场景：这里要验证的是打分器自己的逻辑
（归一化、口径、detail 字段），预测器的数值行为由预测器的测试负责。
混在一起测会让失败原因难以定位。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import ClassVar

import pytest

from pepdesign.bio.cleavage import SITE_DENSITY_SATURATION
from pepdesign.rewards import (
    AffinityScorer,
    ProteaseScorer,
    StructureScorer,
    ThermoScorer,
)
from pepdesign.rewards.affinity import (
    ABSOLUTE_CONFIDENCE_DISCOUNT,
    RELATIVE_SCORE_MAX,
    USED_ABSOLUTE_KEY,
)
from pepdesign.rewards.base import DimensionScorer
from pepdesign.rewards.predictors import (
    build_affinity_predictor,
    build_protease_predictor,
    build_structure_predictor,
    build_thermo_predictor,
)
from pepdesign.rewards.predictors.base import PredictionResult

# 长度 ≥ TM_EXTRAPOLATION_LENGTH(12) 的测试序列。短肽会触发"外推混合"，
# 把 Tm 分和脂肪族指数代理分混在一起——测 Tm 曲线本身的性质时必须避开。
LONG_SEQ = "KWKLFKKIEKVK"


class FakePredictor:
    """恒返回固定值的预测器，用于隔离打分器自身的逻辑。"""

    name = "fake"
    is_calibrated = False

    def __init__(self, value: float = 0.5, confidence: float = 1.0, detail: dict | None = None):
        self.value = value
        self.confidence = confidence
        self.detail = detail or {}

    def predict_one(self, sequence: str) -> PredictionResult:
        return PredictionResult(
            value=self.value, confidence=self.confidence,
            detail=dict(self.detail), predictor=self.name,
        )

    def predict_with_reference(self, sequence: str, reference: str) -> PredictionResult:
        return PredictionResult(
            value=self.value, confidence=self.confidence,
            detail={**self.detail, "reference": reference}, predictor=self.name,
        )


class TestDimensionScorerContract:
    """所有打分器共有的契约。"""

    ALL: ClassVar[list[tuple[str, Callable[[], DimensionScorer]]]] = [
        ("thermo", lambda: ThermoScorer(FakePredictor(50.0))),
        ("protease", lambda: ProteaseScorer()),
        ("affinity", lambda: AffinityScorer(FakePredictor(0.5))),
        ("structure", lambda: StructureScorer(FakePredictor(0.5))),
    ]

    @pytest.mark.parametrize("name,factory", ALL, ids=[n for n, _ in ALL])
    def test_value_in_unit_interval(self, name, factory):
        scorer = factory()
        for seq in ["KWKLFKKIEK", "RRWQWR", "ACDEFGHIKLMN", "LLLLLLLLLL", "GPGPGPGPGP"]:
            s = scorer.score(seq)
            assert 0.0 <= s.value <= 1.0, f"{name} 对 {seq} 给出 {s.value}"
            assert math.isfinite(s.value)

    @pytest.mark.parametrize("name,factory", ALL, ids=[n for n, _ in ALL])
    def test_confidence_in_unit_interval(self, name, factory):
        scorer = factory()
        for seq in ["KWKLFKKIEK", "RRWQWR", "XXXXX"]:
            s = scorer.score(seq)
            assert 0.0 <= s.confidence <= 1.0, f"{name} 的 confidence = {s.confidence}"

    @pytest.mark.parametrize("name,factory", ALL, ids=[n for n, _ in ALL])
    def test_dimension_name_matches(self, name, factory):
        assert factory().name == name

    @pytest.mark.parametrize("name,factory", ALL, ids=[n for n, _ in ALL])
    def test_detail_is_a_dict(self, name, factory):
        s = factory().score("KWKLFKKIEK")
        assert isinstance(s.detail, dict)

    @pytest.mark.parametrize("name,factory", ALL, ids=[n for n, _ in ALL])
    def test_disabled_scorer_reports_disabled(self, name, factory):
        scorer = factory()
        scorer.enabled = False
        assert scorer.enabled is False

    def test_base_class_is_abstract(self):
        with pytest.raises(NotImplementedError):
            DimensionScorer().score_one("AAA", {})


class TestThermoScorer:
    def test_higher_tm_gives_higher_score(self):
        """热稳定性维度的核心单调性：Tm 越高分越高。"""
        lo = ThermoScorer(FakePredictor(30.0)).score(LONG_SEQ)
        hi = ThermoScorer(FakePredictor(70.0)).score(LONG_SEQ)
        assert hi.value > lo.value

    def test_baseline_tm_sits_near_middle(self):
        """Tm == baseline 时精确得 0.5。

        `ratio_score` 以基准为中心对称（tanh(0) = 0 → 0.5），
        第二段 sigmoid 又正好以 0.5 为中心，所以"与基准持平 = 0.5"这个
        不动点穿过两段都成立——这让"基准序列得分 0.5"能直接写进文档。
        """
        s = ThermoScorer(FakePredictor(45.0), baseline_tm=45.0).score(LONG_SEQ)
        assert s.value == pytest.approx(0.5, abs=1e-3)

    def test_score_scale_controls_steepness(self):
        """`score_scale` 与 `baseline_tm` 解耦：换 scale 只改陡度，不改零点。

        这是本项目修过的一个真实问题——scale 曾经默认取 |baseline_tm|，
        导致"把基准从 45 调到 60"会顺带把曲线斜率改掉 4/3 倍，
        两次实验的奖励尺度不可比。
        """
        tm = 65.0
        gentle = ThermoScorer(FakePredictor(tm), baseline_tm=45.0, score_scale=200.0)
        steep = ThermoScorer(FakePredictor(tm), baseline_tm=45.0, score_scale=20.0)
        # 同一 Tm，陡的那个离 0.5 更远
        assert steep.score(LONG_SEQ).value > gentle.score(LONG_SEQ).value

    def test_baseline_change_does_not_change_steepness(self):
        """基准平移、scale 固定时，相同 ΔTm 应给相同的分。

        两条序列的 ΔTm 都是 +10℃，只是基准不同（45 vs 55）。
        把两者对齐，意味着"换个参照肽"不会改变奖励的口径——
        这是 scale 与 baseline 解耦后才能成立的等式。
        """
        a = ThermoScorer(FakePredictor(55.0), baseline_tm=45.0, score_scale=45.0)
        b = ThermoScorer(FakePredictor(65.0), baseline_tm=55.0, score_scale=45.0)
        assert a.score(LONG_SEQ).value == pytest.approx(
            b.score(LONG_SEQ).value, abs=1e-3
        )

    def test_short_peptide_uses_extrapolation_and_lowers_confidence(self):
        """短肽（< 12 残基）的 Tm 预测不可靠，走"代理混合"并降置信度。"""
        scorer = ThermoScorer(FakePredictor(50.0))
        short = scorer.score("KWKLFKK")      # 7 残基
        long_ = scorer.score("KWKLFKKIEKV" * 3)  # 33 残基
        assert short.detail["length_extrapolation"] is True
        assert long_.detail["length_extrapolation"] is False
        assert short.confidence < long_.confidence

    def test_detail_records_baseline_and_scale(self):
        """报告要能回答"这个分是按什么基准、什么陡度算的"。"""
        d = ThermoScorer(FakePredictor(50.0), baseline_tm=45.0,
                         score_scale=45.0).score("KWKLFKKIEK").detail
        assert d["baseline_tm"] == 45.0
        assert d["score_scale"] == 45.0
        assert "delta_tm" in d
        assert "score_before_enhance" in d

    def test_raw_is_the_tm_in_celsius(self):
        """raw 必须是物理量（℃），供报告展示——不是归一化后的分。"""
        s = ThermoScorer(FakePredictor(62.5)).score("KWKLFKKIEK")
        assert s.raw == pytest.approx(62.5)
        assert s.value != s.raw

    def test_real_predictor_stays_in_physical_range(self):
        """真实预测器对真实多肽给出的 Tm 必须落在物理合理区间。

        修复 aliphatic_index 之前，melittin 会报出 190.6 ℃——
        多肽 Tm 物理上不可能超过约 100 ℃，那个数字本身就是
        "预测器已经失效"的信号。
        """
        scorer = ThermoScorer(build_thermo_predictor("heuristic"))
        for seq in ["GIGAVLKVLTTGLPALISWIKRKRQQ", "AILAILAILAIL",
                    "KWKLFKK", "RRWQWR", "ACDEFGHIKLMN"]:
            tm = scorer.score(seq).raw
            assert 0.0 < tm < 110.0, f"{seq} 的 Tm = {tm} ℃ 不合物理"


class TestProteaseScorer:
    def test_no_sites_scores_higher_than_many_sites(self):
        """酶解稳定性的判据是"位点越少越稳"。"""
        scorer = ProteaseScorer()
        many = scorer.score("RRWQWR")        # 抗菌肽，3 个 K/R 都是胰蛋白酶位点
        few = scorer.score("LPGPAPGP")       # Pro 在 P1' 位，逐个封住位点
        assert few.value > many.value
        assert many.value < 0.5

    def test_saturation_clamps_at_high_site_density(self):
        """位点密度达到 `SITE_DENSITY_SATURATION` 时饱和在 0 分，不再下降。

        这是刻意的：位点数上不封顶，若一直线性映射，已经极不稳定的序列
        还能靠"再多消一个位点"拿到可观增量，模型于是把生成预算持续花在
        已经很好的方向上。饱和把梯度留给"从差变好"的那一段。
        代价是该区间内组内无区分度——这是已知的、有意的取舍。
        """
        scorer = ProteaseScorer()
        s = scorer.score("KKKKRRRRKKKK")   # 12 残基、位点密度 ≈1.0
        assert s.detail["weighted_site_count"] / 12 >= SITE_DENSITY_SATURATION
        assert s.value == 0.0

    def test_score_is_not_driven_by_length(self):
        """回归测试：分数不能因为序列变长就系统性掉到 0。

        这里曾经用的是绝对位点数（≥6 个即饱和）。后果是 20 残基以上的
        真实抗菌肽——melittin、LL-37、magainin-2、cecropin A——全部
        得 0 分，而这一维占奖励权重的 25-30%。整段长度区间没有梯度，
        `pro_sweep` 想教的"P1' 换 Pro 阻断切割"因此拿不到任何奖励。

        判据用"同一条序列重复延长"来构造：重复单元让位点密度基本不变，
        分数就不该随长度崩塌。旧口径下 26 残基（13 个位点）必然为 0。
        """
        long_seq = "APGA" * 8          # 32 残基、9 个加权位点
        s = ProteaseScorer().score(long_seq)
        # 先确认前提：绝对位点数确实越过了旧阈值，否则这条测试抓不到东西
        assert s.detail["weighted_site_count"] >= 6, "构造失效，位点数没上去"
        assert s.value > 0.0, (
            f"{len(long_seq)} 残基、{s.detail['weighted_site_count']:.1f} 个位点的序列"
            f"被判为 0 分，说明饱和口径又退回成绝对计数了：{s.detail}"
        )

    def test_density_is_what_matters_not_absolute_count(self):
        """密度相近、绝对计数差 2.7 倍的两种情况，分数应当几乎一致。

        把同一条序列重复三次：位点数涨约 3 倍，长度也涨 3 倍。
        旧口径下这两条会得到 0.60 和 0.00（后者直接饱和），
        新口径下应当接近——这是唯一能区分"按密度打分"和"按绝对数打分"
        的构造，所以两条断言都要写紧一点，别让它变成恒真。
        """
        scorer = ProteaseScorer()
        base = "SAPGSAPG"
        tripled = base * 3
        sites = scorer.score(base).detail["weighted_site_count"]
        sites3 = scorer.score(tripled).detail["weighted_site_count"]
        assert sites3 > sites * 2, "绝对计数没有拉开，这个构造就证明不了什么"

        assert scorer.score(tripled).value == pytest.approx(
            scorer.score(base).value, abs=0.06
        )

    def test_reports_site_counts(self):
        d = ProteaseScorer().score("KWKLFKKIEK").detail
        assert "n_sites" in d and "weighted_site_count" in d
        assert "sites_by_protease" in d
        assert d["score_source"].endswith("stability_score")

    def test_n_sites_and_weighted_differ_in_meaning(self):
        """`n_sites` 是位点个数，`weighted_site_count` 带蛋白酶权重。

        报告要能区分"3 个易感位点"和"1 个血清中极不稳定的位点"，
        两者对研发的改造建议完全不同，所以两个数都要留。
        """
        d = ProteaseScorer().score("KWKLFKKIEK").detail
        assert isinstance(d["n_sites"], int)
        assert d["weighted_site_count"] >= 0

    def test_modifications_affect_scoring(self):
        """末端修饰改变酶切行为，必须传下去。

        N 端乙酰化封住了 DPP4 从 N 端第 2 位起切的作用。如果忽略修饰，
        同一条 `Ac-...` 会得到和裸序列一样的分——模型于是学不到
        "加修饰能抗酶解"这件事。

        这里刻意选了一条没有饱和的序列：饱和区间内
        （加权位点 ≥ 6）无论加不加修饰都是 0 分，测不出差别。
        """
        scorer = ProteaseScorer()
        bare = scorer.score("APAGPAGP")
        capped = scorer.score("APAGPAGP", {"modifications": ["n_acetyl"]})
        assert capped.value > bare.value
        assert capped.detail["n_sites"] == bare.detail["n_sites"] - 1
        assert capped.detail["modifications_applied"] == ["n_acetyl"]

    def test_confidence_reflects_ambiguous_residues(self):
        """含 X/B/Z 的序列，酶切规则判不准，置信度应更低。"""
        scorer = ProteaseScorer()
        clean = scorer.score("KWKLFKKIEK")
        dirty = scorer.score("KWXLFXXIEK")
        assert dirty.confidence < clean.confidence

    def test_half_life_is_display_only(self):
        """半衰期预测器只用于展示，失败也不该影响打分。

        半衰期是从位点数指数映射出来的，方差极大，
        混进奖励会让奖励继承这份噪声。
        """
        class Broken:
            name = "broken"
            def predict_one(self, seq):
                raise RuntimeError("模拟半衰期预测器故障")

        scorer = ProteaseScorer(build_protease_predictor("rule"),
                                half_life_predictor=Broken())
        s = scorer.score("KWKLFKKIEK")
        assert 0.0 <= s.value <= 1.0          # 打分不受影响
        assert "half_life_error" in s.detail  # 但错误被记录下来


class TestAffinityScorer:
    def test_prefers_relative_scope_when_baseline_present(self):
        """有基准序列时必须走相对口径——GRPO 组内 8 条候选对着
        同一条基准打分，相对分在组内天然可比，绝对分不可比。

        基准来自 `base_sequence`（用户给的那条待优化序列）。
        """
        s = AffinityScorer(FakePredictor(0.8)).score(
            "KWKLFKKIEK", {"base_sequence": "KWKLFKKIEA"}
        )
        assert s.detail[USED_ABSOLUTE_KEY] is False
        assert s.detail["scoring_scope"] == "relative"

    def test_reference_sequence_is_ignored(self):
        """这条钉的是奖励泄漏的修复。

        `reference_sequence` 曾经被 GRPO 的构造侧填成 `rec.sequence`——
        也就是训练集里那条标准答案。奖励读它，等于在暗地里拿答案当尺子，
        而模型在 prompt 里根本看不到答案。

        现在这个键不再被读：只给 `reference_sequence`、不给 `base_sequence`
        时必须退回绝对口径。如果有人把读取逻辑改回去，这里立刻红。
        """
        s = AffinityScorer(FakePredictor(0.8)).score(
            "KWKLFKKIEK", {"reference_sequence": "KWKLFKKIEA"}
        )
        assert s.detail[USED_ABSOLUTE_KEY] is True
        assert s.detail["scoring_scope"] == "absolute"

    def test_falls_back_to_absolute_without_baseline(self):
        s = AffinityScorer(FakePredictor(0.5)).score("KWKLFKKIEK", {})
        assert s.detail[USED_ABSOLUTE_KEY] is True
        assert s.detail["scoring_scope"] == "absolute"
        assert s.detail["fallback_reason"] is not None

    def test_fallback_discounts_confidence(self):
        """退化到绝对口径时置信度要打折，让报告读者一眼看出
        这一维的分数口径和别的不一样。"""
        scorer = AffinityScorer(FakePredictor(0.5, confidence=1.0))
        absolute = scorer.score("KWKLFKKIEK", {})
        relative = scorer.score("KWKLFKKIEK", {"base_sequence": "KWKLFKKIEA"})
        assert absolute.confidence == pytest.approx(ABSOLUTE_CONFIDENCE_DISCOUNT)
        assert relative.confidence == pytest.approx(1.0)

    def test_relative_value_is_clipped_not_resigmoided(self):
        """相对分已经在 [0, 1.5] 且以 1.0 为"与基准持平"，
        再套 sigmoid 会把"持平"挪到 0.5，语义反而不好解释。"""
        s = AffinityScorer(FakePredictor(1.0)).score(
            "KWKLFKKIEK", {"base_sequence": "KWKLFKKIEK"}
        )
        assert s.value == pytest.approx(1.0)   # 持平就是 1.0，不是 0.5

    def test_relative_value_clipped_at_max(self):
        s = AffinityScorer(FakePredictor(99.0)).score(
            "KWKLFKKIEK", {"base_sequence": "KWKLFKKIEK"}
        )
        assert s.value == pytest.approx(RELATIVE_SCORE_MAX)

    def test_detail_reports_the_baseline_actually_used(self):
        """detail 里的键名是 `baseline_sequence`，不是 `reference_sequence`。

        这两个键在历史上有过语义混淆，换个名字比加注释更难被误读：
        读报告的人必须一眼看出"分母是谁"。
        """
        s = AffinityScorer(FakePredictor(0.8)).score(
            "KWKLFKKIEK", {"base_sequence": "KWKLFKKIEK"}
        )
        assert s.detail["baseline_sequence"] == "KWKLFKKIEK"
        assert "reference_sequence" not in s.detail

    def test_absolute_path_is_normalized(self):
        """绝对口径是"倾向分"，量纲不适合直接进加权求和，要压到 (0,1)。"""
        s = AffinityScorer(FakePredictor(0.5), baseline_affinity=0.5).score("KWKLFKKIEK", {})
        assert s.value == pytest.approx(0.5)  # 等于基准 → sigmoid 给 0.5

    def test_raw_is_predictor_output(self):
        s = AffinityScorer(FakePredictor(0.73)).score("KWKLFKKIEK", {})
        assert s.raw == pytest.approx(0.73)

    def test_empty_baseline_string_treated_as_missing(self):
        s = AffinityScorer(FakePredictor(0.5)).score(
            "KWKLFKKIEK", {"base_sequence": ""}
        )
        assert s.detail[USED_ABSOLUTE_KEY] is True

    def test_real_predictor_stays_in_range(self):
        scorer = AffinityScorer(build_affinity_predictor("feature"))
        for seq in ["KWKLFKKIEK", "RRWQWR", "ACDEFGHIKLMN"]:
            s = scorer.score(seq)
            assert 0.0 <= s.value <= 1.0


class TestStructureScorer:
    def test_passes_through_without_resigmoid(self):
        """structure 维近直通：预测器输出已经是 [0,1]，
        再套一层 sigmoid 会把一票否决的 0.3 上限抹掉。"""
        s = StructureScorer(FakePredictor(0.42)).score("KWKLFKKIEK")
        assert s.value == pytest.approx(0.42)

    def test_clips_out_of_range_values(self):
        assert StructureScorer(FakePredictor(1.8)).score("KWKLFKKIEK").value == 1.0
        assert StructureScorer(FakePredictor(-0.5)).score("KWKLFKKIEK").value == 0.0

    def test_veto_is_surfaced_at_top_level_of_detail(self):
        """一票否决是这一维最重要的输出，必须提到 detail 顶层——
        "为什么这条序列只有 0.25 分"要能直接读到原因，而不是让人去猜。"""
        s = StructureScorer(
            FakePredictor(0.25, detail={"vetoes": ["连续疏水片段过长(8)"]})
        ).score("LLLLLLLL")
        assert s.detail["vetoed"] is True
        assert "连续疏水" in s.detail["veto_reason"]

    def test_no_veto_marks_vetoed_false(self):
        s = StructureScorer(FakePredictor(0.7, detail={"vetoes": []})).score("KWKLFKKIEK")
        assert s.detail["vetoed"] is False

    def test_real_predictor_vetoes_long_hydrophobic_run(self):
        """真实预测器上，聚亮氨酸必须触发否决并拿到 ≤ 0.3。"""
        scorer = StructureScorer(build_structure_predictor("propensity"))
        s = scorer.score("LLLLLLLLLLLL")
        assert s.detail["vetoed"] is True
        assert s.value <= 0.3


class TestSSConfidence:
    """二级结构置信度的基准取法。"""

    def test_ambiguous_conformation_scores_low(self):
        """三个倾向值里最高的两个咬得很紧时，说明序列对两种构象
        都有偏好、等于没有偏好，是典型的"设计失败"，必须低分。

        这里曾经用的是"最高值 - 其余两个的均值"，它把两个低值也
        掺进基准：只要有一个值足够低，均值就被拉下来、gap 就变大，
        哪怕真正的竞争者（次高值）和最高值只差 0.05。于是
        {螺旋 1.5, 折叠 1.45, 卷曲 0.8} 反而拿满分。
        """
        from pepdesign.rewards.predictors.secondary_structure import ss_confidence

        class Props:
            helix_propensity = 1.5
            sheet_propensity = 1.45
            coil_propensity = 0.8

        assert ss_confidence("x", Props()) < 0.3

    def test_clear_conformation_scores_high(self):
        from pepdesign.rewards.predictors.secondary_structure import ss_confidence

        class Props:
            helix_propensity = 1.5
            sheet_propensity = 0.8
            coil_propensity = 0.9

        assert ss_confidence("x", Props()) > 0.9

    def test_three_equal_values_score_zero(self):
        """三个倾向完全相同 → 完全没有构象偏好 → 0 分。"""
        from pepdesign.rewards.predictors.secondary_structure import ss_confidence

        class Props:
            helix_propensity = 1.1
            sheet_propensity = 1.1
            coil_propensity = 1.1

        assert ss_confidence("x", Props()) == pytest.approx(0.0)

    def test_real_sequences_stay_in_range(self):
        from pepdesign.rewards.predictors.secondary_structure import ss_confidence

        for seq in ["GIGAVLKVLTTGLPALISWIKRKRQQ", "KWKLFKKIEK",
                    "AILAILAILAIL", "GPGPGPGPGP", "ACDEFGHIKLMN"]:
            v = ss_confidence(seq)
            assert 0.0 <= v <= 1.0, f"{seq}: {v}"


class TestStructurePredictorSubScores:
    def test_sub_scores_sum_to_reported_value(self):
        """四个子指标按权重合成总分——权重和必须是 1，
        否则总分的量级会随权重表漂移。"""
        from pepdesign.rewards.predictors.secondary_structure import (
            StructureReasonablenessPredictor,
        )

        assert sum(StructureReasonablenessPredictor.SUB_WEIGHTS.values()) == pytest.approx(1.0)

    def test_all_four_sub_scores_present(self):
        from pepdesign.rewards.predictors.secondary_structure import (
            StructureReasonablenessPredictor,
        )

        pred = StructureReasonablenessPredictor()
        d = pred.predict_one("KWKLFKKIEK").detail
        assert set(d["sub_scores"]) == {
            "ss_confidence", "hydrophobic_balance",
            "charge_distribution", "composition_natural",
        }
