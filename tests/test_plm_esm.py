"""第五维（ESM-2 掩码边缘概率）的纯逻辑测试。

本文件不 import torch（conftest.py 的硬约束），所以它测的是
"不需要模型权重就能确定"的那部分：掩码变体的生成、log 概率到分数的
归一化、工厂注册表的一致性、以及"这一维默认不参与训练"这条契约。

真正的前向（批量掩码前向 + 回填 log 概率）在
`tests/test_train_cpu_smoke.py` 里用一个随机初始化的小 ESM 端到端验证。

默认权重为 0 的单独测试
----------------------------------
意见书对奖励体系的核心批评是"自己验证自己"。PLM 通道是用来打破这个
闭环的外部裁判——而外部裁判一旦进入被优化的目标，就不再是裁判了
（Goodhart 定律）。所以"默认不参与加权求和"不是实现细节，
它是这一维存在的理由本身，必须有测试钉住：
打开 PLM 通道不该改变默认配置下的任何一条奖励数值。
"""

from __future__ import annotations

import math

import pytest

from pepdesign.rewards.predictors.esm import (
    ESM_MASK_TOKEN,
    PLM_BAD_LOGPROB,
    PLM_GOOD_LOGPROB,
    logprob_lookup_row,
    marginal_logprobs_to_score,
    masked_variants,
)
from pepdesign.rewards.predictors.factory import (
    DEFAULT_BACKENDS,
    PLM_BACKENDS,
    PREDICTOR_KINDS,
    build_predictor,
)
from pepdesign.utils.config import REWARD_PRESETS, RewardConfig, RewardWeights

SEQ = "KWKLFKKIEKVK"


# ---------------------------------------------------------------------------
# 掩码变体
# ---------------------------------------------------------------------------
def test_masked_variants_masks_exactly_one_position_each():
    """L 条变体，每条恰好掩掉一个位置，位置依次递增。

    用 `partition` 拆而不是字符下标：`<mask>` 是 6 个字符但只代表 1 个残基，
    所以 `v[i+1:]` 这种写法会错位（本测试的第一版就是这么错的）。
    """
    variants = masked_variants(SEQ)
    assert len(variants) == len(SEQ)
    for i, v in enumerate(variants):
        assert v.count(ESM_MASK_TOKEN) == 1
        left, _, right = v.partition(ESM_MASK_TOKEN)
        # 掩码左边和右边都必须与原文一致——只要错一位，
        # 回填的 log 概率就会整体错位，而分数照样是"看起来正常"的数。
        assert left == SEQ[:i]
        assert right == SEQ[i + 1:]
        # 字符串比原文长（`<mask>` 多 5 个字符），这是预期行为：
        # 分词之后每条变体的 token 数仍然相等，所以能批量前向。
        assert len(v) == len(SEQ) + len(ESM_MASK_TOKEN) - 1


def test_masked_variants_preserves_length_for_short_sequences():
    """长度为 1 的序列也要能处理（边界，真实语料里有 5 残基的短肽）。"""
    assert masked_variants("A") == [ESM_MASK_TOKEN]


def test_masked_variants_rejects_empty():
    with pytest.raises(ValueError, match="不能为空"):
        masked_variants("")


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------
def test_score_is_clipped_to_unit_interval():
    """归一化输出必须落在 [0,1]：它要和其余四维同尺度参与加权求和。"""
    good, _ = marginal_logprobs_to_score([PLM_GOOD_LOGPROB] * 10)
    bad, _ = marginal_logprobs_to_score([PLM_BAD_LOGPROB] * 10)
    assert good == pytest.approx(1.0)
    assert bad == pytest.approx(0.0)

    # 越界值被截断，而不是溢出
    over, _ = marginal_logprobs_to_score([0.0] * 5)
    under, _ = marginal_logprobs_to_score([-100.0] * 5)
    assert over == 1.0 and under == 0.0


def test_score_is_linear_inside_the_range_not_sigmoid():
    """区间内必须是线性的，不能是 sigmoid。

    线性保留了"好多少"的信息量。sigmoid 会把两端压平，
    结果是"明显更好"和"稍微更好"在组内的分差被压到接近 0——
    而 GRPO 的优势正是在组内分差上算的，分差没了梯度就没了。
    """
    mid = (PLM_GOOD_LOGPROB + PLM_BAD_LOGPROB) / 2
    quarter = (mid + PLM_BAD_LOGPROB) / 2

    s_mid, _ = marginal_logprobs_to_score([mid] * 4)
    s_quarter, _ = marginal_logprobs_to_score([quarter] * 4)

    # 中点恰好在 0.5（线性映射的直接推论；sigmoid 也是 0.5，所以要看下一条）
    assert s_mid == pytest.approx(0.5)
    # 四分之一处在 0.25，这正是线性的标志：sigmoid 在这里会给出
    # 明显偏离 0.25 的值（~0.27）
    assert s_quarter == pytest.approx(0.25, abs=1e-9)


def test_raw_logprob_is_returned_unscaled():
    """原始平均 log 概率要原样返回——它才是有绝对意义的量。

    归一化分数只在本项目的 good/bad 区间内可比；报告里要引用的是
    原始值（nat/残基），它跨实验、跨模型都有意义。
    把归一化后的数写进报告，换一次区间就对不上了。
    """
    _, raw = marginal_logprobs_to_score([-2.0, -3.0])
    assert raw == pytest.approx(-2.5)


def test_score_rejects_empty_and_nonfinite():
    with pytest.raises(ValueError, match=r"不能为空|无法打分"):
        marginal_logprobs_to_score([])
    with pytest.raises(ValueError, match="有限值"):
        marginal_logprobs_to_score([float("nan")])
    with pytest.raises(ValueError, match="good > bad"):
        marginal_logprobs_to_score([-2.0], good=-4.0, bad=-1.5)


def test_score_is_monotonic_in_logprob():
    """log 概率越高，分数必须越高（方向不能反）。"""
    scores = [
        marginal_logprobs_to_score([lp] * 3)[0]
        for lp in (-4.0, -3.5, -3.0, -2.5, -2.0, -1.5)
    ]
    assert scores == sorted(scores), f"分数不随 log 概率单调递增: {scores}"
    assert all(math.isfinite(s) for s in scores)


# ---------------------------------------------------------------------------
# token 回填
# ---------------------------------------------------------------------------
def test_logprob_lookup_row_rejects_wrong_token_count():
    """tokenizer 没有按单氨基酸分词时必须报错，不能默默错位。

    这是本模块最隐蔽的风险点：ESM 的 tokenizer 是按单残基分词的，
    所以"第 i 个字符 = 第 i+1 个 token"。如果哪天换了别的 tokenizer
    （或 ESM 改了分词方式），回填的 log 概率会整体错位一位——
    而错位之后分数仍然是个"看起来正常"的数，不会有任何报错。
    """

    class BadTokenizer:
        def convert_tokens_to_ids(self, tokens):
            # 假装它做了 BPE：把相邻两个氨基酸合成一个 token
            return [1] * (len(tokens) // 2)

    with pytest.raises(ValueError, match="没有按单氨基酸分词"):
        logprob_lookup_row(BadTokenizer(), SEQ)


def test_logprob_lookup_row_rejects_unknown_residues():
    """词表外的残基要报出来，而不是当成 0 号 token 去算概率。"""

    class SparseTokenizer:
        def convert_tokens_to_ids(self, tokens):
            return [1 if t in "ACDEFGHIKLMNPQRSTVWY" else None for t in tokens]

    with pytest.raises(ValueError, match="词表外"):
        logprob_lookup_row(SparseTokenizer(), "ACDX")


# ---------------------------------------------------------------------------
# 工厂注册表
# ---------------------------------------------------------------------------
def test_plm_is_registered_everywhere_the_other_dimensions_are():
    """加了第五维之后，`DEFAULT_BACKENDS` / `PREDICTOR_KINDS` 必须同步。

    这两张表是"配置里允许的后端名"的唯一来源。加编码时漏了它们，
    配置校验（用 Literal）会放过、`build_predictor` 才 KeyError——
    也就是"配置写得进去、跑到才炸"。
    """
    assert "plm" in PREDICTOR_KINDS
    assert "plm" in DEFAULT_BACKENDS
    assert DEFAULT_BACKENDS["plm"] == "off", (
        "默认后端必须是 off：默认配置下不能凭空多出一维"
    )
    assert PLM_BACKENDS == ("off", "esm")


def test_off_backend_constructs_and_is_harmless():
    """`off` 后端必须能构造（reward_audit 会遍历所有 kind 构造），
    且输出的置信度必须是 0——0 置信度 ⇒ 有效权重 0 ⇒ 不影响总奖励。

    如果它返回一个 0.5、置信度 1.0 的"中性分数"，那就等于在审计报告里
    塞了一个编出来的数，而读报告的人无从分辨。
    """
    p = build_predictor("plm", "off")
    res = p.predict_one(SEQ)
    assert res.confidence == 0.0
    assert res.detail is not None and "未启用" in str(res.detail)


def test_unknown_plm_backend_lists_valid_options():
    with pytest.raises(ValueError, match="未知的 plm 后端"):
        build_predictor("plm", "esm2")


# ---------------------------------------------------------------------------
# "默认不进奖励"这条契约
# ---------------------------------------------------------------------------
def test_plm_weight_defaults_to_zero_in_every_preset():
    assert RewardWeights().plm == 0.0
    for name, preset in REWARD_PRESETS.items():
        assert preset["plm"] == 0.0, f"预设 {name!r} 的 plm 权重不是 0"


def test_normalized_includes_plm_instead_of_silently_dropping_it():
    """`normalized()` 必须把 plm 算进分母。

    这里曾经是逐字段写死的实现（`thermo + protease + affinity + structure`
    再逐个重建对象）。加第五维时那种写法会把 plm 静默重置成类默认值
    并且不计入分母：配置里写 `plm: 0.15` 能通过所有校验、
    打印出来的字典里也看得到它，但归一化之后就变回 0 了。

    这条测试同时钉住两件事：plm 出现在结果里、且占比正确。
    """
    w = RewardWeights(thermo=0.30, protease=0.25, affinity=0.30,
                      structure=0.15, plm=0.15).normalized()
    assert w.plm == pytest.approx(0.15 / 1.15)
    assert sum(w.model_dump().values()) == pytest.approx(1.0)


def test_default_scorer_has_exactly_four_dimensions():
    """默认配置下维度集合必须不变。

    新增能力不该悄悄改变既有实验的结果：`plm_backend` 默认 `"off"`，
    所以默认的 `CompositeScorer` 还是那四维，总奖励和加这一维之前
    逐比特相同。
    """
    from pepdesign.rewards.composite import CompositeScorer

    scorer = CompositeScorer(RewardConfig())
    assert set(scorer.scorers) == {"thermo", "protease", "affinity", "structure"}
    b = scorer.score(SEQ)
    assert set(b.dimensions) == {"thermo", "protease", "affinity", "structure"}


def test_enabling_plm_as_audit_does_not_change_the_total():
    """`plm_backend="esm"` + 权重 0（默认）⇒ 多一维但总奖励不变。

    这是"审计模式"的定义：分数被算出来、被记录、进审计报告，
    但不参与加权求和。这条断言把"审计"和"训练"两种模式区分开——
    少了它，"开了 backend 但总奖励变了"这种 bug 不会被发现。
    """
    from pepdesign.rewards.composite import CompositeScorer
    from pepdesign.rewards.plm import PlmScorer

    # 用一个确定的假预测器，避免这条纯逻辑测试去下载真的 ESM 权重
    class _FixedPlm:
        name = "plm_fake"
        is_calibrated = False

        def predict_one(self, sequence):
            from pepdesign.rewards.predictors.base import PredictionResult

            return PredictionResult(value=0.9, confidence=1.0,
                                    detail={"mean_logprob": -1.6}, predictor=self.name)

    base_dims = ["thermo", "protease", "affinity", "structure"]
    scorers = [s for n, s in CompositeScorer(RewardConfig()).scorers.items() if n in base_dims]
    without = CompositeScorer(RewardConfig(), scorers=scorers)
    with_plm = CompositeScorer(
        RewardConfig(), scorers=[*scorers, PlmScorer(_FixedPlm())]   # type: ignore[arg-type]
    )

    b1 = without.score(SEQ)
    b2 = with_plm.score(SEQ)
    assert "plm" in b2.dimensions
    assert b2.dimensions["plm"].value == pytest.approx(0.9)
    # 权重 0 ⇒ 贡献 0 ⇒ 总分不变
    assert b2.dimensions["plm"].weight == 0.0
    assert b2.total == pytest.approx(b1.total), (
        "plm 权重为 0 时总奖励不该改变——审计模式不能影响训练信号"
    )


def test_plm_with_nonzero_weight_does_change_the_total():
    """反向：把 plm 权重调成非零，它就必须真正参与加权求和。

    只测"权重 0 时不影响总分"是不够的——那用一个永远返回 0 权重的
    实现也能过。必须同时证明非零权重真的接上了。
    """
    from pepdesign.rewards.composite import CompositeScorer
    from pepdesign.rewards.plm import PlmScorer

    class _FixedPlm:
        name = "plm_fake"
        is_calibrated = True   # 不打折，便于断言

        def predict_one(self, sequence):
            from pepdesign.rewards.predictors.base import PredictionResult

            return PredictionResult(value=1.0, confidence=1.0,
                                    detail={"mean_logprob": -1.5}, predictor=self.name)

    cfg = RewardConfig(weights=RewardWeights(plm=0.5))
    dims = ["thermo", "protease", "affinity", "structure"]
    scorers = [s for n, s in CompositeScorer(RewardConfig()).scorers.items() if n in dims]
    scorer = CompositeScorer(cfg, scorers=[*scorers, PlmScorer(_FixedPlm())])  # type: ignore[arg-type]

    b = scorer.score(SEQ)
    d = b.dimensions["plm"]
    assert d.weight > 0.0, "非零权重没有生效"
    assert d.weighted > 0.0
