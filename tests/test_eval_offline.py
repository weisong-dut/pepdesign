"""离线评估层：指标、显著性、以及"什么时候不该下结论"。

这个文件能在只装 core 依赖的环境里跑——`eval/` 刻意只用 numpy
（不碰 scipy/sklearn/pandas/torch），所以它属于第一层验证，
不需要 `-m smoke`。

其中一半的测试是在验证保护机制而不是数学：本项目只有 6 条带实测值
的记录，而在这种规模上最危险的不是"算错"，是算出一个看着很确定的数。
所以 `test_report_refuses_to_conclude_on_small_n` 这类断言的优先级
高于任何一个相关系数的正确性。
"""

from __future__ import annotations

import math

import pytest

from pepdesign.eval.offline import (
    MIN_N_FOR_SIGNIFICANCE,
    average_ranks,
    correlation_report,
    count_ties,
    mae,
    orientation_for,
    pearson_r,
    permutation_p_value,
    rmse,
    spearman_ic,
    top_k_enrichment,
)
from pepdesign.eval.reward_audit import audit_reward_predictors, summarize_audit


# ---------------------------------------------------------------------------
# 方向：不能猜
# ---------------------------------------------------------------------------
def test_orientation_is_explicit_for_known_assays():
    """MIC/IC50 越小越好，Tm 越大越好。两者方向相反，这是全部相关性的地基。"""
    assert orientation_for("MIC") == -1
    assert orientation_for("ic50") == -1
    assert orientation_for("Tm") == +1
    # 大小写、连字符、下划线都不该影响查表
    assert orientation_for("IC-50") == -1
    assert orientation_for("I_C_5_0") == -1
    assert orientation_for(" Tm ") == +1


def test_unknown_assay_raises_instead_of_guessing():
    """未知测定类型必须报错。

    猜错方向的后果是所有相关系数的符号整体反转，而反转后的数字
    看上去完全正常（-0.7 也像是个有意义的数），只是结论全反。
    这是"宁可停下来问，也不要给一个可能的答案"的典型场景。
    """
    with pytest.raises(KeyError, match="未知的测定类型"):
        orientation_for("some_new_assay")
    # 报错信息要带着已知类型，不然使用者不知道去哪加
    with pytest.raises(KeyError, match="ASSAY_ORIENTATION"):
        orientation_for("some_new_assay")


# ---------------------------------------------------------------------------
# 秩与相关系数
# ---------------------------------------------------------------------------
def test_average_ranks_handles_ties_and_is_order_independent():
    """并列取平均秩。

    不处理并列（比如按出现顺序分配 0,1,2）会让相关系数取决于
    数据在文件里的行序——同一份数据换个行序得出不同结论。
    本项目的实测数据里就有两条 Tm 都是 64.6。
    """
    assert list(average_ranks([10.0, 20.0, 30.0])) == [0.0, 1.0, 2.0]
    # 前两个并列 → 共享平均秩 0.5
    assert list(average_ranks([5.0, 5.0, 9.0])) == [0.5, 0.5, 2.0]
    # 三个并列 → 共享平均秩 1.0
    assert list(average_ranks([5.0, 5.0, 5.0])) == [1.0, 1.0, 1.0]
    # 打乱顺序后，每个元素拿到的秩不变
    vals = [3.0, 1.0, 4.0, 1.0, 5.0]
    base = list(average_ranks(vals))
    assert list(average_ranks(list(reversed(vals)))) == list(reversed(base))


def test_spearman_bounds_and_direction():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert spearman_ic(x, x) == pytest.approx(1.0)
    assert spearman_ic(x, list(reversed(x))) == pytest.approx(-1.0)
    # 单调但非线性——这正是用秩相关而不是 Pearson 的理由
    assert spearman_ic(x, [1.0, 4.0, 9.0, 16.0, 25.0]) == pytest.approx(1.0)
    assert pearson_r(x, [1.0, 4.0, 9.0, 16.0, 25.0]) < 0.99


def test_spearman_is_zero_not_nan_when_there_is_no_signal():
    """预测值全相同时返回 0，不是 nan。

    nan 会一路传播成报告里的 `NaN`，而"这批预测没有任何区分度"
    本身就是个需要被看见的结论，不该以缺失值的形式出现。
    """
    assert spearman_ic([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]) == 0.0
    assert spearman_ic([1.0], [1.0]) == 0.0
    assert not math.isnan(spearman_ic([1.0, 1.0], [1.0, 1.0]))


def test_length_mismatch_raises_instead_of_truncating():
    """长度不等必须报错。

    截断到较短的一侧会把"预测值与实测值错位"变成"少几个样本"，
    而错位的相关性照样能算出 -0.9 这种很有说服力的数。
    """
    with pytest.raises(ValueError, match="一一对应"):
        spearman_ic([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="一一对应"):
        mae([1.0], [1.0, 2.0])


def test_error_metrics_use_measured_units():
    assert mae([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]) == pytest.approx(1.0)
    assert rmse([1.0, 3.0], [1.0, 1.0]) == pytest.approx(math.sqrt(2.0))


def test_top_k_enrichment_both_directions():
    """Top-K 富集：`higher_is_better` 要真的翻方向。

    "整体相关性不错但前 K 条全是废的"是真实存在的失败模式，
    而它和"单纯不相关"的处置完全不同。
    """
    scores = [0.1, 0.9, 0.2, 0.8, 0.3]
    labels = [0.0, 1.0, 0.0, 0.0, 1.0]
    top2 = top_k_enrichment(scores, labels, k=2, higher_is_better=True)
    assert top2["hits"] == 1            # 取 0.9 和 0.8，只有 0.9 是好
    assert top2["precision"] == pytest.approx(0.5)
    assert top2["recall"] == pytest.approx(0.5)   # 2 个正例里命中 1 个

    # 翻方向后取到的是最小的两个（0.1 和 0.2），它们的 label 都是 0：
    # 同一批数据、同一个 K，方向搞反就得到完全相反的结论——
    # 这就是"方向必须显式查表、不许猜"的价格。
    low2 = top_k_enrichment(scores, labels, k=2, higher_is_better=False)
    assert low2["hits"] == 0
    assert low2["precision"] == 0.0
    assert low2["k"] == 2
    assert top2["hits"] != low2["hits"]
    # k 超过样本数时收敛到全部，而不是报错
    assert top_k_enrichment(scores, labels, k=99)["k"] == len(scores)


# ---------------------------------------------------------------------------
# 置换检验
# ---------------------------------------------------------------------------
def test_permutation_p_is_small_for_a_perfect_monotone_signal():
    """完全单调时 p 必须很小——否则检验本身就是坏的。"""
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    p = permutation_p_value(y, y, n_perm=2000, seed=0)
    assert p < 0.01


def test_permutation_p_is_not_significant_for_a_random_signal():
    """和一个与 y 无关的固定序列比，p 应该大。"""
    y = [5.0, 1.0, 4.0, 2.0, 8.0, 3.0, 7.0, 6.0]
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    p = permutation_p_value(x, y, n_perm=2000, seed=0)
    assert p > 0.05


def test_permutation_p_never_reports_zero():
    """用 (count+1)/(n_perm+1) 而不是 count/n_perm。

    后者可能给出恰好 0，而 0 会被读成"不可能是巧合"。
    置换检验能分辨的最小 p 值是 1/(n_perm+1)，加一才是诚实的写法。
    """
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    p = permutation_p_value(y, y, n_perm=500, seed=0)
    assert p > 0.0
    assert p >= 1.0 / 501.0


def test_permutation_p_is_reproducible_given_the_seed():
    """同一种子必须给同一个 p。

    置换检验是随机的，不固定种子的话同一份数据跑两次会有不同的 p 值，
    报告就无法被引用——而"报告可引用"正是这层存在的意义。
    """
    x = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.6, 5.3]
    y = [2.0, 7.0, 1.0, 8.0, 2.0, 9.0, 3.0, 6.0]
    assert permutation_p_value(x, y, n_perm=500, seed=7) == \
        permutation_p_value(x, y, n_perm=500, seed=7)


def test_count_ties_counts_repeated_observations():
    assert count_ties([1.0, 2.0, 3.0]) == 0
    assert count_ties([64.6, 64.6, 55.8]) == 2   # 真实数据里的情况
    assert count_ties([]) == 0


# ---------------------------------------------------------------------------
# 报告：这部分是本模块的重点
# ---------------------------------------------------------------------------
def test_report_refuses_to_conclude_on_small_n():
    """n 不足时 `conclusive=False`，且要说明为什么。

    这是本层最重要的一条约定。n=6 且完全单调时，置换检验给的 p 值
    可以很小（≈0.003），光看 p 值会以为有结论——但那个 p 值
    来自 6 条标记着 placeholder 的数据。所以报告必须把样本量
    摆在与 p 值同等显眼的位置，并把 caveat 写成人能读的句子。
    """
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    rep = correlation_report(y, y, n_perm=500)

    assert rep["n"] == 6
    assert rep["spearman"] == pytest.approx(1.0)
    assert rep["conclusive"] is False
    assert rep["p_value"] < 0.05, "这个用例的意义就在于 p 很小但依然不下结论"
    assert any("样本量" in c for c in rep["caveats"])
    assert rep["min_n_for_significance"] == MIN_N_FOR_SIGNIFICANCE


def test_report_concludes_once_n_is_large_enough():
    """n 够了就必须敢下结论——否则这条保护会退化成"永远不下结论"。"""
    y = [float(i) for i in range(30)]
    rep = correlation_report(y, y, min_n=20, n_perm=500)
    assert rep["conclusive"] is True
    assert not any("样本量" in c for c in rep["caveats"])


def test_report_mentions_ties_as_a_caveat():
    """并列要单独报出来：它会同时压低秩相关的上限、抬高 p 值。"""
    measured = [64.6, 64.6, 64.6, 55.8, 53.4, 45.4]
    pred = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    rep = correlation_report(pred, measured, n_perm=200)

    assert rep["ties_in_measured"] == 3
    assert any("并列" in c for c in rep["caveats"])


def test_report_flags_constant_measurements():
    """实测值全相同时，相关性无定义，要明说而不是给一个数。"""
    rep = correlation_report([1.0, 2.0, 3.0], [5.0, 5.0, 5.0], n_perm=100)
    assert rep["spearman"] == 0.0
    assert any("全部相同" in c for c in rep["caveats"])


# ---------------------------------------------------------------------------
# 奖励审计：接到真实数据的形状上
# ---------------------------------------------------------------------------
def _record(seq: str, **assays: float) -> dict:
    return {
        "sequence": seq,
        "activities": [
            {"assay_type": k, "value": v, "source_ref": "unit-test"}
            for k, v in assays.items()
        ],
    }


def test_audit_applies_orientation_to_measured_values():
    """MIC 越小越好，所以进相关系数之前要取负。

    这条不生效的话，MIC 那一路的相关系数符号会整体反转，
    而 -0.3 和 +0.3 看上去都是"合理"的数。
    """
    # 两条记录：预测端只需检查符号是否被翻转，
    # 所以直接用 MIC 的两个值构造，检查 audit 内部把它取负。
    from pepdesign.eval.reward_audit import _measured_by_assay

    oriented = _measured_by_assay(_record("AAAA", MIC=10.0))
    assert oriented["MIC"] == -10.0     # 越小越好 → 取负后"越大越好"
    assert _measured_by_assay(_record("AAAA", Tm=60.0))["Tm"] == 60.0

    # 同一测定出现多次时取"最好的那条"（统一符号后取 max）
    assert _measured_by_assay(_record("AAAA", MIC=10.0))["MIC"] == -10.0
    rec = {"sequence": "AAAA", "activities": [
        {"assay_type": "MIC", "value": 10.0},
        {"assay_type": "MIC", "value": 30.0},
    ]}
    assert _measured_by_assay(rec)["MIC"] == -10.0


def test_audit_reports_structure_and_missing_dimensions():
    """报告的形状：有数据的维度给出统计量，没有的明说没有。"""
    records = [_record(f"GIGAVLKVLTTGLPALISWIKRKRQ{i}", Tm=50.0 + i, MIC=20.0 - i)
               for i in range(4)]
    rep = audit_reward_predictors(records, n_perm=200)

    assert rep["records_total"] == 4
    assert rep["records_with_activities"] == 4
    assert set(rep["dimensions"]) == {"thermo", "affinity"}
    for entry in rep["dimensions"].values():
        assert entry["status"] == "ok"
        assert entry["n"] == 4
        assert entry["conclusive"] is False          # 4 < 20，必须拒答
        assert entry["predictor"] and entry["backend"]

    # 没有实测数据对应的维度不能被悄悄省略。
    # `plm` 也在这一组里，但它"没有实测对应"的原因和另外两个不同：
    # protease/structure 是本仓库缺数据，plm 是那件事本身没有可测定的物理量。
    # 报告里必须能区分这两种"没做"，所以 plm 有一条单独的理由文案。
    assert set(rep["not_evaluated"]) == {"protease", "structure", "plm"}
    assert "不是'缺数据'" in rep["not_evaluated"]["plm"]
    assert rep["conclusive_dimensions"] == []


def test_audit_skips_records_without_activities():
    records = [_record("GIGAVLKVLTTGLPALISWIKRKRQQ", Tm=55.0), {"sequence": "AAAA"}]
    rep = audit_reward_predictors(records, n_perm=200)
    assert rep["records_total"] == 2
    assert rep["records_with_activities"] == 1


def test_audit_dimension_with_no_data_is_marked_not_silently_dropped():
    """一条带实测数据的记录都没有时，维度仍要出现在报告里。"""
    rep = audit_reward_predictors([{"sequence": "AAAA"}], n_perm=100)
    assert rep["records_with_activities"] == 0
    assert rep["dimensions"]["thermo"]["status"] == "no_measured_data"
    assert rep["dimensions"]["thermo"]["n"] == 0


def test_summarize_audit_is_plain_text_and_says_when_it_cannot_conclude():
    records = [_record(f"GIGAVLKVLTTGLPALISWIKRKRQ{i}", Tm=50.0 + i) for i in range(3)]
    rep = audit_reward_predictors(records, n_perm=100)
    text = summarize_audit(rep)

    assert "样本不足" in text
    assert "未评估" in text
    # 输出的文本必须能被 Windows 的 GBK 控制台编码：
    # 源码按 UTF-8 读，只有真正 print 出去的字符会经过控制台编码，
    # 这里曾出现 UnicodeEncodeError（该字符不属于 GBK）。
    text.encode("gbk")
