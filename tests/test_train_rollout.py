"""rollout 组装：解析 → 打分 → 分组 → 算优势。

这一层的输入是已经解码好的文本，输出是浮点数，中间没有任何 torch，
因此可以在不依赖 GPU 的环境中完整断言。下面每一条测试都对应一个
"不会崩、只会悄悄训坏"的场景：

  * 组边界切错（第 1 组的序列和第 2 组的序列被拿来互相比较）；
  * meta 展开顺序错（优势串到别的 prompt 上）；
  * 解析失败的序列混进组内基线（拉低均值 → 同伴优势虚高 → 奖励漏洞）；
  * 就地改调用方的 meta（上一步的失败泄漏到下一步）。

本文件不 import torch（conftest.py 的硬约束）。
"""

from __future__ import annotations

import pytest

from pepdesign.prompts.templates import render_completion
from pepdesign.rewards.base import DimensionScore, RewardBreakdown
from pepdesign.train.rollout import (
    WEAK_PARSE_STRATEGIES,
    RolloutBatch,
    RolloutGroup,
    assemble_rollouts,
    expand_metas,
    parse_completions,
)

# ---------------------------------------------------------------------------
# 测试替身：把奖励分布控制到确定的数值上
# ---------------------------------------------------------------------------


class FakeScorer:
    """按序列内容查表给分，用于精确控制组内的奖励分布。

    用替身而不是真的 `CompositeScorer`，是因为真的那个算出来的分数
    取决于生物物理计算，想构造"组内奖励全相同"或"组内奖励严格递增"
    这两种边界情形会非常别扭——而这两种恰恰是最需要测的。
    真打分器与这里的衔接由文件末尾的集成测试覆盖。
    """

    def __init__(self, table: dict[str, float], invalid: set[str] | None = None):
        self.table = table
        self.invalid = invalid or set()
        self.score_group_calls: list[tuple[list[str], list[dict]]] = []

    def score_group(self, sequences, metas):
        self.score_group_calls.append((list(sequences), [dict(m) for m in metas]))
        out = []
        for seq, meta in zip(sequences, metas, strict=True):
            reason = meta.get("invalid_reason")
            if reason is None and seq in self.invalid:
                reason = "test_marked_invalid"
            out.append(
                RewardBreakdown(
                    sequence=seq,
                    total=self.table.get(seq, 0.0),
                    dimensions={},
                    penalties={},
                    invalid_reason=reason,
                )
            )
        return out

    def group_advantages(self, scores, mode="std"):
        """与 `CompositeScorer.group_advantages` 同语义的最小实现。

        只用来验证 `assemble_rollouts` 把 `mode` 和分组后的分数
        正确传了下来。分组逻辑本身的正确性由 test_grpo_advantage.py 覆盖，
        这里不重复测它。
        """
        valid = [s.total for s in scores if s.invalid_reason is None]
        out = [0.0] * len(scores)
        if not valid:
            return out
        mean = sum(valid) / len(valid)
        if mode == "none":
            denom = 1.0
        else:
            var = sum((v - mean) ** 2 for v in valid) / len(valid)
            std = var ** 0.5
            if std < 1e-6:
                return out
            denom = std
        return [
            0.0 if s.invalid_reason is not None else (s.total - mean) / denom
            for s in scores
        ]


def completion(seq: str, strategy: str = "改造策略：P1'位换 Pro") -> str:
    """造一段格式合法的生成结果（含 ### 标记，解析器认这个格式）。"""
    return render_completion(strategy, seq)


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


def test_parse_completions_marks_failures_without_touching_caller_meta():
    """解析失败的序列被判无效，且不修改调用方传进来的 meta。

    就地写 `invalid_reason` 是这里最容易犯的错：`metas` 通常是上层从数据集
    取出来、下一个 step 还要复用的对象。就地写会让"上一步的解析失败"
    泄漏到下一步——下一步明明解析成功了，却因为残留字段被判无效。
    表现是"loss 偶尔莫名卡住一步"，极难定位（因为重跑往往就不复现了，
    取决于上一步的采样结果）。
    """
    metas = [{"task": "stabilize", "base_sequence": "KWKLFKKIEK"}]
    snapshot = dict(metas[0])

    parsed, scoring_metas = parse_completions(
        ["这段文本里没有任何序列标记"], metas
    )

    assert parsed[0].ok is False
    assert scoring_metas[0]["invalid_reason"].startswith("parse_failed:")
    # 原 dict 必须原封不动
    assert metas[0] == snapshot
    assert "invalid_reason" not in metas[0]
    # 是一个新对象，不是同一个引用
    assert scoring_metas[0] is not metas[0]


def test_parse_completions_keeps_meta_fields_on_success():
    """成功解析时，meta 的业务字段（base_sequence / objectives / task）
    必须原样带到打分器那里。

    丢了 `base_sequence` 的后果在 `data/build_grpo.py` 的开头用一整段解释过：
    相似度约束静默失效，模型把"稳定性优化"做成了"重新设计"——
    奖励还很高，因为全新设计的序列天然避开了原序列的酶切位点。
    """
    metas = [{"task": "stabilize", "base_sequence": "KWKLFKKIEK", "objectives": {"thermo": 1.0}}]
    parsed, scoring_metas = parse_completions([completion("KWKLFKKIEKP")], metas)

    assert parsed[0].ok is True
    assert parsed[0].sequence == "KWKLFKKIEKP"
    assert "invalid_reason" not in scoring_metas[0]
    assert scoring_metas[0]["base_sequence"] == "KWKLFKKIEK"
    assert scoring_metas[0]["objectives"] == {"thermo": 1.0}


def test_parse_completions_cyclic_expectation_from_meta():
    """`cyclic_type` 决定解析器是否按环肽来断言长度。

    这个字段只有 de_novo 任务的 meta 才带，其余任务用 "none" 兜底——
    兜底写错（比如用 `meta["cyclic_type"]` 直接取值）会在非 de_novo 任务上
    抛 KeyError，而那是在训练循环里，报错信息会被采样日志淹没。
    """
    metas = [{"task": "de_novo", "cyclic_type": "head_to_tail"}]
    # 不带这个字段也要能跑（其余四类任务就是这样）
    parsed, _ = parse_completions([completion("KWKLFKKIEK")], [{"task": "stabilize"}])
    assert parsed[0].ok is True

    parsed, _ = parse_completions([completion("KWKLFKKIEK")], metas)
    assert parsed[0].ok is True


def test_parse_completions_rejects_length_mismatch():
    with pytest.raises(ValueError, match="不一致"):
        parse_completions(["a", "b"], [{"task": "x"}])


# ---------------------------------------------------------------------------
# meta 展开顺序
# ---------------------------------------------------------------------------


def test_expand_metas_is_prompt_major():
    """`[m0, m0, m1, m1]` 而不是 `[m0, m1, m0, m1]`。

    这两种顺序在 G=2 且 prompt 数为偶数时长度一样、元素一样、
    只是位置全错，而且不会报错——它会让每个 prompt 的序列拿着别的
    prompt 的 meta 去打分。选择 G=2 做这个测试是刻意的：
    它是能让 `metas * group_size` 这个错误写法刚好蒙混过关的最小参数。
    """
    metas = [{"i": 0}, {"i": 1}]
    assert expand_metas(metas, 2) == [{"i": 0}, {"i": 0}, {"i": 1}, {"i": 1}]
    assert expand_metas(metas, 3) == [{"i": 0}] * 3 + [{"i": 1}] * 3
    assert expand_metas([], 4) == []


# ---------------------------------------------------------------------------
# assemble_rollouts：分组、计数、优势
# ---------------------------------------------------------------------------


def test_assemble_groups_by_prompt_not_by_batch():
    """优势必须在组内算，不能整批一起算。

    这是 GRPO 与"把 GRPO 当成带 baseline 的 REINFORCE"的分界线：
    整批算等于把所有 prompt 的样本混进同一个基线，第 1 组的序列
    会因为"碰巧和更强的对手分在一批"而拿到负优势。
    这里用两组奖励量级相差很大的序列来验：如果优势是整批算的，
    第 1 组的好序列会拿到负优势。
    """
    texts = [
        completion("AAAAA"), completion("CCCCC"),          # prompt 0
        completion("DDDDD"), completion("EEEEE"),          # prompt 1
    ]
    metas = [{"task": "t0"}, {"task": "t1"}]
    scorer = FakeScorer({"AAAAA": 0.9, "CCCCC": 0.1, "DDDDD": 0.2, "EEEEE": 0.1})

    batch = assemble_rollouts(texts, metas, group_size=2, scorer=scorer)

    assert len(batch.groups) == 2
    g0, g1 = batch.groups
    assert g0.texts == texts[:2]
    assert g1.texts == texts[2:]
    assert g0.meta["task"] == "t0"
    assert g1.meta["task"] == "t1"

    # 组 0 内 AAAAA(0.9) > CCCCC(0.1) → 优势一正一负
    assert g0.advantages[0] > 0 > g0.advantages[1]
    # 组 1 内 DDDDD(0.2) > EEEEE(0.1) → 同样一正一负。
    # 如果优势是整批算的，DDDDD 的 0.2 低于整批均值，会拿到负优势。
    assert g1.advantages[0] > 0 > g1.advantages[1]

    # 扁平的 advantages 顺序必须与 texts 一致（prompt-major）
    assert batch.advantages == g0.advantages + g1.advantages
    assert len(batch) == 4


def test_assemble_passes_mode_through_to_group_advantages():
    """`advantage_mode` 要真的传到组内优势那一步。

    传丢了的后果：配置里写着 `none`（不做标准化），实际跑的是 `std`——
    梯度量级完全变了，而日志上唯一能看出差别的地方是 reward std，
    那一列很容易被当成噪声。
    """
    texts = [completion("AAAAA"), completion("CCCCC")]
    metas = [{"task": "t"}]
    scorer = FakeScorer({"AAAAA": 0.9, "CCCCC": 0.1})

    std = assemble_rollouts(texts, metas, group_size=2, scorer=scorer, advantage_mode="std")
    none = assemble_rollouts(texts, metas, group_size=2, scorer=scorer, advantage_mode="none")

    # std 口径下优势被标准化到 ±1 附近；none 口径下就是"偏离组内均值"
    assert std.advantages[0] == pytest.approx(1.0)
    assert std.advantages[1] == pytest.approx(-1.0)
    assert none.advantages[0] == pytest.approx(0.4)      # 0.9 - 0.5
    assert none.advantages[1] == pytest.approx(-0.4)     # 0.1 - 0.5


def test_assemble_excludes_invalid_from_group_baseline():
    """解析失败的序列不参与组内均值/标准差，但仍然占位。

    把它算进均值会系统性拉低基线，让正常序列看起来"更好"——
    等于在鼓励模型输出一些解析不了的东西来抬高同伴的优势。
    这类奖励漏洞在 RL 里一定会被找到（模型不需要理解它，
    只需要反复采样到那个方向）。
    """
    texts = [
        completion("AAAAA"),                 # 有效，0.8
        "这段文本解析不出序列",                 # 无效
        completion("CCCCC"),                 # 有效，0.2
    ]
    metas = [{"task": "t"}]
    scorer = FakeScorer({"AAAAA": 0.8, "CCCCC": 0.2})

    batch = assemble_rollouts(texts, metas, group_size=3, scorer=scorer)
    g = batch.groups[0]

    # 两个有效分数的均值是 0.5，标准差 0.3
    assert g.mean_reward() == pytest.approx(0.5)
    assert g.reward_std() == pytest.approx(0.3)
    # 无效那条占位为 0，不参与
    assert g.advantages[1] == 0.0
    assert g.advantages[0] == pytest.approx(1.0)
    assert g.advantages[2] == pytest.approx(-1.0)
    assert g.parse_failures == 1
    assert batch.reward_report["invalid"] == 1


def test_assemble_all_invalid_group_gives_zero_advantages():
    """整组都解析失败时返回全零优势，而不是 nan。"""
    texts = ["解析失败一", "解析失败二"]
    batch = assemble_rollouts(texts, [{"task": "t"}], group_size=2, scorer=FakeScorer({}))

    assert batch.advantages == [0.0, 0.0]
    assert batch.groups[0].mean_reward() == 0.0
    assert batch.groups[0].reward_std() == 0.0
    assert batch.usable_ratio == 0.0
    assert batch.reward_report["degenerate_groups"] == 1


def test_assemble_degenerate_group_counted():
    """组内奖励完全相同 → 优势全零，且被计进 degenerate_groups。

    这不是超参问题，是"这一步没有可学的信息"。它必须是一个可见的数字：
    常见的错误处理是让 std 变成 0 然后优势变 nan，整个 batch 的梯度被污染，
    而日志上只表现为"loss 突然变成 nan"。
    """
    texts = [completion("AAAAA"), completion("CCCCC")]
    scorer = FakeScorer({"AAAAA": 0.5, "CCCCC": 0.5})
    batch = assemble_rollouts(texts, [{"task": "t"}], group_size=2, scorer=scorer)

    assert batch.advantages == [0.0, 0.0]
    assert batch.reward_report["degenerate_groups"] == 1
    assert batch.reward_report["mean"] == pytest.approx(0.5)


def test_assemble_usable_ratio_counts_only_valid():
    texts = [
        completion("AAAAA"), completion("CCCCC"),   # 组 0：全有效
        completion("DDDDD"), "解析失败",              # 组 1：一半有效
    ]
    scorer = FakeScorer({"AAAAA": 0.6, "CCCCC": 0.4, "DDDDD": 0.5})
    batch = assemble_rollouts(texts, [{"task": "a"}, {"task": "b"}], group_size=2, scorer=scorer)

    assert batch.usable_ratio == 0.75
    assert batch.parse_report["total"] == 4


def test_assemble_rejects_wrong_sample_count_and_tiny_group():
    """长度校验与 G>=2 校验。

    G=1 时必须直接报错：组内没有可比对象，优势恒为 0，
    训练不会有任何梯度。让它跑起来只会浪费几小时算力，
    而且日志上看起来和"奖励设计得不好"一模一样。
    """
    texts = [completion("AAAAA")] * 3
    scorer = FakeScorer({"AAAAA": 1.0})

    with pytest.raises(ValueError, match="group_size"):
        assemble_rollouts(texts, [{"task": "t"}], group_size=1, scorer=scorer)
    with pytest.raises(ValueError, match="group_size"):
        assemble_rollouts(texts, [{"task": "t"}], group_size=2, scorer=scorer)


def test_assemble_scores_the_same_order_it_groups():
    """打分器看到的序列顺序必须和分组顺序一致。

    这是一个"错了也不会报错"的接缝：`score_group` 返回的列表被按
    `[gi*G:(gi+1)*G]` 切开。如果两者顺序不一致（比如打分器内部
    按别的规则排过序），第 1 组的分数会配到第 2 组的序列上，
    优势跟着错位。断言"打分器收到的序列就是传进去的那些、顺序不变"
    是能查到这一点的唯一方式。
    """
    texts = [completion(s) for s in ("AAAAA", "CCCCC", "DDDDD", "EEEEE")]
    metas = [{"task": "t0"}, {"task": "t1"}]
    scorer = FakeScorer(dict.fromkeys(("AAAAA", "CCCCC", "DDDDD", "EEEEE"), 0.5))

    assemble_rollouts(texts, metas, group_size=2, scorer=scorer)

    assert len(scorer.score_group_calls) == 1
    seen, seen_metas = scorer.score_group_calls[0]
    # 打分器拿到的是解析后的序列（"AAAAA"），不是原始生成文本
    # （"### 设计序列 / AAAAA / ### ..."）。后者含非氨基酸字符，
    # 会被 validate 判成非法——整批静默失效。
    assert seen == ["AAAAA", "CCCCC", "DDDDD", "EEEEE"]
    assert [m["task"] for m in seen_metas] == ["t0", "t0", "t1", "t1"]


# ---------------------------------------------------------------------------
# 与真实打分器的衔接
# ---------------------------------------------------------------------------


def test_assemble_with_real_scorer_keeps_meta_contract():
    """换成真的 `CompositeScorer` 也要能跑通，且 meta 真的被用上。

    这条测试的价值在于它是接口契约的检查：
    `assemble_rollouts` 往打分器里塞的 meta 字段名，必须和
    `CompositeScorer.score` 真正读的那些对得上。替身测试永远发现不了
    字段名写错——替身只会读它自己关心的字段。
    """
    from pepdesign.rewards.composite import CompositeScorer
    from pepdesign.utils.config import RewardConfig

    scorer = CompositeScorer(RewardConfig())
    metas = [{"task": "stabilize", "base_sequence": "KWKLFKKIEK"}]
    # 两条差异明显的序列：一条改了 P1' 位，一条是纯疏水堆叠
    texts = [completion("KWKLFKKIEKP"), completion("WWWWWWWWWW")]

    batch = assemble_rollouts(texts, metas, group_size=2, scorer=scorer)

    assert len(batch.groups) == 1
    g = batch.groups[0]
    assert len(g.scores) == 2
    assert g.scores[0].sequence == "KWKLFKKIEKP"
    # 真实打分器给每条都填了分维度
    assert g.scores[0].dimensions
    # 优势要么有效要么全零，但不能是 nan（退化组是很可能发生的，
    # 因为两条随机序列的加权总分可能恰好相等）
    for a in g.advantages:
        assert a == a, "优势出现了 nan"
        assert abs(a) < 1e6


# ---------------------------------------------------------------------------
# 结果结构的直接测试
# ---------------------------------------------------------------------------


def test_rollout_batch_summary_shape():
    batch = RolloutBatch(groups=[])
    assert len(batch) == 0
    assert batch.usable_ratio == 0.0
    summary = batch.summary()
    assert summary["groups"] == 0 and summary["samples"] == 0


def test_rollout_group_std_needs_two_valid_scores():
    """只有 1 条有效分数时 std 定义为 0（样本量不足），不是除以 0。"""
    g = RolloutGroup(prompt_index=0, meta={})
    assert g.reward_std() == 0.0
    g.scores = [RewardBreakdown(sequence="A", total=1.0, dimensions={}, penalties={})]
    assert g.reward_std() == 0.0


# ---------------------------------------------------------------------------
# 弱解析（line_scan / whole_text）在 RL 里必须按失败处理
#
# 对应一个真实观察到的现象：把随机初始化的模型接到 GRPO 循环上跑，
# 它吐出来的乱码里 `IOCTL`、`CONTENT`、`DISCRIMINATORBURNING` 都能被
# `line_scan` 抠成"多肽"——字母表上 20 个标准氨基酸码加上 B/Z/X/U/O
# 这些模糊码，几乎覆盖了英文大写字母。第三个有 19 个残基，
# 正好落在 12-30 的合法长度区间里，长度校验也拦不住。
#
# 对评测来说这没问题（给定一份固定输出，尽力抠出来），
# 但在 RL 里它会变成奖励、进而变成梯度，于是"写不写 ### 标记"这件事
# 被反向优化掉了。下面的断言把这条界线钉死。
# ---------------------------------------------------------------------------

#: 一个能被启发式抠成 19 残基"多肽"的英文词。
#: 用真实观察到的那一个，而不是编一个刚好越界的：长度 19 在合法区间内，
#: 正是这条边界最容易被忽略的地方。
WEAK_TEXT = "DISCRIMINATORBURNING"


def test_the_weak_sample_is_actually_a_weak_parse():
    """先证明前提：解析器确实会把这个英文词抠成序列。

    如果哪天解析器收紧了启发式、不再抠它，下面几条测试就会变成
    "因为没解析出来所以被判无效"——碰巧还是绿的，但验证的东西已经完全变了。
    所以这条断言先在这里把前提钉住，让那种情况显式失败。
    """
    parsed, _ = parse_completions([WEAK_TEXT], [{"task": "de_novo"}])
    assert parsed[0].ok is True
    assert parsed[0].strategy in WEAK_PARSE_STRATEGIES, (
        f"预期弱解析，实际 strategy={parsed[0].strategy!r}；"
        f"解析器行为变了，这组测试的前提不再成立"
    )
    assert parsed[0].sequence == WEAK_TEXT
    assert 12 <= len(parsed[0].sequence) <= 30, "它得落在合法长度区间里才有说服力"


def test_weak_parse_is_invalid_by_default_in_rl():
    """默认（RL 口径）下，弱解析拿不到分数、也不产生优势。

    "拿不到分数"用的是和硬解析失败同一套机制（`invalid_reason`），
    而不是"给个低分"。两者的差别很关键：低分会进组内均值，
    拉低基线、让同组其他序列的优势凭空变大；无效则被排除在外。
    """
    parsed, metas = parse_completions([WEAK_TEXT], [{"task": "de_novo"}])
    assert parsed[0].ok is True, "解析结果本身仍然是成功的，无效是 RL 层加的判断"
    assert metas[0]["invalid_reason"] == "weak_parse:line_scan"


def test_weak_parse_is_scored_when_explicitly_disabled():
    """`require_strong_parse=False` 回到评测口径：弱解析照常打分。

    保留这个开关是因为"抠出来的是不是真序列"这件事本身有分析价值——
    但别为了把 `usable_ratio` 做好看去关它。
    """
    _, metas = parse_completions(
        [WEAK_TEXT], [{"task": "de_novo"}], require_strong_parse=False
    )
    assert "invalid_reason" not in metas[0]


def test_weak_parse_yields_no_advantage_and_no_reward():
    """端到端：弱解析的序列既不计入奖励，也不参与组内优势。

    组里另一条是正常格式的好序列——如果弱解析被算了分，
    它会作为基线的一部分参与比较；这里断言它被完全排除。
    """
    scorer = FakeScorer({WEAK_TEXT: 99.0, "KWKLFKKIEK": 1.0})
    batch = assemble_rollouts(
        [WEAK_TEXT, completion("KWKLFKKIEK")],
        [{"task": "de_novo"}],
        group_size=2,
        scorer=scorer,
    )
    g = batch.groups[0]
    assert g.scores[0].invalid_reason == "weak_parse:line_scan"
    # 虽然查表能查到 99 分，但这条不该被算进任何统计
    assert g.advantages[0] == 0.0
    assert batch.reward_report["mean"] == 1.0, "弱解析的 99 分混进了均值"
    assert batch.reward_report["invalid"] == 1
    # 组内只剩一条有效分数 → std 为 0 → 这一步这一组没有梯度。
    # 这正是"弱解析按无效处理"的代价，也是它该有的样子：
    # 与其拿一条假的基线去训，不如这一步不训。
    assert g.reward_std() == 0.0
    assert batch.usable_ratio == 0.5


def test_weak_parse_counts_toward_the_effective_fail_rate():
    """两个失败率口径要分清楚，否则会打出自相矛盾的日志。

    `parse_stats.fail_rate` 只数"解析器完全没抠出东西"的那些，
    所以一批全是弱解析的 rollout 会让它报 0%，而 `usable_ratio` 是 0%——
    两个数打架，而"日志里打架的两个数"是最难查的一类线索。
    所以 `assemble_rollouts` 把 `fail_rate` 改成有效失败率
    （硬失败 + 弱解析），解析器自己的口径挪到 `raw_fail_rate` 备查。
    """
    scorer = FakeScorer({WEAK_TEXT: 1.0, "KWKLFKKIEK": 2.0})
    batch = assemble_rollouts(
        [WEAK_TEXT, completion("KWKLFKKIEK")],
        [{"task": "de_novo"}],
        group_size=2,
        scorer=scorer,
    )
    report = batch.parse_report
    assert report["total"] == 2
    assert report["raw_fail_rate"] == 0.0, "解析器自己没失败，它确实抠出了东西"
    assert report["weak_parse"] == 1
    assert report["fail_rate"] == 0.5, "有效失败率必须把弱解析算进去"
    # 有效失败率与可用率的一致性：这里两者都是 0.5，对得上
    assert report["fail_rate"] == pytest.approx(1 - batch.usable_ratio)


def test_hard_parse_failure_is_still_counted_separately():
    """硬失败和弱解析要分得开——它们的处方完全不同。

    两者都是 `fail_rate` 的一部分，但 `weak_parse` 只数弱解析。
    "抠不出东西"是 SFT 没教够格式；"抠出英文词"是模型在赌启发式。
    """
    scorer = FakeScorer({WEAK_TEXT: 1.0, "KWKLFKKIEK": 2.0})
    batch = assemble_rollouts(
        [WEAK_TEXT, "!!! ???", completion("KWKLFKKIEK"), completion("AAAAA")],
        [{"task": "de_novo"}, {"task": "de_novo"}],
        group_size=2,
        scorer=scorer,
    )
    report = batch.parse_report
    assert report["total"] == 4
    assert report["weak_parse"] == 1, "只有英文词那一条是弱解析"
    assert report["raw_fail_rate"] == 0.25, "只有 '!!! ???' 是硬失败"
    assert report["fail_rate"] == 0.5, "有效失败率 = 硬失败 1 + 弱解析 1，共 2/4"


# ---------------------------------------------------------------------------
# 逐维组内 std：判断"是哪几维在动"
# ---------------------------------------------------------------------------
def _breakdown(seq: str, **dims: float) -> RewardBreakdown:
    """造一条带逐维分数的奖励分解。`dims` 是 `维度名=value`。"""
    return RewardBreakdown(
        sequence=seq,
        total=sum(dims.values()),
        dimensions={
            name: DimensionScore(dimension=name, value=v, raw=v)
            for name, v in dims.items()
        },
        penalties={},
    )


class TestDimensionStds:
    """总分有方差 ≠ 每一维都有方差。

    这是加这个字段的全部理由：GRPO 的优势由总分算出来，
    总分里只要有一维在动，整组优势就有值。于是"某一维已经死了"
    会被总分掩盖——它名义上占着权重、实际对优势零贡献。

    实测到的两处退化（详见奖励审计）：`protease` 在疏水/碱性肽上恒为 0
    （位点密度饱和封顶），`structure` 触发一票否决时恒为 0.3。
    这两维合计占默认权重的 40%。
    """

    def test_reports_std_per_dimension_not_just_the_total(self):
        """构造"总分在动、但只有一维在动"的组——这正是要暴露的情形。"""
        group = RolloutGroup(
            prompt_index=0,
            meta={},
            texts=["a", "b", "c"],
            scores=[
                _breakdown("a", thermo=0.5, protease=0.0),
                _breakdown("b", thermo=0.9, protease=0.0),
                _breakdown("c", thermo=0.1, protease=0.0),
            ],
        )
        stds = group.dimension_stds()

        assert group.reward_std() > 0.2, "总分明明有方差"
        assert stds["thermo"] > 0.2, "thermo 在动"
        assert stds["protease"] == 0.0, (
            "protease 一维根本没动，必须报 0——"
            "只看总分的话这一维的死活是看不出来的"
        )

    def test_needs_at_least_two_valid_samples_or_it_is_unmeasurable(self):
        """不足 2 条有效样本时给 0。

        这里 0 的含义是"量不出来"，不是"量出来是 0"。
        两者在排查时指向不同方向，但都表现为"这一维没有贡献优势"，
        所以接口上统一给 0，由调用方决定怎么解读。
        """
        group = RolloutGroup(
            prompt_index=0, meta={}, texts=["a"],
            scores=[_breakdown("a", thermo=0.5)],
        )
        assert group.dimension_stds()["thermo"] == 0.0

    def test_invalid_samples_do_not_enter_the_dimension_std(self):
        """无效样本不参与逐维 std——与组内优势的口径保持一致。

        它们已经在优势计算里被排除（`group_advantages` 只对有效样本
        求 mean/std 但保留占位）。这里若把它们算进去，
        逐维 std 就会和真正驱动优势的那个量对不上。
        """
        bad = _breakdown("bad", thermo=0.0)
        bad.invalid_reason = "weak_parse:line_scan"
        group = RolloutGroup(
            prompt_index=0, meta={}, texts=["a", "b", "bad"],
            scores=[
                _breakdown("a", thermo=0.2),
                _breakdown("b", thermo=0.4),
                bad,
            ],
        )
        stds = group.dimension_stds()
        # 有效两条 0.2 / 0.4 → 总体标准差 0.1；若把 0.0 也算进来会变成 ~0.163
        assert stds["thermo"] == pytest.approx(0.1)

    def test_all_invalid_group_reports_nothing(self):
        """整组无效 → 没有任何维度可报，而不是报一堆 0。"""
        bad = _breakdown("bad", thermo=0.0)
        bad.invalid_reason = "parse_failed"
        group = RolloutGroup(prompt_index=0, meta={}, texts=["bad"], scores=[bad])
        assert group.dimension_stds() == {}
