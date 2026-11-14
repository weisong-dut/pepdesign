"""奖励加固的回归测试：均聚物、任务约束、末端修饰、逐维 std 聚合。

这几条都是实际跑出来的漏洞，不是理论上的担心：

* 在原始四维权重下，T20/N20/Q20/H20 的总分是 0.74~0.76，高于
  LL-37（0.479）和 magainin-2（0.312）。原因是 `protease` 对
  "没有已知酶切位点"的序列直接给 1.0，而 structure 没有低复杂度判据。
  GRPO 会沿着这条梯度走到"高奖励但不像肽"的区域。
* `de_novo` 的 `length_range` / `charge_range` 和 `multi_objective` 的
  `allow_length_change` 以前只写在 prompt 文本里，奖励函数完全不读，
  模型可以无视指令还拿高分。
* 模型输出 `Ac-...-NH2` 时，解析器把修饰剥离了，但 `parse_completions`
  只复制原 meta，没有把 `parsed.modifications` 写回去，于是
  `protease.py` 读不到末端封闭，prompt 里教的策略在 RL 里拿不到奖励。
* `run_step` 把"有效样本 <2、量不出来"的组按 0.0 计入逐维 std 平均，
  小样本组越多、日志上的 std 越低，看起来像"这一维死了"。
"""

from __future__ import annotations

import pytest

from pepdesign.rewards.base import (
    DimensionScore,
    PenaltyConfig,
    RewardBreakdown,
    compute_penalties,
)
from pepdesign.rewards.composite import CompositeScorer
from pepdesign.train.rollout import (
    RolloutGroup,
    aggregate_dimension_stds,
    parse_completions,
)

# 五条真实肽；加固后它们的低复杂度惩罚必须保持为空。
REAL_PEPTIDES = {
    "LL-37": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
    "magainin-2": "GIGKFLHSAKKFGKAFVGEIMNS",
    "melittin": "GIGAVLKVLTTGLPALISWIKRKRQQ",
    "indolicidin": "ILPWKWPWWPWRR",
    "histatin-1": "DSHEKRHHGYRRKFHEKHHSHREFPFYGDYGSNYLYDN",
}


class TestHomopolymerExploit:
    def test_polar_homopolymers_are_no_longer_high_reward(self):
        """T20/N20/Q20/H20 是原始审计漏掉、但实测能刷分的四个。"""
        scorer = CompositeScorer()
        for aa in "TNQH":
            b = scorer.score(aa * 20, {"task": "de_novo"})
            assert b.total < 0.10, (
                f"{aa}20 仍然拿到 {b.total}——低复杂度惩罚没有生效，"
                f"GRPO 会继续往这条路上走"
            )
            assert "low_complexity" in b.penalties
            assert "homopolymer_run" in b.penalties

    def test_two_residue_repeats_are_penalized(self):
        """QHQH… / QTQT… 的最大残基占比只有 0.5，绕得过 low_complexity。

        它们的 Shannon 熵只有 1 bit（真实肽中位 3.48），显然不是肽。
        """
        scorer = CompositeScorer()
        for seq in ("QH" * 10, "QT" * 10, "TTNN" * 5):
            b = scorer.score(seq, {"task": "de_novo"})
            assert "low_entropy" in b.penalties, f"{seq} 没触发低熵惩罚"
            assert b.total < 0.45, f"{seq} 仍然拿到 {b.total}"

    def test_short_alternating_peptide_is_penalized(self):
        """7-mer 的 QHQHQHQ 熵只有 1 bit；短肽用 1.5 bit 的阈值抓它。"""
        b = CompositeScorer().score("QHQHQHQ", {"task": "de_novo"})
        assert "low_entropy" in b.penalties
        assert b.total < 0.30

    def test_short_real_peptide_is_not_penalized(self):
        """真实短肽的熵最低 1.84，1.5 的阈值不会误伤。"""
        b = CompositeScorer().score("GIGKFLH", {"task": "de_novo"})
        assert "low_entropy" not in b.penalties

    def test_real_peptides_do_not_trigger_low_complexity_penalties(self):
        scorer = CompositeScorer()
        for name, seq in REAL_PEPTIDES.items():
            b = scorer.score(seq, {"task": "de_novo"})
            assert "low_complexity" not in b.penalties, f"{name} 被误伤了"
            assert "homopolymer_run" not in b.penalties, f"{name} 被误伤了"
            assert "low_entropy" not in b.penalties, f"{name} 被误伤了"
            assert b.total > 0.20, f"{name} 的正常分数被压得太低：{b.total}"

    def test_penalty_is_graded_not_a_boolean(self):
        """占比越高罚得越重，模型才能收到"逐步远离"的信号。"""
        scorer = CompositeScorer()
        mild = scorer.score("A" * 10 + "CDEFGHIKLM", {"task": "de_novo"})
        severe = scorer.score("A" * 15 + "CDEFG", {"task": "de_novo"})
        assert severe.penalties["low_complexity"] > mild.penalties["low_complexity"]

    def test_zero_strength_does_not_add_empty_keys(self):
        cfg = PenaltyConfig(
            aggregation=0.0,
            hydrophobic_run=0.0,
            charge=0.0,
            low_complexity=0.0,
            homopolymer_run=0.0,
            low_entropy=0.0,
            invalid=0.0,
        )
        penalties = compute_penalties("A" * 20, cfg)
        assert "low_complexity" not in penalties
        assert "homopolymer_run" not in penalties


class TestTaskConstraints:
    def test_de_novo_length_violation_is_penalized(self):
        b = CompositeScorer().score(
            "ACDEFGHIKL", {"task": "de_novo", "length_range": [20, 25]}
        )
        assert b.penalties.get("length") == pytest.approx(0.6)

    def test_de_novo_length_in_range_has_no_penalty(self):
        b = CompositeScorer().score(
            "ACDEFGHIKL", {"task": "de_novo", "length_range": [8, 12]}
        )
        assert "length" not in b.penalties

    def test_multi_objective_length_change_is_penalized(self):
        b = CompositeScorer().score(
            "ACDEFGHIKLM",
            {
                "task": "multi_objective",
                "allow_length_change": False,
                "base_sequence": "ACDEFGHIKLMN",
            },
        )
        assert b.penalties.get("length") == pytest.approx(0.30)

    def test_charge_range_violation_is_penalized(self):
        # 把其它惩罚关掉，单独看任务约束这一条。
        cfg = PenaltyConfig(
            aggregation=0.0,
            hydrophobic_run=0.0,
            charge=0.0,
            low_complexity=0.0,
            homopolymer_run=0.0,
            low_entropy=0.0,
            invalid=0.0,
        )
        b = CompositeScorer(penalty_cfg=cfg).score(
            "KWKWKWKWKW", {"task": "de_novo", "charge_range": [-10.0, 0.0]}
        )
        assert "charge_target" in b.penalties
        assert b.penalties["charge_target"] > 0


class TestModificationPropagation:
    def test_terminal_modifications_reach_the_scoring_meta(self):
        """`Ac-...-NH2` 的末端封闭必须能被 protease scorer 看到。"""
        text = "### 设计序列\nAc-KWKLFKK-NH2\n### 设计理由\n测试改造点。\n### 结束"
        parsed, metas = parse_completions([text], [{"task": "stabilize"}])

        assert parsed[0].ok
        assert parsed[0].sequence == "KWKLFKK"
        assert set(metas[0]["modifications"]) == {"n_acetyl", "c_amide"}

    def test_no_modifications_does_not_add_the_key(self):
        text = "### 设计序列\nKWKLFKK\n### 设计理由\n测试改造点。\n### 结束"
        _, metas = parse_completions([text], [{"task": "stabilize"}])
        assert "modifications" not in metas[0]


def _breakdown(seq: str, **dims: float) -> RewardBreakdown:
    return RewardBreakdown(
        sequence=seq,
        total=sum(dims.values()),
        dimensions={
            name: DimensionScore(dimension=name, value=v, raw=v)
            for name, v in dims.items()
        },
        penalties={},
    )


class TestAggregateDimensionStds:
    def test_unmeasurable_groups_do_not_drag_the_average_down(self):
        """只有 1 条有效样本的组必须被排除，而不是按 0.0 计入。"""
        one = RolloutGroup(
            prompt_index=0,
            meta={},
            texts=["a"],
            scores=[_breakdown("a", thermo=0.5)],
        )
        three = RolloutGroup(
            prompt_index=1,
            meta={},
            texts=["a", "b", "c"],
            scores=[
                _breakdown("a", thermo=0.5),
                _breakdown("b", thermo=0.9),
                _breakdown("c", thermo=0.1),
            ],
        )

        out = aggregate_dimension_stds([one, three])
        # 0.5/0.9/0.1 的总体标准差
        assert out["thermo"] == pytest.approx(0.3266, abs=1e-4)

    def test_all_unmeasurable_returns_empty(self):
        one = RolloutGroup(
            prompt_index=0,
            meta={},
            texts=["a"],
            scores=[_breakdown("a", thermo=0.5)],
        )
        assert aggregate_dimension_stds([one]) == {}


class TestNonstandardResidues:
    """奖励只接受 20 种标准氨基酸。

    这是实测漏洞：`XBZUOXBZUOXBZUOXBZUOA`（19 个模糊残基 + 1 个 A）
    在 de_novo 下能拿 0.60，高于 95% 的真实肽。原因是 `validate` 对长片段
    允许 X/B/Z/U/O，而低复杂度判据看不出"5 个符号近似均匀"。
    解析器把 `RGDfV` 归一化成 RGDFV 后会丢手性，所以用
    `noncanonical_residues` 标记单独判无效。
    """

    def test_ambiguous_heavy_sequence_is_invalid(self):
        scorer = CompositeScorer()
        b = scorer.score("XBZUOXBZUOXBZUOXBZUOA", {"task": "de_novo"})
        assert b.invalid_reason == "nonstandard_residue"
        assert b.total == scorer.cfg.clip_min

    def test_x_homopolymer_is_invalid(self):
        b = CompositeScorer().score("X" * 20, {})
        # 全 X 会先被 validate 判成 all_ambiguous；只要不是"可打分"即可。
        assert b.invalid_reason is not None

    def test_d_amino_tag_is_invalid(self):
        b = CompositeScorer().score(
            "RGDFV", {"modifications": ["noncanonical_residues"]}
        )
        assert b.invalid_reason == "noncanonical_residues"

    def test_standard_sequence_is_not_invalid(self):
        b = CompositeScorer().score("KWKLFKKIEK", {})
        assert b.invalid_reason is None


class TestFeasibleCyclicType:
    """数据里的 `disulfide` 标签不能直接当 prompt 约束。

    814 条 disulfide 记录里只有一部分的 Cys 数/配对准许二硫键环化。
    对不可行的记录，prompt 和奖励都不能再要求"保持二硫键"，
    否则 SFT 的答案（记录本身）会同时被要求满足和违反同一个约束。
    """

    def test_linear_stays_none(self):
        from pepdesign.bio.cyclization import feasible_cyclic_type

        assert feasible_cyclic_type("ACDEFGHIK", "none") == "none"

    def test_disulfide_without_cys_becomes_none(self):
        from pepdesign.bio.cyclization import feasible_cyclic_type

        assert feasible_cyclic_type("ACDEFGHIK", "disulfide") == "none"

    def test_feasible_disulfide_is_kept(self):
        from pepdesign.bio.cyclization import feasible_cyclic_type

        seq = "EATCITEMSVMMACWKQNDFNDTVCAEEIRMFYDCVAKAE"
        assert feasible_cyclic_type(seq, "disulfide") == "disulfide"
