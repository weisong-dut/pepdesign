"""校准集的改造算子。

校准集由"规则引擎给出改造建议，再用预测器验证收益"的对照数据构成，
同时用于 SFT few-shot 示例与奖励标定。因此改造算子改动的位置必须与
文档声称的位点一致，否则训练数据会教一个与说明不符的手法。
"""

from __future__ import annotations

from pepdesign.bio.cleavage import analyze, suggest_stabilization
from pepdesign.data.build_calibration import apply_pro_sweep, apply_terminal_cap


class TestProSweep:
    """`apply_pro_sweep` 必须落在 P1' 上。"""

    SEQ = "KWKLFKKIEK"

    def test_mutates_p1_prime_not_p1(self):
        """改动下标必须等于 P1 下标 + 1，即 P1' 位。

        `suggest_stabilization` 的 `position` 是 1-based P1 下标，P1' 的
        0-based 下标恰好等于它；位点从 `analyze` 重新推导，避免与实现共用假设。
        """
        site = analyze(self.SEQ).sites[0]
        new_seq = apply_pro_sweep(self.SEQ)[0]
        assert len(new_seq) == len(self.SEQ)
        mutated = [
            i
            for i, (a, b) in enumerate(zip(self.SEQ, new_seq, strict=True))
            if a != b
        ]
        assert mutated == [site.position + 1], (
            f"改动下标 {mutated}，期望 [{site.position + 1}]（P1' 位）"
        )

    def test_replaced_residue_matches_the_suggestion(self):
        """替换上去的是 P，且被替换的残基正是建议里的 `from`。"""
        suggestion = suggest_stabilization(self.SEQ, max_suggestions=1)[0]
        new_seq, note = apply_pro_sweep(self.SEQ)
        assert note.startswith(f"P1'位 {suggestion['from']}->P")
        pos = int(suggestion["position"])
        assert self.SEQ[pos] == suggestion["from"]
        assert new_seq[pos] == "P"

    def test_it_actually_removes_a_cleavage_site(self):
        """改造后加权位点数应下降。"""
        before = analyze(self.SEQ).weighted_site_count
        after = analyze(apply_pro_sweep(self.SEQ)[0]).weighted_site_count
        assert after < before

    def test_changes_exactly_one_position(self):
        """一次只改一处。"""
        new_seq, _ = apply_pro_sweep(self.SEQ)
        diff = sum(1 for a, b in zip(self.SEQ, new_seq, strict=True) if a != b)
        assert diff == 1

    def test_terminal_suggestions_are_not_treated_as_site_mutations(self):
        """`suggest_stabilization` 的末端封闭建议不是位点替换，必须拒掉。"""
        # 一条只有 DPP4 风险、没有可做 P1' 替换的位点的序列：
        # N 端是 Xaa-Pro，suggest_stabilization 只会给出"N 端封闭"。
        out = apply_pro_sweep("PWKWKWKWKK")
        assert out is None or out[1].startswith("P1'位")


class TestTerminalCap:
    def test_never_changes_the_sequence(self):
        """末端封闭改变修饰而非序列。"""
        for seq in ("KWKLFKKIEK", "GIGAVLKVLTTGLPALISWIKRKRQQ"):
            for terminal in ("n", "c"):
                r = apply_terminal_cap(seq, terminal)
                if r is not None:
                    assert r[0] == seq

    def test_n_acetyl_is_refused_when_n_terminal_is_negative(self):
        """N 端带负电时乙酰化削弱静电作用，不应建议。"""
        assert apply_terminal_cap("DEKWKLFKK", "n") is None

    def test_c_amide_is_refused_when_c_terminal_is_positive(self):
        """C 端带正电时酰胺化收益低。"""
        assert apply_terminal_cap("KWKLFKKIEK", "c") is None

    def test_reports_the_two_caps_differently(self):
        n = apply_terminal_cap("KWKLFKKIEK", "n")
        c = apply_terminal_cap("GIGAVLKVLTTGLPALIA", "c")
        assert n is not None and "乙酰化" in n[1]
        assert c is not None and "酰胺化" in c[1]
