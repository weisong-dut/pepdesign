"""序列性质计算的数值正确性。

重点锁定 `aliphatic_index` 的量纲与值域：输入为摩尔百分数（0-100），
正确区间是 [0, 390]。量纲错误会同时破坏 Tm 预测器与短肽代理分的标定。
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from pepdesign.bio.aa import RESIDUE_MASS, WATER_MASS
from pepdesign.bio.sequence import (
    MAX_PEPTIDE_LEN,
    MIN_PEPTIDE_LEN,
    aliphatic_index,
    amino_acid_composition,
    aromaticity,
    canonicalize,
    compute_properties,
    compute_properties_batch,
    coverage,
    gravy,
    has_noncanonical_residues,
    hydrophobic_moment,
    is_cyclic_capable,
    isoelectric_point,
    max_hydrophobic_run,
    molecular_weight,
    net_charge,
    net_charge_fast,
    standard_only,
    strip_terminal_modifications,
    validate,
)


class TestAliphaticIndex:
    """Ikai (1980) 脂肪族指数：AI = A% + 2.9·V% + 3.9·(I% + L%)，值域 [0, 390]。"""

    def test_maximum_is_exactly_390(self):
        """polyleucine / polyisoleucine 必须恰好取到 390；任何量纲错误都会使其偏离。"""
        assert aliphatic_index("L" * 20) == pytest.approx(390.0)
        assert aliphatic_index("I" * 20) == pytest.approx(390.0)
        # I 和 L 在公式里同权重，混合也应是 390
        assert aliphatic_index("IL" * 10) == pytest.approx(390.0)

    def test_valine_only_gives_290(self):
        """全 Val：V% = 100 → 2.9 × 100 = 290。"""
        assert aliphatic_index("V" * 20) == pytest.approx(290.0)

    def test_alanine_only_gives_100(self):
        """全 Ala：A% = 100，系数 1.0 → 100。"""
        assert aliphatic_index("A" * 20) == pytest.approx(100.0)

    def test_residues_outside_the_formula_contribute_zero(self):
        """公式只统计 A/V/I/L 四个残基，其余残基 AI = 0。"""
        for aa in "RNDCQEGHMFKPSTWY":
            assert aliphatic_index(aa * 10) == pytest.approx(0.0)

    def test_never_exceeds_390_on_real_peptides(self):
        """真实序列上的 AI 必须落在 [0, 390]。"""
        peptides = [
            "GIGAVLKVLTTGLPALISWIKRKRQQ",   # melittin
            "KWKLFKKIEK", "RRWQWR", "ACDEFGHIKLMN",
            "AILAILAILAIL", "ILVAGKILVAGK",
        ]
        for p in peptides:
            ai = aliphatic_index(p)
            assert 0.0 <= ai <= 390.0, f"{p} 的 AI = {ai} 超出 [0,390]"

    def test_known_value_for_melittin(self):
        """melittin 的 AI ≈ 135，作为防止回退的数值锚点。"""
        assert aliphatic_index("GIGAVLKVLTTGLPALISWIKRKRQQ") == pytest.approx(135.0, abs=0.5)

    def test_is_monotonic_in_aliphatic_content(self):
        """脂肪族残基占比越高，AI 越大。"""
        seqs = ["K" * 10, "AK" * 5, "AAKK", "AL" * 5, "LL" * 5, "L" * 10]
        vals = [aliphatic_index(s) for s in seqs]
        assert all(b >= a for a, b in pairwise(vals)), vals

    def test_empty_sequence_is_zero(self):
        assert aliphatic_index("") == 0.0


class TestComposition:
    def test_counts_sum_to_length(self):
        for s in ["ACDEFGHIKLMNPQRSTVWY", "AAAA", "KWKLFKKIEK"]:
            comp = amino_acid_composition(s)
            assert sum(comp.values()) == pytest.approx(len(s))

    def test_percent_sums_to_100(self):
        """percent=True 返回摩尔百分数，即 AI 公式的输入量纲。"""
        comp = amino_acid_composition("ACDEFGHIKLMNPQRSTVWY", percent=True)
        assert sum(comp.values()) == pytest.approx(100.0)
        assert max(comp.values()) == pytest.approx(5.0)

    def test_composition_of_homopolymer(self):
        comp = amino_acid_composition("LLLL", percent=True)
        assert comp["L"] == pytest.approx(100.0)

    def test_composition_is_sparse(self):
        """只列出出现过的残基；消费方使用 `comp.get(aa, 0.0)` 取缺失键。"""
        comp = amino_acid_composition("LLLL", percent=True)
        assert set(comp) == {"L"}
        assert "A" not in comp
        # 消费方必须用 .get 取缺失键
        assert comp.get("A", 0.0) == 0.0


class TestGravy:
    def test_hydrophobic_homopolymer_is_positive(self):
        assert gravy("I" * 10) > 0
        assert gravy("L" * 10) > 0

    def test_hydrophilic_homopolymer_is_negative(self):
        assert gravy("K" * 10) < 0
        assert gravy("D" * 10) < 0

    def test_matches_manual_average_of_kd(self):
        """GRAVY 就是 KD 值的算术平均，用定义直接核对。"""
        from pepdesign.bio.aa import KYTE_DOOLITTLE

        s = "ACDEFGHIKL"
        expected = sum(KYTE_DOOLITTLE[a] for a in s) / len(s)
        assert gravy(s) == pytest.approx(expected, abs=1e-9)


class TestNetCharge:
    def test_positive_for_basic_peptide(self):
        assert net_charge("K" * 5) > 0
        assert net_charge("R" * 5) > 0

    def test_negative_for_acidic_peptide(self):
        assert net_charge("D" * 5) < 0
        assert net_charge("E" * 5) < 0

    def test_histidine_is_mostly_uncharged_at_ph74(self):
        """pH 7.4 下 His 侧链（pKa 6.0）仅约 3.8% 质子化，净电荷约为 +0.038。"""
        q = net_charge("H")
        assert 0.0 < q < 0.2, f"His 在 pH 7.4 的净电荷 {q} 不合理"

    def test_fast_table_agrees_with_exact_within_tolerance(self):
        """快速表（His 取 +0.1）与精确 H-H 方程应接近；两者服务于不同规模的场景。"""
        for s in ["KWKLFKKIEK", "RRWQWR", "ACDEFGHIKLMN", "HHHHH"]:
            diff = abs(net_charge(s) - net_charge_fast(s))
            assert diff < 1.0, f"{s}: 精确 {net_charge(s)} vs 快速 {net_charge_fast(s)}"

    def test_is_monotonic_in_ph(self):
        """pH 越高，净电荷越低（碱性 pH 下羧基去质子化、氨基去质子化）。"""
        s = "KWKLFKKIEK"
        qs = [net_charge(s, ph) for ph in (4.0, 6.0, 7.4, 9.0, 11.0)]
        assert all(b <= a for a, b in pairwise(qs)), qs

    def test_empty_sequence(self):
        assert net_charge("") == pytest.approx(0.0)


class TestIsoelectricPoint:
    def test_acidic_peptide_has_low_pi(self):
        assert isoelectric_point("DDDDD") < 5.0

    def test_basic_peptide_has_high_pi(self):
        assert isoelectric_point("KKKKK") > 9.0

    def test_charge_is_near_zero_at_pi(self):
        """pI 即净电荷为 0 的 pH。"""
        for s in ["KWKLFKKIEK", "ACDEFGHIKLMN", "GIGAVLKVL"]:
            pi = isoelectric_point(s)
            assert abs(net_charge(s, pi)) < 0.5, f"{s}: pI={pi} 处净电荷不为 0"

    def test_neutral_peptide_is_near_7(self):
        pi = isoelectric_point("ACDEFGHIKLMN")
        assert 4.0 < pi < 9.0


class TestAromaticity:
    def test_only_aromatic_residues_count(self):
        assert aromaticity("FWY") == pytest.approx(1.0)
        assert aromaticity("AAA") == pytest.approx(0.0)

    def test_fraction_of_sequence(self):
        assert aromaticity("FWYA") == pytest.approx(0.75)


class TestHydrophobicMoment:
    def test_requires_positive_window(self):
        """window < 1 无意义，必须报错。"""
        for w in (0, -1):
            with pytest.raises(ValueError):
                hydrophobic_moment("ACDEFGHIKLMN", window=w)

    def test_amphipathic_helix_has_higher_moment_than_random(self):
        """两亲性螺旋的疏水矩应显著高于同组成的乱序序列。"""
        import random

        s = "LKKLLKLLKKLLKL"
        chars = list(s)
        random.Random(0).shuffle(chars)
        shuffled = "".join(chars)

        ordered_mu = hydrophobic_moment(s, angle_deg=100.0)
        shuffled_mu = hydrophobic_moment(shuffled, angle_deg=100.0)

        # 组成相同（只是顺序不同）
        assert sorted(s) == sorted(shuffled)
        assert ordered_mu > shuffled_mu * 2

    def test_is_non_negative(self):
        for s in ["ACDEFGHIKLMN", "LLLKKK", "RRWQWR"]:
            assert hydrophobic_moment(s) >= 0.0

    def test_helix_angle_is_100_degrees(self):
        """α-螺旋周角为 100°/残基；改成 180° 会错误计算沿链分布。"""
        import inspect

        sig = inspect.signature(hydrophobic_moment)
        assert sig.parameters["angle_deg"].default == pytest.approx(100.0)


class TestHydrophobicRun:
    def test_counts_longest_consecutive_run(self):
        assert max_hydrophobic_run("LLLKLLL") == 3
        assert max_hydrophobic_run("LLLLL") == 5

    def test_alanine_counts_as_hydrophobic(self):
        """Ala 属于 HYDROPHOBIC_AA（KD 1.8），`AAALLLLAAA` 是 10 个连续疏水残基。"""
        assert max_hydrophobic_run("AAALLLLAAA") == 10

    def test_glycine_and_proline_do_not_count(self):
        """G/P 不在 HYDROPHOBIC_AA 中；二者 KD 偏亲水，聚甘氨酸是常用柔性 linker。"""
        assert max_hydrophobic_run("GGGGGG") == 0
        assert max_hydrophobic_run("PPPPPP") == 0
        assert max_hydrophobic_run("GGGGGGLLLLLL") == 6

    def test_run_broken_by_polar_residue(self):
        assert max_hydrophobic_run("LLLKLLL") == 3
        assert max_hydrophobic_run("LLLDLLL") == 3


class TestMolecularWeight:
    def test_linear_peptide_adds_water(self):
        """线性肽 = 残基质量之和 + 一分子水。"""
        s = "ACDEFGHIKL"
        expected = sum(RESIDUE_MASS[a] for a in s) + WATER_MASS
        assert molecular_weight(s) == pytest.approx(expected, abs=1e-6)

    def test_cyclic_is_18_da_lighter(self):
        """头尾环化脱去一分子水，环肽比线性形式轻 18.02 Da。"""
        s = "KWKLFKKIEK"
        linear = molecular_weight(s)
        cyclic = compute_properties(s, cyclic=True).molecular_weight
        assert linear - cyclic == pytest.approx(WATER_MASS, abs=1e-6)

    def test_single_residue(self):
        assert molecular_weight("A") == pytest.approx(RESIDUE_MASS["A"] + WATER_MASS)

    def test_scales_with_length(self):
        assert molecular_weight("AAAA") < molecular_weight("AAAAAAAA")


class TestCyclizationConsistency:
    def test_cyclic_capable_matches_head_to_tail_min_length(self):
        """`is_cyclic_capable` 与 `bio/cyclization.HEAD_TO_TAIL_MIN_LEN` 必须使用同一下界。"""
        from pepdesign.bio.cyclization import HEAD_TO_TAIL_MIN_LEN

        assert is_cyclic_capable("A" * HEAD_TO_TAIL_MIN_LEN) is True
        assert is_cyclic_capable("A" * (HEAD_TO_TAIL_MIN_LEN - 1)) is False


class TestValidation:
    def test_rejects_empty(self):
        ok, reason = validate("")
        assert not ok and reason

    def test_rejects_too_short(self):
        ok, _ = validate("A" * (MIN_PEPTIDE_LEN - 1))
        assert not ok

    def test_rejects_too_long(self):
        ok, _ = validate("A" * (MAX_PEPTIDE_LEN + 1))
        assert not ok

    def test_accepts_normal_peptide(self):
        ok, reason = validate("KWKLFKKIEK")
        assert ok, reason

    def test_accepts_boundary_lengths(self):
        assert validate("A" * MIN_PEPTIDE_LEN)[0]
        assert validate("A" * MAX_PEPTIDE_LEN)[0]


class TestCanonicalize:
    def test_uppercases_and_strips_noise(self):
        assert canonicalize("acdef-g hik") == "ACDEFGHIK"
        assert canonicalize("  ACDE  ") == "ACDE"

    def test_preserves_ambiguous_residues(self):
        """X/B/Z 是合法模糊码，规范化时不能抹掉。"""
        assert canonicalize("AXBZ") == "AXBZ"


class TestStandardOnlyAndCoverage:
    def test_standard_only_filters_ambiguous(self):
        assert standard_only("ACDXBZ") == "ACD"

    def test_coverage_is_fraction(self):
        assert coverage("ACDE") == pytest.approx(1.0)
        assert coverage("ACDX") == pytest.approx(0.75)


class TestNoncanonicalDetection:
    def test_mixed_case_flags_d_residues(self):
        """`RGDfV` 的小写 `f` 是 D-Phe；upper() 后手性信息丢失，必须标记。"""
        assert has_noncanonical_residues("RGDfV") is True

    def test_all_upper_is_fine(self):
        assert has_noncanonical_residues("RGDFV") is False
        assert has_noncanonical_residues("KWKLFKKIEK") is False

    def test_all_lower_is_fine(self):
        """全小写表示大小写不敏感地书写序列，不是 D-型残基。"""
        assert has_noncanonical_residues("rgdfv") is False

    def test_selenocysteine_and_pyrrolysine(self):
        assert has_noncanonical_residues("ACU") is True
        assert has_noncanonical_residues("ACO") is True


class TestStripTerminalModifications:
    def test_strips_n_acetyl(self):
        body, mods = strip_terminal_modifications("Ac-KWKLFKK-NH2")
        assert body == "KWKLFKK"
        assert "n_acetyl" in mods and "c_amide" in mods

    def test_does_not_strip_all_caps_ac(self):
        """修饰标注必须大小写敏感：全大写 `AC` 是 Ala-Cys 两个残基。"""
        body, mods = strip_terminal_modifications("AC-KWKLFKK")
        assert "n_acetyl" not in mods
        assert body.startswith("AC")

    def test_no_modifications_returns_original(self):
        body, mods = strip_terminal_modifications("KWKLFKKIEK")
        assert body == "KWKLFKKIEK"
        assert mods == []

    def test_c_amide_variants(self):
        for raw in ("KWKLFKK-NH2", "KWKLFKK-NH₂", "KWKLFKK-酰胺化"):
            body, mods = strip_terminal_modifications(raw)
            assert body == "KWKLFKK", raw
            assert "c_amide" in mods, raw


class TestComputeProperties:
    def test_returns_all_expected_fields(self):
        props = compute_properties("KWKLFKKIEK")
        for field in ("molecular_weight", "gravy", "net_charge_ph74",
                      "isoelectric_point", "aliphatic_index", "coverage"):
            assert hasattr(props, field)

    def test_aliphatic_index_in_range_through_the_properties_snapshot(self):
        """同一约束在 compute_properties 入口上同样成立；`predictors/tm.py` 走这条路径。"""
        for s in ["GIGAVLKVLTTGLPALISWIKRKRQQ", "AILAILAILAIL", "RRWQWR"]:
            assert 0.0 <= compute_properties(s).aliphatic_index <= 390.0

    def test_cyclic_flag_changes_molecular_weight_only(self):
        """环化只改变分子量；两端 +1/-1 电荷抵消，净电荷不变。"""
        s = "KWKLFKKIEK"
        lin = compute_properties(s, cyclic=False)
        cyc = compute_properties(s, cyclic=True)
        assert lin.molecular_weight != cyc.molecular_weight
        assert lin.net_charge_ph74 == pytest.approx(cyc.net_charge_ph74)

    def test_batch_matches_single(self):
        seqs = ["KWKLFKKIEK", "RRWQWR", "ACDEFGHIKLMN"]
        batch = compute_properties_batch(seqs)
        assert len(batch) == len(seqs)
        for s, p in zip(seqs, batch, strict=True):
            assert p.aliphatic_index == pytest.approx(compute_properties(s).aliphatic_index)

    def test_is_in_core_length(self):
        assert compute_properties("A" * 10).is_in_core_length
        assert not compute_properties("A" * 50).is_in_core_length


class TestNoInfinityOrNaN:
    """奖励函数对每条样本都要算这些性质，任何一个 nan 都会污染整批梯度。"""

    @pytest.mark.parametrize("seq", [
        "A", "AC", "ACDEFGHIKLMNPQRSTVWY",
        "KWKLFKKIEK", "RRWQWR", "XXXXX", "BBBBB", "A" * 200,
    ])
    def test_all_properties_are_finite(self, seq):
        p = compute_properties(seq)
        for name in ("molecular_weight", "gravy", "net_charge_ph74",
                     "isoelectric_point", "aromaticity", "aliphatic_index",
                     "hydrophobic_moment", "aggregation_propensity"):
            v = getattr(p, name)
            assert math.isfinite(v), f"{seq}: {name} = {v} 不是有限值"

    def test_ambiguous_only_sequence_does_not_crash(self):
        p = compute_properties("XXXXXX")
        assert p.coverage == pytest.approx(0.0)
