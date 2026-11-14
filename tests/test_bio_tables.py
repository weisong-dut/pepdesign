"""氨基酸常数表的数值与完备性。

这些表的数值直接决定所有下游分数。测试同时锁定两件事：关键数值与文献一致，
以及每个"每残基一值"的表的键集合完整（缺键的 KeyError 只在数据恰好出现该残基时触发）。
"""

from __future__ import annotations

import pytest

from pepdesign.bio.aa import (
    ALLOWED_AA,
    AMBIGUOUS_AA,
    AROMATIC_AA,
    C_TERM_PKA,
    COIL_PROPENSITY,
    EISENBERG,
    HELIX_PROPENSITY,
    HYDROPHOBIC_AA,
    KYTE_DOOLITTLE,
    N_TERM_PKA,
    NEGATIVE_AA,
    ONE_TO_THREE,
    POLAR_AA,
    POSITIVE_AA,
    RESIDUE_MASS,
    SHEET_PROPENSITY,
    SIDE_CHAIN_CHARGE_PH7,
    SIDE_CHAIN_PKA,
    STANDARD_AA,
    THREE_TO_ONE,
    WATER_MASS,
)

# 所有"每个残基一个值"的表，都要覆盖完整的 ALLOWED_AA
PER_RESIDUE_TABLES = {
    "KYTE_DOOLITTLE": KYTE_DOOLITTLE,
    "EISENBERG": EISENBERG,
    "HELIX_PROPENSITY": HELIX_PROPENSITY,
    "SHEET_PROPENSITY": SHEET_PROPENSITY,
    "COIL_PROPENSITY": COIL_PROPENSITY,
    "SIDE_CHAIN_CHARGE_PH7": SIDE_CHAIN_CHARGE_PH7,
    "RESIDUE_MASS": RESIDUE_MASS,
}


class TestTableCompleteness:
    def test_standard_aa_is_the_canonical_20(self):
        assert len(STANDARD_AA) == 20
        assert set(STANDARD_AA) == set("ACDEFGHIKLMNPQRSTVWY")
        # 无重复
        assert len(set(STANDARD_AA)) == len(STANDARD_AA)

    @pytest.mark.parametrize("name", sorted(PER_RESIDUE_TABLES))
    def test_covers_every_allowed_residue(self, name):
        """表必须覆盖 ALLOWED_AA 全集；缺键的 KeyError 只在数据出现该残基时触发。"""
        table = PER_RESIDUE_TABLES[name]
        missing = set(ALLOWED_AA) - set(table)
        assert not missing, f"{name} 缺少以下残基: {sorted(missing)}"

    @pytest.mark.parametrize("name", sorted(PER_RESIDUE_TABLES))
    def test_no_unknown_keys(self, name):
        """多出的键通常是拼错的残基码，会静默地永远不被使用。"""
        table = PER_RESIDUE_TABLES[name]
        extra = set(table) - set(ALLOWED_AA)
        assert not extra, f"{name} 含非法键: {sorted(extra)}"

    def test_one_to_three_is_exactly_the_canonical_20(self):
        assert len(ONE_TO_THREE) == 20
        assert set(ONE_TO_THREE) == set(STANDARD_AA)

    def test_three_to_one_covers_standard_plus_modified(self):
        """`THREE_TO_ONE` 是超集：20 个标准码加修饰残基码；修饰残基映射回母体氨基酸。"""
        from pepdesign.bio.aa import ALLOWED_AA

        # 20 个标准码必须在，且映射正确
        for one, three in ONE_TO_THREE.items():
            assert THREE_TO_ONE[three] == one

        # 多出的码只能映射到合法字母（标准或模糊）
        for three, one in THREE_TO_ONE.items():
            assert one in ALLOWED_AA, f"{three} 映射到了非法字母 {one!r}"

    def test_modified_residues_map_to_their_parent(self):
        """修饰残基映射回母体氨基酸；删除它们会让含修饰的样本静默解析失败。"""
        assert THREE_TO_ONE["MSE"] == "M"   # 硒代甲硫氨酸
        assert THREE_TO_ONE["SEP"] == "S"   # 磷酸化丝氨酸
        assert THREE_TO_ONE["TPO"] == "T"   # 磷酸化苏氨酸
        assert THREE_TO_ONE["PTR"] == "Y"   # 磷酸化酪氨酸
        assert THREE_TO_ONE["SEC"] == "U"
        assert THREE_TO_ONE["PYL"] == "O"

    def test_multiple_codes_per_residue_are_intentional(self):
        """一个字母对应多个三字母码是有意的（修饰残基），且必须都指向同一个字母。"""
        from collections import defaultdict

        by_one: dict[str, list[str]] = defaultdict(list)
        for three, one in THREE_TO_ONE.items():
            by_one[one].append(three)
        for one, codes in by_one.items():
            assert one in ALLOWED_AA
            if len(codes) > 1:
                assert len(set(codes)) == len(codes), f"{one} 的三字母码有重复"

    def test_ambiguous_residues_are_not_standard(self):
        assert not (set(AMBIGUOUS_AA) & set(STANDARD_AA))
        assert set(AMBIGUOUS_AA) == {"X", "B", "Z", "U", "O"}

    def test_residue_classes_partition_consistently(self):
        """分类集合之间应当互斥，且全部落在标准字母表内。"""
        classes = {
            "POLAR": POLAR_AA, "HYDROPHOBIC": HYDROPHOBIC_AA,
            "AROMATIC": AROMATIC_AA, "POSITIVE": POSITIVE_AA,
            "NEGATIVE": NEGATIVE_AA,
        }
        for name, s in classes.items():
            assert s <= set(STANDARD_AA), f"{name} 含非标准残基: {sorted(s - set(STANDARD_AA))}"

        # 带电残基必须被 POLAR 覆盖（带电即极性）
        assert (POSITIVE_AA | NEGATIVE_AA) <= POLAR_AA
        # 正负电荷集合互斥
        assert not (POSITIVE_AA & NEGATIVE_AA)
        # 芳香残基都是疏水的（本项目的定义下）
        assert AROMATIC_AA <= HYDROPHOBIC_AA

    def test_glycine_and_proline_are_not_hydrophobic(self):
        """G/P 被刻意排除在 HYDROPHOBIC_AA 外；二者 KD 偏亲水，聚甘氨酸是常用柔性 linker。"""
        assert "G" not in HYDROPHOBIC_AA
        assert "P" not in HYDROPHOBIC_AA
        # 与 KD 标度保持一致：两个被排除的残基 KD 确实偏亲水
        assert KYTE_DOOLITTLE["G"] < 0 and KYTE_DOOLITTLE["P"] < 0


class TestKyteDoolittle:
    def test_anchors_match_the_published_scale(self):
        """KD 标度的几个标志性取值（Ile 最疏水、Arg 最亲水）。"""
        assert KYTE_DOOLITTLE["I"] == pytest.approx(4.5)
        assert KYTE_DOOLITTLE["V"] == pytest.approx(4.2)
        assert KYTE_DOOLITTLE["R"] == pytest.approx(-4.5)
        assert KYTE_DOOLITTLE["K"] == pytest.approx(-3.9)
        assert KYTE_DOOLITTLE["G"] == pytest.approx(-0.4)
        assert KYTE_DOOLITTLE["P"] == pytest.approx(-1.6)

    def test_ordering_of_hydrophobics(self):
        """疏水性排序是这套标度的核心信息。"""
        assert KYTE_DOOLITTLE["I"] > KYTE_DOOLITTLE["V"] > KYTE_DOOLITTLE["L"]
        assert KYTE_DOOLITTLE["F"] > KYTE_DOOLITTLE["A"]

    def test_charged_residues_are_hydrophilic(self):
        """带电残基的 KD 均为负；D/E/N/Q 在原始标度中共享 -3.5。"""
        for aa in "DEKR":
            assert KYTE_DOOLITTLE[aa] < 0, f"{aa} 带电却是疏水的"
        assert KYTE_DOOLITTLE["D"] == KYTE_DOOLITTLE["E"] == -3.5
        # K 是带电残基里最亲水的
        assert KYTE_DOOLITTLE["K"] < KYTE_DOOLITTLE["D"]

    def test_ambiguous_residues_are_neutral_or_inherited(self):
        """X 取中性 0；B/Z 取对应两种残基的公共值（B=N/D，Z=Q/E）。"""
        assert KYTE_DOOLITTLE["X"] == 0.0
        assert KYTE_DOOLITTLE["B"] == KYTE_DOOLITTLE["N"] == KYTE_DOOLITTLE["D"]
        assert KYTE_DOOLITTLE["Z"] == KYTE_DOOLITTLE["Q"] == KYTE_DOOLITTLE["E"]


class TestEisenberg:
    def test_anchors(self):
        assert EISENBERG["I"] == pytest.approx(1.38)
        assert EISENBERG["R"] == pytest.approx(-2.53)
        assert EISENBERG["X"] == 0.0

    def test_scale_differs_from_kd(self):
        """Eisenberg 与 KD 是两套标度；疏水矩必须使用 Eisenberg，不能混用。"""
        for aa in STANDARD_AA:
            assert EISENBERG[aa] != KYTE_DOOLITTLE[aa], f"{aa} 两套标度数值相同，可疑"

    def test_both_scales_rank_ile_most_hydrophobic(self):
        """两套标度在"最疏水残基"上应当一致。"""
        for table in (KYTE_DOOLITTLE, EISENBERG):
            std = {a: table[a] for a in STANDARD_AA}
            assert max(std, key=std.get) == "I"


class TestPropensities:
    def test_chou_fasman_is_not_pace_scholtz(self):
        """本表是 Chou & Fasman (1978) 的 Pα：Glu(1.51) 高于 Ala(1.42)，与 Pace & Scholtz 相反。"""
        assert HELIX_PROPENSITY["E"] > HELIX_PROPENSITY["A"]
        # Pace-Scholtz 的 Glu 会低于 0.5，本表不应如此
        assert HELIX_PROPENSITY["E"] > 1.0

    def test_helix_breakers_are_lowest(self):
        """Pro 和 Gly 是经典螺旋破坏者，应取到最小值附近。"""
        std = {a: HELIX_PROPENSITY[a] for a in STANDARD_AA}
        lowest_two = sorted(std, key=std.get)[:2]
        assert set(lowest_two) == {"P", "G"}
        assert std["P"] == pytest.approx(0.57)

    def test_sheet_anchors(self):
        assert SHEET_PROPENSITY["V"] == pytest.approx(1.70)
        assert SHEET_PROPENSITY["I"] == pytest.approx(1.60)
        # Glu 是最强的折叠破坏者
        std = {a: SHEET_PROPENSITY[a] for a in STANDARD_AA}
        assert min(std, key=std.get) == "E"

    def test_coil_anchors(self):
        assert COIL_PROPENSITY["G"] == pytest.approx(1.56)
        # Ile/Val 是最强的成结构残基（卷曲倾向最低）
        std = {a: COIL_PROPENSITY[a] for a in STANDARD_AA}
        assert min(std, key=std.get) == "I"

    def test_three_states_are_not_identical(self):
        """三个倾向表不能相同，否则 ss_confidence 恒为 0。"""
        for aa in STANDARD_AA:
            triple = (HELIX_PROPENSITY[aa], SHEET_PROPENSITY[aa], COIL_PROPENSITY[aa])
            assert len(set(triple)) > 1, f"{aa} 的三个倾向值完全相同"


class TestChargeAndPka:
    def test_charged_residues(self):
        assert SIDE_CHAIN_CHARGE_PH7["D"] == -1.0
        assert SIDE_CHAIN_CHARGE_PH7["E"] == -1.0
        assert SIDE_CHAIN_CHARGE_PH7["K"] == 1.0
        assert SIDE_CHAIN_CHARGE_PH7["R"] == 1.0

    def test_histidine_is_fractional_not_integer(self):
        """His 取 +0.1；pH 7.4 下仅约 3.8% 质子化，取 +1 会系统性高估正电荷。"""
        h = SIDE_CHAIN_CHARGE_PH7["H"]
        assert 0.0 < h < 1.0
        assert h == pytest.approx(0.1)

    def test_uncharged_residues_are_zero(self):
        for aa in "ACGNILMPQSTVWY":
            assert SIDE_CHAIN_CHARGE_PH7[aa] == 0.0

    def test_pka_values_match_reference(self):
        assert SIDE_CHAIN_PKA["D"] == pytest.approx(3.9)
        assert SIDE_CHAIN_PKA["E"] == pytest.approx(4.3)
        assert SIDE_CHAIN_PKA["H"] == pytest.approx(6.0)
        assert SIDE_CHAIN_PKA["C"] == pytest.approx(8.3)
        assert SIDE_CHAIN_PKA["K"] == pytest.approx(10.5)
        assert SIDE_CHAIN_PKA["R"] == pytest.approx(12.5)
        assert pytest.approx(9.6) == N_TERM_PKA
        assert pytest.approx(2.4) == C_TERM_PKA

    def test_pka_covers_every_charged_residue(self):
        """有非零 pH 7.4 电荷的残基都必须有侧链 pKa，否则等电点求解会漏项。"""
        for aa, q in SIDE_CHAIN_CHARGE_PH7.items():
            if aa in STANDARD_AA and q != 0.0:
                assert aa in SIDE_CHAIN_PKA, f"{aa} 带电但没有 pKa"

    def test_ordering_is_chemically_sensible(self):
        """Asp/Glu 的 pKa 必须远低于 7.4；Lys/Arg 必须远高于 7.4。"""
        assert SIDE_CHAIN_PKA["D"] < 7.4 and SIDE_CHAIN_PKA["E"] < 7.4
        assert SIDE_CHAIN_PKA["K"] > 7.4 and SIDE_CHAIN_PKA["R"] > 7.4


class TestResidueMass:
    def test_anchors(self):
        assert RESIDUE_MASS["G"] == pytest.approx(57.052)
        assert RESIDUE_MASS["W"] == pytest.approx(186.213)
        assert pytest.approx(18.015) == WATER_MASS

    def test_isoleucine_and_leucine_are_isomers(self):
        """Ile 与 Leu 是同分异构体，残基质量必须完全相同。"""
        assert RESIDUE_MASS["I"] == RESIDUE_MASS["L"]

    def test_glycine_is_lightest(self):
        std = {a: RESIDUE_MASS[a] for a in STANDARD_AA}
        assert min(std, key=std.get) == "G"
        assert max(std, key=std.get) == "W"

    def test_all_masses_are_physically_plausible(self):
        """残基质量应落在 [50, 250] Da。"""
        for aa in STANDARD_AA:
            m = RESIDUE_MASS[aa]
            assert 50.0 < m < 250.0, f"{aa} 的残基质量 {m} 超出合理范围"
