"""清洗阶段的分流规则。

`clean.py` 是数据进入训练语料前的最后一道闸；每条规则决定样本的丢弃原因，
该原因会进入 `quality_report`，因此判据与输出名必须稳定。
重点覆盖 `noncanonical_residue`：它会静默改错标签，而不是挡住坏数据。
"""

from __future__ import annotations

import pytest

from pepdesign.data.clean import clean_record, run_cleaning
from pepdesign.data.schema import PeptideRecord
from pepdesign.utils.config import DataConfig


@pytest.fixture(scope="module")
def cfg() -> DataConfig:
    return DataConfig()


def reason_of(sequence: str, cfg: DataConfig, **kw) -> str | None:
    """清洗一条序列，返回丢弃原因（None = 被保留）。"""
    return clean_record(PeptideRecord(sequence=sequence, source="uniprot", **kw), cfg).reason


class TestNoncanonicalResidues:
    """小写字母 / U / O 代表的残基在单字母表里无法表达，必须丢弃。"""

    def test_d_amino_acid_is_dropped_not_silently_capitalized(self, cfg):
        """`RGDfV` 必须被丢弃；upper() 会把手性不同的分子变成合法序列。"""
        assert reason_of("RGDfV", cfg) == "noncanonical_residue"

    def test_selenocysteine_and_hydroxyproline_are_dropped(self, cfg):
        """U = 硒代半胱氨酸、O = 羟脯氨酸，都不是 20 种标准残基。"""
        assert reason_of("ACDUEFGHIK", cfg) == "noncanonical_residue"
        assert reason_of("ACDOEFGHIK", cfg) == "noncanonical_residue"

    def test_all_uppercase_is_kept(self, cfg):
        """全大写序列应保留。"""
        assert reason_of("KWKLFKKIEK", cfg) is None

    def test_acetyl_prefix_is_not_mistaken_for_d_residue(self, cfg):
        """`Ac-` 是修饰标注而非小写 C；检测小写必须发生在剥离修饰之后。"""
        r = clean_record(PeptideRecord(sequence="Ac-KWKLFKKIEK"), cfg)
        assert r.reason is None
        assert r.record is not None
        assert "n_acetyl" in r.record.modifications
        assert r.record.has_noncanonical is False
        # 修饰标注不进序列本体
        assert r.record.sequence == "KWKLFKKIEK"


class TestOtherDropReasons:
    def test_too_short(self, cfg):
        assert reason_of("A", cfg) == f"too_short(<{cfg.min_length})"

    def test_too_long(self, cfg):
        assert reason_of("A" * (cfg.max_length + 1), cfg).startswith("too_long")

    def test_low_complexity(self, cfg):
        """同残基重复序列作为 SFT 目标会教出退化解。"""
        assert reason_of("R" * 12, cfg) == "low_complexity"

    def test_low_coverage(self, cfg):
        """含 X 比例过低的序列不进入语料。"""
        reason = reason_of("KWKXXXXXX" + "X" * 10, cfg)
        assert reason is not None and reason.startswith("low_coverage")

    def test_clean_sequence_survives(self, cfg):
        r = clean_record(PeptideRecord(sequence="KWKLFKKIEK"), cfg)
        assert r.reason is None
        assert r.record is not None
        assert r.record.sequence == "KWKLFKKIEK"

    def test_properties_are_snapshotted_on_kept_records(self, cfg):
        """保留下来的记录要带一份性质快照，供后续按性质分层采样。"""
        r = clean_record(PeptideRecord(sequence="KWKLFKKIEK"), cfg)
        assert r.record is not None
        assert r.record.properties is not None


class TestDropReasonAccounting:
    def test_every_dropped_record_lands_in_the_report(self, cfg):
        """所有丢弃原因都进报告；统计时去掉 `reason` 中的参数部分，避免同类原因被拆分。"""
        records = [
            PeptideRecord(sequence="KWKLFKKIEK"),            # 保留
            PeptideRecord(sequence="RGDfV"),                 # 非天然
            PeptideRecord(sequence="RGDfV" + "ACDEFG"),      # 非天然（同因）
            PeptideRecord(sequence="A"),                     # too_short
            PeptideRecord(sequence="R" * 12),                # low_complexity
        ]
        kept, report = run_cleaning(records, cfg)
        assert len(kept) == 1
        # 两个非天然残基的记录合并成一个计数键，而不是各记一条
        assert report.drop_reasons["noncanonical_residue"] == 2
        assert report.drop_reasons["too_short"] == 1
        assert report.drop_reasons["low_complexity"] == 1
        assert report.raw_count == 5 and report.kept_count == 1


class TestUncleavedPrecursor:
    """成熟肽不带起始甲硫氨酸；以 M 开头且无切割记录说明它仍是翻译产物。

    被丢弃的序列本身合法，但会引入前体蛋白的分布，而推理输入不带信号肽。
    """

    def test_translation_product_is_dropped(self, cfg):
        """以 M 起始且没有切割记录 → uncleaved_precursor。"""
        seq = "MGNNEEYFIDVNDLSIDVFDVVEQGGAVTALTADHG"
        assert reason_of(seq, cfg) == "uncleaved_precursor"

    def test_cut_precursor_is_kept_even_if_it_starts_with_m(self, cfg):
        """判据是 `lineage.chain_cut`，而非开头是否为 M。"""
        # 避免使用含 `U` / `O` 的序列：它们会先被 `noncanonical_residue` 拦掉。
        reason = reason_of(
            "MKWLFKKIEK", cfg, lineage={"chain_cut": True, "precursor_len": 40}
        )
        assert reason is None

    def test_mature_peptide_untouched(self, cfg):
        """不带 M 的正常序列不受影响。"""
        assert reason_of("KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK", cfg) is None

    def test_reason_shows_up_in_the_quality_report(self, cfg):
        """uncleaved_precursor 必须出现在 quality_report 中。"""
        records = [
            PeptideRecord(sequence="KWKLFKKIEK"),                      # 保留
            PeptideRecord(sequence="MGNNEEYFIDVNDLSIDVFDVVEQGGAVTALTA"),  # 前体
            PeptideRecord(sequence="MQLNLGAIFLALLGVMATATSVLAEPHRHQGPIFD"),  # 前体
        ]
        kept, report = run_cleaning(records, cfg)
        assert len(kept) == 1
        assert report.drop_reasons["uncleaved_precursor"] == 2


class TestProteinFragments:
    """大蛋白的结构域片段不是多肽，必须丢弃。

    这类片段的长度与字符均合规，只能靠名称识别。
    """

    def test_envelope_glycoprotein_fragment_is_dropped(self, cfg):
        """HIV gp120 的 V3 环等包膜糖蛋白片段应判为 protein_fragment。"""
        reason = reason_of(
            "CTRPNNNTRKGIHIGPGSTFFATGEIIGDIRQAHCNISKA", cfg,
            name="Envelope glycoprotein",
        )
        assert reason == "protein_fragment"

    def test_flagellin_and_von_willebrand_are_dropped(self, cfg):
        assert reason_of("ADFAAETANLSRSQILQQAGTAMVAQANQLPQSVLKLLQ", cfg,
                         name="Flagellin FliC") == "protein_fragment"
        assert reason_of("APVSVTTPYVEDIPEPPLHDLYC", cfg,
                         name="von Willebrand factor") == "protein_fragment"

    def test_matching_is_case_insensitive(self, cfg):
        """UniProt 名称大小写不统一，匹配需大小写不敏感。"""
        assert reason_of("APVSVTTPYVEDIPEPPLHDLYC", cfg,
                         name="VON WILLEBRAND FACTOR") == "protein_fragment"

    def test_real_peptides_are_kept(self, cfg):
        """黑名单只针对大蛋白名；fibrinopeptide、trypsin inhibitor、defensin 属于真实肽。"""
        assert reason_of("ADSGEGDVSGEDADEGDGDG", cfg,
                         name="Beta-defensin 1") is None
        assert reason_of("RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA",
                         cfg, name="Pancreatic secretory trypsin inhibitor") is None
        assert reason_of("ADSGEGDFLAEGGGVR", cfg,
                         name="Fibrinopeptide A") is None

    def test_unnamed_records_are_not_dropped(self, cfg):
        """`name` 缺省为空串的记录不能被误伤。"""
        assert reason_of("KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK", cfg) is None

    def test_reason_shows_up_in_the_quality_report(self, cfg):
        """protein_fragment 的丢弃量必须出现在报告中。"""
        records = [
            PeptideRecord(sequence="KWKLFKKIEK", name="Magainin-2"),
            PeptideRecord(sequence="CTRPNNNTRKGIHIGPGSTFFATGEIIGDIRQAHCNISKA",
                          name="Envelope glycoprotein"),
            PeptideRecord(sequence="ADFAAETANLSRSQILQQAGTAMVAQANQLPQSVLKLLQ",
                          name="Flagellin FliC"),
        ]
        kept, report = run_cleaning(records, cfg)
        assert len(kept) == 1
        assert report.drop_reasons["protein_fragment"] == 2


class TestUnknownResidueTolerance:
    """含 X/B 的序列零容忍。

    X 表示该位置残基未知，无法用于多肽合成；对 5-60 残基语料，容忍度应为 0。
    """

    def test_x_is_rejected_even_at_one_residue_in_twenty(self, cfg):
        """23 残基中 2 个 X = 91.3%，阈值 0.9 时会被放过，实际必须拒绝。"""
        seq = "APVSVTTPYXEDIPXPPLHDLYC"          # 23 残基、2 个 X = 91.3%
        assert seq.count("X") == 2
        assert reason_of(seq, cfg).startswith("low_coverage")

    def test_b_is_rejected_too(self, cfg):
        """B = Asx，同样是不确定的残基。"""
        assert reason_of("ACDBEFGHIKLMNPQRSTVWY", cfg).startswith("low_coverage")

    def test_pure_standard_sequence_is_kept(self, cfg):
        assert reason_of("KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK", cfg) is None

    def test_the_tolerance_is_configurable(self, cfg):
        """阈值可配置；1.0 不应硬编码进 clean.py。"""
        seq = "APVSVTTPYXEDIPXPPLHDLYC"
        lenient = DataConfig(min_standard_ratio=0.9)
        assert reason_of(seq, lenient) is None
        assert reason_of(seq, cfg).startswith("low_coverage")
