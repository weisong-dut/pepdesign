"""模型输出解析。

解析器是奖励链路的第一环，它的错误不会以异常的形式出现——
解析错了就是解析错了，会安静地把一条错误的序列送进奖励函数，
模型于是朝着一个错误的目标优化。所以这里的测试重点全在
"什么情况下不该被解析成序列"。

三个真实的失败模式（都曾经在代码里存在）：
  1. 三字母码转换把 `HIS` 腐蚀成 `H`；
  2. 英文散文被当成合法序列（氨基酸字母表几乎覆盖整个拉丁字母，
     所以"字母占比"这个判据根本没有判别力）；
  3. `AC-` 被当成乙酰化标注剥掉，丢掉真实的 Ala-Cys。
"""

from __future__ import annotations

import pytest

from pepdesign.prompts.parser import (
    ParsedSequence,
    extract_reasoning,
    extract_sequence,
    normalize_candidate,
    parse_stats,
)


class TestThreeLetterCodes:
    """三字母码 → 单字母的转换。

    转换本身是必要的（模型经常输出 `Lys-Arg-Trp`），
    但守卫条件曾经没有任何判别力：`if _THREE_LETTER_RE.search(s)`
    恰好就是 sub 会命中的同一条件，于是任何含三字母码子串的文本都会被转换。
    """

    @pytest.mark.parametrize("code", [
        "HIS", "ALA", "GLY", "TRP", "LYS", "ARG", "PRO", "SER",
    ])
    def test_standalone_three_letter_code_is_left_alone(self, code):
        """单独一个三字母码不做转换，保持原样。

        这是刻意的：`HIS` 既可能是"组氨酸"的三字母码，
        也可能就是 His-Ile-Ser 三个残基的全大写写法——
        在字符层面无法区分。转换需要证据（连字符分隔，或
        多个三字母码的覆盖率够高），单个码没有证据，所以不动它。

        旧实现的 bug 是无条件转换：`HIS` → `H`，
        把三个字母吃掉两个。当 `HIS` 出现在更长文本里时，
        它会和周围文本粘成一条虚构序列。
        """
        seq, _ = normalize_candidate(code)
        assert seq == code

    def test_hyphenated_three_letter_sequence_converts(self):
        """连字符分隔的三字母码是明确的序列信号，必须转换。"""
        seq, _ = normalize_candidate("Lys-Arg-Trp")
        assert seq == "KRW"

    def test_longer_hyphenated_sequence(self):
        seq, _ = normalize_candidate("Ala-Cys-Asp-Glu-Phe")
        assert seq == "ACDEF"

    def test_his_is_not_corrupted_inside_a_longer_sequence(self):
        """`HIS` 与其它残基混排时不能被吃掉。"""
        seq, _ = normalize_candidate("Ala-HIS-Gly")
        assert seq == "AHG"

    def test_english_words_are_not_converted_when_not_sequence_like(self):
        """散文里的 `met`/`his`/`Pro` 不该被拆成单字母插入突变。

        守卫条件现在是"三字母码覆盖率 ≥ 50%"或"存在连字符分隔证据"，
        普通的英文句子两条都不满足。
        """
        text = "The design was optimized by adding salt bridges to the Pro residue"
        seq, _ = normalize_candidate(text)
        # 不应变成一串稀疏的单字母（那才是被腐蚀的表现）
        assert "MET" not in seq.upper()


class TestAcetylPrefix:
    def test_ac_prefix_is_stripped_and_recorded(self):
        seq, mods = normalize_candidate("Ac-KWKLFKK")
        assert seq == "KWKLFKK"
        assert "n_acetyl" in mods

    def test_all_caps_ac_is_not_an_acetyl_marker(self):
        """`AC-KWKLFKK` 的 `AC` 是 Ala-Cys 两个残基。

        这条规则必须大小写敏感：`Ac-`（首字母大写）是乙酰化的标准写法，
        全大写 `AC-` 是氨基酸序列。混为一谈会丢掉真实残基，
        而且剩下的序列依然合法，validate() 检查不出来。
        """
        seq, mods = normalize_candidate("AC-KWKLFKK")
        assert "n_acetyl" not in mods
        assert seq == "ACKWKLFKK"

    def test_ac_without_hyphen_is_not_stripped(self):
        """没有连字符分隔就不算标注——否则 `ACKW` 会被误伤。"""
        seq, mods = normalize_candidate("ACKW")
        assert "n_acetyl" not in mods
        assert seq == "ACKW"

    def test_c_amide_subscript_two(self):
        """`-NH₂`（下标二）是 C 端酰胺化最标准的写法。

        这里曾经错写成 `-NH₃`（下标三，那是氨），导致最常见的写法
        反而匹配不上。
        """
        seq, mods = normalize_candidate("KWKLFKK-NH₂")
        assert seq == "KWKLFKK"
        assert "c_amide" in mods

    def test_c_amide_ascii(self):
        seq, mods = normalize_candidate("KWKLFKK-NH2")
        assert seq == "KWKLFKK"
        assert "c_amide" in mods

    def test_both_termini(self):
        seq, mods = normalize_candidate("Ac-KWKLFKK-NH2")
        assert seq == "KWKLFKK"
        assert "n_acetyl" in mods and "c_amide" in mods


class TestNoncanonicalResidues:
    def test_d_amino_acid_is_flagged(self):
        """`RGDfV` 的小写 `f` 是 D-Phe（西仑吉肽骨架）。

        大小写一 upper() 就变成 RGDFV——手性不同、活性完全不同的序列。
        """
        seq, mods = normalize_candidate("RGDfV")
        assert "noncanonical_residues" in mods
        assert seq == "RGDFV"

    def test_all_lowercase_is_not_flagged(self):
        """全小写只是"大小写不敏感地写了一条序列"，不是 D-型残基。"""
        seq, mods = normalize_candidate("rgdfv")
        assert "noncanonical_residues" not in mods
        assert seq == "RGDFV"

    def test_selenocysteine_flagged(self):
        _, mods = normalize_candidate("ACU")
        assert "noncanonical_residues" in mods


class TestPegRemoval:
    def test_peg_is_removed_by_normalize(self):
        seq, mods = normalize_candidate("KWKLFKKIEK-PEG4")
        assert seq == "KWKLFKKIEK"
        assert "pegylated" in mods

    def test_pegylated_sequence_survives_the_guards(self):
        """PEG 化的序列必须能走完整条解析链路，而不是被守卫拒收。

        这里有一个容易忽略的点：`_looks_like_sequence` 算"字母占比"时
        分母只剥了末端修饰（Ac/NH2），没剥 PEG，于是
        `KWKLFKKIEK-PEG4` 的分母把 PEG 的 3 个字母也算进去，
        比例 10/13 = 0.77 < 0.80 被判死——整条 PEG 化序列被拒收。
        而 PEG 化是多肽延长半衰期最常用的手段之一，
        `normalize_candidate` 里的 `_PEG_RE` 本就是为它写的，
        却因为守卫先拒了而根本走不到。
        """
        for raw in ("KWKLFKKIEK-PEG4", "KWKLFKKIEK-PEG", "KWKLFKKIEK-PEG2000"):
            r = extract_sequence(raw)
            assert r.ok, f"{raw} 被拒收了"
            assert r.sequence == "KWKLFKKIEK", raw
            assert "pegylated" in r.modifications, raw

    def test_peg_combined_with_terminal_modifications(self):
        r = extract_sequence("Ac-KWKLFKKIEK-PEG4")
        assert r.ok and r.sequence == "KWKLFKKIEK"
        assert "n_acetyl" in r.modifications and "pegylated" in r.modifications


class TestEnglishProseRejection:
    """氨基酸字母表覆盖了几乎整个拉丁字母，所以"字母占比高"
    完全无法区分散文和序列。有效判据是大小写形式和词形。"""

    @pytest.mark.parametrize("prose", [
        "The design was optimized by adding salt bridges",
        "This peptide shows improved thermostability",
        "We recommend a helix forming sequence here",
        "The model generated the following candidate",
    ])
    def test_prose_is_not_extracted(self, prose):
        """整段英文不能被当成序列。

        这里的关键认知是：比例判据没有判别力。
        "The design was optimized..." 的字母全是合法氨基酸码，
        比例 100%，旧实现会把它高高兴兴地收下。
        """
        r = extract_sequence(prose)
        # 要么解析失败，要么至少不是把整句当序列
        if r.ok:
            assert len(r.sequence) < len(prose.replace(" ", "")), (
                f"整段散文被当成了序列: {r.sequence!r}"
            )

    def test_prose_word_shape_rejected(self):
        r = extract_sequence("Thisisagreatdesign")
        # 单独一个长英文单词：大小写形式是混合的（T 大写其余小写）
        assert not r.ok or r.sequence != "THISISAGREATDESIGN"

    def test_normal_sequence_still_works(self):
        """不能因为收紧规则把正常序列也拒了。"""
        r = extract_sequence("KWKLFKKIEKV")
        assert r.ok
        assert r.sequence == "KWKLFKKIEKV"


class TestFallbackStrategies:
    """五级回退，每一级都要有测试命中，否则等于没有这级。"""

    def test_strategy_1_marker(self):
        text = "### 设计序列\nKWKLFKKIEK\n### 设计理由\n增强螺旋性\n### 结束"
        r = extract_sequence(text)
        assert r.ok and r.strategy == "marker"
        assert r.sequence == "KWKLFKKIEK"

    def test_strategy_2_xml_tag(self):
        text = "<answer>KWKLFKKIEK</answer>"
        r = extract_sequence(text)
        assert r.ok and r.strategy == "xml_tag"
        assert r.sequence == "KWKLFKKIEK"

    def test_strategy_3_code_fence(self):
        text = "Here is the design:\n```\nKWKLFKKIEK\n```"
        r = extract_sequence(text)
        assert r.ok and r.strategy == "code_fence"
        assert r.sequence == "KWKLFKKIEK"

    def test_strategy_4_line_scan(self):
        text = "I designed this peptide.\nKWKLFKKIEK\nIt should be stable."
        r = extract_sequence(text)
        assert r.ok and r.strategy == "line_scan"
        assert r.sequence == "KWKLFKKIEK"

    def test_strategy_5_whole_text(self):
        """全文兜底只在逐行扫描接不住的时候才轮得到。

        短肽（3-4 残基）正好是这种情况：整段文本就是一条短序列，
        但逐行扫描有 `_SEQ_LIKE_MIN_LEN = 5` 的硬门槛（3 个字母的
        英文短词无法在字符组成上与 3 残基肽区分），所以扫不出候选，
        最后由全文兜底按"字母占比 ≥ 90%"接住。
        """
        r = extract_sequence("KRW")
        assert r.ok and r.strategy == "whole_text"
        assert r.sequence == "KRW"

    def test_long_sequence_is_caught_by_line_scan_not_whole_text(self):
        """长序列在第 4 层就被接住了，不会流到全文兜底——
        换行/多词写法也归逐行扫描管（见 `_whole_text_looks_like_sequence`
        的说明）。"""
        r = extract_sequence("KWKLFKKIEK")
        assert r.ok and r.strategy == "line_scan"

    def test_marker_takes_priority_over_line_scan(self):
        text = "Some intro line with ACDEFG in it\n### 设计序列\nKWKLFKKIEK\n"
        r = extract_sequence(text)
        assert r.strategy == "marker"
        assert r.sequence == "KWKLFKKIEK"


class TestFailureReasons:
    def test_empty_output(self):
        r = extract_sequence("")
        assert not r.ok and r.reason == "empty_output"

    def test_whitespace_only(self):
        r = extract_sequence("   \n  \n")
        assert not r.ok

    def test_no_sequence_found(self):
        r = extract_sequence("I could not design a peptide for this target.")
        assert not r.ok
        assert r.reason in ("no_sequence_found", "too_short")

    def test_failed_result_has_empty_sequence(self):
        r = extract_sequence("no sequence here at all")
        assert r.sequence == ""
        assert r.strategy == "failed"


class TestCyclicLengthRelaxation:
    """环肽的下限是 4（常见 5-8 残基短环）。

    注意长度下限有两套，别混为一谈（见 `extract_sequence` 的文档）：
    强信号路径（标记/XML/代码块）只要求 `min_len`（默认 2），
    启发式路径另有 `_SEQ_LIKE_MIN_LEN = 5` 的硬门槛。
    `expect_cyclic` 影响的是前者传入的 `lo`。
    """

    def test_cyclic_uses_four_as_floor(self):
        r = extract_sequence("<answer>KWKF</answer>", expect_cyclic=True)
        assert r.ok and r.sequence == "KWKF"

    def test_cyclic_floor_is_four_not_three(self):
        """3 残基在环肽下限之下，应被拒。"""
        r = extract_sequence("<answer>KWK</answer>", expect_cyclic=True)
        assert not r.ok

    def test_explicit_min_len_five_is_the_documented_behavior(self):
        """调用方显式传 `min_len=5` 时，`expect_cyclic=True` 才是放宽到 4。

        这才是 `expect_cyclic` 的设计本意：线性肽要 ≥5，
        环肽可以短到 4。传默认的 2 时它反而是收紧，所以真要用它
        就该显式传 5。
        """
        assert not extract_sequence("<answer>KWKF</answer>", min_len=5).ok
        r = extract_sequence("<answer>KWKF</answer>", min_len=5, expect_cyclic=True)
        assert r.ok and r.sequence == "KWKF"


class TestMaxLengthConstant:
    def test_uses_module_constant_not_hardcoded_200(self):
        """`max_len` 默认值必须引用 `MAX_PEPTIDE_LEN` 常量。

        以前这里手写 200，而 `min_len` 引用常量——两者不同步时
        会在边界上出现"短的一端改了、长的一端没改"的静默不一致。
        """
        from pepdesign.bio.sequence import MAX_PEPTIDE_LEN

        long_seq = "A" * (MAX_PEPTIDE_LEN + 1)
        r = extract_sequence(long_seq)
        assert not r.ok


class TestExtractReasoning:
    def test_extracts_reasoning_section(self):
        text = "### 设计序列\nKWKLFKKIEK\n### 设计理由\n增强了螺旋性\n### 结束"
        assert "螺旋" in extract_reasoning(text)

    def test_missing_reasoning_returns_empty(self):
        assert extract_reasoning("### 设计序列\nKWKLFKKIEK\n") == ""


class TestParsedSequenceContract:
    def test_failed_result_has_empty_sequence(self):
        """ok=False 时 sequence 必须是空串——调用方依赖这条约定
        来判断"这条样本没有可用序列"。"""
        r = ParsedSequence(ok=False, reason="x")
        assert r.sequence == ""

    def test_as_dict_has_length(self):
        r = ParsedSequence(ok=True, sequence="KWKLFKK", strategy="marker")
        d = r.as_dict()
        assert d["length"] == 7
        assert d["ok"] is True


class TestParseStats:
    def test_counts_by_strategy(self):
        results = [
            extract_sequence("### 设计序列\nKWKLFKKIEK\n"),
            extract_sequence("KWKLFKKIEK"),
            extract_sequence("no sequence at all here"),
        ]
        stats = parse_stats(results)
        assert stats["total"] == 3
        assert stats["ok"] == 2
        assert stats["fail_rate"] == pytest.approx(1 / 3, abs=1e-4)  # 统计值保留 4 位小数
        # 失败样本按 strategy="failed" 与失败原因分别归档
        assert stats["strategy_distribution"]["failed"] == 1
        assert stats["fail_reasons"]

    def test_empty_list_returns_empty_dict(self):
        assert parse_stats([]) == {}
