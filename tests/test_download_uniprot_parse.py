"""UniProt 条目解析的纯逻辑测试。不需要网络——喂的是构造好的 JSON。

单独钉住的原因
------------------------
这三个函数各自修掉了一个"采回来的数据静默变差"的问题，而它们
都不会报错——只会让下游拿到一批看起来正常、其实分布错了的序列：

  * `_mature_sequence` 不切前体 → 40% 的训练序列带着信号肽；
  * `_classify_function` 顺序写反 → 抗真菌类别变成 0 条（数据还在，标签没了）；
  * `_location_range` 没兜住 `end: null` → 整个采集在 `int(None)` 上崩。

前两个是"数据错了但日志全绿"，第三个是"采到一半崩"。共同点是：
只有抽查数据本身才能发现。所以这里用构造的条目把它们钉死。
"""

from __future__ import annotations

import pytest

from pepdesign.data.download import (
    _classify_function,
    _location_range,
    _mature_sequence,
    _parse_uniprot_entry,
)


def _entry(seq: str, features: list[dict] | None = None, keywords: list[str] | None = None) -> dict:
    """造一个最小的 UniProt 条目（只带解析用得上的字段）。"""
    return {
        "primaryAccession": "TEST0001",
        "proteinDescription": {
            "recommendedName": {"fullName": {"value": "Test peptide"}}
        },
        "sequence": {"value": seq, "length": len(seq)},
        "organism": {"scientificName": "Test organism"},
        "keywords": [{"name": k} for k in (keywords or [])],
        "features": features or [],
    }


def _loc(start: int, end: int | None = None) -> dict:
    return {
        "location": {
            "start": {"value": start},
            "end": {"value": end} if end is not None else None,
        }
    }


# ---------------------------------------------------------------------------
# _location_range：`end` 为 null 的情况
# ---------------------------------------------------------------------------
def test_location_range_handles_null_end():
    """`"end": null` 要当成单残基，而不是崩在 `int(None)`。

    UniProt 对修饰残基这类单点 feature 会返回 `"end": null`。
    原来的写法是 `.get("end", {}).get("value", start)`——它只兜住了
    "键不存在"，兜不住"键在但值是 None"，于是整个采集抛
    `TypeError: int() argument must be ... not 'NoneType'`。
    """
    assert _location_range(_loc(5)) == (5, 5)
    assert _location_range(_loc(5, 9)) == (5, 9)


def test_location_range_returns_none_without_start():
    assert _location_range({"location": {}}) is None
    assert _location_range({}) is None


# ---------------------------------------------------------------------------
# _mature_sequence：切前体
# ---------------------------------------------------------------------------
def test_cuts_mature_peptide_by_chain_feature():
    """有 Chain feature 时按它切，切出来的就是成熟肽。"""
    full = "MSIGNALPEPTIDE" + "MATUREPEPTIDE"
    entry = _entry(full, features=[{"type": "Chain", "description": "Mature",
                                    **_loc(15, len(full))}])
    seq, info = _mature_sequence(entry, full)
    assert seq == "MATUREPEPTIDE"
    assert info["chain_cut"] is True
    assert info["cut_by"] == "chain"


def test_falls_back_to_signal_when_chain_is_missing():
    """没有 Chain 时用 Signal 的末尾切——这条是必须的，不是兜底。

    实测 2,041 条前体只标了 Signal、没标 Chain（Histatin-1 就是）。
    没有这条回退，它们会整条原样留下，前 19 位是信号肽。
    """
    full = "MKFFVFALVLALMISMISA" + "DSHEKRHHGYRRKFHEKHHSHREFPFY"
    entry = _entry(full, features=[{"type": "Signal", **_loc(1, 19)}])
    seq, info = _mature_sequence(entry, full)
    assert seq == "DSHEKRHHGYRRKFHEKHHSHREFPFY"
    assert info["cut_by"] == "signal_or_propeptide"


def test_propeptide_is_cut_together_with_signal():
    """信号肽 + 前导肽都要切掉，取最远端当起点。

    只切 Signal 的话，带前导肽的条目会残留一段前导序列。
    """
    full = "MSIG" + "PROPEP" + "MATURE"
    entry = _entry(full, features=[
        {"type": "Signal", **_loc(1, 4)},
        {"type": "Propeptide", **_loc(5, 10)},
    ])
    seq, _ = _mature_sequence(entry, full)
    assert seq == "MATURE"


def test_no_cut_when_chain_covers_the_whole_sequence():
    """Chain 覆盖全长时不算切割——很多成熟肽条目本身就整条都是 Chain。"""
    seq0 = "KWKLFKKIEK"
    entry = _entry(seq0, features=[{"type": "Chain", **_loc(1, len(seq0))}])
    seq, info = _mature_sequence(entry, seq0)
    assert seq == seq0
    assert info["chain_cut"] is False


def test_no_cut_when_there_is_nothing_to_cut():
    """既没 Chain 也没 Signal 时原样返回，并说明原因。

    这一类实测有 1,957 条（细菌素这类条目 UniProt 注释不全），
    它们由清洗层的 `uncleaved_precursor` 规则处理，不在这里丢。
    """
    seq0 = "MGNNEEYFIDVNDLSIDVFDVVEQGGAVTALTADHG"
    entry = _entry(seq0)
    seq, info = _mature_sequence(entry, seq0)
    assert seq == seq0
    assert info["chain_cut"] is False
    assert info["reason"] == "no_chain_no_signal"


def test_picks_the_longest_chain():
    """多段 Chain 时取最长的——前体被切成几段成熟肽的情况存在。"""
    full = "AAAA" + "BBBBBBBB" + "CCCCCC"
    entry = _entry(full, features=[
        {"type": "Chain", **_loc(1, 4)},
        {"type": "Chain", **_loc(5, 12)},
        {"type": "Chain", **_loc(13, len(full))},
    ])
    seq, info = _mature_sequence(entry, full)
    assert seq == "BBBBBBBB"
    assert info["chain"] == [5, 12]


# ---------------------------------------------------------------------------
# _classify_function：优先级顺序
# ---------------------------------------------------------------------------
def test_antifungal_wins_over_antibacterial():
    """这条钉的是一个真实 bug：抗真菌类别曾经是 0 条。

    原来的实现把 `Antimicrobial` 放在 `if` 的第一位，而宽谱抗菌肽
    几乎都带这个关键词，于是抗真菌 / 抗病毒分支永远轮不到。
    实测 583 条带 Fungicide 的条目全被标成了 antibacterial。

    正确行为是"最特异的先判"：一个既抗菌又抗真菌的肽，标 antifungal
    信息量更大（抗菌已经是默认类别了）。
    """
    assert _classify_function(["Antimicrobial", "Fungicide"], "other") == "antifungal"
    assert _classify_function(["Antibiotic", "Antifungal"], "other") == "antifungal"


def test_antiviral_wins_over_antibacterial():
    assert _classify_function(["Antimicrobial", "Antiviral protein"], "other") == "antiviral"


def test_falls_back_to_the_query_function():
    """认不出来时用查询给的粗分类，不擅自改成 other。"""
    assert _classify_function([], "cell_penetrating") == "cell_penetrating"
    assert _classify_function(["Some other keyword"], "antitumor") == "antitumor"


# ---------------------------------------------------------------------------
# _parse_uniprot_entry：端到端
# ---------------------------------------------------------------------------
def test_parse_records_the_precursor_length_in_lineage():
    """切过之后要把前体长度记下来，否则事后无法审计切得对不对。"""
    full = "MNFSKIFIFVVLAVLLLCSQTE" + "GRLKKLGKKIEGAGKRVFKAAEKALPVVAGVKALG"
    entry = _entry(full, features=[{"type": "Chain", **_loc(23, len(full))}])
    rec = _parse_uniprot_entry(entry, "antibacterial")
    assert rec is not None
    assert len(rec.sequence) < len(full)
    assert rec.lineage["precursor_len"] == len(full)
    assert rec.lineage["chain_cut"] is True


def test_parse_collects_modifications_without_flagging_noncanonical():
    """修饰只记录、不置 `has_noncanonical`。

    UniProt 的 `sequence.value` 只用标准字母，磷酸化/酰胺化是注释，
    不改序列字符。曾经把两者绑定，后果是清洗层把 903 条正常序列
    当成"含非天然残基"丢掉了。
    """
    seq = "DSHEKRHHGYRRKFHEKHHSHREFPFY"
    entry = _entry(seq, features=[
        {"type": "Modified residue", "description": "Phosphoserine", **_loc(2)},
        {"type": "Modified residue", "description": "Phosphoserine", **_loc(9)},
        {"type": "Modified residue", "description": "Sulfotyrosine", **_loc(11)},
    ])
    rec = _parse_uniprot_entry(entry, "other")
    assert rec is not None
    assert rec.modifications == ["Phosphoserine", "Sulfotyrosine"]  # 去重且保序
    assert rec.has_noncanonical is False


def test_parse_returns_none_for_empty_sequence():
    assert _parse_uniprot_entry(_entry(""), "other") is None


@pytest.mark.parametrize("hint,expect", [("disulfide", "disulfide"), ("none", "none")])
def test_parse_keeps_the_query_level_cyclic_hint(hint, expect):
    """查询给的环化提示优先——它来自关键词，比序列里数 Cys 可靠。"""
    rec = _parse_uniprot_entry(_entry("KWKLFKKIEK"), "antibacterial", hint)
    assert rec is not None
    assert rec.cyclic_type == expect


# ---------------------------------------------------------------------------
# 采集查询的"产出契约"
# ---------------------------------------------------------------------------
class TestQueryContracts:
    """每条 UniProt 查询都要声明"我捞回来的东西应该长什么样"，并被执行。

    这组测试钉的是一个机制，不是某一条查询。背景：曾出现两次
    "关键词和目标语义不对齐"且均为静默失败——

      * `keyword:"Antineoplastic"` 实测命中 0 条，查询实际只剩另一半
        在起作用，捞回来的是蛇毒细胞毒素，却被标成"抗肿瘤"；
      * `keyword:"Antibiotic"` 是子串匹配，连 `Antibiotic resistance`
        一起返回，590 条耐药蛋白（外排泵、核糖体蛋白）混进语料。

    两次都是靠人工抽查记录内容才发现的。契约机制把这件事变成自动的：
    没命中声明关键词的记录直接丢弃 + 报警，比例进 manifest。
    """

    def test_every_query_declares_require_any(self):
        """没有契约的查询就是下一个 Antineoplastic。

        查询返回几千条、名字看着也对、函数不报错——只有抽查才知道
        它名不副实。声明 `require_any` 是让机器替人去查。
        """
        from pepdesign.data.download import UNIPROT_QUERIES

        for spec in UNIPROT_QUERIES:
            assert spec.get("require_any"), (
                f"查询 {spec['name']} 没有声明 require_any——"
                f"它捞回来的东西是否符合类目名，就没有任何自动校验。"
            )

    def test_dead_keywords_are_not_used(self):
        """`Antineoplastic` 与 `Cyclotide` 在 UniProt 里命中 0 条，不许再用。

        这条测试的价值在于把实测结论固化下来：这两个关键词看起来
        非常合理（一个像"抗肿瘤"、一个像"环肽"），任何人重写查询时
        都会想用它们。写死在这里，配上一句"实测命中 0 条"，
        可以避免同类问题再次出现。
        """
        from pepdesign.data.download import _FUNCTION_BY_KEYWORD, UNIPROT_QUERIES

        blob = " ".join(s["query"] for s in UNIPROT_QUERIES)
        assert "Antineoplastic" not in blob
        assert "Cyclotide" not in blob
        assert "Antineoplastic" not in {k for k, _ in _FUNCTION_BY_KEYWORD}

    def test_toxins_are_excluded_globally(self):
        """毒类关键词必须在排除表里：目标是药物候选，毒素不属于候选。

        实测依据：蛇毒细胞毒素 48 条全部带 `Cardiotoxin` 关键词；
        二硫键查询捞回来的 1,945 条里有 1,066 条是毒素（55%）。
        """
        from pepdesign.data.download import UNIPROT_EXCLUDE_KEYWORDS

        for kw in ("Toxin", "Cardiotoxin", "Neurotoxin"):
            assert kw in UNIPROT_EXCLUDE_KEYWORDS, f"{kw} 没被排除"

    def test_no_antitumor_query_because_uniprot_has_no_usable_keyword(self):
        """当前刻意没有"抗肿瘤肽"查询——UniProt 没有可用的关键词。

        与其留一个名不副实的类别（把蛇毒毒素标成抗肿瘤），不如不声明这个能力。
        `FunctionType` 里的 `antitumor` 保留（下游 prompt 映射还在），
        但当前没有采集源产出它。
        """
        from pepdesign.data.download import UNIPROT_QUERIES

        assert "antitumor" not in {s["function"] for s in UNIPROT_QUERIES}

    def test_no_cell_penetrating_query_because_the_keywords_are_wrong(self):
        """当前刻意没有"细胞穿膜肽"查询——曾经那条查询的关键词是错的。

        用过的关键词是 `Cell projection` / `Cell adhesion`，它们标的是
        细胞突起与黏附相关蛋白，跟穿膜肽没有任何关系。捞回来的是
        56 条大蛋白的结构域片段（Flagellin、TMEM17、von Willebrand factor）。

        这条测试的价值在于：那两个关键词看起来非常合理——"细胞突起"
        "细胞黏附"都带"细胞"，重写查询的人很容易再想用它。
        写死在这里，可以避免同类问题再次出现。

        这条同时也是在钉契约机制的边界：`require_any` 拦不住这一类。
        它验证"记录确实带这个关键词"（确实带了），
        但不验证"这个关键词真的等于类目名"。假关键词（命中 0 条的
        `Antineoplastic`）它抓得住，真关键词配错类目它抓不住。
        """
        from pepdesign.data.download import UNIPROT_QUERIES

        assert "cell_penetrating" not in {s["function"] for s in UNIPROT_QUERIES}
        blob = " ".join(s["query"] for s in UNIPROT_QUERIES)
        assert "Cell projection" not in blob
        assert "Cell adhesion" not in blob


class TestCheckContract:
    """契约校验的报警逻辑。"""

    def test_warns_when_coverage_is_low(self, caplog):
        """契约达成率低 → 报警，指出查询可能名不副实。"""
        import logging

        from pepdesign.data.download import _check_contract

        with caplog.at_level(logging.WARNING):
            _check_contract(
                "bad_query",
                {"query": "keyword:\"Nope\"", "require_any": ["Nope"]},
                {"kept": 10, "contract_miss": 990, "excluded": {}},
            )
        assert any("契约达成率" in r.message for r in caplog.records)

    def test_no_warning_when_coverage_is_good(self, caplog):
        """达成率高就不该刷警告——否则真出问题时没人看。"""
        import logging

        from pepdesign.data.download import _check_contract

        with caplog.at_level(logging.WARNING):
            _check_contract(
                "good_query",
                {"query": "x", "require_any": ["X"]},
                {"kept": 1000, "contract_miss": 10, "excluded": {}},
            )
        assert not any("契约达成率" in r.message for r in caplog.records)

    def test_reports_nothing_collected_as_an_error(self, caplog):
        """一条都没采到 = 查询是坏的，要报 ERROR 而不是静默返回空。"""
        import logging

        from pepdesign.data.download import _check_contract

        with caplog.at_level(logging.ERROR):
            _check_contract(
                "dead_query",
                {"query": "keyword:\"Antineoplastic\"", "require_any": ["Antineoplastic"]},
                {"kept": 0, "contract_miss": 0, "excluded": {}},
            )
        assert any("一条都没采到" in r.message for r in caplog.records)

    def test_logs_what_was_excluded(self, caplog):
        """排除了什么要报出来——静默排除等于数据凭空少了。"""
        import logging

        from pepdesign.data.download import _check_contract

        with caplog.at_level(logging.INFO):
            _check_contract(
                "q",
                {"query": "x", "require_any": ["X"]},
                {"kept": 100, "contract_miss": 0, "excluded": {"Toxin": 42, "Cardiotoxin": 8}},
            )
        assert any("Toxin" in r.message and "50" in r.message for r in caplog.records)
