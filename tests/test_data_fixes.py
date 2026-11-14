"""数据修复的回归测试。

覆盖已经出过问题的不变量：合成变体保持长度与 Cys 位置、变体降权、
de_novo 检索满足先采样约束、`break_amphipathic` 确实降低疏水矩、
修复理由写出实际位点。
"""
from __future__ import annotations

import random
from dataclasses import replace

from pepdesign.bio.sequence import hydrophobic_moment_profile
from pepdesign.data.build_sft import (
    build_denovo_index,
    conservative_variant,
    cyclic_balanced_pool,
    degrade_sequence,
    describe_repair,
    search_denovo_candidate,
)
from pepdesign.data.schema import PeptideRecord


def _rec(seq: str, *, cyclic: str = "none", source: str = "uniprot", sid: str = "x"):
    rec = PeptideRecord(
        sequence=seq, name="t", source=source, source_id=sid,
        function="other", cyclic_type=cyclic,
    )
    rec.compute_properties()
    return rec


#: 在 `feasible_cyclic_type` 下确实可行的二硫键环肽
CYCLIC_SEQ = "IIEKGEEHCGPLIEAHKECMRALGFKV"
LINEAR_SEQ = "FISAIASMLGKFL"


def test_conservative_variant_keeps_length_and_cys():
    for seed in range(30):
        variant = conservative_variant(CYCLIC_SEQ, random.Random(seed))
        if variant is None:
            continue
        assert len(variant) == len(CYCLIC_SEQ)
        assert [i for i, c in enumerate(variant) if c == "C"] == [
            i for i, c in enumerate(CYCLIC_SEQ) if c == "C"
        ]
        # 只改一个残基，且确实改了
        assert sum(1 for a, b in zip(variant, CYCLIC_SEQ, strict=True) if a != b) == 1
        assert variant != CYCLIC_SEQ


def test_cyclic_pool_downweights_synthetic_variants():
    real = _rec(CYCLIC_SEQ, cyclic="disulfide", sid="real")
    syn = replace(real, source="synthetic", source_id="syn")
    linear = [_rec("ACDEFGHIKLMNPQRSTVWY", sid=f"l{i}") for i in range(60)]

    pool = cyclic_balanced_pool(
        [real, syn, *linear], cyclic_ratio=0.3, variant_weight=0.2
    )
    n_real = sum(1 for r in pool if r.source_id == "real")
    n_syn = sum(1 for r in pool if r.source_id == "syn")

    assert n_real >= 1
    # 变体副本数应明显少于真实环肽（variant_weight=0.2 时约 1/5）
    assert n_syn < n_real
    cyclic_share = (n_real + n_syn) / len(pool)
    assert 0.2 <= cyclic_share <= 0.45


def test_search_denovo_candidate_respects_sampled_constraints():
    records = [
        _rec("ACDEFGHIKLMNPQRSTVWY", sid=f"l{i}") for i in range(12)
    ]
    records.append(_rec("ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMN", sid="long"))
    index = build_denovo_index(records)

    rng = random.Random(7)
    for _ in range(30):
        found = search_denovo_candidate(index, rng, max_attempts=50)
        assert found is not None
        rec, cyclic, (lo, hi), (q_lo, q_hi) = found
        assert lo <= rec.length <= hi
        q = rec.properties["net_charge_ph74"]
        assert q_lo - 1e-9 <= q <= q_hi + 1e-9
        assert cyclic in ("none", "disulfide")


def test_break_amphipathic_actually_reduces_moment():
    for seed in range(20):
        result = degrade_sequence(
            LINEAR_SEQ, random.Random(seed), strategy="break_amphipathic"
        )
        if result is None:
            continue
        bad, strategy = result
        assert strategy == "break_amphipathic"
        window = min(9, len(LINEAR_SEQ))
        assert max(hydrophobic_moment_profile(bad, window=window)) < max(
            hydrophobic_moment_profile(LINEAR_SEQ, window=window)
        ) - 1e-9


def test_describe_repair_names_the_real_position():
    text = describe_repair("ACDEF", "ACWEF", "break_amphipathic")
    assert "第 3 位" in text
    assert "D→W" in text


def test_repair_reasoning_contains_actual_site():
    from pepdesign.data.build_sft import generate_repair_reasoning

    good = _rec(CYCLIC_SEQ, cyclic="disulfide", sid="g")
    bad = good.sequence[:5] + "A" + good.sequence[6:]
    reasoning = generate_repair_reasoning(good, bad, "break_amphipathic")
    assert "第" in reasoning and "位" in reasoning
