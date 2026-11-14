"""数据清洗、过滤与近似去重。

流水线顺序：canonicalize → 长度过滤 → 字符过滤 → 去重 → 同源切分 → 质量报告。
去重必须在切分之前，否则同家族序列会跨 train/test 造成泄漏。去重使用
k-mer 集合的 Jaccard 相似度，比编辑距离更适合大规模两两比较。
"""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pepdesign.bio.sequence import (
    canonicalize,
    coverage,
    is_cyclic_capable,
    summarize,
    validate,
)
from pepdesign.data.schema import DataQualityReport, PeptideRecord
from pepdesign.utils.config import DataConfig
from pepdesign.utils.logger import get_logger

logger = get_logger(__name__)

# k-mer 的 k。k=3 在 5–30 残基区间区分度最好：k=2 组合太少，k=5 粒度太粗。
DEFAULT_KMER: int = 3

# 大蛋白名称标记：命中即视为完整蛋白的结构域片段而非多肽。
# 这些条目长度合法、字符合法，只有名称能暴露来源；丢弃量计入 drop_reasons。
LARGE_PROTEIN_MARKERS: tuple[str, ...] = (
    "envelope glycoprotein",
    "envelope protein",
    "von willebrand",
    "flagellin",
    "transmembrane protein",
    "crystallin",
)

# 长度分箱边界，用于质量报告中的长度分布。
LENGTH_BINS: tuple[tuple[int, int, str], ...] = (
    (5, 8, "5-8(短肽)"),
    (9, 14, "9-14(中肽)"),
    (15, 22, "15-22(中长肽)"),
    (23, 30, "23-30(长肽)"),
    (31, 60, "31-60(超长肽)"),
)


# ---------------------------------------------------------------------------
# 1. 单条记录清洗
# ---------------------------------------------------------------------------
@dataclass
class CleanResult:
    """单条记录的清洗结果。`reason` 非 None 表示被丢弃。"""

    record: PeptideRecord | None
    reason: str | None = None


def clean_record(rec: PeptideRecord, cfg: DataConfig) -> CleanResult:
    """清洗单条记录。返回失败原因而不是抛异常，便于统计丢弃原因分布。"""
    seq = canonicalize(rec.sequence)
    if not seq:
        return CleanResult(None, "empty_sequence")

    # --- 非天然残基：丢弃，而不是照常保留 ---
    # canonicalize 会把小写字母大写，RGDfV 会变成 RGDFV（手性不同的另一条
    # 序列），却仍挂着原记录的功效标注，因此这类数据不能放行。
    if rec.has_noncanonical:
        return CleanResult(None, "noncanonical_residue")

    # --- 前体蛋白过滤：成熟肽不带起始甲硫氨酸 ---
    # 以 M 开头的序列通常是未被切掉信号肽的前体。下载层已按 Chain 切割，
    # 未切且以 M 起始的记录在此丢弃。
    if seq.startswith("M") and not (rec.lineage or {}).get("chain_cut"):
        return CleanResult(None, "uncleaved_precursor")

    # --- 大蛋白结构域片段：丢弃 ---
    # 只能靠名称识别（见 LARGE_PROTEIN_MARKERS）。放在长度过滤之前，
    # 使丢弃原因反映来源问题而非长度问题。
    name_lower = (rec.name or "").lower()
    if any(marker in name_lower for marker in LARGE_PROTEIN_MARKERS):
        return CleanResult(None, "protein_fragment")

    # --- 长度过滤 ---
    if len(seq) < cfg.min_length:
        return CleanResult(None, f"too_short(<{cfg.min_length})")
    if len(seq) > cfg.max_length:
        return CleanResult(None, f"too_long(>{cfg.max_length})")

    # --- 字符合法性 ---
    ok, reason = validate(seq)
    if not ok:
        return CleanResult(None, reason)

    # --- 标准残基比例 ---
    # 比例太低说明序列大部分是未知区域，拿它训练等于让模型学一堆 X。
    cov = coverage(seq)
    if cov < cfg.min_standard_ratio:
        return CleanResult(None, f"low_coverage({cov:.0%})")

    # --- 组成合理性 ---
    # 单一残基占比过高的序列作为 SFT 目标会让模型学到退化解，直接过滤。
    counts = Counter(seq)
    if counts.most_common(1)[0][1] / len(seq) > 0.6:
        return CleanResult(None, "low_complexity")

    # --- 环肽一致性检查 ---
    if rec.is_cyclic and not is_cyclic_capable(seq):
        # 标注为环肽但序列两端是 Pro，无法头尾环化 → 标注有误，降级为线性肽
        # 而不是丢弃：序列本身还是有价值的。
        rec.cyclic_type = "none"
        rec.lineage["downgraded"] = "cyclic_constraint_violated"

    rec.sequence = seq
    rec.compute_properties()  # 生成性质快照，供后续按性质分层
    return CleanResult(rec, None)


# ---------------------------------------------------------------------------
# 2. 近似去重
# ---------------------------------------------------------------------------
def kmer_set(seq: str, k: int = DEFAULT_KMER) -> frozenset[str]:
    """序列的 k-mer 集合。长度不足 k 时退化为整个序列本身。"""
    if len(seq) < k:
        return frozenset({seq})
    return frozenset(seq[i : i + k] for i in range(len(seq) - k + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard 相似度 = |A∩B| / |A∪B|，取值 [0,1]。"""
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0


def information_score(rec: PeptideRecord) -> tuple:
    """给记录排序用的"信息量"打分。重复组内保留分最高的那条。

    打分维度（从高到低）：
      1. 有实测活性数据 —— 这类样本最稀缺，是标定奖励函数的依据；
      2. 有靶点标注 —— 能支持"靶点信息引导设计"；
      3. 有明确功能分类（非 other）；
      4. 环肽 —— 环肽样本天然稀少，比线性肽更值得保留；
      5. 序列长度 —— 同组里长的信息量更大（包含了短的作为子串的信息）。

    tuple 比较是按位字典序，正好实现"逐级降序比较"的语义。
    """
    return (
        1 if rec.has_experimental_data else 0,
        1 if rec.target else 0,
        0 if rec.function == "other" else 1,
        1 if rec.is_cyclic else 0,
        rec.length,
    )


def deduplicate(
    records: Sequence[PeptideRecord],
    *,
    threshold: float = 0.9,
    k: int = DEFAULT_KMER,
) -> tuple[list[PeptideRecord], int]:
    """近似去重，返回 (去重后列表, 移除条数)。

    倒排索引 + 贪心聚类：只与共享 k-mer 的已保留序列比较 Jaccard，
    避免 O(n²) 全量比较。阈值 0.9 约等于允许 1–2 个残基差异。
    """
    if not records:
        return [], 0

    # 按信息量降序，保证每个重复组里先处理的是"最值得保留"的那条
    indexed = sorted(records, key=information_score, reverse=True)
    kmers = [kmer_set(r.sequence, k) for r in indexed]

    inverted: dict[str, list[int]] = defaultdict(list)
    for i, ks in enumerate(kmers):
        for token in ks:
            inverted[token].append(i)

    kept_idx: list[int] = []
    kept_set: set[int] = set()   # 已被保留的下标，用于快速判断"跟谁比"
    removed = 0

    for i, _ in enumerate(indexed):
        my_kmers = kmers[i]
        overlap: dict[int, int] = defaultdict(int)
        for token in my_kmers:
            for j in inverted[token]:
                if j < i:  # 只和已经处理过的（=信息量更高的）比较
                    overlap[j] += 1

        is_dup = False
        for j, inter in overlap.items():
            if j not in kept_set:
                continue
            other = kmers[j]
            union = len(my_kmers) + len(other) - inter
            if union and inter / union >= threshold:
                is_dup = True
                break
        if is_dup:
            removed += 1
            continue

        kept_set.add(i)
        kept_idx.append(i)

    deduped = [indexed[i] for i in kept_idx]
    logger.info("去重完成：%d -> %d，移除 %d 条", len(records), len(deduped), removed)
    return deduped, removed


# ---------------------------------------------------------------------------
# 3. 类别平衡
# ---------------------------------------------------------------------------
def upweight_cyclic(
    records: Sequence[PeptideRecord],
    *,
    target_ratio: float = 0.15,
    max_repeat: int = 3,
) -> list[PeptideRecord]:
    """对环肽样本做有限上采样（最多 max_repeat 轮），缓解类别不平衡。

    无限复制会导致过拟合；序列是离散符号，不能使用 SMOTE 类插值方法。
    """
    cyclic = [r for r in records if r.is_cyclic]
    linear = [r for r in records if not r.is_cyclic]
    if not cyclic or not linear:
        return list(records)

    current_ratio = len(cyclic) / len(records)
    if current_ratio >= target_ratio:
        logger.info("环肽占比 %.1f%% 已达标，不做上采样", current_ratio * 100)
        return list(records)

    # 需要的环肽条数：cyclic_n / (total + extra) = target
    target_cyclic = math.ceil(target_ratio * len(linear) / (1 - target_ratio))
    need = max(0, target_cyclic - len(cyclic))
    extra: list[PeptideRecord] = []
    for round_idx in range(max_repeat):
        if len(extra) >= need:
            break
        for rec in cyclic:
            if len(extra) >= need:
                break
            clone = copy.deepcopy(rec)
            clone.lineage["upsampled_round"] = round_idx + 1
            extra.append(clone)

    logger.info(
        "环肽上采样：%d -> %d 条（目标占比 %.0f%%，实际上限 %d 轮）",
        len(cyclic), len(cyclic) + len(extra), target_ratio * 100, max_repeat,
    )
    return list(records) + extra


# ---------------------------------------------------------------------------
# 4. 质量报告
# ---------------------------------------------------------------------------
def length_bin(length: int) -> str:
    for lo, hi, label in LENGTH_BINS:
        if lo <= length <= hi:
            return label
    return f">{LENGTH_BINS[-1][1]}"


def build_quality_report(
    raw_count: int,
    kept: Sequence[PeptideRecord],
    drop_reasons: dict[str, int],
    dedup_removed: int,
) -> DataQualityReport:
    """汇总质量报告：来源、丢弃原因、长度/功能/来源分布与性质概览。"""
    report = DataQualityReport(
        raw_count=raw_count,
        kept_count=len(kept),
        drop_reasons=dict(drop_reasons),
        dedup_removed=dedup_removed,
    )
    if not kept:
        return report

    length_dist: Counter = Counter()
    func_dist: Counter = Counter()
    src_dist: Counter = Counter()
    for rec in kept:
        length_dist[length_bin(rec.length)] += 1
        func_dist[rec.function] += 1
        src_dist[rec.source] += 1

    report.length_distribution = dict(length_dist)
    report.function_distribution = dict(func_dist)
    report.source_distribution = dict(src_dist)
    report.cyclic_ratio = round(sum(1 for r in kept if r.is_cyclic) / len(kept), 4)
    report.mean_length = round(sum(r.length for r in kept) / len(kept), 2)
    report.property_summary = summarize([r.sequence for r in kept])
    return report


# ---------------------------------------------------------------------------
# 5. 总流水线
# ---------------------------------------------------------------------------
def run_cleaning(
    raw_records: Iterable[PeptideRecord],
    cfg: DataConfig | None = None,
    *,
    balance_cyclic: bool = True,
) -> tuple[list[PeptideRecord], DataQualityReport]:
    """完整清洗流水线：过滤 -> 去重 -> 平衡 -> 报告。

    只做清洗，不做切分；切分需要全局同源聚类信息，见 split.py。
    """
    cfg = cfg or DataConfig()
    raw_list = list(raw_records)
    logger.info("开始清洗，原始样本 %d 条", len(raw_list))

    kept: list[PeptideRecord] = []
    drop_reasons: Counter = Counter()
    for rec in raw_list:
        result = clean_record(rec, cfg)
        if result.record is None:
            # 归并带参数的原因标签，否则原因种类过多，报告里看不出重点。
            key = (result.reason or "unknown").split("(")[0]
            drop_reasons[key] += 1
        else:
            kept.append(result.record)

    logger.info("过滤后剩余 %d 条，丢弃原因分布: %s",
                len(kept), dict(drop_reasons.most_common()))

    kept, dedup_removed = deduplicate(kept, threshold=cfg.dedup_identity)

    if balance_cyclic:
        kept = upweight_cyclic(kept, target_ratio=cfg.min_cyclic_ratio)

    report = build_quality_report(len(raw_list), kept, dict(drop_reasons), dedup_removed)
    logger.info(
        "清洗完成：%d -> %d（保留率 %.1f%%），环肽占比 %.1f%%，平均长度 %.1f",
        report.raw_count, report.kept_count, report.keep_rate * 100,
        report.cyclic_ratio * 100, report.mean_length,
    )
    return kept, report
