"""多源公开数据集采集。

UniProt 通过 REST API 游标分页拉取；APD3 / DRAMP / DBAASP / PDB 等人工
下载源只扫描本地目录并校验格式；内部源读取脱敏后的 JSONL/CSV。
每次采集写出 manifest.json，记录来源、条数、查询参数与版本信息。
"""

from __future__ import annotations

import contextlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pepdesign.data.schema import ActivityRecord, PeptideRecord
from pepdesign.utils.config import DataConfig, resolve_path
from pepdesign.utils.io import read_jsonl, write_json, write_jsonl
from pepdesign.utils.logger import get_logger

logger = get_logger(__name__)

# UniProt 官方程序化接口，支持游标分页与字段裁剪。
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

# 只请求需要的字段：默认 JSON 每条十几 KB，裁剪后仅几百字节。
# feature 字段用于切出成熟肽并取修饰信息。
UNIPROT_FIELDS = ",".join([
    "accession",
    "id",
    "protein_name",
    "sequence",
    "length",
    "organism_name",
    "keyword",
    "ft_disulfid",
    # ---- feature 字段：切成熟肽与取修饰信息用 ----
    "ft_signal",     # 信号肽范围
    "ft_chain",      # 成熟链范围
    "ft_propep",     # 前导肽范围
    "ft_mod_res",    # 修饰残基
])

# UniProt 查询及其产出契约。require_any / exclude_any 在采集时校验并报告达成率；
# 关键词与类目语义不符是静默失败，契约用于把人工抽查变成自动检查。
UNIPROT_QUERIES: list[dict] = [
    {
        "name": "antimicrobial_peptides",
        "query": '(keyword:"Antimicrobial" OR keyword:"Antibiotic") AND (length:[5 TO 60])',
        "function": "antibacterial",
        "require_any": ["Antimicrobial", "Antibiotic"],
    },
    {
        "name": "antiviral_peptides",
        "query": '(keyword:"Antiviral protein") AND (length:[5 TO 60])',
        "function": "antiviral",
        "require_any": ["Antiviral protein"],
    },
    {
        "name": "antifungal_peptides",
        "query": '(keyword:"Fungicide" OR keyword:"Antifungal") AND (length:[5 TO 60])',
        "function": "antifungal",
        "require_any": ["Fungicide", "Antifungal"],
    },
    # 刻意没有细胞穿膜肽查询：UniProt 无可靠关键词，Cell projection /
    # Cell adhesion 实际是细胞突起与黏附蛋白，需改用 CPPsite 类专门数据源。
    {
        # 二硫键命中的条目多为环肽，是获取环肽最有效的入口；长度上限取 60，
        # 以覆盖 40–60 残基的 defensin 类。
        "name": "disulfide_cyclic",
        "query": '(keyword:"Disulfide bond") AND (length:[5 TO 60])',
        "function": "other",
        "cyclic_hint": "disulfide",
        "require_any": ["Disulfide bond"],
        "max_records": 20000,
    },
    # 以下关键词在 UniProt 中语义明确，且落在多肽长度区间内，
    # 用于提高底层真实序列的多样性。
    {
        "name": "lantibiotics",
        "query": '(keyword:"Lantibiotic") AND (length:[5 TO 60])',
        "function": "antibacterial",
        "cyclic_hint": "disulfide",
        "require_any": ["Lantibiotic"],
    },
    {
        "name": "bacteriocins",
        "query": '(keyword:"Bacteriocin") AND (length:[5 TO 60])',
        "function": "antibacterial",
        "require_any": ["Bacteriocin"],
    },
    {
        "name": "defensins",
        "query": '(keyword:"Defensin") AND (length:[5 TO 60])',
        "function": "antibacterial",
        "cyclic_hint": "disulfide",
        "require_any": ["Defensin"],
    },
    {
        "name": "neuropeptides",
        "query": '(keyword:"Neuropeptide") AND (length:[5 TO 60])',
        "function": "other",
        "require_any": ["Neuropeptide"],
    },
    {
        "name": "peptide_hormones",
        "query": '(keyword:"Hormone") AND (length:[5 TO 60])',
        "function": "other",
        "require_any": ["Hormone"],
    },
    {
        "name": "protease_inhibitors",
        "query": '(keyword:"Protease inhibitor") AND (length:[5 TO 60])',
        "function": "inhibitor",
        "require_any": ["Protease inhibitor"],
    },
    # 第二批扩充：每条查询均使用语义明确的关键词，并带产出契约。
    {
        # Knottin 是环肽的又一条有效入口；环肽在 UniProt 没有专属关键词。
        "name": "knottins",
        "query": '(keyword:"Knottin") AND (length:[5 TO 60])',
        "function": "other",
        "cyclic_hint": "disulfide",
        "require_any": ["Knottin"],
    },
    {
        # 直接使用 Peptide 关键词：覆盖激素、神经肽等各类小肽；
        # 清洗层会按长度、非标准残基与前体片段过滤。
        "name": "peptides_generic",
        "query": '(keyword:"Peptide") AND (length:[5 TO 60])',
        "function": "other",
        "require_any": ["Peptide"],
        "max_records": 20000,
    },
    {
        "name": "cytokines",
        "query": '(keyword:"Cytokine") AND (length:[5 TO 60])',
        "function": "immunomodulatory",
        "require_any": ["Cytokine"],
    },
    {
        "name": "growth_factors",
        "query": '(keyword:"Growth factor") AND (length:[5 TO 60])',
        "function": "other",
        "require_any": ["Growth factor"],
    },
    {
        "name": "antioxidants",
        "query": '(keyword:"Antioxidant") AND (length:[5 TO 60])',
        "function": "other",
        "require_any": ["Antioxidant"],
    },
    {
        "name": "amyloid_peptides",
        "query": '(keyword:"Amyloid") AND (length:[5 TO 60])',
        "function": "other",
        "require_any": ["Amyloid"],
    },
    {
        "name": "antihypertensives",
        "query": '(keyword:"Antihypertensive") AND (length:[5 TO 60])',
        "function": "other",
        "require_any": ["Antihypertensive"],
    },
    {
        "name": "glycopeptides",
        "query": '(keyword:"Glycopeptide") AND (length:[5 TO 60])',
        "function": "other",
        "require_any": ["Glycopeptide"],
    },
]
# 刻意没有抗肿瘤肽查询：Antineoplastic 命中 0 条，Cytolysis 命中的是蛇毒
# 细胞毒素；该方向应改用 DRAMP 等专门库。毒类关键词记录会被 exclude_any 丢弃。

#: 所有查询共用的排除表：命中的记录一律丢掉。
UNIPROT_EXCLUDE_KEYWORDS: tuple[str, ...] = (
    "Toxin",
    "Cardiotoxin",
    "Neurotoxin",
    "Hemolytic",
)


@dataclass
class DownloadManifest:
    """采集清单。用于数据集版本溯源。"""

    timestamp: str = ""
    sources: dict[str, dict] = field(default_factory=dict)
    total_records: int = 0

    def record(self, source: str, count: int, **extra: object) -> None:
        entry = self.sources.setdefault(source, {"count": 0})
        entry["count"] += count
        entry.update(extra)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "total_records": self.total_records,
            "sources": self.sources,
        }


# ---------------------------------------------------------------------------
# 1. UniProt API 采集
# ---------------------------------------------------------------------------
def _http_get(url: str, *, retries: int = 5, timeout: int = 60) -> tuple[bytes, dict]:
    """带指数退避重试的 GET，返回 (响应体, 响应头)。

    需显式捕获 http.client.HTTPException：RemoteDisconnected 不属于
    URLError，连接中途断开时不会自动重试。
    """
    import http.client

    last_exc: Exception | None = None
    # 连接级异常单独兜住：它们不保证已经收到完整响应，但重试是安全的（GET 幂等）
    transient = (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        http.client.HTTPException,
        ConnectionError,
        OSError,
    )
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    # UniProt 要求带上 User-Agent 标明来源，便于他们统计与联系
                    "User-Agent": "pepdesign-research/0.1 (academic peptide design)",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read(), dict(resp.headers)
        except transient as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning("请求失败（第 %d 次）: %s，%ds 后重试", attempt + 1, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"请求重试 {retries} 次仍失败: {url}") from last_exc


#: 关键词 → 功能分类，按特异性从高到低排列；宽谱关键词放最后兜底。
#: 刻意没有 Antineoplastic → antitumor：该关键词在 UniProt 中命中 0 条。
_FUNCTION_BY_KEYWORD: list[tuple[str, str]] = [
    ("Fungicide", "antifungal"),
    ("Antifungal", "antifungal"),
    ("Antiviral protein", "antiviral"),
    # Antineoplastic 在 UniProt 中命中 0 条，不列为抗肿瘤来源。
    ("Antimicrobial", "antibacterial"),
    ("Antibiotic", "antibacterial"),
]


def _classify_function(keywords: list[str], fallback: str) -> str:
    """按关键词判功能，认不出来就用查询给的粗分类兜底。"""
    for keyword, func in _FUNCTION_BY_KEYWORD:
        if keyword in keywords:
            return func
    return fallback


def _location_range(feature: dict) -> tuple[int, int] | None:
    """取一个 feature 的 (start, end)，1-based 闭区间。取不到返回 None。

    `end` 可能缺失（单残基 feature，比如"修饰残基"只有一个位置），
    这时按 UniProt 的约定把 end 当成 start。
    """
    loc = feature.get("location") or {}
    start = (loc.get("start") or {}).get("value")
    if start is None:
        return None
    # end 可能缺键或值为 null，两种形态都要兜住。
    end = (loc.get("end") or {}).get("value")
    if end is None:
        end = start
    return int(start), int(end)


def _mature_sequence(entry: dict, seq: str) -> tuple[str, dict]:
    """用 UniProt 的 Chain feature 切出成熟肽，返回 `(序列, 切割说明)`。

    多个 Chain 时取最长的；无 Chain 时退回 Signal / Propeptide 的最远端，
    切不动则保留原序列。
    """
    features = entry.get("features", []) or []
    chains: list[tuple[int, int, str]] = []
    for f in features:
        if f.get("type") != "Chain":
            continue
        rng = _location_range(f)
        if rng:
            chains.append((*rng, f.get("description", "")))

    if chains:
        start, end, desc = max(chains, key=lambda c: c[1] - c[0])
        # 覆盖全长时不算切割（很多成熟肽条目本身就整条都是 Chain）
        if start <= 1 and end >= len(seq):
            return seq, {"chain_cut": False, "reason": "chain_covers_full"}
        cut = seq[start - 1: end]
        if cut:
            return cut, {
                "chain_cut": True,
                "cut_by": "chain",
                "chain": [start, end],
                "chain_desc": desc,
                "precursor_len": len(seq),
            }

    # ---- 回退：没标 Chain 时，用 Signal / Propeptide 的末尾当切割点 ----
    # 取两者的最远端作为起点；只取 Signal 会让带前导肽的条目残留前导序列。
    cut_after = 0
    used: list[str] = []
    for f in features:
        if f.get("type") not in ("Signal", "Propeptide"):
            continue
        rng = _location_range(f)
        if rng and rng[1] > cut_after:
            cut_after = rng[1]
            used.append(f"{f['type']}[{rng[0]},{rng[1]}]")

    if cut_after > 0:
        cut = seq[cut_after:]
        # 切完太短则不切，避免得到无意义的碎片。
        if len(cut) >= 5:
            return cut, {
                "chain_cut": True,
                "cut_by": "signal_or_propeptide",
                "cut_features": used,
                "cut_after": cut_after,
                "precursor_len": len(seq),
            }

    return seq, {"chain_cut": False, "reason": "no_chain_no_signal"}


def _modifications(entry: dict) -> list[str]:
    """取出修饰残基描述列表（去重、保持出现顺序）。"""
    seen: dict[str, None] = {}
    for f in entry.get("features", []) or []:
        if f.get("type") != "Modified residue":
            continue
        desc = (f.get("description") or "").strip()
        if desc:
            seen.setdefault(desc, None)
    return list(seen)


def _parse_uniprot_entry(
    entry: dict, function: str, cyclic_hint: str = "none"
) -> PeptideRecord | None:
    """把一条 UniProt JSON 条目转成 PeptideRecord。

    返回 None 表示记录不可用（缺序列等）；不在解析层抛异常，
    避免单条异常导致整批采集失败。
    """
    seq = entry.get("sequence", {}).get("value", "")
    if not seq:
        return None

    accession = entry.get("primaryAccession", "")
    protein_name = ""
    desc = entry.get("proteinDescription", {})
    if desc:
        rec_name = desc.get("recommendedName", {})
        if rec_name:
            protein_name = rec_name.get("fullName", {}).get("value", "")
        if not protein_name:
            subs = desc.get("submissionNames", [])
            if subs:
                protein_name = subs[0].get("fullName", {}).get("value", "")

    keywords = [kw.get("name", "") for kw in entry.get("keywords", [])]

    # ---- 切成熟肽（必须在判环化之前：二硫键个数要按切完的序列数）----
    precursor_len = len(seq)
    seq, cut_info = _mature_sequence(entry, seq)
    if not seq:
        return None

    features = entry.get("features", []) or []

    # 环化类型判定：优先用查询时给的提示，其次看条目里有没有二硫键特征
    cyclic_type = cyclic_hint
    if cyclic_type == "none":
        has_disulfide = any(f.get("type") == "Disulfide bond" for f in features)
        # 只有 4 个以上 Cys 才认为它可能是靠二硫键成环的肽（2 个 Cys 太常见了，
        # 很多线性肽也有单对二硫键）
        if has_disulfide and seq.count("C") >= 4:
            cyclic_type = "disulfide"

    modifications = _modifications(entry)
    # 不要用 modifications 置 has_noncanonical：UniProt 的 sequence 字段
    # 只用标准一字母码，修饰写在注释里；两者绑定会误丢大量正常序列。

    organism = entry.get("organism", {}).get("scientificName", "")

    return PeptideRecord(
        sequence=seq,
        name=protein_name,
        source="uniprot",
        source_id=accession,
        function=_classify_function(keywords, function),  # type: ignore[arg-type]
        cyclic_type=cyclic_type,  # type: ignore[arg-type]
        organism=organism,
        modifications=modifications,
        lineage={
            "keywords": keywords[:10],
            "uniprot_length": entry.get("sequence", {}).get("length"),
            "precursor_len": precursor_len,
            **cut_info,
        },
    )


def fetch_uniprot(
    *,
    query: str,
    function: str = "other",
    cyclic_hint: str = "none",
    max_records: int = 5000,
    page_size: int = 500,
    require_any: Sequence[str] | None = None,
    exclude_any: Sequence[str] | None = None,
    stats: dict | None = None,
) -> Iterator[PeptideRecord]:
    """按游标分页拉取 UniProt 条目，并按产出契约过滤。

    必须用响应头 Link 中的游标翻页，from/size 偏移翻页会漏条目或重复。
    `require_any` 缺失即丢弃并计数，`exclude_any` 命中的记录直接丢弃。
    `stats` 为可变 dict，用于把计数带回调用方。
    """
    require_any = list(require_any or [])
    exclude_any = list(exclude_any or [])
    if stats is not None:
        stats.setdefault("kept", 0)
        stats.setdefault("excluded", {})
        stats.setdefault("contract_miss", 0)
        stats.setdefault("unparsable", 0)

    params = {
        "query": query,
        "format": "json",
        "size": str(min(page_size, 500)),
        "fields": UNIPROT_FIELDS,
    }
    url = f"{UNIPROT_SEARCH_URL}?{urllib.parse.urlencode(params)}"
    fetched = 0
    page = 0

    while url and fetched < max_records:
        body, headers = _http_get(url)
        payload = json.loads(body)
        results = payload.get("results", [])
        if not results:
            break

        page += 1
        for entry in results:
            keywords = [kw.get("name", "") for kw in entry.get("keywords", [])]

            hit = next((k for k in exclude_any if k in keywords), None)
            if hit is not None:
                if stats is not None:
                    stats["excluded"][hit] = stats["excluded"].get(hit, 0) + 1
                continue

            if require_any and not any(k in keywords for k in require_any):
                if stats is not None:
                    stats["contract_miss"] += 1
                continue

            rec = _parse_uniprot_entry(entry, function, cyclic_hint)
            if rec is None:
                if stats is not None:
                    stats["unparsable"] += 1
                continue
            if stats is not None:
                stats["kept"] += 1
            yield rec
            fetched += 1
            if fetched >= max_records:
                logger.warning(
                    "达到 max_records=%d 上限，%s 的采集被截断（可能还有更多数据）",
                    max_records, query,
                )
                return

        logger.info("UniProt 第 %d 页完成，累计 %d 条", page, fetched)

        # 用正则从 Link 头取 next 游标：fields 参数本身含逗号，
        # 按逗号切分会把 URL 截断。
        link = headers.get("Link") or headers.get("link") or ""
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = match.group(1) if match else ""

        # 礼貌爬取：UniProt 的公开政策要求不要高频请求
        time.sleep(0.5)


#: 契约达成率低于此值就报警。宽泛关键词组合的命中率本来不是 1，
#: 要抓的是"关键词根本不存在"这一类，其达成率会接近 0。
_CONTRACT_MIN_COVERAGE = 0.5


def _check_contract(name: str, spec: dict, stats: dict) -> None:
    """校验查询产出是否符合声明的契约，不达阈值就报警。

    该机制只验证记录带有所需关键词，不能验证关键词与类目语义一致；
    后者仍需人工核对查询定义。
    """
    kept = stats.get("kept", 0)
    miss = stats.get("contract_miss", 0)
    total = kept + miss
    if total == 0:
        logger.error(
            "UniProt 子集 %s 一条都没采到——查询 %r 可能是坏的（关键词不存在？）。",
            name, spec["query"],
        )
        return

    coverage = 1.0 - miss / total
    if coverage < _CONTRACT_MIN_COVERAGE:
        logger.warning(
            "UniProt 子集 %s 的契约达成率只有 %.0f%%（%d/%d 条没命中 %s）——"
            "查询 %r 的语义可能和类目名对不上，请抽查记录内容确认。",
            name, coverage * 100, miss, total,
            spec.get("require_any"), spec["query"],
        )
    excluded = stats.get("excluded", {})
    if excluded:
        logger.info(
            "UniProt 子集 %s 排除了 %d 条非药物候选：%s",
            name, sum(excluded.values()),
            ", ".join(f"{k}x{v}" for k, v in sorted(excluded.items())),
        )


def collect_uniprot(cfg: DataConfig, out_dir: Path) -> tuple[list[PeptideRecord], DownloadManifest]:
    """执行全部 UniProt 查询并落盘。"""
    manifest = DownloadManifest(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
    records: list[PeptideRecord] = []

    for spec in UNIPROT_QUERIES:
        name = spec["name"]
        logger.info("开始采集 UniProt 子集: %s", name)
        stats: dict = {}
        subset = list(
            fetch_uniprot(
                query=spec["query"],
                function=spec["function"],
                cyclic_hint=spec.get("cyclic_hint", "none"),
                require_any=spec.get("require_any"),
                exclude_any=UNIPROT_EXCLUDE_KEYWORDS,
                stats=stats,
                # 单条查询默认上限从 5,000 提到 20,000：放宽后的
                # disulfide 查询实测有 13,716 条，旧的 5,000 会截断大半。
                max_records=int(spec.get("max_records", 20000)),
            )
        )
        _check_contract(name, spec, stats)

        # 同一个 accession 可能被多个 query 命中，在采集阶段就用 source_id 去重，
        # 避免后面清洗时做无谓的工作
        seen = {r.source_id for r in records if r.source_id}
        fresh = [r for r in subset if r.source_id not in seen]
        records.extend(fresh)
        manifest.record(
            f"uniprot:{name}", len(fresh),
            query=spec["query"],
            raw_returned=len(subset),
            deduped=len(subset) - len(fresh),
            # 契约执行情况一起进清单——"这条查询到底捞回来什么"是数据集
            # 可追溯性的一部分，不该只留在日志里（日志会被滚掉）。
            excluded=stats.get("excluded", {}),
            contract_miss=stats.get("contract_miss", 0),
            unparsable=stats.get("unparsable", 0),
        )
        logger.info(
            "UniProt 子集 %s 采集完成: 新增 %d 条（排除 %d 条非候选、%d 条契约未命中）",
            name, len(fresh), sum(stats.get("excluded", {}).values()),
            stats.get("contract_miss", 0),
        )

    manifest.total_records = len(records)
    write_jsonl(out_dir / "uniprot.jsonl", (r.to_dict() for r in records))
    return records, manifest


# ---------------------------------------------------------------------------
# 2. 人工下载源的目录扫描
# ---------------------------------------------------------------------------
# 各人工下载源期望的文件名模式。校验文件名可避免把说明文档或残留文件当数据。
MANUAL_SOURCE_SPECS: dict[str, dict] = {
    "apd3": {
        "patterns": ["*.fasta", "*.fa", "APD3_*.txt"],
        "format": "fasta",
        "note": "APD3 抗菌肽数据库，官网按机构授权后提供批量下载",
    },
    "dramp": {
        "patterns": ["DRAMP*.fasta", "*.fasta"],
        "format": "fasta",
        "note": "DRAMP 治疗肽库，含结构与修饰标注",
    },
    "dbaasp": {
        "patterns": ["*.csv", "*.xlsx"],
        "format": "csv",
        "note": "DBAASP 含 MIC/溶血等活性数据，是奖励函数标定的重要来源",
    },
    "pdb_cyclic": {
        "patterns": ["*.fasta", "*.pdb"],
        "format": "fasta",
        "note": "PDB 环肽结构，用于结构维度的标注与验证",
    },
    "internal": {
        "patterns": ["*.jsonl", "*.csv"],
        "format": "jsonl",
        "note": "内部专家标注与湿实验数据（已脱敏）",
    },
}


def parse_fasta(text: str) -> Iterator[tuple[str, str]]:
    """极简 FASTA 解析器，产出 (header, sequence)。

    不引入 Biopython，避免大文件上的对象封装开销。
    """
    header = ""
    chunks: list[str] = []
    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith(">"):
            if header or chunks:
                yield header, "".join(chunks)
            header = line[1:].strip()
            chunks = []
        elif line:
            chunks.append(line.strip())
    if header or chunks:
        yield header, "".join(chunks)


def scan_manual_source(
    source: str,
    raw_dir: Path,
    *,
    cyclic_type: str = "none",
) -> list[PeptideRecord]:
    """扫描人工下载目录。目录不存在时返回空列表并给出明确提示。"""
    spec = MANUAL_SOURCE_SPECS.get(source)
    if spec is None:
        raise KeyError(f"未知的人工源: {source}")

    src_dir = raw_dir / source
    if not src_dir.exists():
        logger.warning(
            "目录不存在，跳过 %s（%s）。请人工下载后放入 %s",
            source, spec["note"], src_dir,
        )
        return []

    files: list[Path] = []
    for pattern in spec["patterns"]:
        files.extend(sorted(src_dir.glob(pattern)))
    # 去重（不同 pattern 可能命中同一文件）
    files = sorted(set(files))
    if not files:
        logger.warning("%s 目录存在但没有符合 %s 的文件", source, spec["patterns"])
        return []

    records: list[PeptideRecord] = []
    for path in files:
        logger.info("读取 %s/%s", source, path.name)
        if spec["format"] == "fasta":
            text = path.read_text(encoding="utf-8", errors="ignore")
            for header, seq in parse_fasta(text):
                if not seq:
                    continue
                records.append(
                    PeptideRecord(
                        sequence=seq,
                        name=header[:120],
                        source=source,  # type: ignore[arg-type]
                        source_id=f"{path.stem}:{header[:40]}",
                        cyclic_type=cyclic_type,  # type: ignore[arg-type]
                        lineage={"file": path.name},
                    )
                )
        elif spec["format"] == "jsonl":
            for item in read_jsonl(path):
                records.append(PeptideRecord.from_dict(item))
        elif spec["format"] == "csv":
            records.extend(_read_csv_source(path, source))
        else:
            logger.warning("暂不支持的格式 %s: %s", spec["format"], path)

    logger.info("%s 共读取 %d 条记录", source, len(records))
    return records


def _read_csv_source(path: Path, source: str) -> list[PeptideRecord]:
    """读取 CSV 形式的源（DBAASP / 内部导出）。

    用标准库 csv，不引入 pandas；列名做大小写不敏感的模糊匹配。
    """
    import csv

    seq_keys = ("sequence", "seq", "peptide", "序列", "氨基酸序列")
    target_keys = ("target", "靶点", "strain", "菌株")
    value_keys = ("mic", "ic50", "activity", "value", "活性")

    records: list[PeptideRecord] = []
    with open(path, encoding="utf-8-sig", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        # 建立 小写列名 -> 原列名 的映射，方便大小写不敏感匹配
        lower_map = {name.lower().strip(): name for name in reader.fieldnames}

        seq_col = next((lower_map[k] for k in seq_keys if k in lower_map), None)
        if seq_col is None:
            logger.warning("%s 找不到序列列，跳过。现有列: %s",
                           path.name, list(reader.fieldnames)[:10])
            return []
        target_col = next((lower_map[k] for k in target_keys if k in lower_map), None)
        value_col = next((lower_map[k] for k in value_keys if k in lower_map), None)

        for row in reader:
            seq = (row.get(seq_col) or "").strip()
            if not seq:
                continue
            activities: list[ActivityRecord] = []
            if value_col:
                raw_val = (row.get(value_col) or "").strip()
                # 活性值缺失或格式异常时跳过该条活性，不留下空记录。
                with contextlib.suppress(ValueError, TypeError):
                    activities.append(
                        ActivityRecord(
                            assay_type="MIC" if "mic" in value_col.lower() else "activity",
                            target=(row.get(target_col) or "") if target_col else "",
                            value=float(raw_val),
                            unit="ug/mL",
                            source_ref=path.stem,
                        )
                    )
            records.append(
                PeptideRecord(
                    sequence=seq,
                    source=source,  # type: ignore[arg-type]
                    source_id=f"{path.stem}:{len(records)}",
                    target=(row.get(target_col) or "") if target_col else "",
                    activities=activities,
                    lineage={"file": path.name},
                )
            )
    return records


# ---------------------------------------------------------------------------
# 3. 总入口
# ---------------------------------------------------------------------------
def collect_all(cfg: DataConfig | None = None) -> DownloadManifest:
    """按配置采集所有启用的数据源，落盘到 data/raw/。"""
    cfg = cfg or DataConfig()
    raw_dir = resolve_path(cfg.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest = DownloadManifest(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
    total = 0

    for source, spec in cfg.sources.items():
        if not spec.get("enabled", False):
            logger.info("跳过未启用的数据源: %s", source)
            continue

        kind = spec.get("kind")
        if source == "uniprot" and kind == "api":
            _, sub_manifest = collect_uniprot(cfg, raw_dir)
            for key, entry in sub_manifest.sources.items():
                manifest.sources[key] = entry
            total += sub_manifest.total_records
        elif kind in ("manual", "local"):
            records = scan_manual_source(source, raw_dir)
            write_jsonl(raw_dir / f"{source}.jsonl", (r.to_dict() for r in records))
            manifest.record(source, len(records), note=spec.get("note", ""))
            total += len(records)
        else:
            logger.warning("未知的数据源类型: %s -> %s", source, kind)

    manifest.total_records = total
    write_json(raw_dir / "manifest.json", manifest.to_dict())
    logger.info("采集完成，共 %d 条记录，清单已写入 %s", total, raw_dir / "manifest.json")
    return manifest
