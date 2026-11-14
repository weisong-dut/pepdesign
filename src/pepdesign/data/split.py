"""数据集切分：同源感知（homology-aware）切分。

不能随机切分的原因
------------------
多肽公开数据里，同一个家族的变体极多。比如 defensin 类抗菌肽，
不同物种、不同亚型之间有 60%-90% 的序列相似度。
如果随机切分，训练集里几乎每一条测试序列都能找到一个"近亲"，
模型在测试集上的表现反映的是记忆能力而不是泛化能力——
离线指标会很漂亮，但上线后在新靶点上的命中率会明显偏低，
这是离线指标与线上效果脱节的典型成因。

做法：先把序列聚成同源簇，再以簇为单位整体分配到某个划分。
这样测试集里的任何一条序列，在训练集里都不存在相似度超过阈值的"近亲"。

聚类算法选择
------------
用并查集（Union-Find）做单链聚类（single-linkage）：
  * 对每条序列只跟"共享 k-mer 的候选"比较（倒排索引加速，与去重同款）；
  * 相似度超过 `cluster_threshold` 就把两者所在簇合并。
单链的语义是"有相似关系的序列传递性地归到一簇"，
对同源家族这种"链式相似"的结构是最自然的建模——
A 像 B、B 像 C，那么 A/B/C 就是一个家族，即便 A 和 C 相似度不高。

切分比例按簇分配时用贪心装箱
-------------------------------
不能按顺序切（簇大小差异很大，前几个大簇就把 train 塞满了），
也不能随机切（大簇被随机分到 test 会让 test 规模失控）。
做法是把簇按大小降序排列，依次放进"当前最小的那个划分"里，
这是经典的 LPT（Longest Processing Time first）装箱启发式，
能保证各划分规模接近目标比例。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from pepdesign.data.clean import DEFAULT_KMER, kmer_set
from pepdesign.data.schema import PeptideRecord
from pepdesign.utils.logger import get_logger
from pepdesign.utils.seed import get_rng

logger = get_logger(__name__)

# 聚类阈值。比去重阈值（0.9）低不少，因为目的不同：
# 去重是"删掉冗余"，聚类是"划出家族边界"。
# 0.5 的经验含义：两条 10 残基的肽共享一半以上的 3-mer，
# 基本可以断定是同源家族（而不是巧合，随机序列的 3-mer Jaccard 期望低于 0.05）。
DEFAULT_CLUSTER_THRESHOLD: float = 0.5


@dataclass
class SplitResult:
    """切分结果及统计。"""

    train: list[PeptideRecord] = field(default_factory=list)
    valid: list[PeptideRecord] = field(default_factory=list)
    test: list[PeptideRecord] = field(default_factory=list)
    n_clusters: int = 0
    largest_cluster: int = 0

    def summary(self) -> dict[str, float | int]:
        total = len(self.train) + len(self.valid) + len(self.test)
        return {
            "total": total,
            "train": len(self.train),
            "valid": len(self.valid),
            "test": len(self.test),
            "n_clusters": self.n_clusters,
            "largest_cluster": self.largest_cluster,
            "train_ratio": round(len(self.train) / total, 4) if total else 0.0,
        }


# ---------------------------------------------------------------------------
# 并查集
# ---------------------------------------------------------------------------
class UnionFind:
    """带路径压缩和按秩合并的并查集。

    路径压缩 + 按秩合并后，单次操作的均摊复杂度是 O(α(n))，
    α 是反阿克曼函数，在实际规模下几乎等于常数 1。
    对几十万条序列的聚类来说，这个常数直接决定了能不能在一台机器上跑完。
    """

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n

    def find(self, x: int) -> int:
        # 迭代版路径压缩：递归版在深链上可能触发 Python 的递归深度限制
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        """合并两个集合。已在同一集合时返回 False。"""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # 按秩合并：把小树挂到大树上，避免树退化成链
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


# ---------------------------------------------------------------------------
# 聚类
# ---------------------------------------------------------------------------
def cluster_sequences(
    records: Sequence[PeptideRecord],
    *,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    k: int = DEFAULT_KMER,
) -> list[list[int]]:
    """把序列聚成同源簇，返回每簇的下标列表。

    性能说明：倒排索引把候选对的数量从 O(n²) 降到 O(n × 平均候选数)。
    但注意这里比去重多了一步 Jaccard 计算——去重时一旦发现重复就提前
    退出，而聚类必须算完整相似度才能决定是否合并。
    所以在几十万条规模上，这一步是最耗时的（本项目的实测瓶颈就在这里）。
    真要上到百万级，应该换成 MinHash/LSH 近似，用一点精度换一个数量级的速度。
    """
    n = len(records)
    if n == 0:
        return []

    kmers = [kmer_set(r.sequence, k) for r in records]

    # 倒排索引：k-mer -> 序列下标
    inverted: dict[str, list[int]] = defaultdict(list)
    for i, ks in enumerate(kmers):
        for token in ks:
            inverted[token].append(i)

    uf = UnionFind(n)

    for i in range(n):
        overlap: dict[int, int] = defaultdict(int)
        for token in kmers[i]:
            for j in inverted[token]:
                if j > i:  # 只处理 j > i 的对，避免同一对被算两次
                    overlap[j] += 1
        for j, inter in overlap.items():
            union = len(kmers[i]) + len(kmers[j]) - inter
            if union and inter / union >= threshold:
                uf.union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)

    result = sorted(clusters.values(), key=len, reverse=True)
    logger.info(
        "同源聚类完成：%d 条序列 -> %d 个簇，最大簇 %d 条",
        n, len(result), len(result[0]) if result else 0,
    )
    return result


# ---------------------------------------------------------------------------
# 按簇切分
# ---------------------------------------------------------------------------
def split_by_clusters(
    records: Sequence[PeptideRecord],
    *,
    train_ratio: float = 0.9,
    valid_ratio: float = 0.05,
    test_ratio: float = 0.05,
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    seed: int = 42,
) -> SplitResult:
    """同源感知切分主入口。

    实现细节：切分前先打乱簇的顺序（固定 seed），
    否则同样大小的簇会按原始顺序被分到同一个划分里，
    而原始顺序通常跟数据来源相关（比如某个库的条目都在一起），
    会导致"测试集全部来自一个数据源"这种隐蔽的分布偏移。
    """
    total_ratio = train_ratio + valid_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"三个比例之和必须为 1，实际为 {total_ratio}")

    records_list = list(records)
    clusters = cluster_sequences(records_list, threshold=cluster_threshold)
    if not clusters:
        return SplitResult()

    rng = get_rng(seed)
    rng.shuffle(clusters)   # 打乱后再排序，保证同规模簇的顺序无偏
    clusters.sort(key=len, reverse=True)  # LPT 需要降序

    total = len(records_list)
    targets = {
        "train": train_ratio * total,
        "valid": valid_ratio * total,
        "test": test_ratio * total,
    }
    assigned = {"train": 0, "valid": 0, "test": 0}
    buckets: dict[str, list[int]] = {"train": [], "valid": [], "test": []}

    for cluster in clusters:
        # LPT：把当前簇放进"距离目标缺口最大"的那个划分。
        # 用 缺口/目标 而不是绝对缺口，是因为三个划分的目标规模不同——
        # test 目标只有 5%，绝对缺口永远最小，会被饿死。
        gaps = {
            name: (targets[name] - assigned[name]) / max(targets[name], 1.0)
            for name in targets
        }
        best = max(gaps, key=lambda name: gaps[name])
        buckets[best].extend(cluster)
        assigned[best] += len(cluster)

    result = SplitResult(
        train=[records_list[i] for i in buckets["train"]],
        valid=[records_list[i] for i in buckets["valid"]],
        test=[records_list[i] for i in buckets["test"]],
        n_clusters=len(clusters),
        largest_cluster=len(clusters[0]) if clusters else 0,
    )
    logger.info("切分完成: %s", result.summary())
    return result


def verify_no_leakage(
    train: Sequence[PeptideRecord],
    test: Sequence[PeptideRecord],
    *,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    k: int = DEFAULT_KMER,
    sample_size: int = 500,
) -> dict[str, float]:
    """验证切分没有泄漏：抽查 test 里的序列，看 train 里有没有近似同源。

    这是数据质量的单元测试，应该作为数据集产出的必过关卡。
    没有这一步，"同源感知切分"只是一个说法，没人知道它到底有没有生效。

    为控制耗时，只抽 `sample_size` 条 test 序列做检查；
    返回的 `max_similarity` 如果接近 1，说明切分逻辑有 bug，必须修。
    """
    if not train or not test:
        return {"checked": 0, "max_similarity": 0.0, "leak_rate": 0.0}

    rng = get_rng(0)
    probe = test if len(test) <= sample_size else rng.sample(list(test), sample_size)

    train_kmers = [kmer_set(r.sequence, k) for r in train]
    inverted: dict[str, list[int]] = defaultdict(list)
    for i, ks in enumerate(train_kmers):
        for token in ks:
            inverted[token].append(i)

    max_sim = 0.0
    leaked = 0
    for rec in probe:
        ks = kmer_set(rec.sequence, k)
        overlap: dict[int, int] = defaultdict(int)
        for token in ks:
            for j in inverted[token]:
                overlap[j] += 1
        best = 0.0
        for j, inter in overlap.items():
            union = len(ks) + len(train_kmers[j]) - inter
            if union:
                best = max(best, inter / union)
        max_sim = max(max_sim, best)
        if best >= threshold:
            leaked += 1

    leak_rate = leaked / len(probe)
    logger.info(
        "泄漏检查：抽查 %d 条，最高相似度 %.3f，疑似泄漏 %d 条（%.2f%%）",
        len(probe), max_sim, leaked, leak_rate * 100,
    )
    if leak_rate > 0.01:
        logger.warning("泄漏率超过 1%%，请检查聚类阈值或切分逻辑")
    return {
        "checked": len(probe),
        "max_similarity": round(max_sim, 4),
        "leak_rate": round(leak_rate, 4),
    }
