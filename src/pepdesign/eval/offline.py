"""离线指标：预测值与实测值之间可算的那些数。

纯 numpy + 标准库，不 import torch / scipy / sklearn / pandas。
这不是洁癖：本层的全部输入都是 `list[float]`，用到的数学是排序和协方差，
numpy 已经够了；而多一个依赖就多一个"这台机器上跑不了评估"的理由。
（`pyproject.toml` 里原本给 eval 额外声明了 scikit-learn/scipy/pandas，
实际一个都用不上，已经按这条约定删掉。）

秩相关（Spearman）而非 Pearson
--------------------------------------------------
奖励模型是个启发式代理，它只声称"这两条序列谁更好"的顺序，
不声称"Tm 差 3.2℃"这种量纲上的准确。所以该考的也是顺序对不对。
而且实测数据里量纲不统一（MIC 是 μg/mL、Tm 是 ℃）、常有离群点，
Pearson 会被一个离群点整个带偏，秩相关不会。

置换检验而非 t 检验
--------------------------------------
1. n 小（本项目实测只有个位数），正态假设不成立；
2. 实测值里有并列（两条序列 Tm 都是 64.6），t 检验对此没有好的处理；
3. 置换检验不需要 scipy，也不需要对分布做任何假设。

关于本模块最重要的那条约定：样本量不足时拒绝判定
-------------------------------------------------------
`correlation_report` 在 n < `min_n_for_significance` 时把结论标成
`conclusive=False`，并把 p 值一起交出来但明确说明它不构成结论。
理由是这类报告最常见的误用方式：拿到一个 0.03 的 p 值就去改奖励函数，
而那个 p 值来自 6 条带 placeholder 标记的数据。

具体地说，小样本的问题不是"算不出小 p 值"——n=6 且完全单调时，
置换检验本来就会给出 p = 2/720 ≈ 0.003。问题在于：
  * 没有功效：n=6 时除非相关性接近完美，否则 p 都很大，
    "不显著"这件事几乎不携带信息（既不能证明无关，也不能证明有关）；
  * 一点扰动就翻盘：改一个数、或把某条序列从并列里挪出来，
    结论可能整个反过来；
  * 并列会稀释：实测值有并列时，秩相关能达到的最大值就不到 1，
    置换分布随之变宽，p 值系统性偏大。
所以本模块报的是"统计量 + 样本量 + 是否够格下结论"，而不是一个光秃秃的
p 值。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from pepdesign.data.schema import ASSAY_ORIENTATION, orientation_for

#: 低于这个样本量就不下结论。20 是秩相关开始有实际功效的量级
#: （n=20、真实 ρ=0.6 时，双侧 α=0.05 的检验功效约 0.8）。
#: 这个数不是统计定理，是一个工程阈值：它必须大于本项目真实拥有的
#: 实测数据量（6 条），否则这条保护形同虚设。
MIN_N_FOR_SIGNIFICANCE = 20

# `ASSAY_ORIENTATION` / `orientation_for` 在 `data/schema.py` 里定义，
# 这里 import 进来继续对外暴露（`from pepdesign.eval.offline import orientation_for`
# 仍然可用），但不在这里另写一份：方向表是领域知识，
# 而且 `PeptideRecord.best_activity` 也要用它。两份表的后果是
# "best_activity 和评估层对同一个测定类型的方向理解不一致"——
# 那会让报告里的结论和数据集里取出来的值相互矛盾，且两处都看不出问题。
__all__ = [
    "ASSAY_ORIENTATION",
    "MIN_N_FOR_SIGNIFICANCE",
    "average_ranks",
    "correlation_report",
    "count_ties",
    "mae",
    "orientation_for",
    "pearson_r",
    "permutation_p_value",
    "rmse",
    "spearman_ic",
    "top_k_enrichment",
]


# ---------------------------------------------------------------------------
# 基本量
# ---------------------------------------------------------------------------
def _as_arrays(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """校验成两个等长的 float 数组。

    长度不等直接报错而不是截断到较短的一侧：截断会让"预测值和实测值
    错位"这件事变成"少几个样本"，而错位的相关性可以轻松算出 -0.9
    这种看起来很有说服力的数——它测的是两条序列的分数被错配之后的
    巧合关系。
    """
    ax = np.asarray(x, dtype=np.float64)
    ay = np.asarray(y, dtype=np.float64)
    if ax.shape != ay.shape:
        raise ValueError(
            f"预测值与实测值必须一一对应，长度分别是 {ax.size} 和 {ay.size}。"
            f"（错位后的相关性照样能算出来，但测的是错配关系。）"
        )
    if ax.ndim != 1:
        raise ValueError(f"只接受一维序列，收到 shape={ax.shape}")
    return ax, ay


def average_ranks(values: np.ndarray) -> np.ndarray:
    """秩，并列取平均秩（标准做法）。

    并列必须处理：本项目 6 条实测里就有两条 Tm 都是 64.6。
    给并列值随便分配先后（比如按出现顺序），会让相关系数取决于
    数据在文件里的排列顺序——同一份数据换个行序就得到不同的结论。
    """
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]

    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        # 并列段 [i, j] 共享平均秩 (i + j) / 2
        ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman_ic(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman 秩相关系数（信息系数 IC）。

    任一序列的秩为常数（全是同一个值）时返回 0 而不是 nan：
    nan 会一路传播到报告里变成一个 `NaN`，而"这批预测没有任何区分度"
    本身就是个需要被看见的结论，不该以缺失值的形式出现。
    """
    ax, ay = _as_arrays(x, y)
    if ax.size < 2:
        return 0.0
    rx, ry = average_ranks(ax), average_ranks(ay)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson_r(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson 相关系数。留作对照：它比秩相关更容易被离群点带偏。"""
    ax, ay = _as_arrays(x, y)
    if ax.size < 2 or ax.std() < 1e-12 or ay.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(ax, ay)[0, 1])


def mae(pred: Sequence[float], true: Sequence[float]) -> float:
    """平均绝对误差。量纲和实测值一致（MIC 就是 μg/mL）。"""
    ax, ay = _as_arrays(pred, true)
    return float(np.abs(ax - ay).mean()) if ax.size else 0.0


def rmse(pred: Sequence[float], true: Sequence[float]) -> float:
    ax, ay = _as_arrays(pred, true)
    return float(np.sqrt(((ax - ay) ** 2).mean())) if ax.size else 0.0


def top_k_enrichment(
    scores: Sequence[float], labels: Sequence[float], *, k: int, higher_is_better: bool = True
) -> dict[str, Any]:
    """Top-K 富集：按预测分排序取前 K 个，看里面有多少是"真的好的"。

    为什么除了相关系数还要看这个：研发实际只会去合成前 K 条序列，
    "整体相关性不错但前 20 条全是废的"是真实存在的失败模式，
    而这两种失败（相关但排不到前面 vs 单纯不相关）的处置完全不同。

    `labels` 是 0/1（1 = 好）。`k` 大于样本数时取全部。
    """
    ax, ay = _as_arrays(scores, labels)
    if ax.size == 0:
        return {"k": 0, "hits": 0, "precision": 0.0, "recall": 0.0, "n": 0}
    k = max(1, min(int(k), ax.size))
    # argsort 默认升序；"越大越好"时取末尾 k 个。
    order = np.argsort(ax, kind="stable")
    picked = order[-k:] if higher_is_better else order[:k]
    hits = int(ay[picked].sum())
    positives = int(ay.sum())
    return {
        "k": k,
        "hits": hits,
        "precision": hits / k,
        "recall": (hits / positives) if positives else 0.0,
        "n": int(ax.size),
    }


# ---------------------------------------------------------------------------
# 显著性：置换检验
# ---------------------------------------------------------------------------
def permutation_p_value(
    x: Sequence[float],
    y: Sequence[float],
    *,
    n_perm: int = 5000,
    seed: int = 42,
) -> float:
    """秩相关的双侧置换检验 p 值。

    做法就是原假设的字面意思：打乱 y 的顺序（等价于"预测和实测无关"），
    重算 n_perm 次统计量，看有多少次能超过观测值。
    用 `(count + 1) / (n_perm + 1)` 而不是 `count / n_perm`：
    后者可能给出恰好 0，而 0 会被读成"不可能是巧合"——
    置换检验能分辨的最小 p 值是 1/(n_perm+1)，加一才是诚实的写法。

    单看这个数不要下结论，配合 `correlation_report` 使用：它会把样本量
    够不够这件事一并报出来。
    """
    ax, ay = _as_arrays(x, y)
    if ax.size < 2:
        return 1.0
    observed = abs(spearman_ic(ax, ay))
    if observed < 1e-12:
        return 1.0
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(int(n_perm)):
        if abs(spearman_ic(ax, rng.permutation(ay))) >= observed - 1e-12:
            exceed += 1
    return (exceed + 1) / (int(n_perm) + 1)


def count_ties(values: Sequence[float]) -> int:
    """并列值的个数（重复出现的观测值数量）。

    单独报出来是因为它同时影响两件事：秩相关能达到的上限，
    以及置换检验的分布。6 条实测里就有两条 Tm 相同。
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0
    _, counts = np.unique(arr, return_counts=True)
    return int(counts[counts > 1].sum())


# ---------------------------------------------------------------------------
# 汇总报告
# ---------------------------------------------------------------------------
def correlation_report(
    pred: Sequence[float],
    measured: Sequence[float],
    *,
    min_n: int = MIN_N_FOR_SIGNIFICANCE,
    n_perm: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    """把上面那些数整理成一份自带可信度声明的报告。

    返回值里有两个字段是刻意加的：
      * `conclusive` —— n 够不够下结论。不够时下面的 p 值仍然给出
        （信息不藏），但调用方不该拿它做决定；
      * `caveats` —— 为什么不够。直接写成人能读的句子，
        因为这些字最后会出现在报告里，而不是只留在注释里。
    """
    ax, ay = _as_arrays(pred, measured)
    n = int(ax.size)
    ties = count_ties(ay)

    caveats: list[str] = []
    if n < min_n:
        caveats.append(
            f"样本量 n={n} 低于阈值 {min_n}，置换检验没有足够功效："
            f"此处的 p 值不构成结论（无论显著还是不显著）。"
            f"低于阈值时'不显著'几乎不携带信息——既不能证明无关，"
            f"也不能证明有关。"
        )
    if ties:
        caveats.append(
            f"实测值里有 {ties} 个并列（重复值），秩相关能达到的上限因此"
            f"低于 1，置换检验的 p 值会系统性偏大。"
        )
    if n and ay.std() < 1e-12:
        caveats.append("实测值全部相同，没有任何可解释的变异，相关性无定义（已记 0）。")

    return {
        "n": n,
        "spearman": round(spearman_ic(ax, ay), 4),
        "pearson": round(pearson_r(ax, ay), 4),
        "mae": round(mae(ax, ay), 4),
        "rmse": round(rmse(ax, ay), 4),
        "p_value": round(permutation_p_value(ax, ay, n_perm=n_perm, seed=seed), 4),
        "n_perm": int(n_perm),
        "ties_in_measured": ties,
        "min_n_for_significance": int(min_n),
        "conclusive": bool(n >= min_n),
        "caveats": caveats,
    }
