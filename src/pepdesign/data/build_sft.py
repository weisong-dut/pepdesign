"""构造 SFT 训练语料。

公开数据库只有天然好序列，没有指令-回答对。本模块用三种方式生成配对：
按序列属性反向构造 de_novo 指令；对好序列做定向劣化，得到
(劣化序列 -> 原序列) 的稳定性改造对；用规则引擎把可验算的事实转成设计理由。
劣化由程序完成，因此训练信号客观可验证。
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from pepdesign.bio.aa import HELIX_PROPENSITY, STABILITY_MOTIFS
from pepdesign.bio.cleavage import analyze, suggest_stabilization
from pepdesign.bio.cyclization import feasible_cyclic_type
from pepdesign.bio.sequence import mean_propensity, net_charge, standard_only
from pepdesign.data.schema import PeptideRecord
from pepdesign.prompts.templates import (
    build_sft_sample,
    de_novo_prompt,
    format_target_context,
    knowledge_prompt,
    multi_objective_prompt,
    stabilize_prompt,
)
from pepdesign.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. 定向劣化：制造配对样本
# ---------------------------------------------------------------------------
# 劣化策略清单。每条对应一种可被奖励函数检测到的设计缺陷，
# 以保证 SFT 负样本与 GRPO 奖励口径一致。
DEGRADE_STRATEGIES = (
    "add_protease_site",      # 把 P1' 位的 Pro 换掉，制造一个酶切位点
    "break_amphipathic",      # 在疏水面上插入极性残基，破坏两亲性
    "hydrophobic_run",        # 插入连续疏水片段，提升聚集倾向
    "charge_clash",           # 引入净电荷冲突，破坏静电匹配
    "mismatch_cyclic",        # 破坏环化约束（去掉成对的 Cys 或破坏 i/i+4 相位）
    "helix_breaker",          # 把螺旋倾向最高窗口里最强的一个螺旋促进者换成 Pro
)

#: 劣化策略 → 被破坏的设计维度。这是唯一事实来源：
#: `stabilize_prompt` 的 weak_dimension 与 `DIMENSION_TO_STRATEGY` 都从它派生。
STRATEGY_TO_DIMENSION: dict[str, str] = {
    "add_protease_site": "酶解稳定性",
    "break_amphipathic": "结构合理性",
    "hydrophobic_run": "成药性（溶解度与聚集）",
    "charge_clash": "结构合理性（静电匹配）",
    "mismatch_cyclic": "环化构象稳定性",
    "helix_breaker": "热稳定性",
}

#: 目标维度名 → 劣化策略，供多目标任务使用。由上一张表反转并显式补充别名。
DIMENSION_TO_STRATEGY: dict[str, str] = {
    **{dim: strat for strat, dim in STRATEGY_TO_DIMENSION.items()},
    "成药性": "hydrophobic_run",
    "结构合理性": "break_amphipathic",
    # charge_clash 破坏与带负电靶点的静电互补，对应"靶点亲和力"维度；
    # 否则该维度样本会回退到次高权重。
    "靶点亲和力": "charge_clash",
}


#: 保守替换分组：组内互换对侧链性质与结构影响最小。
#: Cys/Pro/Gly 刻意不在任何组里——Cys 关系二硫键配对，Pro/Gly 关系主链构象，
#: 动了它们会同时改变环化可行性与二级结构，已经超出"保守"的范畴。
_CONSERVATIVE_GROUPS: tuple[str, ...] = ("AVILM", "FYW", "ST", "KRH", "DE", "NQ")


def conservative_variant(seq: str, rng: random.Random) -> str | None:
    """在侧链性质相近的残基之间做一次随机替换，用于扩充环肽池。

    不改变 Cys 数量与位置，因此环化可行性不变；变体由调用方单独标记。
    """
    group_of = {aa: g for g in _CONSERVATIVE_GROUPS for aa in g}
    positions = [
        i for i, aa in enumerate(seq)
        if aa in group_of and len(group_of[aa]) > 1
    ]
    if not positions:
        return None
    pos = rng.choice(positions)
    aa = seq[pos]
    new = rng.choice([c for c in group_of[aa] if c != aa])
    return seq[:pos] + new + seq[pos + 1:]


def cyclic_balanced_pool(
    records: list[PeptideRecord],
    *,
    cyclic_ratio: float = 0.30,
    variant_weight: float = 0.25,
) -> list[PeptideRecord]:
    """按目标环肽比例构造加权记录池，对可行环肽过采样。

    `variant_weight` 给合成保守变体降权，避免变体在环肽答案中占比过高。
    代价是可行环肽序列的重复次数上升。
    """
    linear: list[PeptideRecord] = []
    real_cyclic: list[PeptideRecord] = []
    syn_cyclic: list[PeptideRecord] = []
    for rec in records:
        if feasible_cyclic_type(rec.sequence, rec.cyclic_type) == "none":
            linear.append(rec)
        elif rec.source == "synthetic":
            syn_cyclic.append(rec)
        else:
            real_cyclic.append(rec)
    if (not real_cyclic and not syn_cyclic) or not linear or cyclic_ratio <= 0:
        return list(records)
    cyclic_ratio = min(cyclic_ratio, 0.9)
    variant_weight = max(0.0, min(1.0, variant_weight))

    # 迭代求真实环肽的副本数 k：变体副本 = max(1, round(k * variant_weight))，
    # 使总环肽占比达到目标。
    k = 1
    while k < 200:
        syn_copies = max(1, round(k * variant_weight)) if syn_cyclic else 0
        total_cyc = len(real_cyclic) * k + len(syn_cyclic) * syn_copies
        if not linear or total_cyc / (len(linear) + total_cyc) >= cyclic_ratio:
            break
        k += 1
    syn_copies = max(1, round(k * variant_weight)) if syn_cyclic else 0
    return linear + real_cyclic * k + syn_cyclic * syn_copies


def strategy_for_dimension(dimension: str) -> str | None:
    """把"想劣化哪个维度"翻成策略名，认不出来返回 None。

    返回 None 而不是抛 KeyError：权重最高的维度可能还没有对应算子，
    调用方应退回次高权重而不是让整批构建失败。
    """
    return DIMENSION_TO_STRATEGY.get(dimension)


def degrade_sequence(
    seq: str,
    rng: random.Random,
    *,
    cyclic_type: str = "none",
    strategy: str | None = None,
) -> tuple[str, str] | None:
    """对一条好序列做定向劣化，返回 (劣化序列, 使用的策略)。

    没有可用位点时返回 None，不硬改；rng 显式传入以保证可复现。
    """
    s = standard_only(seq)
    if len(s) < 6:
        return None

    candidates = list(DEGRADE_STRATEGIES)
    if cyclic_type == "none":
        candidates.remove("mismatch_cyclic")   # 线性肽没有环化约束可破坏
    if strategy is not None:
        if strategy not in candidates:
            return None
        candidates = [strategy]
    rng.shuffle(candidates)

    for strat in candidates:
        result = _apply_degradation(s, strat, rng, cyclic_type)
        if result is not None:
            return result, strat
    return None


def _apply_degradation(
    s: str, strat: str, rng: random.Random, cyclic_type: str
) -> str | None:
    """执行单个劣化策略。返回 None 表示该策略在这条序列上不适用。"""
    chars = list(s)
    n = len(chars)

    if strat == "add_protease_site":
        # 找到"P1' 位是 Pro"的位置（即酶切位点已经被 Pro 保护住了），
        # 把那个 Pro 换成别的残基，就等于人为打开了一个酶切位点。
        protected = [
            i + 1 for i in range(n - 1)
            if s[i] in "KR" and s[i + 1] == "P"
        ]
        if not protected:
            return None
        pos = rng.choice(protected)
        # 替换成一个中性的、不会引入新位点的残基
        chars[pos] = rng.choice("AGSTV")
        return "".join(chars)

    if strat == "break_amphipathic":
        # 找螺旋疏水面上的一段，把中间的疏水残基换成极性残基，
        # 直接破坏两亲面的连续性
        from pepdesign.bio.sequence import hydrophobic_moment_profile

        if n < 8:
            return None
        window = min(9, n)
        profile = hydrophobic_moment_profile(s, window=window)
        if not profile:
            return None
        best_start = profile.index(max(profile))
        window_end = min(best_start + window, n)
        hydro_positions = [
            i for i in range(best_start, window_end) if s[i] in "AVILMFWY"
        ]
        if not hydro_positions:
            return None
        # 必须验证劣化确实降低了疏水矩：只换疏水残基不保证性质变差，
        # 未变差的样本在下游是纯噪声。
        base_max = max(profile)
        # 先试中间，再试同一窗口里的其他疏水位点，顺序固定以保证可复现
        order = [hydro_positions[len(hydro_positions) // 2], *hydro_positions]
        seen: set[int] = set()
        for pos in order:
            if pos in seen:
                continue
            seen.add(pos)
            cand = list(s)
            cand[pos] = rng.choice("NQST")
            cand_seq = "".join(cand)
            cand_max = max(hydrophobic_moment_profile(cand_seq, window=window))
            if cand_max < base_max - 1e-9:
                return cand_seq
        return None

    if strat == "hydrophobic_run":
        # 把一段极性残基整体换成疏水残基，制造连续疏水片段（>5 个）。
        # 同样要验证 aggregation_propensity 真的上升，而不是只看片段变长。
        from pepdesign.bio.sequence import aggregation_propensity

        if n < 9:
            return None
        starts = [
            i for i in range(n - 5)
            if all(s[j] in "DEKRNQSTG" for j in range(i, i + 5))
        ]
        if not starts:
            return None
        base_agg = aggregation_propensity(s)
        for start in rng.sample(starts, len(starts)):
            cand = list(chars)
            for j in range(start, start + 5):
                cand[j] = rng.choice("AILMFV")
            cand_seq = "".join(cand)
            if aggregation_propensity(cand_seq) > base_agg + 1e-9:
                return cand_seq
        return None

    if strat == "charge_clash":
        # 把多个正电荷残基替换成负电荷，制造净电荷的大幅偏移
        # （模拟"稳定性改造时误改了静电相互作用"这个真实的失败模式）
        pos = [i for i, c in enumerate(s) if c in "KR"]
        if len(pos) < 2:
            return None
        for i in rng.sample(pos, min(2, len(pos))):
            chars[i] = "E"
        if abs(net_charge("".join(chars))) > abs(net_charge(s)) + 3:
            return "".join(chars)
        return None

    if strat == "helix_breaker":
        # 在螺旋倾向最高的窗口里把最强促进者换成 Pro：Tm 模型中
        # pro_gly_ratio 系数 −18、helix_propensity 为 +22，换 Pro 同时压低两项。
        if n < 8:
            return None
        window = min(9, n)
        best_start, best_score = 0, -1.0
        for start in range(n - window + 1):
            score = sum(HELIX_PROPENSITY[c] for c in s[start : start + window]) / window
            if score > best_score:
                best_start, best_score = start, score

        # 只换本身是螺旋促进者的残基；换掉本来不倾向螺旋的残基不会改变性质。
        candidates = [
            i for i in range(best_start, best_start + window)
            if s[i] not in "PG" and HELIX_PROPENSITY[s[i]] >= 0.9
        ]
        if not candidates:
            return None
        pos = max(candidates, key=lambda i: HELIX_PROPENSITY[s[i]])

        chars[pos] = "P"
        degraded = "".join(chars)

        # 验证螺旋倾向确实下降，而不只是序列变了。
        if mean_propensity(degraded, "helix") >= mean_propensity(s, "helix"):
            return None
        return degraded

    if strat == "mismatch_cyclic":
        if cyclic_type == "disulfide":
            # 破坏二硫键：把一对 Cys 之一换掉，剩下奇数个 Cys
            cys = [i for i, c in enumerate(s) if c == "C"]
            if len(cys) < 2:
                return None
            chars[cys[0]] = "S"  # Ser 是 Cys 最常见的天然替换，生物上合理
            return "".join(chars)
        if cyclic_type in ("staple", "lactam"):
            # 破坏 i/i+4 的成对关系
            for i in range(n - 4):
                if (s[i] in "DE") and (s[i + 4] in "KR"):
                    chars[i] = "A"
                    return "".join(chars)
            return None
        if cyclic_type == "head_to_tail":
            # 头尾环化被破坏：把两端换成 Pro
            chars[0] = "P"
            chars[-1] = "P"
            return "".join(chars)
        return None

    return None


# ---------------------------------------------------------------------------
# 2. 规则引擎生成"设计理由"
# ---------------------------------------------------------------------------
def _skeleton_sentence(props: dict) -> str:
    if props.get("amphipathic_face", 0) > 0.4:
        return (
            f"序列形成清晰的两亲性螺旋面（疏水矩 "
            f"{props.get('max_hydrophobic_moment', 0):.2f}，"
            f"疏水面集中度 {props.get('amphipathic_face', 0):.2f}）"
        )
    return (
        f"序列以 {props.get('helix_propensity', 0):.2f} 的平均螺旋倾向"
        f"偏向柔性构象"
    )


def _charge_sentence(props: dict) -> str:
    q = props.get("net_charge_ph74", 0.0)
    if abs(q) >= 3:
        return f"pH 7.4 下净电荷 {q:+.1f}，有利于与带负电的靶点表面产生静电吸引"
    if abs(q) <= 1:
        return f"净电荷接近中性（{q:+.1f}），溶解性依赖极性侧链"
    # |q| 在 1~3 之间：既不构成明显优势也不构成风险，不写这一句
    return ""


def generate_reasoning(rec: PeptideRecord) -> str:
    """用规则引擎生成设计理由，只描述可由 compute_properties 复算的事实。

    不写因果性或机理性的推断，避免模型学会编造无法验证的解释。
    """
    props = rec.properties or rec.compute_properties()
    parts: list[str] = [_skeleton_sentence(props)]

    # --- 电荷 ---
    charge = _charge_sentence(props)
    if charge:
        parts.append(charge)

    # --- 稳定性相关的处理 ---
    report = analyze(rec.sequence, modifications=tuple(rec.modifications))
    if report.raw_site_count == 0:
        parts.append("序列不含常见血清蛋白酶的识别位点，酶解稳定性良好")
    elif report.raw_site_count <= 2:
        dom = report.dominant_protease
        parts.append(f"仅存在 {report.raw_site_count} 处潜在酶切位点（主要来自 {dom}）")
    else:
        # 位点多时给出改造方向，而不是声称它稳定。
        sugg = suggest_stabilization(rec.sequence, max_suggestions=1)
        hint = sugg[0]["reason"] if sugg else "建议对酶切位点做保护性改造"
        parts.append(f"存在 {report.raw_site_count} 处酶切位点，{hint}")

    # --- 聚集风险 ---
    agg = props.get("aggregation_propensity", 0.0)
    if agg > 0.5:
        parts.append(f"聚集倾向偏高（{agg:.2f}），需关注溶解度")
    if props.get("max_hydrophobic_run", 0) >= 5:
        parts.append(f"存在 {props['max_hydrophobic_run']} 个连续疏水残基，已接近风险阈值")

    # 限制在 3 句以内：训练模板里写的是"不超过三句话"，
    # 生成的样本必须遵守自己定的规范，否则模型学到的是不一致的格式
    return "；".join(parts[:3]) + "。"


#: 劣化策略 → 修复动作的描述模板。`{pos}` 是实际改动的位点（1-based），
#: `{bad}` / `{good}` 是劣化序列/答案序列在该位的残基。
_REPAIR_PHRASES: dict[str, str] = {
    "add_protease_site": "第 {pos} 位恢复为 {good}，重新封闭此前被打开的酶切位点",
    "break_amphipathic": "第 {pos} 位 {bad}→{good}，恢复疏水面的连续性",
    "hydrophobic_run": "第 {start}–{end} 位恢复极性残基，打断连续疏水片段",
    "charge_clash": "第 {pos} 位恢复 {good}，修正净电荷偏移",
    "mismatch_cyclic": "第 {pos} 位恢复 {good}，修复环化几何",
    "helix_breaker": "第 {pos} 位 {bad}→{good}，恢复螺旋倾向",
}


def describe_repair(bad_seq: str, good_seq: str, strategy: str) -> str:
    """把一次修复的实际 diff 写成一句话。认不出就返回空串。"""
    if len(bad_seq) != len(good_seq):
        return "恢复原始序列长度与环化几何"
    diff = [
        (i + 1, bad_seq[i], good_seq[i])
        for i in range(len(bad_seq))
        if bad_seq[i] != good_seq[i]
    ]
    if not diff:
        return ""
    template = _REPAIR_PHRASES.get(strategy)
    if template is None:
        return ""
    pos, bad, good = diff[0]
    return template.format(
        pos=pos, bad=bad, good=good,
        start=diff[0][0], end=diff[-1][0],
    )


def generate_repair_reasoning(
    rec: PeptideRecord, bad_seq: str, strategy: str
) -> str:
    """生成修复型理由：答案序列的可验算属性 + `describe_repair` 给出的实际 diff。

    不能复用 `generate_reasoning` 的通用建议，否则理由可能与实际改动无关。
    """
    props = rec.properties or rec.compute_properties()
    parts: list[str] = [_skeleton_sentence(props)]
    charge = _charge_sentence(props)
    if charge:
        parts.append(charge)
    action = describe_repair(bad_seq, rec.sequence, strategy)
    if action:
        parts.append(action)
    if len(parts) < 3:
        # 前两句不足以凑满三句时，补聚集风险（同样是可验算的属性）
        agg = props.get("aggregation_propensity", 0.0)
        if agg > 0.5:
            parts.append(f"聚集倾向偏高（{agg:.2f}），需关注溶解度")
    return "；".join(parts[:3]) + "。"


# ---------------------------------------------------------------------------
# 3. 三种任务样本的构造
# ---------------------------------------------------------------------------
def _length_range(
    rec: PeptideRecord, rng: random.Random | None = None
) -> tuple[int, int]:
    """由序列长度反推长度约束区间，默认在真实长度两侧留 ±2。

    传入 `rng` 时两侧余量随机抖动到 ±4，避免约束成为答案的精确描述。
    """
    n = rec.length
    if rng is None:
        return (max(5, n - 2), n + 2)
    # 区间宽于答案自身，避免答案恰好落在约束中点而被反推。
    return (max(5, n - rng.randint(0, 4)), n + rng.randint(0, 4))


#: 功能类别 → 中文名。SFT 与 GRPO 共用一份，避免两处各写一遍后漂移。
FUNCTION_CN: dict[str, str] = {
    "antibacterial": "抗菌", "antitumor": "抗肿瘤", "antiviral": "抗病毒",
    "antifungal": "抗真菌", "immunomodulatory": "免疫调节",
    "antiinflammatory": "抗炎", "cell_penetrating": "细胞穿膜",
    "targeting": "靶向", "inhibitor": "抑制剂", "other": "生物活性",
}


@dataclass(frozen=True)
class DenovoCandidate:
    """de_novo 检索索引里的一条候选（预存长度/电荷/可行环化类型）。"""

    rec: PeptideRecord
    length: int
    charge: float
    cyclic: str


@dataclass
class DenovoIndex:
    """de_novo 检索索引：按功能分组的候选 + 按环化类型预计算的采样权重。"""

    by_function: dict[str, list[DenovoCandidate]]
    #: cyclic → (functions, weights)。权重是"该 function 下可行该环化类型的记录数"。
    pools: dict[str, tuple[list[str], list[int]]]
    #: 每条序列被选为答案的次数；检索时按 1/(1+次数) 加权以抑制重复。
    usage: dict[str, int] = field(default_factory=dict)
    #: 全池长度分布；采样约束时先抽目标长度，避免 40–60 的长肽几乎不被选中。
    length_values: list[int] = field(default_factory=list)
    length_weights: list[int] = field(default_factory=list)


def build_denovo_index(records: list[PeptideRecord]) -> DenovoIndex:
    """按功能分组预计算 de_novo 检索所需的字段。

    预计算避免每次检索都重跑 `compute_properties` / `feasible_cyclic_type`。
    """
    by_function: dict[str, list[DenovoCandidate]] = {}
    for rec in records:
        cyclic = feasible_cyclic_type(rec.sequence, rec.cyclic_type)
        charge = float(
            rec.properties.get("net_charge_ph74", net_charge(rec.sequence))
        )
        by_function.setdefault(rec.function, []).append(
            DenovoCandidate(rec=rec, length=rec.length, charge=charge, cyclic=cyclic)
        )

    pools: dict[str, tuple[list[str], list[int]]] = {}
    for cyclic in ("none", "disulfide"):
        funcs: list[str] = []
        weights: list[int] = []
        for f, items in by_function.items():
            n = sum(1 for c in items if c.cyclic == cyclic)
            if n > 0:
                funcs.append(f)
                weights.append(n)
        pools[cyclic] = (funcs, weights)

    len_counts = Counter(c.length for items in by_function.values() for c in items)
    return DenovoIndex(
        by_function=by_function,
        pools=pools,
        length_values=sorted(len_counts),
        length_weights=[len_counts[n] for n in sorted(len_counts)],
    )


def sample_denovo_constraints(
    rng: random.Random,
    *,
    length_values: list[int] | None = None,
    length_weights: list[int] | None = None,
) -> tuple[int, int, float, float]:
    """独立于任何序列采样一组设计约束（长度区间、电荷区间）。

    传入长度分布时先按整体分布抽目标长度，保证长肽也有被选中的机会。
    """
    if length_values:
        target = rng.choices(length_values, weights=length_weights, k=1)[0]
        lo = max(5, target - rng.randint(1, 6))
        hi = min(60, target + rng.randint(1, 6))
    else:
        lo = rng.randint(5, 50)
        hi = min(60, lo + rng.randint(2, 10))
    q_lo = round(rng.uniform(-4.0, 4.0) * 2) / 2
    q_hi = min(8.0, q_lo + round(rng.uniform(2.0, 6.0) * 2) / 2)
    return lo, hi, q_lo, q_hi


def search_denovo_candidate(
    index: DenovoIndex,
    rng: random.Random,
    *,
    max_attempts: int = 20,
    cyclic_ratio: float = 0.30,
) -> tuple[PeptideRecord, str, tuple[int, int], tuple[float, float]] | None:
    """约束先行检索：返回 `(记录, 环化类型, 长度区间, 电荷区间)`。

    SFT 与 GRPO 共用这一份检索逻辑——两阶段面对的"需求"必须同源，
    否则 SFT 教的是"约束内选择"、RL 却在另一套约束分布上爬坡。
    """
    if not index.by_function:
        return None

    for _ in range(max_attempts):
        cyclic = "disulfide" if rng.random() < cyclic_ratio else "none"
        # 按池大小加权选功能类别，不能均匀选：function 分布极不均，
        # 均匀选择会把小类别放大成主要任务；权重与环化类型绑定。
        funcs, weights = index.pools[cyclic]
        if not funcs:
            continue
        function = rng.choices(funcs, weights=weights, k=1)[0]
        lo, hi, q_lo, q_hi = sample_denovo_constraints(
            rng,
            length_values=index.length_values,
            length_weights=index.length_weights,
        )

        # 无候选时放宽约束而不是换答案迁就；同时要求候选数 >= 5，
        # 避免窄约束与小候选集变成隐性记忆通道。
        best: tuple[list[DenovoCandidate], int, int, float, float] | None = None
        for widen in range(6):
            lo_c = max(5, lo - 2 * widen)
            h = min(60, hi + 2 * widen)
            ql = max(-8.0, q_lo - widen)
            qh = min(8.0, q_hi + widen)
            cands = [
                c for c in index.by_function[function]
                if c.cyclic == cyclic
                and lo_c <= c.length <= h
                and ql - 1e-9 <= c.charge <= qh + 1e-9
            ]
            if not cands:
                continue
            if best is None:
                best = (cands, lo_c, h, ql, qh)
            if len(cands) >= 5:
                best = (cands, lo_c, h, ql, qh)
                break
        if best is None:
            continue
        cands, lo_c, h, ql, qh = best
        # 使用次数越多权重越低，避免短序列因满足更多约束而被反复选中。
        weights = [1.0 / (1.0 + index.usage.get(c.rec.sequence, 0)) for c in cands]
        chosen = rng.choices(cands, weights=weights, k=1)[0]
        index.usage[chosen.rec.sequence] = index.usage.get(chosen.rec.sequence, 0) + 1
        return chosen.rec, cyclic, (lo_c, h), (round(ql, 1), round(qh, 1))
    return None


def build_denovo_sample(
    index: DenovoIndex,
    rng: random.Random,
    *,
    max_attempts: int = 20,
    cyclic_ratio: float = 0.30,
) -> dict | None:
    """约束先行：采样需求 → 在库里检索满足约束的序列 → 生成样本。

    约束先于答案存在，答案只是其中一个满足者，模型无法从约束反推唯一序列。
    """
    found = search_denovo_candidate(
        index, rng, max_attempts=max_attempts, cyclic_ratio=cyclic_ratio
    )
    if found is None:
        return None
    rec, cyclic, (lo_c, h), (ql, qh) = found

    target_name = rec.target or _infer_target_name(rec)
    motif = _infer_binding_motif(rec)
    target_desc = format_target_context(
        target_name="", target_sequence="", binding_motif=motif
    ).replace("靶点名称：\n", "").strip()

    spec = de_novo_prompt(
        target_name=target_name,
        target_desc=target_desc,
        function=FUNCTION_CN.get(rec.function, "生物活性"),
        length_range=(lo_c, h),
        cyclic_type=cyclic,
        net_charge_range=(ql, qh),
        extra_constraints=_extra_constraints(rec),
    )
    return build_sft_sample(
        spec, sequence=rec.sequence, reasoning=generate_reasoning(rec)
    )


def build_stabilize_sample(
    rec: PeptideRecord, rng: random.Random
) -> tuple[dict, str] | None:
    """构造"稳定性优化"样本，返回 (样本, 劣化策略)。

    流程：好序列 -> 定向劣化 -> 把劣化后的序列作为输入、原序列作为答案。
    这样模型学到的就是"识别并修复具体缺陷"，而不是泛泛的"重新设计"。
    """

    degraded = degrade_sequence(
        rec.sequence, rng, cyclic_type=feasible_cyclic_type(rec.sequence, rec.cyclic_type)
    )
    if degraded is None:
        return None
    bad_seq, strategy = degraded

    # 指令里的目标维度必须由劣化策略决定，否则会教模型做错误归因。
    dimension = STRATEGY_TO_DIMENSION[strategy]

    spec = stabilize_prompt(
        sequence=bad_seq,
        target_name=rec.target or _infer_target_name(rec),
        weak_dimension=dimension,
        preserve="靶点亲和力" if strategy != "break_amphipathic" else "整体活性",
        cyclic_type=feasible_cyclic_type(rec.sequence, rec.cyclic_type),
    )
    reasoning = generate_repair_reasoning(rec, bad_seq, strategy)
    sample = build_sft_sample(spec, sequence=rec.sequence, reasoning=reasoning)
    # 记录实际劣化策略，便于在数据层直接审计 prompt 与缺陷是否匹配。
    sample.setdefault("meta", {})["degrade_strategy"] = strategy
    return sample, strategy


#: 多目标任务的权重预设。SFT 与 GRPO 共用这一份，保证权重语义一致。
MULTI_OBJECTIVE_PRESETS: tuple[dict[str, float], ...] = (
    {"热稳定性": 0.4, "酶解稳定性": 0.4, "靶点亲和力": 0.2},
    {"热稳定性": 0.2, "酶解稳定性": 0.3, "靶点亲和力": 0.5},
    {"热稳定性": 0.35, "酶解稳定性": 0.25, "靶点亲和力": 0.25, "成药性": 0.15},
    {"热稳定性": 0.15, "酶解稳定性": 0.15, "靶点亲和力": 0.6, "成药性": 0.1},
    {"热稳定性": 0.3, "酶解稳定性": 0.3, "靶点亲和力": 0.2, "成药性": 0.2},
)


def build_multi_objective_sample(rec: PeptideRecord, rng: random.Random) -> tuple[dict, str] | None:
    """构造"多目标协同优化"样本。

    劣化策略跟随主导权重，使权重排序成为真实约束：最高权重维度就是缺陷所在。
    """
    objectives = MULTI_OBJECTIVE_PRESETS[rng.randrange(len(MULTI_OBJECTIVE_PRESETS))]

    # 按权重降序取第一个既有算子又能生效的维度；最高权重维度未必有算子。
    degraded = None
    strategy = None
    for dimension, _weight in sorted(objectives.items(), key=lambda kv: -kv[1]):
        candidate = strategy_for_dimension(dimension)
        if candidate is None:
            continue
        result = degrade_sequence(
            rec.sequence,
            rng,
            cyclic_type=feasible_cyclic_type(rec.sequence, rec.cyclic_type),
            strategy=candidate,
        )
        if result is not None:
            degraded, strategy = result
            break
    if degraded is None:
        return None
    bad_seq = degraded

    spec = multi_objective_prompt(
        sequence=bad_seq,
        objectives=objectives,
        target_name=rec.target or _infer_target_name(rec),
        cyclic_type=feasible_cyclic_type(rec.sequence, rec.cyclic_type),
    )
    reasoning = generate_repair_reasoning(rec, bad_seq, strategy)
    sample = build_sft_sample(spec, sequence=rec.sequence, reasoning=reasoning)
    sample.setdefault("meta", {})["degrade_strategy"] = strategy
    return sample, strategy


# ---------------------------------------------------------------------------
# 4. 知识问答样本（把 SOP 固化成文本能力）
# ---------------------------------------------------------------------------
KNOWLEDGE_QA: tuple[tuple[str, str], ...] = (
    (
        "多肽序列设计的标准流程是什么？",
        "标准流程分五步：①靶点需求分析，明确功能方向、靶点特征与约束条件；"
        "②骨架选型，在 α-螺旋、β-折叠、线性柔性等骨架中按功能需求选择；"
        "③序列生成，基于骨架与约束生成候选序列；④初筛评估，用计算工具预测"
        "热稳定性、酶解稳定性、靶点亲和力与结构合理性；⑤稳定性优化，"
        "针对短板维度做定向改造，常用手段见稳定性改造策略清单。",
    ),
    (
        "评价一条多肽序列质量的四个核心维度是什么？",
        "①热稳定性，以解链温度 Tm 为主要指标；②酶解稳定性，以血清蛋白酶"
        "识别位点数量与血清半衰期衡量；③靶点亲和力，基于结合特征预测；"
        "④结构合理性，检查二级结构置信度、疏水平衡与电荷分布。"
        "四个维度之间存在制衡关系，需要按研发目标设定权重做协同优化。",
    ),
    (
        "如何提升多肽的酶解稳定性？",
        "常用策略：①在酶切位点的 P1' 位引入 Pro，利用蛋白酶无法切割"
        "Xaa-Pro 键的特性阻断该位点；②N 端乙酰化或 C 端酰胺化，"
        "封闭末端以抵抗氨肽酶与羧肽酶；③在关键位点使用 D-型氨基酸或"
        "非天然氨基酸，破坏蛋白酶识别构象；④主链 N-甲基化；"
        "⑤头尾环化消除两端切口；⑥PEG 化降低蛋白酶可及性。",
    ),
    (
        "为什么环化能提升多肽的稳定性？",
        "环化同时带来两个效应：①消除游离的 N 端与 C 端，"
        "直接屏蔽氨肽酶与羧肽酶的外切作用；②刚性化主链构象，"
        "降低构象熵、提升结合前的预组织程度，既提高 Tm 也提高亲和力。"
        "代价是合成难度上升，且过短的序列环张力过大会导致环化效率低。",
    ),
    (
        "为什么强化学习阶段要用 KL 约束？",
        "KL 约束限制新策略偏离参考策略（SFT 模型）的距离。"
        "没有这个约束，模型会为了最大化奖励把序列分布推到一个"
        "完全不像天然肽的区域——奖励是预测器的近似，"
        "近似误差在分布外会急剧放大，模型会利用这些误差钻空子。"
        "KL 项相当于给优化路径加了一条「不要走太远」的护栏。",
    ),
    (
        "20 种标准氨基酸按侧链性质如何分类？",
        "疏水脂肪族：Ala、Val、Ile、Leu、Met；芳香族：Phe、Trp、Tyr；"
        "极性不带电：Ser、Thr、Cys、Asn、Gln；酸性：Asp、Glu；"
        "碱性：Lys、Arg、His；特殊构象：Gly（无侧链、柔性最高）、Pro（环状侧链、破坏螺旋）。"
        "分类不是目的，作用是让替换时能判断「性质相近」与「性质突变」。",
    ),
    (
        "二硫键在什么条件下形成，又会在什么条件下断裂？",
        "两个 Cys 的巯基在氧化环境中脱氢形成 S–S 键，要求两个残基在三维空间上足够接近。"
        "细胞外与内质网是氧化环境，有利于成键；胞内还原环境、以及还原剂（DTT、β-巯基乙醇）"
        "会把它断开。序列层面只能数 Cys 个数，能否成键还取决于构象。",
    ),
    (
        "多肽环化有哪几种常见方式，各自代价是什么？",
        "①头尾环化：N 端与 C 端形成酰胺键，消除两端切口；"
        "②二硫键环化：两对 Cys 氧化成键，构象更接近天然；"
        "③侧链订书钉（staple）：i/i+4 或 i/i+7 的非天然桥接，能稳定 α-螺旋。"
        "代价依次是合成难度上升、对 Cys 数量与位置敏感、以及非天然单体成本高。",
    ),
    (
        "血清中常见的蛋白酶有哪些，识别规律是什么？",
        "Trypsin 类在 Lys/Arg 的羧基侧切割；chymotrypsin 类在 Phe/Tyr/Trp 之后切割；"
        "pepsin 在酸性环境偏爱疏水残基之间；elastase 偏小侧链残基。"
        "设计时先数序列里的 K/R/F/Y/W，它们是最主要的切割风险位点。",
    ),
    (
        "为什么在 P1' 位引入 Pro 能提高酶解稳定性？",
        "Pro 的环状侧链使 Xaa–Pro 键的构象受限，trypsin/chymotrypsin 的活性口袋"
        "难以把它摆成可切割的构象。注意这是「阻断」而不是「消除」："
        "位点仍在，但在血清中的切割速率显著下降。多处连续引入 Pro 会破坏二级结构，需权衡。",
    ),
    (
        "净电荷如何影响多肽的溶解性与膜结合？",
        "净电荷绝对值大（通常 |q| > 3）意味着水溶性好，但也可能与带相反电荷的"
        "非靶点表面发生非特异结合；接近中性的序列溶解性依赖极性侧链，"
        "更容易聚集。抗菌肽常带正电，是为了与带负电的细菌膜表面产生静电吸引。",
    ),
    (
        "疏水矩（hydrophobic moment）衡量的是什么？",
        "把疏水残基的疏水性按螺旋周期投影到圆柱面上，得到的向量长度。"
        "高疏水矩说明疏水残基集中在一侧，形成两亲性螺旋面——这正是膜结合与"
        "蛋白-蛋白界面常见的结构特征。它和 GRAVY 不同：GRAVY 只看总体疏水程度，"
        "疏水矩看的是「分布是否均匀」。",
    ),
    (
        "GRAVY 指数的含义是什么？",
        "所有残基疏水指数的算术平均。正值表示整体偏疏水，负值表示偏亲水。"
        "它是一个粗粒度指标，无法区分「疏水残基集中在一侧」与「均匀分布」——"
        "后者用疏水矩衡量。",
    ),
    (
        "哪些序列特征会提高聚集倾向？",
        "①连续 5 个以上的疏水残基；②高 β-折叠倾向的片段；③净电荷接近中性；"
        "④芳香族残基密集。降低聚集的常见做法是打断疏水片段、引入带电残基、"
        "或做末端封闭，但每一步都要检查是否损伤靶点结合面。",
    ),
    (
        "螺旋倾向如何影响热稳定性？",
        "螺旋倾向高的残基（Ala、Leu、Met 等）有利于形成 α-螺旋，而 Pro、Gly 会"
        "打断或降低螺旋含量。热稳定性与二级结构含量相关，但不是唯一因素："
        "盐桥、二硫键、表面电荷分布同样影响 Tm。用单一指标下结论时要说清楚这是代理指标。",
    ),
    (
        "末端乙酰化与酰胺化分别解决什么问题？",
        "N 端乙酰化封闭游离氨基，抵抗氨肽酶的外切；C 端酰胺化封闭羧基，"
        "抵抗羧肽酶，同时去掉一个负电荷、略微提高净正电。"
        "两者都不改变序列主体，是对「末端被啃掉」这一类降解最直接的处理。",
    ),
    (
        "非天然氨基酸能带来什么，代价是什么？",
        "D-型氨基酸、N-甲基化、β-氨基酸可以破坏蛋白酶识别的构象，"
        "显著提高血清稳定性；N-甲基化还能增加膜渗透性。"
        "代价是固相合成难度与成本上升，而且很多计算预测器只在 20 种标准残基上训练过，"
        "对非天然残基的预测并不可信。",
    ),
    (
        "多肽的固相合成大致流程是什么，为什么长肽和环肽更难？",
        "Fmoc 固相合成：树脂上逐轮「脱保护—偶联—洗涤」，最后裂解并纯化。"
        "每一步偶联效率通常低于 100%，步骤越多总收率越低（长肽的典型瓶颈）；"
        "环化还要求线性前体在极稀溶液中成环以抑制分子间聚合，因此环肽的"
        "收率与纯度通常低于同长度线性肽。",
    ),
    (
        "如何表征一条合成多肽的纯度与结构？",
        "HPLC 测纯度与保留时间，质谱确认分子量，圆二色谱（CD）看二级结构含量，"
        "必要时用 NMR 或晶体结构确认构象。活性与稳定性结论必须建立在这些表征之上，"
        "否则无法区分「设计有效」与「样品本身不合格」。",
    ),
    (
        "环肽为什么可能比线性肽更容易透过细胞膜？",
        "环化减少了可旋转键与构象熵，主链酰胺可以形成分子内氢键、降低极性表面积，"
        "从而减少跨膜时的去溶剂化代价。但这个效应依赖于环的大小与N-甲基化程度，"
        "分子量过大的环肽仍然难以透膜——「环化」不是膜渗透的充分条件。",
    ),
)


def build_knowledge_samples() -> list[dict]:
    """把内置 SOP 问答转成 SFT 样本，用于领域规范对齐。"""

    samples = []
    for question, answer in KNOWLEDGE_QA:
        spec = knowledge_prompt(question=question)
        samples.append(
            {
                "messages": [
                    *spec.as_messages(),
                    {"role": "assistant", "content": answer},
                ],
                "task_type": "knowledge",
                "meta": {"question": question},
            }
        )
    return samples


# ---------------------------------------------------------------------------
# 5. 辅助推断（在没有标注时给出合理的占位）
# ---------------------------------------------------------------------------
def _infer_target_name(rec: PeptideRecord) -> str:
    """没有靶点标注时的兜底名称：按功能类别生成泛化描述，不留空。"""
    mapping = {
        "antibacterial": "细菌细胞膜（革兰阴性菌外膜）",
        "antitumor": "肿瘤细胞表面受体",
        "antiviral": "病毒包膜蛋白",
        "antifungal": "真菌细胞膜",
        "cell_penetrating": "细胞膜（穿膜需求）",
        "immunomodulatory": "免疫细胞表面受体",
    }
    return mapping.get(rec.function, "目标结合位点")


def _infer_binding_motif(rec: PeptideRecord) -> str:
    """从序列推断可能的功能基序：取两亲性最强的 7 残基窗口。

    这是推断而非标注，生成理由时应注明。
    """
    from pepdesign.bio.sequence import hydrophobic_moment_profile

    s = standard_only(rec.sequence)
    if len(s) < 7:
        return s
    profile = hydrophobic_moment_profile(s, window=7)
    if not profile:
        return s
    start = profile.index(max(profile))
    return s[start : start + 7]


def _charge_range(
    rec: PeptideRecord, rng: random.Random | None = None
) -> tuple[float, float] | None:
    """由序列的实测净电荷反推约束区间，默认两侧留 ±2。

    传入 `rng` 时余量随机抖动，使同一记录能生成多个不同 prompt。
    """
    q = rec.properties.get("net_charge_ph74", net_charge(rec.sequence))
    if rng is None:
        return (round(q - 2, 1), round(q + 2, 1))
    # 与长度同理：±1~4 的随机余量，避免约束成为答案的"精确描述"
    return (round(q - rng.randint(1, 4), 1), round(q + rng.randint(1, 4), 1))


def _extra_constraints(rec: PeptideRecord) -> str:
    """把领域标注转成自然语言的额外约束。"""
    parts = []
    # 只保留改造算子认识的修饰；UniProt 的原始修饰名与 STABILITY_MOTIFS
    # 对不上，原样写入会让 prompt 要求一个答案里不会出现的约束。
    recognized = [STABILITY_MOTIFS[m] for m in rec.modifications if m in STABILITY_MOTIFS]
    if recognized:
        parts.append(f"需要应用以下修饰：{'; '.join(recognized)}")
    agg = rec.properties.get("aggregation_propensity", 0.0)
    if agg > 0.5:
        parts.append("注意控制疏水片段长度以降低聚集风险")
    if rec.organism:
        parts.append(f"参考来源物种的序列特征（{rec.organism}）")
    return "；".join(parts)
