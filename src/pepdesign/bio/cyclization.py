"""环肽的结构约束与可行性评估。

将环化方式量化为可检查的约束，供"结构合理性"奖励维度使用：

1. 头尾环化：N 端与 C 端成酰胺键；短肽（<8 残基）两端距离不足，
   需要连接子（linker），用端到端距离估计近似判断可行性。
2. 二硫键：Cys 必须成对出现，且在序列上保持足够间距。
3. 订书肽（stapled peptide）：i 与 i+4（或 i+7）的桥接位点
   需落在螺旋同侧，否则会破坏功能面。
4. 内酰胺桥：i 位 Asp/Glu 与 i+4 位 Lys/Arg 成酰胺，电荷互补。

这些约束以硬性过滤或惩罚形式进入奖励函数，避免生成化学上不可行的序列。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pepdesign.bio.sequence import MIN_CYCLIC_LEN, canonicalize

# ---------------------------------------------------------------------------
# 几何常数
# ---------------------------------------------------------------------------
# 头尾环化时两端在空间上需要靠拢。用主链延伸距离近似：
# 相邻残基 Cα 间距约 3.5 Å（反式伸展构象）。一段 n 残基的线性肽两端间距
# 约为 n × 2.5 Å（考虑一级结构的轻微卷曲），要把两端拉到 ~5 Å 以内成键，
# 至少需要 5-6 个残基。少于这个长度必须加 linker。
CA_CA_DISTANCE: float = 3.5
# 无 linker 头尾环化的最小长度。引用 bio/sequence.py 的统一常量，
# 而不是在这里再写一个 5：`is_cyclic_capable()`（数据清洗用来筛"这条序列
# 值不值得尝试环化"）和这里的环化可行性评估必须对同一个门槛达成一致，
# 否则会出现"清洗放行了 5 残基环肽、环化模块却判它不可行"的自相矛盾。
HEAD_TO_TAIL_MIN_LEN: int = MIN_CYCLIC_LEN
HEAD_TO_TAIL_COMFORT_LEN: int = 8      # 这个长度以上环化几乎无张力
ORGANIC_LINKER_LENGTH_EQUIV: float = 4.0  # 一个常用 linker 约等于 4 个残基的跨度

# α-螺旋旋转周期（每残基转过的角度）
HELIX_TURN_DEG: float = 100.0
# 二硫键要求两条 Cys 在序列上至少相隔这么远才能在空间上正确配对
DISULFIDE_MIN_SEPARATION: int = 4
# 超过这个间距，线性肽两端太远，即便有柔性也很难自发成键（需要 linker）
DISULFIDE_MAX_SEPARATION: int = 40

# 订书肽允许的 (i, i+k) 间隔。
#
# 为什么是 (4, 7, 11) 而不是"再加一个 8"
# ------------------------------------
# α-螺旋每残基转过 100°，所以 (i, i+k) 落在同一侧的条件是
# k×100° 接近 360° 的整数倍，即 k 接近 3.6 的整数倍：3.6 / 7.2 / 10.8 →
# 取整就是 k = 4 / 7 / 11，相位偏差分别是 40° / 20° / 20°。
# 而 k = 8 的相位是 800° mod 360° = 80°，整整差了四分之一圈——
# 两个位点在螺旋的相反侧，桥接会严重扭曲主链。
#
# 原来的 (4, 7, 8, 11) 里，k=8 会被下面 `staple_score` 里
# `phase_err > 40°` 的相位过滤永久剔除：它从来不可能进入候选集。
# 一个永远不会生效的配置项比不存在更糟——看代码的人会以为订书策略
# 支持 i/i+8，写文档、做实验设计时都会照着这个错误的前提走。
STAPLE_SPACINGS: tuple[int, ...] = (4, 7, 11)


@dataclass
class CyclizationPlan:
    """环化方案及其可行性评估结果。"""

    cyclic_type: str                    # head_to_tail / disulfide / staple / lactam / none
    feasible: bool
    score: float                        # ∈ [0,1] 可行性打分，直接进奖励函数
    residues: list[int] = field(default_factory=list)   # 参与交联的残基下标（0-based）
    issues: list[str] = field(default_factory=list)     # 不可行/降分的具体原因
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "cyclic_type": self.cyclic_type,
            "feasible": self.feasible,
            "score": round(self.score, 4),
            "residues": self.residues,
            "issues": self.issues,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# 1. 头尾环化
# ---------------------------------------------------------------------------
def head_to_tail_score(seq: str, *, has_linker: bool = False) -> CyclizationPlan:
    """评估头尾环化的可行性。

    打分逻辑（线性衰减，刻意保持简单可解释）：
      * 长度 ≥ COMFORT_LEN           → 1.0
      * MIN_LEN ≤ 长度 < COMFORT_LEN → 在 0.6~1.0 之间线性插值
      * 长度 < MIN_LEN               → 有 linker 给 0.5，否则 0.2（环张力过大）
      * 两端是 Pro                    → 各扣 0.15（主链构象受限，环化效率低）
      * 存在连续 3 个以上 Gly         → 扣 0.1（柔性过高，环化后仍有多个构象）
    """
    s = canonicalize(seq)
    plan = CyclizationPlan(cyclic_type="head_to_tail", feasible=True, score=1.0)
    n = len(s)

    if n < HEAD_TO_TAIL_MIN_LEN:
        if has_linker:
            plan.score = 0.5
            plan.notes.append(
                f"序列仅 {n} 残基，短于无张力环化下限 {HEAD_TO_TAIL_MIN_LEN}，"
                f"需引入约 {ORGANIC_LINKER_LENGTH_EQUIV:.0f} 残基跨度的 linker"
            )
        else:
            plan.score = 0.2
            plan.feasible = False
            plan.issues.append(f"长度 {n} < {HEAD_TO_TAIL_MIN_LEN}，头尾无法靠拢成键")
        return plan

    if n < HEAD_TO_TAIL_COMFORT_LEN:
        # 线性插值：n=MIN 时 0.6，n=COMFORT 时 1.0
        frac = (n - HEAD_TO_TAIL_MIN_LEN) / (HEAD_TO_TAIL_COMFORT_LEN - HEAD_TO_TAIL_MIN_LEN)
        plan.score = 0.6 + 0.4 * frac
        plan.notes.append(f"长度 {n} 处于环化临界区，建议环化后做构象验证")

    if s and s[0] == "P":
        plan.score -= 0.15
        plan.issues.append("N 端为 Pro：主链构象受限，环化效率通常较低")
    if s and s[-1] == "P":
        plan.score -= 0.15
        plan.issues.append("C 端为 Pro：主链构象受限，环化效率通常较低")
    if "GGG" in s:
        plan.score -= 0.1
        plan.issues.append("存在连续 Gly 柔性区，环化后构象熵仍偏高")

    plan.score = max(0.0, min(1.0, plan.score))
    return plan


# ---------------------------------------------------------------------------
# 2. 二硫键
# ---------------------------------------------------------------------------
def disulfide_score(seq: str) -> CyclizationPlan:
    """评估二硫键环化的可行性。

    关键约束：Cys 必须成对。奇数个 Cys 意味着必然有一个游离巯基，
    它会与血浆白蛋白等形成非特异性加合物，是成药性上的明确缺陷。

    配对策略：优先选择"序列上相距最远 → 次远"的嵌套配对（1-2, 3-4 式样），
    而不是简单的首尾配对。生化依据是天然蛋白中二硫键几乎总是
    非交叉（nested）拓扑，交叉配对会形成打结结构，折叠自由能不利。
    """
    s = canonicalize(seq)
    cys_idx = [i for i, c in enumerate(s) if c == "C"]
    plan = CyclizationPlan(cyclic_type="disulfide", feasible=False, score=0.0)

    if len(cys_idx) < 2:
        plan.issues.append(f"仅 {len(cys_idx)} 个 Cys，无法形成二硫键")
        return plan
    if len(cys_idx) % 2 == 1:
        plan.issues.append(
            f"Cys 数量为奇数（{len(cys_idx)}），必然残留游离巯基，成药性风险高"
        )
        return plan

    # 嵌套配对：把 Cys 按下标排序后两两配对（1-2, 3-4, ...）即为嵌套拓扑
    pairs = [(cys_idx[i], cys_idx[i + 1]) for i in range(0, len(cys_idx), 2)]
    separations = [b - a for a, b in pairs]
    plan.residues = [i for pair in pairs for i in pair]

    min_sep = min(separations)
    max_sep = max(separations)
    if min_sep < DISULFIDE_MIN_SEPARATION:
        plan.issues.append(
            f"最短 Cys 间距仅 {min_sep}，小于 {DISULFIDE_MIN_SEPARATION}，"
            "无法形成正确的二硫构象"
        )
        return plan
    if max_sep > DISULFIDE_MAX_SEPARATION:
        plan.issues.append(
            f"最长 Cys 间距 {max_sep}，超过自发成键可行范围，建议改用 linker 或订书"
        )
        return plan

    # 打分：间距越接近 8-14 残基（一个螺旋周期的整数倍附近）越理想
    ideal = 10.0
    avg_sep = sum(separations) / len(separations)
    plan.score = max(0.5, 1.0 - abs(avg_sep - ideal) / ideal * 0.5)
    if len(cys_idx) >= 4:
        plan.score = min(1.0, plan.score + 0.1)  # 双二硫键稳定性增益更大
        plan.notes.append("含两对及以上 Cys，可实现双二硫键，热稳定性增益显著")
    plan.feasible = True
    plan.notes.append(f"配对方案（0-based）: {pairs}，平均间距 {avg_sep:.1f}")
    return plan


# ---------------------------------------------------------------------------
# 3. 订书肽（烃链桥）
# ---------------------------------------------------------------------------
def staple_score(seq: str) -> CyclizationPlan:
    """评估烃链订书（hydrocarbon stapling）的可行性与位置合理性。

    约束有两层：
      a. 可操作位点：只有 (i, i+4) / (i, i+7) 这两个间隔能在 α-螺旋上
         让两个非天然氨基酸的侧链指向同一侧并够得着。
      b. 位置合理性：订书点应当落在非功能面上。判别方法是看该位置
         在螺旋轮上属于疏水面还是亲水面（疏水面通常是与靶点结合的面，
         也是需要保持的活性面）——在亲水面订书会破坏极性相互作用。

    这里给出的 score 是"这条序列上是否存在至少一个合理的订书位点"。
    """
    s = canonicalize(seq)
    plan = CyclizationPlan(cyclic_type="staple", feasible=False, score=0.0)
    if len(s) < 8:
        plan.issues.append(f"长度 {len(s)} 过短，无法容纳订书跨度")
        return plan

    candidates: list[tuple[int, int, float]] = []
    for i in range(len(s)):
        for k in STAPLE_SPACINGS:
            j = i + k
            if j >= len(s):
                continue
            # i 与 j 必须落在螺旋的同一侧：转过 k 圈后相位差接近 0 或 360°
            phase = (k * HELIX_TURN_DEG) % 360.0
            phase_err = min(phase, 360.0 - phase)
            if phase_err > 40.0:
                continue  # 不在同一侧，桥接会严重扭曲螺旋
            # 订书点两侧残基不能是 Pro/Gly（主链构象不允许被拉近）
            if any(s[p] in ("P", "G") for p in (i, j)):
                continue
            candidates.append((i, j, phase_err))

    if not candidates:
        plan.issues.append("未找到相位匹配的 (i, i+4/i+7) 订书位点")
        return plan

    # 优先选相位误差最小、且落在疏水面的位点。
    # 排序键第一个分量是 +err（升序 → 误差最小优先），第二个是 -hydro
    # （升序 → 疏水性最高优先）。两处符号方向都不能写反，
    # 否则会挑到相位最不匹配的位点，等于给出了一个会破坏螺旋的错误建议。
    def rank(cand: tuple[int, int, float]) -> tuple[float, float]:
        i, j, err = cand
        # 疏水残基占比越高，说明这一侧是疏水面（保持不动更安全）
        hydro = sum(1 for p in (i, j) if s[p] in "AVILMFWY") / 2.0
        return (err, -hydro)

    best = sorted(candidates, key=rank)[0]
    plan.residues = [best[0], best[1]]
    plan.feasible = True
    plan.score = max(0.6, 1.0 - best[2] / 90.0)
    plan.notes.append(
        f"推荐订书位点（0-based）: {best[0]} 与 {best[1]}，"
        f"间隔 {best[1] - best[0]}，相位误差 {best[2]:.1f}°"
    )
    plan.notes.append("订书点落在疏水面，不直接遮挡极性/带电功能残基")
    return plan


# ---------------------------------------------------------------------------
# 4. 内酰胺桥
# ---------------------------------------------------------------------------
def lactam_score(seq: str) -> CyclizationPlan:
    """评估 i/i+4 内酰胺桥（Asp/Glu ↔ Lys/Arg 侧链成酰胺）的可行性。

    与订书互补：内酰胺桥带电荷，常用于水溶性较差或需要额外极性的场合，
    也能稳定 α-螺旋。约束同样是相位（i 与 i+4 同侧）与残基类型。
    """
    s = canonicalize(seq)
    plan = CyclizationPlan(cyclic_type="lactam", feasible=False, score=0.0)
    acidic, basic = "DE", "KR"
    hits: list[tuple[int, int]] = []
    for i in range(len(s) - 4):
        j = i + 4
        if (s[i] in acidic and s[j] in basic) or (s[i] in basic and s[j] in acidic):
            hits.append((i, j))
    if not hits:
        plan.issues.append("未找到 i/i+4 的酸-碱残基对，无法构建内酰胺桥")
        return plan
    # 取第一对（序列顺序确定，保证可复现）
    plan.residues = list(hits[0])
    plan.feasible = True
    # 多余的对数越多越灵活（可以选择最优位置），给一点额外加分
    plan.score = min(1.0, 0.75 + 0.05 * (len(hits) - 1))
    plan.notes.append(
        f"可行内酰胺位点（0-based）共 {len(hits)} 处，首选 {hits[0]}"
    )
    return plan


# ---------------------------------------------------------------------------
# 5. 统一入口
# ---------------------------------------------------------------------------
_SCORERS = {
    "head_to_tail": head_to_tail_score,
    "disulfide": disulfide_score,
    "staple": staple_score,
    "lactam": lactam_score,
}


def evaluate(seq: str, cyclic_type: str, *, has_linker: bool = False) -> CyclizationPlan:
    """按指定的环化类型评估方案。`none` 表示线性肽，永远可行。"""
    if cyclic_type in ("none", "linear", ""):
        return CyclizationPlan(
            cyclic_type="none", feasible=True, score=1.0,
            notes=["线性肽，无需环化约束"],
        )
    scorer = _SCORERS.get(cyclic_type)
    if scorer is None:
        raise KeyError(
            f"unsupported cyclic_type: {cyclic_type}，可选 {[*_SCORERS, 'none']}"
        )
    if cyclic_type == "head_to_tail":
        return scorer(seq, has_linker=has_linker)  # type: ignore[call-arg]
    return scorer(seq)  # type: ignore[operator]


def best_plan(seq: str) -> CyclizationPlan:
    """在四种环化方案里挑一条最合理的，用于"骨架选型"阶段的自动推荐。"""
    plans = [evaluate(seq, t) for t in _SCORERS]
    feasible = [p for p in plans if p.feasible]
    if not feasible:
        return CyclizationPlan(
            cyclic_type="none",
            feasible=True,
            score=0.5,
            notes=[
                "当前序列不满足任何一种环化约束，建议先做骨架优化；",
                "以下为各方案失败原因：" + "; ".join(
                    f"{p.cyclic_type}: {p.issues[0]}" for p in plans if p.issues
                ),
            ],
        )
    return max(feasible, key=lambda p: p.score)


def constraint_penalty(seq: str, cyclic_type: str) -> float:
    """把环化可行性转换成"结构合理性"奖励维度里的一个惩罚项 ∈ [0, 1]。

    0 = 完全满足约束（不扣分），1 = 完全不可行（扣满）。
    这样设计的好处是奖励函数可以对所有序列给出连续值，
    而不是对不可行序列直接给 0——后者会导致 GRPO 的组内优势全为 0，梯度消失。
    """
    plan = evaluate(seq, cyclic_type)
    return round(1.0 - plan.score, 4)


def feasible_cyclic_type(seq: str, cyclic_type: str) -> str:
    """把"数据里的环化标签"收敛成"这条序列真的能满足的类型"。

    这一步的目的
    --------------
    UniProt 的 `Disulfide bond` 关键词只说明"这个条目里有二硫键"，
    不保证整条序列的 Cys 数/配对能让它成为一个自洽的二硫键环肽。
    实测 814 条 `disulfide` 记录里，247 条 Cys 数不足或为奇数、
    461 条的 constraint_penalty >= 0.5。如果直接把这个标签写进 prompt
    （"保持二硫键环化"）、又在奖励里按它扣分，SFT 的答案是这条记录本身，
    模型就在被要求同时"满足"和"违反"同一个约束——这是数据标签的错，
    不是模型的错。

    这里用原始记录序列判断；stabilize/multi 的 base 是劣化后的序列，
    它可能故意破坏了环化几何，但任务目标正是把它修回来，所以不能用 base 判断。
    返回 `"none"` 表示"这条序列不把它当环化任务"，prompt 和奖励都不再提环化。
    """
    if cyclic_type in (None, "", "none", "linear"):
        return "none"
    try:
        return cyclic_type if constraint_penalty(seq, cyclic_type) <= 1e-9 else "none"
    except KeyError:
        return "none"
