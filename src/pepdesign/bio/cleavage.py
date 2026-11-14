"""蛋白酶酶切位点识别与酶解稳定性评估。

规则层承担日常批量打分：明确、零成本、完全可解释。模型层
（predictors/protease_sites.py）作为可选增强，遵循同一接口，
上层奖励函数无需改动。

N 端乙酰化、C 端酰胺化与头尾环化会消除对应的外切酶位点。
`modifications` 参数接收这些修饰并扣除其屏蔽的位点，避免模型学到
"加修饰即可加分"的错误因果。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pepdesign.bio.aa import PROTEASE_RULES
from pepdesign.bio.sequence import canonicalize

# 位点密度（加权位点数 / 序列长度）的归一化尺度：达到该密度即饱和到 0 分。
#
# 为什么是密度而不是绝对位点数
# ----------------------------
# 这里原先是一个绝对计数常量（`SITE_SATURATION = 6.0`，"位点数达到 6 个
# 即认为酶解风险很高"）。6 这个数在 5-12 残基的短肽上是对的，但它不随
# 长度缩放，于是在长肽上直接失效。实测（同一套打分，只换归一化口径）：
#
#     序列           长度  加权位点   旧分(绝对)  新分(密度)
#     indolicidin     13      5.0       0.17        0.49
#     KWKLFKKIEK      10      7.2       0.00        0.04
#     magainin-2      23     12.2       0.00        0.29
#     melittin        26     13.0       0.00        0.33
#     LL-37           37     21.6       0.00        0.22
#     cecropin A      37     16.2       0.00        0.55
#
# 旧口径下所有 ≥13 残基的真实抗菌肽都是 0 分；随机序列里 20-mer 有
# 77% 是 0、30-mer 98%。后果不是"分低"而是没有梯度：13 个位点和 21 个
# 位点同分，而 `pro_sweep` 想教的"P1' 换 Pro 阻断切割"通常只降 1-2 个位点，
# 在这条奖励上拿不到任何收益。数据侧也印证了：pipeline 日志里
# `saturated:protease: 164` 次候选改造被拒，校准集只做到目标的三分之一。
#
# 换成密度之后，随机序列的位点密度在各长度上基本恒定（实测约 0.37），
# 长短肽不再被系统性地判成"稳定性差异"；真实抗菌肽则被拉开档次。
#
# 取值 0.75 的依据：随机序列密度 ≈0.37（得分 ≈0.51），真实抗菌肽落在
# 0.38-0.72（得分 0.04-0.49），最有价值的那一段完整落在曲线中段，
# 且两端都不贴边饱和。
#
# 已知取舍（写清楚，免得下一个人以为是疏漏）：密度口径不再区分
# "20-mer 上 10 个位点"和"40-mer 上 20 个位点"。绝对意义上后者可切断的
# 键更多、血清半衰期更短——但那由 `ProteaseHalfLifePredictor` 用绝对
# 口径单独估算并写进 detail，两者分工在 rewards/protease.py 里有说明。
# 奖励这一层的职责是给出可学习的梯度，不是复现半衰期的绝对值。
SITE_DENSITY_SATURATION: float = 0.75

# 各蛋白酶在"血清降解"这一目标下的相对重要性权重。
# 为什么要加权而不是简单求和：血清中真正主导短肽清除的是胰蛋白酶样活性
# （凝血因子、激肽释放酶等都属丝氨酸蛋白酶，偏好碱性残基）与 DPP4，
# 而胃蛋白酶在血液里几乎没有活性。等权求和会让一个胃蛋白酶位点
# 和胰蛋白酶位点等价，把优化方向带偏。
PROTEASE_WEIGHTS: dict[str, float] = {
    "trypsin": 1.0,
    "chymotrypsin": 0.8,
    "elastase": 0.5,
    "pepsin": 0.2,
    "carboxypeptidase": 0.6,
    "dpp4": 0.9,
}

# 支持的末端修饰 -> 它们各自封闭掉哪一类酶切位点
TERMINAL_MODIFICATIONS: dict[str, str] = {
    "n_acetyl": "封闭 N 端，屏蔽氨肽酶/DPP4 类从 N 端起切的作用",
    "c_amide": "封闭 C 端，屏蔽羧肽酶从 C 端起切的作用",
    "head_to_tail_cyclic": "头尾环化，N/C 两端都不再游离，外切酶位点全部消失",
    "pegylated": "PEG 屏蔽，降低蛋白酶可及性（按比例折减）",
}


@dataclass
class CleavageSite:
    """一个酶切位点。`position` 是 P1 残基的下标（切割发生在其 C 端之后）。"""

    protease: str
    position: int
    p1: str
    p2: str  # P1' 位，用于解释为什么某些位置被判为"切不动"
    weight: float

    def as_dict(self) -> dict:
        return {
            "protease": self.protease,
            "position": self.position,
            "p1": self.p1,
            "p2": self.p2,
            "weight": self.weight,
        }


@dataclass
class CleavageReport:
    """一条序列的酶解稳定性完整报告。"""

    sequence: str
    sites: list[CleavageSite] = field(default_factory=list)
    # 按蛋白酶汇总的位点数量，用于前端展示"主要降解途径"
    sites_by_protease: dict[str, int] = field(default_factory=dict)
    # 加权位点数：考虑了各酶在血清中的实际重要性
    weighted_site_count: float = 0.0

    @property
    def raw_site_count(self) -> int:
        return len(self.sites)

    @property
    def dominant_protease(self) -> str | None:
        """位点最多的那个酶——报告里会直接写"主要风险来自胰蛋白酶样切割"。"""
        if not self.sites_by_protease:
            return None
        return max(self.sites_by_protease.items(), key=lambda kv: kv[1])[0]

    def as_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "raw_site_count": self.raw_site_count,
            "weighted_site_count": round(self.weighted_site_count, 3),
            "sites_by_protease": self.sites_by_protease,
            "dominant_protease": self.dominant_protease,
            "sites": [s.as_dict() for s in self.sites],
        }


def find_sites(
    seq: str,
    *,
    proteases: tuple[str, ...] | None = None,
    modifications: tuple[str, ...] = (),
) -> list[CleavageSite]:
    """找出所有酶切位点。

    参数
    ----
    seq : 序列（会被 canonicalize）
    proteases : 限定考察哪些酶，None = 全部
    modifications : 已应用的末端修饰，见 TERMINAL_MODIFICATIONS

    实现要点
    --------
    * P1' 抑制规则：真核蛋白酶普遍"切不动 Xaa-Pro 键"（脯氨酸的环状主链
      无法进入蛋白酶 S1' 口袋完成催化构象）。这条规则对 trypsin / chymotrypsin /
      elastase 都成立，所以刻意把 Pro 放在酶切位点后一位，是设计上最常用的
      "单点抗酶解"策略——代码里必须体现这条规则，否则模型学不到这个策略。
    * 外切酶从末端起算：羧肽酶作用于 C 端、DPP4 作用于 N 端第 2 位，
      位置计算方式和内切酶不同，需要分开处理。
    """
    s = canonicalize(seq)
    if not s:
        return []

    # 末端修饰生效时的屏蔽集合
    n_blocked = any(m in modifications for m in ("n_acetyl", "head_to_tail_cyclic"))
    c_blocked = any(m in modifications for m in ("c_amide", "head_to_tail_cyclic"))

    names = proteases if proteases is not None else tuple(PROTEASE_RULES.keys())
    sites: list[CleavageSite] = []

    for name in names:
        rule = PROTEASE_RULES.get(name)
        if rule is None:
            raise KeyError(f"unknown protease rule: {name}")
        weight = PROTEASE_WEIGHTS.get(name, 0.5)

        if name == "carboxypeptidase":
            # 外切酶：只看 C 端残基。C 端被封闭则完全无位点。
            if not c_blocked and s[-1] in rule["sites"]:
                sites.append(
                    CleavageSite(name, len(s) - 1, s[-1], "", weight)
                )
            continue

        if name == "dpp4":
            # 二肽基肽酶 IV：切 N 端第 2 位之后的键，P1 = 第 2 位残基。
            # 需要至少 2 个残基，且 N 端未被封闭。
            if not n_blocked and len(s) >= 2 and s[1] in rule["sites"]:
                sites.append(CleavageSite(name, 1, s[1], s[2] if len(s) > 2 else "", weight))
            continue

        # 内切酶：扫描所有 i，i 处残基是 P1，i+1 是 P1'
        for i, c in enumerate(s[:-1]):  # 最后一个残基后面没有键，不可能是内切位点
            if c not in rule["sites"]:
                continue
            nxt = s[i + 1]
            if nxt in rule["blocked_by"]:
                continue
            sites.append(CleavageSite(name, i, c, nxt, weight))

    return sites


def analyze(
    seq: str,
    *,
    modifications: tuple[str, ...] = (),
    proteases: tuple[str, ...] | None = None,
) -> CleavageReport:
    """完整分析：位点列表 + 汇总统计。"""
    s = canonicalize(seq)
    sites = find_sites(s, proteases=proteases, modifications=modifications)

    by_protease: dict[str, int] = {}
    for site in sites:
        by_protease[site.protease] = by_protease.get(site.protease, 0) + 1

    weighted = sum(site.weight for site in sites)
    # PEG 化是"部分屏蔽"而不是"完全消除"，按 30% 可及性折减
    if "pegylated" in modifications:
        weighted *= 0.3

    return CleavageReport(
        sequence=s,
        sites=sites,
        sites_by_protease=by_protease,
        weighted_site_count=round(weighted, 4),
    )


def site_penalty(n_weighted_sites: float, length: int) -> float:
    """加权位点数 + 序列长度 → [0, 1] 的惩罚分（1 = 最不稳定）。

    单独抽成函数是因为"多少个位点算多"和"怎么把它变成分数"是两件事，
    而前者已经被证明是最容易搞错的一环（见 `SITE_DENSITY_SATURATION`
    上面的实测表）。把口径集中在这一个函数里，改的时候只有一个地方要看。
    """
    if length <= 0:
        # 空序列没有密度可言。它本来就过不了 validate，给 0 惩罚即可，
        # 不要在这里造一个 inf/nan 出来——奖励链路上任何非有限值都会
        # 一路传到 GRPO 的优势计算里。
        return 0.0
    return min(1.0, (n_weighted_sites / length) / SITE_DENSITY_SATURATION)


def stability_score(seq: str, *, modifications: tuple[str, ...] = ()) -> float:
    """酶解稳定性得分 ∈ [0, 1]，越高越稳定。

    映射方式：score = 1 - penalty，其中
        penalty = min(1, (weighted_site_count / len(seq)) / SITE_DENSITY_SATURATION)

    两个刻意的设计
    --------------
    1. 用加权位点数，不用原始计数。血清中主导短肽清除的是胰蛋白酶样
       活性和 DPP4，胃蛋白酶在血液里几乎没活性；等权计数会把优化方向带偏。
    2. 按长度归一化。绝对计数在长肽上会整体饱和到 0，这一维就没有梯度了
       （来龙去脉和实测数据见 `SITE_DENSITY_SATURATION`）。

    饱和仍然保留：密度很高时封顶在 0，避免模型把生成预算持续花在
    "从已经很好变得更好"的方向上。代价是该区间内组内无区分度——
    这是已知的、有意的取舍，而不是疏漏。

    注意：这个分数是稠密奖励，每条序列都能算出唯一值，
    非常适合 GRPO 的组内比较（group 内序列越分散，优势信号越强）。
    """
    report = analyze(seq, modifications=modifications)
    penalty = site_penalty(report.weighted_site_count, len(report.sequence))
    return round(1.0 - penalty, 4)


def suggest_stabilization(
    seq: str, *, max_suggestions: int = 5
) -> list[dict[str, str | int]]:
    """基于位点分布给出可执行的改造建议。

    这是 SOP "稳定性优化" 阶段的规则引擎部分，输出会作为 few-shot 示例
    塞回给模型的提示词里——"模型 + 规则"的组合比纯模型更可靠：
    规则保证每一步改造都有生化依据，模型负责在规则约束下搜索组合方案。
    """
    report = analyze(seq)
    suggestions: list[dict[str, str | int]] = []

    # 建议 1：在 P1' 位引入 Pro，直接消除该位点
    for site in report.sites:
        if site.protease == "dpp4" or site.protease == "carboxypeptidase":
            continue  # 外切酶位点不能靠 P1' 突变解决
        if site.p2 == "P":
            continue
        if site.position + 1 >= len(report.sequence):
            continue
        suggestions.append(
            {
                "position": site.position + 1,  # 1-based，方便研发人员对照
                "from": site.p2,
                "to": "P",
                "reason": f"{site.protease} 在 {site.position + 1} 位后识别 {site.p1}，"
                f"将 P1' 位改为 Pro 可阻断该切割",
            }
        )
        if len(suggestions) >= max_suggestions:
            return suggestions

    # 建议 2：N 端封闭（如果存在 DPP4 风险）
    if "dpp4" in report.sites_by_protease:
        suggestions.append(
            {
                "position": 1,
                "from": report.sequence[0],
                "to": "Ac-",
                "reason": "N 端乙酰化可消除 DPP4 介导的二肽切除，"
                "这是 GLP-1 类多肽最经典的半衰期延长手段",
            }
        )

    # 建议 3：C 端酰胺化
    if "carboxypeptidase" in report.sites_by_protease:
        suggestions.append(
            {
                "position": len(report.sequence),
                "from": report.sequence[-1],
                "to": "-NH2",
                "reason": "C 端酰胺化屏蔽羧肽酶切口，同时移除末端负电荷提升活性",
            }
        )
    return suggestions
