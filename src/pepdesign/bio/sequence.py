"""多肽序列的解析与物化性质计算。

这个模块是全项目口径的唯一来源：数据清洗、奖励函数、离线评估、SOP 报告
全都调用这里的函数，不允许各模块自己再算一遍。原因在 aa.py 里已经说过——
强化学习里如果奖励和评估用了两套口径，模型会朝着一个和最终指标不一致的方向优化。

实现约定
--------
* 所有函数纯函数、无副作用、CPU 可跑、不依赖 torch。这保证了奖励函数可以在
  vLLM 推理进程之外独立、批量、并行地计算（GRPO 的数据流正是如此）。
* 对含模糊残基（X/B/Z）的序列，性质计算会跳过对应位置而不是报错，
  同时在返回结构里给出 `coverage`，让调用方知道这个分数有多少信息量。
  这正是奖励函数"不能因为一条脏序列就把整轮训练打断"所需要的鲁棒性。
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from pepdesign.bio.aa import (
    ALLOWED_AA,
    AMBIGUOUS_AA,
    AROMATIC_AA,
    C_TERM_PKA,
    COIL_PROPENSITY,
    EISENBERG,
    EISENBERG_MEAN,
    HELIX_PROPENSITY,
    HYDROPHOBIC_AA,
    KYTE_DOOLITTLE,
    N_TERM_PKA,
    NEGATIVE_AA,
    RESIDUE_MASS,
    SHEET_PROPENSITY,
    SIDE_CHAIN_CHARGE_PH7,
    SIDE_CHAIN_PKA,
    STANDARD_AA,
    WATER_MASS,
)

# 合法序列的长度区间。5-30 为主要训练分布；放宽到 2-200 以容纳长肽与
# 复杂环肽样本，质量打分时对超出主分布的部分降权。
MIN_PEPTIDE_LEN: int = 2
MAX_PEPTIDE_LEN: int = 200
CORE_PEPTIDE_LEN: tuple[int, int] = (5, 30)

# 净电荷的三道门槛，层层递进而不是三个随手写的数：
#   * PROMPT  —— 写进系统提示词、对模型提的硬要求；
#   * SOFT    —— 软惩罚的起点，故意低于硬要求，给模型留提前量：
#                等到真的越过 8 才扣分，模型已经在这条边界上反复试探了；
#   * VETO    —— 结构合理性的一票否决线，到这个量级已经不是"不太理想"
#                而是"这条序列在生理条件下不可能有合理行为"。
# 三处以前分别硬编码为 8 / 6 / 12，散在 prompts、rewards、predictors 三个
# 模块里，改一个忘一个就会出现"提示词说 8、惩罚按 6 算、否决按 12 判"
# 这种各说各话的状态。统一在这里定义。
CHARGE_PROMPT_LIMIT: float = 8.0
CHARGE_SOFT_LIMIT: float = 6.0
CHARGE_VETO_LIMIT: float = 12.0

# 环化所需的最小长度。以前 bio/sequence.py 用 4、bio/cyclization.py 用 5，
# 于是 is_cyclic_capable() 放行的 4 残基序列到了 take_cyclic() 会被拒，
# 数据清洗和环化策略对"什么算可环化"的判断不一致。
# 取 5：头尾成酰胺键在 4 残基上几何张力过大，且 4 残基不足以形成
# 稳定的转角结构——这正是环肽实战里的经验下限。
MIN_CYCLIC_LEN: int = 5


# ==========================================================================
# 1. 解析与校验
# ==========================================================================
def canonicalize(raw: str) -> str:
    """把任意输入的序列规范化成大写、无空白、单字母的字符串。

    数据来自 UniProt / PDB / DRAMP 等不同源，格式五花八门：
    小写、带换行、带 '-'(缺失残基)、带 '.'(链断裂)、带数字。统一在这里处理，
    避免下游每个函数都做一遍防御性 strip。
    """
    if raw is None:
        return ""
    s = str(raw).strip().upper()
    # 去掉 FASTA 头（以 '>' 开头的整行）以及所有空白
    if s.startswith(">"):
        s = s.split("\n", 1)[-1]
    for ch in (" ", "\t", "\n", "\r", "-", ".", "*", "\x00"):
        s = s.replace(ch, "")
    # 数字也一并去掉：序列里的数字只可能来自标注
    # （PDB 的残基编号 "A12"、修饰写法 "NH2"、"PEG2000"、长度括号 "KWK(12)"），
    # 不可能是残基代码。留着它们会让 validate() 把整条序列判为非法，
    # 而实际上这些序列是完全可用的——这是"清洗要理解领域写法"的一个具体例子。
    s = "".join(ch for ch in s if not ch.isdigit())
    return s


# ---------------------------------------------------------------------------
# 末端修饰标注的识别
# ---------------------------------------------------------------------------
# 这两条正则故意大小写敏感，这不是疏忽：
#
#   * `Ac-`（首字母大写、次字母小写）是 N 端乙酰化的标准写法；
#   * 全大写 `AC-` 表示的是 Ala-Cys 两个残基。
#
# 加上 re.IGNORECASE 之后两者就无法区分，`AC-KWKLFKK`（Ala-Cys-Lys-Trp-…）
# 会被剥掉开头两个残基、并被打上"已乙酰化"的标记——而结果序列
# `KWKLFKK` 依然是一条完全合法的肽，validate() 检查不出来，
# 错误会一路混进训练语料和奖励计算。这正是 aa.py 里说的
# "静默的错误比明确的报错危险得多"。
N_ACETYL_RE = re.compile(r"^\s*(?:Acetyl|Ac|乙酰基?)\s*[-－]\s*")
# C 端酰胺化。-NH2 / -NH₂ / -CONH2 / 酰胺化，同样要求存在连字符分隔，
# 否则 `GNH2` 这类（Gly-Asn 后面恰好跟了个数字）会被误剥。
#
# 下标数字 ₂（U+2082）与 ₃（U+2083）都接受：文献和数据库中最常见的
# C 端酰胺化写法是 `-NH₂`，也存在误写成 `-NH₃` 的情况。两个模式都要求
# 连字符分隔，误匹配风险很低；只收 ₃ 会漏掉标准写法。
C_AMIDE_RE = re.compile(r"\s*[-－]\s*(?:NH2|NH₂|NH₃|CONH2|酰胺化?)\s*$")


def strip_terminal_modifications(raw: str) -> tuple[str, list[str]]:
    """剥离 N 端乙酰化 / C 端酰胺化标注，返回 (序列本体, 修饰列表)。

    放在 bio/ 层的原因
    ------------------------------------
    有两个地方需要做这件事，它们面对的是同一个问题：
      * `prompts/parser.py` —— 处理模型生成的 `Ac-KWKLFKK-NH2`；
      * `data/schema.py`    —— 处理数据库里的 `ACDEFG-NH2` 写法。
    各写一份的后果在项目里真实发生过：一处认 `AC-`、另一处按不同规则切，
    同一条序列在"训练语料路径"和"推理解析路径"上被解析成不同的结果。
    这属于本模块开头说的"全项目口径的唯一来源"。

    必须在 `canonicalize` 之前调用：canonicalize 会删掉 '-' 和数字，
    那时 `Ac-KWK-NH2` 已经变成 `ACKWKNH`，再也分不清哪些字母是标注。
    """
    mods: list[str] = []
    s = str(raw or "")

    m = N_ACETYL_RE.match(s)
    if m:
        mods.append("n_acetyl")
        s = s[m.end():]

    m = C_AMIDE_RE.search(s)
    if m:
        mods.append("c_amide")
        s = s[: m.start()]

    return s, mods


def has_noncanonical_residues(raw: str) -> bool:
    """判断原始写法里是否含手性/非标准残基信息（D-型氨基酸、U、O）。

    多肽领域用小写表示 D-型残基（`RGDfV` 里的 `f` 就是 D-Phe，西仑吉肽骨架），
    用 U/O 表示硒代半胱氨酸/吡咯赖氨酸。这些信息在 `canonicalize` 里会被
    静默丢掉——大写化 + 过滤之后 `RGDfV` 变成了 `RGDFV`，手性和活性都变了，
    而下游没有任何环节能发现这件事。

    所以检测必须在 canonicalize 之前做，并把结果作为标记保留下来，
    由数据清洗层决定是丢弃、还是单独走一条"非标准残基"的分流。

    判据用"大小写混排"而不是"含小写字母"，这一点很关键：D-型残基的
    标准写法是大写序列里夹一个小写字母（`RGDfV`），而全小写的 `rgdfv`
    只是随手写的正经序列、`ac-kwklfkk` 里的 `c` 是修饰标注的一部分。
    把"含小写"直接当成手性标记，会把这两类正常的输入全部误标成非标准残基。

    调用前应当先剥离末端修饰标注，避免 `Ac-` 的 `c` 混进来。
    """
    s = str(raw or "")
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return False
    mixed_case = any(ch.isupper() for ch in letters) and any(
        ch.islower() for ch in letters
    )
    return mixed_case or any(ch in "UO" for ch in s.upper())


def validate(seq: str) -> tuple[bool, str]:
    """校验序列格式是否可用（长度、字符集），返回 (是否合法, 原因)。

    单开一个函数返回原因而不是抛异常，是因为数据清洗阶段需要统计各类
    失败原因的分布；异常只给出类型，无法形成统计。

    这个函数不判断"这是不是一条肽"，只判断"它能不能当序列用"。
    ------------------------------------------------------------
    `ALLOWED_AA` 是 25 个字母（20 标准 + `XBZUO`），而英文单词的字母几乎
    全落在这个集合里，所以 `CONTENT`、`IOCTL`、`DISCRIMINATORBURNING`
    这类词都能通过。这不是漏写了检查，是刻意不加的：
    任何"看起来像英文单词就拒绝"的启发式都会误杀真实序列
    （`ATTENTION`、`TARGET` 都是合法的氨基酸串），而且误杀是静默的——
    样本直接消失，没人知道它存在过。

    正确的防线在入口，不在这个函数里：
      * RL 采样侧走 `prompts/parser.require_strong_parse`——它结合
        输出格式的上下文判断，那里"没有按格式回答"本身就是强信号；
      * 数据入库侧要对来源负责（见 `DataSource` 的说明），
        外部库来的序列不该走到这里被逐条猜。
    新接一个数据入口时，这条边界要重新想一遍，别指望 `validate()` 兜住。
    """
    if not seq:
        return False, "empty"
    if len(seq) < MIN_PEPTIDE_LEN:
        return False, "too_short"
    if len(seq) > MAX_PEPTIDE_LEN:
        return False, "too_long"
    bad = set(seq) - set(ALLOWED_AA)
    if bad:
        return False, f"illegal_char:{''.join(sorted(bad))}"
    # 全是 X 的序列没有信息量，属于典型噪声样本
    if all(c in AMBIGUOUS_AA for c in seq):
        return False, "all_ambiguous"
    return True, "ok"


def standard_only(seq: str) -> str:
    """只保留 20 种标准氨基酸（砍掉 X/B/Z/U/O）。

    用于必须用严格氨基酸表的场景：疏水矩的周期求和、二级结构倾向统计。
    直接丢弃模糊残基会让相位错位，所以这类计算只能在纯净序列上做。
    """
    return "".join(c for c in seq if c in STANDARD_AA)


def coverage(seq: str) -> float:
    """标准残基占比。奖励函数用它给分数打置信度折扣：
    一条 50% 都是 X 的序列，其"预测 Tm"本来就不该被当真。"""
    if not seq:
        return 0.0
    return sum(1 for c in seq if c in STANDARD_AA) / len(seq)


def is_cyclic_capable(seq: str) -> bool:
    """判断序列是否具备头尾环化的基本条件。

    环化需要 N 端游离氨基与 C 端游离羧基成酰胺键，因此两端不能被封闭基团占据；
    带 Pro 结尾的序列环化效率通常很低（主链构象受限）。这里只做"是否值得尝试"的
    粗筛，真实的环化可行性评估属于 SOP 里的骨架选型阶段。
    """
    return len(seq) >= MIN_CYCLIC_LEN and seq[0] != "P" and seq[-1] != "P"


# ==========================================================================
# 2. 组成类性质
# ==========================================================================
def amino_acid_composition(seq: str, *, percent: bool = False) -> dict[str, float]:
    """氨基酸组成。`percent=True` 时返回摩尔百分数（用于 aliphatic_index）。"""
    s = standard_only(seq)
    n = len(s)
    if n == 0:
        return {}
    counts = {aa: s.count(aa) for aa in STANDARD_AA if s.count(aa) > 0}
    if percent:
        return {aa: cnt / n * 100.0 for aa, cnt in counts.items()}
    return {aa: float(cnt) for aa, cnt in counts.items()}


def molecular_weight(seq: str) -> float:
    """多肽分子量（Da）。

    残基质量表已是"脱去一分子水"后的残基形式，所以 N 个残基的理论肽链质量
    需要再补回一分子水（两端各一个 H / OH）。对于头尾环化的序列，
    环化时还会再脱去一分子水，故 cyclic=True 时要减掉。
    """
    return _molecular_weight(seq, cyclic=False)


def _molecular_weight(seq: str, *, cyclic: bool) -> float:
    mass = sum(RESIDUE_MASS.get(c, 0.0) for c in seq)
    mass += WATER_MASS
    if cyclic:
        mass -= WATER_MASS  # 头尾成酰胺键，再脱去一分子水
    return round(mass, 3)


def gravy(seq: str) -> float:
    """GRAVY = Grand Average of hydropathicitY，全序列平均亲疏水性（Kyte-Doolittle）。

    经验区间：> 0 偏疏水（易穿膜、易聚集、溶解性差），< 0 偏亲水。
    对多肽药物而言，GRAVY 通常希望落在 -1.0 ~ 0 之间：
    太负说明极性太强进不了细胞/结合口袋，太正则溶解度差、易聚集。
    """
    s = standard_only(seq)
    if not s:
        return 0.0
    return round(sum(KYTE_DOOLITTLE[c] for c in s) / len(s), 4)


def net_charge(seq: str, ph: float = 7.4) -> float:
    """用 Henderson-Hasselbalch 方程计算给定 pH 下的净电荷。

    比"数一下 D/E/K/R 个数"精确，因为它区分了部分质子化：
      * 酸性侧链 (D/E/C/Y) 带负电的比例 f = 1 / (1 + 10^(pKa - pH))
      * 碱性侧链 (K/R/H) 带正电的比例 f = 1 / (1 + 10^(pH - pKa))
      * N 端 +1、C 端 -1 同样按 pKa 折算
    pH 7.4 下 His 只带 +0.038（pKa 6.0 代入上式），这个细节决定了"人工数残基"
    和"真实电荷"的差异，在做静电亲和力相关判断时不能忽略。
    （注意别和 `net_charge_fast` 用的固定电荷表搞混：那张表按文献习惯取 +0.1，
    两张表服务于不同场景，见 aa.py 里 SIDE_CHAIN_CHARGE_PH7 的说明。）
    """
    s = standard_only(seq)
    if not s:
        return 0.0
    q = 0.0
    # N 端氨基：pH < pKa 时带正电
    q += 1.0 / (1.0 + 10 ** (ph - N_TERM_PKA))
    # C 端羧基：pH > pKa 时带负电
    q -= 1.0 / (1.0 + 10 ** (C_TERM_PKA - ph))
    for c in s:
        if c in SIDE_CHAIN_PKA:
            pka = SIDE_CHAIN_PKA[c]
            if c in NEGATIVE_AA or c in ("C", "Y"):
                q -= 1.0 / (1.0 + 10 ** (pka - ph))
            else:  # K / R / H
                q += 1.0 / (1.0 + 10 ** (ph - pka))
    return round(q, 4)


def net_charge_fast(seq: str) -> float:
    """pH 7.4 净电荷的快速近似（固定电荷表）。

    在大规模数据清洗（几十万条）里，H-H 的对数运算会成为瓶颈，
    这种场合用固定表足够——反正清洗只需要粗筛"这条序列是不是电荷极端异常"。
    但奖励函数里必须用精确版，因为奖励是要被优化的目标，近似误差会被放大。
    """
    return round(sum(SIDE_CHAIN_CHARGE_PH7.get(c, 0.0) for c in seq), 4)


def isoelectric_point(seq: str, tol: float = 1e-3, max_iter: int = 100) -> float:
    """等电点 pI：净电荷为 0 时的 pH，用二分法求解。

    pI 直接决定多肽在生理 pH 下的带电状态，进而影响溶解度与肾清除。
    二分而不是牛顿法：net_charge 关于 pH 单调递减，二分保证收敛且不需要导数。
    """
    s = standard_only(seq)
    if not s:
        return 7.0
    lo, hi = 0.0, 14.0
    if net_charge(s, lo) < 0:
        return lo
    if net_charge(s, hi) > 0:
        return hi
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if net_charge(s, mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return round((lo + hi) / 2.0, 3)


def aromaticity(seq: str) -> float:
    """芳香族残基（F/W/Y）占比。关系到 π-π 堆积、紫外吸收与膜穿透能力。"""
    s = standard_only(seq)
    if not s:
        return 0.0
    return round(sum(1 for c in s if c in AROMATIC_AA) / len(s), 4)


def aliphatic_index(seq: str) -> float:
    """脂肪族指数（Ikai, 1980）：AI = A% + 2.9·V% + 3.9·(I% + L%)。

    其中的 % 是摩尔百分数（0-100，不是摩尔分数），所以理论值域是
    [0, 390]——一条 100% I/L 的序列取满 390，典型球蛋白落在 60-120。
    注意量纲：分子分母各乘一次 100 会得到 100 倍的数值，
    使下游按该量纲标定的阈值全部失效。

    本是衡量球蛋白热稳定性的经典指标——脂肪族侧链比例高的蛋白往往 Tm 更高，
    因为疏水核心堆积更紧密、构象熵损失更被补偿。用在多肽上，
    它是序列层面唯一能免费拿到的热稳定性代理量，因此被"结构合理性"维度采纳。
    """
    pct = amino_acid_composition(seq, percent=True)
    if not pct:
        return 0.0
    ai = (
        pct.get("A", 0.0)
        + 2.9 * pct.get("V", 0.0)
        + 3.9 * (pct.get("I", 0.0) + pct.get("L", 0.0))
    )
    return round(ai, 3)


# ==========================================================================
# 3. 二级结构与两亲性
# ==========================================================================
def mean_propensity(seq: str, scale: str = "helix") -> float:
    """平均二级结构倾向值。scale ∈ {helix, sheet, coil}。

    这个量回答的是"这条序列整体愿意变成什么形状"。它不预测真实构象
    （那需要结构预测模型），但在奖励函数里作为一个便宜的"无义突变探测器"很有效：
    如果一条已知是 α-螺旋的两亲性抗菌肽，被改造成平均螺旋倾向大幅下降，
    大概率是模型为了刷其他奖励而把序列改坏了。
    """
    tables = {
        "helix": HELIX_PROPENSITY,
        "sheet": SHEET_PROPENSITY,
        "coil": COIL_PROPENSITY,
    }
    if scale not in tables:
        raise ValueError(f"unknown propensity scale: {scale}")
    table = tables[scale]
    s = standard_only(seq)
    if not s:
        return 1.0
    return round(sum(table[c] for c in s) / len(s), 4)


def hydrophobic_moment(seq: str, *, angle_deg: float = 100.0, window: int | None = None) -> float:
    """疏水矩 μH —— 两亲性（amphipathicity）的定量刻画（Eisenberg 1984）。

    算法：把序列看成按固定周期排列的残基，用疏水性向量做矢量和：
        μH = sqrt( (Σ Hn·sin(n·δ))² + (Σ Hn·cos(n·δ))² ) / N
    其中 δ 是每个残基沿该二级结构轴旋转的角度：
      * δ = 100° → α-螺旋（理想 3.6 残基/圈，即 100°）
      * δ = 160° → β-折叠（理想 2.25 残基/圈）
    先减去标度均值使随机序列期望为 0，得到的值才可跨序列横向比较。

    window 参数：只在滑动窗口内取最大值。原因是很多活性肽（如抗菌肽）
    只有局部一两个螺旋面是两亲的，全序列平均会把信号抹平。

    window 必须 ≥ 1。传 0 或负数在这里会切出空片段，
    最后在 `_moment_of` 里除以片段长度变成 ZeroDivisionError——
    报错信息和真正的原因（窗口参数非法）完全对不上，所以在这里显式拦掉。
    """
    if window is not None and window < 1:
        raise ValueError(f"window 必须 ≥ 1，收到 {window}")

    s = standard_only(seq)
    if len(s) < 2:
        return 0.0
    delta = math.radians(angle_deg)
    if window is not None and window < len(s):
        # 滑动窗口取最大 μH——识别"最强两亲片段"
        return round(
            max(
                _moment_of(s[i : i + window], delta)
                for i in range(len(s) - window + 1)
            ),
            4,
        )
    return round(_moment_of(s, delta), 4)


def _moment_of(fragment: str, delta: float) -> float:
    """对一个片段求和疏水矩（不含窗口逻辑，供上面复用）。"""
    sin_sum = 0.0
    cos_sum = 0.0
    for idx, c in enumerate(fragment):
        h = EISENBERG[c] - EISENBERG_MEAN  # 零均值化
        sin_sum += h * math.sin(idx * delta)
        cos_sum += h * math.cos(idx * delta)
    return math.hypot(sin_sum, cos_sum) / len(fragment)


def hydrophobic_moment_profile(seq: str, window: int = 11, angle_deg: float = 100.0) -> list[float]:
    """逐窗口的疏水矩曲线。

    用途：SOP 报告里画"两亲性沿序列分布"图，也是评估阶段判断
    "模型是不是把活性面改花了"的可视化依据。
    """
    s = standard_only(seq)
    if len(s) < window:
        return [round(_moment_of(s, math.radians(angle_deg)), 4)] if s else []
    delta = math.radians(angle_deg)
    return [
        round(_moment_of(s[i : i + window], delta), 4)
        for i in range(len(s) - window + 1)
    ]


def helical_wheel(seq: str) -> list[list[int]]:
    """把序列投射到螺旋轮（helical wheel）上，返回每一"扇区"的残基下标。

    α-螺旋每圈 3.6 个残基，所以轮上 18 个位置约等于 5 圈。
    同一下标 mod 18 的残基在空间上大致同侧——这就是判断"疏水面/亲水面是否分明"
    的几何依据，比只算一个标量 μH 更能解释"为什么这条序列构象合理"。
    """
    wheel: list[list[int]] = [[] for _ in range(18)]
    # 轮盘均分成 18 个扇区（每扇区 360/18 = 20°），α-螺旋每残基转过约 100°，
    # 换算成扇区就是 100/20 = 5 个。全程整数运算：原先写成
    # `int(round(i * 100.0 / 20.0))` 绕了一圈浮点，结果与 `i * 5` 完全等价，
    # 却多了一次舍入和一个多余的 int()，读的人还得自己把量纲化简一遍。
    for i in range(len(seq)):
        wheel[(i * 5) % 18].append(i)
    return wheel


def amphipathic_face_score(seq: str, window: int = 11) -> float:
    """两亲性"基本面"分数 ∈ [0, 1]：疏水残基在螺旋轮上是否聚在一侧。

    做法：用最优窗口的螺旋轮投影，计算"疏水残基所在扇区的分布集中度"。
    集中度用圆周方差衡量——若所有疏水残基挤在 3-4 个连续扇区，说明存在清晰疏水面。
    返回归一化到 [0,1] 的分数（1 = 界限分明）。
    """
    s = standard_only(seq)
    if len(s) < 4:
        return 0.0
    if len(s) > window:
        # 选疏水矩最大的窗口作为"功能面"候选
        best_i, best_mu = 0, -1.0
        delta = math.radians(100.0)
        for i in range(len(s) - window + 1):
            mu = _moment_of(s[i : i + window], delta)
            if mu > best_mu:
                best_i, best_mu = i, mu
        s = s[best_i : best_i + window]

    # 把疏水残基的角度映射到单位圆上求矢量和长度，长度越接近 1 越集中
    xs, ys = [], []
    for i, c in enumerate(s):
        if c in HYDROPHOBIC_AA:
            ang = math.radians(i * 100.0)
            xs.append(math.cos(ang))
            ys.append(math.sin(ang))
    if len(xs) < 2:
        return 0.0
    r = math.hypot(sum(xs) / len(xs), sum(ys) / len(ys))
    return round(min(1.0, r), 4)


# ==========================================================================
# 4. 电荷分布与聚集倾向
# ==========================================================================
def charge_asymmetry(seq: str) -> float:
    """电荷沿序列的不对称度 ∈ [0, 1]。

    对短肽而言，正电荷集中在 N 端一侧、疏水段集中在 C 端一侧，
    是抗菌肽/穿膜肽的经典拓扑（"正电头 + 疏水尾"）。
    这里用 N 半段与 C 半段的净电荷差做归一化，0 = 完全对称，1 = 极端不对称。
    """
    s = standard_only(seq)
    if len(s) < 4:
        return 0.0
    mid = len(s) // 2
    q_n = net_charge_fast(s[:mid])
    q_c = net_charge_fast(s[mid:])
    return round(min(1.0, abs(q_n - q_c) / max(1.0, abs(q_n) + abs(q_c))), 4)


def max_hydrophobic_run(seq: str) -> int:
    """最长连续疏水残基片段长度。

    经验上连续 ≥5 个疏水残基会显著提升聚集与血浆蛋白结合（影响半衰期），
    这在多肽设计里是需要主动规避的模式，因此作为一条硬性惩罚规则。
    """
    best = cur = 0
    for c in seq:
        if c in HYDROPHOBIC_AA:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def aggregation_propensity(seq: str, window: int = 7) -> float:
    """聚集倾向的启发式评分 ∈ [0, 1]，越高越容易自聚集。

    说明：真实工具（TANGO / AGGRESCAN / SAP）是基于实验标定过的算法，
    生产环境应当接入它们。这里实现的是一个透明的启发式代理，只用两件事：
      1. 滑动窗口的平均 KD 疏水性（窗口内越疏水越容易暴露黏性表面）
      2. β-折叠倾向（聚集的骨架构象前提）
    这样做的好处是：完全可解释、CPU 上零成本、可批量跑几十万条，
    适合作为强化学习的稠密奖励项；同时它和真工具同向（正相关），
    不会把优化方向带偏——这正是"留给后续替换成真工具"的接口设计。
    """
    s = standard_only(seq)
    if len(s) < 3:
        return 0.0
    w = min(window, len(s))
    worst = 0.0
    for i in range(len(s) - w + 1):
        frag = s[i : i + w]
        hyd = sum(KYTE_DOOLITTLE[c] for c in frag) / w
        # 把疏水性(-4.5~4.5)线性映射到 [0,1]，0 以下视为不聚集
        hyd_norm = max(0.0, min(1.0, (hyd + 1.0) / 3.0))
        sheet = sum(SHEET_PROPENSITY[c] for c in frag) / w
        sheet_norm = max(0.0, min(1.0, (sheet - 0.7) / 0.8))
        worst = max(worst, 0.6 * hyd_norm + 0.4 * sheet_norm)
    return round(worst, 4)


def flexibility_score(seq: str) -> float:
    """柔性评分 ∈ [0, 1]，基于 Karplus-Schulz 柔性标度的均值映射。

    柔性高的区域通常是活性位点（需要诱导契合），但整条序列都柔性说明构象不稳定，
    在"结构合理性"里作为一把双刃剑使用：活性口袋区柔性高加分，
    全局柔性过高扣分。
    """
    # Karplus & Schulz (1985) 归一化柔性标度，F 最高、P/G 高，W/C 最低
    flex = {
        "A": 0.984, "R": 0.903, "N": 1.048, "D": 1.068, "C": 0.906,
        "Q": 1.037, "E": 1.094, "G": 1.031, "H": 0.950, "I": 0.886,
        "L": 0.935, "K": 1.102, "M": 0.952, "F": 0.915, "P": 1.049,
        "S": 1.046, "T": 0.997, "W": 0.904, "Y": 0.929, "V": 0.931,
    }
    s = standard_only(seq)
    if not s:
        return 0.5
    raw = sum(flex.get(c, 1.0) for c in s) / len(s)
    # 标度实测范围约 [0.886, 1.102]，线性映射到 [0,1]
    return round(min(1.0, max(0.0, (raw - 0.886) / (1.102 - 0.886))), 4)


# ==========================================================================
# 5. 汇总结构：一次算完，避免重复遍历
# ==========================================================================
@dataclass
class SequenceProperties:
    """一条序列的全部物化性质快照。

    做成 dataclass 而不是 dict：奖励函数、评估报告、API 响应三处都要用它，
    字段名写错要能在导入期/类型检查期发现，而不是等到线上 KeyError。

    `heuristic` 字段显式标记哪些指标是启发式代理，写进报告时一并输出——
    对研发人员诚实交代"这个分数是怎么来的"，是 ToB 产品能不能被信任的前提。
    """

    sequence: str
    length: int
    molecular_weight: float
    gravy: float
    net_charge_ph74: float
    isoelectric_point: float
    aromaticity: float
    aliphatic_index: float
    helix_propensity: float
    sheet_propensity: float
    coil_propensity: float
    hydrophobic_moment: float          # 全序列平均 μH
    max_hydrophobic_moment: float      # 最优 11 窗口的 μH（更贴近功能面）
    amphipathic_face: float
    charge_asymmetry: float
    max_hydrophobic_run: int
    aggregation_propensity: float
    flexibility: float
    coverage: float
    heuristic: tuple[str, ...] = field(
        default=("aggregation_propensity", "amphipathic_face", "charge_asymmetry"),
        repr=False,
    )

    @property
    def is_in_core_length(self) -> bool:
        """是否落在训练语料主分布（5-30 残基）内。评估时据此分组看指标。"""
        return CORE_PEPTIDE_LEN[0] <= self.length <= CORE_PEPTIDE_LEN[1]


def compute_properties(
    seq: str, *, window: int = 11, cyclic: bool = False
) -> SequenceProperties:
    """计算一条序列的全部性质。训练/推理路径上的统一入口。

    `cyclic` 参数
    -------------
    以前没有这个参数，分子量一律按线性肽算（残基质量之和 + 一分子水）。
    对头尾环化的序列这是错的：环化时两端会再缩合脱去一分子水，
    所以环肽的分子量比线性形式小 18.02 Da。18 Da 在质谱上正好是
    一个能分辨的差值，而分子量又是报告里给研发人员看的字段——
    一条环肽报出偏大 18 Da 的分子量，形同给错了身份。

    只有 `head_to_tail` 会脱去这一分子水；二硫键/订书肽/内酰胺桥
    都是侧链交联，主链两端仍保持游离，分子量与线性形式一致。

    净电荷不受环化影响：主链两端的 -NH3+/COO- 在成酰胺键后确实消失，
    但那两个电荷恰好互相抵消（+1 和 -1），所以净电荷不变——
    这个结论值得写下来，因为它看起来像是"应该有影响"的地方。
    """
    s = canonicalize(seq)
    return SequenceProperties(
        sequence=s,
        length=len(s),
        molecular_weight=_molecular_weight(s, cyclic=cyclic),
        gravy=gravy(s),
        net_charge_ph74=net_charge(s, 7.4),
        isoelectric_point=isoelectric_point(s),
        aromaticity=aromaticity(s),
        aliphatic_index=aliphatic_index(s),
        helix_propensity=mean_propensity(s, "helix"),
        sheet_propensity=mean_propensity(s, "sheet"),
        coil_propensity=mean_propensity(s, "coil"),
        hydrophobic_moment=hydrophobic_moment(s, angle_deg=100.0),
        max_hydrophobic_moment=hydrophobic_moment(s, angle_deg=100.0, window=window),
        amphipathic_face=amphipathic_face_score(s, window=window),
        charge_asymmetry=charge_asymmetry(s),
        max_hydrophobic_run=max_hydrophobic_run(s),
        aggregation_propensity=aggregation_propensity(s),
        flexibility=flexibility_score(s),
        coverage=coverage(s),
    )


def compute_properties_batch(
    seqs: Iterable[str], *, window: int = 11, cyclic: bool = False
) -> list[SequenceProperties]:
    """批量版本。目前是朴素循环；接口先定下来，将来换多进程/C 扩展不影响调用方。

    之所以现在就抽象出 batch 接口：GRPO 每步要为 group_size × batch_size
    条生成序列打分（比如 8 × 16 = 128 条），单条调用会成为训练吞吐瓶颈。
    """
    return [compute_properties(s, window=window, cyclic=cyclic) for s in seqs]


def summarize(seqs: Sequence[str]) -> dict[str, float]:
    """统计一组序列的聚合指标，供数据质量报表与运营看板使用。"""
    props = compute_properties_batch(seqs)
    if not props:
        return {}
    n = len(props)
    return {
        "count": float(n),
        "mean_length": round(sum(p.length for p in props) / n, 2),
        "mean_gravy": round(sum(p.gravy for p in props) / n, 4),
        "mean_net_charge": round(sum(p.net_charge_ph74 for p in props) / n, 4),
        "mean_hydrophobic_moment": round(
            sum(p.max_hydrophobic_moment for p in props) / n, 4
        ),
        "mean_aggregation": round(sum(p.aggregation_propensity for p in props) / n, 4),
        "high_aggregation_ratio": round(
            sum(1 for p in props if p.aggregation_propensity > 0.6) / n, 4
        ),
        "mean_coverage": round(sum(p.coverage for p in props) / n, 4),
    }
