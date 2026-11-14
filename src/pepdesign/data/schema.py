"""数据模型：一条多肽记录的结构。

数据用 dataclass + 显式 from_dict/to_dict，而不是 pydantic：数据由机器产出、
量级大，序列化开销更低；to_dict 是显式白名单，保证落盘 schema 稳定。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pepdesign.bio.sequence import (
    canonicalize,
    compute_properties,
    has_noncanonical_residues,
    strip_terminal_modifications,
)

# 多肽功能分类。用枚举字符串而非自由文本，以支持按功能分层采样与分组评估。
FunctionType = Literal[
    "antibacterial",     # 抗菌
    "antitumor",         # 抗肿瘤
    "antiviral",         # 抗病毒
    "antifungal",        # 抗真菌
    "immunomodulatory",  # 免疫调节
    "antiinflammatory",  # 抗炎
    "cell_penetrating",  # 细胞穿膜
    "targeting",         # 靶向
    "inhibitor",         # 抑制剂
    "other",
]

CyclicType = Literal["none", "head_to_tail", "disulfide", "staple", "lactam"]

# 数据来源标注。来源决定用途：公开库可做 SFT，带实验值的数据才能标定奖励，
# 评估集必须与训练集来源隔离。
DataSource = Literal["uniprot", "apd3", "dramp", "dbaasp", "pdb", "internal", "synthetic"]

#: 实测值方向：+1 = 越大越好，-1 = 越小越好。
#: 表中没有的测定类型直接报错，不做默认假设——方向猜反会让相关系数整体反
#: 转且不易察觉。
ASSAY_ORIENTATION: dict[str, int] = {
    # 越小越好：抑菌/抑制浓度、解离常数
    "mic": -1,
    "ic50": -1,
    "ec50": -1,
    "kd": -1,
    "ki": -1,
    # 越大越好：熔解温度、半衰期
    "tm": +1,
    "dtm": +1,
    "halflife": +1,
    "t12": +1,
}


def orientation_for(assay_type: str) -> int:
    """返回该测定类型的方向（+1 越大越好 / -1 越小越好），查不到则抛错。

    评估层直接复用，不另写一份表；方向判断错误会导致结论整体反转。
    """
    key = str(assay_type).strip().lower().replace("-", "").replace("_", "")
    try:
        return ASSAY_ORIENTATION[key]
    except KeyError as exc:
        raise KeyError(
            f"未知的测定类型 {assay_type!r}，无法确定'越大越好'还是'越小越好'。"
            f"请把它加进 data.schema.ASSAY_ORIENTATION 并写明方向——"
            f"猜一个方向的代价是所有相关系数和取值方向的整体反转，"
            f"而反转后的数字看上去完全正常。已知类型：{sorted(ASSAY_ORIENTATION)}"
        ) from exc


@dataclass
class ActivityRecord:
    """一条活性/性质实测记录。

    一条序列可挂多条记录（不同靶点、菌株、测定方法），保留原始测值，
    需要时再聚合。
    """

    assay_type: str = ""          # MIC / IC50 / Tm / 血清半衰期 ...
    target: str = ""              # 靶点或菌株名
    value: float | None = None
    unit: str = ""
    method: str = ""              # 测定方法，用于判断不同数据是否可比
    source_ref: str = ""          # 文献 PMID / 数据库 ID，便于溯源

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PeptideRecord:
    """标准化的多肽记录。数据管道中流转的统一单元。"""

    # ---- 核心字段 ----
    sequence: str
    name: str = ""
    source: DataSource = "uniprot"
    source_id: str = ""

    # ---- 领域标注 ----
    function: FunctionType = "other"
    cyclic_type: CyclicType = "none"
    target: str = ""
    organism: str = ""
    # 是否含非天然/修饰残基。这类序列不能进 SFT 语料（模型只生成 20 种
    # 标准氨基酸），但可作为改造方向参考。
    has_noncanonical: bool = False
    modifications: list[str] = field(default_factory=list)

    # ---- 性质快照（惰性计算并缓存）----
    properties: dict[str, Any] = field(default_factory=dict)

    # ---- 实验数据 ----
    activities: list[ActivityRecord] = field(default_factory=list)

    # ---- 管道元信息：用于回溯这条样本的处理过程 ----
    lineage: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        """构造时规范化序列，并把末端修饰标注从序列字符串中剥离。

        必须先于 canonicalize：'-'、数字与大小写都会被它抹掉。
        只识别 `Ac-` 前缀和 `-NH2` / `-CONH2` 后缀，宁可漏识别也不误识别。
        """
        raw = str(self.sequence or "").strip()

        # 以下三件事必须在 canonicalize 之前完成，它们依赖的字符会被抹掉。

        # 1) 剥离末端修饰，规则与 prompts/parser.py 共用 strip_terminal_modifications。
        body, term_mods = strip_terminal_modifications(raw)
        for m in term_mods:
            if m not in self.modifications:
                self.modifications.append(m)

        # 2) 检测非标准残基。必须放在第 1 步之后：`Ac-` 中的 `c` 是修饰标注，
        #    先检测会把带乙酰化的样本误标成含 D-型残基。
        if has_noncanonical_residues(body):
            self.has_noncanonical = True

        # 3) 规范化
        self.sequence = canonicalize(body)

    # ------------------------------------------------------------------
    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def is_cyclic(self) -> bool:
        return self.cyclic_type != "none"

    @property
    def has_experimental_data(self) -> bool:
        return any(a.value is not None for a in self.activities)

    def compute_properties(self, *, force: bool = False) -> dict[str, Any]:
        """计算并缓存物化性质快照。"""
        if self.properties and not force:
            return self.properties
        props = compute_properties(self.sequence)
        self.properties = asdict(props)
        return self.properties

    def best_activity(self, assay_type: str | None = None) -> float | None:
        """取最强的活性值，方向由 `orientation_for` 决定。

        先按测定类型的方向归一化再取最大；未知测定类型抛 KeyError。
        """
        sign = orientation_for(assay_type) if assay_type is not None else None
        vals = [
            (a.value, sign if sign is not None else orientation_for(a.assay_type))
            for a in self.activities
            if a.value is not None and (assay_type is None or a.assay_type == assay_type)
        ]
        if not vals:
            return None
        # 归一化到"越大越好"之后再取最大。
        return max(vals, key=lambda pair: pair[1] * pair[0])[0]

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """显式白名单序列化。字段顺序稳定 → 落盘 diff 友好。"""
        return {
            "sequence": self.sequence,
            "name": self.name,
            "source": self.source,
            "source_id": self.source_id,
            "function": self.function,
            "cyclic_type": self.cyclic_type,
            "target": self.target,
            "organism": self.organism,
            "has_noncanonical": self.has_noncanonical,
            "modifications": list(self.modifications),
            "properties": self.properties,
            "activities": [a.to_dict() for a in self.activities],
            "lineage": self.lineage,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PeptideRecord:
        """反序列化。对未知字段宽容忽略（而不是报错），
        因为语料文件可能是旧版本产出的，加字段不应该让旧数据读不了。"""
        activities = [
            ActivityRecord(**a) for a in d.get("activities", []) if isinstance(a, dict)
        ]
        known = {
            "sequence", "name", "source", "source_id", "function", "cyclic_type",
            "target", "organism", "has_noncanonical", "modifications",
            "properties", "lineage",
        }
        kwargs = {k: v for k, v in d.items() if k in known}
        kwargs["activities"] = activities
        return cls(**kwargs)


@dataclass
class DataQualityReport:
    """清洗阶段的质量报告，落盘为 JSON 与 Markdown，作为数据集版本元数据。"""

    raw_count: int = 0
    kept_count: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)
    dedup_removed: int = 0
    length_distribution: dict[str, int] = field(default_factory=dict)
    function_distribution: dict[str, int] = field(default_factory=dict)
    source_distribution: dict[str, int] = field(default_factory=dict)
    cyclic_ratio: float = 0.0
    mean_length: float = 0.0
    property_summary: dict[str, float] = field(default_factory=dict)

    @property
    def keep_rate(self) -> float:
        return self.kept_count / self.raw_count if self.raw_count else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["keep_rate"] = round(self.keep_rate, 4)
        return d

    def to_markdown(self) -> str:
        """生成人读的 Markdown 报告。"""
        lines = [
            "# 数据质量报告",
            "",
            f"- 原始样本数：{self.raw_count}",
            f"- 保留样本数：{self.kept_count}（保留率 {self.keep_rate:.1%}）",
            f"- 去重移除：{self.dedup_removed}",
            f"- 环肽占比：{self.cyclic_ratio:.1%}",
            f"- 平均长度：{self.mean_length:.1f}",
            "",
            "## 丢弃原因分布",
            "",
        ]
        for reason, cnt in sorted(self.drop_reasons.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{reason}`：{cnt}")
        lines += ["", "## 功能分布", ""]
        for func, cnt in sorted(self.function_distribution.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {func}：{cnt}")
        lines += ["", "## 来源分布", ""]
        for src, cnt in sorted(self.source_distribution.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {src}：{cnt}")
        lines += ["", "## 性质概览", ""]
        for key, val in self.property_summary.items():
            lines.append(f"- {key}：{val}")
        return "\n".join(lines) + "\n"
