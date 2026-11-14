"""指令模板：把研发需求翻译成模型可理解、且能被解析器读回的文本。

模板使用 `### 设计序列` 等强标记保证输出可确定性解析；训练与推理必须共用同一套
模板，否则格式不一致会导致模型表现下降。任务类型包括 de_novo、stabilize、
multi_objective 与 knowledge，结构化约束直接写入指令。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# 电荷阈值统一从 bio 层取：prompt、软惩罚与结构否决必须共用同一套数值。
from pepdesign.bio.sequence import CHARGE_PROMPT_LIMIT

TaskType = Literal["de_novo", "stabilize", "multi_objective", "knowledge"]

# --------------------------------------------------------------------------
# 标记符：训练、解析、推理三处共用，改这里就全改
# --------------------------------------------------------------------------
SEQ_OPEN = "### 设计序列"
SEQ_CLOSE = "### 设计理由"
END_MARK = "### 结束"


SYSTEM_PROMPT = f"""你是多肽药物序列设计专家，服务于抗肿瘤、自身免疫、抗菌方向的先导化合物研发。

设计时必须遵守以下规范：
1. 只使用 20 种标准氨基酸的单字母代码（ACDEFGHIKLMNPQRSTVWY）。
2. 除非特别说明，默认设计线性肽；要求环化时按指定的环化方式给出序列。
3. 不要输出解释性的散文中夹杂序列，序列必须单独成行。
4. 优先保证靶点亲和力不下降，这是活性底线；在此前提下优化稳定性。
5. 避免出现连续 5 个以上疏水残基，避免净电荷绝对值超过 {CHARGE_PROMPT_LIMIT:.0f}（pH 7.4 下）。

输出格式（严格遵循）：
### 设计序列
<仅由大写单字母氨基酸组成的一行序列>
### 设计理由
<不超过三句话，说明关键改造点与其生化依据>
### 结束
"""


# --------------------------------------------------------------------------
# 每种任务一个模板构造函数
# --------------------------------------------------------------------------
@dataclass
class PromptSpec:
    """构造好的 prompt 及其元信息；`meta` 在 RL 阶段传给奖励函数。"""

    system: str
    user: str
    task_type: TaskType
    meta: dict = field(default_factory=dict)

    def as_messages(self) -> list[dict[str, str]]:
        """转成 chat 格式，直接喂给 tokenizer.apply_chat_template。"""
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


def de_novo_prompt(
    *,
    target_name: str,
    target_desc: str = "",
    function: str = "抗菌",
    length_range: tuple[int, int] = (10, 20),
    cyclic_type: str = "none",
    net_charge_range: tuple[float, float] | None = None,
    extra_constraints: str = "",
) -> PromptSpec:
    """从头设计的指令模板。

    结构化参数便于把约束变成可验证指标；自由文本由 `extra_constraints` 兜底。
    """
    lines = [
        f"【设计任务】针对 {target_name} 从头设计一条多肽序列。",
        f"【功能方向】{function}",
        f"【长度要求】{length_range[0]}-{length_range[1]} 个氨基酸",
    ]
    if target_desc:
        lines.append(f"【靶点说明】{target_desc}")
    if cyclic_type and cyclic_type != "none":
        cyclic_map = {
            "head_to_tail": "头尾环化",
            "disulfide": "二硫键环化（需成对 Cys）",
            "staple": "烃链订书（需标注订书位点）",
            "lactam": "内酰胺桥（i/i+4 酸碱残基对）",
        }
        lines.append(f"【环化方式】{cyclic_map.get(cyclic_type, cyclic_type)}")
    if net_charge_range:
        lo, hi = net_charge_range
        lines.append(f"【净电荷要求】pH 7.4 下介于 {lo:+.1f} 到 {hi:+.1f} 之间")
    if extra_constraints:
        lines.append(f"【其他约束】{extra_constraints}")

    lines.append("")
    lines.append("请给出满足上述要求的设计序列，并简要说明设计思路。")

    return PromptSpec(
        system=SYSTEM_PROMPT,
        user="\n".join(lines),
        task_type="de_novo",
        meta={
            "target_name": target_name,
            "function": function,
            "length_range": list(length_range),
            "cyclic_type": cyclic_type,
            "net_charge_range": list(net_charge_range) if net_charge_range else None,
        },
    )


CYCLIC_LABELS = {
    "head_to_tail": "头尾环化",
    "disulfide": "二硫键环化",
    "staple": "订书肽侧链交联",
    "lactam": "内酰胺环化",
}


def _cyclic_label(cyclic_type: str) -> str:
    return CYCLIC_LABELS.get(cyclic_type, cyclic_type)


def stabilize_prompt(
    *,
    sequence: str,
    target_name: str = "",
    weak_dimension: str = "酶解稳定性",
    preserve: str = "靶点亲和力",
    modifications: tuple[str, ...] = (),
    cyclic_type: str = "none",
    extra_constraints: str = "",
) -> PromptSpec:
    """稳定性优化指令模板，显式写明"在保持 X 不变的前提下提升 Y"。"""
    mods = ""
    if modifications:
        mods = "、".join(modifications)

    lines = [
        "【优化任务】对以下先导化合物序列做稳定性优化。",
        f"【原始序列】{sequence}",
        f"【优化目标】提升{weak_dimension}",
        f"【必须保持】{preserve}不下降",
    ]
    if cyclic_type not in ("", "none"):
        lines.append(
            f"【环化约束】保持{_cyclic_label(cyclic_type)}，不要破坏环化几何。"
        )
    if target_name:
        lines.append(f"【靶点】{target_name}")
    if mods:
        lines.append(f"【可用修饰】{mods}")
    if extra_constraints:
        lines.append(f"【其他约束】{extra_constraints}")

    lines.append("")
    lines.append(
        "请注意：不要改动参与靶点结合的关键残基；"
        "优先采用在酶切位点 P1' 位引入 Pro、末端封闭、非天然氨基酸替换等策略。"
    )
    lines.append("请给出优化后的序列并说明改造点。")

    return PromptSpec(
        system=SYSTEM_PROMPT,
        user="\n".join(lines),
        task_type="stabilize",
        meta={
            "base_sequence": sequence,
            "target_name": target_name,
            "weak_dimension": weak_dimension,
            "preserve": preserve,
            "modifications": list(modifications),
            "cyclic_type": cyclic_type,
        },
    )


def multi_objective_prompt(
    *,
    sequence: str,
    objectives: dict[str, float],
    target_name: str = "",
    allow_length_change: bool = False,
    cyclic_type: str = "none",
) -> PromptSpec:
    """多目标协同优化模板。

    `objectives` 为维度名到权重的映射，权重同时用于 prompt 与奖励函数，两边必须一致。
    """
    obj_str = "、".join(f"{k}(权重 {v:.0%})" for k, v in objectives.items())
    lines = [
        "【优化任务】对以下序列做多目标协同优化，兼顾各维度权衡。",
        f"【原始序列】{sequence}",
        f"【优化目标及权重】{obj_str}",
    ]
    if target_name:
        lines.append(f"【靶点】{target_name}")
    lines.append(
        "【长度约束】" + ("允许增减长度" if allow_length_change else "保持长度不变")
    )
    if cyclic_type not in ("", "none"):
        lines.append(
            f"【环化约束】保持{_cyclic_label(cyclic_type)}，不要破坏环化几何。"
        )
    lines.append("")
    lines.append("请给出优化后的序列，并说明你如何在各目标之间做权衡。")
    return PromptSpec(
        system=SYSTEM_PROMPT,
        user="\n".join(lines),
        task_type="multi_objective",
        meta={
            "base_sequence": sequence,
            "objectives": dict(objectives),
            "target_name": target_name,
            "allow_length_change": allow_length_change,
            "cyclic_type": cyclic_type,
        },
    )


def knowledge_prompt(*, question: str, context: str = "") -> PromptSpec:
    """领域知识问答模板。"""
    user = f"【问题】{question}"
    if context:
        user += f"\n\n【参考资料】{context}"
    return PromptSpec(
        system=SYSTEM_PROMPT,
        user=user,
        task_type="knowledge",
        meta={"question": question},
    )


# --------------------------------------------------------------------------
# 组装成训练样本
# --------------------------------------------------------------------------
def render_completion(reasoning: str, sequence: str) -> str:
    """把序列与理由渲染成标准输出格式；序列在前，便于解析提前停止。"""
    return f"{SEQ_OPEN}\n{sequence}\n{SEQ_CLOSE}\n{reasoning}\n{END_MARK}"


def build_sft_sample(
    spec: PromptSpec,
    *,
    sequence: str,
    reasoning: str = "",
) -> dict:
    """构造一条 SFT 训练样本（chat messages），由 tokenizer 负责特殊 token。"""
    return {
        "messages": [
            *spec.as_messages(),
            {"role": "assistant", "content": render_completion(reasoning, sequence)},
        ],
        "task_type": spec.task_type,
        "meta": spec.meta,
    }


def build_grpo_sample(spec: PromptSpec) -> dict:
    """构造一条 GRPO prompt 样本；`meta` 完整保留供奖励函数使用。"""
    return {
        "prompt": spec.as_messages(),
        "task_type": spec.task_type,
        "meta": spec.meta,
    }


def format_target_context(
    *,
    target_name: str,
    target_sequence: str = "",
    binding_motif: str = "",
    max_len: int = 512,
) -> str:
    """把靶点信息压缩成一段上下文。

    训练侧上下文只有 2048 token，按"名称 → 结合基序 → 序列前 N 残基"裁剪，
    优先保留相关信息。
    """
    parts = [f"靶点名称：{target_name}"]
    if binding_motif:
        parts.append(f"关键结合基序：{binding_motif}")
    if target_sequence:
        seq = target_sequence[:max_len]
        truncated = len(target_sequence) > max_len
        suffix = f"...（全长 {len(target_sequence)} 残基，此处截断显示）" if truncated else ""
        parts.append(f"靶点序列片段：{seq}{suffix}")
    return "\n".join(parts)
