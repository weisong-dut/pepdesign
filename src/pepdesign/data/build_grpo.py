"""构造 GRPO 的训练 prompt 集。

GRPO 只需要 prompt，答案由模型采样生成。每个样本的 meta 携带奖励函数
所需的上下文（`base_sequence`、`cyclic_type`、任务类型等）。
prompt 在任务类型、目标权重、长度/电荷约束与环化方式上做多样化。
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence

from pepdesign.bio.cyclization import feasible_cyclic_type
from pepdesign.data.build_sft import (
    MULTI_OBJECTIVE_PRESETS,
    STRATEGY_TO_DIMENSION,
    _infer_target_name,
    build_denovo_index,
    cyclic_balanced_pool,
    degrade_sequence,
    search_denovo_candidate,
    strategy_for_dimension,
)
from pepdesign.data.schema import PeptideRecord
from pepdesign.prompts.templates import (
    build_grpo_sample,
    de_novo_prompt,
    multi_objective_prompt,
    stabilize_prompt,
)
from pepdesign.utils.logger import get_logger

logger = get_logger(__name__)

# 三种任务的比例。stabilize 占比最高，对应先导化合物优化这一主要场景。
TASK_MIX: dict[str, float] = {
    "stabilize": 0.50,
    "de_novo": 0.35,
    "multi_objective": 0.15,
}

# 多目标权重预设与 SFT 阶段共用 build_sft.MULTI_OBJECTIVE_PRESETS，不在此重复定义。

#: 环化方式采样权重。de_novo 只请求 SFT 教过的 none / disulfide；
#: stabilize/multi 的类型由 feasible_cyclic_type 按记录决定。
CYCLIC_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("none", 0.65),
    ("disulfide", 0.35),
)

FUNCTION_CN = {
    "antibacterial": "抗菌",
    "antitumor": "抗肿瘤",
    "antiviral": "抗病毒",
    "antifungal": "抗真菌",
    "immunomodulatory": "免疫调节",
    "antiinflammatory": "抗炎",
    "cell_penetrating": "细胞穿膜",
    "targeting": "靶向",
    "inhibitor": "抑制剂",
    "other": "生物活性",
}


def _weighted_choice(rng: random.Random, items: Sequence[tuple[str, float]]) -> str:
    """按权重随机选一项。使用显式传入的 rng，保证整条流程的随机性同源。"""
    total = sum(w for _, w in items)
    r = rng.random() * total
    acc = 0.0
    for name, weight in items:
        acc += weight
        if r <= acc:
            return name
    return items[-1][0]


def build_stabilize_prompts(
    records: Sequence[PeptideRecord], rng: random.Random, count: int
) -> list[dict]:
    """构造稳定性优化的 prompt：先劣化序列，再让模型在 RL 中修复。

    与 SFT 不同，这里不提供标准答案，修复程度由奖励量化。
    """
    samples: list[dict] = []
    attempts = 0
    max_attempts = count * 10  # 防止在劣化失败率高时死循环

    while len(samples) < count and attempts < max_attempts:
        attempts += 1
        rec = records[rng.randrange(len(records))]
        # 数据里的 disulfide 标签不保证 Cys 数自洽，按原始序列判断可行性，
        # 不可行则退回线性肽。
        cyclic = feasible_cyclic_type(rec.sequence, rec.cyclic_type)
        degraded = degrade_sequence(rec.sequence, rng, cyclic_type=cyclic)
        if degraded is None:
            continue
        bad_seq, strategy = degraded

        # 使用 build_sft 的共享映射，避免两处副本漂移导致 prompt 与缺陷不符。
        dimension = STRATEGY_TO_DIMENSION[strategy]

        spec = stabilize_prompt(
            sequence=bad_seq,
            target_name=rec.target or _infer_target_name(rec),
            weak_dimension=dimension,
            preserve="靶点亲和力" if strategy != "break_amphipathic" else "整体活性",
            cyclic_type=cyclic,
        )
        sample = build_grpo_sample(spec)
        # base_sequence 供相似度约束与亲和力基准；不写 reference_sequence，
        # 避免把标准答案泄漏进奖励。
        sample["meta"].update(
            {
                "base_sequence": bad_seq,
                "degrade_strategy": strategy,
                "task": "stabilize",
                # 环化方式必须进 meta，否则 CompositeScorer 不会加环化约束。
                "cyclic_type": cyclic,
            }
        )
        samples.append(sample)

    if attempts >= max_attempts:
        logger.warning("稳定性 prompt 构造提前结束：尝试 %d 次只得到 %d 条", attempts, len(samples))
    return samples


def build_denovo_prompts(
    records: Sequence[PeptideRecord], rng: random.Random, count: int
) -> list[dict]:
    """构造从头设计的 prompt：先采样约束区间，再到索引里检索满足它的记录。

    与 SFT 阶段共用 `search_denovo_candidate`，保证两阶段约束分布同源。
    """
    index = build_denovo_index(list(records))
    samples: list[dict] = []
    attempts = 0
    max_attempts = count * 5
    while len(samples) < count and attempts < max_attempts:
        attempts += 1
        found = search_denovo_candidate(index, rng)
        if found is None:
            continue
        rec, cyclic, (lo, hi), (q_lo, q_hi) = found

        spec = de_novo_prompt(
            target_name=rec.target or _infer_target_name(rec),
            target_desc="",
            function=FUNCTION_CN.get(rec.function, "生物活性"),
            length_range=(lo, hi),
            cyclic_type=cyclic,
            net_charge_range=(q_lo, q_hi),
        )
        sample = build_grpo_sample(spec)
        sample["meta"].update(
            {
                "task": "de_novo",
                "cyclic_type": cyclic,
                "length_range": [lo, hi],
                "charge_range": [q_lo, q_hi],
                "function": rec.function,
            }
        )
        samples.append(sample)
    return samples


def build_multi_objective_prompts(
    records: Sequence[PeptideRecord], rng: random.Random, count: int
) -> list[dict]:
    """构造多目标协同优化的 prompt。"""
    samples: list[dict] = []
    attempts = 0
    max_attempts = count * 10

    while len(samples) < count and attempts < max_attempts:
        attempts += 1
        rec = records[rng.randrange(len(records))]
        cyclic = feasible_cyclic_type(rec.sequence, rec.cyclic_type)
        objectives = MULTI_OBJECTIVE_PRESETS[
            rng.randrange(len(MULTI_OBJECTIVE_PRESETS))
        ]

        # 劣化策略跟随主导权重，与 SFT 侧逻辑一致；否则 prompt 里的权重对模型无意义。
        degraded = None
        strategy = None
        for dimension, _weight in sorted(objectives.items(), key=lambda kv: -kv[1]):
            candidate = strategy_for_dimension(dimension)
            if candidate is None:
                continue
            result = degrade_sequence(
                rec.sequence, rng, cyclic_type=cyclic, strategy=candidate
            )
            if result is not None:
                degraded, strategy = result
                break
        if degraded is None:
            continue
        bad_seq = degraded

        spec = multi_objective_prompt(
            sequence=bad_seq,
            objectives=objectives,
            target_name=rec.target or _infer_target_name(rec),
            allow_length_change=False,
            cyclic_type=cyclic,
        )
        sample = build_grpo_sample(spec)
        sample["meta"].update(
            {
                "task": "multi_objective",
                # 权重直接传给奖励函数，使奖励与 prompt 目标一致；
                # 不写 reference_sequence，理由同 build_stabilize_prompts。
                "objectives": objectives,
                "base_sequence": bad_seq,
                "degrade_strategy": strategy,
                # 缺少 cyclic_type 时环化约束不会被评分。
                "cyclic_type": cyclic,
            }
        )
        samples.append(sample)
    return samples


def build_grpo_dataset(
    records: Sequence[PeptideRecord],
    *,
    total: int = 2000,
    seed: int = 42,
    mix: dict[str, float] | None = None,
    cyclic_ratio: float = 0.30,
) -> list[dict]:
    """构造完整的 GRPO prompt 集。

    `cyclic_ratio` 与 SFT 阶段共用同一个加权池策略，保证两阶段的环肽占比同源。
    """
    if not records:
        logger.warning("记录为空，无法构造 GRPO 数据集")
        return []

    rng = random.Random(seed)
    mix = mix or TASK_MIX
    pool = cyclic_balanced_pool(list(records), cyclic_ratio=cyclic_ratio)

    parts = {
        "stabilize": build_stabilize_prompts(
            pool, rng, int(total * mix.get("stabilize", 0.5))
        ),
        "de_novo": build_denovo_prompts(
            pool, rng, int(total * mix.get("de_novo", 0.35))
        ),
        "multi_objective": build_multi_objective_prompts(
            pool, rng, int(total * mix.get("multi_objective", 0.15))
        ),
    }

    merged = [sample for sub in parts.values() for sample in sub]
    rng.shuffle(merged)   # 打乱以避免训练时同类任务连续出现

    dist = Counter(s["meta"]["task"] for s in merged)
    logger.info("GRPO 数据集构造完成：%d 条，任务分布 %s", len(merged), dict(dist))
    return merged
