"""把解码后的生成文本转换为 GRPO 训练所需的优势。

本模块不 import torch：生成本身不在此处，解析、打分、分组与优势计算均为纯逻辑。
解析失败或弱解析的样本标记为无效，不参与组内均值/标准差，但占位为 0.0；
优势必须逐组计算，不能跨 prompt 混合。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pepdesign.bio.sequence import CORE_PEPTIDE_LEN
from pepdesign.prompts.parser import ParsedSequence, extract_sequence, parse_stats
from pepdesign.rewards.base import RewardBreakdown
from pepdesign.rewards.composite import CompositeScorer
from pepdesign.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 解析强度的分级 —— "解析成功"有两种含义，别把它们混成一件事
# ---------------------------------------------------------------------------
#: 命中这些策略说明输出符合格式契约：`### 设计序列` 标记、XML 标签或代码块。
STRONG_PARSE_STRATEGIES: frozenset[str] = frozenset(
    {"marker", "xml_tag", "code_fence"}
)

#: 启发式兜底策略：从自由文本中挑出最像序列的片段。单字母氨基酸码覆盖范围广，
#: 随机文本也可能被识别为合法长度的序列，因此 RL 中按无效处理。
WEAK_PARSE_STRATEGIES: frozenset[str] = frozenset({"line_scan", "whole_text"})


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
def parse_completions(
    texts: Sequence[str],
    metas: Sequence[dict],
    *,
    require_strong_parse: bool = True,
) -> tuple[list[ParsedSequence], list[dict]]:
    """解析一批生成结果，返回 `(解析结果, 用于打分的 meta)`。

    返回新的 meta 列表，不修改入参。`min_len` 取 `CORE_PEPTIDE_LEN[0]`；
    `require_strong_parse=True` 时，弱解析（`WEAK_PARSE_STRATEGIES`）标记为无效。
    """
    if len(texts) != len(metas):
        raise ValueError(
            f"文本条数 ({len(texts)}) 与 meta 条数 ({len(metas)}) 不一致"
        )

    parsed_list: list[ParsedSequence] = []
    scoring_metas: list[dict] = []

    for text, meta in zip(texts, metas, strict=True):
        cyclic = str(meta.get("cyclic_type", "none") or "none")
        parsed = extract_sequence(
            text,
            min_len=CORE_PEPTIDE_LEN[0],
            expect_cyclic=(cyclic != "none"),
        )
        parsed_list.append(parsed)

        # 新 dict：保留原 meta 字段，避免把解析结论写回调用方复用的对象。
        scoring_meta = dict(meta)
        # 末端修饰需并入评分 meta：`protease.py` 读取 `meta["modifications"]`，
        # 缺失时 `Ac-...-NH2` 的酶切位点屏蔽拿不到奖励。
        if parsed.modifications:
            merged_mods = set(meta.get("modifications") or ()) | set(parsed.modifications)
            scoring_meta["modifications"] = sorted(merged_mods)
        if not parsed.ok:
            # 标记为"没有分数"而非"分数低"：优势计算会排除该样本，避免拉低组内基线。
            scoring_meta["invalid_reason"] = f"parse_failed:{parsed.reason}"
        elif require_strong_parse and parsed.strategy in WEAK_PARSE_STRATEGIES:
            # 弱解析不构成"按格式输出"的证据；RL 中按无效处理。
            scoring_meta["invalid_reason"] = f"weak_parse:{parsed.strategy}"
        scoring_metas.append(scoring_meta)

    return parsed_list, scoring_metas


# ---------------------------------------------------------------------------
# 结果结构
# ---------------------------------------------------------------------------
@dataclass
class RolloutGroup:
    """同一 prompt 采样出的 G 条序列及其组内优势。"""

    prompt_index: int
    meta: dict
    texts: list[str] = field(default_factory=list)
    parsed: list[ParsedSequence] = field(default_factory=list)
    scores: list[RewardBreakdown] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)

    @property
    def parse_failures(self) -> int:
        return sum(1 for p in self.parsed if not p.ok)

    def mean_reward(self) -> float:
        vals = [s.total for s in self.scores if s.invalid_reason is None]
        return sum(vals) / len(vals) if vals else 0.0

    def reward_std(self) -> float:
        """组内奖励标准差。

        接近 0 表示该组没有可学信号，优势将全为 0、这一步梯度为零。
        """
        vals = [s.total for s in self.scores if s.invalid_reason is None]
        if len(vals) < 2:
            return 0.0
        mean = sum(vals) / len(vals)
        return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5

    def dimension_stds(self) -> dict[str, float]:
        """组内每个奖励维度各自的标准差；有效样本不足 2 条时该维记 0。

        总分有方差不代表每一维都在贡献；逐维 std 用于发现已退化的维度。
        """
        by_dim: dict[str, list[float]] = {}
        for s in self.scores:
            if s.invalid_reason is not None:
                continue
            for name, score in s.dimensions.items():
                by_dim.setdefault(name, []).append(score.value)

        out: dict[str, float] = {}
        for name, vals in by_dim.items():
            if len(vals) < 2:
                out[name] = 0.0
                continue
            mean = sum(vals) / len(vals)
            out[name] = round(
                (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5, 4
            )
        return out


def aggregate_dimension_stds(groups: Sequence[RolloutGroup]) -> dict[str, float]:
    """跨组平均逐维 std，只统计有效样本 >= 2 的组。

    `dimension_stds` 对样本不足的组返回 0.0（表示无法测量，而非确实为 0），
    这些组整体排除，避免拉低平均。
    """
    by_dim: dict[str, list[float]] = {}
    for group in groups:
        per_group: dict[str, list[float]] = {}
        for score in group.scores:
            if score.invalid_reason is not None:
                continue
            for name, dim in score.dimensions.items():
                per_group.setdefault(name, []).append(dim.value)
        for name, vals in per_group.items():
            if len(vals) < 2:
                continue
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            by_dim.setdefault(name, []).append(std)

    return {
        name: round(sum(vals) / len(vals), 4)
        for name, vals in sorted(by_dim.items())
        if vals
    }


@dataclass
class RolloutBatch:
    """一个 training step 的全部 rollout 结果，顺序与训练循环一致。"""

    groups: list[RolloutGroup]

    #: 扁平序列，顺序为 prompt-major：必须先放 prompt 0 的 G 条。
    #: 该顺序须与 `grpo.py` 中张量拼接顺序一致，否则优势错位。
    texts: list[str] = field(default_factory=list)
    metas: list[dict] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)

    parse_report: dict = field(default_factory=dict)
    reward_report: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def usable_ratio(self) -> float:
        """解析成功且未被标记为无效的比例。

        低于 0.5 表示这一步多数样本没有奖励。解析失败与解析成功但非法对应
        不同的排查方向。
        """
        total = len(self.texts)
        if not total:
            return 0.0
        ok = sum(
            1 for g in self.groups for s in g.scores if s.invalid_reason is None
        )
        return round(ok / total, 4)

    def summary(self) -> dict:
        return {
            "groups": len(self.groups),
            "samples": len(self.texts),
            "parse": self.parse_report,
            "reward": self.reward_report,
        }


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------
def assemble_rollouts(
    texts: Sequence[str],
    metas: Sequence[dict],
    *,
    group_size: int,
    scorer: CompositeScorer,
    advantage_mode: str = "std",
    require_strong_parse: bool = True,
) -> RolloutBatch:
    """执行解析 → 打分 → 分组 → 优势计算，返回 `RolloutBatch`。

    `texts` 长度须为 `len(metas) * group_size`，顺序为 prompt-major；
    优势逐组计算，`scorer` 由调用方注入以便测试替换。
    """
    if group_size < 2:
        raise ValueError(
            f"group_size 必须 >= 2，收到 {group_size}。"
            f"GRPO 的优势来自组内相对比较，G=1 时组内没有可比对象，"
            f"优势恒为 0，训练不会有任何梯度。"
        )
    if len(metas) * group_size != len(texts):
        raise ValueError(
            f"文本条数 ({len(texts)}) != prompt 数 ({len(metas)}) × group_size ({group_size})。"
            f"请检查生成时的采样设置是否与 num_generations 一致。"
        )

    parsed_all, scoring_metas = parse_completions(
        texts, expand_metas(metas, group_size),
        require_strong_parse=require_strong_parse,
    )
    # 传解析后的序列而非原始文本：原始文本含标记字符，会使 `validate` 判定全部非法。
    # 解析失败的样本其 meta 已带 `invalid_reason`，`score` 会优先使用。
    scores_all = scorer.score_group([p.sequence for p in parsed_all], scoring_metas)

    groups: list[RolloutGroup] = []
    flat_advantages: list[float] = []
    for gi, meta in enumerate(metas):
        lo, hi = gi * group_size, (gi + 1) * group_size
        group = RolloutGroup(
            prompt_index=gi,
            meta=meta,
            texts=list(texts[lo:hi]),
            parsed=parsed_all[lo:hi],
            scores=scores_all[lo:hi],
        )
        # 逐组计算优势：GRPO 的优势定义是组内相对比较，整批计算会混合不同 prompt。
        group.advantages = scorer.group_advantages(group.scores, mode=advantage_mode)
        flat_advantages.extend(group.advantages)
        groups.append(group)

    batch = RolloutBatch(
        groups=groups,
        texts=list(texts),
        metas=[dict(m) for m in expand_metas(metas, group_size)],
        advantages=flat_advantages,
    )

    valid_rewards = [s.total for s in scores_all if s.invalid_reason is None]
    batch.parse_report = parse_stats(parsed_all)

    # `parse_stats.fail_rate` 只统计硬失败；RL 口径还需计入弱解析。
    # 这里补 `raw_fail_rate`（硬失败）与 `weak_parse` 单独计数，避免日志口径矛盾。
    if batch.parse_report:
        total = len(parsed_all)
        hard_fail = sum(1 for p in parsed_all if not p.ok)
        n_weak = (
            sum(1 for p in parsed_all if p.ok and p.strategy in WEAK_PARSE_STRATEGIES)
            if require_strong_parse
            else 0
        )
        batch.parse_report["raw_fail_rate"] = round(hard_fail / total, 4)
        batch.parse_report["weak_parse"] = n_weak
        batch.parse_report["fail_rate"] = round((hard_fail + n_weak) / total, 4)
    batch.reward_report = {
        "mean": round(sum(valid_rewards) / len(valid_rewards), 4) if valid_rewards else 0.0,
        "invalid": sum(1 for s in scores_all if s.invalid_reason is not None),
        # 组内奖励标准差接近 0 的组数：这些组没有梯度。
        "degenerate_groups": sum(1 for g in groups if g.reward_std() < 1e-6),
    }
    return batch


def expand_metas(metas: Sequence[dict], group_size: int) -> list[dict]:
    """把每个 prompt 的 meta 在组内重复，得到与 texts 对齐的扁平列表。

    使用逐组展开而非 `metas * group_size`，后者会产生错误的非 prompt-major 顺序。
    """
    out: list[dict] = []
    for m in metas:
        out.extend([m] * group_size)
    return out
