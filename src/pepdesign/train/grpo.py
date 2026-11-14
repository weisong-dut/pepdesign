"""阶段二：GRPO 训练循环，自写，不依赖 trl 的 GRPOTrainer。

单个 step：采样 G 条 → 解析 → 多维打分 → 组内优势 → 重要性比 →
裁剪损失 + KL → 反传。优势计算复用 `rewards/composite.py:group_advantages`。

关键不变量：`old_logps` 必须在优化器更新前、`no_grad` 下计算；参考模型是
SFT 后的策略而非基座；`dr_grpo` 的分母为 `样本数 × max_completion_length`。
优势口径见 `train/advantages.py`。
"""

from __future__ import annotations

import copy
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from pepdesign.models.loader import (
    build_tiny_model,
    chat_tokenize,
    enable_gradient_checkpointing,
    generation_stop_ids,
    load_tokenizer,
    probe_hardware,
    resolve_plan,
    tiny_lora_config,
)
from pepdesign.models.lora import inject_lora, load_adapter, save_adapter
from pepdesign.train.advantages import (
    advantage_mode,
    clip_bounds,
    loss_denominator,
    masked_mean,
)
from pepdesign.train.checkpoint import (
    CheckpointManager,
    on_step_end,
    resolve_resume_path,
    training_fingerprint,
)
from pepdesign.train.dataset import (
    build_scheduler_spec,
    estimate_grpo_num_training_steps,
    load_grpo_prompts,
)
from pepdesign.train.masking import resolve_ellipsis, truncate_prompt
from pepdesign.train.rollout import (
    WEAK_PARSE_STRATEGIES,
    aggregate_dimension_stds,
    assemble_rollouts,
)
from pepdesign.utils.config import ProjectConfig, save_config
from pepdesign.utils.logger import get_logger, silence_noisy_libraries
from pepdesign.utils.seed import set_seed
from pepdesign.utils.tracking import MetricWriter

logger = get_logger(__name__)

METRIC_KEYS: tuple[str, ...] = (
    "loss", "mean_reward", "reward_std", "kl", "clip_fraction",
    "parse_fail_rate", "weak_parse_ratio", "degenerate_groups",
    "usable_ratio", "grad_norm",
)


# ---------------------------------------------------------------------------
# 单步结果
# ---------------------------------------------------------------------------
@dataclass
class StepMetrics:
    """一步训练的观测指标。

    GRPO 的 loss 是替代目标，下降不代表模型变好；`mean_reward`、`reward_std`、
    `degenerate_groups`、`clip_fraction`、`kl` 与 `parse_fail_rate` 反映实际训练状态。
    """

    step: int = 0
    loss: float = 0.0
    mean_reward: float = 0.0
    reward_std: float = 0.0
    kl: float = 0.0
    clip_fraction: float = 0.0
    parse_fail_rate: float = 0.0
    #: 启发式兜底解析的占比（已计入 `parse_fail_rate`）。两类解析失败的排查方向不同：
    #: 完全无法解析指向 SFT 格式学习不足，弱解析偏高则说明模型在输出"像序列的英文词"。
    weak_parse_ratio: float = 0.0
    degenerate_groups: int = 0
    usable_ratio: float = 0.0
    grad_norm: float = 0.0
    seconds: float = 0.0
    example_sequences: list[str] = field(default_factory=list)
    #: 逐维跨组平均组内标准差 `{维度名: std}`。
    #: 总分方差会掩盖已退化的维度；逐维 std 用于发现零贡献维度。
    dimension_stds: dict[str, float] = field(default_factory=dict)

    def as_log_line(self) -> str:
        return (
            f"step {self.step:4d} | loss {self.loss:+.4f} | reward {self.mean_reward:.4f}"
            f" (std {self.reward_std:.4f}) | kl {self.kl:.4f}"
            f" | clip {self.clip_fraction:.3f} | 解析失败 {self.parse_fail_rate:.1%}"
            f" (弱解析 {self.weak_parse_ratio:.1%})"
            f" | 退化组 {self.degenerate_groups} | grad {self.grad_norm:.3f}"
            f" | {self.seconds:.1f}s"
        )

    def dimension_log_line(self) -> str:
        """逐维组内 std 的单行摘要；无维度数据时返回空串。"""
        if not self.dimension_stds:
            return ""
        parts = [
            f"{name}={std:.4f}"
            for name, std in sorted(self.dimension_stds.items(), key=lambda kv: kv[1])
        ]
        # 按 std 升序排——最需要关注的零方差维度排在最前面
        return "           逐维组内 std（升序，越靠前越该查）: " + "  ".join(parts)


# ---------------------------------------------------------------------------
# 一步之内的指标聚合（纯 Python，可无 torch 测试）
# ---------------------------------------------------------------------------
def aggregate_step_metrics(
    parts: list[tuple[StepMetrics, int, int]], *, step: int
) -> StepMetrics:
    """把一步中各 micro-batch 的指标聚合成一条。

    `parts` 为 `(指标, 序列条数, 组数)`。`mean_reward` / `kl` / 解析类指标按序列
    条数加权，`reward_std` 与逐维 std 按组数加权，`degenerate_groups` 求和。
    """
    if not parts:
        raise ValueError("aggregate_step_metrics 收到空的 parts")
    total_n = sum(n for _, n, _ in parts)
    total_groups = sum(g for _, _, g in parts)
    if total_n <= 0:
        # 没有产出序列（如生成失败）：保留首个 part 的形态避免除零，计数类字段照常求和。
        first = parts[0][0]
        return StepMetrics(
            step=step,
            mean_reward=first.mean_reward,
            degenerate_groups=sum(m.degenerate_groups for m, _, _ in parts),
            example_sequences=list(first.example_sequences),
        )

    def wmean(attr: str) -> float:
        return sum(getattr(m, attr) * n for m, n, _ in parts) / total_n

    # 逐维 std 按组数加权平均，与 reward_std 同一口径（都是"平均组内离散度"）。
    # 某个 micro-batch 里没出现过的维度不参与分母——否则它会把该维拉低。
    dim_names = {k for m, _, _ in parts for k in m.dimension_stds}
    dim_stds: dict[str, float] = {}
    for name in sorted(dim_names):
        num = sum(
            m.dimension_stds.get(name, 0.0) * g
            for m, _, g in parts
            if name in m.dimension_stds
        )
        den = sum(g for m, _, g in parts if name in m.dimension_stds)
        dim_stds[name] = round(num / den, 4) if den > 0 else 0.0

    return StepMetrics(
        step=step,
        mean_reward=wmean("mean_reward"),
        # 按组数加权（不是条数），理由见上面的文档。
        reward_std=(
            sum(m.reward_std * g for m, _, g in parts) / total_groups
            if total_groups > 0
            else 0.0
        ),
        kl=wmean("kl"),
        clip_fraction=wmean("clip_fraction"),
        parse_fail_rate=wmean("parse_fail_rate"),
        weak_parse_ratio=wmean("weak_parse_ratio"),
        degenerate_groups=sum(m.degenerate_groups for m, _, _ in parts),
        usable_ratio=wmean("usable_ratio"),
        dimension_stds=dim_stds,
        # 样例只留第一个 micro-batch 的：它们是给人看的样本，
        # 混在一起反而看不出"这条属于哪一批"。
        example_sequences=list(parts[0][0].example_sequences),
    )


# ---------------------------------------------------------------------------
# 模型准备
# ---------------------------------------------------------------------------
def build_reference_model(model):
    """克隆一份冻结的当前策略作为 KL 参考模型。

    参考模型必须是 SFT 后的策略，不能是基座，否则 KL 会把策略往预训练分布拉回。
    返回的模型关闭梯度与 `use_cache`。
    """
    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    # 参考模型只做前向：必须显式关闭 use_cache，否则会为 [B, L] 分配一份 KV cache。
    if hasattr(ref, "config"):
        ref.config.use_cache = False
    return ref


def _device(model) -> str:
    return str(next(model.parameters()).device)


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
def ellipsis_token_id(tokenizer) -> int:
    """返回截断占位符替换用的真实 token id。

    按 unk → pad → eos 依次兜底，三者皆无时报错。
    """
    for attr in ("unk_token_id", "pad_token_id", "eos_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            return int(tid)
    raise ValueError(
        "tokenizer 的 unk/pad/eos token id 全为 None，无法给截断占位符选一个真实 id。"
        "检查 tokenizer 是否加载正确。"
    )


def _truncate_prompt_ids(
    tokenizer, messages: list[dict], full_ids: list[int], max_prompt_length: int
) -> tuple[list[int], bool]:
    """把一条 prompt 压到 `max_prompt_length` 内，返回 `(ids, 是否截断)`。

    `keep_prefix` 取单独渲染 system 消息的 token 数；若前缀断言不成立，
    退化为 `keep_prefix=0` 并告警。
    """
    keep = 0
    if messages and messages[0].get("role") == "system":
        sys_ids = chat_tokenize(tokenizer, [messages[0]])
        if full_ids[: len(sys_ids)] == sys_ids:
            keep = len(sys_ids)
        else:
            logger.warning(
                "chat template 渲染出的前缀与单独渲染 system 消息不一致，"
                "无法在截断时保证 system prompt 完整；本条按 keep_prefix=0 处理。"
                "（换模板后要重新确认这条路径。）"
            )
    kept = truncate_prompt(full_ids, max_prompt_length, keep_prefix=keep)
    return resolve_ellipsis(kept, ellipsis_token_id(tokenizer)), len(kept) != len(full_ids)


def sample_completions(
    model, tokenizer, prompts: list[dict], cfg: ProjectConfig
) -> tuple[list[str], list[list[int]], int, list[int]]:
    """对每个 prompt 采样 G 条，返回 `(生成文本, 完整 id, gen_start, prompt_lens)`。

    顺序为 prompt-major。`gen_start` 是左 padding 后的输入宽度，也是 completion 的
    起始列；`prompt_lens` 是截断后每条 prompt 的真实长度，用于计算左 padding 宽度。
    """
    import torch

    from pepdesign.train.masking import pad_left

    # ---- 逐条编码 → 截断 → 再 padding ----
    # 顺序不能换：`width` 即 `gen_start`，`prompt_lens` 必须来自截断后的 ids；
    # 两者不一致会导致掩码整体错位。
    raw_ids: list[list[int]] = []
    n_truncated = 0
    for p in prompts:
        ids = chat_tokenize(tokenizer, p["prompt"], add_generation_prompt=True)
        ids, was_truncated = _truncate_prompt_ids(
            tokenizer, p["prompt"], ids, cfg.grpo.max_prompt_length
        )
        raw_ids.append(ids)
        n_truncated += int(was_truncated)

    # 截断率需可见，否则 max_prompt_length 可能长期是惰性配置。
    if n_truncated:
        logger.info(
            "本批 %d/%d 条 prompt 超过 max_prompt_length=%d，已从中间截断"
            "（保留 system 前缀与结尾约束）。",
            n_truncated, len(prompts), cfg.grpo.max_prompt_length,
        )

    batched = pad_left(raw_ids, tokenizer.pad_token_id)
    gen_start = int(batched["width"])
    # 每条 prompt 的真实长度（不含左侧 padding），是掩码的输入之一。
    prompt_lens = [len(r) for r in raw_ids]
    enc = {
        "input_ids": torch.tensor(batched["input_ids"], dtype=torch.long, device=_device(model)),
        "attention_mask": torch.tensor(
            batched["attention_mask"], dtype=torch.long, device=_device(model)
        ),
    }

    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=cfg.grpo.max_completion_length,
            do_sample=True,
            temperature=cfg.grpo.temperature,
            top_p=cfg.grpo.top_p,
            top_k=cfg.grpo.top_k if cfg.grpo.top_k > 0 else 0,
            num_return_sequences=cfg.grpo.num_generations,
            pad_token_id=tokenizer.pad_token_id,
            # 必须显式传 eos_token_id：漏掉会导致回合结束后继续生成且不报错。
            eos_token_id=sorted(generation_stop_ids(tokenizer)),
            # 生成到 "### 结束" 强制停止：SFT 后模型对 eos 概率不够高，温度采样时会继续
            # 生成第二组输出；`stop_strings` 需要同时传 tokenizer。
            stop_strings=["### 结束"],
            tokenizer=tokenizer,
        )
    if was_training:
        model.train()

    gen_only = [row[gen_start:].tolist() for row in out]
    texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
    return texts, [row.tolist() for row in out], gen_start, prompt_lens


# ---------------------------------------------------------------------------
# 对数概率与损失
# ---------------------------------------------------------------------------
#: 计算对数概率时一次处理的 token 数。这是显存参数而非性能参数，
#: 调大只会提高显存峰值。
LOGPROB_CHUNK: int = 32


@contextmanager
def _dropout_disabled(model):
    """临时把模型内所有 Dropout 的 p 置 0，退出时恢复。

    重要性比要求 π_old 与采样时策略一致；同时保留 train 模式以维持
    gradient checkpointing 生效。
    """
    import torch

    saved = [(m, m.p) for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    for module, _ in saved:
        module.p = 0.0
    try:
        yield
    finally:
        for module, p in saved:
            module.p = p


def token_logprobs(model, input_ids, attention_mask):
    """计算逐 token 对数概率，返回 `[B, L-1]`，对齐到"预测下一个 token"。

    `logits[:, :-1]` 对应 `input_ids[:, 1:]`，调用方传入的掩码需做相同切片。
    分块计算并使用 `gather + logsumexp`，避免整块 fp32 张量驻留计算图。
    """
    import torch

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]

    n_tokens = logits.shape[1]
    pieces: list = []
    for start in range(0, n_tokens, LOGPROB_CHUNK):
        # 切片是视图，不占额外显存。
        sl = logits[:, start : start + LOGPROB_CHUNK, :]
        tg = targets[:, start : start + LOGPROB_CHUNK].unsqueeze(-1)
        # 目标 token 对应的 logit。
        picked = sl.gather(-1, tg).squeeze(-1)
        # 配分函数的对数：直接在视图上计算，不要 .float()，否则会新建张量并驻留计算图。
        lse = torch.logsumexp(sl, dim=-1)
        # 相减时才升 fp32：两个都是 [B, c] 的小张量。
        pieces.append(picked.float() - lse.float())

    return torch.cat(pieces, dim=1)


def grpo_loss(
    logps,
    old_logps,
    ref_logps,
    advantages,
    completion_mask,
    *,
    loss_type: str,
    epsilon: float,
    clip_higher: float,
    beta: float,
    max_completion_length: int | None = None,
):
    """带裁剪的替代损失与 KL 惩罚，返回 `(loss, 指标)`。

    逐 token：`L_t = -min(r_t·A, clip(r_t)·A) + β·k3_t`，按 `loss_type` 归一化。
    分母由 `advantages.loss_denominator` 统一给出。
    """
    import torch

    adv = advantages.detach()
    lo, hi = clip_bounds(loss_type, epsilon, clip_higher)

    ratio = torch.exp(logps - old_logps)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, lo, hi) * adv
    # min 是悲观下界：优势为正时限制涨幅，为负时限制跌幅。
    surrogate = torch.min(unclipped, clipped)

    # k3 估计子：exp(d) − d − 1，d = logπ_ref − logπ_θ；恒非负、恒有梯度。
    d = (ref_logps - logps).clamp(-30.0, 30.0)
    per_token_kl = torch.exp(d) - d - 1.0

    per_token = -surrogate + beta * per_token_kl

    mask = completion_mask.float()
    n_valid = mask.sum().clamp(min=1.0)
    # `dr_grpo` 的分母必须是配置中的 completion 长度，而非 mask 全宽
    # （prompt+completion），否则 loss 与梯度被整体缩小。
    denom = loss_denominator(
        loss_type,
        token_mask=completion_mask.detach().cpu().int().tolist(),
        max_completion_length=(
            max_completion_length
            if max_completion_length is not None
            else completion_mask.shape[1]
        ),
    )

    if loss_type == "grpo":
        # 先序列内平均、再对样本平均：分母为样本数，因此每条序列先除以自身有效 token 数。
        per_seq = (per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        loss = per_seq.sum() / denom
    else:
        loss = (per_token * mask).sum() / denom

    with torch.no_grad():
        # clip_fraction 口径：重要性比落在 [lo, hi] 之外的 token 占比，
        # 与 trl 日志口径一致。
        outside = ((ratio < lo) | (ratio > hi)).float()
        metrics = {
            "kl": float((per_token_kl * mask).sum() / n_valid),
            "clip_fraction": float((outside * mask).sum() / n_valid),
        }
    return loss, metrics


# ---------------------------------------------------------------------------
# 一步完整流程
# ---------------------------------------------------------------------------
def run_step(model, ref_model, tokenizer, prompts, cfg: ProjectConfig, scorer):
    """采样 → 打分 → 优势 → 前向 → 损失，返回 `(loss, RolloutBatch, StepMetrics)`。

    包含反传之外的完整计算图；反传由调用方执行，以便统一处理梯度累积与日志。
    """
    import torch

    meta_mode = advantage_mode(
        cfg.grpo.advantage_normalization, cfg.grpo.scale_rewards
    )

    device = _device(model)
    texts, full_ids, gen_start, prompt_lens = sample_completions(model, tokenizer, prompts, cfg)
    batch = assemble_rollouts(
        texts,
        [p["meta"] for p in prompts],
        group_size=cfg.grpo.num_generations,
        scorer=scorer,
        advantage_mode=meta_mode,
    )

    # 掩码的构造整个放进 masking.build_rollout_masks（纯 list 运算、有测试）。
    # 那里处理的是这个函数最容易错的部分：左 padding 的宽度、
    # 停止 token 该不该算、填充怎么排除——见那个函数的说明。
    from pepdesign.train.masking import build_rollout_masks

    att_rows, mask_rows = build_rollout_masks(
        full_ids,
        gen_start=gen_start,
        prompt_lens=prompt_lens,
        group_size=cfg.grpo.num_generations,
        # 停止集必须含模板的回合终止符，否则掩码会一路跑到填充边缘。
        # 这里和上面 generate 用的是同一个函数，两处不会漂移。
        stop_ids=generation_stop_ids(tokenizer),
    )

    input_ids = torch.tensor(full_ids, dtype=torch.long, device=device)
    attention = torch.tensor(att_rows, dtype=torch.long, device=device)
    # [1:] 是与 token_logprobs 输出（logits[:, :-1]）逐位对齐的位移，
    # 语义与 masking.shift_mask_for_labels 完全一致。
    completion_mask = torch.tensor(mask_rows, dtype=torch.long, device=device)[:, 1:]

    # 三次前向都在"dropout 关闭 + 仍是 train 模式"下算：
    # 前者保证 π_old / π_ref / π_θ 是同一个确定性策略（第 1 步 ratio=1、
    # KL=0），后者保证 gradient checkpointing 仍然生效，不会多占激活显存。
    # 见 `_dropout_disabled` 的说明。
    with _dropout_disabled(model):
        with torch.no_grad():
            old_logps = token_logprobs(model, input_ids, attention)
            ref_logps = token_logprobs(ref_model, input_ids, attention)

        logps = token_logprobs(model, input_ids, attention)

    advantages = torch.tensor(
        batch.advantages, dtype=torch.float32, device=device
    ).unsqueeze(1)

    loss, loss_metrics = grpo_loss(
        logps, old_logps, ref_logps, advantages, completion_mask,
        loss_type=cfg.grpo.loss_type,
        epsilon=cfg.grpo.epsilon,
        clip_higher=cfg.grpo.clip_higher,
        beta=cfg.grpo.beta,
        # dr_grpo 的分母必须是配置里的 completion 长度，不是
        # `completion_mask.shape[1]`（那是 prompt+completion 的全宽）。
        # 传错会让 loss/梯度整体缩小约 2.6 倍，而且随 batch 最长 prompt 变化。
        max_completion_length=cfg.grpo.max_completion_length,
    )

    group_stds = [g.reward_std() for g in batch.groups]
    strategy_dist = batch.parse_report.get("strategy_distribution", {}) or {}
    n_total = int(batch.parse_report.get("total", 0)) or len(batch.texts)
    weak = sum(n for k, n in strategy_dist.items() if k in WEAK_PARSE_STRATEGIES)

    # 逐维组内 std 跨组求平均——只统计有效样本 >= 2 的组。
    # 见 `aggregate_dimension_stds` 的说明：不足 2 条的组返回 0.0 是
    # "量不出来"，不能当成"量出来是 0"拉低平均。
    dim_stds = aggregate_dimension_stds(batch.groups)

    metrics = StepMetrics(
        loss=float(loss.detach()),
        mean_reward=batch.reward_report.get("mean", 0.0),
        reward_std=round(sum(group_stds) / len(group_stds), 4) if group_stds else 0.0,
        kl=loss_metrics["kl"],
        clip_fraction=loss_metrics["clip_fraction"],
        parse_fail_rate=float(batch.parse_report.get("fail_rate", 0.0)),
        weak_parse_ratio=round(weak / n_total, 4) if n_total else 0.0,
        degenerate_groups=int(batch.reward_report.get("degenerate_groups", 0)),
        usable_ratio=batch.usable_ratio,
        dimension_stds=dim_stds,
        example_sequences=list(batch.texts[:1]),
    )
    return loss, batch, metrics


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------
def peak_gpu_memory_gb() -> float:
    """本进程至今的 CUDA 显存峰值（GiB）。CPU 上恒为 `0.0`。

    独立函数而非 METRIC_KEYS 字段
    ------------------------------------------------------
    `METRIC_KEYS` 是一份固定契约，说的是"这一步发生了什么学习信号"
    （见 `StepMetrics` 的类文档：reward_std 有没有可学信号、kl 有没有崩、
    解析失败率还在不在）。显存峰值不是学习信号，它是工程约束，
    而且它的聚合口径是 max 而不是加权平均——塞进
    `aggregate_step_metrics` 那套按序列条数加权的逻辑里，
    算出来的会是一个"平均峰值"这种没有意义的数。

    必须输出的原因
    -------------------
    `token_logprobs` 的分块修复（见那里的文档）解决的是前向的峰值，
    而那个 bug 两次都是在前向崩的，即反向的峰值从来没有被观测过。
    反向里每个分块都会分配一份整词表大小的梯度缓冲（`[B, L-1, V]`，
    本项目实测 5.54 GB），所以"改完之后够不够"在这行日志出现之前
    是个推测，不是结论。

    `max_memory_allocated` 是自进程启动以来的峰值、单调不减，所以
    每一步读到的都是"至今为止最坏的一刻"，不需要 reset 也不会漏掉
    生成阶段那部分——而生成阶段的 KV cache 同样是真实占用。
    读的是 `allocated` 而不是 `reserved`：前者才是"张量真的占了多少"，
    后者包含分配器的缓存池，会把结论说大。
    """
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1 << 30)


def run_grpo(cfg: ProjectConfig, *, model=None, ref_model=None, tokenizer=None,
             prompts=None, scorer=None, max_steps: int | None = None) -> Path:
    """跑完阶段二，返回 adapter 保存目录。

    所有参数都可注入：冒烟测试传入两层小模型和少量 prompt，
    真实训练全用默认值从配置和磁盘加载。
    """
    import torch

    silence_noisy_libraries()
    set_seed(cfg.grpo.seed)
    logger.info("=== 阶段二：GRPO（四维奖励） ===")

    if scorer is None:
        from pepdesign.rewards.composite import CompositeScorer

        scorer = CompositeScorer(cfg.reward)

    if tokenizer is None:
        tokenizer = load_tokenizer(cfg.grpo.model_name_or_path)

    if model is None:
        from pepdesign.models.loader import load_base_model

        hw = probe_hardware()
        plan = resolve_plan(
            cuda_available=hw["cuda_available"],
            bf16_supported=hw["bf16_supported"],
            mps_available=hw["mps_available"],
            want_bf16=cfg.grpo.bf16,
            want_fp16=cfg.grpo.fp16,
            quant=cfg.quant,
        )
        # 返回 (model, plan)，第二个已经打过日志了，这里丢掉。
        model, _plan = load_base_model(
            cfg.grpo.model_name_or_path, quant=cfg.quant,
            want_bf16=cfg.grpo.bf16, want_fp16=cfg.grpo.fp16, plan=plan,
        )
        if cfg.grpo.sft_adapter_path:
            # 必须先 `inject_lora` 再 `load_adapter`，顺序不能反。
            #
            # 手写后端（本项目默认）的 `load_adapter` 只负责把权重灌进
            # 已经存在的 `LoRALinear` 里；它靠 `_is_manual_model(model)`
            # 判断走哪条路，而刚 `load_base_model` 出来的基座身上没有任何
            # `LoRALinear`——于是会掉进 peft 分支，去读一个手写格式的 adapter
            # （`format="pepdesign.manual-lora.v1"`、key 名也完全不同），
            # 结果是一句和真正原因无关的加载错误。
            # 这条契约 `load_adapter` 自己的 docstring 就写着；
            # `serve/engine.py` 遵守了，这里以前漏了。
            #
            # 冒烟测不出来：冒烟路径传的是已经注入好的 model=，
            # 从不经过"从磁盘加载已训好的 adapter"这一段。
            model = inject_lora(model, cfg.lora)
            # is_trainable=True 不能省：peft 的 from_pretrained 默认走推理语义，
            # 会把 adapter 冻住（细节见 models/lora.py:load_adapter）。
            # 漏掉的后果不是报错，而是下面收集可训练参数时拿到空列表。
            model = load_adapter(
                model, cfg.grpo.sft_adapter_path, is_trainable=True
            )
        else:
            logger.warning(
                "没有指定 sft_adapter_path，GRPO 将从基座开始。"
                "这违反两阶段设计：参考模型会是没做过 SFT 的模型，"
                "KL 约束会把策略往预训练分布上拽。除非在做消融实验，"
                "否则应当先跑 scripts/train_sft.py。"
            )

    if prompts is None:
        prompts = load_grpo_prompts(cfg.grpo.dataset_path)

    if not prompts:
        raise RuntimeError(
            f"GRPO prompt 集为空（{cfg.grpo.dataset_path}）。"
            f"请先跑数据构建：data/pipeline.py 的第 5 步会产出 grpo_prompts.jsonl。"
        )

    device = _device(model)
    model.train()
    model = model.to(device)

    # 只训练 LoRA 参数。基座在 inject_lora 里已经冻住了，
    # 这里再显式过滤一次：传入的 model 可能是调用方构造的（冒烟测试），
    # 不能假定它的 requires_grad 一定正确。
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("没有任何可训练参数，检查 LoRA 是否注入成功")
    optimizer = torch.optim.AdamW(
        params, lr=cfg.grpo.learning_rate, weight_decay=cfg.grpo.weight_decay
    )

    # 步数口径见 dataset.estimate_grpo_num_training_steps：
    # 分母是"每步吃多少条 prompt"（batch × grad_accum），num_generations 不在分母里。
    # 以前这里只按 batch 算，而外层循环每步又乘了 grad_accum，
    # 于是实际遍历 prompt 集的次数恰好等于 grad_accum（grpo_lora.yaml 里是 4 遍），
    # 而配置里的 num_train_epochs 从来没被读过。
    if max_steps is not None and max_steps > 0:
        # 冒烟测试 / 命令行显式指定的步数，优先级高于配置。
        steps = int(max_steps)
    else:
        steps = estimate_grpo_num_training_steps(cfg.grpo, len(prompts))

    grad_accum = max(1, cfg.grpo.gradient_accumulation_steps)
    batch_size = max(1, cfg.grpo.per_device_train_batch_size)
    # 这一步只是为了让"实际遍历了几遍语料"在日志里可见：上面那个 bug 的
    # 隐蔽之处就在于多跑了 3 遍语料，而所有日志看起来都正常。
    passes = steps * grad_accum * batch_size / max(1, len(prompts))

    # 学习率调度。自己写的循环里没有 TrainingArguments，调度器必须显式构造——
    # 不构造的话 GRPOConfig 的 lr_scheduler_type / warmup_ratio 就是两个
    # "配了也不生效"的字段（详见 dataset.build_scheduler_spec 的说明）。
    from transformers import get_scheduler

    spec = build_scheduler_spec(cfg.grpo, num_training_steps=steps)
    scheduler = get_scheduler(
        name=spec["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=spec["num_warmup_steps"],
        num_training_steps=spec["num_training_steps"],
    )

    # ---- 指标外送 / checkpoint / 断点续训 ----
    writer = MetricWriter.from_config(
        cfg.grpo.report_to, output_dir=cfg.grpo.output_dir, prefix="grpo"
    )
    ckpt = CheckpointManager(
        output_dir=cfg.grpo.output_dir,
        save_total_limit=cfg.grpo.save_total_limit,
        lora_cfg=cfg.lora,
        phase="grpo",
    )
    fingerprint = training_fingerprint(cfg.grpo, phase="grpo", num_steps=steps)
    resume_path = resolve_resume_path(cfg.grpo.resume_from_checkpoint, cfg.grpo.output_dir)
    start_step = 0
    restored_history: list[StepMetrics] = []
    if resume_path is not None:
        state = ckpt.load_into(
            resume_path,
            model=model, optimizer=optimizer, scheduler=scheduler,
            expected_fingerprint=fingerprint,
            learning_rate=cfg.grpo.learning_rate,
        )
        start_step = int(state["step"])
        if state.get("history"):
            # 落盘时走的是 dataclasses.asdict，读回来就是构造参数。
            restored_history = [StepMetrics(**row) for row in state["history"]]
        logger.info("从 step %d 接着跑（共 %d 步）", start_step, steps)

    # 校准集预热：在正式 RL 之前用校准对把策略的初始分布拉到
    # 合理区域，避免 RL 早期在巨大的序列空间里盲目探索。
    # 详见 train/calibration.py。
    #
    # 续训时必须跳过：预热的作用是把起点挪到合理区域，
    # 而续训的起点已经是一个训过的策略了。再跑一遍等于在继续 RL 之前
    # 先把策略往校准集的分布上拽一次——那会悄悄改掉"接着跑"的语义，
    # 而且日志上看不出任何异常。
    if cfg.grpo.calibration_enabled and start_step == 0:
        from pepdesign.train.calibration import run_calibration_warmup

        run_calibration_warmup(model, tokenizer, cfg, optimizer)
    elif cfg.grpo.calibration_enabled and start_step > 0:
        logger.info("续训（start_step=%d），跳过校准预热。", start_step)

    # ---- 参考模型必须在续训加载与校准预热之后克隆 ----
    # 以前这里在加载 checkpoint / 校准之前就克隆了，于是：
    #   * 续训时 π_ref 是"续训前的模型"，不是当前起点；
    #   * 校准时 π_ref 是"校准前的 SFT"，KL 从第 1 步起就把模型往
    #     校准前的分布拽——预热刚把起点挪走，KL 又把它拉回来。
    # 参考模型的定义是"RL 开始时的策略"，所以必须等模型状态稳定后再克隆。
    if ref_model is None:
        if start_step > 0:
            # 参考模型没有存进 checkpoint：续训时只能从当前（已被 GRPO
            # 更新过的）策略重新克隆。KL 锚点因此会从 θ_0 变成 θ_N，
            # 相当于把 KL 预算重置一次。首次全量跑通常不会续训；
            # 真要长时间续训，应该在 checkpoint 里额外保存 ref adapter。
            logger.warning(
                "续训（start_step=%d）：参考模型将从当前 checkpoint 重新克隆，"
                "KL 锚点会重置到 step %d 的策略。连续多次续训会让策略逐渐"
                "远离最初的 SFT 起点；如果这不是想要的，请从原始 adapter 重跑。",
                start_step, start_step,
            )
        ref_model = build_reference_model(model)

    # 梯度检查点必须在 `build_reference_model` 之后开：它会往输入
    # embedding 上挂一个 forward hook，而 deepcopy 会把挂上的 hook 一起复制。
    # 代价是校准预热这一段没有梯度检查点（500 条、batch 4、512 token，
    # 4-bit 3B 在 24G 卡上够用）；换来的是 ref 与当前起点严格一致。
    enable_gradient_checkpointing(model, enabled=cfg.grpo.gradient_checkpointing)
    # 即使 `gradient_checkpointing=False`，logprob 前向也不需要 KV cache。
    # `enable_gradient_checkpointing` 只在开启时才把 use_cache 置 False；
    # 关掉 GC 的实验如果忘了这一条，`token_logprobs` 会为 [B,L] 分配一份
    # 本可以避免的 cache（B=32、L≈665 时约 2.4 GB）。
    if hasattr(model, "config"):
        model.config.use_cache = False

    logger.info(
        "GRPO 开始：%d 个 prompt，每 prompt 采样 %d 条，共 %d 步"
        "（本次从第 %d 步开始），梯度累积 %d（≈ %.2f 遍语料），loss_type=%s，beta=%.3f",
        len(prompts), cfg.grpo.num_generations, steps, start_step + 1,
        grad_accum, passes, cfg.grpo.loss_type, cfg.grpo.beta,
    )
    if (
        cfg.grpo.temperature != 1.0
        or cfg.grpo.top_p != 1.0
        or cfg.grpo.top_k != 0
    ):
        logger.warning(
            "采样参数 temperature=%.2f / top_p=%.2f / top_k=%d 不是默认值。"
            "`token_logprobs` 用的是原始 logits，没有对温度缩放和 top-p/top-k "
            "截断做校正，所以 π_old 不再是真正的采样分布——第 1 步 ratio 就"
            "不等于 1，重要性采样有偏。要么保持 1.0/1.0/0，要么在 logprob "
            "里显式除以 temperature（top-p/top-k 的截断还需更大的改动）。",
            cfg.grpo.temperature, cfg.grpo.top_p, cfg.grpo.top_k,
        )

    save_config(cfg, Path(cfg.grpo.output_dir) / "resolved_config.yaml")
    out_dir = Path(cfg.grpo.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history: list[StepMetrics] = list(restored_history)
    if history:
        logger.info("已恢复 %d 步的历史指标，本次从 step %d 继续。", len(history), start_step + 1)

    def _save() -> None:
        ckpt.save(
            step, total_steps=steps, model=model, tokenizer=tokenizer,
            optimizer=optimizer, scheduler=scheduler,
            history=history, fingerprint=fingerprint,
        )

    # `try/finally` 只为了保证 writer 被关掉，循环体本身不受影响。
    try:
        # 从 start_step + 1 开始：续训时前面那些步已经跑过了。
        # GRPO 的数据位置不需要像 SFT 那样"跳过前 K 个 micro-batch"——
        # 它的 chunk 下标本来就是由 step 算出来的算术表达式（见下面两行），
        # 所以从第 N 步开始天然就落在正确的数据位置上。
        for step in range(start_step + 1, steps + 1):
            t0 = time.time()
            optimizer.zero_grad()

            step_loss = 0.0
            # 每个 micro-batch 的指标都留着，这一步结束时聚合，而不是让下一个覆盖上一个。
            # 元组是 (指标, 序列条数, 组数)——两个权重口径不同，见 aggregate_step_metrics。
            parts: list[tuple[StepMetrics, int, int]] = []
            for micro in range(grad_accum):
                start = ((step - 1) * grad_accum + micro) * batch_size
                chunk = [prompts[(start + i) % len(prompts)] for i in range(batch_size)]
                # 这个 with 必须包住 backward，不能只包住 run_step。
                # gradient checkpointing 的反向会重新跑一遍前向；如果这里在
                # run_step 返回后就恢复 dropout 的 p，重算时用的是 p>0，
                # 而原始前向是 p=0——torch 的 checkpoint 会报
                # `Recomputed values have different metadata`，或者（旧版 torch）
                # 静默得到不一致的梯度。真实配置正是 GC=true + dropout=0.05，
                # 冒烟配置用 GC=false/dropout=0 掩盖了这个问题。
                with _dropout_disabled(model):
                    loss, batch, m = run_step(model, ref_model, tokenizer, chunk, cfg, scorer)
                    # 和 SFT 一致：NaN/Inf 立刻停，不要让它静默写坏 adapter。
                    # GRPO 的 loss 比 SFT 更容易出现非有限值（重要性比指数、
                    # KL 的 exp、logprob 的 logsumexp 都在放大数值范围），
                    # 而一旦 backward 进去，optimizer.step() 会把 NaN 写进
                    # 每一个 LoRA 参数，后续所有 step 都是坏的。
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"第 {step} 步 GRPO loss 不是有限值（{loss.item()}）。"
                            f"常见原因：重要性比爆炸（检查 old_logps 是否为 NaN）、"
                            f"KL 项溢出、或 lr 过大。先看上一步的 kl / clip_fraction / "
                            f"parse_fail_rate 日志，再决定是调小 lr 还是回滚 checkpoint。"
                        )
                    # 除以累积步数：不除的话等效学习率会随 grad_accum 线性放大，
                    # 而"改了累积步数但忘了改 lr"是调参时最常见的意外。
                    (loss / grad_accum).backward()
                step_loss += float(loss.detach()) / grad_accum
                parts.append((m, len(batch.texts), len(batch.groups)))

            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grpo.max_grad_norm)
            # loss 有限不代表梯度有限：fp16 下溢/溢出、或个别参数在反向里
            # 产生 inf，clip_grad_norm_ 会把范数算成 inf/nan，随后
            # optimizer.step() 就把 nan 写进所有 LoRA 参数。SFT 有
            # GradScaler 兜底，GRPO 没有，所以这里必须单独查。
            if not torch.isfinite(grad_norm):
                raise RuntimeError(
                    f"第 {step} 步梯度范数不是有限值（{grad_norm}）。"
                    f"先回滚到上一个 checkpoint，再检查 lr / fp16 溢出 / "
                    f"logprob 分块里的数值稳定性。"
                )
            optimizer.step()
            scheduler.step()

            metrics = aggregate_step_metrics(parts, step=step)
            metrics.loss = step_loss
            metrics.grad_norm = float(grad_norm)
            metrics.seconds = time.time() - t0
            history.append(metrics)

            if step % max(1, cfg.grpo.logging_steps) == 0 or step == 1:
                logger.info(metrics.as_log_line())
                # 逐维 std 单独一行，只在有数据时打（冒烟/早期可能量不出来）。
                # 它回答的是"总分有方差，但到底是哪几维在动"——
                # 某一维长期为 0 说明它名义上占着权重、实际对优势零贡献。
                dim_line = metrics.dimension_log_line()
                if dim_line:
                    logger.info(dim_line)
                # 显存峰值单独一行，只在真有 CUDA 时打（CPU 冒烟上恒为 0）。
                # 它是"全量训练会不会 OOM"的唯一直接证据——理由见
                # peak_gpu_memory_gb 的文档：前向问题修复后，
                # 反向的峰值还没有人观测过。
                peak = peak_gpu_memory_gb()
                if peak > 0.0:
                    logger.info("           峰值显存（至今）: %.2f GiB", peak)

            # 与 SFT 共用同一个步末收尾函数（见 train/checkpoint.py）。
            # GRPO 没有验证集，所以 evaluate 传 None——
            # 它的监控信号是每一步的 reward / KL / 解析失败率，见 StepMetrics。
            on_step_end(
                step=step, total_steps=steps,
                eval_steps=0, save_steps=cfg.grpo.save_steps,
                writer=writer, scalars=grpo_scalars(metrics),
                evaluate=None, save=_save,
            )
    finally:
        writer.close()

    # 这里以前没传 cfg：手写后端下 `save_adapter` 会因为
    # `cfg is None` 直接抛 ValueError（见 models/lora.py:373），
    # 即默认配置下 GRPO 一个 adapter 都存不下来——中间的和最终的都不能。
    # 之所以一直没被发现，是因为冒烟配置把 save_steps 设成 0，
    # 这条路径从来没有被跑到过。修复它的是"传 cfg"，
    # 而防止它再发生的是 CheckpointManager 把 LoRAConfig 变成构造参数：
    # 保存路径上再也没有一个地方可以忘记它。
    adapter_dir = save_adapter(model, out_dir / "final", tokenizer, cfg.lora)
    _log_training_summary(history, num_generations=cfg.grpo.num_generations)
    logger.info("=== GRPO 完成，adapter 在 %s ===", adapter_dir)
    return adapter_dir


def grpo_scalars(metrics: StepMetrics) -> dict[str, float]:
    """把一步的指标摊成 `{名字: 数}`，喂给 tensorboard。

    名字直接用 `METRIC_KEYS`，不另起一套：那 10 个键已经是
    "这一步发生了什么"的完整词汇表（loss 本身没有量纲、`mean_reward` 才是目标、
    `reward_std` 说明还有没有可学信号……），日志里打的、指标里存的、
    曲线里画的应当是同一组名字。另起名字会让"日志里有、曲线里没有"
    变成一件要靠记忆去比对的事。

    逐维 std 是动态键集（取决于这一步开了哪几个奖励维度），
    所以走 `dim/` 前缀单开一组，而不是塞进 `METRIC_KEYS`——
    后者是固定契约，往里加动态键会让"这个键应该总是存在"这个假设失效。
    """
    out = {key: float(getattr(metrics, key)) for key in METRIC_KEYS}
    for name, std in metrics.dimension_stds.items():
        out[f"dim/{name}_std"] = float(std)
    return out


def _log_training_summary(history: list[StepMetrics], *, num_generations: int) -> None:
    """训练收尾时打一段诊断，把"训练是不是真的在进步"摊开说。

    只打 mean reward 的首尾对比和几个健康指标，不下"训练成功"的结论——
    判断好不好要看 eval 层的离线指标，训练日志只负责提供线索。
    """
    if not history:
        return
    first, last = history[0], history[-1]
    logger.info(
        "训练小结：reward %.4f → %.4f（%+.4f）| kl %.4f → %.4f | "
        "退化组 %d → %d | 解析失败 %.1f%% → %.1f%%",
        first.mean_reward, last.mean_reward, last.mean_reward - first.mean_reward,
        first.kl, last.kl,
        first.degenerate_groups, last.degenerate_groups,
        100 * first.parse_fail_rate, 100 * last.parse_fail_rate,
    )
    # 峰值显存放进小结，而不是只在逐步日志里打：判断"全量 375 步会不会 OOM"
    # 靠的就是这一个数。逐步日志里读到的已经是同一个值
    # （`max_memory_allocated` 单调不减），但收尾这一行保证"跑完了就一定看得到"，
    # 不需要回头翻日志——而一次 20 步的验证跑完最该留下的就是这个数。
    peak = peak_gpu_memory_gb()
    if peak > 0.0:
        logger.info(
            "峰值显存：%.2f GiB（allocated，不含分配器缓存池）——"
            "拿它对照卡的容量，就知道全量训练还剩多少余量",
            peak,
        )
    # 两种"没梯度"的原因完全不同，必须先分开，否则会把人指向错误的方向：
    #   (a) 一条都没解析成功 → 所有奖励都是 0 → 组内当然没有离散度。
    #       这时候改温度、调奖励、换 prompt 全都没用，问题是格式没学会；
    #   (b) 解析成功了但组内分数就是一样 → 才是温度/奖励饱和/prompt 难度的问题。
    # 以前这里只有 (b) 的消息，而 (a) 的现场跑出来正好长成 (b) 的样子。
    if all(m.parse_fail_rate >= 1.0 for m in history):
        logger.warning(
            "所有 rollout 都解析失败：奖励恒为 0，因此优势恒为 0、没有梯度。\n"
            "  这是格式问题，不是超参问题——先别动温度和奖励。按这个顺序查：\n"
            "  ①max_completion_length 够不够：本项目 SFT 答案长度在 62~158 token，\n"
            "    低于这个数就是被截断了，模型永远没机会输出完整答案；\n"
            "  ②策略是不是还没学会输出格式（SFT 步数不够 / adapter 没挂上）；\n"
            "  ③prompt 模板和解析器的格式约定是不是对得上。"
        )
    elif all(m.reward_std < 1e-6 for m in history):
        # 判据用 reward_std 而不是 degenerate_groups：聚合之后
        # `StepMetrics.degenerate_groups` 是跨 micro-batch 的和
        # （见 aggregate_step_metrics），grad_accum=4 时一个全退化的 step
        # 是 16，而这里要比的是"步数"，两者量纲不同，比出来恒为 False——
        # 警告会在最该响的时候不响。reward_std 是按组数加权的平均组内
        # 标准差，它全为 0 就等价于"每个组内部都没有离散度"。
        logger.warning(
            "每一步的组内奖励离散度都是 0：优势恒为 0，这个训练没有梯度。"
            "先别调学习率——按这个顺序查：①温度是不是太低了"
            "（组大小 G=%d，改小 G 或调高温度）；"
            "②奖励是不是饱和了（预测器输出都顶到 0 或 1）；"
            "③prompt 是不是太简单/太难（模型对所有采样都给同一个分）。",
            num_generations,
        )


# ---------------------------------------------------------------------------
# CPU 冒烟
# ---------------------------------------------------------------------------
def run_grpo_smoke(cfg: ProjectConfig, *, steps: int = 2) -> dict:
    """CPU 冒烟：两层随机初始化小模型 + 真奖励函数，跑几步完整 GRPO。

    验证的是结构性事实（这些在 GPU 上跑同样成立）：
      * 生成 → 解码 → 解析 → 打分 → 优势 这条链能从头走到尾；
      * torch 算出的损失与 `advantages.py` 的纯 Python 实现数值一致
        （见 `_cross_check_loss`）——这是"两份实现不会漂移"的保证；
      * 反传能把梯度送到 LoRA 参数上，优化器 step 之后参数真的变了；
      * 解析失败、组内奖励退化这些分支不会让循环崩掉。
    """
    import torch

    from pepdesign.rewards.composite import CompositeScorer

    set_seed(cfg.grpo.seed)
    tokenizer = load_tokenizer(cfg.grpo.model_name_or_path)
    model = build_tiny_model(vocab_size=len(tokenizer))
    model = inject_lora(model, tiny_lora_config())
    ref_model = build_reference_model(model)
    # 与 run_grpo 同序：参考模型是 deepcopy 出来的，梯度检查点的 embedding hook
    # 要在 clone 之后再挂，否则会被一起复制过去。
    enable_gradient_checkpointing(model, enabled=cfg.grpo.gradient_checkpointing)

    prompts = load_grpo_prompts(cfg.grpo.dataset_path, limit=max(steps * 2, 4))
    scorer = CompositeScorer(cfg.reward)

    cross_check = _cross_check_loss(cfg)

    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    losses: list[float] = []
    grad_norms: list[float] = []
    nonzero_advantages = 0
    last_metrics: StepMetrics | None = None
    for i in range(steps):
        chunk = [prompts[(i * 2 + j) % len(prompts)] for j in range(2)]
        optimizer.zero_grad()
        # 与 run_grpo 同序：dropout 必须关到 backward 结束，
        # 否则 gradient checkpointing 的重算会和原始前向对不上（见 run_grpo 的注释）。
        with _dropout_disabled(model):
            loss, batch, metrics = run_step(model, ref_model, tokenizer, chunk, cfg, scorer)
            loss.backward()
        # 梯度范数要在 step 之前取：step 之后参数变了、梯度还在，
        # 但那是"更新前的梯度"，语义上属于这一步。
        # 不用 clip_grad_norm_（它会原地改梯度）：这里只想知道范数是多少。
        grads = [
            p.grad.detach()
            for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        grad_norms.append(float(
            torch.sqrt(sum((g * g).sum() for g in grads)) if grads else 0.0
        ))
        optimizer.step()
        losses.append(float(loss.detach()))
        nonzero_advantages += sum(1 for a in batch.advantages if a != 0.0)
        last_metrics = metrics
        logger.info("冒烟步 %d: %s", i + 1, metrics.as_log_line())
        if batch.parse_report.get("total", 0) != len(batch.texts):
            raise AssertionError("解析统计的样本数与生成条数不一致")

    after = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    changed = sum(1 for a, b in zip(before, after, strict=True) if not torch.equal(a, b))

    return {
        "steps": steps,
        "losses": losses,
        "losses_finite": all(math.isfinite(x) for x in losses),
        "tensors_changed": changed,
        "trainable_tensors": len(before),
        # --- 下面三个是"这一步到底有没有学习信号"的直接证据 ---
        # `tensors_changed > 0` 不能代替它们：AdamW 是解耦权重衰减，
        # 梯度全为 0 时它照样会把参数按 (1 - lr·wd) 缩一下。
        # 所以"参数变了"只证明 optimizer 挂对了参数，
        # 不证明奖励算出了优势、优势产生了梯度。
        "grad_norms": grad_norms,
        "nonzero_advantages": nonzero_advantages,
        "parse_fail_rate": float(last_metrics.parse_fail_rate) if last_metrics else None,
        "weak_parse_ratio": float(last_metrics.weak_parse_ratio) if last_metrics else None,
        "usable_ratio": float(last_metrics.usable_ratio) if last_metrics else None,
        "loss_cross_check": cross_check,
        "note": (
            "小模型 + 随机初始化 + 真奖励函数；只证明链路通与数值一致，不证明学得好。"
            "**随机初始化的模型生成不出 `### 设计序列` 格式，解析成功率通常是 0，"
            "于是奖励全部无效、优势全为 0、梯度也全为 0——这是正确行为，不是故障。**"
            "要看「有优势时梯度会不会来」，见 test_nonzero_advantages_produce_gradients。"
        ),
    }


def _cross_check_loss(cfg: ProjectConfig) -> dict:
    """把 torch 版损失和纯 Python 版在同一组人造数值上对比。

    这是防止"两份实现漂移"的手段：`advantages.py` 里的
    `clipped_surrogate` / `kl_penalty` / `loss_denominator` 是给日志和
    无 GPU 测试用的纯函数，`grpo_loss` 里是真正反传的 torch 实现。
    公式写两遍的风险是它们慢慢分家，而这个对比把它变成一个会失败的断言。

    用固定的人造数值而不是随机数：随机数在 CI 上偶尔会撞到
    `ratio` 恰好落在裁剪边界的情形，导致本该稳定的对比变得不稳定。
    """
    import torch

    logps = torch.tensor([[-0.5, -1.2, -2.0, -0.3]])
    old_logps = torch.tensor([[-0.6, -1.0, -2.1, -0.35]])
    ref_logps = torch.tensor([[-0.55, -1.1, -1.9, -0.4]])
    advantages = torch.tensor([[1.5]])
    mask = torch.tensor([[1, 1, 1, 0]])

    loss, metrics = grpo_loss(
        logps, old_logps, ref_logps, advantages, mask,
        loss_type="dr_grpo",
        epsilon=cfg.grpo.epsilon,
        clip_higher=cfg.grpo.clip_higher,
        beta=cfg.grpo.beta,
        max_completion_length=4,
    )

    # ---- 纯 Python 版本，逐项手算 ----
    from pepdesign.train.advantages import clipped_surrogate, kl_penalty, policy_ratio

    lp = [-0.5, -1.2, -2.0, -0.3]
    olp = [-0.6, -1.0, -2.1, -0.35]
    rlp = [-0.55, -1.1, -1.9, -0.4]
    a = 1.5
    lo, hi = clip_bounds("dr_grpo", cfg.grpo.epsilon, cfg.grpo.clip_higher)

    ratios = policy_ratio(lp, olp)
    kls = kl_penalty(lp, rlp)
    per_token = [
        -clipped_surrogate(r, a, lo, hi)[1] + cfg.grpo.beta * k
        for r, k in zip(ratios, kls, strict=True)
    ]
    denom = loss_denominator("dr_grpo", token_mask=[[1, 1, 1, 0]], max_completion_length=4)
    expect = masked_mean(per_token, [1, 1, 1, 0]) * 3 / denom

    return {
        "torch": round(float(loss.detach()), 8),
        "python": round(expect, 8),
        "agree": abs(float(loss.detach()) - expect) < 1e-6,
        "kl_torch": round(metrics["kl"], 8),
        "kl_python": round(masked_mean(kls, [1, 1, 1, 0]), 8),
    }
