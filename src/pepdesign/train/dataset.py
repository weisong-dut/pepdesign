"""训练语料的读取、校验与步数/调度器口径。

本模块不 import torch，负责三件事：读取并逐条校验 `data/processed/*.jsonl`；
估算两阶段步数（决定 warmup 与 cosine 衰减点）；把 `lr_scheduler_type` /
`warmup_ratio` 换算成 `transformers.get_scheduler` 的参数。

SFT 与 GRPO 必须共用同一套步数与调度口径。训练语料已被 pipeline 清洗，
坏行直接报错并指出文件、行号与缺失键，不静默跳过。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pepdesign.utils.config import (
    GRPOConfig,
    SFTConfig,
    TrainCommonConfig,
    resolve_path,
)
from pepdesign.utils.logger import get_logger

logger = get_logger(__name__)

#: SFT 样本必需 `messages` + `task_type`；GRPO 样本必需 `prompt` + `task_type` + `meta`。
#: 键由 `prompts/templates.py` 写入，改动需同步。
SFT_REQUIRED_KEYS: tuple[str, ...] = ("messages", "task_type")
GRPO_REQUIRED_KEYS: tuple[str, ...] = ("prompt", "task_type", "meta")


# ---------------------------------------------------------------------------
# 逐行读取
# ---------------------------------------------------------------------------
def iter_jsonl_checked(
    path: str | Path, required_keys: tuple[str, ...]
) -> list[dict]:
    """读取 JSONL 并校验必备键，坏行抛错并指出行号。

    使用 `json.loads` 而非 `utils.io.read_jsonl`，以便解析失败时定位行号。
    """
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"训练语料不存在: {p}\n"
            f"请先运行数据构建：python -m pepdesign.data.pipeline 或 "
            f"scripts/00_make_sample_data.py 之后重跑 pipeline。"
        )

    out: list[dict] = []
    with open(p, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{p.name} 第 {lineno} 行不是合法 JSON: {exc}"
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(
                    f"{p.name} 第 {lineno} 行不是 JSON 对象，实际是 {type(item).__name__}"
                )
            missing = [k for k in required_keys if k not in item]
            if missing:
                raise ValueError(
                    f"{p.name} 第 {lineno} 行缺少必备字段 {missing}。"
                    f"已有的字段: {sorted(item)}。"
                    f"这通常说明语料是旧版本生成的，请重新跑一遍数据构建。"
                )
            out.append(item)
    return out


def load_sft_samples(path: str | Path, *, limit: int | None = None) -> list[dict]:
    """读取 SFT 语料并校验 `messages` 结构。

    要求非空列表、每条含 role/content，且最后一条 role 为 assistant；
    completion-only loss 依赖最后一条 assistant 定位答案区间。
    """
    rows = iter_jsonl_checked(path, SFT_REQUIRED_KEYS)
    if limit is not None:
        rows = rows[:limit]

    for i, row in enumerate(rows):
        msgs = row["messages"]
        if not isinstance(msgs, list) or not msgs:
            raise ValueError(f"第 {i} 条样本的 messages 不是非空列表: {type(msgs).__name__}")
        for j, m in enumerate(msgs):
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                raise ValueError(
                    f"第 {i} 条样本的 messages[{j}] 缺少 role/content: {m!r}"
                )
        if msgs[-1]["role"] != "assistant":
            raise ValueError(
                f"第 {i} 条样本的最后一个 message 是 {msgs[-1]['role']!r}，不是 assistant。"
                f"completion-only loss 靠最后一个 assistant 消息定位答案区间，"
                f"这里传错文件（比如把 grpo_prompts.jsonl 当成 sft_train.jsonl）"
                f"会让模型去学'复述指令'。"
            )

    logger.info("载入 SFT 语料 %s：%d 条", Path(str(path)).name, len(rows))
    return rows


def load_grpo_prompts(path: str | Path, *, limit: int | None = None) -> list[dict]:
    """读取 GRPO prompt 集并校验格式。

    `prompt` 须为非空 messages 列表且不含 assistant；`meta["task"]` 必须存在，
    奖励函数依赖它选择权重预设。
    """
    rows = iter_jsonl_checked(path, GRPO_REQUIRED_KEYS)
    if limit is not None:
        rows = rows[:limit]

    for i, row in enumerate(rows):
        prompt = row["prompt"]
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f"第 {i} 条 prompt 不是非空列表: {type(prompt).__name__}")
        roles = [m.get("role") for m in prompt if isinstance(m, dict)]
        if "assistant" in roles:
            raise ValueError(
                f"第 {i} 条 prompt 里含 assistant 消息。GRPO 的答案是模型自己采样出来的，"
                f"prompt 里带答案等于把标准答案泄漏给了策略（这条大概率是 SFT 语料）。"
            )
        meta = row["meta"]
        if not isinstance(meta, dict) or not meta.get("task"):
            raise ValueError(
                f"第 {i} 条的 meta 缺少 task 字段（现有键: "
                f"{sorted(meta) if isinstance(meta, dict) else type(meta).__name__}）。"
                f"奖励函数靠 meta['task'] 选权重预设，缺失会静默退回 balanced 权重。"
            )

    logger.info("载入 GRPO prompt 集 %s：%d 条", Path(str(path)).name, len(rows))
    return rows


# ---------------------------------------------------------------------------
# 配置映射
# ---------------------------------------------------------------------------
def estimate_sft_num_training_steps(cfg: SFTConfig, dataset_size: int) -> int:
    """估算 SFT 总步数，用于换算 warmup 比例与 cosine 衰减点。

    一个 optimizer step 消耗 `batch_size × gradient_accumulation_steps` 条样本；
    按每个 micro-batch 装满估算，真实步数不超过该值。`max_steps` 优先。
    """
    if cfg.max_steps is not None and cfg.max_steps > 0:
        return int(cfg.max_steps)
    per_step = max(1, cfg.per_device_train_batch_size * max(1, cfg.gradient_accumulation_steps))
    steps_per_epoch = -(-max(1, dataset_size) // per_step)   # 向上取整
    return max(1, int(-(-steps_per_epoch * cfg.num_train_epochs // 1)))


def warmup_steps_for(warmup_ratio: float, num_training_steps: int) -> int:
    """把 warmup 比例换算成步数，SFT 与 GRPO 共用。

    `ratio <= 0` 返回 0；`ratio > 0` 至少 1 步且不超过总步数。
    """
    if num_training_steps < 1:
        raise ValueError(f"num_training_steps 必须 >= 1，收到 {num_training_steps}")
    if warmup_ratio <= 0:
        return 0
    return max(1, min(int(num_training_steps * warmup_ratio), num_training_steps))


def estimate_grpo_num_training_steps(cfg: GRPOConfig, num_prompts: int) -> int:
    """估算 GRPO 总步数（跑满 `num_train_epochs`）。

    一个 micro-batch 消耗 `per_device_train_batch_size` 条 prompt，每条 prompt
    采样 `num_generations` 条 completion；该字段不进入分母。此字段语义为 prompt 数，
    与 trl 中同名字段定义为 completion 数不同。`max_steps` 优先。
    """
    if cfg.max_steps is not None and cfg.max_steps > 0:
        return int(cfg.max_steps)
    per_step = max(
        1, cfg.per_device_train_batch_size * max(1, cfg.gradient_accumulation_steps)
    )
    steps_per_epoch = -(-max(1, num_prompts) // per_step)   # 向上取整
    return max(1, int(-(-steps_per_epoch * cfg.num_train_epochs // 1)))


def build_scheduler_spec(cfg: TrainCommonConfig, *, num_training_steps: int) -> dict[str, Any]:
    """把 `lr_scheduler_type` / `warmup_ratio` 换算成 `get_scheduler` 的参数。

    两阶段共用，避免自写循环下这两个字段配置后不生效。
    `num_training_steps` 由调用方传入，须与真实步数一致。
    """
    return {
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "num_warmup_steps": warmup_steps_for(cfg.warmup_ratio, num_training_steps),
        "num_training_steps": num_training_steps,
    }
