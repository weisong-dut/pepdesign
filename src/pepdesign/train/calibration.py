"""校准集预热：在正式 RL 之前，用专家校准对对策略做少量监督学习。

流程位置：SFT → 校准预热 → GRPO。预热把策略拉到"采样结果已是有意义的改造方案"
的起点，使 GRPO 的组内比较具备区分度。预热步数受限，学习率相对 SFT 打折；
预热复用 SFT 的掩码与交叉熵路径，并复用 GRPO 的 optimizer。
"""

from __future__ import annotations

from pathlib import Path

from pepdesign.models.loader import chat_tokenize
from pepdesign.train.masking import apply_mask_to_labels, build_completion_mask, pad_batch
from pepdesign.utils.config import ProjectConfig, resolve_path
from pepdesign.utils.io import read_jsonl
from pepdesign.utils.logger import get_logger

logger = get_logger(__name__)

#: 预热的最多 optimizer step 数。500 对、batch 4、1 epoch 为 125 步，
#: 上限需 ≥125，否则"1 个 epoch"会被静默截短。
MAX_CALIBRATION_STEPS: int = 200


def resolve_warmup_path(cfg: ProjectConfig) -> Path:
    """返回预热样本文件路径。

    `calibration_pairs.jsonl` 存在同名 `calibration_warmup.jsonl` 时优先使用后者，
    避免重复转换；否则返回原路径。
    """
    p = resolve_path(cfg.grpo.calibration_path)
    if p.name == "calibration_pairs.jsonl":
        sibling = p.with_name("calibration_warmup.jsonl")
        if sibling.exists():
            return sibling
    return p


def load_calibration_samples(cfg: ProjectConfig, *, limit: int | None = None) -> list[dict]:
    """读取预热数据，统一为 `{"messages": [...]}` 形式。

    warmup 文件直接返回；校准对文件通过 `calibration_to_warmup_samples` 现场转换。
    `limit` 限制返回条数。
    """
    path = resolve_warmup_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"校准集不存在: {path}。请先跑数据构建（pipeline 的第 6 步会产出 "
            f"calibration_pairs.jsonl 与 calibration_warmup.jsonl），"
            f"或把 grpo.calibration_enabled 设为 false 跳过预热。"
        )

    rows = read_jsonl(path)
    if not rows:
        raise RuntimeError(f"校准集为空: {path}")

    if "messages" in rows[0]:
        samples = rows
    else:
        from pepdesign.data.build_calibration import (
            CalibrationPair,
            calibration_to_warmup_samples,
        )

        # 按 dataclass 字段名过滤，忽略落盘时附带的审计字段：
        # 直接 **row 会在数据新增字段时抛 TypeError。
        fields = CalibrationPair.__dataclass_fields__
        pairs = [CalibrationPair(**{k: v for k, v in r.items() if k in fields}) for r in rows]
        samples = calibration_to_warmup_samples(pairs)
        logger.info("从校准对现场转换出 %d 条预热样本", len(samples))

    if limit is not None:
        samples = samples[:limit]
    return samples


def calibration_step_loss(model, tokenizer, samples, *, max_length: int = 512):
    """一批预热样本的交叉熵损失，与 SFT 共用 `masking.py` 的掩码逻辑。

    只对答案 token 计算 loss，prompt 部分不计入梯度。
    """
    import torch

    # 顺序：先验证前缀、再截断、最后算掩码。前缀断言必须在截断前，
    # 否则超长 prompt 被截断后会以无关原因失败；掩码必须在截断后，否则会覆盖已截断的 token。
    encoded = []
    for row in samples:
        prompt_ids = chat_tokenize(
            tokenizer, row["messages"][:-1], add_generation_prompt=True
        )
        full_ids = chat_tokenize(tokenizer, row["messages"])
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise AssertionError(
                "chat template 在 add_generation_prompt 开/关下产出的前缀不一致，"
                "无法据此定位答案区间。"
            )
        if len(prompt_ids) >= max_length:
            raise ValueError(
                f"max_length={max_length} 装不下 prompt（{len(prompt_ids)} 个 token），"
                f"截断后答案区一个 token 都不剩，这一步不会有任何梯度。"
                f"请调大 max_length 或检查模板是不是被改长了。"
            )
        full_ids = full_ids[:max_length]
        mask = build_completion_mask(len(prompt_ids), len(full_ids))
        encoded.append((full_ids, mask))

    batched = pad_batch(
        [e[0] for e in encoded],
        tokenizer.pad_token_id,
        token_masks=[e[1] for e in encoded],
    )
    labels = [
        apply_mask_to_labels(ids, mask)
        for ids, mask in zip(batched["input_ids"], batched["token_mask"], strict=True)
    ]

    device = next(model.parameters()).device
    input_ids = torch.tensor(batched["input_ids"], dtype=torch.long, device=device)
    attention = torch.tensor(batched["attention_mask"], dtype=torch.long, device=device)
    label_t = torch.tensor(labels, dtype=torch.long, device=device)

    n_supervised = int((label_t != -100).sum())
    if n_supervised == 0:
        raise RuntimeError(
            "预热批次里没有任何被监督的 token。"
            "通常意味着 max_length 比 prompt 还短，答案整段被截掉了——"
            "这时 loss 会是 nan，但不会有任何报错。"
        )

    out = model(input_ids=input_ids, attention_mask=attention)
    # labels 已是"位置 t 该预测的 token"，交叉熵内部的 shift 由 F.cross_entropy 完成，
    # 不要再手动切一刀。
    return torch.nn.functional.cross_entropy(
        out.logits.view(-1, out.logits.size(-1)).float(),
        label_t.view(-1),
        ignore_index=-100,
    )


def run_calibration_warmup(model, tokenizer, cfg: ProjectConfig, optimizer) -> list[float]:
    """用校准集跑若干步监督学习，原地更新 model，返回每步 loss。

    复用传入的 optimizer，使学习率、权重衰减与梯度裁剪参数和 RL 阶段一致。
    """
    import torch

    samples = load_calibration_samples(cfg)
    lr = cfg.grpo.learning_rate * cfg.grpo.calibration_lr_scale
    for group in optimizer.param_groups:
        group["lr"] = lr

    batch_size = max(1, cfg.grpo.per_device_train_batch_size)
    steps = min(
        MAX_CALIBRATION_STEPS,
        max(1, int(len(samples) * cfg.grpo.calibration_epochs / batch_size)),
    )
    logger.info(
        "校准预热：%d 条样本，%d 步，lr=%.2e（RL 阶段的 %.0f%%）",
        len(samples), steps, lr, 100 * cfg.grpo.calibration_lr_scale,
    )

    losses: list[float] = []
    model.train()
    for step in range(steps):
        start = (step * batch_size) % len(samples)
        chunk = [samples[(start + i) % len(samples)] for i in range(batch_size)]
        optimizer.zero_grad()
        loss = calibration_step_loss(model, tokenizer, chunk)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg.grpo.max_grad_norm
        )
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 0 or (step + 1) % max(1, cfg.grpo.logging_steps) == 0:
            logger.info("预热步 %d/%d: loss %.4f", step + 1, steps, losses[-1])

    if losses and losses[-1] > losses[0]:
        logger.warning(
            "预热 loss 没有下降（%.4f → %.4f）。这通常是正常的——"
            "校准集只有几百条、步数很少，loss 的波动可能大于趋势。"
            "但如果在完整训练后 reward 仍然不涨，检查一下这几步是不是"
            "把模型带偏了（把 learning_rate 或 calibration_lr_scale 调小重试）。",
            losses[0], losses[-1],
        )

    if cfg.grpo.reset_optimizer_after_calibration:
        # 清零而非重建：GRPO 循环持有 optimizer 引用，清零 state 效果相同。
        optimizer.state.clear()
        logger.info("已清空 optimizer 状态（reset_optimizer_after_calibration=True）")

    for group in optimizer.param_groups:
        group["lr"] = cfg.grpo.learning_rate

    return losses
