"""阶段一：监督微调（LoRA），自写训练循环，不依赖 trl。

流程：chat 语料 → tokenizer + 掩码 → 长度分桶批量 → 手写循环 → LoRA adapter。

训练循环实现混合精度（bf16 autocast / fp16 GradScaler）、梯度累积、梯度裁剪、
warmup + cosine 调度、日志、checkpoint 保存与续训。不使用 packing，以长度分桶
加动态 padding 控制 pad 占比。可验证部分拆为纯函数，由配置映射测试与 CPU 冒烟
测试覆盖。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

from pepdesign.models.loader import (
    build_tiny_model,
    chat_tokenize,
    enable_gradient_checkpointing,
    load_base_model,
    load_tokenizer,
    probe_hardware,
    resolve_plan,
    tiny_lora_config,
)
from pepdesign.models.lora import inject_lora, save_adapter
from pepdesign.train.checkpoint import (
    CheckpointManager,
    list_checkpoints,
    on_step_end,
    resolve_resume_path,
    training_fingerprint,
)
from pepdesign.train.dataset import (
    build_scheduler_spec,
    estimate_sft_num_training_steps,
    load_sft_samples,
)
from pepdesign.train.masking import (
    apply_mask_to_labels,
    build_completion_mask,
    pad_batch,
    summarize_lengths,
)
from pepdesign.utils.config import ProjectConfig, save_config
from pepdesign.utils.logger import get_logger, silence_noisy_libraries
from pepdesign.utils.seed import set_seed
from pepdesign.utils.tracking import MetricWriter

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 纯函数层：不 import torch
# ---------------------------------------------------------------------------
def encode_sft_sample(
    tokenizer,
    row: dict,
    *,
    max_seq_length: int,
    completion_only: bool,
) -> dict:
    """把一条 chat 样本编码为 `(input_ids, token_mask)` 并按预算截断。

    答案区间由同一 tokenizer 两次渲染确定：`full_ids[len(prompt_ids):]`，
    前提是 `full_ids` 的前缀逐 token 等于 `prompt_ids`。超长时优先丢弃答案
    放不下的样本；答案放得下时从左侧截 prompt、保留答案。
    """
    messages = row["messages"]
    prompt_ids = chat_tokenize(tokenizer, messages[:-1], add_generation_prompt=True)
    full_ids = chat_tokenize(tokenizer, messages)

    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise AssertionError(
            "chat template 在 add_generation_prompt=True/False 下产出的前缀不一致，"
            "无法据此定位答案区间。请检查 tokenizer 的 chat_template 是否含"
            "随调用变化的成分。"
        )
    if len(prompt_ids) >= len(full_ids):
        # 答案区间为 0 个 token：该样本对 completion-only loss 无贡献，进入 batch 只会稀释梯度。
        return {"input_ids": [], "token_mask": [], "dropped": "empty_completion"}

    answer_ids = list(full_ids[len(prompt_ids):])
    if len(answer_ids) >= max_seq_length:
        # 答案超出预算时不截断答案，直接丢弃整条。
        return {"input_ids": [], "token_mask": [], "dropped": "answer_too_long"}

    # prompt 预算为总长减去答案长度，超出部分从左侧截掉。
    budget = max_seq_length - len(answer_ids)
    prompt_ids = list(prompt_ids[-budget:]) if budget > 0 else []

    ids = prompt_ids + answer_ids
    mask = build_completion_mask(len(prompt_ids), len(ids))
    if not completion_only:
        # 全序列计算 loss：把 prompt 部分的掩码也置 1。
        mask = [1] * len(ids)
    return {"input_ids": ids, "token_mask": mask, "dropped": None}


def build_sft_examples(
    tokenizer,
    rows: Sequence[dict],
    *,
    max_seq_length: int,
    completion_only: bool,
) -> tuple[list[dict], dict[str, int]]:
    """批量编码，返回 `(可用样本, 丢弃原因计数)`。

    丢弃必须计数并记录日志，静默丢弃无法从 loss 曲线察觉。
    """
    examples: list[dict] = []
    dropped: dict[str, int] = {}
    for row in rows:
        enc = encode_sft_sample(
            tokenizer, row,
            max_seq_length=max_seq_length,
            completion_only=completion_only,
        )
        if enc["dropped"]:
            dropped[enc["dropped"]] = dropped.get(enc["dropped"], 0) + 1
            continue
        examples.append({"input_ids": enc["input_ids"], "token_mask": enc["token_mask"]})

    if dropped:
        logger.warning(
            "编码时丢弃了 %d 条样本：%s。"
            "answer_too_long 占比高说明 max_seq_length 配小了。",
            sum(dropped.values()), dropped,
        )
    return examples, dropped


def batch_indices_by_length(
    lengths: Sequence[int],
    batch_size: int,
    *,
    shuffle: bool = True,
    seed: int = 42,
) -> list[list[int]]:
    """按长度分桶后组 batch，返回每步的样本下标。

    同批样本长度接近，动态 padding 的 pad 占比低。`shuffle` 打乱桶间顺序，
    `seed` 固定分桶结果以保证可复现。
    """
    if batch_size < 1:
        raise ValueError(f"batch_size 必须 >= 1，收到 {batch_size}")
    n = len(lengths)
    if n == 0:
        return []

    order = sorted(range(n), key=lambda i: (lengths[i], i))
    buckets = [order[i: i + batch_size] for i in range(0, n, batch_size)]

    if shuffle:
        import random

        rng = random.Random(seed)
        rng.shuffle(buckets)
    return buckets


# ---------------------------------------------------------------------------
# torch 层
# ---------------------------------------------------------------------------
def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "SFT 训练需要 torch，当前环境没有装。请装训练依赖：\n"
            "  pip install -e '.[train]'\n"
            "纯 CPU 的逻辑验证（掩码/优势/解析/参数量）不需要 torch，"
            "直接跑 pytest -m 'not smoke' 即可。"
        ) from exc
    return torch


def make_sft_dataset(torch, examples: Sequence[dict]):
    """把样本列表包装为 `torch.utils.data.Dataset`。

    `__getitem__` 返回变长 list，padding 推迟到 `collate_sft` 完成。
    """

    class _SFTDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(examples)

        def __getitem__(self, i: int) -> dict:
            return examples[i]

    return _SFTDataset()


def collate_sft(torch, batch: Sequence[dict], pad_token_id: int) -> dict:
    """把一批变长样本补到本批最长，并生成 labels。

    掩码必须与 ids 一起补齐，否则 labels 与 input_ids 长度不一致。
    """
    batched = pad_batch(
        [b["input_ids"] for b in batch],
        pad_token_id,
        token_masks=[b["token_mask"] for b in batch],
    )
    labels = [
        apply_mask_to_labels(ids, mask)
        for ids, mask in zip(batched["input_ids"], batched["token_mask"], strict=True)
    ]
    return {
        "input_ids": torch.tensor(batched["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(batched["attention_mask"], dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def train_loop(
    model,
    examples: Sequence[dict],
    cfg: ProjectConfig,
    *,
    tokenizer,
    device,
    plan,
    eval_examples: Sequence[dict] | None = None,
    max_steps: int | None = None,
) -> dict:
    """手写 SFT 训练循环，返回训练指标字典。

    每个 optimizer step 依次执行 grad_accum 次前向/反向、梯度裁剪、参数更新与
    调度器更新。顺序约束：`unscale_` 在 `clip_grad_norm_` 之前；日志使用未缩放的
    `raw_loss`；bf16 用 autocast，fp16 用 GradScaler。
    """
    torch = _require_torch()
    from transformers import get_scheduler

    device_obj = torch.device(device) if isinstance(device, str) else device
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(
            "没有任何 requires_grad=True 的参数，训练不会有任何效果。"
            "检查 LoRA 是否真的注入成功（inject_lora 之后可训练参数应远小于总参数）。"
        )

    optimizer = torch.optim.AdamW(
        params,
        # 学习率统一由 `resolved_lr()` 取得；续训时也用它把配置值压回调度器。
        lr=cfg.sft.resolved_lr(),
        weight_decay=cfg.sft.weight_decay,
    )

    lengths = [len(e["input_ids"]) for e in examples]
    batch_size = max(1, cfg.sft.per_device_train_batch_size)
    grad_accum = max(1, cfg.sft.gradient_accumulation_steps)
    # 步数须在构造调度器前估准：过小则后半程 lr 为 0，过大则到不了衰减段。
    # 一个 optimizer step 消耗 batch_size × grad_accum 条样本。
    num_steps = (
        int(max_steps) if max_steps is not None and max_steps > 0
        else estimate_sft_num_training_steps(cfg.sft, len(examples))
    )

    sched_args = build_scheduler_spec(cfg.sft, num_training_steps=num_steps)
    scheduler = get_scheduler(
        sched_args["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=sched_args["num_warmup_steps"],
        num_training_steps=sched_args["num_training_steps"],
    )
    logger.info(
        "SFT 计划：%d 条样本，batch=%d × 累积=%d，共 %d 个 optimizer step"
        "（warmup %d 步）",
        len(examples), batch_size, grad_accum, num_steps, sched_args["num_warmup_steps"],
    )

    # 混合精度：bf16 与 fp16 只能开一个，且 fp16 才需要 GradScaler。
    use_bf16 = plan.dtype == "bfloat16"
    use_fp16 = plan.dtype == "float16"
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer 没有 pad_token_id，无法做动态 padding。")

    history = {"loss": [], "steps": [], "lr": [], "completion_tokens": []}
    step = 0
    micro_seen = 0
    running_loss = 0.0
    running_tokens = 0
    started = time.time()

    # ---- 指标外送 ----
    # 无可用后端时为 no-op（见 utils/tracking.py）。
    writer = MetricWriter.from_config(
        cfg.sft.report_to, output_dir=cfg.sft.output_dir, prefix="sft"
    )
    if not writer.enabled and cfg.sft.report_to:
        logger.warning(
            "report_to=%s 但没有任何后端可用，本次不写指标（训练不受影响）。",
            list(cfg.sft.report_to),
        )
    if cfg.sft.eval_steps > 0 and not eval_examples:
        # 配置了 eval_steps 但没有验证集时需要告警，否则该配置静默失效。
        logger.warning(
            "eval_steps=%d 但本次没有验证集（eval_dataset_path 为空或文件不存在），"
            "训练中途不会评估。", cfg.sft.eval_steps,
        )

    # ---- checkpoint 与断点续训 ----
    ckpt = CheckpointManager(
        output_dir=cfg.sft.output_dir,
        save_total_limit=cfg.sft.save_total_limit,
        lora_cfg=cfg.lora,
        phase="sft",
    )
    fingerprint = training_fingerprint(cfg.sft, phase="sft", num_steps=num_steps)
    resume_path = resolve_resume_path(cfg.sft.resume_from_checkpoint, cfg.sft.output_dir)
    resumed_from: str | None = None
    if resume_path is not None:
        state = ckpt.load_into(
            resume_path,
            model=model, optimizer=optimizer, scheduler=scheduler,
            expected_fingerprint=fingerprint,
            learning_rate=cfg.sft.resolved_lr(),
        )
        step = int(state["step"])
        resumed_from = str(resume_path)
        if state.get("history"):
            # history 的形状是 dict[str, list]，直接接着往后 append 即可。
            history = {k: list(v) for k, v in state["history"].items()}
        logger.info("从 step %d 接着跑（共 %d 步）", step, num_steps)

    def _micro_batches(skip_micro_batches: int = 0):
        """无限产出 micro-batch，由 `num_steps` 控制停止。

        写成无限循环使 `max_steps` 与 `num_train_epochs` 共用同一路径。
        `skip_micro_batches` 用于续训时恢复数据位置；分桶只由长度、batch 与 seed 决定。
        """
        epoch = 0
        to_skip = skip_micro_batches
        while True:
            for bucket in batch_indices_by_length(
                lengths, batch_size, seed=cfg.sft.seed + epoch
            ):
                if to_skip > 0:
                    to_skip -= 1
                    continue
                yield [examples[j] for j in bucket]
            epoch += 1

    gen = _micro_batches(skip_micro_batches=step * grad_accum)
    optimizer.zero_grad(set_to_none=True)

    eval_history: dict[int, float] = {}

    def _evaluate() -> float:
        assert eval_examples is not None
        return evaluate_loss(model, eval_examples, tokenizer, device_obj, plan, cfg)

    def _save() -> None:
        ckpt.save(
            step, total_steps=num_steps, model=model, tokenizer=tokenizer,
            optimizer=optimizer, scheduler=scheduler,
            history=history, fingerprint=fingerprint,
        )

    # `try/finally` 保证 writer 被关闭：SummaryWriter 有 120 秒 flush 间隔。
    try:
        while step < num_steps:
            batch_examples = next(gen)
            if not batch_examples:
                continue
            batch = collate_sft(torch, batch_examples, pad_token_id)
            batch = {k: v.to(device_obj) for k, v in batch.items()}
            # 答案 token 数恒为 0 说明掩码被吃空，loss 会因 labels 全为 ignore_index 变成 nan。
            running_tokens += int((batch["labels"] != -100).sum().item())

            # 除以 grad_accum：梯度累积下不除会让等效学习率放大 N 倍。
            # 日志记录未缩放的 raw_loss，使 train/eval loss 可比。
            if use_bf16 or use_fp16:
                with torch.autocast(device_type=device_obj.type, dtype=autocast_dtype):
                    raw_loss = model(**batch).loss
            else:
                raw_loss = model(**batch).loss
            loss = raw_loss / grad_accum

            if not torch.isfinite(loss):
                # loss 非有限值时立即停止，避免用坏梯度继续训练。
                raise RuntimeError(
                    f"第 {step} 步 loss 不是有限值（{loss.item()}）。"
                    f"最常见的原因：学习率过大、fp16 溢出，"
                    f"或者 labels 全是 ignore_index（答案掩码被吃掉了）。"
                    f"可以先把 sft.learning_rate 调小一个量级再试。"
                )

            scaler.scale(loss).backward()
            running_loss += float(raw_loss.item())
            micro_seen += 1

            if micro_seen % grad_accum == 0:
                # unscale 必须在裁剪之前：GradScaler 放大过的梯度直接裁剪会失去阈值意义。
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, cfg.sft.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                lr = scheduler.get_last_lr()[0]
                history["loss"].append(round(running_loss / grad_accum, 4))
                # 1-based，与日志和 on_step_end 的 step 口径一致。
                history["steps"].append(step + 1)
                history["lr"].append(round(lr, 8))
                history["completion_tokens"].append(running_tokens)
                if (step + 1) % max(1, cfg.sft.logging_steps) == 0 or step == 0:
                    logger.info(
                        "step %d/%d | loss %.4f | lr %.3g | 本步答案 token 数 %d",
                        step + 1, num_steps, history["loss"][-1], lr, running_tokens,
                    )
                running_loss = 0.0
                running_tokens = 0
                step += 1

                # 所有步末收尾统一在 on_step_end 中执行：评估、写指标、存 checkpoint。
                merged = on_step_end(
                    step=step, total_steps=num_steps,
                    eval_steps=cfg.sft.eval_steps, save_steps=cfg.sft.save_steps,
                    writer=writer,
                    scalars={
                        "loss": history["loss"][-1],
                        "lr": lr,
                        "completion_tokens": history["completion_tokens"][-1],
                    },
                    evaluate=_evaluate if eval_examples else None,
                    save=_save,
                )
                if "eval_loss" in merged:
                    eval_history[step] = merged["eval_loss"]
    finally:
        writer.close()

    elapsed = time.time() - started
    result = {
        "num_steps": step,
        "final_loss": history["loss"][-1] if history["loss"] else None,
        "first_loss": history["loss"][0] if history["loss"] else None,
        "history": history,
        "eval_history": eval_history,
        "checkpoints": [str(p) for _, p in list_checkpoints(cfg.sft.output_dir)],
        "resumed_from": resumed_from,
        "elapsed_sec": round(elapsed, 1),
        "num_examples": len(examples),
        "trainable_params": sum(p.numel() for p in params),
        "dtype": plan.dtype,
    }

    if eval_examples:
        result["eval_loss"] = evaluate_loss(
            model, eval_examples, tokenizer, device_obj, plan, cfg
        )
    logger.info(
        "SFT 结束：%d 步，loss %s → %s，用时 %.1fs",
        step, result["first_loss"], result["final_loss"], elapsed,
    )
    return result


def evaluate_loss(model, examples, tokenizer, device, plan, cfg) -> float:
    """在给定样本上计算平均 loss，不更新参数。

    使用 `model.eval()` 关闭 dropout，`torch.no_grad()` 节省激活显存。
    """
    torch = _require_torch()

    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    batch_size = max(1, cfg.sft.per_device_eval_batch_size)
    batches = batch_indices_by_length(
        [len(e["input_ids"]) for e in examples], batch_size, shuffle=False
    )
    with torch.no_grad():
        for b in batches:
            batch = collate_sft(torch, [examples[j] for j in b], tokenizer.pad_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            total += float(out.loss.item())
            n += 1
    if was_training:
        model.train()
    return round(total / max(n, 1), 4)


# ---------------------------------------------------------------------------
# 组装与入口
# ---------------------------------------------------------------------------
def preflight_tokenizer(tokenizer, cfg: ProjectConfig) -> dict:
    """检查 tokenizer 是否具备 chat_template 与 pad_token。

    缺失时直接报错；pad_token == eos_token 只记录日志，不改配置。
    """
    template = getattr(tokenizer, "chat_template", None)
    report = {
        "has_chat_template": bool(template),
        "has_pad_token": tokenizer.pad_token_id is not None,
        "pad_is_eos": (
            tokenizer.pad_token_id is not None
            and tokenizer.pad_token_id == tokenizer.eos_token_id
        ),
    }
    if not template:
        raise ValueError(
            "tokenizer 没有 chat_template，无法渲染 messages 格式的语料。"
            "本项目语料是对话格式，必须要有模板（Qwen / Llama-3 系都自带）。"
        )
    if not report["has_pad_token"]:
        raise ValueError(
            "tokenizer 没有 pad_token_id，无法做动态 padding。"
            "通常的修法是 `tokenizer.pad_token = tokenizer.eos_token`，"
            "但要注意这与 pad_is_eos 属于同一类问题，见 preflight_tokenizer 的说明。"
        )
    if report["pad_is_eos"]:
        logger.info(
            "tokenizer 的 pad_token == eos_token（本项目基座的默认情况）。"
            "对训练无影响（pad 位置的 loss 已被 ignore_index 掩掉），"
            "但推理时要注意停止条件不能只看 eos。"
        )
    return report


def build_trainer(cfg: ProjectConfig, *, tokenizer=None, model=None, examples=None):
    """装配训练所需的 `(model, examples, tokenizer, device, plan)`。

    名字保留用于兼容调用方，但不返回 trainer 对象。先加载基座再注入 LoRA，
    使无可用可训练参数的错误在计算开始前暴露。
    """
    if tokenizer is None:
        tokenizer = load_tokenizer(cfg.sft.model_name_or_path)

    preflight_tokenizer(tokenizer, cfg)

    hw = probe_hardware()
    plan = resolve_plan(
        cuda_available=hw["cuda_available"],
        bf16_supported=hw["bf16_supported"],
        mps_available=hw["mps_available"],
        want_bf16=cfg.sft.bf16,
        want_fp16=cfg.sft.fp16,
        quant=cfg.quant,
    )

    if model is None:
        model, _plan = load_base_model(
            cfg.sft.model_name_or_path, quant=cfg.quant,
            want_bf16=cfg.sft.bf16, want_fp16=cfg.sft.fp16, plan=plan,
        )
        model = inject_lora(model, cfg.lora)

    # 必须在注入 LoRA 之后：梯度检查点会给输入 embedding 挂 forward hook。
    enable_gradient_checkpointing(model, enabled=cfg.sft.gradient_checkpointing)

    if examples is None:
        rows = load_sft_samples(cfg.sft.dataset_path)
        length_preflight(cfg, tokenizer, rows)
        examples, _dropped = build_sft_examples(
            tokenizer, rows,
            max_seq_length=cfg.sft.max_seq_length,
            completion_only=cfg.sft.completion_only_loss,
        )

    device = next(model.parameters()).device
    return model, examples, tokenizer, device, plan


def run_sft(cfg: ProjectConfig) -> Path:
    """跑完阶段一，返回 adapter 保存目录。

    两阶段串联，调用方需要知道上一阶段的产物位置。
    """
    silence_noisy_libraries()
    set_seed(cfg.sft.seed)
    logger.info("=== 阶段一：SFT（LoRA，自写训练循环，无 trl）===")

    model, examples, tokenizer, device, plan = build_trainer(cfg)
    if not examples:
        raise RuntimeError(
            f"编码后没有可用样本（语料 {cfg.sft.dataset_path}）。"
            f"检查 max_seq_length 是否配得太小。"
        )

    eval_examples = None
    if cfg.sft.eval_dataset_path:
        try:
            eval_rows = load_sft_samples(cfg.sft.eval_dataset_path)
            eval_examples, _ = build_sft_examples(
                tokenizer, eval_rows,
                max_seq_length=cfg.sft.max_seq_length,
                completion_only=cfg.sft.completion_only_loss,
            )
        except FileNotFoundError:
            # 验证集不是必需品；不存在时告警并跳过，不中断训练。
            logger.warning("验证集 %s 不存在，本次训练不评估。", cfg.sft.eval_dataset_path)

    # 落盘最终生效的配置，包含命令行覆盖。
    cfg_path = save_config(cfg, Path(cfg.sft.output_dir) / "resolved_config.yaml")
    logger.info("最终生效的配置已保存: %s", cfg_path)

    result = train_loop(
        model, examples, cfg,
        tokenizer=tokenizer, device=device, plan=plan, eval_examples=eval_examples,
    )

    out = save_adapter(model, Path(cfg.sft.output_dir) / "final", tokenizer, cfg.lora)
    logger.info("=== SFT 完成，adapter 在 %s ===", out)
    logger.info("本次训练指标：%s", {
        k: v for k, v in result.items() if k != "history"
    })
    logger.info(
        "下一步：python scripts/train_grpo.py（GRPOConfig.sft_adapter_path 默认指向 %s）",
        cfg.grpo.sft_adapter_path,
    )
    return out


def length_preflight(cfg: ProjectConfig, tokenizer, samples, *, sample_n: int = 64) -> dict:
    """抽样估算语料 token 长度分布，检查 `max_seq_length` 是否合适。

    只抽 `sample_n` 条；超长样本会被静默丢弃或截断，因此需要预检。
    """
    if not samples:
        return {"count": 0}
    step = max(1, len(samples) // sample_n)
    picked = samples[::step][:sample_n]
    lengths = []
    for row in picked:
        # 走 chat_tokenize 而非 apply_chat_template：BatchEncoding 的 len() 是键数，
        # 会让本函数永远不报警。
        ids = chat_tokenize(tokenizer, row["messages"])
        lengths.append(len(ids))
    report = summarize_lengths(lengths, cfg.sft.max_seq_length)
    if report.get("should_warn"):
        logger.warning(
            "抽样 %d 条中有 %d 条 (%.1f%%) 超过 max_seq_length=%d。"
            "超长样本会在编码阶段被丢弃（见 encode_sft_sample），"
            "请调大 sft.max_seq_length 或检查语料。",
            report["count"], report["truncated"],
            100 * report["truncated_ratio"], cfg.sft.max_seq_length,
        )
    else:
        logger.info("语料长度预检：%s", report)
    return report


def run_sft_smoke(cfg: ProjectConfig, *, steps: int = 3) -> dict:
    """CPU 冒烟：用随机初始化的小模型运行几步 SFT，返回观测指标。

    不加载真实权重，验证梯度传递、LoRA 注入、loss 有限性与参数更新；
    不证明训练效果。
    """
    torch = _require_torch()

    set_seed(cfg.sft.seed)
    tokenizer = load_tokenizer(cfg.sft.model_name_or_path)
    model = build_tiny_model(vocab_size=len(tokenizer))
    model = inject_lora(model, tiny_lora_config())
    enable_gradient_checkpointing(model, enabled=cfg.sft.gradient_checkpointing)
    device = next(model.parameters()).device

    rows = load_sft_samples(cfg.sft.dataset_path, limit=steps * 4)
    if not rows:
        raise RuntimeError("语料为空，无法冒烟")

    examples, _ = build_sft_examples(
        tokenizer, rows,
        max_seq_length=cfg.sft.max_seq_length,
        completion_only=cfg.sft.completion_only_loss,
    )
    if not examples:
        raise RuntimeError("编码后没有可用样本，无法冒烟")

    # 小模型的 packing/分桶都不走真实路径，直接看 LoRA 参数有没有动。
    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    hw = probe_hardware()
    plan = resolve_plan(
        cuda_available=hw["cuda_available"], bf16_supported=hw["bf16_supported"],
        mps_available=hw["mps_available"], want_bf16=False, want_fp16=False,
        quant=cfg.quant,
    )
    result = train_loop(
        model, examples, cfg,
        tokenizer=tokenizer, device=device, plan=plan, max_steps=steps,
    )

    after = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    changed = sum(
        1 for a, b in zip(before, after, strict=True) if not torch.equal(a, b)
    )
    result.update({
        "trainable_tensors": len(before),
        "tensors_changed": changed,
        "note": "小模型 + 随机初始化，只证明训练链路通，不证明学得好",
    })
    return result
