"""LoRA 对外门面：注入、保存、加载、合并。

纯逻辑层（build_peft_config / validate_lora_config / lora_trainable_summary /
lora_config_for_model）不依赖 torch；后端层按 `LoRAConfig.backend` 分发到手写实现
或 peft；门面层负责分发。静默失效的配置错误在这里变成显式校验或断言。
"""

from __future__ import annotations

from pathlib import Path

from pepdesign.utils.config import LoRAConfig, resolve_path
from pepdesign.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 纯逻辑半区
# ---------------------------------------------------------------------------
def build_peft_config(cfg: LoRAConfig) -> dict:
    """把 `LoRAConfig` 翻译成 `peft.LoraConfig` 的构造参数。

    返回 dict 便于测试字段映射。只输出 peft 认得的键：不传 `backend` 等本模块字段。
    `bias` 虽为 peft 默认值也显式传入，避免随版本变化。
    """
    validate_lora_config(cfg)
    return {
        "r": cfg.r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "target_modules": cfg.resolved_target_modules(),
        "bias": cfg.bias,
        "task_type": cfg.task_type,
    }


def validate_lora_config(cfg: LoRAConfig) -> None:
    """跨字段校验：`alpha/r` 缩放落在 [0.25, 8]，`r` 较大且 dropout=0 时告警。

    pydantic 只约束单字段，字段之间的关系在构造后检查。
    """
    scaling = cfg.scaling
    if not 0.25 <= scaling <= 8.0:
        raise ValueError(
            f"LoRA 缩放 alpha/r = {cfg.lora_alpha}/{cfg.r} = {scaling:.3f} 超出 [0.25, 8]。"
            f"缩放太大会让 LoRA 分支压过基座（小语料上几个 step 就被带偏），"
            f"太小则学不动。默认取值是 alpha=32, r=16（缩放 2.0）。"
            f"如果确实要改这个比例，请显式确认后再放宽这里的检查。"
        )

    if cfg.lora_dropout == 0.0 and cfg.r >= 32:
        logger.warning(
            "LoRA 配置 r=%d 且 dropout=0：小语料 + 大秩 + 无正则，过拟合风险高。"
            "多肽语料通常在万条量级，建议 r<=16 或给 dropout 0.05。",
            cfg.r,
        )


#: LOGOS-3B 的结构参数，取自 `LOGOS-Hub/LOGOS-pretrain-3B` 的 config.json。
#: 它是标准 Llama-3.2-3B 结构（model_type=llama），无混合注意力、无 MoE。
#: `head_dim` 在 config.json 中显式给出，不靠 hidden_size // num_attention_heads 推算。
#: `tie_word_embeddings=true`，embedding 与 lm_head 共享权重，总参数量只算一次。
LOGOS_3B: dict[str, int] = {
    "hidden_size": 3072,            # config.json: hidden_size
    "num_hidden_layers": 28,        # config.json: num_hidden_layers
    "num_attention_heads": 24,      # config.json: num_attention_heads
    "num_key_value_heads": 8,       # config.json: num_key_value_heads（GQA，比例 24/8）
    "head_dim": 128,                # config.json: head_dim（显式给出）
    "intermediate_size": 8192,      # config.json: intermediate_size
    # 逐项累加：embed 394,060,800 + 每层 100,669,440 × 28 + final norm 3,072
    #        = 3,212,808,192
    "total_params": 3_212_808_192,
}

#: Qwen3-0.6B 的结构参数，用于 CPU 冒烟及另一套 GQA 比例（16/8）的对照。
QWEN3_0_6B: dict[str, int] = {
    "hidden_size": 1024,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 3072,
    "total_params": 596_049_920,
}


def lora_trainable_summary(cfg: LoRAConfig, **overrides: int) -> dict:
    r"""估算 LoRA 可训练参数量与占比，不加载模型。

    单层参数量为 :math:`r \times (d_{in} + d_{out})`，GQA 下 k/v 与 q 的输出维度不同；
    `**overrides` 换模型时必须同时给出头数与 head_dim。
    """
    p = {**LOGOS_3B, **overrides}

    q_out = p["num_attention_heads"] * p["head_dim"]
    kv_out = p["num_key_value_heads"] * p["head_dim"]
    dims = {
        "q_proj": (p["hidden_size"], q_out),
        "k_proj": (p["hidden_size"], kv_out),
        "v_proj": (p["hidden_size"], kv_out),
        # o_proj 是"把多头拼回来的输出投影"，入维是 q_out、出维是 hidden。
        "o_proj": (q_out, p["hidden_size"]),
        "gate_proj": (p["hidden_size"], p["intermediate_size"]),
        "up_proj": (p["hidden_size"], p["intermediate_size"]),
        # down_proj 是三者里唯一"反过来"的：intermediate → hidden。
        "down_proj": (p["intermediate_size"], p["hidden_size"]),
    }

    unknown = [m for m in cfg.resolved_target_modules() if m not in dims]
    per_layer = 0
    for module in cfg.resolved_target_modules():
        if module not in dims:
            # 不认识的模块名跳过并记录，不静默当 0：日志里一个"看起来正常但偏小"
            # 的数字比明确告警更危险。
            continue
        d_in, d_out = dims[module]
        per_layer += cfg.r * (d_in + d_out)

    n_params = per_layer * p["num_hidden_layers"]
    return {
        "r": cfg.r,
        "alpha": cfg.lora_alpha,
        "scaling": round(cfg.scaling, 4),
        "backend": cfg.backend,
        "target_modules": cfg.resolved_target_modules(),
        "unknown_target_modules": unknown,
        "per_layer_params": per_layer,
        "estimated_trainable_params": n_params,
        "total_params": p["total_params"],
        "estimated_share": round(n_params / p["total_params"], 6),
        # adapter 文件大小：参数量 × dtype 字节数（bf16 为 2），
        # 与"参数量"不是同一个数。
        "adapter_bytes_bf16": n_params * 2,
    }


def lora_config_for_model(cfg: LoRAConfig, preset: str = "logos-3b") -> dict:
    """按模型预设算参数量；支持 logos-3b 与 qwen3-0.6b，预设名非法时抛错并列出可选值。"""
    presets = {"logos-3b": LOGOS_3B, "qwen3-0.6b": QWEN3_0_6B}
    if preset not in presets:
        raise ValueError(f"未知的模型预设: {preset!r}，可选值: {sorted(presets)}")
    return lora_trainable_summary(cfg, **presets[preset])


# ---------------------------------------------------------------------------
# torch 半区
# ---------------------------------------------------------------------------
def _require_peft():
    """惰性导入 peft；缺依赖时给出可操作的报错。"""
    try:
        import peft  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 只在缺依赖的环境触发
        raise ImportError(
            "LoRA 需要 peft，当前环境没有装。训练/推理依赖请装 train extra："
            "  pip install -e '.[train]'"
        ) from exc
    from peft import LoraConfig, get_peft_model

    return LoraConfig, get_peft_model


def _inject_lora_peft(model, cfg: LoRAConfig):
    """peft 后端的注入路径，作为手写实现的数值对照基准。"""
    LoraConfig, get_peft_model = _require_peft()
    kwargs = build_peft_config(cfg)
    logger.info(
        "注入 LoRA（peft 后端）: r=%d, alpha=%d (缩放 %.2f), 目标模块 %s",
        cfg.r, cfg.lora_alpha, cfg.scaling, kwargs["target_modules"],
    )
    peft_model = get_peft_model(model, LoraConfig(**kwargs))
    _assert_lora_actually_injected(peft_model, cfg)
    return peft_model


def inject_lora(model, cfg: LoRAConfig):
    """把 LoRA 注入基座并冻结基座参数，返回可训练的模型。

    `manual` 原地替换目标层并返回同一对象，`peft` 返回 PeftModel；
    两种后端都在注入后断言可训练参数量非零。
    """
    if cfg.backend == "manual":
        from pepdesign.models.lora_layer import count_lora_params, inject_manual_lora

        model = inject_manual_lora(model, cfg)
        n_trainable, _ = count_lora_params(model)
        if n_trainable == 0:
            raise RuntimeError(
                f"手写 LoRA 注入后没有任何可训练参数："
                f"target_modules={cfg.resolved_target_modules()}。"
            )
        return model

    if cfg.backend == "peft":
        return _inject_lora_peft(model, cfg)

    # pydantic 的 Literal 已经拦住了非法取值，走到这里说明
    # 有人绕过配置直接构造了别的东西——明确报出来而不是静默走默认分支。
    raise ValueError(
        f"未知的 LoRA 后端: {cfg.backend!r}，可选值: ['manual', 'peft']"
    )


def _trainable_counts(model) -> tuple[int, int]:
    """返回 `(可训练参数量, 总参数量)`。两处断言共用，避免各数一遍数岔。"""
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return n_trainable, n_total


def _assert_lora_actually_injected(peft_model, cfg: LoRAConfig) -> None:
    """断言 LoRA 确实被注入：peft 匹配不到模块时会静默注入 0 个 adapter。"""
    n_trainable, n_total = _trainable_counts(peft_model)
    if n_trainable == 0:
        raise RuntimeError(
            f"LoRA 注入后没有任何可训练参数：target_modules={cfg.resolved_target_modules()} "
            f"在该模型里可能一个都没匹配上。请检查模块名（Llama 系是 "
            f"q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj）。"
        )
    logger.info("LoRA 注入完成：可训练参数 %s / 总计 %s (%.3f%%)",
                f"{n_trainable:,}", f"{n_total:,}", 100.0 * n_trainable / max(n_total, 1))


def save_adapter(model, path: str | Path, tokenizer=None, cfg: LoRAConfig | None = None) -> Path:
    """只保存 adapter，基座不动，并在目录中写入 tokenizer 以便复现。

    手写后端必须传 cfg：adapter_config.json 需要记录 r / alpha / target_modules。
    """
    if _is_manual_model(model):
        if cfg is None:
            raise ValueError(
                "保存手写后端的 adapter 必须传 cfg："
                "adapter_config.json 里要写入 r / alpha / target_modules，"
                "缺了这些，加载时无法判断该挂到哪些层上。"
            )
        from pepdesign.models.lora_layer import save_manual_adapter

        return save_manual_adapter(model, path, cfg, tokenizer)

    out = resolve_path(path)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(out))
    logger.info("adapter 已保存（peft 后端）: %s", out)
    return out


def _is_manual_model(model) -> bool:
    """判断模型是否由手写后端注入。

    只认 `_pepdesign_manual_lora` 标记；peft 的 LoraLayer 也有 `lora_A`，
    不能用 `hasattr` 判断。
    """
    return any(
        getattr(module, "_pepdesign_manual_lora", False)
        for module in model.modules()
    )


def load_adapter(
    model, path: str | Path, *, is_trainable: bool = False, cfg: LoRAConfig | None = None
):
    """在基座上挂载已训好的 adapter，`is_trainable` 默认 False。

    手写后端要求先 `inject_lora` 注入骨架；`is_trainable=True` 时断言可训练参数非零，
    防止续训时拿到空参数列表。
    """
    if _is_manual_model(model):
        from pepdesign.models.lora_layer import count_lora_params, load_manual_adapter

        model = load_manual_adapter(model, path, is_trainable=is_trainable)
        if is_trainable:
            n_trainable, _ = count_lora_params(model)
            if n_trainable == 0:
                raise RuntimeError(
                    f"以 is_trainable=True 加载了 {path}，但没有任何可训练参数。"
                )
        return model

    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover
        raise ImportError("加载 LoRA adapter 需要 peft：pip install -e '.[train]'") from exc

    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"adapter 目录不存在: {p}。"
            f"GRPO 阶段默认从 SFT 的产出继续训（GRPOConfig.sft_adapter_path），"
            f"如果还没跑过 SFT，请先把该字段改成本地已有的 adapter 路径。"
        )
    logger.info("加载 adapter（peft 后端）: %s（可训练=%s）", p, is_trainable)
    if isinstance(model, PeftModel):
        # 已经是 PeftModel：把权重加载进现有 adapter 槽位，
        # 再包一层会让后续保存与 resume 的 key 错位。
        model.load_adapter(
            str(p), adapter_name="default", is_trainable=is_trainable
        )
        peft_model = model
    else:
        peft_model = PeftModel.from_pretrained(
            model, str(p), is_trainable=is_trainable
        )

    if is_trainable:
        n_trainable, _ = _trainable_counts(peft_model)
        if n_trainable == 0:
            raise RuntimeError(
                f"以 is_trainable=True 加载了 {p}，但没有任何可训练参数。\n"
                f"  正常情况下这一条不会触发（参数就是上面那个开关控制的）。\n"
                f"  真触发了的话，检查 adapter_config.json 里的 "
                f"`inference_mode` 是否为 false，以及 adapter 文件是否完整。"
            )
    return peft_model


def merge_and_save(model, path: str | Path, tokenizer=None, cfg: LoRAConfig | None = None) -> Path:
    """把 adapter 合并进基座，导出成普通模型目录。

    合并是精确的矩阵加法，不删除原 adapter。手写后端会原地改动模型并 unwrap，
    之后不能继续训练；4-bit/8-bit 基座需先 dequantize。
    """
    out = resolve_path(path)
    out.mkdir(parents=True, exist_ok=True)

    if _is_manual_model(model):
        from pepdesign.models.lora_layer import (
            merge_manual_lora,
            unwrap_manual_lora,
        )

        merge_manual_lora(model)
        # 必须 unwrap：否则 save_pretrained 写出 ...q_proj.base.weight，
        # 标准 from_pretrained 会把目标层静默重新随机初始化。
        unwrap_manual_lora(model)
        model.save_pretrained(str(out))
        if tokenizer is not None:
            tokenizer.save_pretrained(str(out))
        logger.info("LoRA 已合并并保存（手写后端）: %s", out)
        return out

    if not hasattr(model, "merge_and_unload"):
        raise TypeError(
            "传入的模型既没有挂着 LoRALinear，也没有 merge_and_unload，"
            "说明它不是注入了 LoRA 的模型（可能是没注入的基座）。"
            "请先 inject_lora 或 load_adapter。"
        )
    merged = model.merge_and_unload()
    merged.save_pretrained(str(out))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(out))
    logger.info("LoRA 已合并并保存（peft 后端）: %s", out)
    return out
