"""诊断 GRPO 生成格式，分 base / lora / adapter 三段定位问题所在层。

`base` 只加载基座，`lora` 注入未训练 LoRA，`adapter` 加载 SFT adapter；
输出在任一段崩坏即可定位问题层。`--set` 使用 `action="append"`，可重复。

用法::

    python scripts/diagnose_grpo_generation.py --stage all --n 1 \
        --set grpo.model_name_or_path=/path/to/logos-3b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pepdesign.models.loader import (  # noqa: E402
    load_base_model,
    load_tokenizer,
    probe_hardware,
    resolve_plan,
)
from pepdesign.models.lora import inject_lora, load_adapter  # noqa: E402
from pepdesign.train.dataset import load_grpo_prompts  # noqa: E402
from pepdesign.train.grpo import sample_completions  # noqa: E402
from pepdesign.utils.config import load_config  # noqa: E402


def _load_model(cfg):
    hw = probe_hardware()
    plan = resolve_plan(
        cuda_available=hw["cuda_available"],
        bf16_supported=hw["bf16_supported"],
        mps_available=hw["mps_available"],
        want_bf16=cfg.grpo.bf16,
        want_fp16=cfg.grpo.fp16,
        quant=cfg.quant,
    )
    model, _ = load_base_model(
        cfg.grpo.model_name_or_path, quant=cfg.quant,
        want_bf16=cfg.grpo.bf16, want_fp16=cfg.grpo.fp16, plan=plan,
    )
    return model


def _generate(model, tokenizer, prompts, cfg, *, autocast=False, greedy=False):
    import contextlib

    import torch

    saved_temp = cfg.grpo.temperature
    if greedy:
        cfg.grpo.temperature = 0.0
        # sample_completions 使用 do_sample=True；温度 0 在部分 transformers
        # 版本会报错或退化，故用一个极小温度近似贪心
        cfg.grpo.temperature = 1e-5
    ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if autocast and torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    try:
        with ctx:
            texts, _ids, gen_start, _lens = sample_completions(
                model, tokenizer, prompts, cfg
            )
    finally:
        cfg.grpo.temperature = saved_temp
    return texts, gen_start


def _format_ok(text: str) -> bool:
    return (
        "### 设计序列" in text
        and "### 设计理由" in text
        and "### 结束" in text
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="诊断 GRPO 生成格式")
    parser.add_argument("--config", default="configs/grpo_lora.yaml")
    parser.add_argument("--n", type=int, default=5, help="看几条 prompt")
    parser.add_argument(
        "--stage", choices=["base", "lora", "adapter", "all"], default="adapter",
        help="base=只加载基座；lora=注入未训练 LoRA；adapter=加载 SFT adapter；all=三段对比",
    )
    parser.add_argument(
        "--autocast", action="store_true",
        help="生成时套 bf16 autocast（与训练前向一致，用于验证精度不一致假设）",
    )
    parser.add_argument(
        "--greedy", action="store_true", help="贪心解码，排除采样随机性",
    )
    parser.add_argument(
        "--no-quant", action="store_true",
        help="用 bf16 加载基座（不量化），用于验证 4-bit + LoRA 的推理组合",
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="KEY=VALUE", help="覆盖配置项，可重复",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    if args.no_quant:
        cfg.quant.enabled = False
    print(f"模型    {cfg.grpo.model_name_or_path}")
    print(f"quant   {'bf16（不量化）' if args.no_quant else '4-bit'}")
    print(f"adapter {cfg.grpo.sft_adapter_path}")
    print(f"quant={cfg.quant}, bf16={cfg.grpo.bf16}, G={cfg.grpo.num_generations}, "
          f"max_new_tokens={cfg.grpo.max_completion_length}")

    tokenizer = load_tokenizer(cfg.grpo.model_name_or_path)
    prompts = load_grpo_prompts(cfg.grpo.dataset_path, limit=args.n)
    # 配置校验要求 num_generations >= 2；诊断时用 2 条足够对比格式
    cfg.grpo.num_generations = 2

    stages = ["base", "lora", "adapter"] if args.stage == "all" else [args.stage]
    model = None
    injected = False
    for stage in stages:
        if model is None:
            model = _load_model(cfg)
        # `lora` 与 `adapter` 都需要先注入；用标志位保证只注入一次
        # （重复注入会把 LoRALinear 套两层，输出直接错乱）
        if stage in ("lora", "adapter") and not injected:
            model = inject_lora(model, cfg.lora)
            injected = True
        if stage == "adapter":
            model = load_adapter(model, cfg.grpo.sft_adapter_path, is_trainable=False)

        texts, gen_start = _generate(
            model, tokenizer, prompts, cfg,
            autocast=args.autocast, greedy=args.greedy,
        )
        ok = sum(1 for t in texts if _format_ok(t))
        print("\n" + "=" * 40)
        print(f"[{stage}] 生成 {len(texts)} 条，三段完整 {ok}/{len(texts)}，"
              f"gen_start={gen_start}")
        for i, text in enumerate(texts):
            print(f"--- {stage} completion {i}（{len(text)} 字符）---")
            print(repr(text[:600]))
        if stage != "adapter":
            print(
                f"\n[{stage}] 这一段如果已经乱码，说明问题不在 adapter；"
                "如果正常，继续看下一段。"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
