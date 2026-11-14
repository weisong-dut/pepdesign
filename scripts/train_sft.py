"""阶段一：LoRA 监督微调的入口。

用法::

    python scripts/train_sft.py --config configs/sft_lora.yaml
    python scripts/train_sft.py --config configs/sft_lora.yaml --dry-run
    python scripts/train_sft.py --config configs/smoke_cpu.yaml --smoke

`--dry-run` 只校验配置与语料、不加载权重；`--smoke` 用随机初始化小模型
验证训练循环。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许直接以脚本方式运行（把 src 加入 sys.path）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pepdesign.utils.config import load_config, save_config
from pepdesign.utils.logger import get_logger, silence_noisy_libraries

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="阶段一：LoRA 监督微调",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default="configs/sft_lora.yaml", help="YAML 配置文件路径")
    p.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="覆盖配置项，可重复。例：--set sft.learning_rate=1e-4",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="只校验配置与语料、跑长度预检，不加载模型")
    p.add_argument("--smoke", action="store_true",
                   help="CPU 冒烟：随机初始化小模型跑几步，验证链路通")
    p.add_argument("--steps", type=int, default=3,
                   help="--smoke 模式下跑多少步（默认 3）")
    return p.parse_args(argv)


def dry_run(cfg) -> int:
    """不加载模型的前置检查。返回进程退出码。

    LoRA 参数量由 `lora_trainable_summary` 按模型真实结构逐项计算，
    可直接用于发现 target_modules 配错或漏层。
    """
    from pepdesign.models.lora import lora_trainable_summary, validate_lora_config
    from pepdesign.train.dataset import (
        estimate_sft_num_training_steps,
        load_sft_samples,
    )

    validate_lora_config(cfg.lora)
    summary = lora_trainable_summary(cfg.lora)
    logger.info(
        "LoRA：backend=%s r=%d alpha=%d 缩放=%.2f 目标模块=%s",
        summary["backend"], summary["r"], summary["alpha"], summary["scaling"],
        summary["target_modules"],
    )
    logger.info(
        "可训练参数 %s / 总计 %s (%.4f%%%s)，bf16 adapter 约 %.1f MiB",
        f"{summary['estimated_trainable_params']:,}",
        f"{summary['total_params']:,}",
        100 * summary["estimated_share"],
        "" if not summary["unknown_target_modules"]
        else f"，不认识的模块 {summary['unknown_target_modules']}",
        summary["adapter_bytes_bf16"] / 1024 / 1024,
    )

    rows = load_sft_samples(cfg.sft.dataset_path)
    logger.info("SFT 语料：%d 条，来自 %s", len(rows), cfg.sft.dataset_path)

    from collections import Counter
    logger.info("任务分布：%s", dict(Counter(r["task_type"] for r in rows)))

    n_steps = estimate_sft_num_training_steps(cfg.sft, len(rows))
    logger.info(
        "计划 %d 个 optimizer step（batch %d × 累积 %d，%s 个 epoch）",
        n_steps, cfg.sft.per_device_train_batch_size,
        cfg.sft.gradient_accumulation_steps, cfg.sft.num_train_epochs,
    )

    # 长度预检需要下载 tokenizer（十几 MB）；取不到时跳过，
    # 不因此中断已经完成的配置与语料检查。
    try:
        from pepdesign.models.loader import load_tokenizer
        from pepdesign.train.sft import length_preflight

        tokenizer = load_tokenizer(cfg.sft.model_name_or_path)
        length_preflight(cfg, tokenizer, rows, sample_n=min(64, len(rows)))
    except Exception as exc:  # noqa: BLE001 - 预检是可选的，失败不应中断
        logger.warning("长度预检跳过（%s: %s）", type(exc).__name__, exc)

    out = save_config(cfg, Path(cfg.sft.output_dir) / "resolved_config.yaml")
    logger.info("配置已落盘：%s", out)
    logger.info("dry-run 通过。去掉 --dry-run 开始真实训练。")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    silence_noisy_libraries()
    cfg = load_config(args.config, overrides=args.overrides)
    logger.info("配置加载完成：%s（覆盖 %d 项）", args.config, len(args.overrides))

    if args.dry_run:
        return dry_run(cfg)

    try:
        if args.smoke:
            from pepdesign.train.sft import run_sft_smoke

            report = run_sft_smoke(cfg, steps=args.steps)
        else:
            from pepdesign.train.sft import run_sft

            report = {"adapter_dir": str(run_sft(cfg))}
    except ImportError as exc:
        # 依赖缺失属于预期情况；直接抛栈会与代码缺陷混淆，
        # 这里给出明确的安装或替代验证指引。
        logger.error("%s", exc)
        logger.error(
            "当前环境缺少训练依赖。可选的下一步：\n"
            "  * 只验证逻辑：python -m pytest -q（不需要 torch）\n"
            "  * CPU 冒烟：pip install torch transformers safetensors "
            "--index-url https://download.pytorch.org/whl/cpu\n"
            "  * 真实训练：换到有 GPU 的机器上 pip install -e '.[train]'\n"
            "（不需要 peft / trl / datasets：LoRA 和训练循环都是本项目自写的。"
            "peft 只在做数值对照测试时才用。）"
        )
        return 2

    for key, value in report.items():
        logger.info("%s: %s", key, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
