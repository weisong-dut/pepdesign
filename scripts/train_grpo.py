"""阶段二：GRPO 强化学习（四维奖励）的入口。

用法::

    python scripts/train_grpo.py --config configs/grpo_lora.yaml
    python scripts/train_grpo.py --config configs/grpo_lora.yaml --dry-run
    python scripts/train_grpo.py --config configs/smoke_cpu.yaml --smoke

`--dry-run` 校验 prompt 集格式、奖励预测器与校准集，均不需要加载模型。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许直接以脚本方式运行（把 src 加入 sys.path）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pepdesign.utils.config import load_config, resolve_path, save_config
from pepdesign.utils.logger import get_logger, silence_noisy_libraries

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="阶段二：GRPO 强化学习",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default="configs/grpo_lora.yaml", help="YAML 配置文件路径")
    p.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="覆盖配置项，可重复。例：--set grpo.beta=0.1",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="只校验 prompt 集 / 预测器 / 校准集，不加载模型")
    p.add_argument("--smoke", action="store_true",
                   help="CPU 冒烟：随机初始化小模型跑几步完整 GRPO")
    p.add_argument("--steps", type=int, default=2,
                   help="--smoke 模式下跑多少步（默认 2）")
    return p.parse_args(argv)


def dry_run(cfg) -> int:
    """不加载模型的前置检查。返回进程退出码。"""
    from collections import Counter

    from pepdesign.train.dataset import load_grpo_prompts
    from pepdesign.train.rollout import expand_metas

    prompts = load_grpo_prompts(cfg.grpo.dataset_path)
    logger.info("GRPO prompt 集：%d 条，来自 %s", len(prompts), cfg.grpo.dataset_path)
    logger.info("任务分布：%s", dict(Counter(p["meta"]["task"] for p in prompts)))

    # 展开后的条数决定了显存占用：batch × G 条序列同时前向。
    # 这个数在训练日志里只体现为 OOM，所以提前算出来。
    n_seq = len(prompts) * cfg.grpo.num_generations
    logger.info(
        "采样规模：%d prompt × G=%d = 每轮 %d 条序列需同时前向"
        "（单条最长 %d token）。显存不够就先调小 per_device_train_batch_size 或 G。",
        len(prompts), cfg.grpo.num_generations, n_seq, cfg.grpo.max_completion_length,
    )

    # 展开顺序须符合 prompt-major 假设；顺序不一致时长度仍对得上，
    # 优势会串到其他 prompt 上且不报错，故在此显式验证。
    expanded = expand_metas([p["meta"] for p in prompts[:2]], cfg.grpo.num_generations)
    if len(expanded) != 2 * cfg.grpo.num_generations:
        logger.error("meta 展开长度不对：%d，期望 %d",
                     len(expanded), 2 * cfg.grpo.num_generations)
        return 1

    try:
        from pepdesign.rewards.composite import CompositeScorer

        scorer = CompositeScorer(cfg.reward)
        logger.info("四维奖励打分器加载完成：预设 %s，权重 %s",
                    scorer.cfg.preset, scorer.cfg.weights.normalized().model_dump())
    except Exception as exc:  # noqa: BLE001 - 预测器可能依赖未装的包
        logger.error("奖励打分器加载失败：%s: %s", type(exc).__name__, exc)
        return 1

    if cfg.grpo.calibration_enabled:
        try:
            from pepdesign.train.calibration import load_calibration_samples

            samples = load_calibration_samples(cfg)
            logger.info("校准集：%d 条样本，来自 %s", len(samples), cfg.grpo.calibration_path)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            return 1
    else:
        logger.warning(
            "校准预热已关闭。SFT 后的策略对'怎么改才更稳定'仍一无所知，"
            "组内 8 条采样的分数会挤在一起、优势接近 0、GRPO 拿不到梯度。"
            "除非有明确理由（比如做消融），否则建议打开 calibration_enabled。"
        )

    adapter = resolve_path(cfg.grpo.sft_adapter_path) if cfg.grpo.sft_adapter_path else None
    if adapter is None:
        logger.warning("sft_adapter_path 为空：GRPO 将从基座开始，跳过 SFT 阶段。")
    elif not adapter.exists():
        logger.error(
            "找不到 SFT adapter：%s。\n"
            "  阶段二必须从阶段一的产物继续。请先运行：\n"
            "      python scripts/train_sft.py --config configs/sft_lora.yaml\n"
            "  或把 grpo.sft_adapter_path 指向本地已有的 adapter 目录。",
            adapter,
        )
        return 1
    else:
        logger.info("SFT adapter：%s", adapter)

    out = save_config(cfg, Path(cfg.grpo.output_dir) / "resolved_config.yaml")
    logger.info("配置已落盘：%s", out)
    logger.info("dry-run 通过。去掉 --dry-run 开始训练。")
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
            from pepdesign.train.grpo import run_grpo_smoke

            report = run_grpo_smoke(cfg, steps=args.steps)

            # 数值一致性检查单独提出来报警：它是"两份损失实现没有漂移"
            # 的唯一保证，不通过说明代码本身有问题，而不是环境不对。
            check = report.get("loss_cross_check", {})
            if check and not check.get("agree", True):
                logger.error(
                    "损失函数的两份实现数值不一致：torch %.8f vs python %.8f。"
                    "train/grpo.py:grpo_loss 与 train/advantages.py 的纯函数版本"
                    "已经分家了，必须先修这个再看别的。",
                    check.get("torch"), check.get("python"),
                )
                return 3
        else:
            from pepdesign.train.grpo import run_grpo

            report = {"adapter_dir": str(run_grpo(cfg))}
    except ImportError as exc:
        logger.error("%s", exc)
        logger.error(
            "当前环境缺少训练依赖。可选的下一步：\n"
            "  * 只验证逻辑：python -m pytest -q（不需要 torch）\n"
            "  * CPU 冒烟：pip install torch transformers safetensors "
            "--index-url https://download.pytorch.org/whl/cpu\n"
            "  * 真实训练：换到有 GPU 的机器上 pip install -e '.[train]'\n"
            "（不需要 peft / trl / datasets：LoRA 和训练循环都是本项目自写的。）"
        )
        return 2

    for key, value in report.items():
        logger.info("%s: %s", key, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
