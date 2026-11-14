"""从 GRPO 训练日志与 SFT TensorBoard 事件文件绘制训练曲线。

用法::

    python scripts/plot_training_curves.py --grpo-log logs/grpo_full.log \
        --sft-tb outputs/sft/tensorboard --out outputs/plots/training_curves.png

GRPO 直接解析日志，SFT 需要 tensorboard 包；绘图需要 matplotlib。
图使用英文标签，避免无中文字体时显示为方块。
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 每步日志的格式：
#: step 15 | loss -0.0313 | reward 0.1957 (std 0.1127) | kl 0.0093 | ...
#: ... | 解析失败 0.0% (弱解析 0.0%) | 退化组 1 | grad 0.774 | 118.4s
_STEP_RE = re.compile(
    r"step\s+(\d+)\s*\|.*?"
    r"loss\s+([+-]?\d+(?:\.\d+)?).*?"
    r"reward\s+([+-]?\d+(?:\.\d+)?)\s*\(std\s+([\d.]+)\).*?"
    r"kl\s+([+-]?\d+(?:\.\d+)?).*?"
    r"退化组\s+(\d+)"
)


def parse_grpo_log(path: Path) -> dict[str, list[float]]:
    steps: list[int] = []
    losses: list[float] = []
    rewards: list[float] = []
    stds: list[float] = []
    kls: list[float] = []
    degenerates: list[float] = []
    if not path.exists():
        raise FileNotFoundError(f"找不到 GRPO 日志：{path}")
    for line in path.read_text(encoding="utf8", errors="ignore").splitlines():
        if "step" not in line or "reward" not in line:
            continue
        m = _STEP_RE.search(line)
        if not m:
            continue
        steps.append(int(m.group(1)))
        losses.append(float(m.group(2)))
        rewards.append(float(m.group(3)))
        stds.append(float(m.group(4)))
        kls.append(float(m.group(5)))
        degenerates.append(float(m.group(6)))
    if not steps:
        raise RuntimeError(f"{path} 里没解析出任何 step 行")
    return {
        "steps": steps, "loss": losses, "reward": rewards,
        "reward_std": stds, "kl": kls, "degenerate": degenerates,
    }


def read_sft_tensorboard(path: Path) -> dict[str, tuple[list[int], list[float]]]:
    """读取 SFT 的标量。没有 tensorboard 包或目录时返回空 dict。"""
    if not path.exists():
        return {}
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        print("没有 tensorboard 包，跳过 SFT 曲线（pip install tensorboard）")
        return {}
    ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
    ea.Reload()
    out: dict[str, tuple[list[int], list[float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        out[tag] = ([e.step for e in events], [e.value for e in events])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="画训练曲线")
    p.add_argument("--grpo-log", default="logs/grpo_full.log")
    p.add_argument("--sft-tb", default="outputs/sft/tensorboard")
    p.add_argument("--out", default="outputs/plots/training_curves.png")
    p.add_argument("--window", type=int, default=10, help="滑动平均窗口（0=不平滑）")
    args = p.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("需要 matplotlib：pip install matplotlib")
        return 1

    def smooth(values, window):
        if window <= 1 or len(values) < window:
            return values
        out = []
        for i in range(len(values)):
            lo = max(0, i - window + 1)
            out.append(sum(values[lo:i + 1]) / (i - lo + 1))
        return out

    grpo = parse_grpo_log(Path(args.grpo_log))
    sft = read_sft_tensorboard(Path(args.sft_tb))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training curves", fontsize=14)

    # 左上：SFT loss（train / eval）
    ax = axes[0][0]
    drew = False
    for tag, style in (("sft/loss", "-"), ("sft/eval_loss", "o-")):
        if tag in sft:
            xs, ys = sft[tag]
            ax.plot(xs, ys, style, label=tag.split("/")[-1], markersize=3)
            drew = True
    if not drew:
        ax.text(0.5, 0.5, "SFT tensorboard not found", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title("SFT loss")
    ax.set_xlabel("step")
    ax.legend()
    ax.grid(alpha=0.3)

    # 右上：GRPO reward（原始 + 滑动平均，std 阴影）
    ax = axes[0][1]
    steps = grpo["steps"]
    reward = grpo["reward"]
    std = grpo["reward_std"]
    ax.plot(steps, reward, alpha=0.25, color="tab:blue", label="reward")
    ax.fill_between(steps,
                    [r - s for r, s in zip(reward, std, strict=True)],
                    [r + s for r, s in zip(reward, std, strict=True)],
                    alpha=0.15, color="tab:blue", label="±1 std")
    ax.plot(steps, smooth(reward, args.window), color="tab:blue",
            label=f"reward (MA{args.window})")
    ax.set_title("GRPO reward")
    ax.set_xlabel("step")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 左下：KL
    ax = axes[1][0]
    ax.plot(steps, grpo["kl"], color="tab:orange", label="kl")
    ax.set_title("GRPO KL")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)

    # 右下：退化组
    ax = axes[1][1]
    ax.plot(steps, grpo["degenerate"], color="tab:red", label="degenerate groups")
    ax.set_title("GRPO degenerate groups (of 16)")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"已保存：{out}")
    print(f"GRPO 共 {len(steps)} 步；reward {reward[0]:.4f} → {reward[-1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
