"""checkpoint 的命名、保留、状态读写，以及两个阶段共用的步末收尾。

`save_steps` / `save_total_limit` / `resume_from_checkpoint` / `eval_steps` 的
唯一消费点在此模块。顶层不 import torch：命名、保留策略与状态读写为纯逻辑，
可直接进快速测试；仅 `CheckpointManager.save` / `load_into` 内部导入 torch。

`trainer_state.json` 最后写入，是目录写完整的提交标记。续训恢复权重、优化器、
调度器、step、history 与数据位置，不恢复 RNG，因此不保证逐位可复现。
"""

from __future__ import annotations

import dataclasses
import math
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pepdesign.utils.config import TrainCommonConfig
from pepdesign.utils.io import write_json
from pepdesign.utils.logger import get_logger

logger = get_logger(__name__)


CHECKPOINT_PREFIX = "checkpoint-"
#: 提交标记：只有写到它，这个目录才算一个完整可用的 checkpoint。
STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"

#: fingerprint 中不一致即报错的键：它们改变后"从第 N 步接着跑"在语义上不成立
#: （步数决定 warmup 与 cosine 衰减点；batch/长度改变同一步对应的数据；phase 必须一致）。
_FATAL_KEYS: tuple[str, ...] = (
    "phase", "num_steps",
    "per_device_train_batch_size", "gradient_accumulation_steps",
    "max_seq_length", "max_prompt_length", "max_completion_length",
    "num_generations",
)

#: fingerprint 中由配置决定、只记日志的键：变化不影响续训语义，但影响结果，需留痕。
_CONFIG_WINS_KEYS: tuple[str, ...] = (
    "learning_rate", "lr_scheduler_type", "warmup_ratio",
    "weight_decay", "max_grad_norm", "seed",
)


# ---------------------------------------------------------------------------
# 命名与保留（纯逻辑）
# ---------------------------------------------------------------------------
def should_fire(step: int, every: int) -> bool:
    """判断 `step` 是否为 `every` 的整数倍，评估与保存共用。

    `every <= 0` 或 `step <= 0` 时返回 False。
    """
    if every <= 0 or step <= 0:
        return False
    return step % every == 0


def checkpoint_dir(output_dir: str | Path, step: int) -> Path:
    """返回 `<output_dir>/checkpoint-<step>`；命名集中于此，别处不要拼接。"""
    return Path(output_dir) / f"{CHECKPOINT_PREFIX}{int(step)}"


def parse_checkpoint_step(path: Path) -> int | None:
    """从目录名解析步号；不是 `checkpoint-<整数>` 时返回 None。"""
    name = path.name
    if not name.startswith(CHECKPOINT_PREFIX):
        return None
    tail = name[len(CHECKPOINT_PREFIX):]
    return int(tail) if tail.isdigit() else None


def list_checkpoints(output_dir: str | Path) -> list[tuple[int, Path]]:
    """列出 `output_dir` 下所有完整 checkpoint，按步号升序。

    仅接受含 `trainer_state.json` 的目录；按步号整数排序，而非目录名字符串。
    """
    root = Path(output_dir)
    if not root.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        step = parse_checkpoint_step(child)
        if step is None:
            continue
        if not (child / STATE_NAME).exists():
            logger.debug("忽略未写完的 checkpoint（缺 %s）：%s", STATE_NAME, child)
            continue
        found.append((step, child))
    return sorted(found, key=lambda item: item[0])


def select_stale_checkpoints(
    existing: Sequence[tuple[int, Path]], save_total_limit: int
) -> list[Path]:
    """返回应删除的旧 checkpoint，保留最新的 `save_total_limit` 个。

    `existing` 须为 `list_checkpoints` 的输出；只返回路径，不执行删除。
    """
    if save_total_limit <= 0:
        return []
    if len(existing) <= save_total_limit:
        return []
    return [path for _, path in existing[: len(existing) - save_total_limit]]


def resolve_resume_path(resume_from: str | None, output_dir: str | Path) -> Path | None:
    """把 `resume_from_checkpoint` 解析为可续训目录。

    接受 None/空串（返回 None）、`"latest"` 或具体路径；找不到时抛错，
    不会静默从头训练。
    """
    if resume_from is None or not str(resume_from).strip():
        return None

    text = str(resume_from).strip()
    if text.lower() == "latest":
        found = list_checkpoints(output_dir)
        if not found:
            raise FileNotFoundError(
                f"resume_from_checkpoint='latest'，但 {Path(output_dir)} 下一个完整的 "
                f"checkpoint 都没有（完整 = 含 {STATE_NAME}）。\n"
                f"常见原因：`save_steps` 是 0（等于从不保存），"
                f"或者它比总步数还大——那样这个 checkpoint 永远不会产生。"
            )
        step, path = found[-1]
        logger.info("resume_from_checkpoint='latest' → 选中最新的 checkpoint-%d", step)
        return path

    path = Path(text)
    if not path.exists():
        raise FileNotFoundError(
            f"resume_from_checkpoint 指向的路径不存在：{path}。\n"
            f"（它应当是某个 `checkpoint-<N>` 目录，或者字符串 'latest'。）"
        )
    if not (path / STATE_NAME).exists():
        raise ValueError(
            f"{path} 不是可续训的 checkpoint：缺 {STATE_NAME}。\n"
            f"两种可能：\n"
            f"  ①它是 `final/`——`final/` 只有权重，没有优化器和调度器状态，"
            f"续训必须从 checkpoint-<N> 出发（否则 lr 曲线会从 warmup 重来、"
            f"Adam 的二阶矩会从 0 重新累积，而这两件事都不会报错）；\n"
            f"  ②它是一个写到一半的残目录（保存时崩溃），换个更早的 checkpoint。"
        )
    return path


# ---------------------------------------------------------------------------
# 状态文件
# ---------------------------------------------------------------------------
def _jsonable(obj: Any) -> Any:
    """把待落盘对象转换为可 JSON 序列化的形状。

    dataclass 转为 dict；非有限浮点转为 None，保留字段语义。
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_trainer_state(
    directory: str | Path,
    *,
    step: int,
    total_steps: int,
    phase: str,
    history: Any = None,
    fingerprint: Mapping[str, Any] | None = None,
) -> Path:
    """写 `trainer_state.json`，必须在 checkpoint 的最后一步调用。

    写入走临时文件 + `os.replace`，断电不会留下半截 JSON；该文件存在即表示目录完整。
    """
    payload = {
        "step": int(step),
        "total_steps": int(total_steps),
        "phase": phase,
        "history": _jsonable(history) if history is not None else None,
        "fingerprint": dict(fingerprint or {}),
    }
    return write_json(Path(directory) / STATE_NAME, payload)


def read_trainer_state(path: str | Path) -> dict:
    """读 `trainer_state.json`。`path` 可以是目录，也可以是文件本身。"""
    p = Path(path)
    if p.is_dir():
        p = p / STATE_NAME
    if not p.exists():
        raise FileNotFoundError(
            f"找不到 {p}。它缺失说明这个目录不是一个完整的 checkpoint："
            f"要么是写到一半崩了，要么指向的是 `final/`（`final/` 只有权重）。"
        )
    import json

    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "step" not in data:
        raise ValueError(f"{p} 的内容不是合法的 trainer_state（缺 step 字段）：{data!r}")
    return data


# ---------------------------------------------------------------------------
# fingerprint：这次训练"是什么"
# ---------------------------------------------------------------------------
def training_fingerprint(
    cfg: TrainCommonConfig, *, phase: str, num_steps: int
) -> dict[str, Any]:
    """把本次训练的可比配置压成字典，随 checkpoint 一起落盘。

    用于续训时判断配置是否仍属于同一次训练。
    """
    record: dict[str, Any] = {"phase": phase, "num_steps": int(num_steps)}
    for key in _FATAL_KEYS + _CONFIG_WINS_KEYS:
        if key in record:
            continue
        record[key] = getattr(cfg, key, None)
    # fingerprint 记录最终生效的学习率：SFT 有 lr 覆盖 learning_rate 的规则，
    # 否则 `--set sft.lr=...` 在续训时会被认为未变化。
    resolved = getattr(cfg, "resolved_lr", None)
    if callable(resolved):
        record["learning_rate"] = resolved()
    return record


def compare_fingerprint(
    recorded: Mapping[str, Any], current: Mapping[str, Any]
) -> tuple[list[str], list[str]]:
    """比较两份 fingerprint，返回 `(致命差异, 配置决定的差异)`。

    致命差异在语义上不再是同一次训练；配置决定的差异需打日志留痕。
    """
    critical: list[str] = []
    config_wins: list[str] = []
    for key in _FATAL_KEYS:
        was, now = recorded.get(key), current.get(key)
        if was != now:
            critical.append(f"{key}: checkpoint 里是 {was!r}，现在是 {now!r}")
    for key in _CONFIG_WINS_KEYS:
        was, now = recorded.get(key), current.get(key)
        if was != now:
            config_wins.append(f"{key}: {was!r} → {now!r}")
    return critical, config_wins


# ---------------------------------------------------------------------------
# 读写一个 checkpoint
# ---------------------------------------------------------------------------
class CheckpointManager:
    """一个实验目录下 checkpoint 的读写与保留。

    构造时要求 `lora_cfg`：手写后端下 `save_adapter` 缺少它会抛错，作为构造参数
    可避免保存路径漏传。
    """

    def __init__(
        self,
        *,
        output_dir: str | Path,
        save_total_limit: int,
        lora_cfg,
        phase: str,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.save_total_limit = int(save_total_limit)
        self.lora_cfg = lora_cfg
        self.phase = phase

    def save(
        self,
        step: int,
        *,
        total_steps: int,
        model,
        tokenizer,
        optimizer,
        scheduler,
        history: Any = None,
        fingerprint: Mapping[str, Any] | None = None,
    ) -> Path:
        """存一个 checkpoint，然后按 `save_total_limit` 清理旧的。"""
        import torch

        from pepdesign.models.lora import save_adapter

        directory = checkpoint_dir(self.output_dir, step)
        if directory.exists() and not (directory / STATE_NAME).exists():
            # 上一次写到一半的残目录：内容会被整体覆写，但需要记录。
            logger.warning("checkpoint-%d 是个未写完的残目录，本次覆写它。", step)
        directory.mkdir(parents=True, exist_ok=True)

        save_adapter(model, directory, tokenizer, self.lora_cfg)
        torch.save(optimizer.state_dict(), directory / OPTIMIZER_NAME)
        torch.save(scheduler.state_dict(), directory / SCHEDULER_NAME)
        # 最后写：该文件存在即表示目录完整。
        write_trainer_state(
            directory,
            step=step,
            total_steps=total_steps,
            phase=self.phase,
            history=history,
            fingerprint=fingerprint,
        )
        logger.info("已保存 checkpoint-%d → %s", step, directory)

        stale = select_stale_checkpoints(
            list_checkpoints(self.output_dir), self.save_total_limit
        )
        for old in stale:
            shutil.rmtree(old, ignore_errors=True)
            # 删除必须记录日志：静默删除会让续训找不到可用 checkpoint。
            logger.info(
                "按 save_total_limit=%d 清理旧 checkpoint：%s",
                self.save_total_limit, old.name,
            )
        return directory

    def load_into(
        self,
        path: str | Path,
        *,
        model,
        optimizer,
        scheduler,
        expected_fingerprint: Mapping[str, Any] | None = None,
        learning_rate: float | None = None,
    ) -> dict:
        """把 `path` 中的状态加载进 model / optimizer / scheduler，返回 trainer_state。

        顺序：先读 state 并比较 fingerprint（在修改模型前报错），再加载权重、
        优化器与调度器，最后按配置调和学习率并校验 `last_epoch`。
        """
        import torch

        from pepdesign.models.lora import load_adapter

        directory = Path(path)
        state = read_trainer_state(directory)
        started = int(state["step"])
        total_steps = state.get("total_steps")

        recorded = state.get("fingerprint") or {}
        if recorded and expected_fingerprint:
            critical, config_wins = compare_fingerprint(recorded, expected_fingerprint)
            if critical:
                raise ValueError(
                    "这个 checkpoint 与当前配置不是同一件事，不能接着跑：\n  "
                    + "\n  ".join(critical)
                    + "\n两条修法：①把配置改回 checkpoint 里的值；"
                    "②去掉 resume_from_checkpoint，当作一次新训练重跑。"
                )
            if config_wins:
                logger.warning(
                    "续训：以下超参和 checkpoint 里记录的不一致，按当前配置执行：%s",
                    "；".join(config_wins),
                )
        elif expected_fingerprint and total_steps not in (
            None, expected_fingerprint.get("num_steps")
        ):
            # 老 checkpoint 没有 fingerprint 时的降级路径：至少比较总步数。
            raise ValueError(
                f"checkpoint 记录的总步数是 {total_steps}，"
                f"当前配置估出来的是 {expected_fingerprint.get('num_steps')}。"
                f"总步数决定 warmup 与 cosine 的衰减点，不一致时续的是另一条曲线，"
                f"而 loss 曲线看起来完全正常。"
            )

        load_adapter(model, directory, is_trainable=True)

        opt_path, sch_path = directory / OPTIMIZER_NAME, directory / SCHEDULER_NAME
        for p, what in ((opt_path, "优化器"), (sch_path, "调度器")):
            if not p.exists():
                raise FileNotFoundError(
                    f"{directory} 缺 {p.name}，无法恢复{what}状态。"
                    f"只恢复权重的续训会让二阶矩从 0 重新累积、lr 曲线从 warmup 重来——"
                    f"那是另一个实验，不是续训。请从 checkpoint-<N> 续训。"
                )
        optimizer.load_state_dict(torch.load(opt_path, map_location="cpu", weights_only=False))
        scheduler.load_state_dict(torch.load(sch_path, map_location="cpu", weights_only=False))

        last_epoch = getattr(scheduler, "last_epoch", None)
        if last_epoch is not None and int(last_epoch) != started:
            raise AssertionError(
                f"调度器的 last_epoch={last_epoch} 与记录的 step={started} 对不上，"
                f"续训后学习率会从一个错误的位置接着走。"
                f"这通常意味着 checkpoint 不是在一个 optimizer step 的边界上存的。"
            )

        # ---- 学习率：由配置决定 ----
        # `LambdaLR.load_state_dict` 会恢复 base_lrs，覆盖成当前配置值后
        # 需重算一次当前 lr，相当于把整条曲线按新 lr 重新标定。
        if learning_rate is not None:
            base_lrs = getattr(scheduler, "base_lrs", None)
            if base_lrs and any(abs(float(b) - float(learning_rate)) > 1e-12 for b in base_lrs):
                logger.warning(
                    "续训：学习率按当前配置执行（checkpoint 里是 %s，现在是 %s）。"
                    "调度器曲线会按新的 base_lr 整体重新标定。",
                    base_lrs[0], learning_rate,
                )
                scheduler.base_lrs = [float(learning_rate)] * len(base_lrs)
                # `last_epoch -= 1` 再 `step()`：用公开 API 重算当前 lr，不碰内部字段。
                scheduler.last_epoch = started - 1
                scheduler.step()

        logger.info(
            "已从 %s 续训：step=%d / %s，优化器与调度器状态均已恢复"
            "（RNG 未恢复，所以不保证逐位可复现）。",
            directory, started, total_steps,
        )
        return state


# ---------------------------------------------------------------------------
# 步末收尾：两个阶段共用的唯一一份判断
# ---------------------------------------------------------------------------
def on_step_end(
    *,
    step: int,
    total_steps: int,
    eval_steps: int,
    save_steps: int,
    writer=None,
    scalars: Mapping[str, float] | None = None,
    evaluate: Callable[[], float | None] | None = None,
    save: Callable[[], Any] | None = None,
) -> dict[str, float]:
    """一个 optimizer step 结束时的收尾：评估 → 写标量 → 存 checkpoint。

    `step` 为已完成的 optimizer step 数，从 1 开始。`evaluate` / `save` 为无参回调，
    只在本该触发时调用；先评估再保存，使 history 含本步 eval_loss。
    """
    merged: dict[str, float] = dict(scalars or {})

    # 最后一步不评估：循环结束后调用方还会再算一次 eval_loss。
    if (
        evaluate is not None
        and should_fire(step, eval_steps)
        and step < total_steps
    ):
        value = evaluate()
        if value is not None:
            merged["eval_loss"] = float(value)
            logger.info("step %d/%d | eval_loss %.4f", step, total_steps, value)

    if writer is not None:
        writer.add_scalars(merged, step)

    if save is not None and should_fire(step, save_steps):
        save()

    return merged
