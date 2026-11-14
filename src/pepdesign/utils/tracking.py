"""训练指标的外送通道（目前是 tensorboard）。

独立模块而非并入 logger.py
------------------------------------------------
`logger.py` 管的是文本日志，而且它被 `config.py` 在导入期就拉进来了；
这里是数值指标的外送，带一个重依赖（torch）和一条必须优雅降级的路。
两件事的失败模式完全不同：日志挂了是看不见信息，指标后端挂了不该影响训练。

所以这个模块的顶层不 import torch，重依赖在 `_load_summary_writer_factory`
里现导——`config.py` 的 `report_to` 校验器要 import 它，而 `config.py` 不能被
torch 绑架（整个快速测试套件都跑在没有 torch 的环境里）。

它为什么存在：`report_to` 这个配置字段在很长一段时间里没有任何消费方。
`pyproject.toml` 里甚至写着 `"tensorboard>=2.16", # report_to: [tensorboard] 的消费方`
——而那个消费方并不存在。凡是"配了不生效"的字段，要么接线，要么删掉；
`report_to` 值得接线，因为"训练曲线落在哪"本来就是这个项目声称的复现证据之一。

降级规则（这条比功能本身重要）
------------------------------
拿不到 tensorboard 时：只警告一次，然后整个 writer 变成 no-op。
训练继续跑，一个标量都不写。理由很直接——指标外送是观测手段，
它挂掉不该让一次跑了几小时的训练失败。反过来，一个"因为没装可视化库所以训练
起不来"的项目，在这个字段上是诚实的、在可用性上是不可用的。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pepdesign.utils.logger import PROJECT_ROOT, get_logger

logger = get_logger(__name__)


def _resolve_path(p: str | Path) -> Path:
    """把相对路径解析到项目根目录下。

    这里自己写一行而不复用 `utils.config.resolve_path`，是为了避开一个
    循环导入：`config.py` 要在 `report_to` 的校验器里 import 本模块
    （那样"后端名写错"才能在加载配置时报错），本模块再回头 import 它就是环。
    规则与 `config.resolve_path` 完全一致（相对路径一律以项目根为基准，
    不依赖 cwd），改那边的时候这里要跟着看一眼。
    """
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


#: 支持的后端名 → 一句话说明。
#: 加新后端时在这里加，不要在 `if/elif` 里加——这个名字表同时被配置校验器
#: 用来报"合法取值有哪些"，漏了的话报错信息会把用户指向一个不存在的选项。
REPORT_BACKENDS: dict[str, str] = {
    "tensorboard": "torch.utils.tensorboard.SummaryWriter，事件写到 <output_dir>/tensorboard/",
}

#: 已经警告过的失败，避免每个 step 刷一条同样的日志。
#: 用模块级可变集合而不是实例属性：同一台机器上 SFT 和 GRPO 各构造一个 writer，
#: "没装 tensorboard"这件事只需要说一次。
_WARNED: set[str] = set()


def resolve_backends(report_to: Sequence[str]) -> tuple[str, ...]:
    """规范化 `report_to`：去空白、转小写、去重、保持声明顺序。

    不认识的后端名直接报错，而不是忽略。这是和 pydantic 的
    `extra="forbid"` 同一个态度：`report_to: [wandb]` 这种拼写如果在运行时被
    静默忽略，就等于又造了一个"配了不生效"的字段——而这类字段正是本项目
    一直在清理的东西。报错信息里必须列出合法取值，否则使用者只能去读源码。
    """
    out: dict[str, None] = {}
    for raw in report_to:
        name = str(raw).strip().lower()
        if not name:
            continue
        if name not in REPORT_BACKENDS:
            raise ValueError(
                f"未知的 report_to 后端 {raw!r}，可选：{sorted(REPORT_BACKENDS)}。"
                f"（不支持的后端名不能静默忽略——那会让配置里写着的东西不生效，"
                f"而日志上完全看不出来。要加新后端请在 utils/tracking.py 的 "
                f"REPORT_BACKENDS 里注册。）"
            )
        out.setdefault(name, None)
    return tuple(out)


def _warn_once(key: str, msg: str, *args: Any) -> None:
    """同一个原因只警告一次。训练循环里每个 step 都会走到这些分支。"""
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(msg, *args)


def _load_summary_writer_factory() -> Any:
    """惰性导入 `torch.utils.tensorboard.SummaryWriter`。这是测试的注入点。

    为什么 `try/except` 要包住导入而不是构造：没装 tensorboard 包时，
    `torch/utils/tensorboard/__init__.py` 第一行就是 `import tensorboard`，
    所以异常发生在导入期，根本走不到构造那一步。
    包错地方的写法（把 try 包在 `SummaryWriter(...)` 外面）在缺包的环境里
    会直接抛 ImportError，而不是降级。
    """
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter


class MetricWriter:
    """把 `{标量名: 值}` 写进配置指定的后端。拿不到后端就变成 no-op。

    用法::

        writer = MetricWriter.from_config(cfg.sft.report_to,
                                          output_dir=cfg.sft.output_dir, prefix="sft")
        writer.add_scalars({"loss": 1.23, "lr": 2e-4}, step=10)
        writer.close()

    `prefix` 是阶段名（`"sft"` / `"grpo"`），最终标签是 `sft/loss`。
    带前缀不是因为好看：两阶段的标量名有重叠（两边都有 `loss`），
    两阶段又常常共用同一个 output 根目录，撞在一起时 tensorboard 会把它们
    画成同一条曲线——看起来像是"loss 突然掉了一截"，而不是两条曲线。
    """

    def __init__(
        self,
        *,
        backends: Sequence[str] = (),
        prefix: str = "",
        log_dir: str | Path | None = None,
    ) -> None:
        self.prefix = prefix
        self.log_dir = _resolve_path(log_dir) if log_dir is not None else None
        self._writers: dict[str, Any] = {}
        self._disabled_reasons: dict[str, str] = {}

        for name in resolve_backends(backends):
            if name == "tensorboard":
                self._init_tensorboard()
            else:  # pragma: no cover - resolve_backends 已经挡住了
                raise AssertionError(f"REPORT_BACKENDS 里有 {name} 但没有对应的分支")

    # -- 构造 ---------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        report_to: Sequence[str],
        *,
        output_dir: str | Path,
        prefix: str = "",
    ) -> MetricWriter:
        """按配置构造。`log_dir` 统一放在 `<output_dir>/tensorboard/`。

        和 adapter / resolved_config 放同一个实验目录，是为了让"这次实验的产物"
        是一个目录——排查时不需要先问"日志到底在哪儿"。
        """
        return cls(
            backends=report_to,
            prefix=prefix,
            log_dir=Path(output_dir) / "tensorboard",
        )

    def _init_tensorboard(self) -> None:
        try:
            factory = _load_summary_writer_factory()
        except ImportError as exc:
            reason = (
                "tensorboard 不可用（report_to 里写了它，但导入失败："
                f"{type(exc).__name__}: {exc}）。"
                "训练继续，本次不写任何标量。"
                "要启用请装：pip install -e '.[train]'（或 pip install tensorboard）。"
            )
            self._disabled_reasons["tensorboard"] = reason
            _warn_once("tensorboard-import", reason)
            return

        try:
            # 只在真的要写的时候才建目录：禁用的 writer 不该在磁盘上留下
            # 一个空的 tensorboard/，那会让"这次跑到底写没写指标"变成一个
            # 要去看时间戳才能回答的问题。
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._writers["tensorboard"] = factory(str(self.log_dir))
        except Exception as exc:  # noqa: BLE001 - 后端初始化失败不该中断训练
            reason = (
                f"tensorboard 初始化失败（{type(exc).__name__}: {exc}）。"
                f"训练继续，本次不写任何标量。目录：{self.log_dir}"
            )
            self._disabled_reasons["tensorboard"] = reason
            _warn_once("tensorboard-init", reason)

    # -- 状态 ---------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """是否真的有后端在写。冒烟测试和脚本用它决定要不要打那行日志。"""
        return bool(self._writers)

    @property
    def disabled_reasons(self) -> dict[str, str]:
        """后端名 → 为什么没启用。配置里写了却没生效时，这里是那个"为什么"。"""
        return dict(self._disabled_reasons)

    # -- 写入 ---------------------------------------------------------------
    def add_scalar(self, name: str, value: float, step: int) -> None:
        self.add_scalars({name: value}, step)

    def add_scalars(self, values: Mapping[str, float], step: int) -> None:
        """一次写一组标量。非有限值会被跳过，见下。

        NaN / inf 写进 tensorboard 是合法的，但它的实际效果是那条曲线从那一
        步开始整段消失（tensorboard 前端的平滑会把整个窗口一起干掉），
        于是诊断信息比"没记这个点"还少。SFT 循环遇到非有限的 loss 会直接抛错，
        GRPO 侧的 `grad_norm` 倒是有可能在某些 batch 上是 inf——
        与其让曲线神秘消失，不如记下名字、跳过这个点。
        """
        if not self._writers:
            return
        tag_values: dict[str, float] = {}
        skipped: list[str] = []
        for name, value in values.items():
            try:
                fvalue = float(value)
            except (TypeError, ValueError):
                skipped.append(name)
                continue
            if not math.isfinite(fvalue):
                skipped.append(name)
                continue
            tag_values[f"{self.prefix}/{name}" if self.prefix else name] = fvalue

        if skipped:
            _warn_once(
                f"nonfinite-{','.join(sorted(skipped))}",
                "这些标量出现了非有限值，已跳过（NaN/inf 写进 tensorboard 会让"
                "整条曲线从该点起消失，比不记这个点更难看出来）：%s",
                sorted(skipped),
            )

        # 遍历副本：下面写失败时会把该后端从 `_writers` 里摘掉，
        # 直接迭代原字典会 `RuntimeError: dictionary changed size during iteration`。
        # （这条是测试先发现的——写失败路径没有它就跑不到。）
        for name, writer in list(self._writers.items()):
            try:
                for tag, fvalue in tag_values.items():
                    writer.add_scalar(tag, fvalue, int(step))
            except Exception as exc:  # noqa: BLE001 - 观测手段不该中断训练
                _warn_once(
                    f"write-{name}",
                    "%s 写入失败（%s: %s），本次运行后续不再尝试。",
                    name, type(exc).__name__, exc,
                )
                self._writers.pop(name, None)

    def close(self) -> None:
        """关掉所有后端。训练循环用 `try/finally` 保证它一定被调到——
        不关的话最后一段事件可能还压在缓冲区里（SummaryWriter 有 120 秒的
        flush 间隔），表现为"曲线最后几步没了"。"""
        for name, writer in list(self._writers.items()):
            try:
                writer.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s 关闭时出错（忽略）：%s: %s", name, type(exc).__name__, exc)
        self._writers.clear()
