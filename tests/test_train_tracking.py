"""指标外送（`utils/tracking.py`）——全部不需要 torch。

这个文件重点测的不是"能不能写 tensorboard"，而是降级路径：

* 拿不到 tensorboard 时必须变成一个安静的 no-op，而不是让训练崩掉；
* 配置里写了不存在的后端名时必须报错，而不是静默什么都不做。

第二条和 `extra="forbid"` 是同一个道理：静默忽略一个配置项，就等于
又造了一个"配了不生效"的字段，而这类字段正是这个项目一直在清理的东西。

「写进去了没有」这条路径用注入假后端来验：真 tensorboard 在本机
根本没装（`.venv` 里没有这个包），而"没装的机器上验不了写路径"本身
就是不该接受的——真写入那条分支的逻辑（前缀、跳过非有限值）是普通的
Python，用一个记账用的假 factory 就能完整覆盖。
"""

from __future__ import annotations

import logging
from typing import ClassVar

import pytest

from pepdesign.utils import tracking
from pepdesign.utils.tracking import MetricWriter, resolve_backends


@pytest.fixture(autouse=True)
def _reset_warned():
    """每个测试都从"还没警告过"开始。

    `_WARNED` 是模块级的（同一台机器上"没装 tensorboard"只需要说一次），
    所以它会跨测试泄漏——不重置的话，第二条测试拿到的是空的警告列表，
    而它会以为"没警告"，从而漏掉真正要验的行为。
    """
    saved = set(tracking._WARNED)
    tracking._WARNED.clear()
    yield
    tracking._WARNED.clear()
    tracking._WARNED.update(saved)


def _no_tensorboard(monkeypatch):
    """把"导入 tensorboard"这一步钉成必定失败。"""

    def _boom():
        raise ImportError("No module named 'tensorboard'")

    monkeypatch.setattr(tracking, "_load_summary_writer_factory", _boom)


class _RecordingWriter:
    """记账用的假 SummaryWriter。`add_scalar` 的调用被记进 `calls`。"""

    instances: ClassVar[list[_RecordingWriter]] = []

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self.calls: list[tuple[str, float, int]] = []
        self.closed = False
        _RecordingWriter.instances.append(self)

    def add_scalar(self, tag, value, step) -> None:
        self.calls.append((tag, value, step))

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def recording(monkeypatch):
    _RecordingWriter.instances = []
    monkeypatch.setattr(
        tracking, "_load_summary_writer_factory", lambda: _RecordingWriter
    )
    return _RecordingWriter


# ---------------------------------------------------------------------------
# resolve_backends：名字表就是"合法取值"的唯一来源
# ---------------------------------------------------------------------------
def test_resolve_backends_normalizes_and_dedupes():
    assert resolve_backends([" TensorBoard ", "tensorboard", ""]) == ("tensorboard",)
    assert resolve_backends([]) == ()


def test_resolve_backends_rejects_unknown_names():
    """写错后端名必须报错，不能静默忽略。

    `report_to: [wandb]` 被静默忽略时，使用者看到的是"配置写了、什么都没发生"，
    而那正是本项目定义的"配了不生效"。报错信息还必须列出合法取值——
    否则使用者只能去读源码才知道能写什么。
    """
    with pytest.raises(ValueError, match="tensorboard"):
        resolve_backends(["wandb"])


# ---------------------------------------------------------------------------
# 降级路径
# ---------------------------------------------------------------------------
def test_missing_backend_degrades_to_noop(monkeypatch, caplog):
    """拿不到 tensorboard：`enabled` 为 False，三个方法都不抛。"""
    _no_tensorboard(monkeypatch)
    writer = MetricWriter(backends=["tensorboard"], prefix="sft", log_dir="outputs/_nope")
    assert writer.enabled is False
    assert "tensorboard" in writer.disabled_reasons
    # 关键：调用方不需要判断 enabled，直接调也不能炸。
    writer.add_scalar("loss", 1.0, 1)
    writer.add_scalars({"loss": 1.0, "lr": 2e-4}, 2)
    writer.close()


def test_missing_backend_warns_only_once(monkeypatch, caplog):
    """同一台机器上"没装 tensorboard"只说一次。

    训练循环每一步都会构造/使用 writer（SFT 和 GRPO 各一个），
    每次都警告的话日志会被刷满，然后所有人学会忽略这个 warning。
    """
    _no_tensorboard(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="pepdesign.utils.tracking"):
        MetricWriter(backends=["tensorboard"], log_dir="outputs/_nope1")
        MetricWriter(backends=["tensorboard"], log_dir="outputs/_nope2")
    warnings = [r for r in caplog.records if "tensorboard" in r.message]
    assert len(warnings) == 1, f"警告了 {len(warnings)} 次，应该只警告 1 次"


def test_disabled_writer_creates_no_directory(recording, tmp_path):
    """`report_to: []` 时连目录都不该建。

    禁用的 writer 在磁盘上留一个空的 `tensorboard/`，会让
    "这次跑到底写没写指标"变成一个要去看时间戳才能回答的问题。
    """
    out = tmp_path / "exp"
    writer = MetricWriter.from_config([], output_dir=out, prefix="sft")
    assert writer.enabled is False
    assert not (out / "tensorboard").exists()

    writer.add_scalars({"loss": 1.0}, 1)
    assert _RecordingWriter.instances == [], "禁用时不该构造任何后端"


# ---------------------------------------------------------------------------
# 写入路径（注入假后端）
# ---------------------------------------------------------------------------
def test_prefix_is_applied_to_every_tag(recording, tmp_path):
    """阶段名前缀不能省：两阶段都有 `loss`，撞在一起会被画成同一条曲线。"""
    writer = MetricWriter.from_config(["tensorboard"], output_dir=tmp_path, prefix="sft")
    writer.add_scalars({"loss": 1.0, "lr": 2e-4}, step=10)
    writer.add_scalar("eval_loss", 0.9, step=10)
    writer.close()

    (inst,) = recording.instances
    assert [c[0] for c in inst.calls] == ["sft/loss", "sft/lr", "sft/eval_loss"]
    assert [c[2] for c in inst.calls] == [10, 10, 10]
    assert inst.closed is True
    assert inst.log_dir.endswith("tensorboard")


def test_writer_lives_under_output_dir(recording, tmp_path):
    """指标和 adapter 落在同一个实验目录里，排查时不需要先问"日志在哪"。"""
    writer = MetricWriter.from_config(["tensorboard"], output_dir=tmp_path / "sft", prefix="sft")
    assert writer.log_dir == tmp_path / "sft" / "tensorboard"
    assert writer.log_dir.exists()
    writer.close()


def test_nonfinite_values_are_skipped(recording, tmp_path, caplog):
    """NaN/inf 不写。

    它们写进 tensorboard 是合法的，但实际效果是那条曲线从该点开始整段消失
    （前端的平滑会把整个窗口一起干掉），诊断价值比"这个点没记"还低。
    """
    writer = MetricWriter.from_config(["tensorboard"], output_dir=tmp_path, prefix="grpo")
    with caplog.at_level(logging.WARNING, logger="pepdesign.utils.tracking"):
        writer.add_scalars({"loss": 1.0, "grad_norm": float("inf"), "kl": float("nan")}, 3)
    writer.close()

    (inst,) = recording.instances
    assert [c[0] for c in inst.calls] == ["grpo/loss"]
    assert any("非有限值" in r.message for r in caplog.records)


def test_backend_failure_during_write_disables_instead_of_raising(recording, tmp_path):
    """写失败（磁盘满、后端内部错误）只丢弃这个后端，不让训练崩。

    指标外送是观测手段。它挂掉不该让一次跑了几小时的训练失败——
    这是这个模块存在的前提，所以要有测试盯着它。
    """

    class _Broken(_RecordingWriter):
        def add_scalar(self, tag, value, step):
            raise OSError("No space left on device")

    recording.instances = []
    import pepdesign.utils.tracking as t

    t._load_summary_writer_factory = lambda: _Broken  # type: ignore[assignment]
    writer = t.MetricWriter(backends=["tensorboard"], prefix="sft", log_dir=tmp_path)
    assert writer.enabled is True
    writer.add_scalars({"loss": 1.0}, 1)      # 不抛
    assert writer.enabled is False, "写失败之后这个后端应当被摘掉"
    writer.add_scalars({"loss": 0.9}, 2)      # 再调也不抛
    writer.close()
