"""checkpoint 的命名、保留、状态读写与步末收尾，全部不需要 torch。

覆盖 `save_steps`/`eval_steps` 边界、`save_total_limit` 保留策略、
`resume_from_checkpoint` 路径解析。均为纯文件系统与 JSON 逻辑。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from pepdesign.train.checkpoint import (
    CheckpointManager,
    _jsonable,
    checkpoint_dir,
    compare_fingerprint,
    list_checkpoints,
    on_step_end,
    parse_checkpoint_step,
    read_trainer_state,
    resolve_resume_path,
    select_stale_checkpoints,
    should_fire,
    training_fingerprint,
    write_trainer_state,
)
from pepdesign.utils.config import GRPOConfig, SFTConfig


def _make_ckpt(root, step: int, *, complete: bool = True):
    """造一个 checkpoint 目录。`complete=False` 模拟写到一半崩掉。"""
    d = checkpoint_dir(root, step)
    d.mkdir(parents=True, exist_ok=True)
    if complete:
        write_trainer_state(d, step=step, total_steps=999, phase="sft", history={})
    return d


# ---------------------------------------------------------------------------
# should_fire：评估与保存共用的唯一一份判断
# ---------------------------------------------------------------------------
def test_should_fire_boundaries():
    """`every <= 0` 是"从不"的哨兵；`step <= 0` 必须排除，因为 `0 % 5 == 0`。"""
    assert should_fire(0, 5) is False, "0 % 5 == 0，但第 0 步不该触发任何事"
    assert should_fire(5, 5) is True
    assert should_fire(10, 5) is True
    assert should_fire(7, 5) is False
    assert should_fire(1, 1) is True

    assert should_fire(3, 0) is False, "every=0 表示从不"
    assert should_fire(3, -1) is False
    assert should_fire(-5, 5) is False


# ---------------------------------------------------------------------------
# 命名与列表
# ---------------------------------------------------------------------------
def test_checkpoint_dir_and_parse_roundtrip():
    assert checkpoint_dir("outputs/sft", 250).name == "checkpoint-250"
    assert parse_checkpoint_step(checkpoint_dir("x", 1250)) == 1250
    assert parse_checkpoint_step(checkpoint_dir("x", 1250).parent / "final") is None
    assert parse_checkpoint_step(checkpoint_dir("x", 1250).parent / "checkpoint-abc") is None


def test_list_checkpoints_sorts_numerically_and_ignores_junk(tmp_path):
    """只认完整目录与 `checkpoint-<数字>`，且按步号整数排序。

    按字符串排序会把 `checkpoint-100` 排在 `checkpoint-20` 前，误删最新项。
    """
    for step in (9, 10, 20, 100):
        _make_ckpt(tmp_path, step)
    (tmp_path / "final").mkdir()                    # 最终产物，不是 checkpoint
    (tmp_path / "tensorboard").mkdir()              # 指标目录
    (tmp_path / "not-a-checkpoint").mkdir()
    _make_ckpt(tmp_path, 30, complete=False)        # 写到一半崩了

    listed = list_checkpoints(tmp_path)
    assert [s for s, _ in listed] == [9, 10, 20, 100]
    assert all(p.name.startswith("checkpoint-") for _, p in listed)


def test_list_checkpoints_on_missing_dir_is_empty(tmp_path):
    assert list_checkpoints(tmp_path / "不存在") == []


# ---------------------------------------------------------------------------
# 保留策略
# ---------------------------------------------------------------------------
def test_select_stale_checkpoints_keeps_the_newest(tmp_path):
    for step in (10, 20, 30, 40):
        _make_ckpt(tmp_path, step)
    existing = list_checkpoints(tmp_path)

    stale = [p.name for p in select_stale_checkpoints(existing, 2)]
    assert stale == ["checkpoint-10", "checkpoint-20"]

    # 刚刚保存的那一个永远不在删除列表里：on_step_end 是先存后删，
    # N=1 时保留的就是本步这个。顺序反过来的话磁盘上什么都不剩。
    assert "checkpoint-40" not in stale

    assert select_stale_checkpoints(existing, 4) == []
    assert select_stale_checkpoints(existing, 99) == []
    # <= 0 表示"不限制"，不是"全删"
    assert select_stale_checkpoints(existing, 0) == []


# ---------------------------------------------------------------------------
# 续训路径解析
# ---------------------------------------------------------------------------
def test_resolve_resume_path_none_means_no_resume(tmp_path):
    assert resolve_resume_path(None, tmp_path) is None
    assert resolve_resume_path("", tmp_path) is None
    assert resolve_resume_path("   ", tmp_path) is None


def test_resolve_resume_path_latest(tmp_path):
    for step in (10, 20, 30):
        _make_ckpt(tmp_path, step)
    assert resolve_resume_path("latest", tmp_path).name == "checkpoint-30"


def test_resolve_resume_path_latest_without_any_checkpoint_raises(tmp_path):
    """找不到 checkpoint 必须报错，不能静默从头训练并覆盖 `final/`。

    报错信息还要点出最常见的原因（`save_steps` 为 0 或大于总步数）。
    """
    with pytest.raises(FileNotFoundError, match="save_steps"):
        resolve_resume_path("latest", tmp_path)


def test_resolve_resume_path_rejects_final_with_two_fixes(tmp_path):
    """指向 `final/` 要报错并给出修法：它只有权重，没有优化器与调度器状态。"""
    (tmp_path / "final").mkdir()
    with pytest.raises(ValueError) as excinfo:
        resolve_resume_path(str(tmp_path / "final"), tmp_path)
    msg = str(excinfo.value)
    assert "checkpoint-" in msg, "报错要指出该用 checkpoint-<N>"
    assert "warmup" in msg, "报错要说明后果，而不是只说'文件不存在'"


def test_resolve_resume_path_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_resume_path(str(tmp_path / "checkpoint-999"), tmp_path)


# ---------------------------------------------------------------------------
# 状态文件
# ---------------------------------------------------------------------------
def test_trainer_state_roundtrip(tmp_path):
    path = write_trainer_state(
        tmp_path, step=250, total_steps=3000, phase="sft",
        history={"loss": [1.0, 0.5]}, fingerprint={"num_steps": 3000},
    )
    assert path.name == "trainer_state.json"
    state = read_trainer_state(tmp_path)          # 传目录也可以
    assert state["step"] == 250
    assert state["total_steps"] == 3000
    assert state["phase"] == "sft"
    assert state["history"] == {"loss": [1.0, 0.5]}
    assert read_trainer_state(path)["step"] == 250


def test_read_trainer_state_missing_file_explains_why(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"trainer_state\.json"):
        read_trainer_state(tmp_path)


def test_read_trainer_state_rejects_foreign_json(tmp_path):
    """外来目录（有人手工拷了别的 JSON 进来）不能当成 checkpoint 用。"""
    (tmp_path / "trainer_state.json").write_text(json.dumps({"foo": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="step"):
        read_trainer_state(tmp_path)


@dataclass
class _FakeMetrics:
    loss: float
    grad_norm: float


def test_jsonable_handles_dataclass_and_nonfinite():
    """落盘前：dataclass → dict，非有限浮点 → None。

    显式 None 表示该指标本步无有效值，比静默变成 0 更好排查。
    """
    out = _jsonable([_FakeMetrics(loss=1.5, grad_norm=float("inf"))])
    assert out == [{"loss": 1.5, "grad_norm": None}]
    assert _jsonable({"a": float("nan")}) == {"a": None}
    assert _jsonable({"a": [1.0, 2.0]}) == {"a": [1.0, 2.0]}


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_records_the_resolved_lr():
    """SFT 的 `lr` 覆盖 `learning_rate`，fingerprint 必须记最终生效的那个。"""
    assert training_fingerprint(
        SFTConfig(learning_rate=2e-4), phase="sft", num_steps=3000
    )["learning_rate"] == 2e-4
    assert training_fingerprint(
        SFTConfig(learning_rate=2e-4, lr=3e-5), phase="sft", num_steps=3000
    )["learning_rate"] == 3e-5


def test_compare_fingerprint_splits_fatal_from_config_wins():
    """致命差异（步数/批量/阶段）与配置覆盖项（lr/调度器/weight_decay/seed）必须分开返回。"""
    recorded = training_fingerprint(SFTConfig(), phase="sft", num_steps=3000)

    # 换了 num_steps：致命
    critical, config_wins = compare_fingerprint(
        recorded, training_fingerprint(SFTConfig(), phase="sft", num_steps=1500)
    )
    assert any("num_steps" in c for c in critical)
    assert config_wins == []

    # 换了 lr：只记日志，不报错
    critical, config_wins = compare_fingerprint(
        recorded,
        training_fingerprint(SFTConfig(learning_rate=1e-4), phase="sft", num_steps=3000),
    )
    assert critical == []
    assert any("learning_rate" in c for c in config_wins)

    # 阶段不匹配（把 GRPO 的 checkpoint 续到 SFT 上）
    critical, _ = compare_fingerprint(
        recorded, training_fingerprint(GRPOConfig(), phase="grpo", num_steps=375)
    )
    assert any("phase" in c for c in critical)


def test_fingerprint_ignores_keys_a_phase_does_not_have():
    """某阶段不存在的键在两边都是 None，不该被判成差异。"""
    fp = training_fingerprint(SFTConfig(), phase="sft", num_steps=3000)
    assert fp["max_seq_length"] == SFTConfig().max_seq_length
    assert fp["max_completion_length"] is None
    critical, config_wins = compare_fingerprint(fp, dict(fp))
    assert critical == [] and config_wins == []


# ---------------------------------------------------------------------------
# on_step_end：两个阶段共用的步末收尾
# ---------------------------------------------------------------------------
class _OrderLog:
    def __init__(self):
        self.events: list[str] = []


def test_on_step_end_order_is_evaluate_then_write_then_save():
    """顺序不能换：先评估再保存，history 才含本步 eval；标量先写以便崩溃时留痕。"""
    log = _OrderLog()
    saved: list[int] = []

    def _evaluate():
        log.events.append("evaluate")
        return 1.25

    def _save():
        log.events.append("save")
        saved.append(1)

    class _W:
        def add_scalars(self, values, step):
            log.events.append(f"write:{sorted(values)}")

    merged = on_step_end(
        step=250, total_steps=3000, eval_steps=250, save_steps=250,
        writer=_W(), scalars={"loss": 1.0}, evaluate=_evaluate, save=_save,
    )
    assert log.events == ["evaluate", "write:['eval_loss', 'loss']", "save"]
    assert merged["eval_loss"] == 1.25
    assert len(saved) == 1


def test_on_step_end_does_not_evaluate_on_the_final_step():
    """最后一步不评估（调用方会再算一次），但仍按配置保存。"""
    calls = {"eval": 0, "save": 0}

    def _evaluate():
        calls["eval"] += 1
        return 0.5

    def _save():
        calls["save"] += 1

    on_step_end(step=3000, total_steps=3000, eval_steps=250, save_steps=250,
                scalars={"loss": 0.1}, evaluate=_evaluate, save=_save)
    assert calls == {"eval": 0, "save": 1}


def test_on_step_end_is_a_noop_on_non_boundary_steps():
    calls = {"eval": 0, "save": 0}
    merged = on_step_end(
        step=7, total_steps=3000, eval_steps=250, save_steps=250,
        scalars={"loss": 1.0},
        evaluate=lambda: calls.__setitem__("eval", calls["eval"] + 1) or 1.0,
        save=lambda: calls.__setitem__("save", calls["save"] + 1),
    )
    assert calls == {"eval": 0, "save": 0}
    assert merged == {"loss": 1.0}


def test_on_step_end_works_without_writer_or_callbacks():
    """GRPO 没有验证集（evaluate=None），冒烟配置 report_to 为空（writer=None）。"""
    assert on_step_end(
        step=50, total_steps=375, eval_steps=0, save_steps=0, scalars={"loss": 1.0}
    ) == {"loss": 1.0}


# ---------------------------------------------------------------------------
# CheckpointManager：构造时就要求 LoRAConfig
# ---------------------------------------------------------------------------
def test_checkpoint_manager_requires_lora_cfg():
    """`lora_cfg` 是构造必填参数，不是有默认值的。

    有默认值时"忘了传"重新成为可能，手写后端下会直接抛 ValueError。
    """
    import inspect

    params = inspect.signature(CheckpointManager.__init__).parameters
    assert params["lora_cfg"].default is inspect.Parameter.empty, (
        "lora_cfg 一旦有默认值，'忘了传' 就重新变成可能"
    )
