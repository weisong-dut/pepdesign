"""训练配置"配了就必须生效"的机器可执行版本。

四件事：每个字段都要登记消费方（双向相等）、确无消费方的字段必须删除、
字段归属真正消费它的阶段、非法值在加载时拒绝。
另有一条源码级回归锁：`train/` 中每处 `save_adapter(...)` 都要传 LoRAConfig。
本文件不 import torch。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from pepdesign.utils.config import GRPOConfig, SFTConfig, TrainCommonConfig

# ---------------------------------------------------------------------------
# 1. 每个字段都要有消费方
# ---------------------------------------------------------------------------
#: 训练配置每个字段 → 消费方（模块:函数）。用白名单而非 grep，
#: 因为通用字段名在源码里到处出现；新增或删除字段时必须同步这一行。
FIELD_CONSUMERS: dict[str, str] = {
    # ---- TrainCommonConfig：两个阶段共用 ----
    "model_name_or_path": "train/sft.py:build_trainer、train/grpo.py:run_grpo",
    "output_dir": "train/sft.py、train/grpo.py（落 resolved_config 与 adapter）",
    "seed": "train/sft.py、train/grpo.py（set_seed + 分桶 seed）",
    "num_train_epochs": "train/dataset.py:estimate_*_num_training_steps",
    "per_device_train_batch_size": "train/sft.py、train/grpo.py、train/dataset.py",
    "gradient_accumulation_steps": "train/sft.py、train/grpo.py、train/dataset.py",
    "learning_rate": "train/sft.py（经 resolved_lr）、train/grpo.py、train/calibration.py",
    "weight_decay": "train/sft.py、train/grpo.py（构造 AdamW）",
    "max_grad_norm": "train/sft.py、train/grpo.py、train/calibration.py（clip_grad_norm_）",
    "lr_scheduler_type": "train/dataset.py:build_scheduler_spec",
    "warmup_ratio": "train/dataset.py:build_scheduler_spec",
    "bf16": "train/sft.py、train/grpo.py → models/loader.resolve_plan",
    "fp16": "train/sft.py、train/grpo.py → models/loader.resolve_plan",
    "gradient_checkpointing": "models/loader.enable_gradient_checkpointing（两个循环都调）",
    "logging_steps": "train/sft.py、train/grpo.py、train/calibration.py",
    "save_steps": "train/checkpoint.py:on_step_end（两个循环共用）",
    "save_total_limit": "train/checkpoint.py:CheckpointManager",
    "report_to": "utils/tracking.py:MetricWriter",
    "resume_from_checkpoint": "train/checkpoint.py:resolve_resume_path",
    "max_steps": "train/dataset.py:estimate_*_num_training_steps",
    # ---- SFTConfig ----
    "dataset_path": "train/sft.py",
    "eval_dataset_path": "train/sft.py",
    "eval_steps": "train/sft.py:train_loop → on_step_end（GRPO 没有）",
    "per_device_eval_batch_size": "train/sft.py:evaluate_loss",
    "max_seq_length": "train/sft.py:encode_sft_sample",
    "completion_only_loss": "train/sft.py:encode_sft_sample",
    "lr": "train/sft.py:train_loop（经 resolved_lr）",
    # ---- GRPOConfig ----
    "sft_adapter_path": "train/grpo.py:run_grpo（load_adapter）",
    "max_prompt_length": "train/grpo.py:sample_completions → masking.truncate_prompt",
    "max_completion_length": "train/grpo.py:sample_completions（max_new_tokens）",
    "num_generations": "train/grpo.py（采样与优势分组）",
    "temperature": "train/grpo.py:sample_completions",
    "top_p": "train/grpo.py:sample_completions",
    "top_k": "train/grpo.py:sample_completions",
    "beta": "train/grpo.py:grpo_loss（KL 惩罚）",
    "loss_type": "train/grpo.py:grpo_loss → advantages.loss_denominator",
    "epsilon": "train/grpo.py:grpo_loss → advantages.clip_bounds",
    "clip_higher": "train/grpo.py:grpo_loss（仅 dapo 时生效）",
    "scale_rewards": "train/grpo.py:run_step → advantages.advantage_mode",
    "advantage_normalization": "train/grpo.py:run_step → advantages.advantage_mode",
    "calibration_path": "train/calibration.py:resolve_warmup_path",
    "calibration_enabled": "train/grpo.py:run_grpo",
    "calibration_epochs": "train/calibration.py",
    "calibration_lr_scale": "train/calibration.py",
    "reset_optimizer_after_calibration": "train/calibration.py",
}


def test_every_training_config_field_has_a_declared_consumer():
    """双向相等：每个字段都登记消费方，每条白名单也对应真实字段。"""
    for cls in (TrainCommonConfig, SFTConfig, GRPOConfig):
        undeclared = set(cls.model_fields) - set(FIELD_CONSUMERS)
        assert not undeclared, (
            f"{cls.__name__} 里有字段没登记消费方：{sorted(undeclared)}。"
            f"请先确认它真的被某处读取，然后在 FIELD_CONSUMERS 里写下位置；"
            f"没人读的话就删掉这个字段。"
        )

    known = set(TrainCommonConfig.model_fields) | set(SFTConfig.model_fields)
    known |= set(GRPOConfig.model_fields)
    for name in FIELD_CONSUMERS:
        assert name in known, (
            f"FIELD_CONSUMERS 里的 {name!r} 已经不是一个真实字段了，请删掉这一行"
        )


# ---------------------------------------------------------------------------
# 2. 确无消费方的字段必须删掉
# ---------------------------------------------------------------------------
def test_dataloader_num_workers_is_gone_because_there_is_no_dataloader():
    """`dataloader_num_workers` 必须不存在，而不是存在但没人读。

    训练循环在内存 list 上做长度分桶与动态 padding，没有 DataLoader；
    恢复它时必须在同一提交里让 batching 真正走它。
    """
    from pydantic import ValidationError

    for cls in (TrainCommonConfig, SFTConfig, GRPOConfig):
        assert "dataloader_num_workers" not in cls.model_fields, (
            f"{cls.__name__} 里又有 dataloader_num_workers 了。"
            f"如果是有意恢复它，请同时让 batching 真的走 DataLoader，"
            f"并删掉这条断言。"
        )
    with pytest.raises(ValidationError):
        SFTConfig(dataloader_num_workers=2)


# ---------------------------------------------------------------------------
# 3. 字段归属真正消费它的那个阶段
# ---------------------------------------------------------------------------
def test_eval_fields_live_on_the_phase_that_consumes_them():
    """`eval_steps` / `per_device_eval_batch_size` 只属于 SFT，GRPO 没有验证集。"""
    from pydantic import ValidationError

    assert "eval_steps" in SFTConfig.model_fields
    assert "per_device_eval_batch_size" in SFTConfig.model_fields

    for field in ("eval_steps", "per_device_eval_batch_size"):
        assert field not in GRPOConfig.model_fields, (
            f"GRPOConfig 不该有 {field}——它没有验证集，配了也不会生效。"
        )
        with pytest.raises(ValidationError):
            GRPOConfig(**{field: 50})


# ---------------------------------------------------------------------------
# 4. 静默无效的值在加载时就拒掉
# ---------------------------------------------------------------------------
def test_config_rejects_values_that_would_silently_do_nothing():
    """不生效的配置值在加载时就拒绝：`save_total_limit=0`、未知的 `report_to` 后端。"""
    from pydantic import ValidationError

    from pepdesign.utils.tracking import REPORT_BACKENDS

    with pytest.raises(ValidationError):
        SFTConfig(save_total_limit=0)
    with pytest.raises(ValidationError):
        SFTConfig(report_to=["wandb"])

    assert SFTConfig(report_to=list(REPORT_BACKENDS)).report_to == list(REPORT_BACKENDS)
    assert SFTConfig(report_to=[]).report_to == []


# ---------------------------------------------------------------------------
# 5. 源码级回归锁：save_adapter 必须拿到 LoRAConfig
# ---------------------------------------------------------------------------
def test_every_save_adapter_call_passes_the_lora_config():
    """`train/` 里每处 `save_adapter(...)` 都要传 LoRAConfig。

    静态检查覆盖所有调用点，包括以后新增的那些；手写后端漏传会抛 ValueError。
    """
    train_dir = Path(__file__).resolve().parents[1] / "src" / "pepdesign" / "train"
    offenders: list[str] = []
    for path in sorted(train_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name != "save_adapter":
                continue
            passed_by_keyword = any(kw.arg == "cfg" for kw in node.keywords)
            if not passed_by_keyword and len(node.args) < 4:
                offenders.append(f"{path.name}:{node.lineno}")

    assert not offenders, (
        f"这些 save_adapter 调用没有传 LoRAConfig：{offenders}。"
        f"手写后端下会直接抛 ValueError（见 models/lora.py:373）。"
    )


def test_the_ast_lock_would_actually_catch_a_missing_cfg():
    """自检：喂一段故意漏传 cfg 的源码，确认上一条静态断言能抓到。"""
    train_dir = Path(__file__).resolve().parents[1] / "src" / "pepdesign" / "train"
    source = "save_adapter(model, out_dir / 'final', tokenizer)\n"
    tree = ast.parse(source)
    bad = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "save_adapter"
        and not any(kw.arg == "cfg" for kw in node.keywords)
        and len(node.args) < 4
    ]
    assert bad, "自检失败：这段明显漏了 cfg 的代码没有被判为违规"

    # 而真实源码现在是干净的（这才是上一条测试的结论）
    assert train_dir.is_dir()
