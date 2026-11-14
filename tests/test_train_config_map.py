"""配置、参数量与设备决策，覆盖"配置说一套、实际跑另一套"的失败模式。

包括 warmup 比例换算、LoRA `target_modules` 缩小导致参数量骤降、
无加速器时 bf16/fp16 的降级。`resolve_plan` 把硬件能力作为入参，
因此本文件不 import torch。
"""

from __future__ import annotations

import pathlib

import pytest

from pepdesign.models.loader import chat_tokenize, resolve_plan
from pepdesign.models.lora import build_peft_config, lora_trainable_summary
from pepdesign.train.dataset import (
    build_scheduler_spec,
    estimate_grpo_num_training_steps,
    estimate_sft_num_training_steps,
    load_grpo_prompts,
    load_sft_samples,
    warmup_steps_for,
)
from pepdesign.utils.config import GRPOConfig, LoRAConfig, QuantConfig, SFTConfig, load_config

# ---------------------------------------------------------------------------
# 设备 / 精度决策
# ---------------------------------------------------------------------------


def test_resolve_plan_falls_back_to_cpu_when_no_accelerator():
    """没有 CUDA / MPS 时回退 fp32 且不量化；bf16/fp16 必须一起关掉。

    否则 `TrainingArguments` 会直接拒绝启动，报错指向 bf16 而非缺少 GPU。
    """
    plan = resolve_plan(
        cuda_available=False, bf16_supported=False, mps_available=False,
        want_bf16=True, want_fp16=False, quant=QuantConfig(enabled=True, load_in_4bit=True),
    )
    assert plan.device == "cpu"
    assert plan.dtype == "float32"
    assert plan.load_in_4bit is False and plan.load_in_8bit is False
    assert plan.is_fallback is True
    assert "fp32" in plan.reason


def test_resolve_plan_cuda_prefers_bf16_but_handles_old_cards():
    """有卡时优先 bf16，不支持 bf16 的老卡（如 V100）退 fp16。"""
    new_gpu = resolve_plan(
        cuda_available=True, bf16_supported=True,
        want_bf16=True, quant=QuantConfig(enabled=True, load_in_4bit=True),
    )
    assert new_gpu.device == "cuda" and new_gpu.dtype == "bfloat16"
    assert new_gpu.load_in_4bit is True
    assert new_gpu.is_fallback is False

    old_gpu = resolve_plan(
        cuda_available=True, bf16_supported=False, want_bf16=True,
        quant=QuantConfig(enabled=True, load_in_4bit=True),
    )
    assert old_gpu.dtype == "float16"
    assert "V100" in old_gpu.reason

    # 显式关掉混合精度
    fp32 = resolve_plan(cuda_available=True, bf16_supported=True, want_bf16=False, want_fp16=False)
    assert fp32.dtype == "float32"

    # 配置里关掉量化时不能自作主张打开
    no_quant = resolve_plan(
        cuda_available=True, bf16_supported=True, quant=QuantConfig(enabled=False)
    )
    assert no_quant.load_in_4bit is False


def test_resolve_plan_mps_forces_fp32_and_no_quant():
    """Apple MPS：强制 fp32 且关量化；MPS 的 bf16 支持分算子，bitsandbytes 无后端。"""
    plan = resolve_plan(
        cuda_available=False, mps_available=True, want_bf16=True,
        quant=QuantConfig(enabled=True, load_in_4bit=True),
    )
    assert plan.device == "mps"
    assert plan.dtype == "float32"
    assert plan.load_in_4bit is False
    # MPS 不算"回退到 CPU"，但也不算有 CUDA——语义上是"非本地 GPU 加速"
    assert plan.is_fallback is False


# ---------------------------------------------------------------------------
# chat template 的返回类型（transformers 5.x 改过默认值）
# ---------------------------------------------------------------------------
class _FakeTokenizer:
    """只实现 `apply_chat_template` 的假 tokenizer。

    transformers ≤4.x 返回 `list[int]`，5.x 默认 `return_dict=True` 时返回 `BatchEncoding`。
    """

    def __init__(self, returned):
        self.returned = returned
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        return self.returned


def test_chat_tokenize_returns_plain_ids():
    tok = _FakeTokenizer([1, 2, 3])
    assert chat_tokenize(tok, [{"role": "user", "content": "hi"}]) == [1, 2, 3]


def test_chat_tokenize_always_passes_return_dict_false():
    """必须显式传 `return_dict=False`；transformers 5.x 的默认值是 `True`。"""
    tok = _FakeTokenizer([1, 2, 3])
    chat_tokenize(tok, [{"role": "user", "content": "hi"}])
    assert tok.calls[0]["return_dict"] is False
    assert tok.calls[0]["tokenize"] is True


def test_chat_tokenize_rejects_batch_encoding_loudly():
    """拿到 `BatchEncoding` 时必须报错，不能把键当成 token id 返回。

    它是 dict 子类，`len()` 返回 2，会让长度预检静默失效。
    """
    tok = _FakeTokenizer({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]})
    with pytest.raises(TypeError, match="return_dict"):
        chat_tokenize(tok, [{"role": "user", "content": "hi"}])


def test_chat_tokenize_forwards_add_generation_prompt():
    tok = _FakeTokenizer([1])
    chat_tokenize(tok, [{"role": "user", "content": "hi"}], add_generation_prompt=True)
    assert tok.calls[0]["add_generation_prompt"] is True


# ---------------------------------------------------------------------------
# "配了不生效"的字段
# ---------------------------------------------------------------------------


def test_removed_trl_only_fields_are_really_gone():
    """`packing` / `assistant_only_loss` 必须不存在，而不是存在但不生效。

    自写训练循环没有实现这两个字段；`extra="forbid"` 拦不住"字段合法但没人读"。
    """
    from pydantic import ValidationError

    for field in ("packing", "assistant_only_loss"):
        assert field not in SFTConfig.model_fields, (
            f"{field} 又回到了 SFTConfig 里。如果是有意恢复了它，"
            f"请同时确认 train/sft.py 真的读了它，并删掉这条断言。"
        )
        # 从 YAML / 命令行传进来时必须报错，而不是静默忽略。
        with pytest.raises(ValidationError):
            SFTConfig(**{field: True})


def test_sft_config_lr_overrides_learning_rate():
    """`lr` 非空时覆盖 `learning_rate`；两个字段同时存在但只有 `lr` 生效。"""
    # 默认 lr 是 None，此时用 learning_rate
    assert SFTConfig().lr is None
    assert SFTConfig(learning_rate=2e-4).learning_rate == 2e-4

    # 同时给时，lr 是那个"覆盖值"。两个字段都还在对象上（不是互斥的），
    # 所以真正的解析必须在取值处做——这正是要测的点。
    both = SFTConfig(learning_rate=2e-4, lr=3e-5)
    resolved = both.lr if both.lr is not None else both.learning_rate
    assert resolved == 3e-5


# ---------------------------------------------------------------------------
# 调度器换算
# ---------------------------------------------------------------------------


def test_scheduler_spec_converts_warmup_ratio_to_steps():
    """`warmup_ratio` 是比例，`get_scheduler` 要的是步数；直接传比例等于不 warmup。"""
    spec = build_scheduler_spec(GRPOConfig(warmup_ratio=0.1), num_training_steps=100)
    assert spec["num_warmup_steps"] == 10
    assert spec["num_training_steps"] == 100
    assert spec["lr_scheduler_type"] == GRPOConfig().lr_scheduler_type


def test_scheduler_spec_never_disables_or_overshoots_warmup():
    """比例很小时至少 1 步，比例很大时不超过总步数，比例恰为 0 时必须是 0 步。"""
    # 3 步 × 0.03 = 0.09 → 取整成 0；比例非零表示想要 warmup，需兜底到 1。
    def _warmup(ratio, steps):
        return build_scheduler_spec(
            GRPOConfig(warmup_ratio=ratio), num_training_steps=steps
        )["num_warmup_steps"]

    assert _warmup(0.03, 3) == 1          # 0.09 → 取整成 0 → 兜底到 1
    assert _warmup(2.0, 500) == 500      # 不能超过总步数
    assert _warmup(0.0, 100) == 0        # 显式的"不要 warmup"，不能被兜底成 1

    with pytest.raises(ValueError, match="num_training_steps"):
        build_scheduler_spec(GRPOConfig(), num_training_steps=0)


def test_warmup_rule_is_shared_between_sft_and_grpo():
    """同一个 `warmup_ratio` 在 SFT 与 GRPO 两侧必须算出同一个整数。"""
    ratio, steps = 0.1, 200
    for cfg_cls in (SFTConfig, GRPOConfig):
        spec = build_scheduler_spec(cfg_cls(warmup_ratio=ratio), num_training_steps=steps)
        assert spec["num_warmup_steps"] == warmup_steps_for(ratio, steps) == 20
        assert spec["num_training_steps"] == steps


def test_sft_warmup_is_an_integer_number_of_steps():
    """SFT 的 warmup 也必须是明确的整数步数，不能依赖"小于 1 当比例处理"的多态行为。"""
    spec = build_scheduler_spec(SFTConfig(warmup_ratio=0.03), num_training_steps=1000)
    assert spec["num_warmup_steps"] == 30
    assert isinstance(spec["num_warmup_steps"], int)

    # 显式的 0 传下去还是 0
    assert build_scheduler_spec(
        SFTConfig(warmup_ratio=0.0), num_training_steps=1000
    )["num_warmup_steps"] == 0

    # 真实配置（600 条语料）算出来的就是这个数——
    # configs/sft_lora.yaml 的注释里写的 "0.03 × 57 步 = 1.71 → 1 步" 靠它钉住。
    n_steps = estimate_sft_num_training_steps(SFTConfig(), 600)
    assert n_steps == 57
    assert build_scheduler_spec(SFTConfig(), num_training_steps=n_steps)["num_warmup_steps"] == 1


def test_estimate_sft_num_training_steps():
    """`max_steps` 优先；否则由 batch × 累积 × epoch 推出总步数。"""
    cfg = SFTConfig(per_device_train_batch_size=4, gradient_accumulation_steps=8)
    # 600 条 / 32 = 18.75 → 19 步一轮 × 3 epoch = 57
    assert estimate_sft_num_training_steps(cfg, 600) == 57
    # 不足一个 batch 也要算 1 步，不能是 0（0 会让 warmup 换算直接抛错）
    assert estimate_sft_num_training_steps(cfg, 1) == 3
    # max_steps 设了就是精确值，和数据集大小无关
    assert estimate_sft_num_training_steps(SFTConfig(max_steps=5), 600) == 5
    # max_steps=-1 是"按 epoch 跑满"的哨兵值，不能被当成 0 步
    assert estimate_sft_num_training_steps(SFTConfig(max_steps=-1), 600) > 0


def test_estimate_grpo_num_training_steps():
    """GRPO 步数的分母是每步消耗的 prompt 数，`num_generations` 不进分母。"""
    cfg = GRPOConfig(
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        num_generations=8, num_train_epochs=1,
    )
    # 200 条 / (4×4) = 12.5 → 13 步跑满一个 epoch
    assert estimate_grpo_num_training_steps(cfg, 200) == 13
    # 老写法给的是 200/4 = 50 步（= 4 遍语料）。这里必须不同，否则没修。
    assert estimate_grpo_num_training_steps(cfg, 200) != 50

    # num_generations 不进分母：G 变化不该改动步数；G=1 由配置校验挡在外面。
    for g in (2, 8, 32):
        assert estimate_grpo_num_training_steps(
            GRPOConfig(
                per_device_train_batch_size=4, gradient_accumulation_steps=4,
                num_generations=g, num_train_epochs=1,
            ),
            200,
        ) == 13

    # epoch 数真的生效了（这正是以前不生效的那一项）
    assert estimate_grpo_num_training_steps(
        GRPOConfig(
            per_device_train_batch_size=4, gradient_accumulation_steps=4,
            num_train_epochs=3,
        ),
        200,
    ) == 39

    # 不足一个 batch 也要算 1 步；max_steps 设了就是精确值；-1 是哨兵不是 0
    assert estimate_grpo_num_training_steps(GRPOConfig(), 1) >= 1
    assert estimate_grpo_num_training_steps(
        GRPOConfig(max_steps=5), 200
    ) == 5
    assert estimate_grpo_num_training_steps(
        GRPOConfig(max_steps=-1), 200
    ) > 0


# ---------------------------------------------------------------------------
# LoRA 配置映射
# ---------------------------------------------------------------------------


def test_peft_config_maps_every_field():
    """peft 后端的字段映射必须一个不多、一个不少；`backend` 不能传给 peft。"""
    cfg = LoRAConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj"])
    mapped = build_peft_config(cfg)

    assert mapped == {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    # bias 是 peft 的默认值，仍然显式传：默认值会随版本变，显式传才可复现
    assert "bias" in mapped
    assert "modules_to_save" not in mapped
    assert "backend" not in mapped, "这是本项目的字段，peft 不认识"


def test_default_lora_targets_all_linear_layers():
    """默认挂全部 7 个线性层而不只是 Q/K/V；缩回去会让参数量下降一个数量级。"""
    cfg = LoRAConfig()
    assert set(cfg.resolved_target_modules()) == {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    }
    assert cfg.r == 16 and cfg.lora_alpha == 32
    assert cfg.backend == "manual"


def test_resolved_target_modules_dedupes_without_reordering():
    """重复的模块名要去重且保持声明顺序，否则参数量估算会翻倍。"""
    cfg = LoRAConfig(target_modules=["q_proj", "v_proj", "q_proj"])
    assert cfg.resolved_target_modules() == ["q_proj", "v_proj"]


def test_lora_scaling_guard_rejects_silent_drift():
    """alpha/r 偏离合理区间时必须报错；改 r 忘了同步 alpha 会让实验不可比。"""
    from pepdesign.models.lora import validate_lora_config

    validate_lora_config(LoRAConfig(r=16, lora_alpha=32))    # 缩放 2.0，正常
    with pytest.raises(ValueError, match="缩放"):
        validate_lora_config(LoRAConfig(r=16, lora_alpha=320))


def test_lora_trainable_summary_matches_hand_computed_logos_3b_numbers():
    """参数量必须逐项算准（LOGOS-3B：hidden 3072、28 层、24/8 GQA 头）。

    偏小一个数量级的估算会让"LoRA 没挂上"无法被察觉。
    """
    summary = lora_trainable_summary(LoRAConfig())

    per_layer = 98304 + 65536 + 65536 + 98304 + 180224 * 3
    assert per_layer == 868_352
    assert summary["per_layer_params"] == per_layer
    assert summary["estimated_trainable_params"] == per_layer * 28 == 24_313_856
    assert summary["total_params"] == 3_212_808_192
    assert summary["scaling"] == 2.0
    assert summary["backend"] == "manual"
    assert summary["unknown_target_modules"] == []
    # 0.76%：远小于 1%，但不是 0.045%（那是只挂 Q/K/V 的数字）。
    # 断言一个区间而不是等值，是为了让"改了 r"这件事在这里可见地反映出来。
    assert 0.0075 < summary["estimated_share"] < 0.0076


def test_lora_adapter_size_is_params_times_dtype_bytes():
    """adapter 文件大小 = 参数量 × dtype 字节数；bf16 下 24,313,856 参数约 46 MiB。"""
    summary = lora_trainable_summary(LoRAConfig())
    assert summary["adapter_bytes_bf16"] == summary["estimated_trainable_params"] * 2
    assert summary["adapter_bytes_bf16"] == 48_627_712
    assert 46_000_000 < summary["adapter_bytes_bf16"] < 49_000_000


def test_lora_trainable_summary_reports_unknown_modules_instead_of_ignoring():
    """不认识的模块名要报出来，不能当成 0 悄悄跳过。"""
    summary = lora_trainable_summary(
        LoRAConfig(target_modules=["q_proj", "out_proj"], r=16)
    )
    assert summary["unknown_target_modules"] == ["out_proj"]
    # 只按认得的那个算：q_proj 单模块 16 × 6144 = 98304，× 28 层
    assert summary["per_layer_params"] == 98_304
    assert summary["estimated_trainable_params"] == 98_304 * 28


def test_lora_trainable_summary_supports_other_model_sizes():
    """换模型时用 overrides 覆盖全部结构参数，不用改代码。

    Qwen3-0.6B 的 GQA 比例与 LOGOS-3B 不同。
    """
    from pepdesign.models.lora import QWEN3_0_6B

    cfg = LoRAConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj"])
    s = lora_trainable_summary(cfg, **QWEN3_0_6B)

    # q_proj: 8 × (1024 + 16×128) = 24576
    # k_proj: 8 × (1024 +  8×128) = 16384
    # v_proj: 同上                = 16384
    assert s["per_layer_params"] == 24576 + 16384 + 16384 == 57344
    assert s["estimated_trainable_params"] == 57344 * 28 == 1_605_632
    assert s["estimated_share"] < 0.01

    # 只覆盖 num_hidden_layers 时，其余结构参数仍用 LOGOS-3B 的默认值——
    # 这是"部分覆盖"的语义，容易误解，所以在这里显式钉住。
    smaller = lora_trainable_summary(LoRAConfig(r=8, lora_alpha=16), num_hidden_layers=1)
    assert smaller["estimated_trainable_params"] == 868_352 * 8 // 16  # r 从 16 降到 8


def test_lora_config_for_model_preset_rejects_unknown_name():
    """预设名写错时必须报错，不能静默退回主基座。"""
    from pepdesign.models.lora import lora_config_for_model

    three = lora_config_for_model(LoRAConfig(), "logos-3b")
    six = lora_config_for_model(LoRAConfig(), "qwen3-0.6b")
    assert three["estimated_trainable_params"] > six["estimated_trainable_params"]

    with pytest.raises(ValueError, match="未知的模型预设"):
        lora_config_for_model(LoRAConfig(), "qwen-9b")


# ---------------------------------------------------------------------------
# 语料校验（直接跑在项目真实产出上）
# ---------------------------------------------------------------------------


def test_real_sft_corpus_passes_validation():
    """仓库产出的 SFT 语料必须能通过读取校验。

    在数据产物与训练代码之间建立会失败的联系，格式漂移时立刻暴露。
    """
    rows = load_sft_samples("data/processed/sft_train.jsonl")
    assert len(rows) > 0
    for row in rows[:20]:
        assert row["messages"][-1]["role"] == "assistant"
        assert "task_type" in row


def test_real_grpo_corpus_passes_validation():
    rows = load_grpo_prompts("data/processed/grpo_prompts.jsonl")
    assert len(rows) > 0
    for row in rows[:20]:
        assert "assistant" not in [m["role"] for m in row["prompt"]]
        assert row["meta"]["task"]


def test_load_sft_samples_rejects_grpo_file():
    """把 prompt 文件当成 SFT 语料传进来必须报错。

    GRPO 的 prompt 没有 assistant 消息，会被当成整段 completion 训练。
    """
    with pytest.raises(ValueError, match="缺少必备字段"):
        load_sft_samples("data/processed/grpo_prompts.jsonl")


def test_load_grpo_prompts_rejects_sft_file():
    with pytest.raises(ValueError, match="缺少必备字段"):
        load_grpo_prompts("data/processed/sft_train.jsonl")


def test_missing_corpus_error_points_at_the_fix():
    """语料不存在时的报错必须指出先运行数据构建。"""
    with pytest.raises(FileNotFoundError, match="数据构建"):
        load_sft_samples("data/processed/不存在的文件.jsonl")


# ---------------------------------------------------------------------------
# 配置文件本身
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["configs/sft_lora.yaml", "configs/grpo_lora.yaml", "configs/smoke_cpu.yaml"],
)
def test_config_files_load_and_are_self_consistent(path):
    """三份配置文件都要能加载，且缩放比例、精度与步数等跨字段不变量成立。"""
    cfg = load_config(path)

    assert cfg.lora.r >= 1
    assert 0.25 <= cfg.lora.scaling <= 8.0
    assert cfg.grpo.num_generations >= 2

    if "smoke" in path:
        # 冒烟配置的核心约束：能在没有 GPU 的机器上跑完
        assert cfg.sft.bf16 is False and cfg.sft.fp16 is False
        assert cfg.grpo.bf16 is False and cfg.grpo.fp16 is False
        assert cfg.sft.max_steps > 0 and cfg.grpo.max_steps > 0
        assert cfg.quant.enabled is False
    else:
        # 真实配置必须是两阶段串起来的：GRPO 从 SFT 的产物继续
        assert cfg.grpo.sft_adapter_path, "真实配置里 sft_adapter_path 不能为空"
        assert cfg.sft.dataset_path.endswith("sft_train.jsonl")
        assert cfg.grpo.dataset_path.endswith("grpo_prompts.jsonl")


def test_infer_is_no_longer_an_unimplemented_section():
    """`infer:` 不在"配了不生效"清单里。

    `serve/` 的读取点消失时，这一节会悄悄退回，而告警不再覆盖它。
    """
    from pepdesign.utils.config import UNIMPLEMENTED_SECTIONS

    assert "infer" not in UNIMPLEMENTED_SECTIONS, (
        "infer 又回到了'配了不生效'清单里——说明 serve/ 的读取点没了。"
        "要么把服务补回来，要么把这一节重新登记进 UNIMPLEMENTED_SECTIONS。"
    )

    import pepdesign.serve.app as serve_app
    import pepdesign.serve.engine as serve_engine

    src = (
        pathlib.Path(serve_app.__file__).read_text(encoding="utf-8")
        + pathlib.Path(serve_engine.__file__).read_text(encoding="utf-8")
    )
    for field in ("max_num_seqs", "max_num_batched_tokens", "batch_timeout_ms",
                  "adapter_path", "merged_model_path", "max_model_len"):
        assert field in src, (
            f"InferConfig.{field} 在 serve/ 里找不到读取点——"
            f"它现在是'配了不生效'的字段了"
        )


def test_load_config_is_quiet_when_infer_is_not_configured(caplog):
    """没显式写 `infer:` 就不要告警；默认值不算"配置了"，否则日志会被噪音淹没。"""
    import logging

    with caplog.at_level(logging.WARNING, logger="pepdesign.utils.config"):
        cfg = load_config("configs/sft_lora.yaml")

    assert not [r for r in caplog.records if "infer" in r.message]
    # 但字段本身仍然存在、仍然有默认值（保留而不是删掉，理由见 InferConfig 文档）
    assert cfg.infer.base_model and cfg.infer.max_model_len > 0


def test_grpo_config_defaults_match_documented_values():
    """默认超参与 config.py 和 configs/grpo_lora.yaml 注释里的数字一致。"""
    cfg = GRPOConfig()
    assert cfg.num_generations == 8
    assert cfg.temperature == 1.0
    assert cfg.beta == 0.04
    assert cfg.loss_type == "dr_grpo"
    assert cfg.epsilon == 0.2
    assert cfg.clip_higher == 0.28
    assert cfg.advantage_normalization == "std"
    assert cfg.scale_rewards is True


def test_published_step_counts_match_the_real_configs():
    """文档里印的 SFT 5625 步 / GRPO 375 步必须与 configs/*.yaml 及语料规模算出来的一致。

    必须读 yaml（类默认的 gradient_accumulation_steps 与 yaml 不同），
    语料规模取实际产物中的数字。
    """
    import json

    from pepdesign.utils.config import resolve_path

    sft = load_config("configs/sft_lora.yaml")
    grpo = load_config("configs/grpo_lora.yaml")

    # 语料规模从产物里读，不再写死在测试里。
    # 写死的话，"重新生成了一份更大的语料"会让下面两条断言的失败信息
    # 指向"步数变了、去改文档"，而真正发生的事情是数据变了——
    # 排查方向会被指错。
    card = json.loads(
        resolve_path("data/processed/dataset_card.json").read_text(encoding="utf-8")
    )
    n_sft = len(
        [
            line
            for line in resolve_path("data/processed/sft_train.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
    )
    n_grpo = len(
        [
            line
            for line in resolve_path("data/processed/grpo_prompts.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
    )

    n_sft_steps = estimate_sft_num_training_steps(sft.sft, n_sft)
    n_grpo_steps = estimate_grpo_num_training_steps(grpo.grpo, n_grpo)

    assert n_sft_steps == 5625, (
        f"SFT 步数变了（语料 {n_sft} 条 → {n_sft_steps} 步，"
        f"dataset_card 记的是 {card['counts']['sft']} 条）。"
        f"configs/sft_lora.yaml 的注释里印的是 5625 步，请同步改。"
        f"（注意根源往往不是步数算错了，而是语料规模变了——"
        f"先看 data.sft_size，再看这里的数字。）"
    )
    assert n_grpo_steps == 375, (
        f"GRPO 步数变了（prompt 集 {n_grpo} 条 → {n_grpo_steps} 步）。"
        f"同上，文档需要同步。"
    )

    # warmup：0.03 × 5625 = 168 步。这个数也印在 yaml 的注释里。
    assert build_scheduler_spec(
        sft.sft, num_training_steps=n_sft_steps
    )["num_warmup_steps"] == 168


def test_grpo_save_steps_actually_fires():
    """`save_steps` 必须小于总步数，否则 checkpoint 永远不会产生。

    `save_steps <= 0` 是"从不保存"的哨兵值，不算违规。
    """
    from pepdesign.utils.config import resolve_path

    def count_lines(path: str) -> int:
        return len(
            [
                line
                for line in resolve_path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        )

    grpo = load_config("configs/grpo_lora.yaml")
    grpo_steps = estimate_grpo_num_training_steps(
        grpo.grpo, count_lines("data/processed/grpo_prompts.jsonl")
    )
    if grpo.grpo.save_steps > 0:
        assert grpo.grpo.save_steps < grpo_steps, (
            f"grpo.save_steps={grpo.grpo.save_steps} 大于等于总步数 {grpo_steps}，"
            f"这个 checkpoint 永远不会被保存。要么调小 save_steps，"
            f"要么把语料/epoch 调大。"
        )

    # SFT 同理：save_steps 与 eval_steps 两个都必须在总步数以内。
    sft = load_config("configs/sft_lora.yaml")
    sft_steps = estimate_sft_num_training_steps(
        sft.sft, count_lines("data/processed/sft_train.jsonl")
    )
    for name in ("save_steps", "eval_steps"):
        every = getattr(sft.sft, name)
        if every > 0:
            assert every < sft_steps, (
                f"sft.{name}={every} 大于等于总步数 {sft_steps}，永远不会触发。"
            )


def test_pipeline_sizes_are_configurable_not_hardcoded():
    """`data.sft_size` / `grpo_size` / `calibration_size` 必须真的能改。

    签名默认值必须是 `None`，具体值由 `DataConfig` 决定。
    """
    import inspect

    from pepdesign.data.pipeline import run_pipeline
    from pepdesign.utils.config import DataConfig

    cfg = DataConfig()
    params = inspect.signature(run_pipeline).parameters
    for name in ("sft_size", "grpo_size", "calibration_size"):
        assert hasattr(cfg, name), f"DataConfig 缺字段 {name}"
        assert params[name].default is None, (
            f"run_pipeline 的 {name} 有具体默认值 {params[name].default!r}——"
            f"那会让配置里写的值被忽略。应当留 None，由 cfg 决定。"
        )


def test_pipeline_cli_entry_point_actually_exists():
    """`python -m pepdesign.data.pipeline` 必须真的可执行。

    缺少 `__main__` 时该命令会以退出码 0 什么都不做。
    """
    import subprocess
    import sys

    from pepdesign.data import pipeline

    assert callable(getattr(pipeline, "main", None)), "pipeline 没有 main()"

    result = subprocess.run(
        [sys.executable, "-m", "pepdesign.data.pipeline", "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"--help 失败: {result.stderr[-300:]}"
    # 三个关键开关都要在帮助里，否则用户不知道它们存在
    assert "--download" in result.stdout
    assert "--set" in result.stdout
