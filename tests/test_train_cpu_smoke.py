"""CPU 冒烟：两层随机初始化的小模型，真跑 SFT 和 GRPO 的训练循环。

这个文件是 CPU 验证方案的第二层。

第一层（其余测试文件）验证的是纯逻辑：掩码对齐、优势口径、解析分组、
配置映射。那些覆盖了出错最多的一段，但覆盖不到"训练到底跑不跑得起来"——
梯度有没有传到 LoRA 参数上、`model.generate` 的返回形状对不对、
优化器 step 之后参数变没变、损失函数的两份实现是否一致。

这一层就是来补这几样的：它造一个两层 / hidden=64 的 Llama，
在 CPU 上把完整的训练循环跑几步。秒级完成，不需要显卡。

它不能证明的事（写在最前面，免得被当成训练成功的证据）
---------------------------------------------------------
  * 学得好不好：随机初始化的模型在真语料上当然学不会；
  * 真实规模下的显存/吞吐：3B + 真数据是另一回事；
  * 超参是否合理：这里只跑 2-3 步，任何超参都"能跑"。

它证明的是结构性事实。这些事实在 GPU 上跑同样成立，
所以它是有意义的——不是"在 CPU 上模拟了一遍训练"。

依赖缺失时整文件跳过
--------------------
`importorskip` 写在模块顶层，所以没装 torch 的环境里收集阶段就会跳过，
既不会报错也不会拖慢其余测试。这正是 `conftest.py` 那条
"tests/ 不许 import torch"约定的例外，而这个例外是有意的：
不碰 torch 就永远验证不了训练循环。

peft 是可选的（`importorskip` 只在模块级对 torch/transformers 硬要求）：
生产路径是手写 LoRA（`models/lora_layer.py`），peft 只在做数值对照时用。
没装 peft 时只有一个对照测试会跳过，其余全跑——这正是"手写实现
不依赖 peft"的可观测证明。
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="CPU 冒烟需要 torch")
pytest.importorskip("transformers", reason="CPU 冒烟需要 transformers")

from pepdesign.models.loader import (  # noqa: E402
    build_tiny_model,
    chat_tokenize,
    load_tokenizer,
    tiny_lora_config,
)
from pepdesign.models.lora import inject_lora  # noqa: E402
from pepdesign.train.checkpoint import list_checkpoints  # noqa: E402
from pepdesign.train.grpo import (  # noqa: E402
    _cross_check_loss,
    _dropout_disabled,
    build_reference_model,
    run_grpo_smoke,
    run_step,
)
from pepdesign.train.sft import collate_sft, run_sft_smoke  # noqa: E402
from pepdesign.utils.config import LoRAConfig, load_config  # noqa: E402

pytestmark = pytest.mark.smoke

#: 冒烟用的 tokenizer 取自冒烟配置里写的那个模型，而不是在这里再抄一份 id。
#: 抄一份的坏处很具体：`run_*_smoke` 走的是配置，测试 fixture 走的是常量，
#: 换基座时只改了一边（两份都"看起来对"），于是冒烟验的是另一个 tokenizer 的模板，
#: 而它恰恰是这套测试里唯一盯着模板行为的东西。派生出来就不存在漂移。
SMOKE_MODEL_ID = load_config("configs/smoke_cpu.yaml").sft.model_name_or_path


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def smoke_cfg():
    return load_config("configs/smoke_cpu.yaml")


@pytest.fixture(scope="module")
def tokenizer():
    """真实 tokenizer。

    为什么不去造一个假的：SFT 和 GRPO 都依赖 chat template，
    而模板是 tokenizer 自带的。假 tokenizer 没有模板，
    `apply_chat_template` 直接不可用——那样测出来的就不是真实路径了。
    这里取的 tokenizer 约 17MB，比模型权重小两个多数量级。

    必须和 `configs/smoke_cpu.yaml` 里写的那个是同一个模型：
    `run_*_smoke` 走 `--smoke` 分支时也是照配置的 `model_name_or_path`
    取 tokenizer 的。两边不一致的话，"chat template 前缀一致"那条测试
    验的就是另一个模板，而它正是这套冒烟里唯一钉住模板行为的断言。

    取不到（没网、没缓存）就跳过：这一层是"有依赖就多验一点"，
    不该因为下载失败让整个测试套件变红。
    """
    try:
        return load_tokenizer(SMOKE_MODEL_ID)
    except Exception as exc:  # noqa: BLE001 - 网络/缓存问题都算"取不到"
        pytest.skip(f"取不到 {SMOKE_MODEL_ID} 的 tokenizer，跳过冒烟：{type(exc).__name__}: {exc}")


@pytest.fixture(scope="module")
def tiny(tokenizer):
    """一个已注入 LoRA 的两层小模型，模块级共享。

    只给只读的测试用。 它是 module scope，所以任何会改到模型状态
    的测试（`backward()` 写 `.grad`、`normal_()` 改权重、`merge()`、
    再注入一次 LoRA）都会把状态泄漏给同文件里后面的测试，
    而症状是"某个测试单独跑能过、整套跑就红"——排查起来很费时间。

    需要改模型状态的测试用 `fresh_tiny`。
    """
    model = build_tiny_model(vocab_size=len(tokenizer))
    return inject_lora(model, tiny_lora_config())


@pytest.fixture
def fresh_tiny(tokenizer):
    """每个测试一份全新的、已注入 LoRA 的小模型。函数级，用完即弃。

    多出来的开销是造一个两层 hidden=64 的模型（毫秒级），
    换来的是"测试之间零状态共享"——对会写梯度 / 改权重 / merge 的测试来说
    这不是优化问题，是正确性问题。
    """
    model = build_tiny_model(vocab_size=len(tokenizer))
    return inject_lora(model, tiny_lora_config())


# ---------------------------------------------------------------------------
# tokenizer / chat template：后面所有测试的前提
# ---------------------------------------------------------------------------
def test_tokenizer_has_a_chat_template(tokenizer):
    """chat template 是整个训练语料格式的载体，没有它什么都做不了。

    本项目基座（LOGOS-3B）自带模板，所以这条测试在正常环境下必然通过——
    它的价值在于如果换成别的模型，这里会立刻失败，
    而不是等到 `apply_chat_template` 在训练循环深处抛一个指向 Jinja 的错误。
    """
    assert getattr(tokenizer, "chat_template", None), "tokenizer 没有 chat_template"
    assert tokenizer.pad_token is not None, "pad_token 未设置，padding 会失败"


def test_chat_template_prefix_is_stable_with_generation_prompt(tokenizer):
    """`add_generation_prompt=True/False` 产出的前缀必须一致。

    整个完成掩码的正确性都建立在这条不变量上：SFT 阶段用
    "前半段 tokenize 一次 + 整段 tokenize 一次"来定位答案区间，
    GRPO 阶段用 prompt 长度来切 completion。模板一旦在这两种调用方式下
    产出不同的前缀，掩码就会错位——而掩码错位不会报错，
    只会让模型在错的 token 上算 loss。

    这条断言在 `calibration.py` 和 `sft.py:run_sft_smoke` 里各有一份运行时
    检查（遇到就抛错）；这里提前一次性验证，见 `test_calibration.py` 之外
    多一层保护。
    """
    messages = [
        {"role": "system", "content": "你是多肽设计专家。"},
        {"role": "user", "content": "设计一条 10 残基的抗菌肽。"},
    ]
    prompt_ids = chat_tokenize(tokenizer, messages, add_generation_prompt=True)
    full_ids = chat_tokenize(
        tokenizer,
        [*messages, {"role": "assistant", "content": "### 设计序列\nKWKLFKKIEK\n### 结束"}],
    )
    assert full_ids[: len(prompt_ids)] == prompt_ids


def test_generation_stop_ids_includes_the_template_terminator(tokenizer):
    """停止集必须含模板的回合终止符，而不只是 eos。

    回归用例：Qwen 的模板终止符恰好就是 eos，
    于是"只传 {pad, eos}"看起来一直是对的；换成 Llama-3 模板的基座之后
    两者分家（`<|eot_id|>` vs `<|end_of_text|>`），`generate` 不再知道
    该在哪里停——rollout 会跑满 `max_completion_length`，多出来的那段
    还会被当成正常 completion 拿到 loss 与优势，而且完全不报错。
    """
    from pepdesign.models.loader import generation_stop_ids

    stop = generation_stop_ids(tokenizer)
    assert tokenizer.eos_token_id in stop

    probe = [{"role": "user", "content": "x"}]
    prompt_ids = chat_tokenize(tokenizer, probe, add_generation_prompt=True)
    full_ids = chat_tokenize(
        tokenizer, [*probe, {"role": "assistant", "content": "y"}]
    )
    assert full_ids[: len(prompt_ids)] == prompt_ids
    terminator = full_ids[-1]

    # 如果基座的模板终止符就是 eos，这条测试会退化成恒真。
    # 显式断言它不同，是为了让"这条测试还有意义"本身可验证——
    # 换回一个终止符等于 eos 的基座时，这里会红，提醒你重新评估。
    assert terminator != tokenizer.eos_token_id, (
        "本基座的模板终止符恰好就是 eos，这条测试已失去意义；"
        "请确认 generation_stop_ids 是否还需要存在。"
    )
    assert terminator in stop, "模板终止符没有进停止集，rollout 不会在回合边界停下"


# ---------------------------------------------------------------------------
# 小模型本身
# ---------------------------------------------------------------------------
def test_tiny_model_matches_logos_3b_gqa_shape(tiny):
    """GQA 结构要和 LOGOS-3B 一致（KV 头数 < 查询头数），架构类也要是 Llama。

    不一致的话 LoRA 注入走的是另一条路径：q_proj 的输出维度在 GQA 下是
    k/v 的三倍（真实基座 24 头 / 8 KV 头），用
    `num_key_value_heads == num_attention_heads` 的模型冒烟能过，
    但证明的不是真实模型上的那条路径。

    断言 `LlamaForCausalLM` 是因为换基座时最容易漏的一处就是架构类：
    小模型照样能建出来、LoRA 照样注入，只是测的成了另一个模型的代码路径。
    """
    cfg = tiny.config
    assert type(tiny).__name__ == "LlamaForCausalLM"
    assert cfg.num_key_value_heads < cfg.num_attention_heads
    assert cfg.num_hidden_layers == 2


def test_tiny_lora_has_few_trainable_params(tiny):
    """LoRA 注入后大部分参数必须是冻住的。

    两者相等说明 `target_modules` 一个都没匹配上——部分实现对此
    不报错，只是把整个模型当成可训练对象。
    `inject_lora` 里那句断言就是为此存在的，这里从外部再确认一次。
    """
    trainable = sum(p.numel() for p in tiny.parameters() if p.requires_grad)
    total = sum(p.numel() for p in tiny.parameters())
    assert 0 < trainable < total * 0.2, f"可训练参数占比过高: {trainable}/{total}"


def test_manual_lora_matches_peft(fresh_tiny):
    """手写 LoRA 和 peft 必须数值等价。 这是整份手写实现的正确性基准。

    对照测试的目的
    --------------------
    自写一个 LoRA 最容易犯的错是"看起来对"：
    `forward` 里 A/B 的顺序反了（应该先 A 降维、再 B 升维）、
    缩放乘错了地方、dropout 加在了 A 和 B 之间、B 忘了初始化为 0。
    这些全都不会报错，训练也照样能跑——loss 会降，
    只是学出来的东西不是 LoRA 论文里的那个。

    验证办法是唯一的客观办法：把同一份 A/B 权重分别灌进两个后端，
    在同一个输入上比前向输出。权重相同、输入相同、输出必须逐元素相等；
    只要公式有一处不同，这里就会红。

    测试里刻意用 `lora_dropout=0`：dropout 是随机的，
    开着它两个后端的随机掩码不同，比不出任何东西（这是
    `pytest.importorskip` 之外第二个"这个测试到底在测什么"的说明）。

    没装 peft 时整条跳过——手动实现的生产路径不依赖 peft，
    所以缺它不影响其余任何测试。
    """
    pytest.importorskip("peft", reason="数值对照需要 peft（生产路径不需要）")

    from pepdesign.models.lora import _inject_lora_peft
    from pepdesign.models.lora_layer import iter_lora_layers

    vocab = fresh_tiny.config.vocab_size
    lora_cfg = LoRAConfig(
        r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj"], backend="manual",
    )

    manual = inject_lora(build_tiny_model(vocab_size=vocab), lora_cfg)
    with_peft = _inject_lora_peft(
        build_tiny_model(vocab_size=vocab),
        lora_cfg.model_copy(update={"backend": "peft"}),
    )

    # 让两份 A/B 完全一致：从手写那份出发，复制到 peft 那份。
    manual_layers = list(iter_lora_layers(manual))
    assert manual_layers, "手写实现没有找到任何 LoRALinear"
    with torch.no_grad():
        for _, layer in manual_layers:
            layer.lora_B.weight.normal_(0, 0.02)   # 否则 ΔW=0，等价性无从检验

        peft_params = dict(with_peft.named_parameters())
        for path, layer in manual_layers:
            for tag in ("A", "B"):
                # peft 的实际 key 形态（多两段）：
                #   base_model.model.<path>.lora_A.default.weight
                # 手写实现的 key 是干净的 `<path>.lora_A.weight`。
                # 这条断言把两者的差异本身钉住——它就是
                # `to_peft_state_dict` 存在的理由。
                key = (
                    f"base_model.model.{path}"
                    f".lora_{tag}.default.weight"
                )
                assert key in peft_params, (
                    f"peft 那边的 key 命名和手写实现不一致: 找不到 {key}。"
                    f"实际有的是: {[k for k in peft_params if 'lora_' in k][:2]}..."
                )
                peft_params[key].copy_(getattr(layer, f"lora_{tag}").weight)

        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        out_manual = manual(input_ids=ids).logits
        out_peft = with_peft(input_ids=ids).logits

    assert torch.allclose(out_manual, out_peft, atol=1e-5), (
        "手写 LoRA 与 peft 的前向输出不一致——两个实现里有一个是错的。"
        f"最大偏差 {(out_manual - out_peft).abs().max().item():.3e}"
    )


def test_manual_lora_peft_state_dict_roundtrip(tiny):
    """`to_peft_state_dict` / `from_peft_state_dict` 必须能互相还原。

    peft 的 key 比本实现多两处：
      1. 前缀 `base_model.model.`（`PeftModel` 又包了一层 `.model`）；
      2. `lora_A` 与 `weight` 之间的 adapter 名字段 `.default.`。
    第 2 处最容易漏，因为漏了之后 key 看起来仍然"很像对的"。

    把这个转换写成显式函数而不是"保存时直接按 peft 命名"，
    是为了让格式差异本身可见、可测——而不是藏在保存路径里，
    换一个 peft 版本就静默读不出来。
    """
    from pepdesign.models.lora_layer import (
        from_peft_state_dict,
        manual_lora_state_dict,
        to_peft_state_dict,
    )

    original = manual_lora_state_dict(tiny)
    assert original, "没有抽出任何 adapter 权重"
    assert all(k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight") for k in original)
    assert all(not k.startswith("base_model.model.") for k in original)

    as_peft = to_peft_state_dict(original)
    assert all(k.startswith("base_model.model.") for k in as_peft)
    assert all(".default.weight" in k for k in as_peft), (
        "peft 的 key 里 lora_A/lora_B 和 weight 之间还有一个 adapter 名字段"
    )
    assert len(as_peft) == len(original)

    roundtrip = from_peft_state_dict(as_peft)
    assert set(roundtrip) == set(original)
    for k in original:
        assert torch.equal(roundtrip[k], original[k])

    # 不带前缀、也不带 adapter 名字段的 key 要原样保留
    # （手工整理过的权重文件、或老版本 peft 的命名，都很常见）
    legacy = {"model.layers.0.q_proj.lora_A.weight": torch.zeros(2, 3)}
    assert set(from_peft_state_dict(legacy)) == {"model.layers.0.q_proj.lora_A.weight"}


def test_load_adapter_is_trainable_only_when_asked(tiny, tmp_path):
    """`load_adapter` 的 `is_trainable` 开关：默认必须把 adapter 冻住。

    这条锁的是一个在本项目里真实发生过的 bug：`run_grpo` 从磁盘加载
    SFT adapter 时没传 `is_trainable=True`，而当时的加载实现默认是
    推理语义（adapter 权重随即 `requires_grad=False`）。
    于是加载不报错、前向能跑、采样也正常，只有 `run_grpo` 收集优化器参数
    时才发现列表是空的——整机第一次跑起来才炸。

    两层测试原本都拦不住它，因为冒烟测试走的是 `inject_lora`
    （它自己会把 `requires_grad` 设对），**从来没有经过"从目录里读回一个
    已训好的 adapter"这条路**。这个往返正是本测试补上的那一段。

    两个方向都断言：默认必须冻住（推理语义没被改坏），显式 True 必须可训
    （续训语义没被改坏）——只测一个方向的话，把默认值直接翻成 True
    也能"通过"。

    手写后端的加载是"先注入、再灌权重"（见 lora_layer.load_manual_adapter
    的说明：按权重文件的 key 现场建层会绕过 target_modules 校验），
    所以下面每次都先 `inject_lora` 出一个骨架再加载。
    """
    from pepdesign.models.lora import load_adapter, save_adapter

    n_injected = sum(p.numel() for p in tiny.parameters() if p.requires_grad)
    assert n_injected > 0

    save_adapter(tiny, tmp_path / "adapter", cfg=tiny_lora_config())
    assert (tmp_path / "adapter" / "adapter_config.json").exists()
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists()

    vocab = tiny.config.vocab_size

    def _fresh_skeleton():
        return inject_lora(build_tiny_model(vocab_size=vocab), tiny_lora_config())

    # 默认（不传）= 推理语义：adapter 冻住。
    frozen = load_adapter(_fresh_skeleton(), tmp_path / "adapter")
    assert sum(p.numel() for p in frozen.parameters() if p.requires_grad) == 0

    # 显式 True = 续训语义：可训练参数量要和当初注入时一致——
    # 只断言 "> 0" 的话，把基座参数也放开（而不是只放开 adapter）同样能过，
    # 而那意味着 GRPO 会去微调整个基座。
    trainable = load_adapter(
        _fresh_skeleton(), tmp_path / "adapter", is_trainable=True
    )
    assert sum(p.numel() for p in trainable.parameters() if p.requires_grad) == n_injected


def test_save_adapter_refuses_without_cfg(tiny, tmp_path):
    """手写后端保存时必须传 cfg，否则报错而不是写半个 adapter。

    `adapter_config.json` 里存着 r / alpha / target_modules，
    没有它就无法判断一份权重该挂在哪些层上——加载时只能靠猜。
    与其写出一个缺配置的目录（下游会以一种很难查的方式失败），
    不如在这里直接拒绝。
    """
    from pepdesign.models.lora import save_adapter

    with pytest.raises(ValueError, match="必须传 cfg"):
        save_adapter(tiny, tmp_path / "adapter_no_cfg")


def test_adapter_config_records_everything_needed_to_reload(tiny, tmp_path):
    """`adapter_config.json` 必须自足：光有它就能重建出该挂哪些层。

    存 `r` 和 `lora_alpha` 而不是只存 `scaling`：合并时要用 `alpha/r` 的
    精确形式，而 `2.0` 反推不出 (8,16) 还是 (16,32)。
    """
    import json

    from pepdesign.models.lora import save_adapter

    cfg = tiny_lora_config()
    save_adapter(tiny, tmp_path / "adapter", cfg=cfg)

    with open(tmp_path / "adapter" / "adapter_config.json", encoding="utf-8") as f:
        saved = json.load(f)

    assert saved["r"] == cfg.r
    assert saved["lora_alpha"] == cfg.lora_alpha
    assert saved["target_modules"] == cfg.resolved_target_modules()
    assert saved["scaling"] == cfg.scaling
    assert saved["num_lora_modules"] > 0


def test_export_peft_adapter_round_trips_to_clean_keys(tiny, tmp_path):
    """手写格式 → peft 格式的导出：键名要变成 peft 的，权重值要逐元素不变。

    这一步是 vLLM 后端的前置条件——vLLM 按 peft 的键名
    （`base_model.model.*.lora_A.default.weight`）找权重，
    而本项目的训练产物用的是干净路径。键名对不上时 vLLM 报的是
    "找不到参数"，看不出是格式问题。

    两件事都要钉，而且第二件更容易被漏掉：
      * 键名形状：前缀 + `.default.` 名字段，缺一个都会加载失败；
      * 数值：转换只改 key、不碰张量。用 `torch.equal` 逐元素比，
        因为"形状对、值错了"是这类转换最危险的失败——它不会报错，
        只会让模型输出悄悄变成另一个东西。
    """
    import torch
    from safetensors.torch import load_file

    from pepdesign.models.lora import save_adapter
    from pepdesign.models.lora_layer import (
        PEFT_KEY_PREFIX,
        export_peft_adapter,
        load_manual_adapter,
        manual_lora_state_dict,
    )

    cfg = tiny_lora_config()
    src = save_adapter(tiny, tmp_path / "adapter", cfg=cfg)
    original = manual_lora_state_dict(tiny)

    dst = export_peft_adapter(
        src, tmp_path / "adapter_peft", base_model_name_or_path="Qwen/Qwen3-0.6B"
    )
    exported = load_file(str(dst / "adapter_model.safetensors"))

    # 1) 键名：全部带 peft 前缀，且插了 adapter 名字段
    assert exported, "导出结果为空"
    for key in exported:
        assert key.startswith(PEFT_KEY_PREFIX), key
        assert ".lora_A.default.weight" in key or ".lora_B.default.weight" in key, key

    # 2) 数值：张量一一对应且逐元素相等
    assert len(exported) == len(original)
    for key, value in original.items():
        module_path, _, tail = key.rpartition(".lora_")
        peft_key = f"{PEFT_KEY_PREFIX}{module_path}.lora_{tail.rsplit('.', 1)[0]}.default.weight"
        assert peft_key in exported, peft_key
        assert torch.equal(exported[peft_key], value), f"{peft_key} 的数值变了"

    # 3) 导出物必须是 peft 的 schema，不能是本模块那个 format 标记——
    #    vLLM 读的是 peft_type / r / target_modules，看不见 format。
    import json

    with open(dst / "adapter_config.json", encoding="utf-8") as f:
        peft_cfg = json.load(f)
    assert peft_cfg["peft_type"] == "LORA"
    assert peft_cfg["r"] == cfg.r
    assert peft_cfg["target_modules"] == cfg.resolved_target_modules()
    assert "format" not in peft_cfg

    # 4) 原始目录没有被改坏：训练侧的加载仍然可用
    reloaded = load_manual_adapter(tiny, src, is_trainable=False)
    assert reloaded is not None


def test_export_peft_adapter_rejects_a_peft_source(tiny, tmp_path):
    """把已经导出过的目录再导出一次，要报错而不是静默成功。

    静默成功的后果是产出一个"看起来对、其实键名被套了两层前缀"的目录，
    而 vLLM 到加载时才会报"找不到参数"。
    """
    import pytest as _pytest

    from pepdesign.models.lora import save_adapter
    from pepdesign.models.lora_layer import export_peft_adapter

    cfg = tiny_lora_config()
    src = save_adapter(tiny, tmp_path / "adapter", cfg=cfg)
    once = export_peft_adapter(src, tmp_path / "peft1", base_model_name_or_path="x")

    with _pytest.raises(ValueError, match="不需要再导出"):
        export_peft_adapter(once, tmp_path / "peft2", base_model_name_or_path="x")


# ---------------------------------------------------------------------------
# LoRA 完整性：合并导出、adapter 校验、后端识别
# ---------------------------------------------------------------------------
def _non_zero_lora(model) -> None:
    """把 B 打成非零，保证 Delta W 不是 0（否则合并前后都一样，测不出问题）。"""
    for name, param in model.named_parameters():
        if name.endswith("lora_B.weight"):
            torch.nn.init.normal_(param, mean=0.0, std=0.02)


def test_manual_merge_and_save_reloads_as_standard_model(fresh_tiny, tmp_path):
    """合并导出的模型必须能被标准 `from_pretrained` 完整读回。

    这是本项目最容易静默失败的一条路径：只 merge 不 unwrap 时，
    保存出来的 key 是 `...q_proj.base.weight` / `...lora_A.weight`，
    标准加载器会把 `...q_proj.weight` 记成 MISSING 并重新随机初始化。
    文件能加载、不报错，但权重已经全丢了。

    测试分两步，缺一不可：
      1. 检查保存文件里没有 `.base.` / `.lora_` 这些非标准 key；
      2. 真的 `from_pretrained` 一次，比较合并前后的 logits。
    只查 key 不查数值，仍然可能放过"key 对了但值错了"。
    """
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    from pepdesign.models.lora import merge_and_save

    _non_zero_lora(fresh_tiny)
    fresh_tiny.eval()
    vocab = fresh_tiny.config.vocab_size
    ids = torch.randint(0, vocab, (2, 8))
    with torch.no_grad():
        before = fresh_tiny(input_ids=ids).logits

    out = merge_and_save(
        fresh_tiny, tmp_path / "merged", cfg=tiny_lora_config()
    )

    saved = load_file(str(out / "model.safetensors"))
    assert saved, "合并模型没有写出权重"
    assert not any(".base." in key for key in saved), "保存了 LoRALinear 的 base.* 非标准 key"
    assert not any(".lora_" in key for key in saved), "保存了 adapter 参数"
    assert any(key.endswith(".q_proj.weight") for key in saved), (
        "没有标准 q_proj.weight，说明 LoRALinear 没有被 unwrap 成 nn.Linear"
    )

    reloaded = AutoModelForCausalLM.from_pretrained(out)
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(input_ids=ids).logits
    assert torch.allclose(before, after, atol=1e-5, rtol=1e-5), (
        "合并保存后重新加载的输出与合并前不一致——权重在保存/加载链路里丢了"
    )


def test_manual_adapter_restores_saved_alpha(tokenizer, tmp_path):
    """加载 adapter 时必须用保存时的 alpha，而不是当前骨架的 alpha。

    `r` 不一致会被形状检查拦住，但 `alpha` 不一致是形状兼容的：
    SFT 保存 alpha=8、GRPO 骨架 alpha=4 时，权重能装进去，
    但缩放从 2.0 变成 1.0，输出整体错了，loss 却照常下降。
    """
    from pepdesign.models.lora import load_adapter, save_adapter

    vocab = len(tokenizer)

    def _make(alpha):
        cfg = LoRAConfig(
            r=4,
            lora_alpha=alpha,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=0.0,
            backend="manual",
        )
        model = inject_lora(build_tiny_model(vocab_size=vocab), cfg)
        return model, cfg

    source, cfg_source = _make(8)
    _non_zero_lora(source)
    source.eval()
    adapter_dir = save_adapter(source, tmp_path / "adapter", cfg=cfg_source)

    target, _ = _make(4)
    load_adapter(target, adapter_dir, is_trainable=False)
    target.eval()

    layer = target.model.layers[0].self_attn.q_proj
    assert layer.alpha == 8, f"没有恢复保存时的 alpha：{layer.alpha}"
    assert layer.scaling == 2.0, f"没有恢复保存时的缩放：{layer.scaling}"

    ids = torch.randint(0, vocab, (2, 8))
    with torch.no_grad():
        y_source = source(input_ids=ids).logits
        y_target = target(input_ids=ids).logits
    assert torch.allclose(y_source, y_target, atol=1e-5, rtol=1e-5), (
        "同一份 adapter 在 alpha 恢复后输出仍不一致"
    )


def test_manual_adapter_rejects_missing_tensor(fresh_tiny, tmp_path):
    """不完整的 adapter 必须报错，不能静默加载。

    缺一个 `lora_A.weight` 时，那一层会保留注入时的随机 A，
    再乘上从文件读到的非零 B —— 得到一个随机的 Delta W。
    这比"没有 adapter"更危险：模型能跑，loss 也正常，但那一层是噪声。
    """
    from safetensors.torch import load_file, save_file

    from pepdesign.models.lora import save_adapter
    from pepdesign.models.lora_layer import load_manual_adapter

    cfg = tiny_lora_config()
    adapter_dir = save_adapter(fresh_tiny, tmp_path / "adapter", cfg=cfg)
    weights_path = adapter_dir / "adapter_model.safetensors"
    state = load_file(str(weights_path))
    victim = sorted(state)[0]
    del state[victim]
    save_file(state, str(weights_path))

    skeleton = inject_lora(
        build_tiny_model(vocab_size=fresh_tiny.config.vocab_size), cfg
    )
    with pytest.raises(ValueError, match=r"不匹配|缺少"):
        load_manual_adapter(skeleton, adapter_dir, is_trainable=False)


def test_manual_merge_rejects_quantized_base():
    """4-bit/8-bit 基座不能直接 merge，必须给出明确错误而不是写坏权重。

    量化层的 `weight.dtype` 是 `uint8`，浮点 Delta W 无法通过 `add_` 写进去。
    这里用一个显式 uint8 的普通 Linear 模拟这条路径——不依赖 bitsandbytes，
    但它锁住的检查逻辑与真实 4-bit 基座是同一条。
    """
    from pepdesign.models.lora_layer import build_lora_linear_class

    LoRALinear = build_lora_linear_class()
    base = torch.nn.Linear(8, 4, bias=False)
    # uint8 的 Parameter 不能要求梯度；模拟量化基座时先冻住再换 dtype。
    base.weight.requires_grad_(False)
    base.weight.data = base.weight.data.to(torch.uint8)
    layer = LoRALinear(base, r=2, alpha=2, dropout=0.0)

    with pytest.raises(RuntimeError, match=r"量化基座|不是浮点"):
        layer.merge()


def test_manual_backend_rejects_bias(tokenizer):
    """手写后端没有实现 LoRA bias，配置写了就必须拒绝，不能静默忽略。"""
    cfg = LoRAConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.0,
        bias="all",
        backend="manual",
    )
    model = build_tiny_model(vocab_size=len(tokenizer))
    with pytest.raises(NotImplementedError, match="bias"):
        inject_lora(model, cfg)


def test_peft_model_is_not_misdetected_as_manual(tokenizer, tmp_path):
    """PEFT 模型不能被当成手写模型。

    peft 的 LoraLayer 也有 `lora_A` / `lora_B`（ModuleDict），
    用 `hasattr` 判断会让 `save_adapter` 走手写分支并报
    `'ModuleDict' object has no attribute 'weight'`。
    """
    pytest.importorskip("peft")
    from pepdesign.models.lora import _inject_lora_peft, _is_manual_model, save_adapter

    cfg = LoRAConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.0,
        backend="peft",
    )
    model = build_tiny_model(vocab_size=len(tokenizer))
    peft_model = _inject_lora_peft(model, cfg)

    assert not _is_manual_model(peft_model), (
        "peft 模型被误判成手写模型——save_adapter/merge_and_save 会走错分支"
    )

    out = save_adapter(peft_model, tmp_path / "peft_adapter", cfg=cfg)
    assert (out / "adapter_config.json").exists()
    assert (out / "adapter_model.safetensors").exists()


# ---------------------------------------------------------------------------
# SFT 冒烟
# ---------------------------------------------------------------------------
def test_sft_smoke_runs_and_updates_weights(smoke_cfg, tokenizer):
    """SFT 链路：数据 → 掩码 → 标签 → 自写循环 → 参数更新。

    `tokenizer` 参数在这里没有被用到，它是用来绑定跳过条件的：
    `run_sft_smoke` 内部自己会去取 tokenizer，取不到时抛的是网络异常，
    而不是 skip。让这个测试和其余测试共用一个 fixture，
    下载失败时整组一起跳过，而不是"一半红了、红的原因还和被测代码无关"。

    三个断言缺一不可：
      * `final_loss` 是有限值 —— 掩码错位、labels 与 logits 长度不匹配
        这类问题经常表现为 loss 变成 nan，而不是抛异常；
      * `tensors_changed > 0` —— 梯度真的传到了 LoRA 参数上。
        lr 配错、参数没接进 optimizer、LoRA 注入位置不对，
        都会让 loss 正常打印而参数一动不动；
      * `trainable_tensors > 0` —— 有东西可训。

    `tensors_changed > 0` 不能当作"学习有效"的证据：
    AdamW 的解耦权重衰减在零梯度下照样会改参数。要验"有梯度时会更新"
    必须人为构造非零梯度，见 `test_nonzero_advantages_produce_gradients`
    和 `test_manual_lora_receives_gradients`。
    """
    cfg = load_config("configs/smoke_cpu.yaml")
    report = run_sft_smoke(cfg, steps=3)

    assert report["num_steps"] == 3
    assert report["final_loss"] is not None
    assert math.isfinite(report["final_loss"]), f"loss 不是有限值: {report['final_loss']}"
    assert report["trainable_tensors"] > 0
    assert report["tensors_changed"] > 0, "3 步之后没有任何参数被更新"
    # 走的是手写后端（smoke_cpu.yaml 里 backend: manual），
    # 所以这条同时证明了"无 trl / 无 peft 的路径能端到端跑通"。
    assert report["note"]


def _lora_layers(model):
    from pepdesign.models.lora_layer import iter_lora_layers

    return list(iter_lora_layers(model))


def _one_batch():
    return collate_sft(
        torch,
        [{"input_ids": [1, 2, 3, 4, 5], "token_mask": [0, 0, 1, 1, 1]}],
        pad_token_id=0,
    )


def test_manual_lora_receives_gradients(fresh_tiny):
    """LoRA 的梯度链路：B 有非零梯度、A 在第 0 步必然为零、基座恒无梯度。

    最后那条"B 有梯度"是应该成立的；"A 为零"乍看像 bug，其实是 LoRA
    的一个数学性质，值得单独讲清楚：

        y = W₀x + (α/r)·B·A·x
        ∂L/∂B = (α/r)·(∂L/∂y)·(A x)ᵀ        ← 与 B 无关，非零
        ∂L/∂A = (α/r)·Bᵀ·(∂L/∂y)·xᵀ        ← 含 B，而 B₀ = 0 → 恒为零

    即：B 初始化为 0 之后，第一步只有 B 会被更新，A 一动不动；
    等 B ≠ 0 了，A 才开始有梯度。这不是实现问题，而是"B 初始化为 0"
    这个设计的直接推论。知道这一点就不会误判——看到 A.grad 全零
    而去"修"计算图，只会把正确的实现改坏。

    另外两个方向分别对应两类不同的错误：
      * `grad is None` → 计算图断了（`no_grad` 漏在外面之类）；
      * 基座 `grad is not None` → 忘了冻结，"LoRA 训练"实际在改全量参数。

    随机初始化的小模型在 GRPO 里会因为优势全零而梯度为零
    （见 test_train_advantages 的说明），但 SFT 的交叉熵 loss
    对任何输入都有非零梯度，所以这里可以放心断言"梯度不全是零"。
    """
    examples = [{"input_ids": [1, 2, 3, 4, 5], "token_mask": [0, 0, 1, 1, 1]}]
    out = fresh_tiny(**collate_sft(torch, examples, pad_token_id=0))
    out.loss.backward()

    pairs = _lora_layers(fresh_tiny)
    assert pairs, "没有找到任何 LoRALinear"

    _, first = pairs[0]
    assert first.lora_B.weight.grad is not None, "lora_B 没有梯度，计算图断了"
    assert torch.count_nonzero(first.lora_B.weight.grad) > 0, "lora_B 的梯度全为零"

    assert first.lora_A.weight.grad is not None, "lora_A 的梯度没被算出来"
    assert torch.count_nonzero(first.lora_A.weight.grad) == 0, (
        "A 在第 0 步竟然有非零梯度——说明 B 的初始化不是全零，"
        "那样训练一开始模型就不等于基座了（见 test_manual_lora_starts_as_identity）"
    )

    # 基座权重一个都不能有梯度
    for path, layer in pairs:
        assert layer.base.weight.grad is None, f"{path} 的基座权重被训练了"


def test_lora_a_gets_gradients_once_b_is_nonzero(fresh_tiny):
    """B 变非零之后，A 必须开始拿到梯度。

    接着上一条的推论：如果 A 永远拿不到梯度，那 LoRA 就退化成
    "只训练 B"（一个秩 ≤ r 的输出侧修正），表达能力大打折扣——
    而训练看起来完全正常，loss 也会降，只是效果比真 LoRA 差。

    构造方式：手动把 B 设成非零（模拟"已经训过一步"），再反传一次。
    """
    with torch.no_grad():
        for _, layer in _lora_layers(fresh_tiny):
            layer.lora_B.weight.normal_(0, 0.02)

    out = fresh_tiny(**_one_batch())
    out.loss.backward()

    _, first = _lora_layers(fresh_tiny)[0]
    assert torch.count_nonzero(first.lora_A.weight.grad) > 0, (
        "B 已经非零了，A 还是没有梯度——计算图里 A 可能被绕过了"
    )


def test_manual_lora_starts_as_identity(fresh_tiny):
    """训练第 0 步，注入了 LoRA 的模型必须与原模型前向输出完全一致。

    这是"B 初始化为 0"那条设计的可观测结论，也是 LoRA 能安全地
    接在预训练模型上的原因：初始状态是恒等变换，不会一上来就把
    基座带偏。如果 A、B 都随机初始化，这里立刻红。

    做法是"同一个模型、注入前后各跑一次"而不是"造两个模型比权重"：
    后者要处理 `.base.` 这一层路径变化，而且 `named_parameters()` 的
    顺序会被注入打乱（每层多出 A/B 两个参数），
    按位置 zip 会错位，本测试的第一版曾因此失败。
    比前向输出既不依赖参数顺序，也正好就是"恒等变换"这个 claim 本身。
    """
    from pepdesign.models.lora_layer import inject_manual_lora

    # 注意用的是没注入过的干净模型：`fresh_tiny` 已经注入过了，
    # 在它上面再注入一次会被 `inject_manual_lora` 的重复注入守卫拦下。
    model = build_tiny_model(vocab_size=fresh_tiny.config.vocab_size)

    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        before = model(input_ids=ids).logits.clone()

    # 原地替换子模块
    inject_manual_lora(model, tiny_lora_config())
    with torch.no_grad():
        after = model(input_ids=ids).logits.clone()

    assert torch.allclose(before, after, atol=1e-6), (
        f"注入 LoRA 改变了前向输出（最大偏差 "
        f"{(before - after).abs().max().item():.3e}）——B 初始化不是全零"
    )


def test_double_injection_is_rejected(fresh_tiny):
    """重复注入 LoRA 必须报错，而且报错信息要指向真正的原因。

    不拦的话有两种后果，都很隐蔽：
      * 目标模块名不在注入后的路径里 → 报"一个都没匹配上"，
        把人引去查模块名拼写（错误的排查方向）；
      * 目标模块名恰好是 `base` → `LoRALinear.base` 是个 nn.Linear，
        LoRA 会嵌套进 LoRA，参数量翻倍、梯度路径扭曲，
        而模型不报错、训练照跑。

    `fresh_tiny` 是已经注入过的模型，正好是这条测试要的输入。
    """
    from pepdesign.models.lora_layer import inject_manual_lora

    with pytest.raises(RuntimeError, match="已经注入过 LoRA"):
        inject_manual_lora(fresh_tiny, tiny_lora_config())


def test_manual_lora_merge_is_exact_and_reversible(fresh_tiny):
    """merge 是精确的，且可以精确还原。

    很多人以为 merge "会掉精度"——实际不会：`W' = W₀ + (α/r)·B·A`
    就是一次矩阵加法，没有近似。精度损失发生在 4bit 量化基座上，
    和 merge 无关。这条测试把"精确"这个说法钉住。

    unmerge 能还原是它的直接推论，但值得单独断言：
    还原不回去说明 merge 时用了错的缩放（比如漏了 α/r），
    而那种错误在前向里看不出来——因为 merge 之后走的是另一条分支。
    """
    import torch

    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        # 把 A/B 设成非零，否则 ΔW=0、merge 与不 merge 本来就一样
        for _, layer in _lora_layers(fresh_tiny):
            layer.lora_B.weight.normal_(0, 0.02)
        before = fresh_tiny(input_ids=ids).logits.clone()

    from pepdesign.models.lora_layer import merge_manual_lora, unmerge_manual_lora

    merge_manual_lora(fresh_tiny)
    with torch.no_grad():
        merged = fresh_tiny(input_ids=ids).logits.clone()
    assert torch.allclose(before, merged, atol=1e-5), "merge 改变了前向输出"

    unmerge_manual_lora(fresh_tiny)
    with torch.no_grad():
        restored = fresh_tiny(input_ids=ids).logits.clone()
    assert torch.allclose(before, restored, atol=1e-5), "unmerge 没能还原"


def test_double_merge_is_rejected(fresh_tiny):
    """重复 merge 必须报错。

    merge 两次等于把 ΔW 加了两遍，而模型不会报错：
    前向照样跑、loss 照样是个有限的数，只是结果悄悄错了。
    这类"不报错但结果错"的操作必须有显式守卫。
    """
    from pepdesign.models.lora_layer import merge_manual_lora

    merge_manual_lora(fresh_tiny)
    with pytest.raises((RuntimeError, Exception), match="merge"):
        merge_manual_lora(fresh_tiny)


def test_build_trainer_runs_length_preflight(smoke_cfg, tokenizer, tiny, caplog):
    """`build_trainer` 自己会跑长度预检，且真的截断时必须出声。

    `max_seq_length` 配小了是本项目里最安静的一类错误：不抛异常、
    loss 曲线完全正常，只是超长样本被静默丢弃，
    模型学的是残缺的答案。所以这里的断言是warning 必须出现，
    而不是"这个函数没崩"——对一条只负责报警的检查来说，后者毫无意义。

    `max_seq_length=16` 是故意调小的：语料真实长度是 323–485 token
    （`--dry-run` 的抽样：mean 443、p95 474），必然触发截断分支。
    """
    import logging

    from pepdesign.train.sft import build_trainer

    cfg = smoke_cfg.model_copy(deep=True)
    cfg.sft.max_seq_length = 16

    with caplog.at_level(logging.WARNING, logger="pepdesign.train.sft"):
        built = build_trainer(cfg, tokenizer=tokenizer, model=tiny)

    # 现在返回的是 (model, examples, tokenizer, device, plan)，不再是 trainer 对象
    assert len(built) == 5
    assert any("max_seq_length" in r.message for r in caplog.records), (
        "调小 max_seq_length 之后没有出现截断告警——预检没被调用，"
        f"或者告警条件失效了。实际日志: {[r.message for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# GRPO 冒烟
# ---------------------------------------------------------------------------
def test_loss_cross_check_torch_matches_pure_python(smoke_cfg):
    """torch 版损失与纯 Python 版在同一组输入上必须数值一致。

    这是整个方案里最关键的一条断言，因为它防的是"两份实现分家"：
    `train/advantages.py` 里的 `clipped_surrogate` / `kl_penalty` /
    `loss_denominator` 是给日志和无 GPU 测试用的纯函数，
    `train/grpo.py:grpo_loss` 里是真正反传的 torch 实现。

    同一公式写两遍的风险是它们慢慢漂移——某个版本给一边加了个 clamp、
    改了归一化口径，另一边没跟上。而表现是"日志上算出来的 KL
    和实际反传的不是一回事"，几乎不可能被发现。

    这里让两份实现在固定的人造数值上直接对比，把它变成一个会失败的断言。
    用固定的数而不是随机数：随机数偶尔会撞到比值恰好落在裁剪边界的情形，
    让本该稳定的对比变得不稳定。
    """
    check = _cross_check_loss(smoke_cfg)
    assert check["agree"], (
        f"损失的两份实现不一致：torch={check['torch']} python={check['python']}"
    )
    assert check["kl_torch"] == pytest.approx(check["kl_python"], abs=1e-6)


def test_grpo_step_produces_finite_gradients(smoke_cfg, tokenizer):
    """一步完整的 GRPO：采样 → 解析 → 打分 → 优势 → 前向 → 反传。

    这里不断言梯度非零，而是断言"梯度存在、有限，且和优势对得上账"。

    不对非零值断言的原因
    ----------------
    随机初始化的两层小模型生成不出 `### 设计序列` 这种格式，解析成功率是 0，
    于是所有奖励被判无效、组内优势全为 0、替代损失恒为 0；同时
    `build_reference_model` 深拷贝的是还没被优化器碰过的策略，
    所以 KL 项在这一步也精确为 0。两个梯度来源都是 0，
    梯度全零是正确行为——没有优势就没有学习信号。

    那这条测试还剩什么价值？问"账对不对得上"：
    优势全零 且 KL 为零  ⇒  loss 必须是 0、梯度必须是 0。
    一个恒为 0 的 loss 从训练循环里静悄悄地过去是最难发现的失败模式，
    这里把它变成一个会失败的断言。

    "有优势时会有梯度"由 `test_nonzero_advantages_produce_gradients` 单独负责——
    那件事没法用真模型测，只能人为构造优势。
    """
    from pepdesign.rewards.composite import CompositeScorer
    from pepdesign.train.dataset import load_grpo_prompts

    cfg = load_config("configs/smoke_cpu.yaml")
    model = build_tiny_model(vocab_size=len(tokenizer))
    model = inject_lora(model, tiny_lora_config())
    ref = build_reference_model(model)
    prompts = load_grpo_prompts(cfg.grpo.dataset_path, limit=1)
    scorer = CompositeScorer(cfg.reward)

    loss, batch, metrics = run_step(model, ref, tokenizer, prompts, cfg, scorer)
    loss.backward()

    assert torch.isfinite(loss), f"loss 不是有限值: {loss}"
    assert len(batch.texts) == cfg.grpo.num_generations

    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "反传之后一个梯度都没有"
    assert all(torch.isfinite(g).all() for g in grads), "梯度里出现了 nan/inf"

    # 指标本身要合理：解析失败率和可用比例都在 [0,1] 里
    assert 0.0 <= metrics.parse_fail_rate <= 1.0
    assert 0.0 <= metrics.usable_ratio <= 1.0

    # --- 账目自洽 ---
    no_signal = all(a == 0.0 for a in batch.advantages)
    if no_signal and metrics.kl == 0.0:
        loss_scalar = float(loss.detach())
        assert loss_scalar == 0.0, (
            f"优势全零、KL 也为零，loss 必须是 0，实际 {loss_scalar}"
        )
        assert all(g.abs().sum() == 0 for g in grads), (
            f"优势全零、KL 也为零，梯度必须全为 0，"
            f"但范数是 {max(float(g.abs().sum()) for g in grads)}。"
            f"说明 loss 里混进了与优势无关的项。"
        )

    # 反过来的方向：真有非零优势时，梯度不能凭空消失
    if not no_signal:
        assert any(g.abs().sum() > 0 for g in grads), "有非零优势却没有梯度"

    # 解析失败与优势失效必须是同一件事：没有一条能被打分 ⇒ 没有一条优势非零
    if metrics.parse_fail_rate == 1.0:
        assert no_signal, "解析全部失败却算出了非零优势——无效序列参与了组内基线"


def test_nonzero_advantages_produce_gradients(smoke_cfg):
    """优势非零 ⇒ 梯度非零，且方向正确。这是"奖励真的接进了 loss"的证明。

    不使用真模型的原因
    ----------------------
    见上一条：随机初始化的模型解析成功率是 0，优势必然全为 0，
    那时候没有梯度是对的。所以"有优势时会怎样"只能人为构造优势来测——
    这不是退而求其次，而是唯一能把符号方向钉死的方法。

    方向断言才是重点
    ----------------
    GRPO 的全部意义是"往高分那边走"。如果 `grpo_loss` 里把符号写反了
    （`+surrogate` 而不是 `-surrogate`），梯度照样非零、loss 照样在变，
    只是模型在朝低分方向学。这种错在真训练里表现为"训着训着变差了"，
    会被归因到数据或超参上。所以这里正负两行各断言一次符号，
    只有两个符号同时错才会漏网。

    old_logps 和 ref_logps 都取 `logps.detach()`：前者让重要性比恒为 1
    （落在裁剪区间内，符号断言不受裁剪干扰），后者让 k3 的 d 恒为 0
    （KL 项精确为 0，不掺进梯度里）。
    """
    from pepdesign.train.grpo import grpo_loss

    cfg = load_config("configs/smoke_cpu.yaml")

    # 两行 = 同组内的两个 rollout。组内优势本来就是零和的，一正一负。
    logps = torch.tensor(
        [[-0.5, -1.0, -1.5, -2.0], [-0.5, -1.0, -1.5, -2.0]], requires_grad=True
    )
    old_logps = logps.detach().clone()
    ref_logps = logps.detach().clone()
    advantages = torch.tensor([[2.0], [-2.0]])
    mask = torch.ones_like(logps, dtype=torch.long)

    loss, metrics = grpo_loss(
        logps,
        old_logps,
        ref_logps,
        advantages,
        mask,
        loss_type=cfg.grpo.loss_type,
        epsilon=cfg.grpo.epsilon,
        clip_higher=cfg.grpo.clip_higher,
        beta=cfg.grpo.beta,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["kl"] == pytest.approx(0.0, abs=1e-9), (
        "参考和策略完全相同时 KL 必须精确为 0（k3 的 d=0 ⇒ exp(0)-0-1=0）"
    )
    assert logps.grad is not None
    assert torch.isfinite(logps.grad).all()
    assert logps.grad.abs().sum() > 0, "有非零优势却没有任何梯度"

    # 逐行看符号。正优势那一行：loss ≈ -A·exp(logπ − logπ_old)，
    # 对 logπ 求导是负的 ⇒ 梯度下降会抬高它的对数概率（也就是抬高它的概率）。
    # 负优势那一行反过来。
    grad_of_good, grad_of_bad = logps.grad.sum(dim=1)
    assert grad_of_good < 0, (
        f"正优势那一行的梯度应当为负（抬高它的概率），实际 {float(grad_of_good)}"
        f"——检查 grpo_loss 里 surrogate 的符号是不是写反了"
    )
    assert grad_of_bad > 0, (
        f"负优势那一行的梯度应当为正（压低它的概率），实际 {float(grad_of_bad)}"
    )


def test_kl_term_produces_gradient_once_the_policy_has_moved(smoke_cfg):
    """策略和参考只要有一点点差别，KL 项就必须产生梯度。

    紧接上面两条，这一条钉的是第一步 KL 恰好为 0 不是 bug这个事实：

        d = logπ_ref − logπ_θ，k3 = exp(d) − d − 1
        dk3/d(logπ_θ) = 1 − exp(d)，在 d = 0 处精确为 0

    而 `build_reference_model` 深拷贝的正是还没更新过的策略，
    `run_grpo_smoke` 的第 1 步就落在这个点上。会触发它的场景是：
    有人看到 step 1 的 `kl 0.0000`，以为 KL 没接进计算图，
    于是去调 `beta`——调多大都一样，因为梯度本身就是 0。

    β 只有在策略偏离参考之后才起作用，这是 k3 的数学性质，不是实现问题。
    """
    from pepdesign.train.grpo import grpo_loss

    cfg = load_config("configs/smoke_cpu.yaml")

    logps = torch.tensor([[-0.5, -1.0, -1.5]], requires_grad=True)
    old_logps = logps.detach().clone()
    # 参考比策略更"喜欢"这些 token：d = ref − logπ > 0
    ref_logps = logps.detach().clone() + 0.3
    # 优势全 0 ⇒ 替代损失恒为 0，此刻唯一的梯度来源就是 KL
    advantages = torch.zeros(1, 1)
    mask = torch.ones_like(logps, dtype=torch.long)

    loss, metrics = grpo_loss(
        logps,
        old_logps,
        ref_logps,
        advantages,
        mask,
        loss_type=cfg.grpo.loss_type,
        epsilon=cfg.grpo.epsilon,
        clip_higher=cfg.grpo.clip_higher,
        beta=cfg.grpo.beta,
    )
    loss.backward()

    assert metrics["kl"] > 0, "d != 0 时 k3 必须为正"
    assert logps.grad is not None
    assert logps.grad.abs().sum() > 0, (
        "优势全 0、但参考与策略有偏离，梯度必须由 KL 项提供——"
        "全 0 说明 KL 根本没接进计算图"
    )
    # d > 0 意味着参考比策略更看好这些 token，
    # dk3/d(logπ) = 1 − exp(d) < 0 ⇒ 梯度下降会把 logπ 往上抬，向参考靠拢。
    assert logps.grad.sum() < 0, "KL 的梯度方向应当把策略拉回参考分布"

    # beta=0 时 KL 项不该再产生任何梯度——它是可关闭的，不是硬编码进来的
    logps2 = torch.tensor([[-0.5, -1.0, -1.5]], requires_grad=True)
    loss2, _ = grpo_loss(
        logps2,
        logps2.detach().clone(),
        logps2.detach().clone() + 0.3,
        advantages,
        mask,
        loss_type=cfg.grpo.loss_type,
        epsilon=cfg.grpo.epsilon,
        clip_higher=cfg.grpo.clip_higher,
        beta=0.0,
    )
    loss2.backward()
    assert float(loss2.detach()) == 0.0, f"beta=0 且优势全 0 时 loss 应为 0，实际 {float(loss2)}"
    assert logps2.grad is None or logps2.grad.abs().sum() == 0


def test_reference_model_is_frozen_and_detached(smoke_cfg, tokenizer):
    """参考模型不能有梯度，也不能和策略共享参数。

    `disable_adapter()` 是个很顺手的写法，但它把参考变成基座——
    在两阶段流程里这意味着 KL 项在把模型往预训练分布上拽（撤销 SFT）。
    所以这里用的必须是 deepcopy 出来的那份 SFT 策略。

    断言两件事：参考模型没有可训练参数；它是另一个对象
    （共享的话，策略一更新参考就跟着变，KL 项恒为 0，等于没有约束）。
    """
    model = build_tiny_model(vocab_size=len(tokenizer))
    model = inject_lora(model, tiny_lora_config())
    ref = build_reference_model(model)

    assert all(not p.requires_grad for p in ref.parameters())
    assert ref.training is False, "参考模型必须处于 eval 模式（关掉 dropout）"
    # 参数张量不能是同一块内存
    a = next(iter(model.parameters()))
    b = next(iter(ref.parameters()))
    assert a.data_ptr() != b.data_ptr(), "参考模型和策略共享了参数"


def test_grpo_smoke_end_to_end(smoke_cfg, tokenizer):
    """端到端跑几步完整 GRPO（含 `optimizer.step()`），账目要自洽。

    与上面 `test_grpo_step_produces_finite_gradients` 的区别是它走的是
    完整的训练循环，而不是单步前向。

    `tensors_changed > 0` 不足以说明"学到东西了"：AdamW 是解耦权重衰减，
    梯度全为 0 时它照样会把参数按 (1 − lr·wd) 缩一下。所以那条断言只证明
    optimizer 挂对了参数。真正说明"这一步有没有学习信号"的是
    `grad_norms` / `nonzero_advantages` / `parse_fail_rate` 几个字段。

    顺带覆盖了弱解析这条分支：随机模型的乱码里
    `DISCRIMINATORBURNING`（19 残基，落在合法长度区间内）会被
    `line_scan` 抠成"多肽"并拿到真分数——修复前这一步的 `mean_reward`
    是 0.2097，全部来自那一条垃圾。现在它按无效处理，
    `weak_parse_ratio` 大于 0 而 `mean_reward` 保持 0。

    `tokenizer` 参数同上：只为绑定跳过条件。
    """
    cfg = load_config("configs/smoke_cpu.yaml")
    report = run_grpo_smoke(cfg, steps=2)

    assert report["steps"] == 2
    assert report["losses_finite"], f"损失里出现了非有限值: {report['losses']}"
    assert report["trainable_tensors"] > 0
    assert report["tensors_changed"] > 0, "2 步之后没有任何 LoRA 参数被更新"
    assert report["loss_cross_check"]["agree"]

    assert len(report["grad_norms"]) == 2
    assert all(math.isfinite(x) for x in report["grad_norms"])
    assert 0.0 <= report["parse_fail_rate"] <= 1.0
    assert 0.0 <= report["weak_parse_ratio"] <= 1.0
    assert 0.0 <= report["usable_ratio"] <= 1.0

    # 三个数必须自洽（随机模型上这三个都是极端值，正好把边界验了）：
    #   * 弱解析是有效失败率的一个子集，不能比它大；
    #   * 有效失败率不会被"可用率"反超——可用率还额外排除了
    #     `validate()` 打回的（非法残基/长度越界），所以它只会更小。
    assert report["weak_parse_ratio"] <= report["parse_fail_rate"] + 1e-9
    assert report["parse_fail_rate"] <= 1 - report["usable_ratio"] + 1e-9, (
        f"有效失败率 {report['parse_fail_rate']} 超过了 1-可用率 "
        f"{1 - report['usable_ratio']}——两个口径对不上"
    )

    # 解析全失败 ⇒ 一条优势都不该非零（无效序列不参与组内均值/标准差）
    if report["parse_fail_rate"] == 1.0:
        assert report["nonzero_advantages"] == 0, (
            "解析全失败却算出了非零优势——无效序列混进了组内基线"
        )

    # 第 1 步的梯度应当是精确的 0，且这完全正常：
    #   * 优势全 0（没有解析成功的 rollout）；
    #   * 参考就是策略本身（build_reference_model 深拷贝的还没被 optimizer
    #     碰过的模型），d ≡ 0 ⇒ k3 及其梯度都精确为 0。
    # 把这条钉成断言，免得以后有人拿"第一步没有梯度"当 bug 去改。
    # 用 1e-9 而不是 == 0：两次独立前向的浮点结果不保证逐位相同。
    if report["nonzero_advantages"] == 0:
        assert report["grad_norms"][0] < 1e-9, (
            f"第 1 步既无优势、参考又等于策略，梯度应当为 0，"
            f"实际 {report['grad_norms'][0]}"
        )
    else:
        assert any(x > 0 for x in report["grad_norms"]), (
            f"有非零优势却每一步梯度都是 0: {report['grad_norms']}"
        )


# ---------------------------------------------------------------------------
# 校准预热
# ---------------------------------------------------------------------------
def test_calibration_warmup_runs_on_cpu(smoke_cfg, tokenizer):
    """校准预热：读校准对 → 转成 SFT 样本 → 跑几步监督学习。

    这条链路单独测，因为它是两阶段之间的一段，最容易在改动中被漏掉：
    它既不属于 SFT（不走 trl 的 trainer），也不属于 GRPO（不走 rollout）。
    """
    from pepdesign.train.calibration import load_calibration_samples, run_calibration_warmup

    # 用真实配置（校准集路径在里面），但不用它的 grpo 超参——
    # 那些是给真实基座在 GPU 上跑的，在 CPU 小模型上没意义。
    cfg = load_config("configs/grpo_lora.yaml")

    # 校准集必须存在才测；不存在说明数据构建还没跑过第 6 步
    try:
        samples = load_calibration_samples(cfg, limit=4)
    except FileNotFoundError as exc:
        pytest.skip(f"没有校准集，跳过：{exc}")
    assert samples and "messages" in samples[0]

    model = build_tiny_model(vocab_size=len(tokenizer))
    model = inject_lora(model, tiny_lora_config())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    # 用冒烟配置的步数上限，避免在 CPU 上跑完几百条校准样本
    cfg.grpo.per_device_train_batch_size = 1
    cfg.grpo.calibration_lr_scale = 0.5
    losses = run_calibration_warmup(model, tokenizer, cfg, optimizer)

    assert losses, "预热一步都没跑"
    assert all(loss == loss for loss in losses), "预热 loss 出现 nan"
    assert optimizer.param_groups[0]["lr"] == cfg.grpo.learning_rate, (
        "预热结束后必须把学习率恢复成 RL 阶段的值，否则 RL 全程用的都是打对折的 lr"
    )


# ---------------------------------------------------------------------------
# 第五维：ESM-2 掩码边缘概率（用随机初始化的小 ESM，不下载真权重）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def esm_tokenizer():
    """真的 ESM tokenizer（约 1MB）。

    必须用真的：本预测器的正确性建立在"`<mask>` 是单个特殊 token"
    和"序列按单氨基酸分词"这两条上，假 tokenizer 证明不了这两件事。
    权重反而可以用随机初始化的——分数对不对不重要（那要真权重），
    链路通不通、形状对不对才是这层要验的。
    """
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"取不到 ESM tokenizer，跳过：{type(exc).__name__}: {exc}")


def _tiny_esm(esm_tokenizer):
    """两层、hidden=32 的随机 ESM。不下载任何权重。"""
    from transformers import EsmConfig, EsmForMaskedLM

    cfg = EsmConfig(
        vocab_size=len(esm_tokenizer),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        pad_token_id=esm_tokenizer.pad_token_id,
    )
    return EsmForMaskedLM(cfg).eval()


def test_esm_masked_variants_tokenize_to_equal_length(esm_tokenizer):
    """L 条掩码变体分词后必须等长——这是能叠成一个 batch 的前提。

    变体在字符串层面是等长的（每条都只是把一个字符换成 `<mask>`），
    但真正重要的是 token 层面的长度：如果 `<mask>` 被分词器切成多个 token，
    不同位置的变体长度就会不同，batch 会直接报错（还算好），
    或者被 padding 成不等长的东西（更糟）。
    """
    from pepdesign.rewards.predictors.esm import masked_variants

    seq = "KWKLFKKIEKVK"
    lengths = {len(esm_tokenizer(v)["input_ids"]) for v in masked_variants(seq)}
    assert lengths == {len(seq) + 2}, f"token 长度不齐：{lengths}（预期 L+2，含首尾特殊 token）"


def test_esm_predictor_scores_on_cpu(esm_tokenizer):
    """端到端：逐位置掩码 → 批量前向 → 回填真实氨基酸的 log 概率 → 归一化。

    三个断言各自对应一类会静默出错的地方：
      * `mean_logprob` 是有限值 → 回填的下标没越界、log_softmax 没出 nan。
        注意这里不能断言"分数应该高/低"：权重是随机初始化的，
        它给出的 log 概率没有任何生物学含义。
      * `value` 落在 [0,1] → 归一化 + clip 生效，第五维和其余四维同尺度。
      * `confidence > 0` → 这次真的算出来了，而不是走了"ESM 不可用"的降级分支。
    """
    from pepdesign.rewards.predictors.esm import ESMPseudoLikelihoodPredictor

    pred = ESMPseudoLikelihoodPredictor()
    # 直接塞进随机小模型，跳过下载真权重那一步
    pred._model = _tiny_esm(esm_tokenizer)
    pred._tokenizer = esm_tokenizer

    res = pred.predict_one("KWKLFKKIEKVK")
    assert res.confidence > 0, f"走了降级分支，ESM 没被用上：{res.detail}"
    assert 0.0 <= res.value <= 1.0
    assert res.detail is not None
    assert torch.isfinite(torch.tensor(res.detail["mean_logprob"]))
    # 随机权重的 log 概率必然很负（词表 33 个 token，均匀分布下约 -3.5）
    assert res.detail["mean_logprob"] < 0
    assert res.detail["length"] == 12


def test_esm_predictor_degrades_when_weights_missing(esm_tokenizer):
    """拿不到权重时必须优雅降级：置信度 0、detail 里写明原因、不抛异常。

    置信度必须是 0（而不是一个看起来中性的 0.5+高置信度）：
    0 在 composite 里会让这一维的有效权重变成 0，也就是"不参与总奖励"。
    给一个编出来的中性分数会让审计报告里出现一个读报告的人分辨不出的假数据。
    """
    from pepdesign.rewards.predictors.esm import ESMPseudoLikelihoodPredictor

    pred = ESMPseudoLikelihoodPredictor(model_name="facebook/不存在的模型")
    res = pred.predict_one("KWKLFKKIEKVK")

    assert res.confidence == 0.0
    assert res.detail is not None and "error" in res.detail
    assert "ESM 未加载" in str(res.detail["error"])


def test_esm_scorer_marks_unavailable_explicitly(esm_tokenizer):
    """`PlmScorer` 在分数不可用时必须在 detail 里打上 `unavailable` 标记。

    没有这个标记的话，审计报告里会出现一个"值 0.5、看起来挺正常"的记录，
    而它其实来自一次失败的加载。这类"沉默的假数据"正是这份审计要防的东西。
    """
    from pepdesign.rewards.plm import PlmScorer
    from pepdesign.rewards.predictors.esm import ESMPseudoLikelihoodPredictor

    scorer = PlmScorer(ESMPseudoLikelihoodPredictor(model_name="facebook/不存在的模型"))
    d = scorer.score_one("KWKLFKKIEKVK", {})

    assert d.dimension == "plm"
    assert d.confidence == 0.0
    assert d.detail.get("unavailable") is True
    assert d.detail.get("error")


# ---------------------------------------------------------------------------
# 梯度检查点：接线与两个常见问题
# ---------------------------------------------------------------------------
def test_gradient_checkpointing_keeps_lora_gradients(fresh_tiny, tokenizer):
    """梯度检查点接上之后，LoRA 仍然拿得到梯度。

    这个字段以前没有任何消费方（两份真实配置都写着 true，代码一处都没调），
    接上之后要盯的正是该问题：检查点区域内的 LoRA 分支拿不到梯度时，
    loss 照降、曲线完全正常，只是 LoRA 从未被更新。

    上游（transformers 5.x）的 `gradient_checkpointing_enable()` 默认
    `use_reentrant=False` 并且自己会调用 `enable_input_require_grads()`，
    所以这里不重新实现那两件事，而是用这条断言证明它确实成立——
    哪天上游改了行为，失败会出现在这里，而不是出现在一个"训了三天没效果"
    的实验结论里。

    用 `fresh_tiny` 而不是模块级共享的 `tiny`：启用梯度检查点会改模型状态
    （挂 forward hook、关 use_cache），而 `tiny` 是 module scope 的。
    """
    from pepdesign.models.loader import enable_gradient_checkpointing

    model = fresh_tiny
    report = enable_gradient_checkpointing(model, enabled=True)
    assert report["enabled"] is True
    assert report["input_require_grads"] is True, (
        "输入 embedding 没有挂上 require_grads hook——检查点区域内的 LoRA "
        "会拿不到梯度，而 loss 曲线看不出任何异常"
    )

    # 一次真实的前反向
    batch = collate_sft(
        torch, [{"input_ids": [1, 2, 3, 4, 5], "token_mask": [0, 0, 1, 1, 1]}],
        pad_token_id=0,
    )
    model(**batch).loss.backward()

    pairs = _lora_layers(model)
    assert pairs
    _, first = pairs[0]
    assert first.lora_B.weight.grad is not None
    assert torch.count_nonzero(first.lora_B.weight.grad) > 0, (
        "开了梯度检查点之后 lora_B 的梯度全为零——这正是该问题"
    )
    assert tokenizer is not None  # 绑定跳过条件，见文件顶部说明


def test_gradient_checkpointing_leaves_generation_cache_alone(fresh_tiny):
    """训练关 KV cache，生成不受影响。

    这两件事以前是同一个字段（`model.config.use_cache`），很容易误以为
    "关了 cache 所以 GRPO 采样会慢三四倍"。实际上 `generate` 读的是
    `model.generation_config.use_cache`——加载时由 config 建出来的独立对象。

    把这条钉住，是因为它一旦不成立，症状是"GRPO 训练慢了一大截"
    而没有任何报错，没人会想到去查梯度检查点。
    """
    from pepdesign.models.loader import enable_gradient_checkpointing

    model = fresh_tiny
    enable_gradient_checkpointing(model, enabled=True)

    assert model.config.use_cache is False, "训练必须关掉 KV cache"
    assert model.generation_config.use_cache is True, (
        "生成侧的 use_cache 被一起关掉了——GRPO 的采样会慢好几倍而没有报错"
    )


def test_gradient_checkpointing_disabled_is_a_noop(fresh_tiny):
    """配置关掉（冒烟配置就是）时必须什么都不做。"""
    from pepdesign.models.loader import enable_gradient_checkpointing

    report = enable_gradient_checkpointing(fresh_tiny, enabled=False)
    assert report["enabled"] is False


# ---------------------------------------------------------------------------
# checkpoint 保存与断点续训
# ---------------------------------------------------------------------------
def _sft_smoke_cfg(output_dir, *, save_steps, save_total_limit, resume=None):
    """冒烟配置 + checkpoint 相关的覆盖。不碰训练超参（步数由 max_steps 定）。

    只改一处训练侧的东西：调度器换成 constant + 关掉 warmup。
    这不是随手改的——`smoke_cpu.yaml` 默认是 cosine + 3 步，于是
    `warmup_steps_for(0.03, 3) = 1`、cosine 在第 3 步恰好衰减到 lr = 0。
    lr 为 0 时 `optimizer.step()` 什么都不做，**优化器状态恢复得对不对
    根本不影响结果**——续训测试会变成一条恒真的断言（实测验证过：
    把恢复优化器状态那行删掉，测试照样通过）。

    constant + 无 warmup 让每一步的 lr 都是 1e-2，于是"优化器状态有没有恢复"
    会直接反映在权重上，断言才有区分度。
    """
    cfg = load_config("configs/smoke_cpu.yaml").model_copy(deep=True)
    cfg.sft.output_dir = str(output_dir)
    cfg.sft.save_steps = save_steps
    cfg.sft.save_total_limit = save_total_limit
    cfg.sft.resume_from_checkpoint = resume
    cfg.sft.report_to = []
    cfg.sft.eval_dataset_path = None
    cfg.sft.lr_scheduler_type = "constant"
    cfg.sft.warmup_ratio = 0.0
    return cfg


def _run_sft(cfg, tokenizer, *, max_steps: int):
    """跑一次真实的 train_loop（两层小模型 + 真语料），返回 `(结果, 模型)`。

    返回模型是为了让调用方能在权重层面比较两次运行——那比只比 loss
    更直接：loss 相等有可能只是因为那一步没动参数，而权重逐元素相等
    才说明"这一步做了什么"也完全一样。
    """
    from pepdesign.models.loader import probe_hardware, resolve_plan
    from pepdesign.train.sft import build_sft_examples, load_sft_samples, train_loop
    from pepdesign.utils.seed import set_seed

    set_seed(cfg.sft.seed)

    model = build_tiny_model(vocab_size=len(tokenizer))
    model = inject_lora(model, tiny_lora_config())

    rows = load_sft_samples(cfg.sft.dataset_path, limit=max_steps * 4)
    examples, _ = build_sft_examples(
        tokenizer, rows,
        max_seq_length=cfg.sft.max_seq_length,
        completion_only=cfg.sft.completion_only_loss,
    )
    hw = probe_hardware()
    plan = resolve_plan(
        cuda_available=hw["cuda_available"], bf16_supported=hw["bf16_supported"],
        mps_available=hw["mps_available"], want_bf16=False, want_fp16=False,
        quant=cfg.quant,
    )
    result = train_loop(
        model, examples, cfg,
        tokenizer=tokenizer, device=next(model.parameters()).device, plan=plan,
        max_steps=max_steps,
    )
    return result, model


def _lora_weights(model):
    return [layer.lora_A.weight.detach().clone() for _, layer in _lora_layers(model)] + [
        layer.lora_B.weight.detach().clone() for _, layer in _lora_layers(model)
    ]


def test_sft_checkpoint_save_and_resume_reproduces_the_run(tmp_path, tokenizer):
    """续训必须真的续上：从 checkpoint-2 接着跑，结果与不中断的一次相同。

    这是"什么才算续上了"能给出的最强陈述，也是这个测试唯一的重点。
    只恢复权重的"续训"会让 Adam 二阶矩从 0 重新累积、lr 曲线从 warmup 重来，
    而这两件事都不会报错——所以断言必须是数值相等，不能只看"没崩"。

    两个断言各管一段，缺一不可：

      * 第三步的 loss 相等 → 权重、优化器状态、数据位置都恢复对了；
      * 跑完之后 LoRA 权重逐元素相等 → 连"这一步做了多大的更新"也一样。
        （只比 loss 有可能被"那一步 lr 恰好为 0"蒙混过去，见 `_sft_smoke_cfg`
        的说明——这个测试第一版就是这么写坏的，靠 mutation 才验出来。）

    顺带锁住三件以前是假的事：

      * `save_steps` 在 SFT 侧以前没有任何消费方（SFT 从来不存中间
        checkpoint），现在每次保存都真的落盘；
      * `save_total_limit` 以前也没有消费方，现在 `checkpoint-1`
        会因为超出保留数被清掉；
      * 每个 checkpoint 里有 `adapter_config.json`——那是 `save_adapter`
        拿到 LoRAConfig 的证据（漏传 cfg 时手写后端会直接抛 ValueError，
        而冒烟配置以前把 save_steps 设成 0，这条路径从没被跑到过）。
    """
    cfg = _sft_smoke_cfg(tmp_path, save_steps=1, save_total_limit=2, resume=None)

    # ---- 第一次：完整跑 3 步 ----
    result_a, model_a = _run_sft(cfg, tokenizer, max_steps=3)
    assert result_a["num_steps"] == 3
    assert result_a["resumed_from"] is None

    names = sorted(p.name for _, p in list_checkpoints(tmp_path))
    assert names == ["checkpoint-2", "checkpoint-3"], (
        f"期望保留最近 2 个 checkpoint（save_total_limit=2），实际 {names}"
    )
    for required in ("adapter_config.json", "optimizer.pt", "scheduler.pt", "trainer_state.json"):
        assert (tmp_path / "checkpoint-2" / required).exists(), (
            f"checkpoint-2 缺 {required}——续训要靠它恢复优化器/调度器状态"
        )

    # ---- 第二次：从 checkpoint-2 续训，只跑第 3 步 ----
    cfg_b = _sft_smoke_cfg(
        tmp_path, save_steps=1, save_total_limit=2,
        resume=str(tmp_path / "checkpoint-2"),
    )
    result_b, model_b = _run_sft(cfg_b, tokenizer, max_steps=3)

    assert result_b["resumed_from"] == str(tmp_path / "checkpoint-2")
    assert result_b["num_steps"] == 3, "续训后总步数不该变，只是从第 3 步开始跑"
    assert len(result_b["history"]["loss"]) == 3, "恢复的历史 + 新跑的这一步"

    # 前两步的 loss 是从 checkpoint 里读回来的，必须与不中断的那次逐位相同。
    assert result_b["history"]["loss"][:2] == result_a["history"]["loss"][:2]
    assert result_b["final_loss"] == pytest.approx(result_a["final_loss"], abs=1e-6), (
        f"续训的第 3 步 loss={result_b['final_loss']} 与不中断的 "
        f"{result_a['final_loss']} 不一致——权重/优化器/数据位置有一个没恢复对"
    )

    # 最强的那条：跑完之后两组 LoRA 权重逐元素相等。
    weights_a, weights_b = _lora_weights(model_a), _lora_weights(model_b)
    assert len(weights_a) == len(weights_b) > 0
    for i, (wa, wb) in enumerate(zip(weights_a, weights_b, strict=True)):
        assert torch.equal(wa, wb), (
            f"第 {i} 个 LoRA 权重张量在续训后与不中断的那次不一样——"
            f"最大差异 {float((wa - wb).abs().max()):.3e}。"
            f"优化器状态或调度器状态没有真正恢复。"
        )


def test_sft_resume_from_missing_checkpoint_raises_instead_of_restarting(tmp_path, tokenizer):
    """配了续训却找不到 checkpoint 时必须报错，不能静默从头训。

    静默从头训的后果不只是"白跑"：它最后会把 `final/` 覆盖掉，
    而整个过程没有任何一条日志能告诉你这件事。
    """
    cfg = _sft_smoke_cfg(
        tmp_path, save_steps=1, save_total_limit=2,
        resume=str(tmp_path / "checkpoint-99"),
    )
    with pytest.raises(FileNotFoundError):
        _run_sft(cfg, tokenizer, max_steps=1)


def test_grpo_checkpoint_is_writable_and_resumable(tmp_path, tokenizer):
    """GRPO 的 checkpoint 能写、能读回来续训——这条以前会直接崩。

    `train/grpo.py` 里两处 `save_adapter(model, path, tokenizer)` 都漏了
    LoRAConfig，而手写后端下 `save_adapter` 会因为 `cfg is None` 抛
    `ValueError`。即默认配置下 GRPO 一个 adapter 都存不下来，
    中间的和最终的都不能。冒烟配置把 `save_steps` 设成 0，
    所以这条路径从来没有被跑到过——直到现在。

    顺带验了续训时跳过校准预热：预热的作用是把起点挪到合理区域，
    而续训的起点已经训过了；再跑一遍会悄悄改掉"接着跑"的语义。
    """
    from pepdesign.rewards.composite import CompositeScorer
    from pepdesign.train.dataset import load_grpo_prompts
    from pepdesign.train.grpo import build_reference_model, run_grpo

    def _cfg(resume=None, calibration=False):
        c = load_config("configs/smoke_cpu.yaml").model_copy(deep=True)
        c.grpo.output_dir = str(tmp_path)
        c.grpo.save_steps = 1
        c.grpo.save_total_limit = 3
        c.grpo.report_to = []
        c.grpo.max_steps = 2
        c.grpo.resume_from_checkpoint = resume
        c.grpo.calibration_enabled = calibration
        return c

    def _run(c):
        model = inject_lora(build_tiny_model(vocab_size=len(tokenizer)), tiny_lora_config())
        return run_grpo(
            c, model=model, ref_model=build_reference_model(model), tokenizer=tokenizer,
            prompts=load_grpo_prompts(c.grpo.dataset_path, limit=4),
            scorer=CompositeScorer(c.reward), max_steps=2,
        )

    # 第一次：跑 2 步，每步都存
    _run(_cfg())
    assert (tmp_path / "checkpoint-1" / "adapter_config.json").exists(), (
        "GRPO 的中间 checkpoint 没有落盘（就是那个漏传 cfg 的 bug）"
    )
    assert (tmp_path / "final" / "adapter_config.json").exists(), (
        "GRPO 的最终 adapter 没有落盘（同一类 bug 的第二处调用点）"
    )

    # 第二次：从 checkpoint-1 续训，并打开校准预热以验证它会被跳过
    c = _cfg(resume=str(tmp_path / "checkpoint-1"), calibration=True)
    _run(c)
    assert (tmp_path / "checkpoint-2").exists(), "续训应当继续按 save_steps 存盘"


# ---------------------------------------------------------------------------
# LoRA 参数的设备与精度对齐
# ---------------------------------------------------------------------------
class TestLoraParameterPlacement:
    """LoRA 分支的参数必须和它包裹的层同设备、同精度。

    这条锁的是一个只在真 GPU 上才暴露的 bug：`LoRALinear.__init__`
    里新建的 `nn.Linear` 默认落在 CPU + float32，而 `base` 这时候通常
    已经在 GPU 上了（`load_base_model` 先 `.to(device)`，之后才 `inject_lora`）。
    第一次前向就炸：

        RuntimeError: Expected all tensors to be on the same device,
        but got mat2 is on cpu, different from other tensors on cuda:0

    CPU 冒烟永远测不出来——所有东西都在 CPU，设备和精度永远对得上。
    所以下面不测"设备"（CPU 上测不了），只测能测的那部分逻辑：
    dtype 的判定分支，以及前向里那两次精度对齐。
    """

    @staticmethod
    def _layer(base_dtype=None, quantized=False):
        from pepdesign.models.lora_layer import build_lora_linear_class

        nn = torch.nn
        LoRALinear = build_lora_linear_class()
        base = nn.Linear(8, 4, bias=False)
        if quantized:
            # 模拟 4-bit 量化层：weight 是整数类型（bnb 的 Params4bit 就是 uint8）。
            # 注意 nn.Linear 本身不会这么干，这里只为触发那条判定分支。
            # requires_grad=False 是必须的：PyTorch 不允许整数张量要求梯度
            base.weight = nn.Parameter(
                torch.zeros(4, 8, dtype=torch.uint8), requires_grad=False
            )
        elif base_dtype is not None:
            base = base.to(base_dtype)
        return LoRALinear(base, r=2, alpha=4), base

    def test_lora_dtype_follows_a_floating_point_base(self):
        """base 是浮点时，LoRA 跟它同精度。

        跟错精度的后果不是报错而是静默的数值问题：
        fp32 的 LoRA 包在 bf16 的 base 上，前向会在 dtype 提升上
        悄悄多算一遍，而参数更新的尺度也跟着变。
        """
        layer, base = self._layer(torch.float64)
        assert layer.lora_A.weight.dtype == torch.float64
        assert layer.lora_B.weight.dtype == torch.float64
        assert layer.lora_A.weight.dtype == base.weight.dtype

    def test_quantized_base_falls_back_to_float32(self):
        """量化层的 weight 是 uint8，不能跟——跟了会得到一个整数精度的 LoRA。

        这是"跟 base 的 dtype"这条规则唯一的例外，也是最容易写错的地方：
        直接抄 `dtype=base.weight.dtype` 在这里会得到 uint8。
        """
        layer, _ = self._layer(quantized=True)
        assert layer.lora_A.weight.dtype == torch.float32
        assert layer.lora_B.weight.dtype == torch.float32

    def test_forward_aligns_input_and_output_dtype(self):
        """前向必须能吃下与 LoRA 参数不同精度的输入，并把输出降回 base 的精度。

        这是 GPU 上真正会遇到的情形：QLoRA 下 base 是 4-bit、
        激活值是 bf16，而 LoRA 参数是 fp32。bf16 输入直接喂给 fp32 权重
        会抛 "expected scalar type BFloat16 but found Float"。

        这里用 float64 的 base + float32 的 LoRA 来复现同一类不匹配
        （CPU 上没有 bf16 的 matmul 支持，用 float64/float32 是等价的最小复现）。
        """
        layer, _ = self._layer(torch.float64)
        # 手动把 LoRA 降成 fp32，模拟"量化基座 → LoRA 走 fp32"那条分支
        layer.lora_A = layer.lora_A.float()
        layer.lora_B = layer.lora_B.float()

        x = torch.randn(2, 8, dtype=torch.float64)
        out = layer(x)                       # 不抛 dtype 错
        assert out.dtype == torch.float64, "输出必须回到 base 的精度"

    def test_matching_dtypes_pay_no_conversion_cost(self):
        """精度本来就一致时，那两次 `.to()` 必须是零开销的（不复制张量）。

        这条是"为什么可以放心在 forward 里加两次 cast"的依据：
        PyTorch 对 dtype 相同的 `.to()` 直接返回原张量。
        如果哪天有人换成 `.float()` 这类强制转换，这条会红。
        """
        layer, _ = self._layer(torch.float32)
        x = torch.randn(2, 8, dtype=torch.float32)
        assert layer.lora_A.weight.dtype == torch.float32

        # forward 内部会对 x 做 .to(float32)——同 dtype 时必须是同一个对象
        same = x.to(layer.lora_A.weight.dtype)
        assert same is x, "同 dtype 的 .to() 不该复制"

    def test_lora_params_stay_trainable_after_placement(self):
        """.to() 之后 LoRA 参数仍然可训练——不能把 requires_grad 弄丢。"""
        layer, _ = self._layer(torch.float64)
        assert layer.lora_A.weight.requires_grad
        assert layer.lora_B.weight.requires_grad
        assert not layer.base.weight.requires_grad, "基座必须保持冻结"


# ---------------------------------------------------------------------------
# token_logprobs 的分块实现
# ---------------------------------------------------------------------------
class TestTokenLogprobsChunking:
    """分块版必须和"整体升 fp32 再 log_softmax"数学等价。

    这条锁的是一个只在真 GPU 上才暴露的 OOM：LOGOS 的词表是 128,275，
    GRPO 一个 micro-batch 的 logits 形状是 `[32, 676, 128275]`——
    bf16 就 5.55 GB，整体 `.float()` 再翻倍到 11.1 GB，24 GB 卡直接打满
    （实测 `Tried to allocate 10.57 GiB`）。

    修法有两条，第二条比第一条关键：
      1. 分块——fp32 临时张量从"整段序列"降到 chunk 个 token；
      2. 改用 `gather + logsumexp` 而不是 `log_softmax`——
         `log_softmax` 的反向要保存整块输出，分块也救不了；
         `logsumexp` 的反向是从输入重算的，不额外保存。

    所以下面既测"分块不改变结果"，也测"和 log_softmax 的数值一致"。
    """

    @staticmethod
    def _naive(logits, targets):
        """改之前的写法：整体升 fp32 再 log_softmax。作为数值基准。"""
        return (
            torch.log_softmax(logits.float(), dim=-1)
            .gather(-1, targets.unsqueeze(-1))
            .squeeze(-1)
        )

    def test_chunked_matches_the_unchunked_reference(self, fresh_tiny, tokenizer):
        """核心断言：分块版和基准版逐元素相等。

        不相等就说明改的是行为而不是性能——那比 OOM 更糟，
        因为 OOM 会当场报错，而悄悄改了数值不会。
        """
        from pepdesign.train.grpo import token_logprobs

        ids = torch.randint(0, len(tokenizer), (2, 24))
        mask = torch.ones_like(ids)

        with torch.no_grad():
            got = token_logprobs(fresh_tiny, ids, mask)
        logits = fresh_tiny(input_ids=ids, attention_mask=mask).logits[:, :-1, :]
        want = self._naive(logits, ids[:, 1:])

        assert got.shape == want.shape, f"形状不一致 {got.shape} vs {want.shape}"
        assert torch.allclose(got, want, atol=1e-4, rtol=1e-4), (
            f"分块与基准的最大偏差 {(got - want).abs().max().item():.2e}"
        )

    def test_result_shape_is_b_minus_one_along_sequence(self, fresh_tiny, tokenizer):
        """返回 `[B, L-1]`——位置 i 的 logits 预测位置 i+1 的 token。

        这个位移必须和 `masking.shift_mask_for_labels` 一致，
        否则掩码和 logprobs 会错开一位而不会报错。
        """
        from pepdesign.train.grpo import token_logprobs

        ids = torch.randint(0, len(tokenizer), (2, 24))
        with torch.no_grad():
            got = token_logprobs(fresh_tiny, ids, torch.ones_like(ids))
        assert got.shape == (2, 23)

    def test_chunk_size_does_not_change_the_result(self, fresh_tiny, tokenizer, monkeypatch):
        """chunk 取 1 / 7 / 1000 结果都一样。

        chunk 是显存参数，不是数值参数。如果它影响了结果，
        说明分块切错了位置（比如忘了把 targets 一起切）。
        """
        from pepdesign.train import grpo

        ids = torch.randint(0, len(tokenizer), (2, 20))
        mask = torch.ones_like(ids)

        outs = []
        for c in (1, 7, 1000):
            monkeypatch.setattr(grpo, "LOGPROB_CHUNK", c)
            with torch.no_grad():
                outs.append(grpo.token_logprobs(fresh_tiny, ids, mask))
        assert torch.allclose(outs[0], outs[1], atol=1e-5)
        assert torch.allclose(outs[1], outs[2], atol=1e-5)

    def test_stays_finite_for_a_peaked_distribution(self, fresh_tiny, tokenizer):
        """输出必须是有限值。

        这一层的输出直接进重要性比和 KL，一个 -inf 就毁掉整批。
        用 `logsumexp` 而不是 `log(sum(exp))` 就是为了这个：
        后者会先溢出成 inf 再取对数。
        """
        from pepdesign.train.grpo import token_logprobs

        ids = torch.randint(0, len(tokenizer), (2, 16))
        with torch.no_grad():
            got = token_logprobs(fresh_tiny, ids, torch.ones_like(ids))
        assert torch.isfinite(got).all(), "出现 inf 或 nan"


class _Bf16LogitsModel(torch.nn.Module):
    """只吐 bf16 logits 的替身模型。

    存在的理由：这个 bug 只在 logits 是 bf16 时才发作——
    `.float()` 对已经是 fp32 的张量是空操作（直接返回原张量），
    所以在 CPU 上拿真模型测永远抓不到它。
    用这个替身在 CPU 上复现"logits 是 bf16"这个关键条件。
    """

    def __init__(self, vocab: int, seq_len: int) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.randn(seq_len, vocab, dtype=torch.bfloat16)
        )

    def forward(self, *, input_ids, attention_mask):  # attention_mask 用不到，只需 input_ids 定形状
        n = input_ids.shape[1]
        return SimpleNamespace(
            logits=self.logits[:n].unsqueeze(0).expand(input_ids.shape[0], -1, -1)
        )


def test_token_logprobs_does_not_copy_the_logits(tokenizer, monkeypatch):
    """这条锁的是第二次 GRPO OOM 的根因。

    第一版修复已经分了块，但在 `logsumexp` 之前写了 `sl.float()`——
    对 bf16 的 logits 这会新建一个 `[B, c, V]` 的 fp32 张量，
    而 `logsumexp` 的反向传播需要保存它的输入，于是每个块留下一份。
    词表 V = 128,275 时，21 个块共 11 GB 被钉在计算图里：

        logps = token_logprobs(...)            # 带梯度的这次
        lse = torch.logsumexp(sl.float(), ...)
        torch.OutOfMemoryError: Tried to allocate 502.00 MiB
        （22.56 GiB allocated，未分配仅 114 MiB —— 真实占用，不是碎片）

    最能说明问题的是同一个函数前两次调用（`no_grad`）都过了、第三次崩：
    `no_grad` 不建图、不驻留，所以问题不在计算本身，而在计算图里留了什么。

    使用替身模型而非 fresh_tiny 的原因
    --------------------------------------------------
    这个 bug 的触发条件是"logits 是 bf16"。CPU 上真模型是 fp32，
    而 `torch.Tensor.float()` 对已经是 fp32 的输入直接返回原张量（空操作），
    于是无论正确实现还是错误实现，都没有副本产生，测试恒过。
    用 `_Bf16LogitsModel` 才能把那个条件搬进 CPU 测试里。

    判据用 dtype：占位替身的 logits 是 bf16，
    所以任何 fp32 的词表维度张量都只可能来自那次 `.float()`。
    """
    from pepdesign.train import grpo

    vocab = len(tokenizer)
    monkeypatch.setattr(grpo, "LOGPROB_CHUNK", 4)   # 缩小分块，放大"按块累积"特征
    model = _Bf16LogitsModel(vocab, seq_len=24)
    ids = torch.randint(0, vocab, (2, 24))
    mask = torch.ones_like(ids)

    saved: list[tuple] = []

    def pack(t):
        saved.append((t.dtype, tuple(t.shape)))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        out = grpo.token_logprobs(model, ids, mask)
        assert out.requires_grad, "带梯度的这次调用没有建图，测试前提不成立"
        out.sum().backward()

    offenders = [
        (str(dt), sh)
        for dt, sh in saved
        if dt == torch.float32 and len(sh) == 3 and sh[-1] == vocab
    ]
    assert not offenders, (
        f"计算图里保存了 fp32 的词表维度张量：{offenders}。"
        f"这是 `logsumexp` 之前那次 `.float()` 造成的，"
        f"会按分块数累积（真实场景 21 块 ≈ 11 GB）。"
        f"正确写法是直接在 logits 的视图上算 logsumexp。"
    )


# ---------------------------------------------------------------------------
# 训练循环的二阶正确性
# ---------------------------------------------------------------------------
def test_dropout_disabled_makes_two_forwards_identical():
    """`old_logps` 与 `logps` 必须在"dropout 关闭"的同一策略下计算。

    真实配置的 LoRA dropout 是 0.05，而采样用 `model.eval()`。
    如果 `old_logps` 在 `model.train()` 下算，π_old 就不是采样策略，
    第 1 步的 ratio 不为 1、KL 不为 0——测试用 dropout=0 时看不出来。
    这里显式把 dropout 拉到 0.9，让差异不可能靠运气通过。
    """
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 8),
        torch.nn.Dropout(0.9),
        torch.nn.Linear(8, 4),
    )
    model.train()
    x = torch.randn(4, 8)

    first = model(x)
    second = model(x)
    assert not torch.allclose(first, second), "测试前提不成立：dropout 没有生效"

    with _dropout_disabled(model):
        a = model(x)
        b = model(x)
    assert torch.equal(a, b), "dropout 关闭后两次前向必须逐元素相同"

    # 退出上下文后必须恢复原来的 p，不能把模型永久改掉。
    drops = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    assert drops[0].p == pytest.approx(0.9)


def test_reference_model_disables_kv_cache():
    """参考模型只做 no_grad 前向，不该为它分配 KV cache。

    B=32、L≈665、28 层、8 个 KV head 时，这份 cache 约 2.4 GB；
    这是之前两次 OOM 里没有被算进去的一块。
    """
    model = build_tiny_model(vocab_size=128)
    model.config.use_cache = True
    ref = build_reference_model(model)
    assert ref.config.use_cache is False


def test_dr_grpo_denominator_uses_configured_completion_length():
    """`dr_grpo` 的分母是「样本数 × max_completion_length」，不是 mask 全宽。

    同一组 logps、同一张 mask，只改 max_completion_length：
    分母减半，loss 必须翻倍。以前传 `mask.shape[1]` 时这两个调用
    会得到相同的 loss，而且真实 prompt 下分母会大 2.6 倍。
    """
    from pepdesign.train.grpo import grpo_loss

    logps = torch.tensor([[-0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0]])
    mask = torch.ones_like(logps, dtype=torch.long)
    advantages = torch.ones(1, 1)

    kwargs = {
        "loss_type": "dr_grpo",
        "epsilon": 0.2,
        "clip_higher": 0.28,
        "beta": 0.0,
    }
    loss_2, _ = grpo_loss(
        logps, logps.clone(), logps.clone(), advantages, mask,
        max_completion_length=2, **kwargs,
    )
    loss_4, _ = grpo_loss(
        logps, logps.clone(), logps.clone(), advantages, mask,
        max_completion_length=4, **kwargs,
    )
    assert float(loss_2.detach()) == pytest.approx(2.0 * float(loss_4.detach()), rel=1e-6)


def test_peft_adapter_loads_into_existing_peft_model(tokenizer, tmp_path):
    """peft 后端的两阶段路径：先 `inject_lora`，再 `load_adapter`。

    以前 `load_adapter` 对已经是 `PeftModel` 的对象又调了一次
    `PeftModel.from_pretrained(model, ...)`，等于把 PeftModel 再包一层；
    GRPO 的 `sft_adapter_path` 和 checkpoint 续训都会走这条路径。
    """
    pytest.importorskip("peft", reason="peft 后端测试需要 peft")
    from pepdesign.models.lora import _inject_lora_peft, load_adapter, save_adapter

    cfg = tiny_lora_config().model_copy(update={"backend": "peft"})
    model = _inject_lora_peft(build_tiny_model(vocab_size=len(tokenizer)), cfg)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.add_(0.01)

    path = save_adapter(model, tmp_path / "peft_adapter", tokenizer=tokenizer, cfg=cfg)

    fresh = _inject_lora_peft(build_tiny_model(vocab_size=len(tokenizer)), cfg)
    loaded = load_adapter(fresh, path, is_trainable=True)

    saved = {k: v.detach() for k, v in model.named_parameters() if "lora_" in k}
    got = {k: v.detach() for k, v in loaded.named_parameters() if "lora_" in k}
    assert saved.keys() == got.keys(), "加载后的 adapter key 与保存时不一致"
    for key in saved:
        assert torch.allclose(saved[key], got[key]), f"{key} 加载后数值不一致"
    assert any(p.requires_grad for p in loaded.parameters()), "加载后没有可训练参数"


def test_grpo_smoke_survives_gc_with_lora_dropout(monkeypatch):
    """GC + LoRA dropout>0 时，dropout 必须一直关到 backward 结束。

    这是 2026-10-26 修 old_logps 时引入的回归：`run_step` 内部把 dropout
    关掉算完 logprob 就恢复了 p，而 gradient checkpointing 的反向会
    重新跑一遍前向。重算时 p 从 0 变回 0.1，torch 的 checkpoint 会报
    `Recomputed values ... have different metadata`，或者静默给出不一致梯度。
    真实配置是 GC=true + dropout=0.05，而 smoke 配置用 GC=false/dropout=0，
    所以这条必须显式构造触发条件，并且走 `run_grpo_smoke` 的真实循环
    （只测 `_dropout_disabled` 本身会漏掉"调用方忘了包 backward"）。
    """
    import pepdesign.train.grpo as grpo

    cfg = load_config("configs/smoke_cpu.yaml")
    cfg.grpo.gradient_checkpointing = True
    base = grpo.tiny_lora_config()
    monkeypatch.setattr(
        grpo,
        "tiny_lora_config",
        lambda: base.model_copy(update={"lora_dropout": 0.1}),
    )

    report = grpo.run_grpo_smoke(cfg, steps=1)
    assert report["steps"] == 1
    assert report["losses_finite"], "GC + dropout 下 loss 不是有限值"
