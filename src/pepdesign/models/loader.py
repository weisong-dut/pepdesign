"""基座与 tokenizer 的加载，以及设备/精度决策。

`resolve_plan()` 把机器能力与配置意愿映射成可打印、可断言的加载方案，
`load_base_model` 负责执行。无 CUDA 时回退 CPU + fp32 并关闭混合精度与量化：
bf16/fp16 在 CPU 上会让 TrainingArguments 拒绝启动。
"""

from __future__ import annotations

from dataclasses import dataclass

from pepdesign.utils.config import LoRAConfig, QuantConfig
from pepdesign.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 设备 / 精度决策（纯逻辑，无 torch）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DevicePlan:
    """模型实际加载方案。"""

    device: str            # "cuda" | "mps" | "cpu"
    dtype: str             # "bfloat16" | "float16" | "float32"
    load_in_4bit: bool
    load_in_8bit: bool
    reason: str            # 人类可读的一句话，直接打进日志

    @property
    def is_fallback(self) -> bool:
        """是否发生了回退（配置想让 GPU 跑，实际在 CPU 上）。"""
        return self.device == "cpu"


def resolve_plan(
    *,
    cuda_available: bool,
    bf16_supported: bool = True,
    mps_available: bool = False,
    want_bf16: bool = True,
    want_fp16: bool = False,
    quant: QuantConfig | None = None,
) -> DevicePlan:
    """把机器能力与配置意愿映射成实际加载方案。

    能力全部由参数传入而非在函数内部探测，便于在无 GPU 环境下覆盖各分支。
    量化在 CPU/MPS 上无条件降级而非报错，并在 reason 中说明。
    """
    q = quant or QuantConfig()

    if cuda_available:
        if want_bf16 and bf16_supported:
            dtype, why = "bfloat16", "CUDA + bf16 支持，混合精度训练"
        elif want_bf16 or want_fp16:
            dtype, why = "float16", "CUDA 但该卡不支持 bf16（如 V100），退回 fp16 + loss scaling"
        else:
            dtype, why = "float32", "CUDA 但显式关闭了混合精度"
        return DevicePlan(
            device="cuda",
            dtype=dtype,
            load_in_4bit=q.enabled and q.load_in_4bit,
            load_in_8bit=q.enabled and q.load_in_8bit,
            reason=why,
        )

    if mps_available:
        return DevicePlan(
            device="mps",
            dtype="float32",
            load_in_4bit=False,
            load_in_8bit=False,
            reason="Apple MPS：强制 fp32（MPS 的 bf16 支持按算子和系统版本分化，"
                   "静默回退会让训练慢一个量级）；bitsandbytes 无 MPS 后端，量化关闭",
        )

    return DevicePlan(
        device="cpu",
        dtype="float32",
        load_in_4bit=False,
        load_in_8bit=False,
        reason="没有可用的 GPU/MPS，回退 CPU + fp32；"
               "量化与混合精度均已关闭（bf16/fp16 在 CPU 上会让 TrainingArguments 拒绝启动）。",
    )


def probe_hardware() -> dict:
    """探测一次硬件能力，结果供 `resolve_plan` 使用。

    未安装 torch 时返回全 False 而不抛异常，便于纯 CPU 环境 import。
    """
    try:
        import torch
    except ImportError:
        return {
            "cuda_available": False,
            "bf16_supported": False,
            "mps_available": False,
            "torch_available": False,
        }
    cuda = torch.cuda.is_available()
    return {
        "cuda_available": cuda,
        # is_bf16_supported 在低版本 torch 上不存在，用 getattr 兜底
        "bf16_supported": bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        if cuda
        else False,
        "mps_available": bool(getattr(torch.backends, "mps", None)
                              and torch.backends.mps.is_available()),
        "torch_available": True,
    }


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------
def _require_transformers():
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "加载模型需要 transformers，当前环境没有装。"
            "请装训练依赖：pip install -e '.[train]'"
        ) from exc
    return transformers


def load_tokenizer(model_name_or_path: str):
    """加载 tokenizer。

    SFT 与 GRPO 必须共用同一个对象，否则生成与重算 logprob 时的 token 边界不一致。
    """
    transformers = _require_transformers()
    tok = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    # 因果语言模型的 tokenizer 常无 pad token；缺失时 padding 会失败或被静默替换成
    # eos，导致 attention mask 失效。
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def chat_tokenize(
    tokenizer, messages: list[dict], *, add_generation_prompt: bool = False
) -> list[int]:
    """用 chat template 渲染并 tokenize，返回 `list[int]`。

    显式传 `return_dict=False` 并校验返回类型：`BatchEncoding` 的 `len()` 返回
    键数而非 token 数，会让长度预检与掩码静默出错。
    """
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=False,
    )
    if not isinstance(ids, list) or (ids and not isinstance(ids[0], int)):
        raise TypeError(
            f"apply_chat_template(return_dict=False) 返回了 {type(ids).__name__}，"
            f"预期 list[int]：{str(ids)[:80]}。"
            f"这通常是 transformers 又改了 return_dict 的行为——"
            f"`list(BatchEncoding)` 拿到的是它的键而不是 token id，"
            f"而 `len()` 会返回 2，于是掩码和长度预检都会静默出错。"
        )
    return ids


#: 探针内容取单个 ASCII 字符：便于从模板渲染的尾巴中认出模板追加的终止符，
#: 也避免汉字被切成多个 token。
_PROBE_USER = "x"
_PROBE_ASSISTANT = "y"


def generation_stop_ids(tokenizer) -> set[int]:
    """返回生成/采样应当停止的 token id 集合。

    终止符由 chat template 的渲染结果推出，识别失败时降级为 {pad, eos} 并告警。
    """
    stop = {
        tid
        for tid in (
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
        )
        if tid is not None
    }

    probe = [{"role": "user", "content": _PROBE_USER}]
    try:
        prompt_ids = chat_tokenize(tokenizer, probe, add_generation_prompt=True)
        full_ids = chat_tokenize(
            tokenizer, [*probe, {"role": "assistant", "content": _PROBE_ASSISTANT}]
        )
        content_ids = tokenizer.encode(_PROBE_ASSISTANT, add_special_tokens=False)
    except Exception as exc:  # noqa: BLE001 - 模板渲染失败不该阻断训练
        logger.warning(
            "无法用探针对话渲染 chat template（%s: %s），"
            "停止集退化成 {pad, eos}。若模板的终止符不是 eos，"
            "rollout 会跑满 max_completion_length。",
            type(exc).__name__, exc,
        )
        return stop

    if len(full_ids) <= len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
        logger.warning(
            "chat template 在 add_generation_prompt=True/False 下的前缀不一致，"
            "无法推出回合终止符；停止集退化成 {pad, eos}。"
        )
        return stop

    tail = full_ids[len(prompt_ids):]
    if len(tail) <= len(content_ids) or tail[: len(content_ids)] != content_ids:
        logger.warning(
            "探针尾巴（%s）与单独编码的回复内容（%s）对不上，"
            "无法推出回合终止符；停止集退化成 {pad, eos}。",
            tail, content_ids,
        )
        return stop

    terminator = tail[-1]
    token_str = tokenizer.convert_ids_to_tokens(terminator)
    if not (
        isinstance(token_str, str)
        and token_str.startswith("<|")
        and token_str.endswith("|>")
    ):
        logger.warning(
            "模板在 assistant 末尾追加的 token（id=%s, %r）不像对话控制 token，"
            "停止集退化成 {pad, eos}。",
            terminator, token_str,
        )
        return stop

    stop.add(int(terminator))
    return stop


def load_base_model(
    model_name_or_path: str,
    *,
    quant: QuantConfig | None = None,
    want_bf16: bool = True,
    want_fp16: bool = False,
    plan: DevicePlan | None = None,
):
    """加载基座模型，按 `resolve_plan` 的结论决定设备、精度、量化。"""
    transformers = _require_transformers()

    if plan is None:
        hw = probe_hardware()
        plan = resolve_plan(
            cuda_available=hw["cuda_available"],
            bf16_supported=hw["bf16_supported"],
            mps_available=hw["mps_available"],
            want_bf16=want_bf16,
            want_fp16=want_fp16,
            quant=quant,
        )

    logger.info("模型加载方案：device=%s, dtype=%s, 4bit=%s —— %s",
                plan.device, plan.dtype, plan.load_in_4bit, plan.reason)
    if plan.is_fallback:
        logger.warning(
            "在 CPU 上训练：可以跑通流程，但 3B 模型的真实微调"
            "在这个规模上不现实。CPU 的用途是验证代码路径（用小模型 + 小步数），"
            "正式训练请换到有 GPU 的机器。"
        )

    import torch

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[plan.dtype]

    kwargs: dict = {"dtype": dtype}
    if plan.load_in_4bit or plan.load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "配置要求 4bit/8bit 量化，需要 bitsandbytes："
                "pip install -e '.[train]'。"
                "在 Windows + CPU 上 bitsandbytes 没有可用后端，"
                "请把 quant.enabled 设为 false（训练脚本会自动回退）。"
            ) from exc
        q = quant or QuantConfig()
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=plan.load_in_4bit,
            load_in_8bit=plan.load_in_8bit,
            bnb_4bit_quant_type=q.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, q.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=q.bnb_4bit_use_double_quant,
        )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path, **kwargs
    )
    return model.to(plan.device), plan


# ---------------------------------------------------------------------------
# 梯度检查点
# ---------------------------------------------------------------------------
def enable_gradient_checkpointing(model, *, enabled: bool = True) -> dict:
    """按配置开启梯度检查点，返回可打印的报告（enabled / reason / use_cache 等）。

    显式设置 `config.use_cache=False`；模型不支持或开启失败时只告警并返回
    enabled=False，不影响训练启动。
    """
    if not enabled:
        return {"enabled": False, "reason": "配置里 gradient_checkpointing=false"}

    if not getattr(model, "supports_gradient_checkpointing", False):
        reason = (
            f"{type(model).__name__} 声明不支持梯度检查点"
            f"（supports_gradient_checkpointing=False），本次运行关闭它。"
            f"显存不够的话这条是第一个要去查的地方。"
        )
        logger.warning("%s", reason)
        return {"enabled": False, "reason": reason}

    original_use_cache = getattr(model.config, "use_cache", None)
    try:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    except Exception as exc:  # noqa: BLE001 - 显存优化不该让训练起不来
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache
        reason = (
            f"开启梯度检查点失败（{type(exc).__name__}: {exc}），本次运行关闭它。"
            f"训练继续，但显存占用会比预期高。"
        )
        logger.warning("%s", reason)
        return {"enabled": False, "reason": reason}

    gets_input_grads = getattr(model, "_require_grads_hook", None) is not None
    report = {
        "enabled": True,
        "reason": "已开启：反向时重算激活值，用算力换显存（结论：单卡能塞下 3B）",
        # 记下来是为了让"LoRA 到底有没有梯度"这件事可观测：
        # 它是 False 时，检查点区域内的 LoRA 分支可能拿不到梯度。
        "input_require_grads": gets_input_grads,
        "use_cache": getattr(model.config, "use_cache", None),
        "generation_use_cache": getattr(
            getattr(model, "generation_config", None), "use_cache", None
        ),
    }
    logger.info(
        "梯度检查点已开启（config.use_cache=%s，generation_config.use_cache=%s，"
        "输入带梯度=%s）",
        report["use_cache"], report["generation_use_cache"], gets_input_grads,
    )
    return report


# ---------------------------------------------------------------------------
# 冒烟用的小模型
# ---------------------------------------------------------------------------
def build_tiny_model(
    *,
    vocab_size: int,
    hidden_size: int = 64,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    intermediate_size: int = 128,
    max_position_embeddings: int = 2048,
    seed: int = 42,
):
    """构造随机初始化的小模型，不下载权重，用于 CPU 端到端冒烟。

    `num_key_value_heads` 必须显式传入：`LlamaConfig` 默认 None 会退化成 MHA，
    与 LOGOS-3B 的 GQA 结构不一致；架构类必须是 `Llama*`。
    """
    transformers = _require_transformers()
    import torch

    torch.manual_seed(seed)
    config = transformers.LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
    )
    return transformers.LlamaForCausalLM(config)


def tiny_lora_config() -> LoRAConfig:
    """冒烟测试用的 LoRA 配置：参数量小，后端固定为 manual。"""
    return LoRAConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj"],
        backend="manual",
    )
