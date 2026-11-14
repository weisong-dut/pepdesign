"""推理引擎：加载基座与 LoRA adapter，按批生成，支持两种后端。

`InferConfig.backend` 选择后端：transformers（默认，纯文本栈，CPU 可运行，
静态批处理）或 vllm（需 CUDA 与单独安装的 vllm 包，提供 PagedAttention、
连续批与前缀缓存）。

vLLM 分支需要 GPU，未在无 CUDA 环境验证；可验证部分（配置分发、peft 格式
导出、缺依赖报错）有测试覆盖，实际加载与生成需在 GPU 环境核对。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from pepdesign.models.loader import (
    generation_stop_ids,
    load_base_model,
    load_tokenizer,
    probe_hardware,
    resolve_plan,
)
from pepdesign.models.lora import inject_lora, load_adapter
from pepdesign.utils.config import InferConfig, LoRAConfig, resolve_path
from pepdesign.utils.logger import get_logger

logger = get_logger(__name__)


def _require_vllm():
    """惰性导入 vllm，失败时给出可操作的报错。

    三类失败分开报告：未安装、版本不兼容、无可用 CUDA。
    """
    try:
        import vllm
    except ImportError as exc:
        raise ImportError(
            "backend='vllm' 需要 vllm 包，当前环境没有。\n"
            "  注意：vllm 会把它钉死的 torch 版本一起装进来，所以它不在 infer 组里。\n"
            "  请在 GPU 机器上执行：pip install -e '.[serve-vllm]'\n"
            "  没有 CUDA 时请把 InferConfig.backend 改回 'transformers'（默认值）。"
        ) from exc

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "vllm 已安装，但当前机器没有可用的 CUDA 设备，vLLM 起不来。\n"
            "  vLLM 需要 GPU（PagedAttention 依赖 CUDA kernel）。\n"
            "  请改用 backend='transformers'（CPU 上也能跑，只是吞吐低）。"
        )
    return vllm


@dataclass
class GenParams:
    """一次生成请求的采样参数，默认值已在引擎边界解析，引擎不再判断 None。"""

    max_tokens: int
    temperature: float
    top_p: float
    n: int = 1
    stop: list[str] = field(default_factory=list)
    echo: bool = False


@dataclass
class GenOutput:
    """一条 prompt 的全部生成结果（`n` 条）。"""

    texts: list[str]
    n_prompt_tokens: int
    n_completion_tokens: list[int]

    @property
    def finish_reason(self) -> str:
        return "length"  # 由调用方按 stop 命中情况覆盖


class ModelNotReadyError(RuntimeError):
    """服务尚未加载完成。映射为 HTTP 503。"""


class InferenceEngine:
    """持有模型与 tokenizer，对外提供批量生成。

    惰性加载：构造时不加载权重，需显式调用 `load()`。
    """

    def __init__(self, cfg: InferConfig, *, lora_cfg: LoRAConfig | None = None) -> None:
        self.cfg = cfg
        self.lora_cfg = lora_cfg or LoRAConfig()
        # `model` 在两种后端下类型不同：transformers 为 HF 模型，vllm 为 vllm.LLM。
        self.model = None
        self.tokenizer = None
        self._plan = None
        self._loaded_at: float | None = None
        self._adapter: str | None = None
        # vLLM 的 LoRA 按请求挂载；保存当前 adapter 的 peft 路径与自增 id。
        self._lora_request = None
        self._lora_id = 0

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self) -> None:
        """加载基座（可选 adapter）并切换到 eval。

        优先级：`merged_model_path` > `adapter_path`（基座 + LoRA）> 仅基座。
        """
        if self.cfg.backend == "vllm":
            self._load_vllm()
            return

        hw = probe_hardware()
        self._plan = resolve_plan(
            cuda_available=hw["cuda_available"],
            bf16_supported=hw["bf16_supported"],
            mps_available=hw["mps_available"],
            want_bf16=True,
            want_fp16=False,
            quant=self.cfg.quantization,
        )
        logger.info("推理设备决策：%s", self._plan.reason)

        self.tokenizer = load_tokenizer(self.cfg.base_model)

        if self.cfg.merged_model_path:
            path = Path(self.cfg.merged_model_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"merged_model_path 不存在: {path}。"
                    f"它应当指向 merge_and_save 导出的目录；"
                    f"如果还没有合并过，把这一项留空、改用 adapter_path。"
                )
            self.model, _ = load_base_model(
                str(path), quant=self.cfg.quantization, plan=self._plan
            )
            self._adapter = None
            logger.info("已加载合并模型：%s", path)
        else:
            self.model, _ = load_base_model(
                self.cfg.base_model, quant=self.cfg.quantization, plan=self._plan
            )
            if self.cfg.adapter_path:
                # 先注入 LoRA 骨架，再写入权重。
                self.model = inject_lora(self.model, self.lora_cfg)
                self.model = load_adapter(
                    self.model, self.cfg.adapter_path, is_trainable=False, cfg=self.lora_cfg
                )
                self._adapter = self.cfg.adapter_path
                logger.info("已挂载 adapter：%s", self._adapter)
            else:
                logger.warning(
                    "既没有 merged_model_path 也没有 adapter_path，"
                    "服务将以未微调的基座运行。这在 A/B 对照里是有用的，"
                    "但如果这不是你的本意，请检查配置。"
                )

        self.model.eval()
        self._loaded_at = time.time()

    # ------------------------------------------------------------------
    # vLLM 后端
    # ------------------------------------------------------------------
    def _ensure_peft_adapter(self, adapter_path: str) -> str:
        """确保存在 peft 格式的 adapter 目录并返回其路径。

        vLLM 按 peft 键名读取权重，需从手写格式导出；导出到兄弟目录
        `<adapter_path>_peft`，避免污染原训练产物目录。
        """
        from pepdesign.models.lora_layer import export_peft_adapter

        src = Path(adapter_path)
        if not src.is_absolute():
            src = resolve_path(src)
        dst = src.parent / f"{src.name}_peft"
        if dst.exists() and (dst / "adapter_model.safetensors").exists():
            return str(dst)
        export_peft_adapter(
            src, dst, base_model_name_or_path=self.cfg.base_model
        )
        return str(dst)

    def _load_vllm(self) -> None:
        """用 vLLM 加载基座（可选 LoRA）。

        该分支需要 CUDA，未在本地验证；kwargs 需在 GPU 环境逐项核对，
        尤其是 `quantization` 的取值写法。
        """
        vllm = _require_vllm()
        self.tokenizer = load_tokenizer(self.cfg.base_model)

        # vLLM 自行管理显存；`_plan` 仅用于 health() 报告。
        self._plan = resolve_plan(
            cuda_available=True,
            bf16_supported=True,
            want_bf16=True,
            quant=self.cfg.quantization,
        )

        model_path = self.cfg.merged_model_path or self.cfg.base_model
        kwargs: dict = {
            "model": str(model_path),
            "dtype": "bfloat16",
            "max_model_len": self.cfg.max_model_len,
            # 以下字段在 transformers 后端不生效，在 vLLM 后端会传给 vllm.LLM。
            "gpu_memory_utilization": self.cfg.gpu_memory_utilization,
            "tensor_parallel_size": self.cfg.tensor_parallel_size,
            # 单批最大并发序列数，vLLM 用作连续批调度上界。
            "max_num_seqs": self.cfg.max_num_seqs,
            # 前缀缓存：多轮优化中重复的靶点前缀可复用 KV。
            "enable_prefix_caching": self.cfg.enable_prefix_caching,
        }
        if self.cfg.quantization.enabled and self.cfg.quantization.load_in_4bit:
            kwargs["quantization"] = "bitsandbytes"
        if self.cfg.adapter_path and not self.cfg.merged_model_path:
            # merged 模型里已经把 LoRA 合进去了，不能再挂 adapter。
            kwargs["enable_lora"] = True
            kwargs["max_lora_rank"] = self.lora_cfg.r

        logger.info("加载 vLLM 后端：%s", {k: v for k, v in kwargs.items() if k != "model"})
        self.model = vllm.LLM(**kwargs)

        if self.cfg.adapter_path and not self.cfg.merged_model_path:
            self.set_adapter(self.cfg.adapter_path)
        elif not self.cfg.adapter_path and not self.cfg.merged_model_path:
            logger.warning(
                "既没有 merged_model_path 也没有 adapter_path，服务以未微调的基座运行。"
            )

        self._loaded_at = time.time()
        logger.info("vLLM 后端加载完成。")

    def _generate_vllm(
        self, prompts: list[str], params: GenParams
    ) -> list[GenOutput]:
        """vLLM 生成路径。

        返回的 `RequestOutput` 与入参 prompts 按下标一一对应；停止符由 vLLM
        处理，prompt token 数取 vLLM 报告的 `prompt_token_ids`。
        """
        from vllm import SamplingParams

        assert self.model is not None
        sampling = SamplingParams(
            n=params.n,
            temperature=params.temperature if params.temperature > 0 else 0.0,
            top_p=params.top_p,
            max_tokens=params.max_tokens,
            stop=list(params.stop) or None,
            # 回合终止符（如 `<|eot_id|>`）不是 eos，需一并传入。
            stop_token_ids=sorted(generation_stop_ids(self.tokenizer)),
        )

        outs = self.model.generate(
            prompts, sampling, lora_request=self._lora_request, use_tqdm=False
        )

        results: list[GenOutput] = []
        for out in outs:
            texts = [c.text for c in out.outputs]
            if params.stop:
                texts = [_truncate_at_stop(t, params.stop) for t in texts]
            results.append(
                GenOutput(
                    texts=texts,
                    n_prompt_tokens=len(out.prompt_token_ids or []),
                    n_completion_tokens=[len(c.token_ids) for c in out.outputs],
                )
            )

        assert len(results) == len(prompts), (
            f"vLLM 返回的结果条数 ({len(results)}) 与 prompt 条数 ({len(prompts)}) "
            f"不一致——下标对应关系已不可靠，必须报错而不是返回。"
        )
        return results

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------
    def generate_batch(
        self, prompts: list[str], params: GenParams
    ) -> list[GenOutput]:
        """对一批 prompt 做一次静态批量前向，返回与入参按下标一一对应的结果。

        `n > 1` 时使用 `num_return_sequences` 在同一次调用中生成多条。
        """
        if not self.is_ready:
            raise ModelNotReadyError(
                "引擎尚未加载。请先调用 load()——HTTP 层会把它映射成 503，"
                "让客户端知道「再试一次」而不是「请求写错了」。"
            )
        if self.cfg.backend == "vllm":
            return self._generate_vllm(prompts, params)

        import torch

        assert self.model is not None and self.tokenizer is not None
        tok = self.tokenizer

        # 左 padding：右 padding 时各行生成起点不同，解码后需按行裁剪，易差一位。
        prev_side = tok.padding_side
        tok.padding_side = "left"
        try:
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        finally:
            tok.padding_side = prev_side

        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        device = next(self.model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}

        gen_kwargs = {
            "max_new_tokens": params.max_tokens,
            "do_sample": params.temperature > 0,
            "num_return_sequences": params.n,
            "pad_token_id": tok.pad_token_id,
            # 必须显式传 eos：`generation_config` 的 eos 是 `<|end_of_text|>`，
            # 对话模板以 `<|eot_id|>` 结束回合。
            "eos_token_id": sorted(generation_stop_ids(tok)),
        }
        if params.temperature > 0:
            # temperature=0 时不传采样参数，避免 transformers 警告"采样参数被忽略"。
            gen_kwargs["temperature"] = params.temperature
            gen_kwargs["top_p"] = params.top_p

        gen_start = int(enc["input_ids"].shape[1])
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)

        rows = out.tolist()
        # n > 1 时 generate 输出为 prompt-major（repeat_interleave），
        # 每条 prompt 占连续 n 行；顺序错位不会报错，只会把结果串到别的需求上。
        results: list[GenOutput] = []
        for i, plen in enumerate(prompt_lens):
            chunk = rows[i * params.n: (i + 1) * params.n]
            texts = [
                tok.decode(r[gen_start:], skip_special_tokens=True) for r in chunk
            ]
            if params.stop:
                texts = [_truncate_at_stop(t, params.stop) for t in texts]
            results.append(
                GenOutput(
                    texts=texts,
                    n_prompt_tokens=int(plen),
                    n_completion_tokens=[len(r) - gen_start for r in chunk],
                )
            )

        assert len(results) == len(prompts), (
            f"批量生成的结果条数 ({len(results)}) 与 prompt 条数 ({len(prompts)}) 不一致——"
            f"结果与需求的下标对应关系已经不可靠，必须报错而不是返回。"
        )
        return results

    # ------------------------------------------------------------------
    # adapter 热切换
    # ------------------------------------------------------------------
    def list_adapters(self) -> list[str]:
        """列出 outputs/ 下所有可用的 adapter 目录。"""
        out = Path("outputs")
        if not out.exists():
            return []
        found = []
        for p in sorted(out.rglob("adapter_config.json")):
            found.append(str(p.parent))
        return found

    def set_adapter(self, path: str | None) -> None:
        """切换 adapter（热插拔）。

        仅支持与当前 LoRA 配置形状一致的 adapter（同 r、同 target_modules）；
        形状不一致时加载会报错。更换形状不同的 adapter 需重新加载模型。
        """
        if not self.is_ready:
            raise ModelNotReadyError("引擎尚未加载，无法切换 adapter。")
        if path is None:
            raise ValueError(
                "set_adapter(None) 不支持——把 adapter 卸掉需要重新从基座加载模型。"
                "如果你要的是「回到未微调状态」，请重启服务并清空 adapter_path。"
            )

        if self.cfg.backend == "vllm":
            # vLLM 只替换下次请求携带的 LoRARequest，不重新加载模型。
            from vllm.lora.request import LoRARequest

            peft_dir = self._ensure_peft_adapter(path)
            self._lora_id += 1
            self._lora_request = LoRARequest(
                f"adapter-{self._lora_id}", self._lora_id, peft_dir
            )
            self._adapter = path
            logger.info("已切换到 adapter（vLLM LoRARequest）：%s -> %s", path, peft_dir)
            return

        assert self.model is not None
        self.model = load_adapter(
            self.model, path, is_trainable=False, cfg=self.lora_cfg
        )
        self._adapter = path
        logger.info("已切换到 adapter：%s", path)

    # ------------------------------------------------------------------
    # 健康检查
    # ------------------------------------------------------------------
    def health(self) -> dict:
        """给 /health 的状态快照。

        必须报告 `device` / `dtype`：不同设备与精度的延迟差一个量级。
        """
        plan = self._plan
        return {
            "status": "ok" if self.is_ready else "loading",
            # 后端必须如实上报，监控与压测脚本据此选择超时和并发。
            "backend": self.cfg.backend,
            "model": self.cfg.base_model if not self.cfg.merged_model_path
                     else self.cfg.merged_model_path,
            "adapter": self._adapter,
            "device": plan.device if plan else "unknown",
            "dtype": plan.dtype if plan else "unknown",
            "max_model_len": self.cfg.max_model_len,
            "detail": "" if self.is_ready else "模型尚未加载完成",
        }


def _truncate_at_stop(text: str, stops: list[str]) -> str:
    """在第一个命中的 stop 字符串处截断。

    取最早出现的匹配位置，而非列表中第一个命中的 stop。
    """
    cut = len(text)
    for s in stops:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]
