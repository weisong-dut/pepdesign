"""模型层：基座加载、LoRA 注入与合并。

三个模块分工：`lora_layer.py` 提供手写 LoRA 实现（LoRALinear / 注入 / 存取 / 合并），
`lora.py` 是对外门面并按 `LoRAConfig.backend` 分发到 manual 或 peft，
`loader.py` 负责基座与 tokenizer 加载及设备/精度决策。

本包 `__init__` 不做 eager import：torch / transformers 属于可选依赖，
顶层引入会让整个包在纯 CPU 环境下 import 失败。
"""

from __future__ import annotations

__all__ = ["loader", "lora", "lora_layer"]
