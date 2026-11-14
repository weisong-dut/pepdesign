"""推理服务层：OpenAI 兼容接口、静态批处理与 LoRA adapter 热切换。

分层：schemas（仅 pydantic）-> batcher（零依赖）-> engine（torch/transformers）
-> app（fastapi/uvicorn），依赖只向下。

本包不做 eager import：顶层导入 fastapi 会让 `batcher` / `schemas` 在缺少
可选依赖时也无法导入，而这两个模块应可在任意环境使用。
"""

from __future__ import annotations

__all__ = ["app", "batcher", "engine", "schemas"]
