"""训练层：阶段一 SFT（LoRA）与阶段二 GRPO（多维奖励）。

两个阶段均使用本包自写的训练循环，不依赖 trl；`train/calibration.py` 在校准集上
复用 SFT 的掩码与损失路径做预热。

模块按是否需要 torch 分组：`masking` / `advantages` / `rollout` / `dataset`
为纯逻辑，可被测试直接导入；`sft` / `grpo` / `calibration` 在函数内部延迟导入
torch。因此本 `__init__` 不做任何 eager import，顶层也不导入 torch。
"""

from __future__ import annotations

__all__ = ["advantages", "dataset", "masking", "rollout"]
