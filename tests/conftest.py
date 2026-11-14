"""pytest 公共配置。

把 `src/` 加入 sys.path，使测试无需 `pip install -e .` 即可 import pepdesign。

约束：tests/ 不允许 import torch/transformers。项目约定可选依赖只在真正
使用处导入（见 rewards/predictors/factory.py 的 _lazy），测试需证明这一点。
任何测试触发 torch 导入都说明违反了该约定，应在代码侧修复。
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
