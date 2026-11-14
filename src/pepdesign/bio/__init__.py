"""生物序列基础层：常量表、序列性质、酶切、环化约束。

这一层是纯 CPU、零依赖的领域内核。
刻意不 import torch / transformers，目的是让奖励函数、数据清洗、SOP 报告
都能在没有 GPU 的机器上跑——评估和运营侧本来就不该需要显卡。
"""

from pepdesign.bio import aa, cleavage, cyclization, sequence

__all__ = ["aa", "cleavage", "cyclization", "sequence"]
