"""运营评估层：离线指标、A/B 对照、业务指标。

本层不依赖 torch / trl / vllm：输入是已生成的预测值与实测值，
只需要排序和相关系数，numpy 足够。因此它可以在只装 core 依赖的
环境中运行，并由纯逻辑测试覆盖（见 tests/test_eval_offline.py）。

* `offline.py`      —— 秩相关、误差、Top-K 富集、置换检验，均为纯函数。
* `reward_audit.py` —— 用上述指标比较奖励预测器与实验实测值。

输出重点是"结论有多少证据"，而不是"模型多好"。样本量不足时，
`correlation_report` 会拒绝判定并说明原因，而不是给出一个无法解释的
p 值。这与奖励层 `is_calibrated=False` 的处理方式一致。
"""

from __future__ import annotations

__all__ = ["offline", "reward_audit"]
