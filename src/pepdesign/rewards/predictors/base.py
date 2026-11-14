"""预测器抽象接口。

奖励函数通过稳定的接口调用预测器，具体实现可按成本与精度替换，
无需改动奖励代码。子类需要提供 `_predict_one`；有批量能力的实现
应覆写 `predict` 或 `_predict_batch`。

`confidence` 与 `is_calibrated` 分开暴露：前者描述单次预测的可信度，
后者描述预测器是否经过实验数据标定，聚合层据此分别处理。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from pepdesign.bio.sequence import SequenceProperties, compute_properties


@dataclass
class PredictionResult:
    """一次预测的结果，包含预测值、置信度与可选的明细。"""

    value: float
    confidence: float = 1.0
    detail: dict | None = None
    predictor: str = ""

    def as_dict(self) -> dict:
        return {
            "value": round(self.value, 4),
            "confidence": round(self.confidence, 4),
            "predictor": self.predictor,
            "detail": self.detail or {},
        }


class BasePredictor(ABC):
    """所有预测器的基类。

    子类至少要实现 `_predict_one`；批量预测默认走朴素循环，
    有批量能力的实现（如 ESM 编码器）应该覆写 `predict` 拿批量加速。
    """

    #: 预测器名字，会出现在奖励明细和评估报告里
    name: str = "base"
    #: 该预测器的输出是否经过实验数据标定
    is_calibrated: bool = False

    def __init__(self, *, cache_size: int = 100_000) -> None:
        # 缓存：GRPO 每步对 group_size × batch_size 条序列打分，后期序列高度同质，
        # 命中率通常可达 30%-60%。
        from pepdesign.utils.io import JsonlCache

        self._cache = JsonlCache(max_size=cache_size)

    # ------------------------------------------------------------------
    def predict(self, sequences: Sequence[str]) -> list[PredictionResult]:
        """批量预测，带缓存。

        返回值必须与输入等长且逐位对应，否则奖励会与序列错位；
        `_predict_batch` 返回数量不符时直接报错。
        """
        results: list[PredictionResult | None] = [None] * len(sequences)
        missing: list[int] = []

        for i, seq in enumerate(sequences):
            cached = self._cache.get(self._cache_key(seq))
            if cached is not None:
                results[i] = cached
            else:
                missing.append(i)

        if missing:
            fresh = self._predict_batch([sequences[i] for i in missing])
            # 子类覆写 _predict_batch 时若数量不符，zip 会静默截断，因此在源头校验。
            if len(fresh) != len(missing):
                raise RuntimeError(
                    f"预测器 {self.name!r} 的 _predict_batch 返回了 "
                    f"{len(fresh)} 条结果，但输入是 {len(missing)} 条。"
                    f"批量接口必须等长逐位对应，否则奖励会与序列错位。"
                )
            # 已校验等长，strict=True 用于兜底校验被移除的情况。
            for idx, res in zip(missing, fresh, strict=True):
                results[idx] = res
                self._cache.put(self._cache_key(sequences[idx]), res)

        # 兜底断言：结果列表不应有 None。
        assert all(r is not None for r in results), (
            f"预测器 {self.name!r} 有 {sum(r is None for r in results)} 条结果缺失"
        )
        return list(results)  # type: ignore[return-value]

    def predict_one(self, sequence: str) -> PredictionResult:
        """单条预测。

        不直接调用 `predict([sequence])[0]`，以便返回为空时给出带预测器与序列的报错。
        """
        results = self.predict([sequence])
        if not results:
            raise RuntimeError(
                f"预测器 {self.name!r} 对序列 {sequence!r} 没有返回任何结果"
            )
        return results[0]

    # ------------------------------------------------------------------
    def _predict_batch(self, sequences: Sequence[str]) -> list[PredictionResult]:
        """默认批量实现：逐条算。有批量能力的子类应该覆写。"""
        return [self._predict_one(s) for s in sequences]

    @abstractmethod
    def _predict_one(self, sequence: str) -> PredictionResult:
        """单条预测。子类必须实现。"""

    # ------------------------------------------------------------------
    def _cache_key(self, sequence: str) -> str:
        """缓存键，默认为序列本身。

        子类若有影响输出的额外参数（如环化状态），必须覆写此方法把参数编入 key。
        """
        return sequence

    # ------------------------------------------------------------------
    @staticmethod
    def props(sequence: str) -> SequenceProperties:
        """取序列的物化性质快照（各预测器共用）。"""
        return compute_properties(sequence)

    def stats(self) -> dict:
        return {"predictor": self.name, "cache": self._cache.stats()}
