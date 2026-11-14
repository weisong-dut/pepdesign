"""靶点亲和力预测。

本模块实现基于序列特征的亲和力代理，目标是在序列被改坏时给出明确惩罚，
而不是精确预测结合自由能。奖励只需在正确方向上单调，并对钻空子的方向为负。
滑动窗口特征覆盖三类主要破坏：疏水作用丧失、盐桥丧失、二级结构被打乱。

`predict_with_reference()` 比较候选与参考序列在同一功能面上的窗口分数，
是 RL 使用的相对口径。
"""

from __future__ import annotations

from pepdesign.bio.aa import KYTE_DOOLITTLE
from pepdesign.bio.sequence import (
    hydrophobic_moment,
    net_charge,
    standard_only,
)
from pepdesign.rewards.predictors.base import BasePredictor, PredictionResult

# 窗口长度：多肽与靶点相互作用的功能基序典型长度是 5-9 个残基
# （对应一个螺旋面或一个 β-发夹的一侧）。取 7 是文献里的常见取值。
DEFAULT_WINDOW: int = 7


def _window_features(window: str) -> dict[str, float]:
    """单个窗口的结合相关特征。

    四个特征分别对应疏水作用、静电作用、两亲性与芳香堆积。
    """
    n = len(window)
    if n == 0:
        return {"hydrophobic": 0.0, "charged": 0.0, "amphipathic": 0.0, "aromatic": 0.0}

    hydro = sum(KYTE_DOOLITTLE.get(c, 0.0) for c in window) / n
    # KD 标度范围约 [-4.5, 4.5]，映射到 [-1, 1]
    hydrophobic = hydro / 4.5

    charged = net_charge(window, 7.4) / max(n, 1) * 3.0  # 每残基平均电荷，放大到可比较
    amphipathic = hydrophobic_moment(window, angle_deg=100.0) if n >= 4 else 0.0
    aromatic = sum(1 for c in window if c in "FWY") / n

    return {
        "hydrophobic": hydrophobic,
        "charged": charged,
        "amphipathic": amphipathic,
        "aromatic": aromatic,
    }


def _window_score(features: dict[str, float]) -> float:
    """把窗口特征合成一个结合倾向分。

    权重反映各相互作用类型的相对贡献。`charged` 取绝对值，因为序列层面
    只能衡量静电能力，无法判断方向是否与靶点匹配。
    """
    return (
        0.45 * abs(features["hydrophobic"])
        + 0.30 * min(abs(features["charged"]), 1.0)
        + 0.15 * features["amphipathic"]
        + 0.10 * features["aromatic"]
    )


class AffinityPredictor(BasePredictor):
    """基于序列特征的靶点亲和力代理（feature-based）。"""

    name = "affinity_feature"

    def __init__(self, *, window: int = DEFAULT_WINDOW, **kwargs) -> None:
        super().__init__(**kwargs)
        self.window = window

    def _cache_key(self, sequence: str) -> str:
        return f"{sequence}|w{self.window}"

    # ------------------------------------------------------------------
    def window_profile(self, sequence: str) -> list[tuple[int, float]]:
        """返回每个窗口的 (起始下标, 分数)。用于解释和可视化。"""
        s = standard_only(sequence)
        w = min(self.window, len(s))
        if w < 2:
            return []
        return [
            (i, round(_window_score(_window_features(s[i : i + w])), 4))
            for i in range(len(s) - w + 1)
        ]

    def _predict_one(self, sequence: str) -> PredictionResult:
        profile = self.window_profile(sequence)
        if not profile:
            return PredictionResult(
                value=0.0, confidence=0.1,
                detail={"reason": "sequence_too_short"},
                predictor=self.name,
            )

        best_idx, best_score = max(profile, key=lambda t: t[1])
        confidence = 0.4 * min(1.0, len(sequence) / 10.0)

        return PredictionResult(
            value=round(best_score, 4),
            confidence=round(confidence, 4),
            detail={
                "best_window_start": best_idx,
                "best_window": standard_only(sequence)[best_idx : best_idx + self.window],
                "n_windows": len(profile),
                "note": "基于序列特征的亲和力代理，未经过结合实验标定",
            },
            predictor=self.name,
        )

    # ------------------------------------------------------------------
    def predict_with_reference(
        self, sequence: str, reference: str
    ) -> PredictionResult:
        """相对参考序列的亲和力保持度 ∈ [0, 1.5]，1 = 完全保持。

        取参考序列最优窗口，在待评序列上找位置对应的窗口并比较分数。
        用位置对应而非各自取最大值，防止模型只保留任意一个高分窗口。
        """
        ref_profile = self.window_profile(reference)
        seq_profile = self.window_profile(sequence)

        if not ref_profile or not seq_profile:
            return PredictionResult(value=0.5, confidence=0.1, predictor=self.name)

        best_ref_idx, best_ref_score = max(ref_profile, key=lambda t: t[1])
        if best_ref_score <= 1e-6:
            # 参考序列本身没有明显功能面，无从比较，给中性分
            return PredictionResult(value=0.5, confidence=0.2, predictor=self.name)

        # 找待评序列上距离 best_ref_idx 最近的窗口（位置对应）
        nearest = min(seq_profile, key=lambda t: abs(t[0] - best_ref_idx))
        ratio = nearest[1] / best_ref_score

        # 长度差异作为额外的置信度折扣
        length_similarity = min(len(sequence), len(reference)) / max(
            len(sequence), len(reference), 1
        )

        return PredictionResult(
            value=round(max(0.0, min(1.5, ratio)), 4),   # 允许超过 1（变得更好）
            confidence=round(0.4 * length_similarity, 4),
            detail={
                "reference_window_start": best_ref_idx,
                "reference_window_score": round(best_ref_score, 4),
                "candidate_window_start": nearest[0],
                "candidate_window_score": round(nearest[1], 4),
                "positional_match": abs(nearest[0] - best_ref_idx) <= 2,
            },
            predictor=self.name,
        )


class ESMEmbeddingAffinityPredictor(BasePredictor):
    """基于 ESM 嵌入的亲和力预测（可选，需要 torch）。

    保留接口与实现骨架作为升级路径：`_embed` 可替换为真实 ESM 前向，
    奖励层无需改动。当前未接回归头、未标定，主线不使用。
    """

    name = "affinity_esm"
    is_calibrated = False

    def __init__(self, *, model_name: str = "facebook/esm2_t12_35M_UR50D", **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _load(self) -> bool:
        """惰性加载模型。加载失败时返回 False，由调用方降级。"""
        if self._model is not None:
            return True
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            return True
        except Exception as exc:  # noqa: BLE001 - 缺依赖或没有权重都走降级路径
            import logging

            logging.getLogger(__name__).warning(
                "ESM 预测器不可用（%s），将回退到特征代理", exc
            )
            return False

    def _embed(self, sequence: str):  # pragma: no cover - 需要 torch 与权重
        """取序列的池化嵌入。"""
        import torch

        assert self._tokenizer is not None and self._model is not None
        inputs = self._tokenizer(sequence, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)
        hidden = outputs.last_hidden_state[0, 1 : len(sequence) + 1]  # 去掉 BOS/EOS

        # 功能面加权池化：用滑动窗口疏水矩做权重
        weights = torch.ones(hidden.shape[0], device=hidden.device)
        s = standard_only(sequence)
        w = min(DEFAULT_WINDOW, len(s))
        if w >= 4 and len(s) >= w:
            from pepdesign.bio.sequence import hydrophobic_moment_profile

            profile = hydrophobic_moment_profile(s, window=w)
            # 把窗口分数摊回每个残基（居中对齐），再做归一化
            per_residue = torch.zeros(hidden.shape[0], device=hidden.device)
            counts = torch.zeros(hidden.shape[0], device=hidden.device)
            for i, score in enumerate(profile):
                center = i + w // 2
                if center < per_residue.shape[0]:
                    per_residue[center] += score
                    counts[center] += 1
            counts[counts == 0] = 1
            per_residue = per_residue / counts
            weights = 1.0 + 3.0 * (per_residue / (per_residue.max() + 1e-6))

        pooled = (hidden * weights.unsqueeze(-1)).sum(dim=0) / weights.sum()
        return pooled

    def _predict_one(self, sequence: str) -> PredictionResult:
        if not self._load():
            # 降级时明确标注实际使用的实现，避免日志中无法区分。
            fallback = AffinityPredictor()
            result = fallback._predict_one(sequence)
            result.predictor = f"{self.name}:fallback->{fallback.name}"
            result.confidence *= 0.5
            return result

        # 回归头未标定，仅输出嵌入范数作为占位
        with __import__("torch").no_grad():
            emb = self._embed(sequence)
            value = float(emb.norm().item())
        return PredictionResult(
            value=value, confidence=0.1,
            detail={"note": "嵌入尚未接回归头标定，仅占位"},
            predictor=self.name,
        )
