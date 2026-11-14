"""ESM-2 掩码边缘概率打分器：独立于规则引擎的第五个信号源。

逐位置将残基替换为 `<mask>`，取模型对真实氨基酸的 log 概率并求平均，
得到序列的伪似然分数。该分数零样本、不需要标定，适合作为"序列是否像天然
蛋白"的审计信号。

分数是相对的：不同长度序列间不可直接比较，须相对同一参照使用。
由于归一化区间 `[PLM_BAD_LOGPROB, PLM_GOOD_LOGPROB]` 是启发式的，
`is_calibrated=False`。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pepdesign.bio.sequence import standard_only
from pepdesign.rewards.predictors.base import BasePredictor, PredictionResult
from pepdesign.utils.logger import get_logger

logger = get_logger(__name__)

#: ESM-2 的掩码 token。ESM 系列统一用 `<mask>`，不是 [MASK]。
ESM_MASK_TOKEN = "<mask>"

#: 默认权重：150M 在 CPU 可运行且对短肽有足够分辨力；更大模型可经配置替换。
DEFAULT_ESM_MODEL = "facebook/esm2_t30_150M_UR50D"

#: 归一化区间：天然蛋白 per-residue log 概率约 -1.5 ~ -3.5，线性映射到 [0,1]。
#: 这是启发式映射，不是标定，因此 is_calibrated=False。
PLM_GOOD_LOGPROB = -1.5
PLM_BAD_LOGPROB = -4.0


# ---------------------------------------------------------------------------
# 纯逻辑层（不 import torch）
# ---------------------------------------------------------------------------
def masked_variants(sequence: str) -> list[str]:
    """生成 L 条"掩掉一个位置"的变体，顺序与位置一一对应。

    返回字符串比输入长（`<mask>` 占 6 个字符但代表一个残基），
    不要在其上做字符级下标运算；需要拆分时用 `partition(ESM_MASK_TOKEN)`。
    """
    if not sequence:
        raise ValueError("序列不能为空")
    return [f"{sequence[:i]}{ESM_MASK_TOKEN}{sequence[i + 1:]}" for i in range(len(sequence))]


def marginal_logprobs_to_score(
    logprobs: Sequence[float],
    *,
    good: float = PLM_GOOD_LOGPROB,
    bad: float = PLM_BAD_LOGPROB,
) -> tuple[float, float]:
    """把逐位置 log 概率压成 `(归一化分数, 原始平均 log 概率)`。

    归一化分数用于奖励；原始平均 log 概率单位 nat/残基，可跨实验比较并写入审计报告。
    归一化用线性截断而非 sigmoid，以保留区间内的线性信息。
    """
    if not logprobs:
        raise ValueError("logprobs 为空，无法打分")
    mean_lp = sum(logprobs) / len(logprobs)
    if not math.isfinite(mean_lp):
        raise ValueError(f"平均 log 概率不是有限值: {mean_lp}")

    span = good - bad
    if span <= 0:
        raise ValueError(f"归一化区间必须满足 good > bad，收到 good={good} bad={bad}")
    normalized = (mean_lp - bad) / span
    return max(0.0, min(1.0, normalized)), mean_lp


def logprob_lookup_row(tokenizer, sequence: str) -> list[int]:
    """返回真实氨基酸在每个位置对应的 token id。

    ESM tokenizer 按单氨基酸分词，第 i 个字符对应第 i+1 个 token；
    前提不成立时回填会整体错位，因此显式校验长度与词表。
    """
    ids = tokenizer.convert_tokens_to_ids(list(sequence))
    if len(ids) != len(sequence):
        raise ValueError(
            f"ESM tokenizer 没有按单氨基酸分词：序列长度 {len(sequence)}，"
            f"但 token 数是 {len(ids)}。本模块的逐位置回填依赖这个前提。"
        )
    if any(i is None or i < 0 for i in ids):
        bad = [c for c, i in zip(sequence, ids, strict=True) if i is None or i < 0]
        raise ValueError(f"序列里有 ESM 词表外的字符: {sorted(set(bad))}")
    return ids


# ---------------------------------------------------------------------------
# 预测器
# ---------------------------------------------------------------------------
class ESMPseudoLikelihoodPredictor(BasePredictor):
    """冻结的 ESM-2 掩码边缘概率，零样本，不需要标定数据。

    `is_calibrated=False` 是因为 `good`/`bad` 归一化区间为启发式，
    聚合层仍会对其施加未标定折扣。
    """

    name = "plm_esm"
    is_calibrated = False

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_ESM_MODEL,
        max_length: int = 512,
        good_logprob: float = PLM_GOOD_LOGPROB,
        bad_logprob: float = PLM_BAD_LOGPROB,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self.max_length = max_length
        self.good_logprob = good_logprob
        self.bad_logprob = bad_logprob
        self._model = None
        self._tokenizer = None
        self._load_error: str | None = None

    # -- 惰性加载 ----------------------------------------------------------
    def _load(self) -> bool:
        """加载 ESM 权重，失败时返回 False 而不抛异常。

        "配置开启但权重不可用"属于运行环境问题，降级为 confidence=0 即可，
        不应让训练中断。
        """
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad = False
            logger.info("ESM 已加载：%s（冻结，仅前向）", self.model_name)
            return True
        except Exception as exc:  # noqa: BLE001 - 缺依赖/没权重/没网都走降级
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "ESM 预测器不可用（%s），第五维将以 confidence=0 返回、"
                "不影响总奖励。装依赖/下权重后重跑即可启用。",
                self._load_error,
            )
            return False

    # -- 打分 --------------------------------------------------------------
    def _predict_one(self, sequence: str) -> PredictionResult:
        """逐位置掩码、批量前向，回填真实氨基酸的 log 概率。

        本预测器没有 `predict_with_reference`：参照处理在 scorer 层完成。
        """
        seq = standard_only(sequence)
        if not seq:
            return PredictionResult(
                value=0.5, confidence=0.0,
                detail={"error": "序列里没有标准氨基酸"},
                predictor=self.name,
            )
        if len(seq) > self.max_length:
            return PredictionResult(
                value=0.5, confidence=0.0,
                detail={"error": f"序列长度 {len(seq)} 超过 max_length={self.max_length}"},
                predictor=self.name,
            )

        if not self._load():
            # 降级时置信度必须为 0：0 置信度使有效权重为 0，不参与总奖励，
            # 也不能用 0.5 冒充中性分。
            return PredictionResult(
                value=0.5, confidence=0.0,
                detail={"error": f"ESM 未加载: {self._load_error}"},
                predictor=self.name,
            )

        import torch

        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None

        variants = masked_variants(seq)
        # 将 L 条等长变体叠成一个 batch，是在 CPU 上运行的关键。
        encoded = tokenizer(variants, return_tensors="pt", padding=False)
        target_ids = logprob_lookup_row(tokenizer, seq)

        with torch.no_grad():
            logits = model(**encoded).logits  # (L, L+2, vocab)

        # ESM 首尾为特殊 token，残基 i 对应位置 i+1。
        offset = 1
        pos = min(offset, logits.shape[1] - 1)
        logprobs: list[float] = []
        for i, target in enumerate(target_ids):
            row = logits[i, pos + i, :]
            lp = torch.log_softmax(row, dim=-1)[target]
            logprobs.append(float(lp.item()))

        value, mean_lp = marginal_logprobs_to_score(
            logprobs, good=self.good_logprob, bad=self.bad_logprob
        )
        return PredictionResult(
            value=value,
            # 归一化区间未标定，给中等偏低置信度；聚合层还会再乘未标定折扣。
            confidence=0.6,
            detail={
                "mean_logprob": round(mean_lp, 4),   # 有绝对意义的原始值
                "length": len(seq),
                "esm_model": self.model_name,
            },
            predictor=self.name,
        )

    # -- 缓存 --------------------------------------------------------------
    def _cache_key(self, sequence: str) -> str:
        """缓存键必须包含所有会改变输出的参数：模型名、归一化区间和序列。"""
        return (
            f"{self.name}|{self.model_name}|{self.good_logprob}|{self.bad_logprob}"
            f"|{standard_only(sequence)}"
        )
