"""构造 GRPO 校准集与预热样本。

用规则引擎生成候选改造，再用与奖励函数相同的口径复算 delta，只保留
目标维度确实改善、其余维度不劣化的 (base, improved) 对。产物是
"规则引擎 + 客观验证"，不是人工专家标注。
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from pepdesign.bio.cleavage import stability_score, suggest_stabilization
from pepdesign.bio.sequence import compute_properties, net_charge, standard_only
from pepdesign.data.schema import PeptideRecord
from pepdesign.prompts.templates import stabilize_prompt
from pepdesign.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CalibrationPair:
    """一条校准样本。`delta` 为各维度的实际改善量，用于抽查与加权采样。"""

    prompt: str
    base_sequence: str
    improved_sequence: str
    strategy: str
    dimension: str
    delta: dict[str, float]
    source_id: str = ""

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "base_sequence": self.base_sequence,
            "improved_sequence": self.improved_sequence,
            "strategy": self.strategy,
            "dimension": self.dimension,
            "delta": self.delta,
            "source_id": self.source_id,
        }


# ---------------------------------------------------------------------------
# 改造算子：把规则引擎的建议真正落到序列上
# ---------------------------------------------------------------------------
def apply_pro_sweep(seq: str) -> tuple[str, str] | None:
    """把酶切位点的 P1' 位换成 Pro 以消除该位点，一次只改一处以保证可归因。"""
    suggestions = suggest_stabilization(seq, max_suggestions=1)
    if not suggestions:
        return None
    target = suggestions[0]

    # 只处理"某位换成 Pro"的建议；N/C 端封闭建议的 position 是首尾下标，
    # 按 P1' 替换会得到无意义序列。
    if target.get("to") != "P":
        return None

    # position 是 P1 的 1-based 下标，而 P1' 的 0-based 下标恰好就是它，
    # 不要减 1。
    pos = int(target["position"])
    if pos < 0 or pos >= len(seq) or seq[pos] == "P":
        return None
    new_seq = seq[:pos] + "P" + seq[pos + 1 :]
    if new_seq == seq:
        return None
    return new_seq, f"P1'位 {seq[pos]}->P（阻断 {target['reason'][:20]}）"


def apply_terminal_cap(seq: str, terminal: str = "n") -> tuple[str, str] | None:
    """末端封闭改造：不改变序列字符串，只新增修饰标注。

    调用方需自行记录 modifications；该算子不能作为仅含氨基酸的预热样本。
    """
    if terminal == "n":
        if seq[0] in "DE":   # N 端带负电时乙酰化会削弱静电作用
            return None
        return seq, "N 端乙酰化（屏蔽氨肽酶/DPP4 切口）"
    if seq[-1] in "KR":      # C 端带正电时酰胺化收益低
        return None
    return seq, "C 端酰胺化（屏蔽羧肽酶切口）"


def apply_hydrophobic_dilution(seq: str) -> tuple[str, str] | None:
    """稀释连续疏水片段以降低聚集倾向。

    取最长连续疏水片段，把中间的一个残基换成极性残基；两端可能参与结合。
    """
    from pepdesign.bio.sequence import max_hydrophobic_run

    if max_hydrophobic_run(seq) < 5:
        return None

    best: tuple[int, int] | None = None
    cur_start = cur_len = 0
    for i, c in enumerate(seq + "X"):
        if i < len(seq) and c in "AVILMFWY":
            if cur_len == 0:
                cur_start = i
            cur_len += 1
        else:
            if cur_len >= 5 and (best is None or cur_len > best[1]):
                best = (cur_start, cur_len)
            cur_len = 0
    if best is None:
        return None

    start, length = best
    pos = start + length // 2
    if seq[pos] not in "AVILMFWY":
        return None
    new_seq = seq[:pos] + "T" + seq[pos + 1 :]
    return new_seq, f"连续疏水片段稀释（位置 {pos + 1} {seq[pos]}->T）"


def apply_helix_stabilization(seq: str) -> tuple[str, str] | None:
    """把埋在螺旋中的 Pro/Gly 换成 Ala，提升热稳定性。

    一次只改一处，保证性质改善可归因。
    """
    from pepdesign.bio.aa import HELIX_PROPENSITY

    if len(seq) < 8:
        return None

    # 只考虑埋在螺旋里的 Pro/Gly：末端或转角处的替换不会改善稳定性。
    candidates = []
    for i in range(1, len(seq) - 1):
        if seq[i] not in "PG":
            continue
        neighbours = HELIX_PROPENSITY[seq[i - 1]] + HELIX_PROPENSITY[seq[i + 1]]
        if neighbours >= 1.9:          # 两侧都倾向螺旋（均值约 0.95×2）
            candidates.append((neighbours, i))
    if not candidates:
        return None

    # 两侧螺旋倾向最高的那处，收益最大
    _, pos = max(candidates)
    new_seq = seq[:pos] + "A" + seq[pos + 1:]
    if new_seq == seq:
        return None
    return new_seq, f"螺旋中断位 {seq[pos]}->A（位置 {pos + 1}，提升热稳定性）"


STRATEGY_FUNCS = {
    "pro_sweep": apply_pro_sweep,
    "hydrophobic_dilution": apply_hydrophobic_dilution,
    "helix_stabilization": apply_helix_stabilization,
}

#: 末端封闭算子（N 端乙酰化 / C 端酰胺化）。它们不改变序列字符串，
#: 而预热样本的输出格式只接受标准氨基酸，故不放进 STRATEGY_FUNCS。
TERMINAL_MOD_FUNCS = {
    "n_terminal_cap": lambda s: apply_terminal_cap(s, "n"),
    "c_terminal_cap": lambda s: apply_terminal_cap(s, "c"),
}

#: 策略 → `evaluate_delta` 返回的验证维度键。键名必须与返回值一致，
#: 否则 `delta.get(key, 0.0)` 恒为 0，全部样本会被静默丢掉。
#: `aggregation` 方向相反，另有饱和预检。
_STRATEGY_TARGET_DIM: dict[str, str] = {
    "pro_sweep": "protease",
    "n_terminal_cap": "protease",
    "c_terminal_cap": "protease",
    "hydrophobic_dilution": "aggregation",
    "helix_stabilization": "thermo_proxy",
}

#: 策略 → 写入 prompt 的薄弱维度描述，须与 `_STRATEGY_TARGET_DIM` 成对维护。
_STRATEGY_WEAK_DIMENSION: dict[str, str] = {
    "pro_sweep": "酶解稳定性",
    "n_terminal_cap": "酶解稳定性",
    "c_terminal_cap": "酶解稳定性",
    "hydrophobic_dilution": "成药性（溶解度与聚集）",
    "helix_stabilization": "热稳定性",
}


# ---------------------------------------------------------------------------
# 改造的客观验证
# ---------------------------------------------------------------------------
def evaluate_delta(base: str, improved: str) -> dict[str, float]:
    """量化改造前后的四维变化，正数表示改善。

    必须与奖励函数使用同一套 bio/ 函数，否则校准样本在 RL 中的得分
    可能与预热方向相反。
    """
    from pepdesign.bio.cleavage import stability_score
    from pepdesign.rewards.predictors.tm import TmPredictor

    p_base = compute_properties(base)
    p_imp = compute_properties(improved)

    # 用奖励函数同款的 Tm 线性模型；aliphatic_index 与 Tm 不是同一把尺子。
    tm = TmPredictor()
    tm_base = tm.predict_one(base).value
    tm_improved = tm.predict_one(improved).value

    # 各维度统一约定为正数 = 改善；方向写反时 is_valid_improvement 会静默放行变差。
    return {
        "protease": round(stability_score(improved) - stability_score(base), 4),
        # 单位摄氏度，量纲与其余项不同；这里只比较符号。
        "thermo_proxy": round(tm_improved - tm_base, 4),
        "amphipathic": round(p_imp.amphipathic_face - p_base.amphipathic_face, 4),
        # 聚集倾向下降才是改善，所以是 base - improved
        "aggregation": round(
            p_base.aggregation_propensity - p_imp.aggregation_propensity, 4
        ),
        # 电荷偏离 0 越少越温和才是改善
        "charge_shift": round(
            abs(net_charge(base)) - abs(net_charge(improved)), 4
        ),
    }


def is_valid_improvement(
    delta: dict[str, float],
    *,
    target_dim: str,
    tolerance: float = 0.02,
) -> bool:
    """判断改造是否有效：目标维度改善超过 tolerance，且其他维度下降不超过 2×tolerance。"""
    if delta.get(target_dim, 0.0) <= tolerance:
        return False
    for key, value in delta.items():
        if key == target_dim:
            continue
        if value < -tolerance * 2:
            return False
    return True


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build_calibration_pairs(
    records: Sequence[PeptideRecord],
    *,
    target_count: int = 500,
    seed: int = 42,
) -> list[CalibrationPair]:
    """从清洗后的记录里构造校准集，优先选择有实验数据的记录。

    产出维度分布不均衡，取决于各改造算子在语料上的适用面，不保证均衡覆盖。
    """
    rng = random.Random(seed)
    pairs: list[CalibrationPair] = []
    #: 已产出的对数按验证维度计数，用于下面的配额排序。
    #: 键是 `_STRATEGY_TARGET_DIM` 的值（protease / aggregation / thermo_proxy）。
    dim_counts: Counter = Counter()
    reject_stats: dict[str, int] = {}

    # 排序优先级：目标维度仍有改善空间的序列优先（评分饱和后 delta 恒为 0），
    # 其次有实验数据，最后长度接近 16。
    pool = sorted(
        records,
        key=lambda r: (
            -int(stability_score(r.sequence) > 0.05),
            -int(r.has_experimental_data),
            abs(r.length - 16),
        ),
    )

    # 策略名固定顺序，起点按记录下标轮转：每条记录只产出一对，
    # 起点顺序直接决定各策略的名额，随机打乱会让适用面广的策略垄断产出。
    strategy_names = list(STRATEGY_FUNCS)

    for rec_index, rec in enumerate(pool):
        if len(pairs) >= target_count:
            break
        base = standard_only(rec.sequence)
        if len(base) < 8:
            continue

        # 起点随记录轮转，后续策略顺序打乱，避免排在后面的永远轮不到。
        offset = rec_index % len(strategy_names)
        rotated = strategy_names[offset:] + strategy_names[:offset]
        tail = rotated[1:]
        rng.shuffle(tail)
        strategies = [(n, STRATEGY_FUNCS[n]) for n in (rotated[0], *tail)]

        # 按"该维度是否已超额"做软排序，让欠配额的维度优先被尝试。
        # 这只是软偏好，不改变策略适用面差异；硬配额会让总数掉到目标以下。
        quota = max(1, target_count // len(strategy_names))
        strategies.sort(key=lambda nf: dim_counts[_STRATEGY_TARGET_DIM[nf[0]]] >= quota)

        for strat_name, func in strategies:
            # 目标维度已无测量空间时跳过，与"改造失败"分开统计：
            # 前者需要换样本或改评分范围，后者需要改算子。
            if strat_name in ("pro_sweep", "n_terminal_cap", "c_terminal_cap"):
                if stability_score(base) <= 0.05:
                    reject_stats["saturated:protease"] = (
                        reject_stats.get("saturated:protease", 0) + 1
                    )
                    continue
            elif (
                strat_name == "hydrophobic_dilution"
                and compute_properties(base).aggregation_propensity < 0.15
            ):
                reject_stats["saturated:aggregation"] = (
                    reject_stats.get("saturated:aggregation", 0) + 1
                )
                continue

            result = func(base)
            if result is None:
                reject_stats[f"{strat_name}:inapplicable"] = (
                    reject_stats.get(f"{strat_name}:inapplicable", 0) + 1
                )
                continue

            improved_seq, description = result

            # 序列未变说明算子不改变序列（如末端修饰），无法用于预热样本。
            # 保留此分支，以便今后误加同类算子时立即暴露。
            if improved_seq == base:
                reject_stats[f"{strat_name}:no_sequence_change"] = (
                    reject_stats.get(f"{strat_name}:no_sequence_change", 0) + 1
                )
                logger.warning(
                    "策略 %s 没有改变序列，已跳过。校准集的输出格式"
                    "（### 设计序列）表达不了修饰类改造，这类算子不能进 STRATEGY_FUNCS。",
                    strat_name,
                )
                continue

            delta = evaluate_delta(base, improved_seq)
            # 目标维度由策略决定，而不是事后从 delta 里挑最大值，否则验证变成自证。
            target_dim = _STRATEGY_TARGET_DIM[strat_name]

            if not is_valid_improvement(delta, target_dim=target_dim):
                reject_stats[f"{strat_name}:invalid"] = (
                    reject_stats.get(f"{strat_name}:invalid", 0) + 1
                )
                continue

            spec = stabilize_prompt(
                sequence=base,
                target_name=rec.target or "目标结合位点",
                weak_dimension=_STRATEGY_WEAK_DIMENSION[strat_name],
                preserve="靶点亲和力",
            )
            pairs.append(
                CalibrationPair(
                    prompt=spec.user,
                    base_sequence=base,
                    improved_sequence=improved_seq,
                    strategy=description,
                    dimension=target_dim,
                    delta=delta,
                    source_id=rec.source_id,
                )
            )
            dim_counts[target_dim] += 1
            break  # 这条记录已经产出一对，换下一条

    logger.info(
        "校准集构造完成：%d 对（目标 %d），拒绝统计: %s",
        len(pairs), target_count,
        dict(sorted(reject_stats.items(), key=lambda kv: -kv[1])[:6]),
    )
    if len(pairs) < target_count:
        logger.warning(
            "只构造出 %d 对，少于目标 %d。可能原因：语料规模不足、"
            "或序列本身已经足够好（没有可改造的缺陷）。"
            "这是正常的——不要为了凑数而放宽验证标准。",
            len(pairs), target_count,
        )
    return pairs


def calibration_to_warmup_samples(pairs: Sequence[CalibrationPair]) -> list[dict]:
    """把校准对转成预热阶段的训练样本（SFT 形式，prompt -> improved_sequence）。

    序列未变的配对会被跳过：`### 设计序列` 只能表达标准氨基酸，修饰类改造
    在该格式下等价于原样输出。配对本身仍保留在 calibration_pairs.jsonl。
    """
    from pepdesign.prompts.templates import SYSTEM_PROMPT, render_completion

    samples = []
    skipped = 0
    for pair in pairs:
        if pair.base_sequence == pair.improved_sequence:
            # 修饰类改造无法在该输出格式中表达。
            skipped += 1
            continue
        samples.append(
            {
                "messages": [
                    # system 段必须与 SFT / GRPO / 推理逐字一致，统一取 templates.SYSTEM_PROMPT。
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": pair.prompt},
                    {
                        "role": "assistant",
                        "content": render_completion(
                            f"改造策略：{pair.strategy}",
                            pair.improved_sequence,
                        ),
                    },
                ],
                "task_type": "calibration",
                "meta": {"strategy": pair.strategy, "dimension": pair.dimension},
            }
        )

    if skipped:
        logger.info(
            "校准预热：%d 对中跳过 %d 对（序列未变，修饰改动无法在"
            "「### 设计序列」里表达），实际产出 %d 条预热样本。",
            len(pairs), skipped, len(samples),
        )
    if not samples:
        logger.warning(
            "校准预热样本为 0 条——所有配对的序列都没变化。"
            "GRPO 会直接跳过预热开始训练（见 train/calibration.py）。"
        )
    return samples
