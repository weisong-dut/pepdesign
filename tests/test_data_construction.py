"""SFT / GRPO / 校准三套语料构造器的测试。

按不变量而非当前输出编写，覆盖三类不会报错、只能靠抽查发现的问题：
空转样本（输出 = 输入）、任务权重与劣化策略不一致、以及预热 system prompt
与其他阶段不一致。
"""

from __future__ import annotations

import random

from pepdesign.bio.sequence import mean_propensity
from pepdesign.data.build_calibration import (
    STRATEGY_FUNCS,
    TERMINAL_MOD_FUNCS,
    CalibrationPair,
    calibration_to_warmup_samples,
)
from pepdesign.data.build_grpo import build_grpo_dataset
from pepdesign.data.build_sft import (
    DEGRADE_STRATEGIES,
    DIMENSION_TO_STRATEGY,
    MULTI_OBJECTIVE_PRESETS,
    STRATEGY_TO_DIMENSION,
    build_multi_objective_sample,
    degrade_sequence,
    strategy_for_dimension,
)
from pepdesign.data.schema import PeptideRecord
from pepdesign.prompts.templates import SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# 定向劣化：这是整个 SFT/GRPO 负样本的来源，之前没有任何测试
# ---------------------------------------------------------------------------
class TestTargetedDegradation:
    """`degrade_sequence(strategy=...)` 必须只走指定的那条策略。"""

    #: 一条同时具备多种"可劣化位点"的序列：
    #:   有 KP（被 Pro 保护的酶切位点）、有连续疏水段、有多处 KR、有埋在螺旋里的 G
    RICH = "KKPLLLAAAVVVKKPFFGGEEKK"

    def test_named_strategy_is_actually_used(self):
        """指定策略后返回的策略名必须就是它。"""
        for name in DEGRADE_STRATEGIES:
            result = degrade_sequence(self.RICH, random.Random(0), strategy=name)
            if result is None:
                continue
            assert result[1] == name, f"要求 {name}，实际用了 {result[1]}"

    def test_inapplicable_strategy_returns_none_instead_of_falling_back(self):
        """指定的策略不可用时返回 None，不换成其他策略。"""
        # 全 Pro/Gly，没有任何 KR 或疏水段，add_protease_site 无从下手
        assert degrade_sequence("PGPGPGPGPG", random.Random(0),
                                strategy="add_protease_site") is None

    def test_unknown_strategy_returns_none(self):
        assert degrade_sequence(self.RICH, random.Random(0),
                                strategy="no_such_strategy") is None

    def test_mismatch_cyclic_is_not_offered_for_linear_peptides(self):
        """线性肽没有环化约束可破坏，这条策略必须被摘掉。"""
        assert degrade_sequence(self.RICH, random.Random(0),
                                strategy="mismatch_cyclic") is None


class TestHelixBreaker:
    """热稳定性劣化算子。

    它是唯一需要验证"性质确实变化"的算子：换成 Pro 不必然破坏螺旋。
    """

    def test_lowers_helix_propensity(self):
        seq = "KKAALAAALAAALAAAKK"
        result = degrade_sequence(seq, random.Random(0), strategy="helix_breaker")
        assert result is not None
        degraded, name = result
        assert name == "helix_breaker"
        assert degraded != seq
        assert mean_propensity(degraded, "helix") < mean_propensity(seq, "helix")

    def test_lowers_predicted_tm(self):
        """插 Pro 同时提高 `pro_gly_ratio` 与降低 `helix_propensity`，预测 Tm 必须下降。"""
        from pepdesign.rewards.predictors.tm import TmPredictor

        seq = "KKAALAAALAAALAAAKK"
        degraded = degrade_sequence(seq, random.Random(0), strategy="helix_breaker")[0]
        tm = TmPredictor()
        assert tm.predict_one(degraded).value < tm.predict_one(seq).value

    def test_returns_none_when_nothing_is_helix_favouring(self):
        """窗口中没有螺旋促进者时返回 None，不强行改位。"""
        assert degrade_sequence("GGGGGGGGGGGG", random.Random(0),
                                strategy="helix_breaker") is None


class TestStrategyDimensionContract:
    """策略 ↔ 维度两张表必须互为逆映射（允许一对多别名，不允许悬空）。"""

    def test_every_strategy_declares_a_dimension(self):
        """每条策略都要声明目标维度，否则 prompt 查表会 KeyError。"""
        for name in DEGRADE_STRATEGIES:
            assert name in STRATEGY_TO_DIMENSION, f"{name} 没有声明目标维度"

    def test_every_dimension_maps_back_to_a_real_strategy(self):
        """反向表中的每个值都必须是真实策略，且与正向表自洽。"""
        for dim, strat in DIMENSION_TO_STRATEGY.items():
            assert strat in DEGRADE_STRATEGIES, f"{dim} → {strat} 不是合法策略"
            assert strat == strategy_for_dimension(dim)

    def test_unknown_dimension_returns_none_not_keyerror(self):
        """未知维度返回 None；多目标权重可能指向尚无算子的维度。"""
        assert strategy_for_dimension("不存在的维度") is None

    def test_every_multi_objective_preset_is_reachable(self):
        """每套权重预设至少要有一维能找到算子，否则这条预设产不出任何样本。"""
        for preset in MULTI_OBJECTIVE_PRESETS:
            assert any(strategy_for_dimension(d) for d in preset), preset


# ---------------------------------------------------------------------------
# 多目标任务：prompt 上写的权重要和实际造出来的缺陷对得上
# ---------------------------------------------------------------------------
class TestMultiObjectiveAlignment:
    """prompt 中写的权重必须与实际劣化出的缺陷一致。"""

    def _record(self, seq: str) -> PeptideRecord:
        return PeptideRecord(sequence=seq, source="uniprot", name="test")

    def test_degraded_dimension_is_always_one_of_the_stated_objectives(self):
        """劣化出的缺陷维度必须落在 prompt 列出的目标中。"""
        rec = self._record("KKPLLLAAAVVVKKPFFGGEEKK")
        hits = 0
        for seed in range(40):
            built = build_multi_objective_sample(rec, random.Random(seed))
            if built is None:
                continue
            sample, strategy = built
            objectives = sample["meta"]["objectives"]
            degraded_dim = STRATEGY_TO_DIMENSION[strategy]
            assert degraded_dim in objectives, (
                f"劣化了 {degraded_dim}，但 prompt 只列了 {list(objectives)}"
            )
            hits += 1
        assert hits > 0, "40 个 seed 一个样本都没造出来，测试本身失效了"

    def test_highest_weighted_reachable_objective_wins(self):
        """最高权重且可劣化的维度应主导策略选择；无法劣化时才降级。"""
        rec = self._record("KKPLLLAAAVVVKKPFFGGEEKK")
        top = fallback = 0
        for seed in range(40):
            built = build_multi_objective_sample(rec, random.Random(seed))
            if built is None:
                continue
            _sample, strategy = built
            objectives = built[0]["meta"]["objectives"]
            ranked = [
                d for d, _ in sorted(objectives.items(), key=lambda kv: -kv[1])
                if strategy_for_dimension(d)
            ]
            if not ranked:
                continue
            if strategy == strategy_for_dimension(ranked[0]):
                top += 1
            else:
                fallback += 1
        assert top > fallback, (
            f"命中最高权重 {top} 次、降级 {fallback} 次——"
            f"权重没有真正主导策略选择，说明又回到了'随机挑一个劣化'"
        )


# ---------------------------------------------------------------------------
# GRPO：奖励泄漏的通道必须被物理切断
# ---------------------------------------------------------------------------
class TestGrpoMetaHasNoAnswer:
    """`reference_sequence` 不应出现在 GRPO meta 中，它会让奖励读到答案。"""

    def _small_corpus(self) -> list[PeptideRecord]:
        return [
            PeptideRecord(sequence="KKPLLLAAAVVVKKPFFGGEEKK", name="a"),
            PeptideRecord(sequence="AALKKAALKKPLLAAKKPLLAAKK", name="b"),
            PeptideRecord(sequence="GGPLLKKAAVVKKPFFGGEEKKLL", name="c"),
            PeptideRecord(sequence="LLKKPPAAGGLLKKPPAAGGLLKK", name="d"),
        ]

    def test_no_reference_sequence_anywhere_in_meta(self):
        """meta 中不得出现 reference_sequence，否则奖励会以答案为准。"""
        samples = build_grpo_dataset(self._small_corpus(), total=40, seed=0)
        assert samples, "一条都没构造出来，测试本身失效了"
        for s in samples:
            meta = s.get("meta") or {}
            assert "reference_sequence" not in meta, (
                f"meta 里又出现了 reference_sequence：{meta.get('task')}"
            )

    def test_base_sequence_is_present_for_optimization_tasks(self):
        """基准序列必须保留：它是亲和力维度的分母与相似度惩罚的依据。"""
        samples = build_grpo_dataset(self._small_corpus(), total=40, seed=0)
        optim = [s for s in samples if (s.get("meta") or {}).get("task") != "de_novo"]
        assert optim, "没有构造出任何优化类任务"
        assert all("base_sequence" in s["meta"] for s in optim)

    def test_prompts_never_contain_an_assistant_message(self):
        """prompt 里带 assistant 段等于把答案直接给出去。"""
        samples = build_grpo_dataset(self._small_corpus(), total=40, seed=0)
        for s in samples:
            assert "assistant" not in s["prompt"]


# ---------------------------------------------------------------------------
# 校准集
# ---------------------------------------------------------------------------
class TestCalibrationWarmup:
    def _pair(self, base: str, improved: str) -> CalibrationPair:
        return CalibrationPair(
            prompt=f"【优化任务】…【原始序列】{base}",
            base_sequence=base,
            improved_sequence=improved,
            strategy="测试策略",
            dimension="protease",
            delta={"protease": 0.1},
        )

    def test_system_prompt_matches_the_shared_template(self):
        """预热 system prompt 必须与 SFT / GRPO / 推理共用同一模板。"""
        samples = calibration_to_warmup_samples(
            [self._pair("KWKLFKKIEK", "KWKLFKKIEP")]
        )
        assert samples[0]["messages"][0]["content"] == SYSTEM_PROMPT

    def test_pairs_whose_sequence_does_not_change_are_skipped(self):
        """输出等于输入的预热样本必须跳过，否则教模型原样复述。

        末端修饰类改造不改变序列字符串，收益只在带修饰的口径下可见。
        """
        pairs = [
            self._pair("KWKLFKKIEK", "KWKLFKKIEK"),   # 序列没变 → 跳过
            self._pair("KWKLFKKIEK", "KWKLFKKIEP"),   # 真的改了 → 保留
        ]
        samples = calibration_to_warmup_samples(pairs)
        assert len(samples) == 1
        body = samples[0]["messages"][2]["content"]
        assert "KWKLFKKIEP" in body and "KWKLFKKIEK\n" not in body

    def test_terminal_modification_operators_are_not_in_the_pool(self):
        """末端修饰算子不在候选池中：它们产不出可用的预热样本。"""
        assert "n_terminal_cap" not in STRATEGY_FUNCS
        assert "c_terminal_cap" not in STRATEGY_FUNCS
        # 但函数本身要保留（它们描述的是真实的改造手法，格式允许时应重新启用）
        assert set(TERMINAL_MOD_FUNCS) == {"n_terminal_cap", "c_terminal_cap"}

    def test_every_operator_in_the_pool_changes_the_sequence(self):
        """`STRATEGY_FUNCS` 中的算子必须改变序列。"""
        probes = [
            "KKPLLLAAAVVVKKPFFGGEEKK",
            "AAGGPLLKKAAVVKKPFFGGEEKK",
            "LLKKPPAAGGLLKKPPAAGGLLKK",
        ]
        fired = 0
        for name, func in STRATEGY_FUNCS.items():
            for seq in probes:
                result = func(seq)
                if result is None:
                    continue
                improved, _desc = result
                assert improved != seq, f"{name} 没有改变序列（{seq}）"
                fired += 1
        assert fired > 0, "一个算子都没触发，测试本身失效了"
