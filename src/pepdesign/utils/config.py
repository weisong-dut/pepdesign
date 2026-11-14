"""配置系统：YAML 转换为强类型 pydantic 对象。

强类型与 `extra="forbid"` 使拼写错误的字段在加载时立即报错，而非静默使用默认值。
配置与代码分离，支持默认配置 -> YAML -> 环境变量 / CLI 的覆盖链，
调参时只需写出差异项。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from pepdesign.utils.logger import PROJECT_ROOT, get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# 基类
# --------------------------------------------------------------------------
class StrictModel(BaseModel):
    """所有配置的基类：禁止未声明字段，赋错值立刻报错。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------
# 1. LoRA 配置
# --------------------------------------------------------------------------
#: Llama 系（含 LOGOS-3B）常用的线性层名。
#: 不带模型名前缀，便于更换基座；该列表同时被默认配置与参数量统计引用。
LINEAR_MODULES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",        # 注意力
    "gate_proj", "up_proj", "down_proj",            # MLP
)


class LoRAConfig(StrictModel):
    """LoRA 低秩适配超参。

    默认配置为 LOGOS-3B + r=16 + 全部线性层，可训练参数约 0.76%
    （逐项计算见 `models/lora.py:lora_trainable_summary`）。

    `r` 为低秩分解的秩，`lora_alpha` 的生效缩放为 alpha/r，`lora_dropout`
    作用于 LoRA 分支输入；`target_modules` 按模块名后缀匹配，
    `backend` 可选 manual（手写实现）或 peft（peft 库实现）。
    """

    r: int = Field(16, ge=1, le=256)
    lora_alpha: int = Field(32, ge=1)
    lora_dropout: float = Field(0.05, ge=0.0, le=0.9)
    target_modules: list[str] = Field(
        default_factory=lambda: list(LINEAR_MODULES)
    )
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"
    backend: Literal["manual", "peft"] = "manual"

    @field_validator("target_modules")
    @classmethod
    def _check_modules(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("target_modules 不能为空，否则 LoRA 没有任何可训练参数")
        return v

    @property
    def scaling(self) -> float:
        """alpha / r，实际作用于 LoRA 分支的缩放系数。"""
        return self.lora_alpha / self.r

    def resolved_target_modules(self) -> list[str]:
        """最终要注入的模块名列表（去重，保持声明顺序）。

        去重避免参数量统计把同一层计算两次。
        """
        seen: dict[str, None] = {}
        for m in self.target_modules:
            seen.setdefault(m, None)
        return list(seen)


# --------------------------------------------------------------------------
# 2. 量化配置
# --------------------------------------------------------------------------
class QuantConfig(StrictModel):
    """量化配置。训练侧对应 QLoRA 的 4bit 基座，推理侧对应 4-bit NF4。

    使用 bitsandbytes NF4（4-bit NormalFloat）而非普通 INT4：
    预训练权重近似零均值正态分布，NF4 对该分布的量化误差更小。
    """

    enabled: bool = True
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    # 计算用 dtype：反量化后矩阵乘的精度。bf16 动态范围大，不易溢出成 NaN。
    bnb_4bit_compute_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    # 双重量化：对量化常数（scale）再量化一次，额外节省约 0.37 bit/参数。
    bnb_4bit_use_double_quant: bool = True


# --------------------------------------------------------------------------
# 3. 训练公共配置
# --------------------------------------------------------------------------
class TrainCommonConfig(StrictModel):
    """SFT 与 GRPO 共用的工程化训练策略配置。"""

    # 全项目唯一的基础模型默认值；改动时需同步 configs、报告等其他引用。
    model_name_or_path: str = "LOGOS-Hub/LOGOS-pretrain-3B"
    output_dir: str = "outputs/sft"
    seed: int = 42
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 4
    # 梯度累积：等效 batch = per_device_batch × grad_accum × 卡数。
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    # LoRA 微调的是增量 ΔW = B·A，权重衰减的语义不同于全量微调，参考实现通常取 0。
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: Literal[
        "linear", "cosine", "cosine_with_restarts", "constant", "constant_with_warmup"
    ] = "cosine"
    # warmup：前 warmup_ratio 比例的训练步将 lr 从 0 线性升至目标值；
    # LoRA 参数为随机初始化，warmup 用于稳定早期梯度。
    warmup_ratio: float = 0.03
    # 混合精度：bf16 训练。相比 fp16 不需要 loss scaling，稳定性更好。
    bf16: bool = True
    fp16: bool = False
    # 梯度检查点：用重算换显存，决定单卡能否容纳 3B 模型。
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    # 每多少步保存 checkpoint；<= 0 表示从不保存。需小于总步数才有意义。
    save_steps: int = 200
    # 只保留最近 N 个 checkpoint。优化器状态与 adapter 占用较大，不设上限会写满磁盘。
    save_total_limit: int = Field(3, ge=1)
    # 指标外送后端，合法取值见 `utils/tracking.REPORT_BACKENDS`。
    report_to: list[str] = Field(default_factory=lambda: ["tensorboard"])
    # 断点续训：None/空串不续训，"latest" 取最新完整 checkpoint，或指定路径。
    # 续训包含优化器与调度器状态。
    resume_from_checkpoint: str | None = None
    # 只跑多少步就停；-1 表示按 num_train_epochs 跑满。
    max_steps: int = -1

    @field_validator("report_to")
    @classmethod
    def _check_report_backends(cls, v: list[str]) -> list[str]:
        """在加载配置时拒绝不认识的 report_to 后端名。

        校验逻辑位于 `utils/tracking.resolve_backends`，此处接入 pydantic 校验链。
        """
        from pepdesign.utils.tracking import resolve_backends

        return list(resolve_backends(v))


class SFTConfig(TrainCommonConfig):
    """阶段一：监督微调配置。

    不包含 `packing` 与 `assistant_only_loss`：当前实现未支持这两个字段，
    替代方案见 `train/sft.py` 的模块文档。
    """

    output_dir: str = "outputs/sft"
    dataset_path: str = "data/processed/sft_train.jsonl"
    eval_dataset_path: str | None = "data/processed/sft_valid.jsonl"
    # `eval_steps` / `per_device_eval_batch_size` 仅属于 SFT；GRPO 没有验证集，
    # 因此不放在公共基类上。
    eval_steps: int = 200
    per_device_eval_batch_size: int = 4
    # 样本最大 token 数。配小不会报错，只会静默丢弃超长样本。
    max_seq_length: int = 2048
    # 只对答案部分（真实序列）计算 loss，指令部分掩码为 ignore_index。
    completion_only_loss: bool = True
    # 若填写则覆盖 learning_rate；取值的唯一入口是 `resolved_lr()`。
    lr: float | None = None

    def resolved_lr(self) -> float:
        """最终生效的学习率：`lr` 非空时它说了算，否则用 `learning_rate`。"""
        return self.lr if self.lr is not None else self.learning_rate


class GRPOConfig(TrainCommonConfig):
    """阶段二：GRPO 强化学习配置。"""

    output_dir: str = "outputs/grpo"
    dataset_path: str = "data/processed/grpo_prompts.jsonl"
    # 从 SFT 产出的 LoRA 继续训练，GRPO 的起点是 SFT 收敛后的策略。
    sft_adapter_path: str = "outputs/sft/final"
    # prompt 的 token 上限，超出部分由 `masking.truncate_prompt` 从中间截断
    # （保留开头 system prompt 与结尾约束）。
    max_prompt_length: int = 512
    max_completion_length: int = 256
    learning_rate: float = 1e-5      # RL 阶段 lr 要比 SFT 小一个量级
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 8

    # ---- GRPO 专有超参 ----
    # 组大小 G：同一 prompt 采样 G 条序列并做组内相对比较，
    # 这是 GRPO 不需要 critic 网络的关键。
    num_generations: int = Field(8, ge=2)
    # RL 阶段需要探索；温度过低会使组内序列趋同、优势为 0。
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    # KL 惩罚系数：约束新策略不偏离 SFT 参考策略过远。
    beta: float = 0.04
    loss_type: Literal["grpo", "dr_grpo", "bnpo", "dapo"] = "dr_grpo"
    # 重要性比裁剪区间；`loss_type="dapo"` 时上界改用 1 + clip_higher。
    epsilon: float = 0.2
    # DAPO 的上界放宽系数：只放宽上界以缓解熵塌缩。
    clip_higher: float = 0.28
    scale_rewards: bool = True
    # 组内优势标准化方式。`scale_rewards=False` 时会固定为 "none"。
    advantage_normalization: Literal["std", "mean", "none"] = "std"

    # ---- 校准集预热 ----
    # 正式 RL 前用专家校准对做若干步训练，将策略初始分布拉向专家合理区域，
    # 减少早期盲目探索。
    calibration_path: str = "data/calibration/calibration_pairs.jsonl"
    calibration_enabled: bool = True
    calibration_epochs: float = 1.0
    calibration_lr_scale: float = 0.5   # 预热阶段 lr 减半，避免过拟合校准集
    # 预热完成后是否丢弃校准集的梯度影响（把 optimizer state 清零）
    reset_optimizer_after_calibration: bool = False


# --------------------------------------------------------------------------
# 4. 奖励配置
# --------------------------------------------------------------------------
class RewardWeights(StrictModel):
    """奖励各维度的权重。

    前四维为训练奖励：热稳定性 / 酶解稳定性 / 靶点亲和力 / 结构合理性。
    `plm`（ESM-2 掩码边缘概率）默认权重 0，即只记录用于独立审计、不参与优化；
    设为非零后作为第五个训练维度参与优化。
    """

    thermo: float = 0.30       # 热稳定性 Tm
    protease: float = 0.25     # 酶解稳定性
    affinity: float = 0.30     # 靶点亲和力
    structure: float = 0.15    # 结构合理性
    # ESM-2 掩码边缘概率分数。默认 0 表示只审计、不优化。
    plm: float = 0.0

    def normalized(self) -> RewardWeights:
        """归一化到和为 1。

        GRPO 对奖励缩放敏感，归一化使更换权重不改变有效学习率。
        实现遍历 `model_dump()`，新增维度自动纳入。
        """
        values = self.model_dump()
        total = sum(values.values())
        if total <= 0:
            raise ValueError(
                f"奖励权重之和必须为正，当前是 {total}：{values}。"
                f"（全零权重会让总奖励恒为 0，GRPO 拿不到任何优势信号。）"
            )
        return RewardWeights(**{k: v / total for k, v in values.items()})


# 场景预设：研发人员在前端下拉框里选"优化目标"，后端映射到这里的权重组合。
# 这是"动态权重调节"落地时的具体形态——比让用户手填五个数字友好得多，
# 也保证了参数的合法范围可控。
#
# 每个预设都必须写全所有维度（含 `plm`）。少写一个不会报错，
# 但 `from_preset` 的"只覆盖显式给出的维度"合并逻辑会让它退回类默认值——
# 而预设表是"最终生效值"的唯一来源，缺项在这里是必须显式写出来的决定。
# `test_reward_presets_cover_every_dimension` 会盯着这一点。
#
# `plm: 0.0` 在四个预设里都是 0，这是上面那段"审计而非优化"的选择；
# 想开启"优化天然度"就在 YAML 里覆盖：`reward.weights.plm: 0.15`。
REWARD_PRESETS: dict[str, dict[str, float]] = {
    "balanced":        {"thermo": 0.30, "protease": 0.25, "affinity": 0.30,
                        "structure": 0.15, "plm": 0.0},
    "stability_first": {"thermo": 0.45, "protease": 0.35, "affinity": 0.15,
                        "structure": 0.05, "plm": 0.0},
    "activity_first":  {"thermo": 0.15, "protease": 0.15, "affinity": 0.55,
                        "structure": 0.15, "plm": 0.0},
    "developability":  {"thermo": 0.30, "protease": 0.30, "affinity": 0.15,
                        "structure": 0.25, "plm": 0.0},
}


class RewardConfig(StrictModel):
    """奖励函数全局配置。

    `preset` / `weights` 的生效规则（优先级从高到低）
    ------------------------------------------------
    1. 单条样本的 `meta["objectives"]`（仅 `task == "multi_objective"`）。
       这些数字被 `prompts.templates.multi_objective_prompt` 原样写进了 prompt，
       模型是照着它优化的，所以奖励必须用同一组值——这一条不可被配置覆盖。
    2. 本配置里显式写出来的 `preset` / `weights`。在 YAML 里写了才生效，
       "保持默认值"不算。
    3. `composite.TASK_PRESETS` 里该任务的预设（`stabilize` → 稳定性优先等），
       未知任务退回 `balanced`。

    第 2 条为什么要强调"显式"：`weights` 的类默认值恰好等于 `preset="balanced"`
    的权重（0.30/0.25/0.30/0.15），所以光看值分不出"用户想要 balanced"和
    "用户根本没提这一项"，而这两种情况的正确行为完全相反——前者应当覆盖任务预设，
    后者必须让 `TASK_PRESETS` 生效。用 `weights_from_config()` 做这个区分。
    """

    preset: Literal["balanced", "stability_first", "activity_first", "developability"] = "balanced"
    weights: RewardWeights = Field(default_factory=RewardWeights)

    # 每个维度内部的"软约束"惩罚，避免模型钻单项漏洞。
    # 比如只优化 Tm 而不顾聚集，会得到一堆溶于不了水的序列。
    aggregation_penalty: float = 0.30      # 聚集倾向惩罚强度
    hydrophobic_run_penalty: float = 0.20  # 连续疏水片段惩罚强度
    charge_penalty: float = 0.15           # 净电荷过于极端（|q| > 6）的惩罚
    invalid_sequence_penalty: float = 0.5  # 生成非法序列的直接惩罚

    # 预测器后端：heuristic = 纯 CPU 启发式；esm = 用 ESM 嵌入 + 回归头；
    # tool = 外部命令行工具（如接入 PeptideCutter/NetMHCpan 等）
    #
    # thermo 的取值没有 "esm"：`predictors/tm_esm.py` 不存在，
    # 写在这里会让配置文件校验通过、运行时才炸。合法的后端名以
    # `predictors.THERMO_BACKENDS` 为准，两边必须同步。
    thermo_backend: Literal["heuristic", "tool"] = "heuristic"
    protease_backend: Literal["rule", "tool"] = "rule"
    affinity_backend: Literal["feature", "esm", "tool"] = "feature"
    structure_backend: Literal["propensity", "tool"] = "propensity"

    # 第五维：蛋白质语言模型（ESM-2）的独立分数。
    #
    # `"off"`（默认）= 不构造这一维，`CompositeScorer` 就只有四个维度，
    # 行为与加这一维之前完全一致（这点很重要：默认路径不能被新功能改变）。
    # `"esm"` = 用冻结的 ESM-2 算掩码边缘概率（pseudo-likelihood）。
    #
    # 为什么默认关：
    #   * 它要下载真实权重（150M 约 600MB / 650M 约 2.6GB），
    #     而纯 CPU 的单元测试和数据构建不该被这件事绑架；
    #   * 它的主要用途是独立审计（权重为 0，只记录不优化），
    #     审计是分析动作、不是训练动作，按需开启更合理。
    #
    # 这个字段和 `weights.plm` 是两个独立的开关：
    #   * backend="off"          → 维度根本不存在，weights.plm 无从生效；
    #   * backend="esm" + plm=0  → 维度存在、被计算并记录，但不参与加权求和；
    #   * backend="esm" + plm>0  → 真正参与训练。
    # 分成两个开关而不是一个，是因为"算了但不优化"（审计）和
    # "算了也优化"（训练）是两种真实存在的用法。
    plm_backend: Literal["off", "esm"] = "off"

    # 各维度的温度参数：把预测值通过 sigmoid/logistic 压到 (0,1) 的陡峭程度。
    #
    # 方向很容易记反：temperature 在分母上，所以
    #   * temperature 小 → 曲线在 center 附近更陡 → 趋近 0/1 硬阈值
    #     （组内一半 0 一半 1，梯度方向抖动，也丢掉了"好多少"的信息）；
    #   * temperature 大 → 曲线更平 → 所有序列得分几乎一样，
    #     组内优势接近 0，这一步训练等于白跑。
    # 两个方向都会损坏训练，默认值是在"多肽奖励"这个尺度上折中的结果。
    # （这一段注释以前写反了，把"太小"说成得分趋同、把"太大"说成退化硬阈值。）
    thermo_temperature: float = 0.12
    affinity_temperature: float = 0.15
    # 注意这里只有两个 temperature，不是四个，这是刻意的：
    #   * 酶解稳定性维复用 bio/cleavage.py 的 stability_score，
    #     它本身就输出 [0,1] 且内部已对"位点数 ≥6"做过饱和处理；
    #   * 结构合理性维的输出同样已在 [0,1]（含一票否决的上限）。
    # 这两维如果再加一层 sigmoid，就会形成两层归一化叠加
    # （第一层把 ℃ 压成 [0,1]，第二层再压一次），有效陡度变成两个参数的
    # 乘积，最终曲线难以解释。
    # 与其加一个不被读取的配置项让人以为能调，不如不加。

    # 热稳定性维度里，ratio_score 的陡度参数（单位 ℃，与 baseline_tm 解耦）。
    # 单独拿出来是因为 baseline_tm 会随参照肽变化，而曲线陡度不应该跟着变，
    # 否则两次实验的奖励尺度不可比。
    thermo_score_scale: float = 45.0

    # 归一化参考基准（默认取数据集中位肽的性质）。
    baseline_tm: float = 45.0
    baseline_affinity: float = 0.5

    # 奖励裁剪：防止某个维度出现极端值主导整个奖励（比如预测器崩了给出 -1e3）
    clip_min: float = -1.0
    clip_max: float = 1.0

    def weights_from_config(self) -> RewardWeights | None:
        """返回本配置里显式指定的四维权重；没指定过返回 None。

        这个方法的返回值有两个消费者，缺一不可：

        * `composite.dynamic_weights` 用它决定"配置要不要压过任务预设"
          （显式给了就压过，没给就让 `TASK_PRESETS` 生效）；
        * `CompositeScorer.stats()` 用它报告真正生效的权重——
          以前那里直接把 `self.weights` 报出来，而打分路径从不读它，
          于是诊断信息报了一份从未生效的权重。

        返回 None 而不是"默认权重"，是因为这两者语义不同：None 表示
        "让任务预设说话"。调用方必须显式处理这个分支，不能拿默认值糊过去。
        """
        # 只看内层 weights 的 fields_set：`weights=RewardWeights()` 里
        # 一个维度都没给，等于没说话，不该被当成"显式指定了 balanced 权重"。
        explicit_weights = bool(self.weights.model_fields_set)
        explicit_preset = "preset" in self.model_fields_set
        if not (explicit_weights or explicit_preset):
            return None
        # 复用 from_preset 的"预设打底 + 只覆盖显式给出的维度"合并规则，
        # 不在这里再写一遍——同一套合并逻辑写两遍必然漂移。
        return RewardConfig.from_preset(self.preset, weights=self.weights).weights

    @classmethod
    def from_preset(cls, preset: str, **overrides: Any) -> RewardConfig:
        """按预设名构造配置，例如 RewardConfig.from_preset("stability_first")。

        允许用 overrides 覆盖任意字段，包括覆盖预设自带的权重：
            RewardConfig.from_preset("stability_first", weights=RewardWeights(thermo=0.8))
        或直接给一个字典：
            RewardConfig.from_preset("stability_first", weights={"thermo": 0.8})

        `weights` 必须先弹出再构造
        ------------------------
        以前写的是 `cls(preset=preset, weights=RewardWeights(...), **overrides)`，
        当 overrides 里带 `weights` 时，关键字参数 `weights` 出现两次，
        直接抛 `TypeError: got multiple values for keyword argument 'weights'`——
        而且是在运行时才炸，配置类本身不会报错。
        这里显式 pop 出来做一层合并，字典形式也一并支持，
        免得调用方还得自己 import RewardWeights 再手工拼一个对象。
        """
        if preset not in REWARD_PRESETS:
            raise KeyError(f"未知预设 {preset}，可选 {list(REWARD_PRESETS)}")

        base = dict(REWARD_PRESETS[preset])
        override_weights = overrides.pop("weights", None)
        if override_weights is not None:
            # 只覆盖显式给出的维度，其余沿用预设——这样调用方可以只改一项。
            # 用 exclude_unset 而不是 model_dump()：后者会把 RewardWeights
            # 里没被显式赋值的字段也按类默认值吐出来，于是
            # `RewardWeights(thermo=0.8)` 会把 protease/affinity/structure
            # 悄悄重置成 0.30/0.25/0.15，把预设里那几个值覆盖掉——
            # 和传字典的行为不一致，而且违背了"只改一项"的直觉。
            if isinstance(override_weights, RewardWeights):
                supplied = override_weights.model_dump(exclude_unset=True)
            else:
                supplied = dict(override_weights)
            unknown = set(supplied) - set(base)
            if unknown:
                raise KeyError(
                    f"未知的权重维度 {sorted(unknown)}，可选 {sorted(base)}"
                )
            base.update(supplied)

        return cls(
            preset=preset, weights=RewardWeights(**base), **overrides
        )


# --------------------------------------------------------------------------
# 5. 推理 / 部署配置
# --------------------------------------------------------------------------
class InferConfig(StrictModel):
    """推理服务配置。

    本节分成两组字段，消费情况完全不同——别把它们混为一谈。

    第一组：由 `serve/` 真实消费（推理服务已落地）
    ---------------------------------------------------
    `base_model` / `adapter_path` / `merged_model_path` / `quantization` /
    `max_model_len` / `max_num_seqs` / `max_num_batched_tokens` /
    `temperature` / `top_p` / `max_tokens` / `repetition_penalty` /
    `host` / `port` / `batch_size` / `batch_timeout_ms` / `request_timeout_s`

    这些字段的读取点在 `src/pepdesign/serve/` 里（engine.py 与 app.py），
    服务真的按它们跑。判据和别处一样：每个字段都能 grep 到读取点。

    第二组：vLLM 后端专属——`backend="vllm"` 时才生效
    ------------------------------------------------------
    `gpu_memory_utilization` / `max_num_seqs` / `enable_prefix_caching` /
    `tensor_parallel_size`（以及 `block_size` / `swap_space`，见下）

    这组字段的消费情况在 2026-10 变了：以前服务只有 transformers 一个后端，
    这五个是惰性的；现在 `serve/engine.py` 里接了 vLLM 分支，
    前四个会被真的传给 `vllm.LLM(...)`。

    仍然不生效的是 `block_size` 与 `swap_space`：它们的写法在不同 vLLM
    版本之间变过（有的版本挪进了 engine 的配置里），所以在没有 GPU 机器
    逐版本核对之前不传——传一个对方不认识的 kwarg 会直接 TypeError，
    这比"配了不读"更糟。保留它们是为了让配置项名与 vLLM 的接口保持一致。

    怎么保证不漏：`backend="transformers"`（默认）时前四个依然是惰性的，
    所以字段名和值在两种后端下是同一套——改 `gpu_memory_utilization`
    不会在 transformers 后端上"看起来生效"。

    对外表述：服务是真的（接口、批处理、adapter 热切换都能跑）；
    `backend="transformers"` 时没有 PagedAttention / 连续批，
    峰值吞吐低于 vLLM，且这条路径在 CPU 上也能跑（本地验证用的就是它）。
    """

    # 后端。"transformers" 是本项目真实实现的那一个：
    # serve/engine.py 用它加载基座 + adapter 并做静态批处理。
    # 选 "vllm" 会在 `InferenceEngine.__init__` 里抛带指引的
    # NotImplementedError——明确的"没实现"比"配了但跑的是别的东西"好。
    backend: Literal["transformers", "vllm"] = "transformers"

    base_model: str = "LOGOS-Hub/LOGOS-pretrain-3B"
    adapter_path: str | None = "outputs/grpo/final"
    merged_model_path: str | None = None  # 若已合并 LoRA，直接加载合并后的模型更快

    # ---- vLLM 后端专属（backend="vllm" 时才被读取，见类文档第二组）----
    # 这几个是 PagedAttention 与多卡并行的参数。`backend="transformers"`
    # （默认）时不会读它们；`backend="vllm"` 时前四个会真的传给
    # `vllm.LLM(...)`。
    #
    # `gpu_memory_utilization` 仅在 `backend="vllm"` 时生效；
    # 在默认的 transformers 后端上会被忽略。
    #
    # PagedAttention 把 KV cache 切成固定大小的 block 来管理（类似操作系统的
    # 虚拟内存分页），消除了传统实现里"按最大长度预分配"造成的内部碎片。
    # 这两项就是"显存利用率提升到 85% 以上"的来源：
    gpu_memory_utilization: float = 0.90   # vLLM 最多占用 90% 显存做权重 + KV cache
    max_num_seqs: int = 32                 # 单批最大并发序列数
    max_num_batched_tokens: int = 8192
    block_size: int = 16                   # PagedAttention 的块大小（token 数）
    # 前缀缓存：多轮优化时同一靶点描述会被反复作为前缀，缓存 KV 可直接复用，
    # 是"多轮优化场景"下延迟优化的关键开关
    enable_prefix_caching: bool = True
    swap_space: int = 4                    # CPU 交换空间（GB），用于容纳瞬时峰值

    # ---- 长上下文 ----
    # 32K 窗口要同时容纳：完整靶点蛋白序列 + 多轮优化历史 + 设计指令。
    # 注意这是推理窗口；训练时受显存限制只用了 2048（见 SFTConfig），
    # 这种"训练短、推理长"的做法依赖 RoPE 的外推能力，
    # 如果效果掉得厉害就需要在长样本上做一轮轻量继续训练。
    max_model_len: int = 32768

    quantization: QuantConfig = Field(default_factory=QuantConfig)

    # ---- 生成默认参数 ----
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    repetition_penalty: float = 1.05

    # ---- 服务 ----
    host: str = "0.0.0.0"
    port: int = 8000
    tensor_parallel_size: int = 1          # 单卡部署，多卡时调大
    # 批量调度：一次提交多条序列优化需求，服务端按此并发度调度
    batch_size: int = 16
    batch_timeout_ms: int = 50             # 攒批等待窗口
    request_timeout_s: float = 300.0


# --------------------------------------------------------------------------
# 6. 数据配置
# --------------------------------------------------------------------------
class DataConfig(StrictModel):
    """数据集构建配置。"""

    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    calibration_dir: str = "data/calibration"

    # 长度过滤：以主分布 5-30 为主，放宽上下界以纳入长肽与环肽（对应遗留问题一）
    min_length: int = 5
    max_length: int = 60
    # 至少要保留多少比例的标准残基，太脏的序列直接丢。
    #
    # 取 1.0（零容忍），不是 0.9。0.9 曾经放进来 71 条含 X/B 的序列：
    # `validate()` 的 `ALLOWED_AA` 是 25 个字母（20 标准 + XBZUO），
    # X 本身是合法的；真正的闸门就是这里。而 对短肽来说 0.9 等于没有闸门——
    # 一条 23 残基的序列允许 2 个 X（91.3% 刚好过线），
    # 实测放进来的是 HIV 包膜糖蛋白、纤维蛋白原这类片段。
    #
    # 为什么短肽不能容忍 X：X 的含义是"这个位置不知道是什么"。
    # 一条 40 残基的抗体片段里有 2 个 X 仍然可用，但一条要在实验室合成的
    # 多肽里出现 X，等于这条序列没法下单。放进 SFT 语料就是在教模型
    # "输出里带 X 也行"，而这个错误在 loss 曲线上完全看不出来。
    # 代价只有 2.7% 的样本，收益是语料里每一条序列都可合成。
    min_standard_ratio: float = 1.0
    # 去重阈值：不做完全去重，而是去"近似重复"。
    # 同源多肽在公开库里大量存在（同一家族不同物种），
    # 只按字符串去重会让模型反复看几乎一样的样本，是过拟合的隐形来源。
    dedup_identity: float = 0.9
    # 环肽样本的最低占比。公开库里环肽天然稀少，
    # 不做主动上采样，模型对环肽的能力会弱到不可用。
    min_cyclic_ratio: float = 0.15

    # ---- 语料规模（构造多少条，不是采集多少条）----
    # 这三个量的实质是序列复用倍数：sft_size ÷ 训练序列数 = 每条序列
    # 平均被用几次。历史设定是 136 条序列 → 8000 条语料，复用 59 次；
    # 那是"序列太少、必须靠包装撑起训练量"的做法。
    #
    # 序列变多之后这个关系要反过来看：扩增出来的是同一批序列的不同任务
    # 包装，不是新序列。所以它该按"覆盖够不够"定，而不是越大越好；
    # 复用超过 ~8 次就只是在让模型背同一批序列。
    #
    # 取 15,000 的依据（实测训练集 2,392 条序列、1,571 簇）：
    #   de_novo      52% → 7,800 条 → 每条序列 3.3 次
    #   stabilize    37% → 5,550 条 → 2.3 次
    #   multi_obj    10% → 1,500 条 → 0.6 次
    #   合计 15,000 / 2,392 = 6.3 倍复用
    #
    # 这个值是跟着序列数走的，不是一个固定常数：它是 2,392 × 6.3 得来的。
    # 序列数变了（比如把 DRAMP 接进来、或者把前体过滤放宽）就要重算——
    # 固定成 20,000 时复用会涨到 8.4 倍，越过了上面那条 8 倍的上限，
    # 而多出来的条数并不会带来新序列，只是同一批序列的重复包装。
    #
    # 复用倍数从 5.6 涨到 6.3，不是因为语料变多了，是因为序列变少了：
    # 采集后清洗砍掉了 326 条大蛋白结构域片段（Envelope glycoprotein 等）
    # 和 80 条含 X 的序列，训练序列 2,671 → 2,392。
    # 这是主动选择——留下的序列少一些，但条条是可合成、可下药的。
    # 代价是离 8 倍上限只剩 1.7 倍余量；再接一个新数据源（DRAMP）
    # 之前，先回头看这个数。
    sft_size: int = Field(15000, ge=1)
    grpo_size: int = Field(2000, ge=1)
    # 校准集同样受序列数约束：它是"给一条序列设计一个改造方案"的配对，
    # 一对要占一条序列。500 对是当前量级下的合理值。
    calibration_size: int = Field(500, ge=1)

    train_ratio: float = 0.9
    valid_ratio: float = 0.05
    test_ratio: float = 0.05

    # 数据源清单（下载脚本读这个）。enabled=False 的源默认不拉，
    # 因为部分源需要人工申请授权或人工下载后放目录。
    sources: dict[str, dict] = Field(
        default_factory=lambda: {
            "uniprot": {
                "enabled": True,
                "kind": "api",
                "note": "UniProt REST，按 keyword 拉取多肽条目",
            },
            "apd3": {
                "enabled": False,
                "kind": "manual",
                "note": "抗菌肽数据库，需人工下载后放入 data/raw/apd3",
            },
            "dramp": {
                "enabled": False,
                "kind": "manual",
                "note": "抗菌肽与治疗肽库，人工下载",
            },
            "dbaasp": {
                "enabled": False,
                "kind": "manual",
                "note": "抗菌肽活性数据，人工下载",
            },
            "pdb_cyclic": {
                "enabled": False,
                "kind": "manual",
                "note": "PDB 环肽结构，用于结构相关标注",
            },
            "internal": {
                "enabled": True,
                "kind": "local",
                "note": "内部专家标注与实验数据（脱敏后）",
            },
        }
    )

    @field_validator("train_ratio", "valid_ratio", "test_ratio")
    @classmethod
    def _ratio_range(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("切分比例必须在 (0,1) 之间")
        return v


# --------------------------------------------------------------------------
# 7. 离线评估配置
# --------------------------------------------------------------------------
class EvalConfig(StrictModel):
    """离线评估（`pepdesign.eval`）。

    这里的字段全部只被 `eval/` 和 `scripts/eval_offline.py` 消费，
    不参与训练——评估读的是训练已经产出的东西，所以它可以在
    训练之外单独跑、反复跑，也不会影响任何一次训练的结果。

    阈值放在配置里的原因
    ---------------------------------------------------
    因为它是一条可以被数据推翻的判断，而不是一个物理常数。
    现在默认 20，是因为本仓库只有 6 条实测数据；等真拿到几百条实验数据，
    这个值应该按可接受的最少样本量重新确定。
    写死在代码里的话，改它需要发版，而这件事本来是个实验决策。
    """

    #: 奖励审计的输入：带实验数据的记录（`data/pipeline.py` 的产物）
    records_path: str = "data/processed/train_records.jsonl"
    #: 报告落盘位置
    report_path: str = "outputs/eval/reward_audit.json"

    #: 低于这个样本量就不下结论。见 `eval/offline.MIN_N_FOR_SIGNIFICANCE`
    #: 对 20 这个数的来历的说明。
    min_n_for_significance: int = 20
    #: 置换检验的置换次数。同时决定 p 值的分辨率下限 1/(n_perm+1)。
    n_permutations: int = 5000
    #: 置换检验的随机种子。必须固定：评估报告要能复现，
    #: 而置换检验是随机的——不固定种子的话同一份数据跑两次会有不同的 p 值，
    #: 报告就没法被引用。
    seed: int = 42
    #: Top-K 富集里的 K（研发实际只会合成前 K 条）
    top_k: int = 20

    @field_validator("min_n_for_significance", "n_permutations", "top_k")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"必须是正整数，收到 {v}")
        return v


# --------------------------------------------------------------------------
# 8. 顶层聚合配置
# --------------------------------------------------------------------------
class ProjectConfig(StrictModel):
    """一份完整实验的全部配置。CLI 加载的就是这个对象。"""

    data: DataConfig = Field(default_factory=DataConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    quant: QuantConfig = Field(default_factory=QuantConfig)
    sft: SFTConfig = Field(default_factory=SFTConfig)
    grpo: GRPOConfig = Field(default_factory=GRPOConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    infer: InferConfig = Field(default_factory=InferConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)


# --------------------------------------------------------------------------
# 9. 加载与覆盖
# --------------------------------------------------------------------------
def resolve_path(p: str | Path) -> Path:
    """把相对路径解析到项目根目录下。

    为什么不依赖 cwd：训练脚本常常从任意目录被调度器拉起
    （cron/systemd/k8s），依赖 cwd 会导致"本地能跑、线上找不到文件"。
    """
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读 YAML 配置文件。"""
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是映射（dict），实际是 {type(data).__name__}: {p}")
    return data


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并字典，override 覆盖 base。用于多层配置叠加。"""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """把 `--set a.b.c=value` 形式的命令行覆盖应用到配置字典上。

    值会被 yaml.safe_load 解析，所以 `--set train.lr=2e-4` 得到 float、
    `--set lora.target_modules=[q_proj]` 得到 list，
    不需要在命令行里手动标注类型。
    """
    out = dict(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"覆盖项格式应为 key=value，收到: {item}")
        dotted, _, raw = item.partition("=")
        keys = dotted.strip().split(".")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw  # 解析不了就按原始字符串处理
        cursor = out
        for k in keys[:-1]:
            cursor = cursor.setdefault(k, {})
        cursor[keys[-1]] = value
    return out


#: 配置里已经定义、但没有任何代码消费的顶层小节 → 一句说明。
#: `load_config` 只在这些小节被显式写入时告警：默认值不算，
#: 因为 `ProjectConfig` 给每个小节都建了默认对象，
#: 对着"没配过的东西"报未实现等于每次启动都刷一条噪音。
#: "配置里写了、但没有任何代码读"的小节。
#:
#: `infer` 已经从这张表里移除了（2026-09）：`serve/` 包实现了推理服务，
#: 该小节里有 16 个字段被真实消费，所以"配了不生效"这句话对它不再成立。
#: 剩下的问题只是它内部有两类字段（vLLM 专属那 5 个仍不被读取），
#: 而那是 InferConfig 类文档里说清楚的事，不该靠这张表来含糊表达。
#:
#: 这张表留成空的、而不是删掉，是因为它是一种机制：
#: 谁再加一个"只写配置没写代码"的小节，就该把它登记进来，
#: 让 `load_config` 在运行日志里说一声。删掉机制的话，后续维护者
#: 只能靠读注释才发现自己的配置没生效。
UNIMPLEMENTED_SECTIONS: dict[str, str] = {}


def load_config(
    path: str | Path | None = None, overrides: list[str] | None = None
) -> ProjectConfig:
    """主入口：加载配置并校验成强类型对象。

    用法::

        cfg = load_config("configs/sft_lora.yaml",
                          overrides=["sft.learning_rate=3e-4", "lora.r=16"])

    "配了不生效"的字段在这一层拦一道
    ---------------------------------
    `pydantic` 的 `extra="forbid"` 只能拦住拼错/不存在的字段；
    拦不住"字段存在但没人读"这一类。后者更麻烦：配置照常通过校验、
    脚本照常启动、日志里那条设置看起来生效了，只是它对结果毫无影响。
    所以这里对 `UNIMPLEMENTED_SECTIONS` 里的小节显式写入时打 warning，
    让"我配了这个"和"这个没接线"在同一屏里同时出现。

    判断"显式写入"用的是 `raw` 里有没有这个键，而不是最终对象的
    `model_fields_set`：`ProjectConfig(**raw)` 只有在 raw 里给了 infer
    时才会把它标成 set，两者等价，但直接看 raw 更直白。
    """
    raw: dict[str, Any] = {}
    if path is not None:
        raw = load_yaml(path)
    if overrides:
        raw = apply_overrides(raw, overrides)

    for section, note in UNIMPLEMENTED_SECTIONS.items():
        if section in raw:
            logger.warning(
                "配置里写了 `%s:` 小节，但 %s。它不会改变任何行为。",
                section, note,
            )

    return ProjectConfig(**raw)


def save_config(cfg: ProjectConfig, path: str | Path) -> Path:
    """把最终生效的配置落盘。

    每个实验目录里都存一份 resolved config，是复现的最低要求：
    光有 git commit 不够，因为配置可能是从命令行覆盖进来的，
    只有落盘的这份才是"当时真实生效的配置"。
    """
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(mode="json"), f, allow_unicode=True, sort_keys=False)
    return p
