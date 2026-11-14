"""一步之内的指标聚合（`grpo.aggregate_step_metrics`）。

这个函数是纯 Python 的，虽然住在 `train/grpo.py` 里，但那一层**顶层不 import
torch**（可选依赖只在真正使用处导入），所以这里可以零安装直接测——
和 tests/conftest.py 里那条"整个 tests/ 不许 import torch"的约定并不冲突。

独立测试文件的原因
-----------------------------------
它修的是一个观测失真类的 bug：以前训练循环写的是 `last_metrics = m`，
每个 micro-batch 覆盖上一个，于是 `grad_accum=4` 时记下来的只有最后一个
micro-batch 的指标。这类 bug 不会崩、不会报错、loss 照样打印，
只会让"照着日志调参"的人对着 1/4 的样本下判断。
所以下面的断言分两半：

  * 口径：条数加权 ≠ 算术平均，组数加权 ≠ 条数加权；
  * 回归锁：结果必须不等于任何一个单独的 part（老行为），
    否则这个函数改回来了也测不出来。
"""

from __future__ import annotations

import logging

import pytest

from pepdesign.train.grpo import (
    METRIC_KEYS,
    StepMetrics,
    _log_training_summary,
    aggregate_step_metrics,
    grpo_scalars,
)


def mk(**kw) -> StepMetrics:
    """造一个 part。默认值刻意都取 0，让每个用例只关心自己设的那几个。"""
    return StepMetrics(**kw)


# ---------------------------------------------------------------------------
# mean_reward / kl 等：按序列条数加权
# ---------------------------------------------------------------------------
def test_means_are_weighted_by_sequence_count():
    """不满的最后一个 micro-batch 不能被等权放大。

    两批各 800 条和 200 条（比如最后一批语料不够），mean_reward 是
    0.10 和 0.60。等权平均给 0.35，条数加权给 0.20。差的这 0.15
    在奖励尺度上足以让"奖励在涨"和"奖励在跌"反过来。
    """
    parts = [(mk(mean_reward=0.10, kl=0.01), 800, 8), (mk(mean_reward=0.60, kl=0.11), 200, 2)]
    out = aggregate_step_metrics(parts, step=7)

    assert out.mean_reward == pytest.approx(0.20)
    assert out.mean_reward != pytest.approx(0.35)   # 不是算术平均
    assert out.kl == pytest.approx(0.03)
    assert out.step == 7
    # 回归锁：不能是任何一个单独的 part（老行为取的是最后一个）
    assert out.mean_reward != pytest.approx(0.10)
    assert out.mean_reward != pytest.approx(0.60)


def test_ratio_fields_are_weighted_by_sequence_count_too():
    """比例类字段（解析失败率、裁剪率、可用率）同样是条数加权。"""
    parts = [
        (mk(parse_fail_rate=0.0, clip_fraction=0.5, weak_parse_ratio=0.0, usable_ratio=1.0), 90, 9),
        (mk(parse_fail_rate=1.0, clip_fraction=0.0, weak_parse_ratio=1.0, usable_ratio=0.0), 10, 1),
    ]
    out = aggregate_step_metrics(parts, step=1)

    assert out.parse_fail_rate == pytest.approx(0.10)
    assert out.clip_fraction == pytest.approx(0.45)
    assert out.weak_parse_ratio == pytest.approx(0.10)
    assert out.usable_ratio == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# reward_std：按组数加权（口径与上面几个字段不同）
# ---------------------------------------------------------------------------
def test_reward_std_is_weighted_by_group_count_not_sequence_count():
    """`reward_std` 是"组内离散度在组之间的平均"，权重必须是组数。

    它在 `run_step` 里的定义是 `mean(g.reward_std() for g in batch.groups)`，
    每个组贡献一项，与该组的序列条数无关——所以聚合时也该按组数加权。

    这个用例把两种加权明确分开：A 批 8 组 / 80 条、std=0.0；
    B 批 2 组 / 20 条、std=0.5。
      * 按组数加权 → (0.0×8 + 0.5×2) / 10 = 0.10  ← 正确
      * 按条数加权 → (0.0×80 + 0.5×20) / 100 = 0.10  ← 这里恰好相同

    所以再看第二条数据：A 批 2 组 / 80 条、B 批 8 组 / 20 条。
      * 按组数加权 → (0.0×2 + 0.5×8) / 10 = 0.40
      * 按条数加权 → (0.0×80 + 0.5×20) / 100 = 0.10
    两者相差 4 倍。若把权重改回条数，第二条断言会当场失败。
    """
    skewed = [(mk(reward_std=0.0), 80, 2), (mk(reward_std=0.5), 20, 8)]
    assert aggregate_step_metrics(skewed, step=1).reward_std == pytest.approx(0.40)


def test_reward_std_is_never_a_variance_decomposition():
    """`reward_std` 不做方差合成——它没有"全体序列的 std"这个含义。

    反例：A 批 mean=0, std=0（全是 0 分）、B 批 mean=1, std=0（全是 1 分）。
    如果把它当"某一层的 (mean, std, n)"去做方差分解，会算出总体 std=0.5；
    但真实的量是"平均组内离散度"= 0——**两组各自内部都毫无区分度，
    这一步对它们没有梯度**，这才是调参的人需要看到的信号。
    0.5 会让人以为奖励分得很开、继续训就行。
    """
    parts = [
        (mk(mean_reward=0.0, reward_std=0.0), 8, 1),
        (mk(mean_reward=1.0, reward_std=0.0), 8, 1),
    ]
    out = aggregate_step_metrics(parts, step=1)

    assert out.mean_reward == pytest.approx(0.5)   # 均值合成是对的
    assert out.reward_std == pytest.approx(0.0)    # 但 std 不是 0.5
    assert out.reward_std != pytest.approx(0.5)


def test_reward_std_falls_back_when_no_groups():
    """一组都没记上时给 0，而不是 ZeroDivisionError。"""
    out = aggregate_step_metrics([(mk(mean_reward=0.3, reward_std=0.9), 4, 0)], step=2)
    assert out.reward_std == 0.0
    assert out.mean_reward == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# 逐维组内 std：把"某一维已经死了"从总分里拆出来
# ---------------------------------------------------------------------------
def test_dimension_stds_are_aggregated_by_group_count():
    """逐维 std 与 `reward_std` 同一口径：按组数加权。

    理由一样——它是"组内离散度在组之间的平均"，每个组贡献一项。
    """
    parts = [
        (mk(dimension_stds={"protease": 0.0}), 80, 2),
        (mk(dimension_stds={"protease": 0.5}), 20, 8),
    ]
    out = aggregate_step_metrics(parts, step=1)
    assert out.dimension_stds["protease"] == pytest.approx(0.40)


def test_dimension_missing_from_a_part_does_not_drag_it_down():
    """某个 micro-batch 里没有定义的维度不参与该维的平均。

    这是与 `reward_std` 唯一的口径差别：`reward_std` 每一批都有值，
    而某个奖励维度可能因为这一批全是无效样本而量不出来。
    把它按 0 计入分母会凭空拉低该维的 std，掩盖真实情况——
    而"量不出来"和"量出来是 0"在排查时指向完全不同的方向。
    """
    parts = [
        (mk(dimension_stds={"protease": 0.6}), 40, 4),
        (mk(dimension_stds={}), 40, 4),                 # 这批没量出来
    ]
    out = aggregate_step_metrics(parts, step=1)
    assert out.dimension_stds["protease"] == pytest.approx(0.60)


def test_dimension_stds_keeps_every_dimension_seen_in_any_part():
    """任一 micro-batch 出现过的维度都要出现在结果里。

    漏掉某一维的后果是"日志上看着挺好，而那一维其实一直是 0"——
    这正是加这个字段要解决的那个问题。
    """
    parts = [
        (mk(dimension_stds={"thermo": 0.3}), 40, 4),
        (mk(dimension_stds={"protease": 0.0, "structure": 0.2}), 40, 4),
    ]
    out = aggregate_step_metrics(parts, step=1)
    assert set(out.dimension_stds) == {"thermo", "protease", "structure"}
    assert out.dimension_stds["protease"] == 0.0


def test_dimension_log_line_sorts_the_dead_dimension_first():
    """日志行按 std 升序排——最该查的零方差维度排在最前面。

    这不是装饰：这一行的用途就是"扫一眼看出哪一维不动了"，
    降序排会把它埋在一堆正常值后面。
    """
    m = mk(dimension_stds={"thermo": 0.31, "protease": 0.0, "structure": 0.12})
    line = m.dimension_log_line()
    assert line.index("protease") < line.index("structure") < line.index("thermo")


def test_dimension_log_line_is_empty_without_data():
    """没有逐维数据时返回空串，调用方据此跳过这一行（不是打一行空的）。"""
    assert mk().dimension_log_line() == ""


def test_scalars_prefix_dimension_keys_so_metric_keys_stay_fixed():
    """逐维 std 走 `dim/` 前缀，不进 `METRIC_KEYS`。

    `METRIC_KEYS` 是一份固定契约（"这几个键总是存在"），
    往里塞动态键会让这个假设失效。前缀分开之后两边的读取方式都简单。
    """
    m = mk(dimension_stds={"thermo": 0.25})
    scalars = grpo_scalars(m)
    assert scalars["dim/thermo_std"] == pytest.approx(0.25)
    for key in METRIC_KEYS:
        assert key in scalars


# ---------------------------------------------------------------------------
# 计数类字段：求和
# ---------------------------------------------------------------------------
def test_degenerate_groups_is_summed_not_averaged():
    """`degenerate_groups` 是计数，求和。

    取平均会得到一个 0.75 这种"有几个组"没有意义的数；更要紧的是
    这条日志的用途是"这一步有多少组完全没梯度"，
    4 个 micro-batch 各退化 1 组，答案就是 4。
    """
    parts = [(mk(degenerate_groups=1), 32, 4) for _ in range(4)]
    assert aggregate_step_metrics(parts, step=1).degenerate_groups == 4


# ---------------------------------------------------------------------------
# 样例与边界
# ---------------------------------------------------------------------------
def test_example_sequences_come_from_the_first_part():
    """样例是给人看的，只留第一批：混在一起反而看不出这条属于哪一批。"""
    parts = [
        (mk(example_sequences=["A" * 20]), 4, 1),
        (mk(example_sequences=["W" * 20]), 4, 1),
    ]
    out = aggregate_step_metrics(parts, step=1)
    assert out.example_sequences == ["A" * 20]


def test_example_sequences_is_copied_not_aliased():
    """必须拷贝：直接引用第一个 part 的列表，后面改动会污染已入 history 的记录。"""
    src = mk(example_sequences=["A" * 20])
    out = aggregate_step_metrics([(src, 4, 1)], step=1)

    out.example_sequences.append("mutated")
    assert src.example_sequences == ["A" * 20]


def test_empty_parts_raises():
    with pytest.raises(ValueError, match="空的 parts"):
        aggregate_step_metrics([], step=1)


def test_zero_sequences_keeps_counts_and_does_not_divide_by_zero():
    """整步一条序列都没产出（生成本身失败）时：

    比例类字段退化成 0/0，这里的选择是保持第一个 part 的形态而不是
    编一个数出来；但计数类字段仍然如实求和，因为它们跟条数无关，
    而"这一步退化了几组"恰恰是这种失败场面下最该看见的东西。
    """
    parts = [(mk(mean_reward=0.25, degenerate_groups=2), 0, 0), (mk(degenerate_groups=3), 0, 0)]
    out = aggregate_step_metrics(parts, step=9)

    assert out.step == 9
    assert out.mean_reward == pytest.approx(0.25)
    assert out.reward_std == 0.0
    assert out.degenerate_groups == 5


def _warnings(caplog) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
    ]


def test_summary_warns_when_every_rollout_failed_to_parse(caplog):
    """全部解析失败时，必须说"这是格式问题"，而不是让人去调温度。

    这正是第一次真跑 `run_grpo` 时的现场：`max_completion_length=32`
    而答案最短要 71 token，于是所有 rollout 都被截断、解析全挂、
    奖励恒为 0。老代码给出的建议是"查温度/查奖励饱和/查 prompt 难度"，
    三条里没有一条能修好它。
    """
    with caplog.at_level(logging.WARNING):
        _log_training_summary(
            [mk(parse_fail_rate=1.0, reward_std=0.0), mk(parse_fail_rate=1.0, reward_std=0.0)],
            num_generations=2,
        )
    text = "\n".join(_warnings(caplog))
    assert "解析失败" in text
    assert "max_completion_length" in text
    # 关键：要明确叫停调温度这条路，而不只是不提它。
    assert "先别动温度" in text


def test_summary_distinguishes_no_spread_from_no_parse(caplog):
    """解析成功但组内分数全一样 —— 这才是温度/奖励/prompt 的问题。

    与上一条的关键区别是 `parse_fail_rate=0`：解析是好的，
    只是组内 8 条拿到的分完全相同。两段诊断不能互相串台。
    """
    with caplog.at_level(logging.WARNING):
        _log_training_summary(
            [mk(parse_fail_rate=0.0, reward_std=0.0, degenerate_groups=16)],
            num_generations=8,
        )
    text = "\n".join(_warnings(caplog))
    assert "离散度都是 0" in text
    assert "G=8" in text, "组大小必须来自配置，不能写死"
    assert "解析失败" not in text


def test_summary_uses_the_configured_group_size_not_a_hardcoded_eight(caplog):
    """回归锁：G 是配置项，诊断里不能出现写死的 8。

    老代码的文案里写死了"组内 8 条几乎一样"，而 `smoke_cpu.yaml`
    用的是 G=2、命令行也常覆盖成 4——照着日志排查的人会去检查一个
    根本不存在的配置。
    """
    for g in (2, 4, 32):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            _log_training_summary(
                [mk(parse_fail_rate=0.0, reward_std=0.0, degenerate_groups=1)],
                num_generations=g,
            )
        assert f"G={g}" in "\n".join(_warnings(caplog))


def test_summary_is_quiet_when_training_is_healthy(caplog):
    """有梯度的时候一条警告都不该有——否则警告会被当成背景噪音。"""
    with caplog.at_level(logging.WARNING):
        _log_training_summary(
            [
                mk(parse_fail_rate=0.25, reward_std=0.12, mean_reward=0.3),
                mk(parse_fail_rate=0.10, reward_std=0.18, mean_reward=0.5),
            ],
            num_generations=8,
        )
    assert _warnings(caplog) == []


def test_summary_on_empty_history_does_not_crash(caplog):
    """一步都没跑（比如 max_steps 被设成 0）时不能抛异常。"""
    with caplog.at_level(logging.WARNING):
        _log_training_summary([], num_generations=8)
    assert _warnings(caplog) == []
