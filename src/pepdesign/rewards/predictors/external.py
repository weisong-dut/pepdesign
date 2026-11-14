"""外部计算生物学工具的适配层。

外部工具通常需要单独安装、授权或联网，且延迟与奖励在线计算的要求冲突；
生产环境一般用离线流水线批量预测并写入缓存表。因此这里的类实现了完整接口，
但在构造时抛出带指引的异常，避免"配置选择 tool 却静默使用其他实现"。
"""

from __future__ import annotations

from pepdesign.rewards.predictors.base import BasePredictor, PredictionResult

# 统一的错误信息模板
_NOT_IMPLEMENTED_HINT = (
    "外部工具适配层尚未接入。生产环境中的推荐做法是："
    "用离线流水线批量调用外部工具并把结果写入预测缓存表，"
    "再实现本类去查表（而不是在奖励计算时实时调用），"
    "这样才能把单条打分延迟控制在内存查表级别。"
    "当前请在配置中改用内置后端（heuristic / rule / feature / propensity）。"
)


class _ExternalToolPredictor(BasePredictor):
    """外部工具预测器的公共基类。

    子类声明 `name` 与 `tool_hint`，构造函数统一拒绝实例化。
    """

    #: 提示应该接入什么工具
    tool_hint: str = ""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}（建议接入: {self.tool_hint}）{_NOT_IMPLEMENTED_HINT}"
        )

    def _predict_one(self, sequence: str) -> PredictionResult:  # pragma: no cover
        raise NotImplementedError(_NOT_IMPLEMENTED_HINT)


class ExternalToolTmPredictor(_ExternalToolPredictor):
    """热稳定性：建议接入 CD 实验数据库或 ThermoNet 类模型。"""

    name = "tm_tool"
    tool_hint = "实验 Tm 数据库 / ThermoNet / 基于 ESM 的回归头"


class ExternalToolProteasePredictor(_ExternalToolPredictor):
    """酶解稳定性：建议接入 PeptideCutter、PROSPER、DeepCleave 类模型。"""

    name = "protease_tool"
    tool_hint = "PeptideCutter(ExPASy) / PROSPER / DeepCleave"


class ExternalToolAffinityPredictor(_ExternalToolPredictor):
    """亲和力：建议接入分子对接或打分函数。"""

    name = "affinity_tool"
    tool_hint = "AutoDock Vina / Glide / Rosetta ddG / 表面等离子共振实验数据"


class ExternalToolStructurePredictor(_ExternalToolPredictor):
    """结构合理性：建议接入二级结构预测与聚集预测工具。"""

    name = "structure_tool"
    tool_hint = "JPred/PSIPRED（二级结构）/ TANGO/AGGRESCAN（聚集）"
