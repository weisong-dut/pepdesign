"""预测器工厂：按配置名构造预测器实例。

后端选择集中在注册表里，新增后端只需注册一行。名字写错时直接抛异常并列出
合法取值，避免静默回退导致"配置未生效但无人发现"。
"""

from __future__ import annotations

from collections.abc import Callable

from pepdesign.rewards.predictors.base import BasePredictor, PredictionResult

# 各类预测器支持的后端名 -> 构造函数。配置文件中的取值必须与这里的 key 一致。
#
# thermo 不提供 "esm"：对应模块尚未实现，避免配置允许但构造时才失败。
THERMO_BACKENDS: tuple[str, ...] = ("heuristic", "tool")
PROTEASE_BACKENDS: tuple[str, ...] = ("rule", "tool")
AFFINITY_BACKENDS: tuple[str, ...] = ("feature", "esm", "tool")
STRUCTURE_BACKENDS: tuple[str, ...] = ("propensity", "tool")
# 第五维：蛋白质语言模型。"off" 表示这一维不存在。
PLM_BACKENDS: tuple[str, ...] = ("off", "esm")

# 各类预测器的合法 kind 名（build_predictor 的 kind 参数取值）
PREDICTOR_KINDS: tuple[str, ...] = ("thermo", "protease", "affinity", "structure", "plm")


def _lazy(module: str, attr: str) -> Callable[..., BasePredictor]:
    """惰性构造：真正用到某后端时才 import 对应模块。

    ESM 依赖 torch/transformers，顶层导入会让 rewards 包在无 torch 环境下不可用。
    """

    def factory(**kwargs):
        import importlib

        mod = importlib.import_module(module)
        cls = getattr(mod, attr)
        return cls(**kwargs)

    return factory


_REGISTRY: dict[str, dict[str, Callable[..., BasePredictor]]] = {
    "thermo": {
        "heuristic": _lazy("pepdesign.rewards.predictors.tm", "TmPredictor"),
        # tool 后端指向外部命令行工具封装；无外部工具时构造会抛带说明的异常
        "tool": _lazy("pepdesign.rewards.predictors.external", "ExternalToolTmPredictor"),
    },
    "protease": {
        "rule": _lazy(
            "pepdesign.rewards.predictors.protease_sites", "ProteaseSitesPredictor"
        ),
        "tool": _lazy(
            "pepdesign.rewards.predictors.external", "ExternalToolProteasePredictor"
        ),
    },
    "affinity": {
        "feature": _lazy(
            "pepdesign.rewards.predictors.binding", "AffinityPredictor"
        ),
        "esm": _lazy(
            "pepdesign.rewards.predictors.binding", "ESMEmbeddingAffinityPredictor"
        ),
        "tool": _lazy(
            "pepdesign.rewards.predictors.external", "ExternalToolAffinityPredictor"
        ),
    },
    "structure": {
        "propensity": _lazy(
            "pepdesign.rewards.predictors.secondary_structure",
            "StructureReasonablenessPredictor",
        ),
        "tool": _lazy(
            "pepdesign.rewards.predictors.external", "ExternalToolStructurePredictor"
        ),
    },
    "plm": {
        # "off" 指向结构性哑对象：value=0.5、confidence=0.0，既不参与奖励也不报错。
        "off": lambda **kwargs: _NullPlmPredictor(**kwargs),
        "esm": _lazy("pepdesign.rewards.predictors.esm", "ESMPseudoLikelihoodPredictor"),
    },
}


class _NullPlmPredictor(BasePredictor):
    """`plm_backend="off"` 时使用的空预测器。

    返回 confidence=0 而非抛异常，因为"这一维未启用"是合法状态。
    """

    name = "plm_off"
    is_calibrated = True   # 它不输出任何可被采信的信息，不需要打折

    def _predict_one(self, sequence: str) -> PredictionResult:
        return PredictionResult(
            value=0.5,
            confidence=0.0,          # 0 置信度 ⇒ 有效权重 0 ⇒ 不影响总奖励
            detail={"note": "plm_backend='off'，第五维未启用"},
            predictor=self.name,
        )


def build_plm_predictor(backend: str = "off", **kwargs) -> BasePredictor:
    return _build("plm", backend, **kwargs)


def _build(kind: str, backend: str, **kwargs) -> BasePredictor:
    """通用的构造逻辑，四类预测器共用。"""
    # 校验 kind 与 backend，报错中列出合法取值，便于定位配置错误。
    if kind not in _REGISTRY:
        raise ValueError(
            f"未知的预测器类别: {kind!r}，可选值: {list(PREDICTOR_KINDS)}"
        )
    table = _REGISTRY[kind]
    if backend not in table:
        raise ValueError(
            f"未知的 {kind} 后端: {backend!r}，可选值: {sorted(table)}"
        )
    try:
        return table[backend](**kwargs)
    except ImportError as exc:
        # 区分模块不存在与依赖未安装，两者排查方向不同。
        raise ImportError(
            f"{kind} 后端 {backend!r} 依赖未安装或模块尚未实现: {exc}。"
            f"请在配置里改用 {sorted(set(table) - {backend})} 中的可用后端。"
        ) from exc
    except AttributeError as exc:
        # 惰性工厂找不到类名时，报错指向注册表。
        raise AttributeError(
            f"{kind} 后端 {backend!r} 的注册项指向了模块中不存在的类名: {exc}。"
            f"请检查 factory.py 的 _REGISTRY 是否与目标模块里的类名一致。"
        ) from exc


def build_thermo_predictor(backend: str = "heuristic", **kwargs) -> BasePredictor:
    return _build("thermo", backend, **kwargs)


def build_protease_predictor(backend: str = "rule", **kwargs) -> BasePredictor:
    return _build("protease", backend, **kwargs)


def build_affinity_predictor(backend: str = "feature", **kwargs) -> BasePredictor:
    return _build("affinity", backend, **kwargs)


def build_structure_predictor(backend: str = "propensity", **kwargs) -> BasePredictor:
    return _build("structure", backend, **kwargs)


# kind -> 默认后端。build_predictor 不传 backend 时按这里取。
# 与 `RewardConfig` 各 backend 字段的默认值保持一致。
DEFAULT_BACKENDS: dict[str, str] = {
    "thermo": "heuristic",
    "protease": "rule",
    "affinity": "feature",
    "structure": "propensity",
    "plm": "off",
}

_BUILDERS: dict[str, Callable[..., BasePredictor]] = {
    "thermo": build_thermo_predictor,
    "protease": build_protease_predictor,
    "affinity": build_affinity_predictor,
    "structure": build_structure_predictor,
    "plm": build_plm_predictor,
}


def build_predictor(kind: str, backend: str | None = None, **kwargs) -> BasePredictor:
    """按"类别 + 后端"构造预测器的统一入口。

    `backend` 为 None 时使用 `DEFAULT_BACKENDS[kind]`。
    用法::

        build_predictor("thermo", "heuristic")
        build_predictor("affinity")              # 用该类的默认后端
    """
    if kind not in _BUILDERS:
        raise ValueError(
            f"未知的预测器类别: {kind!r}，可选值: {list(PREDICTOR_KINDS)}"
        )
    if backend is None:
        backend = DEFAULT_BACKENDS[kind]
    return _BUILDERS[kind](backend, **kwargs)
