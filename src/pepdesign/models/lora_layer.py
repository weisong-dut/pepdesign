"""从零实现的 LoRA，不依赖 peft。

提供 `LoRALinear`、注入、存取、合并，以及与 peft 的 key 互转；
数值等价性由 `test_manual_lora_matches_peft` 与 peft 对照。

本模块可在未安装 torch 的环境 import：torch 相关的类在
`build_lora_linear_class` 中运行时生成。
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pepdesign.utils.config import LoRAConfig, resolve_path
from pepdesign.utils.logger import get_logger

logger = get_logger(__name__)

#: adapter 目录里配置文件的文件名。与 peft 保持一致，便于互相识别。
ADAPTER_CONFIG_NAME = "adapter_config.json"
#: adapter 权重的文件名。同样是 peft 的约定名。
ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
#: 本模块写出的 adapter_config.json 里 `format` 字段的取值。
#: 加载时靠它区分手写格式与 peft 格式。
FORMAT_TAG = "pepdesign.manual-lora.v1"


# ---------------------------------------------------------------------------
# 核心层
# ---------------------------------------------------------------------------
class LoRALinear:
    """`LoRALinear` 的类型占位；实际类由 `build_lora_linear_class` 运行时生成。"""


def build_lora_linear_class():
    """构造 `LoRALinear` 的 `nn.Module` 子类。

    运行时造类是为了不在模块顶层 `import torch`，保证本包在无 torch 环境下可 import。
    """
    import torch
    from torch import nn

    def _lora_dtype(layer):
        """LoRA 分支的计算精度：取 `lora_A` 权重的 dtype。"""
        return layer.lora_A.weight.dtype

    class LoRALinear(nn.Module):
        r"""挂了 LoRA 分支的线性层：``y = W₀x + (α/r)·B(A(dropout(x)))``。

        base 冻结；`lora_A` 形状 (r, d_in)，`lora_B` 形状 (d_out, r) 且初始化为 0；
        `merged=True` 时前向只用 base，并阻止重复 merge。
        """

        def __init__(
            self,
            base: nn.Linear,
            r: int,
            alpha: int,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            if r < 1:
                raise ValueError(f"LoRA 秩 r 必须 >= 1，收到 {r}")
            if not isinstance(base, nn.Linear):
                raise TypeError(
                    f"LoRALinear 只能包 nn.Linear，收到 {type(base).__name__}"
                )

            #: 手写后端的唯一标记；peft 的 LoraLayer 也有 `lora_A`，
            #: 不能用 `hasattr` 区分，`_is_manual_model` 与 `iter_lora_layers` 依赖它。
            self._pepdesign_manual_lora = True
            self.base = base
            # 冻结基座是 LoRA 定义的一部分；漏掉会静默退化成全量微调。
            # 放在 __init__ 里，调用方无需也无法忘记设置。
            for p in self.base.parameters():
                p.requires_grad = False

            d_in = base.in_features
            d_out = base.out_features
            self.r = r
            self.alpha = alpha
            self.scaling = alpha / r
            self.merged = False

            # LoRA 参数必须与 base 同设备、同精度：新建的 nn.Linear 默认在 CPU/float32。
            # 量化层的 weight.dtype 是 uint8，只在浮点时跟随，量化基座由 forward 对齐。
            ref = getattr(base, "weight", None)
            if ref is not None:
                lora_dtype = ref.dtype if ref.dtype.is_floating_point else torch.float32
                self.lora_A = nn.Linear(d_in, r, bias=False).to(
                    device=ref.device, dtype=lora_dtype
                )
                self.lora_B = nn.Linear(r, d_out, bias=False).to(
                    device=ref.device, dtype=lora_dtype
                )
            else:  # pragma: no cover - nn.Linear 一定有 weight，这里只是兜底
                self.lora_A = nn.Linear(d_in, r, bias=False)
                self.lora_B = nn.Linear(r, d_out, bias=False)
            self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

            self.reset_lora_parameters()

        def reset_lora_parameters(self) -> None:
            """按 LoRA 论文初始化：A 用 Kaiming，B 置零。"""
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

        @property
        def delta_weight(self):
            r"""返回 ``ΔW = (α/r)·B·A``，形状与 `base.weight` 相同。"""
            return self.scaling * (self.lora_B.weight @ self.lora_A.weight)

        def forward(self, x):
            out = self.base(x)
            if self.merged:
                # 已合并：ΔW 已经在 base.weight 里了，不能再加一遍。
                return out
            # 顺序：dropout → A → B → 乘缩放。
            # 输入与输出各做一次精度对齐：QLoRA 下 base 为 4-bit/bf16 而 LoRA 为 fp32；
            # dtype 相同时 `.to()` 零开销。
            lora = self.lora_B(self.lora_A(self.lora_dropout(x).to(_lora_dtype(self))))
            return out + (self.scaling * lora).to(out.dtype)

        def _check_mergeable(self) -> None:
            """合并前检查：拒绝重复 merge 与非浮点（量化）基座。"""
            if self.merged:
                raise RuntimeError(
                    "这个 LoRALinear 已经 merge 过了。重复 merge 会把 ΔW 加两遍，"
                    "而模型不会报错、只是结果悄悄错了。如需重新合并请先 unmerge。"
                )
            weight = getattr(self.base, "weight", None)
            if weight is None or not weight.dtype.is_floating_point:
                raise RuntimeError(
                    "无法把手写 LoRA 合并进量化基座："
                    f"{type(self.base).__name__} 的 weight.dtype="
                    f"{getattr(weight, 'dtype', None)} 不是浮点类型。"
                    "4-bit/8-bit 基座请先 dequantize 再 merge，"
                    "或者直接使用 adapter 推理路径（不合并）。"
                )

        def merge(self) -> None:
            """把 ΔW 精确加进 base.weight。可逆，见 `unmerge`。"""
            self._check_mergeable()
            with torch.no_grad():
                self.base.weight.add_(self.delta_weight)
            self.merged = True

        def unmerge(self) -> None:
            """把 ΔW 从 base.weight 里精确减回去，恢复成"挂 adapter"的形态。"""
            if not self.merged:
                raise RuntimeError("这个 LoRALinear 没有 merge 过，无法 unmerge。")
            with torch.no_grad():
                self.base.weight.sub_(self.delta_weight)
            self.merged = False

        def extra_repr(self) -> str:
            return (
                f"in={self.base.in_features}, out={self.base.out_features}, "
                f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:g}, "
                f"merged={self.merged}"
            )

    return LoRALinear


# ---------------------------------------------------------------------------
# 注入
# ---------------------------------------------------------------------------
def iter_linear_modules(model) -> Iterator[tuple[str, Any]]:
    """遍历 `(模块路径, nn.Linear)`，顺序取自 `named_modules()`。"""
    import torch.nn as nn

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


def find_target_modules(model, target_modules: list[str]) -> list[str]:
    """返回所有以 `target_modules` 中名字结尾的线性层路径。

    采用结尾匹配（peft 约定）：层数多时无法把每条完整路径写进配置。
    返回完整列表，空列表由 `inject_manual_lora` 报错。
    """
    hits: list[str] = []
    for name, _ in iter_linear_modules(model):
        if any(name.endswith(suffix) for suffix in target_modules):
            hits.append(name)
    return hits


def inject_manual_lora(model, cfg: LoRAConfig):
    """把 LoRA 原地注入基座：替换目标线性层并冻结其余参数，返回同一对象。

    `LoRALinear` 持有原 nn.Linear 的引用而不复制权重。
    """
    # 手写后端没有 LoRA bias：配置要求 bias 时明确拒绝，而不是静默忽略。
    if cfg.bias != "none":
        raise NotImplementedError(
            f"手写 LoRA 后端不支持 bias={cfg.bias!r}。"
            "请把 LoRAConfig.bias 设为 'none'，或改用 backend='peft'。"
        )

    LoRALinear = build_lora_linear_class()

    # 重复注入必须拦住：已替换的层不再是 nn.Linear，匹配失败会指向错误的排查方向；
    # 若目标名恰为 base，还会嵌套出两层 LoRA。
    already = [name for name, _ in iter_lora_layers(model)]
    if already:
        raise RuntimeError(
            f"这个模型已经注入过 LoRA 了（找到 {len(already)} 个 LoRALinear，"
            f"例如 {already[0]}）。重复注入会嵌套出两层 LoRA。"
            f"如果确实需要重新注入，请从基座重新加载一个干净的模型。"
        )

    targets = cfg.resolved_target_modules()
    paths = find_target_modules(model, targets)
    if not paths:
        # 匹配不到必须抛错：静默注入 0 个 adapter 会让训练照跑而 loss 为常数。
        available = sorted({n.split(".")[-1] for n, _ in iter_linear_modules(model)})
        raise RuntimeError(
            f"LoRA 的 target_modules={targets} 在该模型里一个都没匹配上。\n"
            f"  模型里实际存在的线性层名（去重后）: {available}\n"
            f"  请检查是不是把 q_proj 写成了 q_projection 之类。"
        )

    # 先收集再替换：边遍历 named_modules 边改模块树是未定义行为。
    to_replace = [(path, dict(model.named_modules())[path]) for path in paths]
    for path, linear in to_replace:
        _set_submodule(
            model,
            path,
            LoRALinear(linear, r=cfg.r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout),
        )

    freeze_non_lora_params(model)

    n_trainable, n_total = count_lora_params(model)
    logger.info(
        "注入 LoRA（手写实现）: r=%d, alpha=%d (缩放 %.2f), 注入 %d 层 %s",
        cfg.r, cfg.lora_alpha, cfg.scaling, len(paths), targets,
    )
    logger.info(
        "可训练参数 %s / 总计 %s (%.4f%%)",
        f"{n_trainable:,}", f"{n_total:,}", 100.0 * n_trainable / max(n_total, 1),
    )
    return model


def _set_submodule(model, path: str, new_module) -> None:
    """按点分路径替换子模块，支持 `ModuleList` 的数字索引。"""
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        # 极小概率：某个被注入的 Linear 直接躺在 ModuleList 里。
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def freeze_non_lora_params(model) -> int:
    """冻结除 `lora_A` / `lora_B` 外的全部参数，返回可训练参数个数。

    Embedding、lm_head 与 norm 未被 LoRA 替换，其 requires_grad 默认为 True，需在此关闭。
    """
    n = 0
    for name, p in model.named_parameters():
        is_lora = name.endswith("lora_A.weight") or name.endswith("lora_B.weight")
        p.requires_grad = is_lora
        if is_lora:
            n += p.numel()
    return n


def count_lora_params(model) -> tuple[int, int]:
    """返回数出来的 `(可训练参数量, 总参数量)`；应与 `lora_trainable_summary` 的估算一致。"""
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return n_trainable, n_total


def iter_lora_layers(model) -> Iterator[tuple[str, Any]]:
    """遍历 `(模块路径, LoRALinear)`；只认 `_pepdesign_manual_lora` 标记。"""
    for name, module in model.named_modules():
        if getattr(module, "_pepdesign_manual_lora", False):
            yield name, module


# ---------------------------------------------------------------------------
# 存 / 取
# ---------------------------------------------------------------------------
def manual_lora_state_dict(model) -> dict:
    """抽出 adapter 权重，key 为相对基座的模块路径。

    `lora_A.weight` 形状 (r, d_in)，`lora_B.weight` 形状 (d_out, r)；
    已 merge 的层拒绝保存。
    """
    state = {}
    for path, layer in iter_lora_layers(model):
        if layer.merged:
            raise RuntimeError(
                f"{path} 已经 merge 进 base.weight 了，此时保存 adapter 会丢掉 ΔW。"
                f"请先 unmerge 再保存。"
            )
        state[f"{path}.lora_A.weight"] = layer.lora_A.weight.detach().cpu()
        state[f"{path}.lora_B.weight"] = layer.lora_B.weight.detach().cpu()
    return state


def save_manual_adapter(model, path: str | Path, cfg: LoRAConfig, tokenizer=None) -> Path:
    """保存 `adapter_model.safetensors` 与 `adapter_config.json`。

    配置记录 r / alpha / target_modules，供加载时恢复超参与校验层；
    存 r 和 alpha 而非 scaling，因为后者无法精确反推。
    """
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "保存 adapter 需要 safetensors：pip install -e '.[train]'"
        ) from exc

    out = resolve_path(path)
    out.mkdir(parents=True, exist_ok=True)

    state = manual_lora_state_dict(model)
    save_file(state, str(out / ADAPTER_WEIGHTS_NAME))

    config = {
        "format": FORMAT_TAG,
        "r": cfg.r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "target_modules": cfg.resolved_target_modules(),
        "scaling": cfg.scaling,
        "bias": cfg.bias,
        "task_type": cfg.task_type,
        "inference_mode": False,
        "num_lora_modules": len({k.rsplit(".lora_", 1)[0] for k in state}),
    }
    # 基座名只用于复现；模型没有 `_name_or_path`（如冒烟小模型）时跳过，不写空值。
    base_name = getattr(getattr(model, "config", None), "_name_or_path", None)
    if base_name:
        config["base_model_name_or_path"] = base_name
    with open(out / ADAPTER_CONFIG_NAME, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    if tokenizer is not None:
        # 一并保存 tokenizer，避免下游复现时无法确定当时的 tokenizer 版本。
        tokenizer.save_pretrained(str(out))

    logger.info("adapter 已保存（手写格式）: %s（%d 个模块）", out, config["num_lora_modules"])
    return out


def load_manual_adapter(model, path: str | Path, *, is_trainable: bool = False):
    """在已注入的骨架上加载手写格式 adapter，`is_trainable` 默认为 False。

    两个方向都显式设置 requires_grad；`is_trainable=True` 时断言可训练参数非零。
    """
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "加载 adapter 需要 safetensors：pip install -e '.[train]'"
        ) from exc
    import torch

    p = resolve_path(path)
    weights = p / ADAPTER_WEIGHTS_NAME
    config_path = p / ADAPTER_CONFIG_NAME
    if not weights.exists():
        raise FileNotFoundError(
            f"adapter 权重不存在: {weights}。"
            f"GRPO 阶段默认从 SFT 的产出继续训（GRPOConfig.sft_adapter_path），"
            f"如果还没跑过 SFT，请先把该字段改成本地已有的 adapter 路径。"
        )

    cfg_dict = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            cfg_dict = json.load(f)
        if cfg_dict.get("format") != FORMAT_TAG:
            raise ValueError(
                f"{config_path} 的 format 是 {cfg_dict.get('format')!r}，"
                f"不是本模块写出的 {FORMAT_TAG!r}。"
                f"这多半是 peft 存出来的 adapter——请用 peft 后端加载"
                f"（LoRAConfig.backend='peft'），或者用 from_peft_state_dict 转换。"
            )
    else:
        logger.warning(
            "%s 不存在：无法校验/恢复保存时的 r、alpha、dropout，"
            "本次将按当前骨架的超参加载。",
            config_path,
        )

    # 先注入再灌权重：按 key 现场建层会绕过 target_modules 校验。
    by_path = dict(iter_lora_layers(model))
    if not by_path:
        raise RuntimeError(
            "当前模型没有手写 LoRA 层，无法加载手写格式 adapter。"
            "请先用 inject_lora(..., backend='manual') 注入骨架；"
            "peft 格式的 adapter 请用 backend='peft' 加载。"
        )
    if any(layer.merged for layer in by_path.values()):
        raise RuntimeError(
            "当前模型已经 merge，LoRA 分支被跳过，此时灌 adapter 权重不会生效。"
            "请先 unmerge 或重新加载一个干净的基座。"
        )

    # ---- 1) 用 adapter_config.json 校验/恢复运行时超参 ----
    # alpha 必须从配置恢复：r 不一致会被形状检查拦住，alpha 不一致则只表现为
    # 输出缩放错误，loss 照常下降。
    first = next(iter(by_path.values()))
    saved_r = cfg_dict.get("r")
    if saved_r is not None and int(saved_r) != first.r:
        raise ValueError(
            f"adapter_config.json 的 r={saved_r}，但当前模型的 LoRA r={first.r}。"
            "请用与保存时一致的 r 重新注入，或加载对应版本的 adapter。"
        )

    saved_alpha = cfg_dict.get("lora_alpha")
    if saved_alpha is not None:
        for layer in by_path.values():
            layer.alpha = int(saved_alpha)
            layer.scaling = layer.alpha / layer.r

    saved_dropout = cfg_dict.get("lora_dropout")
    if saved_dropout is not None:
        p_drop = float(saved_dropout)
        for layer in by_path.values():
            if isinstance(layer.lora_dropout, torch.nn.Dropout):
                layer.lora_dropout.p = p_drop
            elif p_drop > 0:
                # 骨架是 Identity 但保存时带 dropout：恢复成 Dropout，
                # 否则 GRPO 续训会少一层正则，而配置上完全看不出来。
                layer.lora_dropout = torch.nn.Dropout(p_drop)

    saved_n = cfg_dict.get("num_lora_modules")
    if saved_n is not None and int(saved_n) != len(by_path):
        raise ValueError(
            f"adapter_config.json 记录 num_lora_modules={saved_n}，"
            f"但当前模型注入了 {len(by_path)} 层。"
            "请检查注入时用的 target_modules 是否与保存时一致。"
        )

    # ---- 2) 权重文件的 key 必须与当前骨架逐一对应 ----
    # 缺 key 时那一层会保留随机 A，产生不报错的随机 ΔW，所以两个方向都校验。
    state = load_file(str(weights))
    expected = set()
    for module_path in by_path:
        expected.add(f"{module_path}.lora_A.weight")
        expected.add(f"{module_path}.lora_B.weight")
    got = set(state)
    missing = sorted(expected - got)
    unexpected = sorted(got - expected)
    if missing or unexpected:
        detail = []
        if missing:
            detail.append(f"缺少 {len(missing)} 个: {missing[:5]}")
        if unexpected:
            detail.append(f"多出 {len(unexpected)} 个: {unexpected[:5]}")
        raise ValueError(
            "adapter 权重与当前模型的 LoRA 层不匹配（"
            + "；".join(detail)
            + "）。请确认 adapter_config.json 里的 r/target_modules 与当前注入一致。"
        )

    for key, tensor in state.items():
        module_path, _, param = key.rpartition(".lora_")
        layer = by_path[module_path]
        if param == "A.weight":
            target = layer.lora_A
        elif param == "B.weight":
            target = layer.lora_B
        else:  # pragma: no cover - expected 集合已经排除了其他形态
            raise ValueError(f"无法识别的 adapter key: {key}")
        if target.weight.shape != tensor.shape:
            raise ValueError(
                f"{key} 的形状是 {tuple(tensor.shape)}，"
                f"但模型里对应的层是 {tuple(target.weight.shape)}。"
                f"大概率是 adapter_config.json 里的 r 或 target_modules 与当前模型不一致。"
            )
        with torch.no_grad():
            target.weight.copy_(tensor)

    n_loaded = len(by_path)
    logger.info("adapter 已加载（手写格式）: %s（%d 个模块，可训练=%s）",
                p, n_loaded, is_trainable)

    # 两个方向都要显式设 requires_grad：注入时 A/B 默认可训练，
    # is_trainable=False 时若不冻住，推理中的 backward 会悄悄改掉 adapter。
    for p_ in model.parameters():
        p_.requires_grad = False
    for _, layer in iter_lora_layers(model):
        layer.lora_A.weight.requires_grad = is_trainable
        layer.lora_B.weight.requires_grad = is_trainable

    n_trainable, _ = count_lora_params(model)
    if is_trainable and n_trainable == 0:
        raise RuntimeError(
            f"以 is_trainable=True 加载了 {p}，但没有任何可训练参数。"
            f"检查 adapter 文件是否完整、inference_mode 是否为 false。"
        )
    if not is_trainable and n_trainable != 0:
        raise RuntimeError(
            f"以 is_trainable=False 加载了 {p}，但仍存在 {n_trainable} 个可训练参数。"
            f"推理语义下 adapter 必须是冻住的。"
        )
    logger.info("可训练参数 %s（is_trainable=%s）", f"{n_trainable:,}", is_trainable)

    return model


def merge_manual_lora(model) -> int:
    """把所有 LoRALinear 的 ΔW 加进 base.weight，返回合并层数。

    先整体检查再逐层合并，避免第 N 层失败时留下无法回滚的半合并状态。
    """
    layers = list(iter_lora_layers(model))
    for _, layer in layers:
        layer._check_mergeable()
    for _, layer in layers:
        layer.merge()
    logger.info("LoRA 已合并进基座：%d 层", len(layers))
    return len(layers)


def unmerge_manual_lora(model) -> int:
    """`merge_manual_lora` 的逆操作，精确还原。"""
    n = 0
    for _, layer in iter_lora_layers(model):
        layer.unmerge()
        n += 1
    logger.info("LoRA 已从基座拆出：%d 层", n)
    return n


def unwrap_manual_lora(model) -> int:
    """把已合并的 `LoRALinear` 替换成普通 `nn.Linear`，返回替换层数。

    不替换时 save_pretrained 会写出 `.base.weight` 等 key，标准加载会将目标层
    静默重新随机初始化。要求所有层都已 merge。
    """
    import torch
    from torch import nn

    layers = list(iter_lora_layers(model))
    not_merged = [path for path, layer in layers if not layer.merged]
    if not_merged:
        raise RuntimeError(
            f"有 {len(not_merged)} 层 LoRALinear 还没有 merge（例如 {not_merged[0]}），"
            "不能直接 unwrap。请先调用 merge_manual_lora()。"
        )

    for path, layer in layers:
        base = layer.base
        weight = getattr(base, "weight", None)
        if weight is None or not weight.dtype.is_floating_point:
            raise RuntimeError(
                f"{path} 的 base.weight.dtype={getattr(weight, 'dtype', None)} 不是浮点类型，"
                "无法导出成标准模型；4-bit/8-bit 基座请先 dequantize。"
            )

        merged = nn.Linear(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
        ).to(device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            merged.weight.copy_(weight)
            if base.bias is not None:
                merged.bias.copy_(base.bias)
        # 合并模型是推理产物：默认冻住，避免有人在导出后又无意地全参训练。
        merged.weight.requires_grad = False
        if merged.bias is not None:
            merged.bias.requires_grad = False
        merged.train(base.training)
        _set_submodule(model, path, merged)

    logger.info("LoRA 层已 unwrap 成普通 Linear：%d 层", len(layers))
    return len(layers)


# ---------------------------------------------------------------------------
# 与 peft 的格式互转
# ---------------------------------------------------------------------------
#: peft 的 key 前缀：PeftModel 包装后路径为 `base_model.model.` + 基座内部路径，
#: 因此完整 key 中会出现两次 `model`。
PEFT_KEY_PREFIX = "base_model.model."
#: peft 在 `lora_A` / `lora_B` 与 `weight` 之间插入的 adapter 名字段，默认 `default`。
PEFT_ADAPTER_NAME = "default"


def to_peft_state_dict(state: dict, *, adapter_name: str = PEFT_ADAPTER_NAME) -> dict:
    """把本模块的 key 命名转换为 peft 格式。

    peft 会加 `base_model.model.` 前缀，并在 lora_A/B 与 weight 之间插入 adapter 名。
    """
    out = {}
    for key, value in state.items():
        module_path, _, tail = key.rpartition(".lora_")
        out[f"{PEFT_KEY_PREFIX}{module_path}.lora_{tail.rsplit('.', 1)[0]}"
            f".{adapter_name}.weight"] = value
    return out


def from_peft_state_dict(state: dict) -> dict:
    """`to_peft_state_dict` 的逆；缺少前缀或 adapter 名段的 key 原样保留。

    形状校验由 `load_manual_adapter` 负责。
    """
    out = {}
    for key, value in state.items():
        k = key[len(PEFT_KEY_PREFIX):] if key.startswith(PEFT_KEY_PREFIX) else key
        module_path, sep, tail = k.rpartition(".lora_")
        if sep:
            # tail 形如 `A.default.weight` 或 `A.weight`
            parts = tail.split(".")
            if len(parts) == 3:
                k = f"{module_path}.lora_{parts[0]}.{parts[2]}"
        out[k] = value
    return out


def export_peft_adapter(
    src_dir: str | Path,
    out_dir: str | Path,
    *,
    base_model_name_or_path: str,
) -> Path:
    """把手写格式的 adapter 目录导出成 peft 格式（vLLM 只认后者）。

    写出 peft schema 的 adapter_config.json。导出到另一个目录，
    两种格式各自保留给训练侧与外部工具消费。
    """
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "导出 peft adapter 需要 safetensors：pip install -e '.[train]'"
        ) from exc

    src = resolve_path(src_dir)
    weights = src / ADAPTER_WEIGHTS_NAME
    if not weights.exists():
        raise FileNotFoundError(
            f"没有找到 {weights}。这个函数要求源目录是 `save_manual_adapter` "
            f"（或训练脚本）写出的手写格式 adapter。"
        )

    cfg_path = src / ADAPTER_CONFIG_NAME
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"没有找到 {cfg_path}。没有它就无法确定 r / alpha / target_modules，"
            f"导出出来的 peft 配置只能靠猜——而猜错的形状在 vLLM 上是加载期报错，"
            f"在手写后端上是无声挂错层。"
        )
    src_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if src_cfg.get("format") != FORMAT_TAG:
        raise ValueError(
            f"{cfg_path} 的 format 是 {src_cfg.get('format')!r}，不是 {FORMAT_TAG!r}；"
            f"这份 adapter 可能本来就是 peft 格式的，不需要再导出一次。"
        )

    out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    state = load_file(str(weights))
    save_file(to_peft_state_dict(state), str(out / ADAPTER_WEIGHTS_NAME))

    peft_cfg = {
        "peft_type": "LORA",
        "task_type": src_cfg.get("task_type", "CAUSAL_LM"),
        "r": src_cfg["r"],
        "lora_alpha": src_cfg["lora_alpha"],
        "lora_dropout": src_cfg["lora_dropout"],
        "bias": src_cfg.get("bias", "none"),
        "target_modules": src_cfg["target_modules"],
        "base_model_name_or_path": base_model_name_or_path,
        # 导出物是给推理用的，标成 True 免得下游误以为还能接着训；
        # 要接着训请用原始的手写格式目录。
        "inference_mode": True,
    }
    with open(out / ADAPTER_CONFIG_NAME, "w", encoding="utf-8") as f:
        json.dump(peft_cfg, f, indent=2, ensure_ascii=False)

    logger.info(
        "已导出 peft 格式 adapter：%s -> %s（%d 个张量）", src, out, len(state)
    )
    return out


def describe_lora_config(cfg: LoRAConfig) -> dict[str, Any]:
    """返回 r、alpha、缩放、目标模块与后端的摘要字典。"""
    return {
        "r": cfg.r,
        "lora_alpha": cfg.lora_alpha,
        "scaling": round(cfg.scaling, 4),
        "target_modules": cfg.resolved_target_modules(),
        "backend": cfg.backend,
    }
