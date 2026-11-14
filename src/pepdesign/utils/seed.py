"""随机种子与可复现性。

固定随机源的原因
--------------------
同一个 prompt 在 GRPO 中会采样多条序列；如果两次采样不同，
两次实验的曲线无法直接对比，也就无法区分"改进"与"随机波动"。

需要同时固定的随机源（少一个都会不可复现）：
  * Python `random`（数据打乱）
  * NumPy（数据增强、采样）
  * torch CPU / CUDA（权重初始化、dropout）
  * CUDA 的 cuBLAS 算法选择（`CUBLAS_WORKSPACE_CONFIG`）

最后一项最容易被忽略：cuBLAS 默认会用 benchmark 模式挑最快的矩阵乘算法，
而不同算法有不同数值误差，两次运行的浮点结果就会分叉。
"""

from __future__ import annotations

import contextlib
import os
import random

from pepdesign.utils.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42, *, deterministic: bool = True) -> None:
    """固定所有随机源。

    参数
    ----
    seed : 随机种子
    deterministic : 是否开启严格确定性。

    代价说明
    --------
    `deterministic=True` 会关闭 cuDNN 的自动调优（autotune），
    并且当某个算子没有确定性实现时直接报错而不是静默用非确定性版本。
    这会让训练慢 5%-15%，同时在开启了梯度累积 + 变长序列的场景下，
    某些 reduce 算子可能触发 "no deterministic implementation" 报错。

    工程上的做法是分场景：
      * 小规模复现实验 / 排障：deterministic=True，宁可慢也要能对齐。
      * 大规模正式训练：deterministic=False，但固定 seed，
        接受"同 seed 也不完全 bit-level 一致"，
        通过多次运行取均值 ± 方差的方式做实验对比（这也是论文的常规做法）。
    """
    # 1) Python 内置
    random.seed(seed)
    # 2) NumPy
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    # 3) PyTorch（可选依赖，纯 CPU 环境下没装也能正常跑）
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # PyTorch 1.8+ 的严格确定性开关，遇到非确定性算子会直接抛错
            #
            # 只吞这两种异常，且都指向同一件事——当前这个 torch 没这个 API：
            #   AttributeError  函数本身不存在（<1.8）
            #   TypeError       warn_only 这个关键字参数还不存在（1.8–1.9）
            # 它们是版本差异，不是运行期故障，跳过即可。
            # 原先写的是 `except Exception: pass`，那会连"CUBLAS 没配好"
            # 这类真问题一起吞掉——而确定性开关失效恰恰是最不该被静默的一类。
            with contextlib.suppress(AttributeError, TypeError):  # pragma: no cover
                torch.use_deterministic_algorithms(True, warn_only=True)
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        logger.debug("未安装 torch，跳过 torch 随机源固定")

    # 4) cuBLAS 确定性要求必须通过环境变量传递，且必须在 CUDA 初始化之前设置
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # 5) 让 tokenizers 的多进程不要互相抢核导致死锁（训练脚本里的常见问题）
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    logger.info("随机种子已固定: seed=%d, deterministic=%s", seed, deterministic)


def get_rng(seed: int | None = None) -> random.Random:
    """获取一个独立的 Random 实例，而不是用全局的 random。

    用途：数据切分、负采样这类操作如果用全局 random，
    顺序会被其他代码的调用干扰（谁先调用谁先消耗随机数），
    导致"同样的 seed 却切分出不同的数据集"。
    独立实例把随机流的边界划清楚。
    """
    return random.Random(seed)
