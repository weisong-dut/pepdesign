"""统一日志。

不使用 print 或 basicConfig 的原因
------------------------------------------------
这个项目有三个完全不同的运行形态，日志需求不一样：

1. 离线数据/训练：要带时间戳、级别、模块名，能落文件给后续排查。
2. vLLM 推理服务：多进程（每个 GPU 一个 worker），日志必须能跨进程写同一个
   文件而不交错，并且要能按请求 ID 追踪（否则线上排查一次慢请求要捞全量日志）。
3. CLI 单次运行：要好看的进度输出，不要一堆 DEBUG 噪声。

所以这里提供一个 `get_logger`，按名字去重（同一模块重复调用返回同一个 logger，
避免 handler 被反复添加导致日志重复打印——这是最常见的 Python logging 问题）。
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# 项目根目录（src/pepdesign/utils/logger.py -> 上溯 4 层到项目根）
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
LOG_DIR: Path = PROJECT_ROOT / "outputs" / "logs"

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 记录已配置过的 logger 名，防止重复添加 handler
_CONFIGURED: set[str] = set()

# 环境变量开关：PEPDESIGN_LOG_LEVEL=DEBUG 可以在不改代码的情况下打开调试日志
_DEFAULT_LEVEL = os.environ.get("PEPDESIGN_LOG_LEVEL", "INFO").upper()


def get_logger(name: str, *, to_file: bool = False, level: str | None = None) -> logging.Logger:
    """获取一个配置好的 logger。

    name 通常传 `__name__`，这样日志里能看出是哪个模块打的。

    注意 `propagate = False`：日志树在训练脚本里会挂到 transformers/trl 的
    root logger 上，如果允许向父级传播，同一条日志会被打印两次
    （一次被本模块的 handler，一次被 transformers 的 handler）。
    """
    logger = logging.getLogger(name)
    if name in _CONFIGURED:
        return logger

    logger.setLevel(level or _DEFAULT_LEVEL)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT)

    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if to_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        # 文件名带日期，方便按天归档；训练一次可能跑好几天，单文件会很大
        path = LOG_DIR / f"{datetime.now():%Y%m%d}.log"
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _CONFIGURED.add(name)
    return logger


def set_global_level(level: str) -> None:
    """运行时统一调整日志级别（CLI 的 --verbose 走这里）。"""
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger("pepdesign").setLevel(lvl)
    for name in _CONFIGURED:
        logging.getLogger(name).setLevel(lvl)


def silence_noisy_libraries() -> None:
    """压低第三方库的日志。

    transformers 在加载模型时会打几十行"Some weights were not initialized"之类的
    warning，在批量推理时会淹没应用日志。这是工程上很实际的一条：
    日志要有信噪比，否则真正的异常没人看得到。
    """
    for noisy in (
        "transformers",
        "datasets",
        "accelerate",
        "vllm",
        "urllib3",
        "filelock",
        "httpx",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
