"""数据层：采集 -> 清洗 -> 切分 -> 语料构造。

子模块：schema / download / clean / split / build_sft / build_grpo /
build_calibration / pipeline。pipeline 负责总编排，一条命令产出全部语料。
"""

from pepdesign.data import (
    build_calibration,
    build_grpo,
    build_sft,
    clean,
    download,
    pipeline,
    schema,
    split,
)

__all__ = [
    "build_calibration",
    "build_grpo",
    "build_sft",
    "clean",
    "download",
    "pipeline",
    "schema",
    "split",
]
