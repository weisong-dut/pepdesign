"""离线评估：将奖励预测器的预测与实验实测值对比。

用法::

    python scripts/eval_offline.py
    python scripts/eval_offline.py --set eval.records_path=data/processed/test_records.jsonl
    python scripts/eval_offline.py --no-write

不需要 GPU 或 torch，仅依赖 numpy / pyyaml / pydantic / orjson。
`conclusive=False`（样本量不足）时退出码仍为 0：这是数据现状而非失败；
数据文件读不到、预测器构造失败等真实错误仍照常抛错。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许直接以脚本方式运行（把 src 加入 sys.path）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pepdesign.eval.reward_audit import (
    audit_reward_predictors,
    summarize_audit,
)
from pepdesign.utils.config import load_config, resolve_path
from pepdesign.utils.io import read_jsonl, write_json
from pepdesign.utils.logger import get_logger, silence_noisy_libraries

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="离线评估：奖励预测器 vs 实验实测值",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default=None, help="可选的 YAML 配置（默认用代码里的默认值）")
    p.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="覆盖配置项，可重复。例：--set eval.min_n_for_significance=10",
    )
    p.add_argument("--no-write", action="store_true", help="只打印，不落盘报告")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    silence_noisy_libraries()
    cfg = load_config(args.config, overrides=args.overrides)

    path = resolve_path(cfg.eval.records_path)
    if not path.exists():
        logger.error("记录文件不存在：%s", path)
        logger.error(
            "评估读的是数据构建的产物。请先跑：python -m pepdesign.data.pipeline"
        )
        return 2

    records = read_jsonl(path)
    report = audit_reward_predictors(
        records,
        min_n=cfg.eval.min_n_for_significance,
        n_perm=cfg.eval.n_permutations,
        seed=cfg.eval.seed,
    )
    report["source"] = str(path)

    print(summarize_audit(report))
    for caveat in report["all_caveats"]:
        # Windows 控制台默认 GBK，无法编码 U+26A0 等字符，print 会抛
        # UnicodeEncodeError，故输出到终端的内容不使用这类符号。
        print(f"  注意：{caveat}")

    if not args.no_write:
        out = write_json(resolve_path(cfg.eval.report_path), report)
        logger.info("报告已落盘：%s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
