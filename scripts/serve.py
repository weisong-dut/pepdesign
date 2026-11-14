"""推理服务入口（OpenAI 兼容接口）。

用法::

    python scripts/serve.py --config configs/serve.yaml
    python scripts/serve.py --config configs/serve.yaml --dry-run
    python scripts/serve.py --config configs/serve.yaml --stub

`--dry-run` 只校验配置、量化决策、adapter 目录与路由表，不加载权重。
`--stub` 使用假引擎返回带明确标记的占位文本，仅用于验证 HTTP 层。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pepdesign.utils.config import load_config
from pepdesign.utils.logger import get_logger, silence_noisy_libraries

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="多肽设计推理服务（OpenAI 兼容接口）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default="configs/serve.yaml", help="YAML 配置文件路径")
    p.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="覆盖配置项，可重复。例：--set infer.port=8080",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="只校验配置与路由，不加载模型")
    p.add_argument("--stub", action="store_true",
                   help="用假引擎起服务（不加载权重），只验证 HTTP 层")
    return p.parse_args(argv)


def dry_run(cfg) -> int:
    """不加载模型的检查。返回进程退出码。"""
    from pepdesign.serve.schemas import SUPPORTED_SAMPLING, UNSUPPORTED_PARAMS

    infer = cfg.infer
    problems: list[str] = []

    logger.info("后端：%s", infer.backend)
    if infer.backend == "vllm":
        # vLLM 缺包或缺卡的问题会在 load() 时才暴露，这里提前到 dry-run 检查。
        from pepdesign.serve.engine import _require_vllm

        try:
            _require_vllm()
            logger.info("vLLM 环境检查通过（包已安装且有可用 CUDA 设备）")
        except (ImportError, RuntimeError) as exc:
            problems.append(f"backend='vllm' 但环境不满足：{exc}")
    logger.info("基座：%s", infer.base_model)
    logger.info("adapter：%s", infer.adapter_path or "（无，将以未微调的基座运行）")
    if infer.merged_model_path:
        logger.info("合并模型：%s", infer.merged_model_path)

    if infer.adapter_path and not Path(infer.adapter_path).exists():
        problems.append(
            f"adapter_path 不存在: {infer.adapter_path}。"
            f"先跑 SFT/GRPO，或把它改成已有 adapter 的路径 / 留空。"
        )
    if infer.merged_model_path and not Path(infer.merged_model_path).exists():
        problems.append(f"merged_model_path 不存在: {infer.merged_model_path}")

    logger.info(
        "成批：max_num_seqs=%d, max_num_batched_tokens=%d, "
        "batch_timeout_ms=%d, batch_size=%d",
        infer.max_num_seqs, infer.max_num_batched_tokens,
        infer.batch_timeout_ms, infer.batch_size,
    )
    if infer.batch_size > infer.max_num_seqs:
        # batch_size 是调度意图、max_num_seqs 是硬上限；前者大于后者时不会生效。
        logger.warning(
            "batch_size (%d) > max_num_seqs (%d)：成批时永远先撞到后者，"
            "前者不会生效。想让批更大请调 max_num_seqs。",
            infer.batch_size, infer.max_num_seqs,
        )

    logger.info("已实现的采样参数：%s", sorted(SUPPORTED_SAMPLING))
    logger.info("未实现（传了会返回 400）：%s", sorted(UNSUPPORTED_PARAMS))

    if problems:
        for msg in problems:
            logger.error("%s", msg)
        return 1

    logger.info("dry-run 通过。去掉 --dry-run 启动服务（需要 torch + fastapi）。")
    return 0


class _StubEngine:
    """不加载权重的假引擎，仅用于验证 HTTP 层。

    实现 app 所需的 `InferenceEngine` 接口，返回带明确 stub 标记的占位文本。
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.tokenizer = None
        self._adapter = cfg.adapter_path

    @property
    def is_ready(self) -> bool:
        return True

    def load(self) -> None:
        logger.warning("--stub：未加载任何模型权重，返回的是占位文本。")

    def generate_batch(self, prompts, params):
        from pepdesign.serve.engine import GenOutput

        return [
            GenOutput(
                texts=[f"[stub 输出] 未加载模型，这条不对应任何真实生成结果。"
                       f"prompt 长度 {len(p)} 字符。" for _ in range(params.n)],
                n_prompt_tokens=len(p),
                n_completion_tokens=[8] * params.n,
            )
            for p in prompts
        ]

    def list_adapters(self) -> list[str]:
        return []

    def set_adapter(self, path: str | None) -> None:
        self._adapter = path

    def health(self) -> dict:
        return {
            "status": "ok", "backend": "stub", "model": self.cfg.base_model,
            "adapter": self._adapter, "device": "cpu", "dtype": "float32",
            "max_model_len": self.cfg.max_model_len,
            "detail": "--stub 模式：未加载真实模型，输出是占位文本",
        }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    silence_noisy_libraries()
    cfg = load_config(args.config, overrides=args.overrides)

    if args.dry_run:
        return dry_run(cfg)

    try:
        import uvicorn
    except ImportError:
        logger.error(
            "启动服务需要 fastapi + uvicorn：\n"
            "  pip install -e '.[infer]'\n"
            "只想校验配置的话用 --dry-run（不需要这两个包）。"
        )
        return 2

    from pepdesign.serve.app import create_app

    # `--stub` 的假引擎须在构造 app 前注入；建完再替换会让"是否加载"
    # 依赖构造时序，难以判断。
    engine = _StubEngine(cfg.infer) if args.stub else None
    app = create_app(cfg.infer, lora_cfg=cfg.lora, engine=engine)

    logger.info("服务启动：http://%s:%d（接口文档在 /docs）", cfg.infer.host, cfg.infer.port)
    uvicorn.run(app, host=cfg.infer.host, port=cfg.infer.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
