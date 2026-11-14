"""数据构建总编排：一条命令从原始数据走到可训练语料。

流程：采集 → 清洗 → 同源切分 → SFT 语料 → GRPO prompts → 校准集 → 质量报告。
最终产出 dataset_card.json，记录数据来源、样本量、丢弃统计与关键超参。
除采集外均为确定性变换，同一 seed 可复现。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

from pepdesign.data.build_calibration import (
    build_calibration_pairs,
    calibration_to_warmup_samples,
)
from pepdesign.data.build_grpo import build_grpo_dataset
from pepdesign.data.build_sft import (
    build_denovo_index,
    build_denovo_sample,
    build_knowledge_samples,
    build_multi_objective_sample,
    build_stabilize_sample,
    cyclic_balanced_pool,
)
from pepdesign.data.clean import run_cleaning
from pepdesign.data.download import collect_all
from pepdesign.data.schema import PeptideRecord
from pepdesign.data.split import split_by_clusters, verify_no_leakage
from pepdesign.utils.config import DataConfig, load_config, resolve_path
from pepdesign.utils.io import read_jsonl, write_json, write_jsonl
from pepdesign.utils.logger import get_logger
from pepdesign.utils.seed import get_rng

logger = get_logger(__name__)

# SFT 三种任务的比例。与 GRPO 的 TASK_MIX 不同：SFT 侧重格式与完整设计流程，
# GRPO 侧重稳定性优化；knowledge 池只有少量问答，故占比压到 1%。
SFT_TASK_MIX: dict[str, float] = {
    "de_novo": 0.52,
    "stabilize": 0.37,
    "multi_objective": 0.10,
    "knowledge": 0.01,
}


@dataclass
class DatasetCard:
    """数据集版本清单。跟着模型一起归档。"""

    created_at: str = ""
    raw_count: int = 0
    cleaned_count: int = 0
    sft_count: int = 0
    grpo_count: int = 0
    calibration_count: int = 0
    split_summary: dict = field(default_factory=dict)
    leakage_check: dict = field(default_factory=dict)
    task_distribution: dict = field(default_factory=dict)
    quality_report: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "created_at": self.created_at,
            "counts": {
                "raw": self.raw_count,
                "cleaned": self.cleaned_count,
                "sft": self.sft_count,
                "grpo": self.grpo_count,
                "calibration": self.calibration_count,
            },
            "split": self.split_summary,
            "leakage_check": self.leakage_check,
            "task_distribution": self.task_distribution,
            "quality_report": self.quality_report,
            "config": self.config,
        }


# ---------------------------------------------------------------------------
# 载入原始数据
# ---------------------------------------------------------------------------
def load_raw_records(cfg: DataConfig) -> list[PeptideRecord]:
    """把 data/raw/ 下所有 JSONL 读成 PeptideRecord。

    采集阶段已把各种格式统一为 JSONL，下游只需处理一种输入格式。
    """
    raw_dir = resolve_path(cfg.raw_dir)
    records: list[PeptideRecord] = []
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        logger.warning(
            "在 %s 下没有找到任何 jsonl 数据文件。请先运行数据采集"
            "（python -m pepdesign.data.pipeline --download），"
            "或把已有语料放入该目录。",
            raw_dir,
        )
        return []

    for path in files:
        for item in read_jsonl(path):
            try:
                records.append(PeptideRecord.from_dict(item))
            except Exception as exc:  # noqa: BLE001 - 单条坏数据不该中断整批
                logger.warning("跳过无法解析的记录（%s）: %s", path.name, exc)
        logger.info("载入 %s：累计 %d 条", path.name, len(records))
    return records


# ---------------------------------------------------------------------------
# SFT 语料
# ---------------------------------------------------------------------------
def augment_cyclic_variants(
    records: list[PeptideRecord], *, seed: int = 42, per_record: int = 3
) -> list[PeptideRecord]:
    """对可行环肽做保守替换变体，扩充环肽池。

    变体不改变 Cys 数量与位置，环化可行性不变；标记为 source="synthetic"。
    切分放在变体扩充之后，同源聚类会把变体与母序列分到同一簇。
    """
    import random as _random

    from pepdesign.bio.cyclization import feasible_cyclic_type
    from pepdesign.data.build_sft import conservative_variant

    if per_record <= 0:
        return list(records)
    rng = _random.Random(seed + 7331)
    out = list(records)
    seen = {r.sequence for r in records}
    added = 0
    for rec in records:
        if feasible_cyclic_type(rec.sequence, rec.cyclic_type) == "none":
            continue
        for _ in range(per_record):
            variant = conservative_variant(rec.sequence, rng)
            if variant is None or variant in seen:
                continue
            seen.add(variant)
            new = replace(
                rec,
                sequence=variant,
                name=f"{rec.name} 变体",
                source="synthetic",
                source_id=f"{rec.source_id}:var{added}",
                properties={},
                activities=[],
                lineage={
                    **rec.lineage,
                    "parent": rec.source_id,
                    "kind": "conservative_variant",
                },
            )
            new.compute_properties()
            out.append(new)
            added += 1
    logger.info("环肽保守变体：新增 %d 条（%d → %d）", added, len(records), len(out))
    return out


def build_sft_corpus(
    records: list[PeptideRecord],
    *,
    total: int = 8000,
    seed: int = 42,
    cyclic_ratio: float = 0.30,
) -> list[dict]:
    """按任务比例混合出 SFT 语料，某类不足时会从头部补齐以维持总量。

    `cyclic_ratio` 控制过采样后的环肽记录占比，见 `cyclic_balanced_pool`。
    """
    rng = get_rng(seed)
    pool = cyclic_balanced_pool(records, cyclic_ratio=cyclic_ratio)
    parts: dict[str, list[dict]] = {}

    # --- 从头设计（约束先行 + 库内检索）---
    n_denovo = int(total * SFT_TASK_MIX["de_novo"])
    index = build_denovo_index(pool)
    denovo: list[dict] = []
    attempts = 0
    max_attempts = n_denovo * 5
    while len(denovo) < n_denovo and attempts < max_attempts:
        attempts += 1
        sample = build_denovo_sample(index, rng)
        if sample is not None:
            denovo.append(sample)
    if len(denovo) < n_denovo:
        logger.warning(
            "de_novo 只构造出 %d/%d 条：采样约束在索引里找不到足够候选。"
            "训练步数按总量配置，这里少样本会让实际步数与计划不一致。",
            len(denovo), n_denovo,
        )
    parts["de_novo"] = denovo

    # --- 稳定性优化（需要劣化成功，可能凑不满）---
    # 记录按打乱后的顺序轮转使用，而不是独立随机抽取：
    # 轮转保证每条记录使用次数相差不超过 1，降低 prompt 重复。
    n_stab = int(total * SFT_TASK_MIX["stabilize"])
    stab_order = list(pool)
    rng.shuffle(stab_order)
    stab_cursor = 0
    stab: list[dict] = []
    attempts = 0
    while len(stab) < n_stab and attempts < n_stab * 10:
        attempts += 1
        rec = stab_order[stab_cursor % len(stab_order)]
        stab_cursor += 1
        built = build_stabilize_sample(rec, rng)
        if built is not None:
            stab.append(built[0])
    parts["stabilize"] = stab

    # --- 多目标 ---
    n_multi = int(total * SFT_TASK_MIX["multi_objective"])
    multi_order = list(pool)
    rng.shuffle(multi_order)
    multi_cursor = 0
    multi: list[dict] = []
    attempts = 0
    while len(multi) < n_multi and attempts < n_multi * 10:
        attempts += 1
        rec = multi_order[multi_cursor % len(multi_order)]
        multi_cursor += 1
        built = build_multi_objective_sample(rec, rng)
        if built is not None:
            multi.append(built[0])
    parts["multi_objective"] = multi

    # --- 知识问答（固定 5 条，重复到目标数量）---
    n_know = int(total * SFT_TASK_MIX["knowledge"])
    know_pool = build_knowledge_samples()
    parts["knowledge"] = [
        know_pool[i % len(know_pool)] for i in range(n_know)
    ] if know_pool else []

    merged = [s for sub in parts.values() for s in sub]
    rng.shuffle(merged)

    from collections import Counter

    dist = Counter(s["task_type"] for s in merged)
    logger.info("SFT 语料构造完成：%d 条，任务分布 %s", len(merged), dict(dist))
    return merged


# ---------------------------------------------------------------------------
# 总流程
# ---------------------------------------------------------------------------
def run_pipeline(
    cfg: DataConfig | None = None,
    *,
    download: bool = False,
    sft_size: int | None = None,
    grpo_size: int | None = None,
    calibration_size: int | None = None,
    seed: int = 42,
) -> DatasetCard:
    """执行完整的数据构建流程。

    `download=False` 时使用本地已有数据；规模参数为 None 时从 DataConfig 读取。
    """
    cfg = cfg or DataConfig()
    sft_size = cfg.sft_size if sft_size is None else sft_size
    grpo_size = cfg.grpo_size if grpo_size is None else grpo_size
    calibration_size = (
        cfg.calibration_size if calibration_size is None else calibration_size
    )
    card = DatasetCard(created_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    card.config = cfg.model_dump(mode="json")

    # ---- 1. 采集（可选）----
    if download:
        logger.info("=== 步骤 1/6：数据采集 ===")
        collect_all(cfg)
    else:
        logger.info("=== 步骤 1/6：跳过采集（使用本地已有数据）===")

    # ---- 2. 载入 + 清洗 ----
    logger.info("=== 步骤 2/6：清洗与去重 ===")
    raw_records = load_raw_records(cfg)
    if not raw_records:
        logger.error("没有可用数据，流程终止")
        return card
    card.raw_count = len(raw_records)

    cleaned, report = run_cleaning(raw_records, cfg)
    card.cleaned_count = len(cleaned)
    card.quality_report = report.to_dict()
    if not cleaned:
        logger.error("清洗后没有剩余数据，流程终止")
        return card

    interim_dir = resolve_path(cfg.interim_dir)
    write_jsonl(interim_dir / "cleaned.jsonl", (r.to_dict() for r in cleaned))
    write_json(interim_dir / "quality_report.json", report.to_dict())
    (interim_dir / "quality_report.md").write_text(report.to_markdown(), encoding="utf-8")

    # ---- 2.5 环肽保守变体扩充 ----
    # 放在切分之前：变体与母序列同源，聚类会把它们分到同一簇，
    # 不会出现 train/valid 泄漏。
    cleaned = augment_cyclic_variants(cleaned, seed=seed)
    write_jsonl(
        interim_dir / "cleaned_augmented.jsonl", (r.to_dict() for r in cleaned)
    )

    # ---- 3. 同源切分 ----
    logger.info("=== 步骤 3/6：同源感知切分 ===")
    split = split_by_clusters(
        cleaned,
        train_ratio=cfg.train_ratio,
        valid_ratio=cfg.valid_ratio,
        test_ratio=cfg.test_ratio,
        seed=seed,
    )
    card.split_summary = split.summary()
    card.leakage_check = verify_no_leakage(split.train, split.test)

    processed_dir = resolve_path(cfg.processed_dir)
    for name, subset in (("train", split.train), ("valid", split.valid), ("test", split.test)):
        write_jsonl(processed_dir / f"{name}_records.jsonl", (r.to_dict() for r in subset))
    write_json(processed_dir / "split_summary.json", split.summary())

    # ---- 4. SFT 语料 ----
    logger.info("=== 步骤 4/6：构造 SFT 语料 ===")
    # 只在 train 上构造训练语料（valid 单独构造，避免训练时见到验证样本）
    sft_train = build_sft_corpus(split.train, total=sft_size, seed=seed)
    sft_valid = build_sft_corpus(split.valid, total=max(50, int(sft_size * 0.05)), seed=seed + 1)
    write_jsonl(processed_dir / "sft_train.jsonl", sft_train)
    write_jsonl(processed_dir / "sft_valid.jsonl", sft_valid)
    card.sft_count = len(sft_train)

    # ---- 5. GRPO prompts ----
    logger.info("=== 步骤 5/6：构造 GRPO prompt 集 ===")
    grpo = build_grpo_dataset(split.train, total=grpo_size, seed=seed)
    write_jsonl(processed_dir / "grpo_prompts.jsonl", grpo)
    card.grpo_count = len(grpo)

    from collections import Counter

    card.task_distribution = {
        "sft": dict(Counter(s["task_type"] for s in sft_train)),
        "grpo": dict(Counter(s["meta"]["task"] for s in grpo)),
    }

    # ---- 6. 校准集 ----
    logger.info("=== 步骤 6/6：构造校准集 ===")
    pairs = build_calibration_pairs(split.train, target_count=calibration_size, seed=seed)
    calib_dir = resolve_path(cfg.calibration_dir)
    write_jsonl(calib_dir / "calibration_pairs.jsonl", (p.to_dict() for p in pairs))
    write_jsonl(calib_dir / "calibration_warmup.jsonl", calibration_to_warmup_samples(pairs))
    card.calibration_count = len(pairs)

    # ---- 产出数据集清单 ----
    write_json(processed_dir / "dataset_card.json", card.to_dict())
    logger.info(
        "数据构建完成: 原始 %d -> 清洗 %d -> SFT %d 条 / GRPO %d 条 / 校准 %d 对",
        card.raw_count, card.cleaned_count, card.sft_count,
        card.grpo_count, card.calibration_count,
    )
    return card


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """`python -m pepdesign.data.pipeline` 的入口。

    用法::

        python -m pepdesign.data.pipeline                      # 使用本地已有原始数据
        python -m pepdesign.data.pipeline --download           # 先执行采集
        python -m pepdesign.data.pipeline --set data.sft_size=30000
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m pepdesign.data.pipeline",
        description="数据构建：原始数据 -> 清洗 -> 切分 -> SFT / GRPO / 校准语料",
    )
    parser.add_argument("--config", default=None, help="YAML 配置文件（默认用内置默认值）")
    parser.add_argument(
        "--download", action="store_true",
        help="先执行采集。默认关闭：采集要网络且耗时长，日常迭代不该每次都重拉。",
    )
    parser.add_argument(
        "--set", dest="overrides", nargs="*", default=[],
        help="覆盖配置项，形如 data.sft_size=30000",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides).data
    card = run_pipeline(cfg, download=args.download)

    logger.info(
        "语料规模：SFT %d 条 / GRPO %d 条 / 校准 %d 对（配置值 %d / %d / %d）",
        card.sft_count, card.grpo_count, card.calibration_count,
        cfg.sft_size, cfg.grpo_size, cfg.calibration_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
