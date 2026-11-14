"""生成示例数据集，用于在无网络、无 GPU 的环境下跑通全链路。

用公开文献中报道的真实多肽序列构造小规模样例集，覆盖清洗、切分、
语料构造、奖励计算、评估报告各步骤。序列不含实验测值，只能验证流程，
不能用于评估模型效果。

用法::

    python scripts/00_make_sample_data.py --out data/raw/sample_peptides.jsonl
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# 允许直接以脚本方式运行（把 src 加入 sys.path）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pepdesign.data.schema import ActivityRecord, PeptideRecord
from pepdesign.utils.io import write_jsonl
from pepdesign.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 真实多肽序列清单
# ---------------------------------------------------------------------------
# 字段：序列, 名称, 功能分类, 环化类型, 靶点/作用对象, 来源物种
CURATED: list[tuple[str, str, str, str, str, str]] = [
    # ---- 抗菌肽 ----
    ("GIGAVLKVLTTGLPALISWIKRKRQQ", "Melittin", "antibacterial",
     "none", "细菌细胞膜", "Apis mellifera"),
    ("GIGKFLHSAKKFGKAFVGEIMNS", "Magainin-2", "antibacterial",
     "none", "细菌细胞膜", "Xenopus laevis"),
    ("KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK", "Cecropin A", "antibacterial",
     "none", "细菌细胞膜", "Hyalophora cecropia"),
    ("LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES", "LL-37", "antibacterial",
     "none", "细菌细胞膜", "Homo sapiens"),
    ("ILPWKWPWWPWRR", "Indolicidin", "antibacterial",
     "none", "细菌细胞膜", "Bos taurus"),
    ("GIGKFLKKAKKFGKAFVKILKK", "Pexiganan (MSI-78)", "antibacterial",
     "none", "细菌细胞膜", "synthetic"),
    ("ALWMTLLKKVLKAAAKAALNAVLVGANA", "Dermaseptin S4", "antibacterial",
     "none", "细菌细胞膜", "Phyllomedusa sauvagei"),
    ("FLPLIGRVLSGIL", "Temporin A", "antibacterial",
     "none", "细菌细胞膜", "Rana temporaria"),
    ("FKCRRWQWRMKKLGAPSITCVRRAF", "Lactoferricin B", "antibacterial",
     "disulfide", "细菌细胞膜", "Bos taurus"),
    ("TRSSRAGLQFPVGRVHRLLRK", "Buforin II", "antibacterial",
     "none", "细菌 DNA/RNA", "Bufo gargarizans"),
    ("KWCFRVCYRGICYRKCR", "Tachyplesin I", "antibacterial",
     "disulfide", "细菌细胞膜", "Tachypleus tridentatus"),
    ("ITSISLCTPGCKTGALMGCNMKTATCHCSIHVSK", "Nisin A", "antibacterial",
     "disulfide", "细胞壁前体脂质 II", "Lactococcus lactis"),
    ("ACYCRIPACIAGERRYGTCIYQGRLWAFCC", "HNP-1 (α-defensin)", "antibacterial",
     "disulfide", "细菌细胞膜", "Homo sapiens"),
    ("DTHFPICIFCCGCCHRSKCGMCCKT", "Hepcidin-25", "antibacterial",
     "disulfide", "铁代谢调控", "Homo sapiens"),
    ("GIGAVLKVLTTGLPALISWIKRKRQQ-NH2", "Melittin amide", "antibacterial",
     "none", "细菌细胞膜", "synthetic"),
    ("RRWQWRMKKLG", "Lactoferricin B 片段", "antibacterial",
     "none", "细菌细胞膜", "synthetic"),
    ("WEAALAEALAEALAEHLAEALAEALEALAA", "GALA", "antiviral",
     "none", "病毒包膜", "synthetic"),
    ("GLFDIVKKVVGALGSL", "Aurein 1.2", "antibacterial",
     "none", "细菌细胞膜", "Litoria aurea"),
    ("RLCRIVVIRVCR", "Bactenecin", "antibacterial",
     "disulfide", "细菌细胞膜", "Bos taurus"),
    # ---- 穿膜肽 ----
    ("RQIKIWFQNRRMKWKK", "Penetratin", "cell_penetrating",
     "none", "细胞膜", "Drosophila antennapedia"),
    ("GRKKRRQRRRPPQ", "TAT (48-60)", "cell_penetrating",
     "none", "细胞膜", "HIV-1"),
    ("RRRRRRRRR", "R9", "cell_penetrating", "none", "细胞膜", "synthetic"),
    # ---- 抗肿瘤 / 靶向 ----
    ("KWKLVFFAKKWK", "靶向 Aβ 的 α-螺旋肽", "antitumor",
     "none", "淀粉样蛋白聚集", "synthetic"),
    ("GRGDSP", "RGD 靶向肽", "targeting",
     "none", "整合素 αvβ3", "synthetic"),
    ("RGDfV", "Cilengitide 骨架", "targeting",
     "head_to_tail", "整合素 αvβ3", "synthetic"),
    ("YHWYGYTPQNVI", "GE11", "targeting",
     "none", "EGFR", "synthetic"),
    ("CGNKRTRGC", "LyP-1", "targeting",
     "disulfide", "p32 受体", "synthetic"),
    # ---- 环肽 ----
    ("CYIQNCPLG", "Oxytocin", "other", "disulfide", "催产素受体", "Homo sapiens"),
    ("CYFQNCPRG", "Vasopressin", "other", "disulfide", "V2 受体", "Homo sapiens"),
    ("AGCKNFFWKTFTSC", "Somatostatin-14", "other", "disulfide",
     "生长抑素受体", "Homo sapiens"),
    ("FCFWKTCT", "Octreotide 骨架", "other", "disulfide",
     "生长抑素受体 SSTR2", "synthetic"),
    # ---- 神经/信号肽 ----
    ("RPKPQQFFGLM", "Substance P", "other", "none", "NK1 受体", "Homo sapiens"),
    ("RPPGFSPFR", "Bradykinin", "other", "none", "B2 受体", "Homo sapiens"),
    ("YGGFM", "Met-enkephalin", "other", "none", "阿片受体", "Homo sapiens"),
    ("TKPR", "Tuftsin", "immunomodulatory", "none", "巨噬细胞", "Homo sapiens"),
    ("HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR", "GLP-1 (7-36)", "inhibitor",
     "none", "GLP-1 受体", "Homo sapiens"),
]

# ---------------------------------------------------------------------------
# 变体生成
# ---------------------------------------------------------------------------
# 变体用于让数据规模足以验证同源去重与同源感知切分；
# 若序列互不相似，这两个模块的缺陷不会被暴露。
VARIANT_STRATEGIES = ("conservative", "terminal_cap", "pro_scan")


def make_variants(
    base: str, rng: random.Random, count: int
) -> list[tuple[str, str]]:
    """从母序列生成变体，返回 (序列, 变体说明) 列表。

    策略取值为 VARIANT_STRATEGIES：保守替换、末端修饰、脯氨酸扫描。
    """
    # 保守替换映射表：按侧链性质分组，组内互换对结构影响最小
    conservative_groups = [
        "AVILM",   # 疏水脂肪族
        "FYW",     # 芳香族
        "ST",      # 羟基
        "KRH",     # 碱性
        "DE",      # 酸性
        "NQ",      # 酰胺
    ]
    group_of = {aa: group for group in conservative_groups for aa in group}
    out: list[tuple[str, str]] = []

    for _ in range(count):
        strategy = rng.choice(VARIANT_STRATEGIES)
        seq = base.replace("-NH2", "")

        if strategy == "conservative":
            pos = rng.randrange(len(seq))
            aa = seq[pos]
            group = group_of.get(aa)
            if group is None or len(group) < 2:
                continue
            new_aa = rng.choice([c for c in group if c != aa])
            variant = seq[:pos] + new_aa + seq[pos + 1 :]
            if variant == seq:
                continue
            out.append((variant, f"保守替换 {pos + 1}{aa}->{new_aa}"))

        elif strategy == "terminal_cap":
            # 末端修饰以字符串前缀/后缀表示，解析器可识别
            cap = rng.choice(["Ac-", "-NH2"])
            variant = f"{cap}{seq}" if cap == "Ac-" else f"{seq}{cap}"
            out.append((variant, f"末端修饰 {cap}"))

        elif strategy == "pro_scan":
            candidates = [
                i + 1 for i in range(len(seq) - 1)
                if seq[i] in "KR" and seq[i + 1] != "P"
            ]
            if not candidates:
                continue
            pos = rng.choice(candidates)
            variant = seq[:pos] + "P" + seq[pos + 1 :]
            if variant == seq:
                continue
            out.append((variant, f"脯氨酸扫描 {pos + 1}{seq[pos]}->P"))

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="生成示例多肽数据集")
    parser.add_argument(
        "--out", default="data/raw/sample_peptides.jsonl", help="输出路径"
    )
    # 默认 8：该值决定同源去重可验证的相似度下限，以及下游语料中
    # 不同底层序列的数量；过小会使语料重复度过高。
    parser.add_argument(
        "--variants-per-seed", type=int, default=8,
        help="每条母序列生成的变体数量（用于验证同源去重逻辑）",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path

    rng = random.Random(args.seed)
    records: list[PeptideRecord] = []

    # ---- 母序列 ----
    for seq, name, func, cyclic, target, organism in CURATED:
        rec = PeptideRecord(
            sequence=seq,
            name=name,
            source="uniprot",           # 这些序列在 UniProt 里都有对应条目
            source_id=f"SAMPLE:{name.replace(' ', '_')}",
            function=func,  # type: ignore[arg-type]
            cyclic_type=cyclic,  # type: ignore[arg-type]
            target=target,
            organism=organism,
            lineage={"sample": True, "kind": "curated"},
        )
        rec.compute_properties()
        records.append(rec)

    # ---- 变体 ----
    n_variants = 0
    for seq, name, func, cyclic, target, organism in CURATED:
        for variant_seq, description in make_variants(seq, rng, args.variants_per_seed):
            rec = PeptideRecord(
                sequence=variant_seq,
                name=f"{name} 变体",
                source="synthetic",
                source_id=f"SAMPLE:{name.replace(' ', '_')}:{n_variants}",
                function=func,  # type: ignore[arg-type]
                cyclic_type=cyclic,  # type: ignore[arg-type]
                target=target,
                organism=organism,
                lineage={"sample": True, "kind": "variant", "change": description,
                         "parent": name},
            )
            rec.compute_properties()
            records.append(rec)
            n_variants += 1

    # ---- 少量带实验测值的记录 ----
    # 用于覆盖"优先保留有实验数据的记录"这条逻辑。
    for rec in records[:6]:
        # 由性质估算 MIC 量级，非真实实验值
        mic = round(2.0 + rec.properties.get("aggregation_propensity", 0.3) * 30, 2)
        rec.activities.append(
            ActivityRecord(
                assay_type="MIC",
                target=rec.target or "Escherichia coli",
                value=mic,
                unit="ug/mL",
                method="broth_microdilution",
                source_ref="SAMPLE(placeholder)",
            )
        )
        rec.activities.append(
            ActivityRecord(
                assay_type="Tm",
                target=rec.target or "n/a",
                value=round(35 + rec.length * 0.8, 1),
                unit="degC",
                method="CD",
                source_ref="SAMPLE(placeholder)",
            )
        )

    count = write_jsonl(out_path, (r.to_dict() for r in records))
    logger.info(
        "示例数据已生成: %s（%d 条：%d 条母序列 + %d 条变体）",
        out_path, count, len(CURATED), n_variants,
    )


if __name__ == "__main__":
    main()
