"""奖励预测器与实验实测值的对照。

奖励层中热稳定性维度使用 `PRIOR_COEFFICIENTS`，即一组手设的、
未拟合的系数（`tm.py` 中标注 `is_calibrated=False`）。
本模块量化这组先验与实验数据之间的偏差，无需 GPU 或模型前向：
输入为 `data/processed/*_records.jsonl` 中带实验数据的记录与预测器输出。

局限：
1. 代理映射是近似。见 `DIMENSION_TO_ASSAY`：MIC 是抑菌浓度，
   不是结合亲和力，用作 affinity 维度的实测值存在信息损失。
   映射以显式表给出，便于检查与推翻。
2. 可用实测数据极少。当前数据中带实测值的记录数不足以支持
   统计结论，此时 `conclusive` 为 False，表示"尚无证据"，
   而不是模块故障。
   想得到真结论，唯一的办法是喂进真实的实验数据。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pepdesign.eval.offline import (
    MIN_N_FOR_SIGNIFICANCE,
    correlation_report,
    orientation_for,
)
from pepdesign.rewards.predictors.factory import DEFAULT_BACKENDS, build_predictor
from pepdesign.utils.logger import get_logger

logger = get_logger(__name__)

#: 奖励维度 → （实测测定类型, 为什么可以这样近似）。
#:
#: 只有两维有对应的实测数据可查：
#:   * thermo  ↔ Tm      —— 同一件事的两种叫法，最干净的一对；
#:   * affinity ↔ MIC    —— 有损的代理。MIC 是"抑制细菌生长的最低浓度"，
#:     它混合了膜亲和、通透性、代谢稳定性等因素；affinity 维度算的是
#:     序列特征层面的相对结合倾向。两者相关但不等价。
#:
#: 另外两维（protease / structure）不在表里：本仓库的数据里没有任何
#: 实验测定对应它们，硬凑一个映射等于编造证据。它们会出现在报告的
#: `not_evaluated` 里，明说"没有可用的实测数据"，而不是悄悄消失。
DIMENSION_TO_ASSAY: dict[str, tuple[str, str]] = {
    "thermo": ("Tm", "热稳定性预测器输出就是 Tm，与实测 Tm 同一量纲"),
    "affinity": ("MIC", "有损代理：MIC 混合了膜亲和/通透性/稳定性，"
                        "不等于结合亲和力，只作为趋势参考"),
}

#: 不进 `DIMENSION_TO_ASSAY` 的维度，各自"为什么没有实测对应"。
#:
#: 默认那句话（"本仓库没有对应测定"）对 protease / structure 是准确的——
#: 它们本该有实验数据（酶切位点、CD 谱），只是本仓库里没有。
#: 而 `plm` 的处境完全不同：它衡量的是"这条序列有多像天然蛋白"，
#: 这不是一个可以直接测定的物理量，没有对应的 assay，
#: 所以它的理由必须单独写，否则会被误读成"缺数据、以后补上"。
NOT_EVALUATED_REASONS: dict[str, str] = {
    "plm": (
        "这一维衡量的是'序列在多大程度上像一条真实存在过的蛋白'，"
        "不是一个可以直接测定的物理量，因此没有对应的实验 assay——"
        "不是'缺数据'，而是它本来就不该用这种相关性检验来验证。"
        "它的正确用法是当外部裁判：看规则引擎认为变好的那些序列，"
        "在 ESM 眼里是不是也更像天然蛋白（见 rewards/plm.py 的模块文档）。"
    ),
}


def _measured_by_assay(record: dict) -> dict[str, float]:
    """把一条记录里的实验值取成 {assay_type: value}。

    规则与 `PeptideRecord.best_activity` 完全一致（同一测定类型有多条时
    取"最强"的那条，方向查 `data.schema.orientation_for`），只是这里直接
    对原始 dict 做——本模块的输入是 JSONL，那些记录未必都能通过 schema 的
    全部校验，而评测层不该被数据层的严格性挡住（读不到数据 = 没有证据，
    但"能算出多少算多少"比"整批拒绝"更有用）。

    方向在这里就应用掉：MIC 越小越好，所以取负号，
    这样后面所有维度都可以统一按"预测值越大越好"来算相关。
    """
    out: dict[str, float] = {}
    for act in record.get("activities") or []:
        assay = act.get("assay_type")
        value = act.get("value")
        if assay is None or value is None:
            continue
        sign = orientation_for(assay)
        # 同一测定类型出现多次时"取最好的那条"：把 sign 乘进去之后
        # 就统一成"越大越好"，所以这里取 max。
        oriented = sign * float(value)
        if assay not in out or oriented > out[assay]:
            out[assay] = oriented
    return out


def audit_reward_predictors(
    records: Sequence[dict],
    *,
    min_n: int = MIN_N_FOR_SIGNIFICANCE,
    n_perm: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    """对每个"有实测数据可查"的奖励维度，算一遍预测值 vs 实测值的相关性。

    报告里的 `conclusive` 在数据量不够时是 False——这是预期结果，
    不是失败。本仓库当前就是这种情况（6 条 placeholder 数据）。
    """
    usable = [r for r in records if r.get("sequence") and r.get("activities")]
    report: dict[str, Any] = {
        "records_total": len(records),
        "records_with_activities": len(usable),
        "dimensions": {},
        "not_evaluated": {},
        "orientation_applied": True,
        "note": (
            "orientation_applied=True 表示实测值已按'越大越好'统一过符号"
            "（MIC 取负），所以所有相关系数都是正的表示方向正确。"
        ),
    }

    for kind, (assay, why) in DIMENSION_TO_ASSAY.items():
        rows = [(r["sequence"], _measured_by_assay(r).get(assay)) for r in usable]
        rows = [(s, v) for s, v in rows if v is not None]
        entry: dict[str, Any] = {
            "assay": assay,
            "rationale": why,
            "predictor": "",
            "backend": DEFAULT_BACKENDS[kind],
        }
        if not rows:
            entry["status"] = "no_measured_data"
            entry["n"] = 0
            report["dimensions"][kind] = entry
            continue

        predictor = build_predictor(kind, DEFAULT_BACKENDS[kind])
        seqs, measured = zip(*rows, strict=True)
        preds = [p.value for p in predictor.predict(list(seqs))]

        entry["predictor"] = predictor.name
        entry["is_calibrated"] = bool(getattr(predictor, "is_calibrated", False))
        entry["status"] = "ok"
        entry.update(
            correlation_report(preds, measured, min_n=min_n, n_perm=n_perm, seed=seed)
        )
        report["dimensions"][kind] = entry

    for kind in DEFAULT_BACKENDS:
        if kind not in DIMENSION_TO_ASSAY:
            report["not_evaluated"][kind] = NOT_EVALUATED_REASONS.get(
                kind,
                "本仓库的数据里没有对应的实验测定，不做评估"
                "（硬凑一个映射等于编造证据）",
            )

    # 顶层给一句总体判断，避免调用方只看单个维度的 p 值。
    conclusive = [
        k for k, v in report["dimensions"].items() if v.get("conclusive")
    ]
    report["conclusive_dimensions"] = conclusive
    report["all_caveats"] = sorted(
        {
            c
            for v in report["dimensions"].values()
            for c in (v.get("caveats") or [])
        }
    )
    logger.info(
        "奖励审计：%d 条记录中 %d 条带实测数据；可下结论的维度：%s",
        report["records_total"], report["records_with_activities"],
        conclusive or "无",
    )
    return report


def summarize_audit(report: dict[str, Any]) -> str:
    """把报告压成几行给人看的话。不下"奖励可靠/不可靠"的结论——
    数据量不够时那种句子只能是暗示，而暗示正是本模块要消灭的东西。"""
    lines = [
        f"记录 {report['records_total']} 条，其中带实测数据 "
        f"{report['records_with_activities']} 条。"
    ]
    for kind, entry in report["dimensions"].items():
        if entry.get("status") != "ok":
            lines.append(f"  {kind:10s}: 无实测数据，未评估")
            continue
        verdict = "可下结论" if entry["conclusive"] else "样本不足，不作结论"
        lines.append(
            f"  {kind:10s}: 预测器 {entry['predictor']}"
            f"（标定={entry['is_calibrated']}） vs 实测 {entry['assay']}"
            f" n={entry['n']} | Spearman {entry['spearman']:+.3f}"
            f" | p={entry['p_value']} → {verdict}"
        )
    for kind, why in report["not_evaluated"].items():
        lines.append(f"  {kind:10s}: 未评估（{why}）")
    return "\n".join(lines)
