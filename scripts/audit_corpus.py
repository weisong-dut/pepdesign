"""语料审计：格式、重复、环肽覆盖、泄漏、修复有效性、GRPO 结构。

用法::

    python scripts/audit_corpus.py

只读 `data/processed/` 与 `data/calibration/`，用项目内的性质函数与
奖励函数校验，不修改任何文件。任一项出现非零计数都应先查清原因再训练。
"""
from __future__ import annotations

import collections
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pepdesign.bio.cyclization import feasible_cyclic_type  # noqa: E402
from pepdesign.bio.sequence import (  # noqa: E402
    aggregation_propensity,
    hydrophobic_moment_profile,
    net_charge,
)
from pepdesign.data.schema import PeptideRecord  # noqa: E402
from pepdesign.rewards.composite import CompositeScorer  # noqa: E402
from pepdesign.utils.config import load_config  # noqa: E402

AA = set("ACDEFGHIKLMNPQRSTVWY")
DATA = ROOT / "data" / "processed"


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean(seq: str) -> str:
    return re.sub(r"\s+", "", seq.replace("Ac-", "").replace("-NH2", "")).upper()


def answer_of(row: dict) -> str | None:
    m = re.search(r"### 设计序列\s*\n\s*([^\n]+)", row["messages"][-1]["content"])
    return m.group(1).strip() if m else None


def _props(seq: str) -> dict:
    rec = PeptideRecord(
        sequence=seq, name="x", source="x", source_id="x",
        function="other", cyclic_type="none",
    )
    rec.compute_properties()
    return rec.properties


def check_sft_format(sft: list[dict]) -> None:
    role_bad = fmt_bad = seq_bad = 0
    for r in sft:
        roles = [m.get("role") for m in r.get("messages", [])]
        if roles != ["system", "user", "assistant"]:
            role_bad += 1
            continue
        if r["task_type"] == "knowledge":
            continue
        content = r["messages"][-1]["content"]
        if not (
            content.startswith("### 设计序列")
            and "### 设计理由" in content
            and content.rstrip().endswith("### 结束")
        ):
            fmt_bad += 1
        a = answer_of(r)
        if a is None or not clean(a) or set(clean(a)) - AA:
            seq_bad += 1
    print(f"[格式] 角色异常 {role_bad} | 三段格式不符(非 knowledge) {fmt_bad} | 序列非法 {seq_bad}")


def check_duplicates(sft: list[dict], grpo: list[dict]) -> None:
    for name, rows, key in (("SFT", sft, "messages"), ("GRPO", grpo, "prompt")):
        total = 0
        for task in ("de_novo", "stabilize", "multi_objective", "knowledge"):
            sub = [r for r in rows if r["task_type"] == task]
            if not sub:
                continue
            seen = collections.Counter(json.dumps(r[key], ensure_ascii=False) for r in sub)
            dup = sum(v - 1 for v in seen.values() if v > 1)
            total += dup
            print(
                f"[重复] {name} {task}: 唯一 {len(seen)}/{len(sub)}, "
                f"重复 {dup}, 最高 {max(seen.values())}"
            )
        print(f"[重复] {name} 合计重复 {total}")


def check_cyclic(sft: list[dict], grpo: list[dict], valid: list[dict]) -> None:
    def dist(rows):
        return dict(collections.Counter(r["meta"].get("cyclic_type") for r in rows))

    for name, rows in (("SFT", sft), ("valid", valid), ("GRPO", grpo)):
        d = dist(rows)
        n = len(rows) or 1
        share = (d.get("disulfide", 0)) / n
        print(f"[环肽] {name}: {d}, disulfide 占比 {share:.1%}")


def check_leakage(sft: list[dict], grpo: list[dict], cal: list[dict]) -> None:
    heldout = {
        clean(r["sequence"]).upper()
        for f in ("valid_records.jsonl", "test_records.jsonl")
        for r in load(DATA / f)
    }
    ans_hit = sum(1 for r in sft if (a := answer_of(r)) and clean(a).upper() in heldout)
    base_hit = sum(
        1 for r in sft if clean(r["meta"].get("base_sequence", "")).upper() in heldout
    )
    grpo_hit = sum(
        1 for r in grpo
        if clean(r["meta"].get("base_sequence", "")).upper() in heldout
    )
    cal_hit = sum(
        1 for r in cal
        if clean(r.get("base_sequence", "")).upper() in heldout
        or clean(r.get("improved_sequence", "")).upper() in heldout
    )
    print(f"[泄漏] SFT base {base_hit} / 答案 {ans_hit} | GRPO {grpo_hit} | 校准 {cal_hit}")


def check_repairs(sft: list[dict]) -> None:
    """按各劣化策略自身的口径验证修复方向。

    例如 `break_amphipathic` 须用滑窗疏水矩最大值，避免口径错配的假阳性。
    """
    def prot(s):
        return sum(1 for i in range(len(s) - 1) if s[i] in "KR" and s[i + 1] == "P")

    stats: collections.Counter = collections.Counter()
    bad: collections.Counter = collections.Counter()
    reason_bad = 0
    for r in sft:
        if r["task_type"] not in ("stabilize", "multi_objective"):
            continue
        strategy = r["meta"].get("degrade_strategy")
        base = clean(r["meta"].get("base_sequence", ""))
        a = answer_of(r)
        if not strategy or not a:
            continue
        ans = clean(a)
        if len(base) != len(ans) or base == ans:
            continue
        stats[strategy] += 1
        if not re.search(r"第 \d+", r["messages"][-1]["content"]):
            reason_bad += 1
        if strategy == "add_protease_site":
            ok = prot(ans) > prot(base)
        elif strategy == "break_amphipathic":
            w = min(9, len(base))
            ok = max(hydrophobic_moment_profile(ans, window=w)) > (
                max(hydrophobic_moment_profile(base, window=w)) + 1e-9
            )
        elif strategy == "hydrophobic_run":
            ok = aggregation_propensity(ans) < aggregation_propensity(base) - 1e-9
        elif strategy == "charge_clash":
            ok = abs(net_charge(ans)) < abs(net_charge(base)) - 1e-9
        elif strategy == "mismatch_cyclic":
            ok = ans.count("C") % 2 == 0 and base.count("C") % 2 != 0
        elif strategy == "helix_breaker":
            ok = _props(ans)["helix_propensity"] > _props(base)["helix_propensity"]
        else:
            ok = True
        if not ok:
            bad[strategy] += 1
    print(f"[修复] 理由无实际位点 {reason_bad}")
    for k in stats:
        print(f"[修复] {k}: {stats[k]} 条, 未改善 {bad.get(k, 0)}")


def check_denovo(sft: list[dict], train: list[dict]) -> None:
    den = [r for r in sft if r["task_type"] == "de_novo"]
    lr_bad = q_bad = cyc_bad = 0
    candidates: list[int] = []
    offsets: list[float] = []
    recs = [(len(r["sequence"]), net_charge(r["sequence"])) for r in train]
    for r in den:
        a = answer_of(r)
        if not a:
            continue
        ans = clean(a)
        meta = r["meta"]
        lo, hi = meta["length_range"]
        q_lo, q_hi = meta["net_charge_range"]
        if not lo <= len(ans) <= hi:
            lr_bad += 1
        if not q_lo - 0.05 <= net_charge(ans) <= q_hi + 0.05:
            q_bad += 1
        if meta.get("cyclic_type", "none") != "none" and (
            feasible_cyclic_type(ans, meta["cyclic_type"]) != meta["cyclic_type"]
        ):
            cyc_bad += 1
        candidates.append(
            sum(1 for (n, q) in recs if lo <= n <= hi and q_lo - 1e-9 <= q <= q_hi + 1e-9)
        )
        if hi > lo:
            offsets.append(abs(len(ans) - (lo + hi) / 2) / (hi - lo))
    print(
        f"[de_novo] 长度越界 {lr_bad} | 电荷越界 {q_bad} | 环化不可行 {cyc_bad}"
    )
    if candidates:
        print(
            f"[de_novo] 同约束候选数中位 {statistics.median(candidates):.0f} | "
            f"候选=1 比例 {sum(1 for n in candidates if n <= 1) / len(candidates):.1%} | "
            f"答案偏离中点均值 {sum(offsets) / max(1, len(offsets)):.3f}"
        )


def check_grpo(grpo: list[dict]) -> None:
    struct_bad = meta_bad = bad_base = 0
    need = {
        "stabilize": {"base_sequence", "cyclic_type"},
        "de_novo": {"cyclic_type"},
        "multi_objective": {"base_sequence", "objectives", "cyclic_type"},
    }
    for r in grpo:
        if [m.get("role") for m in r.get("prompt", [])] != ["system", "user"]:
            struct_bad += 1
        if need.get(r["task_type"], set()) - set(r.get("meta", {})):
            meta_bad += 1
        base = r["meta"].get("base_sequence")
        if base is not None and (not clean(base) or set(clean(base)) - AA):
            bad_base += 1
    print(f"[GRPO] 结构异常 {struct_bad} | meta 缺字段 {meta_bad} | base 非法 {bad_base}")

    cfg = load_config(str(ROOT / "configs/grpo_lora.yaml"))
    scorer = CompositeScorer(cfg.reward)
    invalid = 0
    checked = 0
    for r in grpo:
        base = r["meta"].get("base_sequence")
        if not base:
            continue
        checked += 1
        out = scorer.score(base, r["meta"])
        if getattr(out, "invalid_reason", None):
            invalid += 1
    print(f"[GRPO] 奖励可解析 base: {checked} 条, invalid {invalid}")


def main() -> int:
    sft = load(DATA / "sft_train.jsonl")
    valid = load(DATA / "sft_valid.jsonl")
    grpo = load(DATA / "grpo_prompts.jsonl")
    cal = load(ROOT / "data" / "calibration" / "calibration_pairs.jsonl")
    train = load(DATA / "train_records.jsonl")
    print(
        f"SFT {len(sft)} | valid {len(valid)} | GRPO {len(grpo)} | "
        f"校准 {len(cal)} | train records {len(train)}"
    )

    check_sft_format(sft)
    check_duplicates(sft, grpo)
    check_cyclic(sft, grpo, valid)
    check_leakage(sft, grpo, cal)
    check_repairs(sft)
    check_denovo(sft, train)
    check_grpo(grpo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
