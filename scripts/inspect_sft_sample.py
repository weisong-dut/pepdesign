"""解码 SFT 训练样本的 input_ids 与 labels，检查对齐。

正确结果：input_ids 为 system + user + assistant 的完整 token；
labels 只保留 assistant 回复，prompt 与 padding 均为 -100。
若 labels 混入 user 内容或与 input_ids 错位，teacher forcing loss 仍会
偏低，但自由生成会退化为重复或乱码。

用法::

    python scripts/inspect_sft_sample.py [n]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from pepdesign.models.loader import load_tokenizer  # noqa: E402
from pepdesign.train.sft import collate_sft, encode_sft_sample  # noqa: E402
from pepdesign.utils.config import load_config  # noqa: E402


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    cfg = load_config("configs/sft_lora.yaml")
    tokenizer = load_tokenizer(cfg.sft.model_name_or_path)

    with open(ROOT / "data/processed/sft_train.jsonl", encoding="utf8") as f:
        rows = [json.loads(line) for line in f if line.strip()][:n]

    for i, row in enumerate(rows):
        enc = encode_sft_sample(
            tokenizer, row,
            max_seq_length=cfg.sft.max_seq_length,
            completion_only=cfg.sft.completion_only_loss,
        )
        if enc["dropped"]:
            print(f"样本{i} 被丢弃: {enc['dropped']}")
            continue
        batch = collate_sft(torch, [enc], tokenizer.pad_token_id)
        ids = batch["input_ids"][0].tolist()
        labels = batch["labels"][0].tolist()

        label_idx = [j for j, t in enumerate(labels) if t != -100]
        answer = tokenizer.decode([labels[j] for j in label_idx], skip_special_tokens=True)

        print("=" * 60)
        print(f"样本{i}  task={row['task_type']}")
        print(f"input_ids 长度 {len(ids)}, labels 非 -100 数量 {len(label_idx)}")
        if label_idx:
            print(f"labels 起始位置 {label_idx[0]}, 结束位置 {label_idx[-1]}")
            prefix = tokenizer.decode(ids[: label_idx[0]], skip_special_tokens=True)
            print(f"labels 之前的输入（应为 system+user+生成提示）:\n{prefix[-300:]!r}")
        print(f"labels 解码（应为 assistant 回复）:\n{answer[:400]!r}")
        print(f"原始 assistant 答案开头:\n{row['messages'][-1]['content'][:300]!r}")

        # 对齐检查：labels 的非 -100 部分应当逐 token 等于
        # input_ids 在同一位置的 token（-100 是掩码，不是移位）
        mismatched = sum(
            1 for j in label_idx if labels[j] != ids[j]
        )
        print(f"labels 与 input_ids 在同位置的 token 不一致数量: {mismatched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
