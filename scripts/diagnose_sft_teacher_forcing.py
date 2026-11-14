"""对比 SFT adapter 的 teacher forcing loss 与自由生成。

判读：loss 低而自由生成乱码，指向暴露偏差或训练方向问题；loss 也高，
说明 adapter 未拟合（与训练日志矛盾时应检查保存/加载）；loss 低且生成
正常，则此前的生成问题与 prompt 截断或采样参数有关。

用法::

    python scripts/diagnose_sft_teacher_forcing.py \
        --adapter outputs/sft_diag/checkpoint-50 --n 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from pepdesign.models.loader import (  # noqa: E402
    chat_tokenize,
    generation_stop_ids,
    load_base_model,
    load_tokenizer,
    probe_hardware,
    resolve_plan,
)
from pepdesign.models.lora import inject_lora, load_adapter  # noqa: E402
from pepdesign.train.masking import pad_left  # noqa: E402
from pepdesign.train.sft import collate_sft, encode_sft_sample  # noqa: E402
from pepdesign.utils.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="teacher forcing vs 自由生成")
    parser.add_argument("--config", default="configs/sft_lora.yaml")
    parser.add_argument("--adapter", default="outputs/sft_diag/checkpoint-50")
    parser.add_argument("--data", default="data/processed/sft_train.jsonl")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    tokenizer = load_tokenizer(cfg.sft.model_name_or_path)
    hw = probe_hardware()
    plan = resolve_plan(
        cuda_available=hw["cuda_available"],
        bf16_supported=hw["bf16_supported"],
        mps_available=hw["mps_available"],
        want_bf16=cfg.sft.bf16,
        want_fp16=cfg.sft.fp16,
        quant=cfg.quant,
    )
    model, _ = load_base_model(
        cfg.sft.model_name_or_path, quant=cfg.quant,
        want_bf16=cfg.sft.bf16, want_fp16=cfg.sft.fp16, plan=plan,
    )
    model = inject_lora(model, cfg.lora)
    model = load_adapter(model, args.adapter, is_trainable=False)
    model.eval()
    device = next(model.parameters()).device

    with open(args.data, encoding="utf8") as f:
        rows = [json.loads(line) for line in f if line.strip()][: args.n]

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
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            loss = model(**batch).loss
        print(f"\n样本{i}  task={row['task_type']}  teacher-forcing loss={float(loss):.4f}")

        prompt_ids = chat_tokenize(
            tokenizer, row["messages"][:-1], add_generation_prompt=True
        )
        batched = pad_left([prompt_ids], tokenizer.pad_token_id)
        ids = torch.tensor(batched["input_ids"], dtype=torch.long, device=device)
        att = torch.tensor(batched["attention_mask"], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(
                input_ids=ids,
                attention_mask=att,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=sorted(generation_stop_ids(tokenizer)),
            )
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        print(f"  自由生成: {text[:300]!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
