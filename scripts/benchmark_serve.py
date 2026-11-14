"""推理服务压测：并发调用 `/v1/completions`，统计延迟、吞吐、成功率。

用法::

    python scripts/benchmark_serve.py --concurrency 8 --requests 32 --max-tokens 128
    python scripts/benchmark_serve.py --url http://127.0.0.1:8000 --prompt "设计一条抗菌肽"

只测已启动的服务，不加载模型、不改数据；token 数取自服务端返回的 usage。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_sample_prompt() -> str:
    """从 GRPO prompt 集里取一条 user 文本，作为压测输入。"""
    path = ROOT / "data/processed/grpo_prompts.jsonl"
    if not path.exists():
        return "请设计一条抗菌肽。"
    with open(path, encoding="utf8") as f:
        row = json.loads(f.readline())
    messages = row.get("prompt", [])
    for m in messages:
        if m.get("role") == "user":
            return m["content"]
    return "请设计一条抗菌肽。"


async def _one(client, url: str, payload: dict) -> tuple[float, int]:
    t0 = time.perf_counter()
    resp = await client.post(url, json=payload)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    tokens = int(data.get("usage", {}).get("completion_tokens", 0) or 0)
    return elapsed, tokens


async def _run(args) -> int:
    import httpx

    url = args.url.rstrip("/") + "/v1/completions"
    prompt = args.prompt or load_sample_prompt()
    payload = {
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    limits = httpx.Limits(
        max_connections=args.concurrency * 2,
        max_keepalive_connections=args.concurrency,
    )
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        # 探活 + 预热（第一次请求包含 CUDA kernel 初始化，不计入统计）
        try:
            await _one(client, url, payload)
        except Exception as exc:  # noqa: BLE001
            print(f"预热请求失败：{type(exc).__name__}: {exc}")
            print("服务没起来？先检查 /health。")
            return 1

        sem = asyncio.Semaphore(args.concurrency)

        async def task():
            async with sem:
                return await _one(client, url, payload)

        started = time.perf_counter()
        results = await asyncio.gather(
            *[task() for _ in range(args.requests)], return_exceptions=True
        )
        wall = time.perf_counter() - started

    latencies: list[float] = []
    token_counts: list[int] = []
    failures = 0
    for r in results:
        if isinstance(r, Exception):
            failures += 1
            continue
        elapsed, tokens = r
        latencies.append(elapsed)
        token_counts.append(tokens)

    ok = len(latencies)
    total_tokens = sum(token_counts)
    print("\n=== 压测结果 ===")
    print(f"并发 {args.concurrency} | 请求 {args.requests} | max_tokens {args.max_tokens}")
    print(f"成功 {ok} / 失败 {failures} | 成功率 {ok / max(1, args.requests):.1%}")
    print(f"墙钟 {wall:.2f}s | 吞吐 {ok / wall:.2f} req/s | "
          f"{total_tokens / wall:.1f} completion tokens/s")
    if latencies:
        lat = sorted(latencies)
        p50 = statistics.median(lat)
        p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
        print(f"延迟 p50 {p50:.2f}s | p95 {p95:.2f}s | max {lat[-1]:.2f}s")
    if token_counts:
        print(f"每条 completion tokens: 平均 {statistics.mean(token_counts):.1f}, "
              f"min {min(token_counts)}, max {max(token_counts)}")
    return 0 if failures == 0 else 2


def main() -> int:
    p = argparse.ArgumentParser(description="推理服务压测")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--requests", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--prompt", default=None, help="默认从 GRPO prompt 集取一条")
    args = p.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
