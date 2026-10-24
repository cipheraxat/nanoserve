"""Online load test against a running server (nanoserve, or any OpenAI-compatible server).

Requests arrive as a Poisson process at --rate req/s and stream their responses. Reports
time to first token (TTFT), inter-token latency (ITL), end-to-end latency and throughput.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time

import httpx
from common import output_lengths, random_prompts, summary_ms, table


async def one_request(client, url, model, prompt, max_tokens, results):
    start = time.perf_counter()
    stamps = []
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True,
    }
    async with client.stream("POST", f"{url}/v1/completions", json=payload) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            json.loads(line[6:])
            stamps.append(time.perf_counter())
    results.append(
        {
            "ttft": stamps[0] - start,
            "itl": [b - a for a, b in zip(stamps, stamps[1:], strict=False)],
            "e2e": stamps[-1] - start,
            "tokens": len(stamps),
        }
    )


async def run(args) -> None:
    prompts = random_prompts(args.num_requests, *args.prompt_len, seed=11)
    out_lens = output_lengths(args.num_requests, *args.output_len, seed=12)
    rng = random.Random(13)
    results: list[dict] = []
    async with httpx.AsyncClient(timeout=None) as client:
        start = time.perf_counter()
        tasks = []
        for prompt, n in zip(prompts, out_lens, strict=True):
            tasks.append(asyncio.create_task(one_request(client, args.url, args.model, prompt, n, results)))
            await asyncio.sleep(rng.expovariate(args.rate))
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start
    ttft = summary_ms([r["ttft"] for r in results])
    itl = summary_ms([g for r in results for g in r["itl"]])
    e2e = summary_ms([r["e2e"] for r in results])
    tokens = sum(r["tokens"] for r in results)
    print(
        f"\n{args.num_requests} requests at {args.rate} req/s, {elapsed:.1f}s, {tokens / elapsed:.1f} output tok/s, "
        f"{len(results) / elapsed:.2f} req/s completed"
    )
    print(
        table(
            [{"metric": "TTFT", **ttft}, {"metric": "ITL", **itl}, {"metric": "E2E", **e2e}],
            ["metric", "mean_ms", "p50_ms", "p99_ms", "max_ms"],
        )
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--num-requests", type=int, default=64)
    p.add_argument("--rate", type=float, default=2.0)
    p.add_argument("--prompt-len", type=int, nargs=2, default=[32, 512])
    p.add_argument("--output-len", type=int, nargs=2, default=[16, 128])
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
