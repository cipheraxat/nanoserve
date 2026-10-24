"""Prefix caching: N requests share one long system prompt and differ only in a short
suffix (a typical chat/agent pattern). Compares prefill work and time to first token with the
prefix cache on and off."""

from __future__ import annotations

import argparse
import time

from common import device_name, random_prompts, save, summary_ms, sync, table

from nanoserve import EngineConfig, LLMEngine, SamplingParams


def run(args, prompts, enable: bool) -> dict:
    engine = LLMEngine(
        EngineConfig(
            model=args.model,
            device=args.device,
            max_num_seqs=args.batch,
            max_num_batched_tokens=args.token_budget,
            enable_prefix_caching=enable,
        )
    )
    engine.generate([[5] * 8], SamplingParams(max_tokens=2))
    engine.stats.ttft_s.clear()
    computed_before = engine.stats.computed_tokens
    sync(engine.runner.device)
    start = time.perf_counter()
    engine.generate(prompts, SamplingParams(max_tokens=args.output_len, temperature=0, ignore_eos=True))
    sync(engine.runner.device)
    elapsed = time.perf_counter() - start
    ttft = summary_ms(engine.stats.ttft_s)
    return {
        "prefix_caching": "on" if enable else "off",
        "seconds": round(elapsed, 2),
        "tokens_computed": engine.stats.computed_tokens - computed_before,
        "cache_hit_rate": f"{engine.block_manager.prefix_hit_rate:.0%}",
        "ttft_p50_ms": ttft["p50_ms"],
        "ttft_p99_ms": ttft["p99_ms"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--device", default="auto")
    p.add_argument("--num-requests", type=int, default=32)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--shared-len", type=int, default=1024)
    p.add_argument("--suffix-len", type=int, default=32)
    p.add_argument("--output-len", type=int, default=32)
    p.add_argument("--token-budget", type=int, default=1024)
    args = p.parse_args()

    shared = random_prompts(1, args.shared_len, args.shared_len, seed=7)[0]
    prompts = [shared + s for s in random_prompts(args.num_requests, args.suffix_len, args.suffix_len, seed=8)]
    rows = [run(args, prompts, False), run(args, prompts, True)]
    device = args.device if args.device != "auto" else str(EngineConfig().resolve_device())
    print(
        f"\n{device_name(device)}, {args.num_requests} requests, {args.shared_len}-token shared prefix + {args.suffix_len}-token suffix"
    )
    print(table(rows, ["prefix_caching", "seconds", "tokens_computed", "cache_hit_rate", "ttft_p50_ms", "ttft_p99_ms"]))
    print("saved", save("prefix", device, {"args": vars(args), "results": rows}))


if __name__ == "__main__":
    main()
