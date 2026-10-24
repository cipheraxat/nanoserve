"""Chunked prefill: while a batch of requests is decoding, long prompts keep arriving.
Without chunking, each long prompt is prefilled in one step and every running decode stalls
for that step. With a token budget, the prompt is split across steps that also carry the
decodes. Measures the inter-token latency (ITL) seen by the decoding requests."""

from __future__ import annotations

import argparse
import time

from common import device_name, random_prompts, save, summary_ms, sync, table

from nanoserve import EngineConfig, LLMEngine, SamplingParams


def run(args, budget: int) -> dict:
    engine = LLMEngine(
        EngineConfig(
            model=args.model,
            device=args.device,
            max_num_seqs=args.decoders + 4,
            max_num_batched_tokens=budget,
            max_model_len=4096,
            enable_prefix_caching=False,
        )
    )
    engine.generate([[5] * 8], SamplingParams(max_tokens=2))
    dev = engine.runner.device
    decoders = [
        engine.add_request(p, SamplingParams(max_tokens=args.decode_tokens, temperature=0, ignore_eos=True))
        for p in random_prompts(args.decoders, 32, 64, seed=3)
    ]
    long_prompts = random_prompts(args.long_prompts, args.long_len, args.long_len, seed=4)
    last_token: dict[str, float] = {}
    gaps: list[float] = []
    step = 0
    start = time.perf_counter()
    while engine.has_unfinished():
        if step >= 8 and step % args.every == 0 and long_prompts:
            engine.add_request(long_prompts.pop(), SamplingParams(max_tokens=1, temperature=0, ignore_eos=True))
        outs = engine.step()
        sync(dev)
        now = time.perf_counter()
        for out in outs:
            if out.request_id in decoders:
                if out.request_id in last_token:
                    gaps.append(now - last_token[out.request_id])
                last_token[out.request_id] = now
        step += 1
    elapsed = time.perf_counter() - start
    itl = summary_ms(gaps)
    return {
        "token_budget": budget if budget < 10**6 else "unlimited (no chunking)",
        "seconds": round(elapsed, 2),
        "itl_p50_ms": itl["p50_ms"],
        "itl_p99_ms": itl["p99_ms"],
        "itl_max_ms": itl["max_ms"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--device", default="auto")
    p.add_argument("--decoders", type=int, default=16)
    p.add_argument("--decode-tokens", type=int, default=160)
    p.add_argument("--long-prompts", type=int, default=6)
    p.add_argument("--long-len", type=int, default=2048)
    p.add_argument("--every", type=int, default=20, help="add a long prompt every N steps")
    p.add_argument("--budgets", type=int, nargs="+", default=[10**7, 512, 256])
    args = p.parse_args()

    rows = [run(args, b) for b in args.budgets]
    device = args.device if args.device != "auto" else str(EngineConfig().resolve_device())
    print(
        f"\n{device_name(device)}, {args.decoders} decoding requests, {args.long_prompts} x {args.long_len}-token prompts arriving"
    )
    print(table(rows, ["token_budget", "seconds", "itl_p50_ms", "itl_p99_ms", "itl_max_ms"]))
    print("saved", save("chunked", device, {"args": vars(args), "results": rows}))


if __name__ == "__main__":
    main()
