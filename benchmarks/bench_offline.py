"""Offline throughput: nanoserve (continuous batching, paged KV) vs Hugging Face `generate`
with static batching on the same workload: N requests with mixed prompt and output lengths.

Static batching pads every batch to its longest prompt and keeps decoding until its longest
request is done, so short requests waste slots. Only the tokens each request asked for are
counted as useful output.
"""

from __future__ import annotations

import argparse
import time

import torch
from common import device_name, output_lengths, random_prompts, save, sync, table

from nanoserve import EngineConfig, LLMEngine, SamplingParams


def run_nanoserve(args, prompts, out_lens) -> dict:
    engine = LLMEngine(
        EngineConfig(
            model=args.model,
            device=args.device,
            max_num_seqs=args.batch,
            max_num_batched_tokens=args.token_budget,
            kv_cache_memory_gb=args.kv_gb,
            enable_prefix_caching=False,
        )
    )
    engine.generate([prompts[0][:16]], SamplingParams(max_tokens=4))  # warmup
    sync(engine.runner.device)
    params = [SamplingParams(max_tokens=n, temperature=0, ignore_eos=True) for n in out_lens]
    start = time.perf_counter()
    outs = engine.generate(prompts, params)
    sync(engine.runner.device)
    elapsed = time.perf_counter() - start
    assert [len(o) for o in outs] == out_lens
    return {
        "system": "nanoserve (continuous batching)",
        "seconds": round(elapsed, 2),
        "output_tok_per_s": round(sum(out_lens) / elapsed, 1),
        "steps": engine.stats.steps,
        "ttft_p50_s": round(sorted(engine.stats.ttft_s)[len(prompts) // 2], 2),
    }


def run_hf(args, prompts, out_lens) -> dict:
    from transformers import AutoModelForCausalLM

    device = torch.device(args.device if args.device != "auto" else EngineConfig().resolve_device())
    dtype = EngineConfig().resolve_dtype(device)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    pad = 0

    def batch_generate(batch, max_new):
        width = max(len(p) for p in batch)
        ids = torch.tensor([[pad] * (width - len(p)) + p for p in batch], device=device)
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in batch], device=device)
        return model.generate(
            ids, attention_mask=mask, max_new_tokens=max_new, min_new_tokens=max_new, do_sample=False, pad_token_id=pad
        )

    batch_generate([prompts[0][:16]], 4)  # warmup
    sync(device)
    start = time.perf_counter()
    for i in range(0, len(prompts), args.batch):
        batch_generate(prompts[i : i + args.batch], max(out_lens[i : i + args.batch]))
    sync(device)
    elapsed = time.perf_counter() - start
    return {
        "system": "HF transformers generate (static batching)",
        "seconds": round(elapsed, 2),
        "output_tok_per_s": round(sum(out_lens) / elapsed, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--device", default="auto")
    p.add_argument("--num-requests", type=int, default=64)
    p.add_argument("--batch", type=int, default=32, help="max concurrent sequences (both systems)")
    p.add_argument("--prompt-len", type=int, nargs=2, default=[32, 512])
    p.add_argument("--output-len", type=int, nargs=2, default=[16, 256])
    p.add_argument("--token-budget", type=int, default=1024)
    p.add_argument("--kv-gb", type=float, default=2.0)
    args = p.parse_args()

    prompts = random_prompts(args.num_requests, *args.prompt_len)
    out_lens = output_lengths(args.num_requests, *args.output_len)
    rows = [run_nanoserve(args, prompts, out_lens), run_hf(args, prompts, out_lens)]
    speedup = rows[0]["output_tok_per_s"] / rows[1]["output_tok_per_s"]
    device = args.device if args.device != "auto" else str(EngineConfig().resolve_device())
    print(
        f"\n{device_name(device)}, {args.num_requests} requests, prompts {args.prompt_len}, outputs {args.output_len}, batch {args.batch}"
    )
    print(table(rows, ["system", "seconds", "output_tok_per_s"]))
    print(f"speedup: {speedup:.2f}x")
    print("saved", save("offline", device, {"args": vars(args), "results": rows, "speedup": round(speedup, 2)}))


if __name__ == "__main__":
    main()
