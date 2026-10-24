"""Decode attention microbenchmark (CUDA): the Triton paged kernel, which reads K/V through
the block table, vs the torch path, which gathers K/V into a padded buffer and calls SDPA."""

from __future__ import annotations

import argparse

import torch
from common import device_name, save, table

from nanoserve.attention import GatherAttentionMetadata, paged_attention_torch
from nanoserve.triton_attention import paged_decode_attention


def setup(batch, ctx_len, block_size=16, h_q=16, h_kv=8, d=128, dtype=torch.bfloat16):
    dev = "cuda"
    blocks_per_seq = (ctx_len + block_size - 1) // block_size
    num_blocks = batch * blocks_per_seq + 1
    k_cache = torch.randn(num_blocks, block_size, h_kv, d, device=dev, dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    tables = torch.randperm(num_blocks, device=dev)[: batch * blocks_per_seq].view(batch, blocks_per_seq).int()
    ctx = torch.full((batch,), ctx_len, device=dev, dtype=torch.int32)
    q = torch.randn(batch, h_q, d, device=dev, dtype=dtype)
    k_pos = torch.arange(ctx_len, device=dev)
    key_slots = tables.long().gather(1, (k_pos // block_size).expand(batch, -1)) * block_size + k_pos % block_size
    meta = GatherAttentionMetadata(
        query_index=torch.arange(batch, device=dev).view(batch, 1),
        output_index=torch.arange(batch, device=dev),
        key_slots=key_slots,
        attn_mask=(k_pos.view(1, 1, 1, -1) < ctx.view(-1, 1, 1, 1)),
    )
    return q, k_cache, v_cache, meta, tables, ctx


def time_ms(fn, iters=50) -> float:
    for _ in range(5):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


SCALE = 128**-0.5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64])
    p.add_argument("--ctx", type=int, nargs="+", default=[512, 2048, 8192])
    args = p.parse_args()
    rows = []
    for batch in args.batches:
        for ctx in args.ctx:
            q, k, v, meta, tables, ctx_t = setup(batch, ctx)
            ref = paged_attention_torch(q, k, v, meta, SCALE)
            out = paged_decode_attention(q, k, v, tables, ctx_t, SCALE)
            err = (out.float() - ref.float()).abs().max().item()
            t_torch = time_ms(lambda q=q, k=k, v=v, meta=meta: paged_attention_torch(q, k, v, meta, SCALE))
            t_triton = time_ms(
                lambda q=q, k=k, v=v, tables=tables, ctx_t=ctx_t: paged_decode_attention(q, k, v, tables, ctx_t, SCALE)
            )
            rows.append(
                {
                    "batch": batch,
                    "context": ctx,
                    "torch_gather_ms": round(t_torch, 3),
                    "triton_paged_ms": round(t_triton, 3),
                    "speedup": f"{t_torch / t_triton:.1f}x",
                    "max_abs_err": f"{err:.1e}",
                }
            )
    print(f"\n{device_name('cuda')}, Qwen3-0.6B attention shape (16 q heads, 8 kv heads, dim 128), bf16, one layer")
    print(table(rows, ["batch", "context", "torch_gather_ms", "triton_paged_ms", "speedup", "max_abs_err"]))
    print("saved", save("attention", "cuda", {"results": rows}))


if __name__ == "__main__":
    main()
