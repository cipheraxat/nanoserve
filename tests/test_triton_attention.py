"""Checks the Triton decode kernel against the torch gather path. Needs CUDA and triton."""

from __future__ import annotations

import pytest
import torch

triton = pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block_size", [16, 32])
def test_paged_decode_matches_torch(dtype, block_size):
    from nanoserve.attention import GatherAttentionMetadata, paged_attention_torch
    from nanoserve.triton_attention import paged_decode_attention

    torch.manual_seed(0)
    dev = "cuda"
    num_blocks, h_q, h_kv, d = 128, 16, 8, 128
    ctx = torch.tensor([1, 17, 100, 333, 64], device=dev)
    batch = len(ctx)
    k_cache = torch.randn(num_blocks, block_size, h_kv, d, device=dev, dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    max_blocks = int((ctx.max() + block_size - 1) // block_size)
    perm = torch.randperm(num_blocks, device=dev)[: batch * max_blocks].view(batch, max_blocks).int()
    q = torch.randn(batch, h_q, d, device=dev, dtype=dtype)

    k_pos = torch.arange(int(ctx.max()), device=dev)
    key_slots = perm.long().gather(1, (k_pos // block_size).expand(batch, -1)) * block_size + k_pos % block_size
    meta = GatherAttentionMetadata(
        query_index=torch.arange(batch, device=dev).view(batch, 1),
        output_index=torch.arange(batch, device=dev),
        key_slots=key_slots,
        attn_mask=(k_pos.view(1, 1, 1, -1) < ctx.view(-1, 1, 1, 1)),
    )
    ref = paged_attention_torch(q, k_cache, v_cache, meta, d**-0.5)
    out = paged_decode_attention(q, k_cache, v_cache, perm, ctx.int(), d**-0.5)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_engine_with_triton_matches_hf(make_engine, tiny_hf):
    """Mixed steps (prefill chunks next to decodes) route decodes through the kernel."""
    from conftest import hf_greedy

    from nanoserve import SamplingParams

    engine = make_engine(device="cuda", block_size=16, max_num_batched_tokens=24, attention_backend="triton")
    assert engine.runner.use_triton
    prompts = [[(7 * i + j) % 500 + 1 for j in range(5 + 9 * i)] for i in range(6)]
    outs = engine.generate(prompts, SamplingParams(max_tokens=12, temperature=0, ignore_eos=True))
    for prompt, out in zip(prompts, outs, strict=True):
        assert out == hf_greedy(tiny_hf, prompt, 12)
