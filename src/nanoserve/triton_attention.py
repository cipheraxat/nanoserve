"""Triton paged-attention kernel for decode steps.

One program per (sequence, query head). It walks the sequence's block table, loads one KV
block at a time straight from the paged cache, and keeps a running max and sum (online
softmax), so keys and values are never gathered into a contiguous buffer. Grouped-query
attention is handled by mapping each query head to its KV head.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    out_ptr,
    scale,
    stride_q_tok,
    stride_q_head,
    stride_kv_block,
    stride_kv_slot,
    stride_kv_head,
    stride_bt_seq,
    stride_o_tok,
    stride_o_head,
    GROUP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // GROUP

    offs_d = tl.arange(0, HEAD_DIM)
    offs_s = tl.arange(0, BLOCK_SIZE)
    q = tl.load(q_ptr + seq * stride_q_tok + head * stride_q_head + offs_d).to(tl.float32)

    context_len = tl.load(context_lens_ptr + seq)
    num_blocks = tl.cdiv(context_len, BLOCK_SIZE)

    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for i in range(0, num_blocks):
        block = tl.load(block_tables_ptr + seq * stride_bt_seq + i).to(tl.int64)
        valid = i * BLOCK_SIZE + offs_s < context_len
        kv_offs = (
            block * stride_kv_block + offs_s[:, None] * stride_kv_slot + kv_head * stride_kv_head + offs_d[None, :]
        )
        k = tl.load(k_cache_ptr + kv_offs, mask=valid[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * scale
        scores = tl.where(valid, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        v = tl.load(v_cache_ptr + kv_offs, mask=valid[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    out = acc / l_i
    tl.store(out_ptr + seq * stride_o_tok + head * stride_o_head + offs_d, out.to(out_ptr.dtype.element_ty))


def paged_decode_attention(
    q: torch.Tensor,  # [B, H_q, D], one query token per sequence
    k_cache: torch.Tensor,  # [num_blocks, block_size, H_kv, D]
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,  # [B, max_blocks] int32
    context_lens: torch.Tensor,  # [B] int32
    scale: float,
) -> torch.Tensor:
    batch, num_q_heads, head_dim = q.shape
    _, block_size, num_kv_heads, _ = k_cache.shape
    assert k_cache.stride() == v_cache.stride()
    q = q.contiguous()
    out = torch.empty_like(q)
    _paged_decode_kernel[(batch, num_q_heads)](
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        out,
        scale,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        block_tables.stride(0),
        out.stride(0),
        out.stride(1),
        GROUP=num_q_heads // num_kv_heads,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        num_warps=4,
    )
    return out
