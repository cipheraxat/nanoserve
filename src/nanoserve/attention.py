"""Paged attention over a block-structured KV cache.

All sequences in a step are packed into one flat token dimension (no padding in the
projections or MLP). Attention is the only place that needs per-sequence structure, so the
model runner builds `AttentionMetadata` once per step and every layer reuses it.

Two backends:
  * torch: gathers each sequence's keys and values from its block table into a padded
    [batch, context, heads, dim] view and runs SDPA with a causal mask that starts at each
    sequence's cached length. Works for any mix of prefill chunks and decodes, on CPU, MPS
    and CUDA.
  * triton: a decode kernel that reads K/V straight from the block table with an online
    softmax, so nothing is gathered. Used on CUDA for decode-only steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AttentionMetadata:
    slot_mapping: torch.Tensor  # [T] flat cache slot for every new token
    query_index: torch.Tensor  # [B, Lq] packed index of each sequence's queries (padded)
    output_index: torch.Tensor  # [T] position of each packed query in the flattened [B * Lq] output
    key_slots: torch.Tensor  # [B, Lk] flat cache slot of every context token (padded)
    attn_mask: torch.Tensor  # [B, 1, Lq, Lk] bool, True where the query may attend
    block_tables: torch.Tensor  # [B, max_blocks] int32, padded with 0
    context_lens: torch.Tensor  # [B] int32, cached + new tokens
    is_decode_only: bool
    use_triton_decode: bool = False


def write_kv(
    k_cache: torch.Tensor, v_cache: torch.Tensor, k: torch.Tensor, v: torch.Tensor, slots: torch.Tensor
) -> None:
    """Scatter this step's keys/values ([T, H_kv, D]) into the paged cache."""
    num_blocks, block_size, num_heads, head_dim = k_cache.shape
    k_cache.view(num_blocks * block_size, num_heads, head_dim).index_copy_(0, slots, k)
    v_cache.view(num_blocks * block_size, num_heads, head_dim).index_copy_(0, slots, v)


def paged_attention_torch(
    q: torch.Tensor,  # [T, H_q, D]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    meta: AttentionMetadata,
    scale: float,
) -> torch.Tensor:
    num_blocks, block_size, num_kv_heads, head_dim = k_cache.shape
    num_q_heads = q.shape[1]

    batch, max_q = meta.query_index.shape
    max_ctx = meta.key_slots.shape[1]
    # index_select rather than advanced indexing: on MPS it is several times faster.
    flat_k = k_cache.view(num_blocks * block_size, num_kv_heads, head_dim)
    flat_v = v_cache.view(num_blocks * block_size, num_kv_heads, head_dim)
    keys = flat_k.index_select(0, meta.key_slots.view(-1)).view(batch, max_ctx, num_kv_heads, head_dim)
    values = flat_v.index_select(0, meta.key_slots.view(-1)).view(batch, max_ctx, num_kv_heads, head_dim)
    queries = q.index_select(0, meta.query_index.view(-1)).view(batch, max_q, num_q_heads, head_dim)

    out = F.scaled_dot_product_attention(
        queries.transpose(1, 2),  # [B, H_q, Lq, D]
        keys.transpose(1, 2),
        values.transpose(1, 2),
        attn_mask=meta.attn_mask,
        scale=scale,
        enable_gqa=num_q_heads != num_kv_heads,
    )
    out = out.transpose(1, 2).reshape(batch * max_q, num_q_heads, head_dim)
    return out.index_select(0, meta.output_index)  # back to packed [T, H_q, D]


def paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    meta: AttentionMetadata,
    scale: float,
) -> torch.Tensor:
    if meta.use_triton_decode:
        from nanoserve.triton_attention import paged_decode_attention

        return paged_decode_attention(q, k_cache, v_cache, meta.block_tables, meta.context_lens, scale)
    return paged_attention_torch(q, k_cache, v_cache, meta, scale)
