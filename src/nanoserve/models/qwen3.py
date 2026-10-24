"""Qwen3 decoder written for packed, paged inference.

Parameter names mirror the Hugging Face checkpoint, except that q/k/v and gate/up are fused
into one matmul each (concatenated at load time), which halves the kernel launches per layer.
The forward pass takes a flat [T] token batch (every sequence's new tokens back to back)
instead of a padded [B, L] batch.
"""

from __future__ import annotations

import torch
from torch import nn

from nanoserve.attention import AttentionMetadata, paged_attention, write_kv


def rope_theta_of(config) -> float:
    theta = getattr(config, "rope_theta", None)
    if theta is None:
        theta = (getattr(config, "rope_parameters", None) or {}).get("rope_theta", 10000.0)
    return float(theta)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_positions: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        freqs = torch.outer(torch.arange(max_positions, dtype=torch.float32), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor):
        cos = self.cos.index_select(0, positions).unsqueeze(1).to(q.dtype)  # [T, 1, D]
        sin = self.sin.index_select(0, positions).unsqueeze(1).to(q.dtype)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.num_heads
        bias = getattr(config, "attention_bias", False)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(config.hidden_size, self.q_size + 2 * self.kv_size, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=bias)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        rope: RotaryEmbedding,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        meta: AttentionMetadata,
    ) -> torch.Tensor:
        num_tokens = hidden.shape[0]
        q, k, v = self.qkv_proj(hidden).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.reshape(num_tokens, self.num_heads, self.head_dim))
        k = self.k_norm(k.reshape(num_tokens, self.num_kv_heads, self.head_dim))
        v = v.reshape(num_tokens, self.num_kv_heads, self.head_dim)
        q, k = rope(positions, q, k)
        write_kv(k_cache, v_cache, k, v, meta.slot_mapping)
        out = paged_attention(q, k_cache, v_cache, meta, self.scale)
        return self.o_proj(out.reshape(num_tokens, self.num_heads * self.head_dim))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(nn.functional.silu(gate) * up)


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden, positions, rope, k_cache, v_cache, meta):
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), positions, rope, k_cache, v_cache, meta)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config, max_positions: int):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        head_dim = self.model.layers[0].self_attn.head_dim
        self.rope = RotaryEmbedding(head_dim, max_positions, rope_theta_of(config))

    @property
    def num_kv_heads(self) -> int:
        return self.config.num_key_value_heads

    @property
    def head_dim(self) -> int:
        return self.model.layers[0].self_attn.head_dim

    def load_hf_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        state_dict = dict(state_dict)
        if "lm_head.weight" not in state_dict or getattr(self.config, "tie_word_embeddings", False):
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
        fused = {"self_attn.qkv_proj": ("q_proj", "k_proj", "v_proj"), "mlp.gate_up_proj": ("gate_proj", "up_proj")}
        for i in range(len(self.model.layers)):
            for target, parts in fused.items():
                module, _ = target.split(".")
                for suffix in ("weight", "bias"):
                    names = [f"model.layers.{i}.{module}.{part}.{suffix}" for part in parts]
                    if all(n in state_dict for n in names):
                        state_dict[f"model.layers.{i}.{target}.{suffix}"] = torch.cat(
                            [state_dict.pop(n) for n in names]
                        )
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        missing = [m for m in missing if not m.startswith("rope.")]
        if missing or unexpected:
            raise ValueError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")

    def forward(
        self,
        input_ids: torch.Tensor,  # [T]
        positions: torch.Tensor,  # [T]
        kv_cache: torch.Tensor,  # [layers, 2, blocks, block_size, H_kv, D]
        meta: AttentionMetadata,
        logits_index: torch.Tensor,  # [S] packed positions that need next-token logits
    ) -> torch.Tensor:
        hidden = self.model.embed_tokens(input_ids)
        for layer_id, layer in enumerate(self.model.layers):
            hidden = layer(hidden, positions, self.rope, kv_cache[layer_id, 0], kv_cache[layer_id, 1], meta)
        hidden = self.model.norm(hidden.index_select(0, logits_index))
        return self.compute_logits(hidden)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.device.type == "mps" and hidden.shape[0] > 8:
            # MPS falls off a fast path for [>8, hidden] x [hidden, ~150k] matmuls (about 8x
            # slower at 32 rows). Slicing the vocab keeps every matmul on the fast path.
            slices = self.lm_head.weight.chunk(16)
            return torch.cat([nn.functional.linear(hidden, w) for w in slices], dim=1).float()
        return self.lm_head(hidden).float()
