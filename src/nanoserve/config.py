from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EngineConfig:
    """Knobs for one engine instance.

    The token budget (`max_num_batched_tokens`) is what makes chunked prefill work: every
    step schedules at most this many new tokens, so a long prompt is split across steps
    instead of stalling the decodes that share the batch.
    """

    model: str = "Qwen/Qwen3-0.6B"
    device: str = "auto"
    dtype: str = "auto"
    block_size: int = 16
    num_kv_blocks: int | None = None
    kv_cache_memory_gb: float = 1.0
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 512
    max_model_len: int = 4096
    enable_prefix_caching: bool = True
    attention_backend: str = "auto"  # "torch" or "triton" (CUDA decode steps only)

    def __post_init__(self) -> None:
        if self.block_size <= 0 or self.block_size & (self.block_size - 1):
            raise ValueError(f"block_size must be a power of two, got {self.block_size}")
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                "max_num_batched_tokens must be at least max_num_seqs so every running sequence can decode in one step"
            )
        if self.attention_backend not in ("auto", "torch", "triton"):
            raise ValueError(f"unknown attention_backend {self.attention_backend!r}")

    def resolve_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def resolve_dtype(self, device: torch.device) -> torch.dtype:
        if self.dtype != "auto":
            return {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[self.dtype]
        return torch.float32 if device.type == "cpu" else torch.bfloat16
