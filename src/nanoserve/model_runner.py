from __future__ import annotations

import logging
import os

import torch

from nanoserve.attention import AttentionMetadata, DecodeAttentionMetadata, GatherAttentionMetadata
from nanoserve.config import EngineConfig
from nanoserve.models.qwen3 import Qwen3ForCausalLM, RotaryEmbedding, rope_theta_of
from nanoserve.scheduler import ScheduledSeq
from nanoserve.sequence import Sequence

logger = logging.getLogger(__name__)


def triton_available(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def load_hf_weights(model_id: str) -> dict[str, torch.Tensor]:
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    path = (
        model_id if os.path.isdir(model_id) else snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"])
    )
    state_dict: dict[str, torch.Tensor] = {}
    for name in sorted(os.listdir(path)):
        if name.endswith(".safetensors"):
            state_dict.update(load_file(os.path.join(path, name)))
    if not state_dict:
        raise FileNotFoundError(f"no .safetensors weights under {path}")
    return state_dict


class ModelRunner:
    """Owns the model weights and the paged KV cache, and turns a scheduled batch into one
    forward pass."""

    def __init__(self, config: EngineConfig, hf_config, state_dict: dict[str, torch.Tensor] | None = None):
        self.config = config
        self.device = config.resolve_device()
        self.dtype = config.resolve_dtype(self.device)
        self.block_size = config.block_size

        with torch.device("meta"):
            model = Qwen3ForCausalLM(hf_config, max_positions=config.max_model_len)
        model = model.to_empty(device=self.device).to(self.dtype)
        model.load_hf_state_dict(state_dict if state_dict is not None else load_hf_weights(config.model))
        model.rope = RotaryEmbedding(model.head_dim, config.max_model_len, rope_theta_of(hf_config)).to(self.device)
        self.model = model.eval()

        num_layers = hf_config.num_hidden_layers
        per_block = num_layers * 2 * config.block_size * model.num_kv_heads * model.head_dim * self.dtype.itemsize
        self.num_blocks = config.num_kv_blocks or max(1, int(config.kv_cache_memory_gb * (1 << 30)) // per_block)
        self.kv_cache = torch.zeros(
            num_layers,
            2,
            self.num_blocks,
            config.block_size,
            model.num_kv_heads,
            model.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        backend = config.attention_backend
        if backend == "triton" and not triton_available(self.device):
            raise RuntimeError("attention_backend='triton' needs CUDA and the triton package")
        self.use_triton = backend == "triton" or (backend == "auto" and triton_available(self.device))
        logger.info(
            "model on %s/%s, %d KV blocks (%.2f GiB), triton decode=%s",
            self.device,
            self.dtype,
            self.num_blocks,
            self.kv_cache.numel() * self.dtype.itemsize / (1 << 30),
            self.use_triton,
        )

    def _to_device(self, data, dtype=torch.int64) -> torch.Tensor:
        return torch.tensor(data, dtype=dtype).to(self.device, non_blocking=True)

    def _gather_meta(self, group: list[tuple[int, int, int, list[int]]], packed: bool) -> GatherAttentionMetadata:
        """Metadata for the torch path. `group` holds (packed start, q_len, ctx_len, block table)."""
        bs, dev = self.block_size, self.device
        q_lens = [g[1] for g in group]
        ctx_lens = [g[2] for g in group]
        max_q, max_ctx = max(q_lens), max(ctx_lens)
        max_blocks = max(len(g[3]) for g in group)
        tables = self._to_device([g[3] + [0] * (max_blocks - len(g[3])) for g in group])
        output_index = [i * max_q + j for i, n in enumerate(q_lens) for j in range(n)]

        q_len_t = self._to_device(q_lens)
        ctx_t = self._to_device(ctx_lens)
        q_start = torch.cumsum(q_len_t, 0) - q_len_t
        j = torch.arange(max_q, device=dev)
        query_valid = j.unsqueeze(0) < q_len_t.unsqueeze(1)
        query_index = torch.where(query_valid, q_start.unsqueeze(1) + j.unsqueeze(0), 0)
        query_pos = (ctx_t - q_len_t).unsqueeze(1) + j.unsqueeze(0)
        k_pos = torch.arange(max_ctx, device=dev)
        attn_mask = (k_pos.view(1, 1, -1) <= query_pos.unsqueeze(2)) & (k_pos.view(1, 1, -1) < ctx_t.view(-1, 1, 1))
        block_idx = (k_pos // bs).clamp(max=max_blocks - 1).unsqueeze(0).expand(len(group), -1)
        key_slots = tables.gather(1, block_idx) * bs + (k_pos % bs).unsqueeze(0)
        token_index = None
        if not packed:
            token_index = self._to_device([g[0] + j for g in group for j in range(g[1])])
        return GatherAttentionMetadata(
            query_index=query_index,
            output_index=self._to_device(output_index),
            key_slots=key_slots,
            attn_mask=attn_mask.unsqueeze(1),
            token_index=token_index,
        )

    def _decode_meta(self, group: list[tuple[int, int, int, list[int]]], packed: bool) -> DecodeAttentionMetadata:
        max_blocks = max(len(g[3]) for g in group)
        return DecodeAttentionMetadata(
            block_tables=self._to_device([g[3] + [0] * (max_blocks - len(g[3])) for g in group], torch.int32),
            context_lens=self._to_device([g[2] for g in group], torch.int32),
            token_index=None if packed else self._to_device([g[0] for g in group]),
        )

    def prepare(self, scheduled: list[ScheduledSeq]):
        bs = self.block_size
        input_ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        logits_index: list[int] = []
        sample_seqs: list[Sequence] = []
        decodes: list[tuple[int, int, int, list[int]]] = []
        others: list[tuple[int, int, int, list[int]]] = []

        for item in scheduled:
            seq, n = item.seq, item.num_tokens
            start = seq.num_computed_tokens
            table = seq.block_table
            entry = (len(input_ids), n, start + n, table)
            (decodes if n == 1 and self.use_triton else others).append(entry)
            input_ids.extend(seq.token_ids[start : start + n])
            positions.extend(range(start, start + n))
            slots.extend(table[p // bs] * bs + p % bs for p in range(start, start + n))
            if start + n == seq.num_tokens:
                logits_index.append(len(input_ids) - 1)
                sample_seqs.append(seq)

        # Each group covers the whole packed batch only when the other group is empty.
        meta = AttentionMetadata(
            slot_mapping=self._to_device(slots),
            decode=self._decode_meta(decodes, packed=not others) if decodes else None,
            gather=self._gather_meta(others, packed=not decodes) if others else None,
        )
        return (
            self._to_device(input_ids),
            self._to_device(positions),
            meta,
            self._to_device(logits_index),
            sample_seqs,
        )

    @torch.inference_mode()
    def execute(self, scheduled: list[ScheduledSeq]) -> tuple[torch.Tensor, list[Sequence]]:
        # Only sequences whose last uncomputed token is in this step get logits; a sequence
        # in the middle of a chunked prefill only fills its cache.
        input_ids, positions, meta, logits_index, sample_seqs = self.prepare(scheduled)
        logits = self.model(input_ids, positions, self.kv_cache, meta, logits_index)
        return logits, sample_seqs
