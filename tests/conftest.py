"""Shared fixtures. Tests build a tiny random Qwen3 with Hugging Face transformers and load
the same weights into nanoserve, so the HF implementation is the reference and nothing
is downloaded."""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from nanoserve import EngineConfig, LLMEngine

VOCAB = 512


class StubTokenizer:
    """Enough of a tokenizer for the engine and server: ids are rendered as 't<id> '."""

    eos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [(ord(c) % (VOCAB - 2)) + 1 for c in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(f"t{i} " for i in ids)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **_):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"
        return self.encode(text) if tokenize else text


@pytest.fixture(scope="session")
def tiny_hf():
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=1024,
        tie_word_embeddings=False,
        initializer_range=0.1,
    )
    return Qwen3ForCausalLM(config).eval()


@pytest.fixture
def make_engine(tiny_hf):
    def make(**overrides) -> LLMEngine:
        cfg = dict(
            model="tiny",
            device="cpu",
            dtype="float32",
            block_size=4,
            num_kv_blocks=256,
            max_num_seqs=8,
            max_num_batched_tokens=64,
            max_model_len=512,
        )
        cfg.update(overrides)
        return LLMEngine(
            EngineConfig(**cfg), tokenizer=StubTokenizer(), hf_config=tiny_hf.config, state_dict=tiny_hf.state_dict()
        )

    return make


def hf_greedy(model, prompt: list[int], max_tokens: int) -> list[int]:
    with torch.no_grad():
        out = model.generate(
            torch.tensor([prompt]), max_new_tokens=max_tokens, min_new_tokens=max_tokens, do_sample=False
        )
    return out[0, len(prompt) :].tolist()
