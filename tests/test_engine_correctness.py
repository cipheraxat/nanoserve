"""End-to-end correctness: whatever the batching, chunking, caching or preemption, greedy
decoding must produce exactly the tokens Hugging Face produces for each prompt alone."""

from __future__ import annotations

import random

import pytest
import torch
from conftest import hf_greedy

from nanoserve import SamplingParams
from nanoserve.scheduler import ScheduledSeq
from nanoserve.sequence import Sequence

MAX_TOKENS = 12


def prompts(n: int, seed: int = 0, lo: int = 1, hi: int = 40) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(1, 500) for _ in range(rng.randint(lo, hi))] for _ in range(n)]


def check_matches_hf(engine, tiny_hf, batch: list[list[int]]) -> None:
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0, ignore_eos=True)
    outputs = engine.generate(batch, params)
    for prompt, out in zip(batch, outputs, strict=True):
        assert out == hf_greedy(tiny_hf, prompt, MAX_TOKENS)


def test_prefill_logits_match_hf(make_engine, tiny_hf):
    engine = make_engine()
    prompt = prompts(1, seed=3, lo=20, hi=20)[0]
    seq = Sequence("r", prompt, SamplingParams())
    engine.block_manager.append_slots(seq, len(prompt))
    logits, _ = engine.runner.execute([ScheduledSeq(seq, len(prompt))])
    with torch.no_grad():
        ref = tiny_hf(torch.tensor([prompt])).logits[0, -1]
    torch.testing.assert_close(logits[0], ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("block_size", [1, 4, 16])
def test_continuous_batching_matches_hf(make_engine, tiny_hf, block_size):
    check_matches_hf(make_engine(block_size=block_size), tiny_hf, prompts(6, seed=block_size))


def test_chunked_prefill_matches_hf(make_engine, tiny_hf):
    engine = make_engine(max_num_batched_tokens=8, max_num_seqs=4)
    check_matches_hf(engine, tiny_hf, prompts(5, seed=1, lo=20, hi=60))
    assert engine.stats.steps > 0


def test_preemption_matches_hf(make_engine, tiny_hf):
    # 20 blocks of 4 tokens cannot hold six 30-token sequences at once.
    engine = make_engine(num_kv_blocks=20, max_num_seqs=6)
    check_matches_hf(engine, tiny_hf, prompts(6, seed=2, lo=25, hi=30))
    assert engine.scheduler.num_preemptions > 0


def test_prefix_caching_matches_hf_and_hits(make_engine, tiny_hf):
    engine = make_engine()
    shared = prompts(1, seed=4, lo=33, hi=33)[0]
    batch = [shared + tail for tail in prompts(4, seed=5, lo=1, hi=10)]
    check_matches_hf(engine, tiny_hf, batch[:1])  # warm the cache
    check_matches_hf(engine, tiny_hf, batch[1:])
    assert engine.block_manager.prefix_hit_tokens >= 3 * 32


def test_full_prompt_cache_hit_still_computes_last_token(make_engine, tiny_hf):
    engine = make_engine()
    prompt = prompts(1, seed=6, lo=16, hi=16)[0]  # exactly four blocks
    check_matches_hf(engine, tiny_hf, [prompt])
    check_matches_hf(engine, tiny_hf, [prompt])
    assert engine.block_manager.prefix_hit_tokens == 12  # three blocks; the last is recomputed


def test_prefix_caching_disabled(make_engine, tiny_hf):
    engine = make_engine(enable_prefix_caching=False)
    shared = prompts(1, seed=7, lo=20, hi=20)[0]
    check_matches_hf(engine, tiny_hf, [shared + [1], shared + [2]])
    assert engine.block_manager.prefix_hit_tokens == 0


def test_tied_embeddings_checkpoint(make_engine, tiny_hf):
    state = {k: v for k, v in tiny_hf.state_dict().items() if k != "lm_head.weight"}
    engine = make_engine()
    engine.runner.model.load_hf_state_dict(state)
    assert torch.equal(engine.runner.model.lm_head.weight, tiny_hf.model.embed_tokens.weight)


def test_stop_on_eos_and_max_tokens(make_engine, tiny_hf):
    prompt = prompts(1, seed=8)[0]
    expected = hf_greedy(tiny_hf, prompt, MAX_TOKENS)
    engine = make_engine()
    out = engine.generate([prompt], SamplingParams(max_tokens=MAX_TOKENS, temperature=0, stop_token_ids=[expected[3]]))
    assert out[0] == expected[: expected.index(expected[3]) + 1]
    assert engine.generate([prompt], SamplingParams(max_tokens=2, temperature=0, ignore_eos=True))[0] == expected[:2]


def test_rejects_requests_that_cannot_fit(make_engine):
    engine = make_engine(num_kv_blocks=4, max_model_len=512)
    with pytest.raises(ValueError, match="KV cache"):
        engine.add_request(list(range(1, 20)), SamplingParams(max_tokens=10))
    with pytest.raises(ValueError, match="max_model_len"):
        engine.add_request(list(range(1, 20)), SamplingParams(max_tokens=600))


def test_abort_frees_blocks(make_engine):
    engine = make_engine()
    rid = engine.add_request(list(range(1, 30)), SamplingParams(max_tokens=50, ignore_eos=True))
    engine.step()
    assert engine.block_manager.num_free_blocks < engine.block_manager.num_blocks
    engine.abort(rid)
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks
    assert not engine.has_unfinished()


def test_seeded_sampling_is_reproducible_across_batches(make_engine):
    params = SamplingParams(max_tokens=10, temperature=0.8, top_p=0.9, seed=1234, ignore_eos=True)
    prompt = prompts(1, seed=9)[0]
    alone = make_engine().generate([prompt], params)[0]
    batched = make_engine().generate(
        [prompt] + prompts(3, seed=10), [params] + [SamplingParams(max_tokens=10, temperature=1.0)] * 3
    )[0]
    assert alone == batched


def test_mixed_decode_and_prefill_routing(make_engine, tiny_hf, monkeypatch):
    """With the decode kernel on, a step can hold decodes (kernel) next to prefill chunks
    (gather path). Swap the kernel for a torch reference so the split and the scatter back
    into the packed batch are checked on any machine."""
    import nanoserve.attention as attention

    def reference_decode(q, k_cache, v_cache, meta, scale):
        block_size = k_cache.shape[1]
        max_ctx = int(meta.context_lens.max())
        k_pos = torch.arange(max_ctx)
        blocks = meta.block_tables.long().gather(
            1, (k_pos // block_size).clamp(max=meta.block_tables.shape[1] - 1).expand(len(q), -1)
        )
        gather = attention.GatherAttentionMetadata(
            query_index=torch.arange(len(q)).view(-1, 1),
            output_index=torch.arange(len(q)),
            key_slots=blocks * block_size + k_pos % block_size,
            attn_mask=(k_pos.view(1, 1, 1, -1) < meta.context_lens.view(-1, 1, 1, 1)),
        )
        calls.append(len(q))
        return attention.paged_attention_torch(q, k_cache, v_cache, gather, scale)

    calls: list[int] = []
    monkeypatch.setattr(attention, "_decode", reference_decode)
    engine = make_engine(max_num_batched_tokens=10, max_num_seqs=6)
    engine.runner.use_triton = True
    check_matches_hf(engine, tiny_hf, prompts(6, seed=11, lo=5, hi=40))
    assert calls, "decode path never used"
