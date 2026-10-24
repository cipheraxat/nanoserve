from __future__ import annotations

import torch

from nanoserve.sampler import Sampler, apply_top_k_top_p
from nanoserve.sampling import SamplingParams


def reference_mask(logits: torch.Tensor, k: int, p: float) -> torch.Tensor:
    keep = torch.zeros_like(logits, dtype=torch.bool)
    order = logits.argsort(descending=True)
    probs = logits.softmax(-1)
    mass = 0.0
    for rank, idx in enumerate(order.tolist()):
        if (k and rank >= k) or (rank > 0 and mass >= p):
            break
        keep[idx] = True
        mass += probs[idx].item()
    return keep


def test_top_k_top_p_matches_reference():
    torch.manual_seed(0)
    logits = torch.randn(6, 50)
    ks = [0, 1, 5, 0, 10, 3]
    ps = [1.0, 1.0, 0.5, 0.9, 0.3, 0.99]
    masked = apply_top_k_top_p(logits, torch.tensor(ks), torch.tensor(ps))
    for row in range(6):
        assert torch.equal(masked[row] != float("-inf"), reference_mask(logits[row], ks[row], ps[row]))


def test_greedy_and_top_k_1_are_argmax():
    torch.manual_seed(0)
    logits = torch.randn(4, 100)
    params = [
        SamplingParams(temperature=0),
        SamplingParams(top_k=1),
        SamplingParams(temperature=0),
        SamplingParams(top_k=1),
    ]
    assert Sampler()(logits, list("abcd"), params) == logits.argmax(-1).tolist()


def test_sampling_follows_distribution():
    logits = torch.log(torch.tensor([[0.7, 0.2, 0.1]])).repeat(4000, 1)
    out = torch.tensor(Sampler()(logits, [str(i) for i in range(4000)], [SamplingParams()] * 4000))
    freq = torch.bincount(out, minlength=3).float() / 4000
    torch.testing.assert_close(freq, torch.tensor([0.7, 0.2, 0.1]), atol=0.03, rtol=0)


def test_seed_is_reproducible():
    logits = torch.randn(1, 1000)
    params = [SamplingParams(seed=42)]
    a = [Sampler()(logits, ["x"], params)[0] for _ in range(1)]
    b = [Sampler()(logits, ["x"], params)[0] for _ in range(1)]
    assert a == b
