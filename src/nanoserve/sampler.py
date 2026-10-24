from __future__ import annotations

import torch

from nanoserve.sampling import SamplingParams


def apply_top_k_top_p(logits: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """Mask logits outside each row's top-k and top-p set with -inf.

    One descending sort serves both filters. Top-p keeps the smallest prefix whose
    probability mass reaches p, and always keeps the most likely token.
    """
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    vocab = logits.shape[-1]
    ranks = torch.arange(vocab, device=logits.device).unsqueeze(0)

    k = torch.where(top_k > 0, top_k, torch.full_like(top_k, vocab)).unsqueeze(1)
    remove = ranks >= k

    probs = sorted_logits.softmax(dim=-1)
    mass_before = probs.cumsum(dim=-1) - probs
    remove |= mass_before >= top_p.unsqueeze(1)
    remove[:, 0] = False

    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.empty_like(logits).scatter_(1, sorted_idx, sorted_logits)


class Sampler:
    """Batched sampling for a step. Greedy rows use argmax; the rest are sampled with the
    exponential race trick (argmax of p / Exp(1)), which equals sampling from p. Rows with a
    seed draw their noise from a per-request CPU generator so results are reproducible
    regardless of what else is in the batch."""

    def __init__(self) -> None:
        self._generators: dict[str, torch.Generator] = {}

    def forget(self, request_id: str) -> None:
        self._generators.pop(request_id, None)

    def _generator(self, request_id: str, seed: int) -> torch.Generator:
        gen = self._generators.get(request_id)
        if gen is None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)
            self._generators[request_id] = gen
        return gen

    def __call__(self, logits: torch.Tensor, request_ids: list[str], params: list[SamplingParams]) -> list[int]:
        device = logits.device
        greedy = torch.tensor([p.greedy for p in params], device=device)
        if bool(greedy.all()):
            return logits.argmax(dim=-1).tolist()

        temperature = torch.tensor([p.temperature if not p.greedy else 1.0 for p in params], device=device)
        top_k = torch.tensor([p.top_k for p in params], device=device)
        top_p = torch.tensor([p.top_p for p in params], device=device, dtype=torch.float32)

        scaled = logits / temperature.unsqueeze(1)
        if bool((top_k > 0).any()) or bool((top_p < 1).any()):
            scaled = apply_top_k_top_p(scaled, top_k, top_p)
        probs = scaled.softmax(dim=-1)

        noise = torch.empty_like(probs).exponential_()
        for row, (rid, p) in enumerate(zip(request_ids, params, strict=True)):
            if p.seed is not None and not p.greedy:
                gen = self._generator(rid, p.seed)
                noise[row] = torch.empty(probs.shape[1]).exponential_(generator=gen).to(device)
        sampled = (probs / noise).argmax(dim=-1)
        sampled = torch.where(greedy, logits.argmax(dim=-1), sampled)
        return sampled.tolist()
