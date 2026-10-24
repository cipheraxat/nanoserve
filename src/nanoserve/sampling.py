from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # 0 disables top-k
    seed: int | None = None
    ignore_eos: bool = False
    stop_token_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0
