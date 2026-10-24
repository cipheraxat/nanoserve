from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

from nanoserve.sampling import SamplingParams


class SequenceStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class FinishReason(str, enum.Enum):
    STOP = "stop"
    LENGTH = "length"
    ABORT = "abort"


@dataclass
class Sequence:
    """One request. `token_ids` is prompt + generated tokens.

    `num_computed_tokens` counts tokens whose K/V already sit in the paged cache, either
    because the model ran on them or because they were matched in the prefix cache.
    Everything after that index still has to go through the model.
    """

    request_id: str
    prompt_token_ids: list[int]
    params: SamplingParams
    token_ids: list[int] = field(init=False)
    num_computed_tokens: int = 0
    num_cached_tokens: int = 0  # prompt tokens served from the prefix cache on admission
    num_registered_blocks: int = 0  # leading full blocks already published to the prefix cache
    block_table: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    finish_reason: FinishReason | None = None
    arrival_time: float = field(default_factory=time.perf_counter)
    first_token_time: float | None = None
    finish_time: float | None = None

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("prompt must contain at least one token")
        self.token_ids = list(self.prompt_token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_output_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def num_uncomputed_tokens(self) -> int:
        return self.num_tokens - self.num_computed_tokens

    @property
    def is_finished(self) -> bool:
        return self.status is SequenceStatus.FINISHED

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
