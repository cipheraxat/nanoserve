from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

from nanoserve.block_manager import BlockManager, cdiv
from nanoserve.config import EngineConfig
from nanoserve.model_runner import ModelRunner
from nanoserve.sampler import Sampler
from nanoserve.sampling import SamplingParams
from nanoserve.scheduler import Scheduler
from nanoserve.sequence import FinishReason, Sequence


@dataclass
class RequestOutput:
    request_id: str
    new_token_ids: list[int]
    output_token_ids: list[int]
    finished: bool
    finish_reason: FinishReason | None
    num_prompt_tokens: int
    num_cached_tokens: int


@dataclass
class EngineStats:
    steps: int = 0
    prompt_tokens: int = 0
    computed_tokens: int = 0
    generated_tokens: int = 0
    finished_requests: int = 0
    step_time_s: float = 0.0
    ttft_s: list[float] = field(default_factory=list)


class LLMEngine:
    """Synchronous engine: `add_request` queues work and each `step` runs one batch.

    `generate` drives the loop for offline use; `AsyncEngine` drives it from a background
    thread for the HTTP server.
    """

    def __init__(self, config: EngineConfig, tokenizer=None, hf_config=None, state_dict=None):
        from transformers import AutoConfig, AutoTokenizer

        self.config = config
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.model)
        hf_config = hf_config or AutoConfig.from_pretrained(config.model)
        self.runner = ModelRunner(config, hf_config, state_dict)
        self.block_manager = BlockManager(self.runner.num_blocks, config.block_size, config.enable_prefix_caching)
        self.scheduler = Scheduler(config, self.block_manager)
        self.sampler = Sampler()
        self.seqs: dict[str, Sequence] = {}
        self.stats = EngineStats()
        self._ids = itertools.count()

        eos = getattr(hf_config, "eos_token_id", None)
        eos_ids = list(eos) if isinstance(eos, list) else [eos]
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(self.tokenizer.eos_token_id)
        self.eos_token_ids = {e for e in eos_ids if e is not None}

    def add_request(
        self,
        prompt: str | list[int],
        params: SamplingParams | None = None,
        request_id: str | None = None,
    ) -> str:
        params = params or SamplingParams()
        request_id = request_id or f"req-{next(self._ids)}"
        if request_id in self.seqs:
            raise ValueError(f"duplicate request id {request_id}")
        token_ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        self.validate_request(token_ids, params)
        seq = Sequence(request_id, token_ids, params)
        self.seqs[request_id] = seq
        self.scheduler.add(seq)
        return request_id

    def validate_request(self, token_ids: list[int], params: SamplingParams) -> None:
        """Reject requests that could never be scheduled. Reads only config, so any thread may call it."""
        if not token_ids:
            raise ValueError("prompt must contain at least one token")
        max_len = len(token_ids) + params.max_tokens
        if max_len > self.config.max_model_len:
            raise ValueError(
                f"prompt ({len(token_ids)} tokens) + max_tokens ({params.max_tokens}) exceeds "
                f"max_model_len ({self.config.max_model_len})"
            )
        if cdiv(max_len, self.config.block_size) > self.block_manager.num_blocks:
            raise ValueError("request would not fit in the KV cache even on its own")

    def abort(self, request_id: str) -> None:
        seq = self.seqs.pop(request_id, None)
        if seq is not None and not seq.is_finished:
            self.scheduler.finish(seq, FinishReason.ABORT)
        self.sampler.forget(request_id)

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    def _finish_reason(self, seq: Sequence, token_id: int) -> FinishReason | None:
        params = seq.params
        if not params.ignore_eos and (token_id in self.eos_token_ids or token_id in params.stop_token_ids):
            return FinishReason.STOP
        if seq.num_output_tokens >= params.max_tokens or seq.num_tokens >= self.config.max_model_len:
            return FinishReason.LENGTH
        return None

    def step(self) -> list[RequestOutput]:
        start = time.perf_counter()
        sched = self.scheduler.schedule()
        if sched.is_empty:
            return []

        logits, sample_seqs = self.runner.execute(sched.scheduled)
        token_ids = (
            self.sampler(logits, [s.request_id for s in sample_seqs], [s.params for s in sample_seqs])
            if sample_seqs
            else []
        )

        for item in sched.scheduled:
            item.seq.num_computed_tokens += item.num_tokens
            self.block_manager.register_full_blocks(item.seq)
        outputs: list[RequestOutput] = []
        for seq, token_id in zip(sample_seqs, token_ids, strict=True):
            first = seq.first_token_time is None
            seq.append_token(token_id)
            if first:
                self.stats.prompt_tokens += seq.num_prompt_tokens
                self.stats.ttft_s.append(seq.first_token_time - seq.arrival_time)
            reason = self._finish_reason(seq, token_id)
            if reason is not None:
                self.scheduler.finish(seq, reason)
                self.sampler.forget(seq.request_id)
                self.stats.finished_requests += 1
            outputs.append(
                RequestOutput(
                    request_id=seq.request_id,
                    new_token_ids=[token_id],
                    output_token_ids=seq.output_token_ids,
                    finished=reason is not None,
                    finish_reason=reason,
                    num_prompt_tokens=seq.num_prompt_tokens,
                    num_cached_tokens=seq.num_cached_tokens,
                )
            )
            if reason is not None:
                del self.seqs[seq.request_id]

        self.stats.steps += 1
        self.stats.computed_tokens += sched.num_batched_tokens
        self.stats.generated_tokens += len(token_ids)
        self.stats.step_time_s += time.perf_counter() - start
        return outputs

    def generate(
        self, prompts: list[str | list[int]], params: SamplingParams | list[SamplingParams] | None = None
    ) -> list[list[int]]:
        """Run prompts to completion and return the generated token ids, in input order."""
        params_list = params if isinstance(params, list) else [params or SamplingParams()] * len(prompts)
        ids = [self.add_request(p, sp) for p, sp in zip(prompts, params_list, strict=True)]
        results: dict[str, list[int]] = {}
        while self.has_unfinished():
            for out in self.step():
                if out.finished:
                    results[out.request_id] = out.output_token_ids
        return [results[i] for i in ids]
