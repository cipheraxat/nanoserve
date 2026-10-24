from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from nanoserve.block_manager import BlockManager
from nanoserve.config import EngineConfig
from nanoserve.sequence import FinishReason, Sequence, SequenceStatus


@dataclass
class ScheduledSeq:
    seq: Sequence
    num_tokens: int  # new tokens this step: 1 for a decode, up to the budget for a prefill chunk


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledSeq] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)

    @property
    def num_batched_tokens(self) -> int:
        return sum(s.num_tokens for s in self.scheduled)

    @property
    def is_empty(self) -> bool:
        return not self.scheduled


class Scheduler:
    """Continuous batching with a per-step token budget.

    Each step:
      1. Running sequences go first, in admission order. A decode costs one token; a
         sequence still in prefill gets the next chunk that fits the remaining budget.
      2. If a running sequence needs a KV block and none is free, the most recently admitted
         running sequence is preempted: its blocks are freed and it goes back to the front of
         the queue to be recomputed later (its prefix may still be in the cache).
      3. Leftover budget admits waiting sequences, FCFS, as long as their next chunk fits in
         free blocks. Nothing new is admitted in a step that had to preempt.
    """

    def __init__(self, config: EngineConfig, block_manager: BlockManager):
        self.config = config
        self.block_manager = block_manager
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0

    def add(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.config.max_num_batched_tokens
        bm = self.block_manager

        i = 0
        while i < len(self.running) and budget > 0:
            seq = self.running[i]
            num_tokens = min(seq.num_uncomputed_tokens, budget)
            while not bm.can_append(seq, num_tokens):
                victim = self.running.pop()
                self._preempt(victim)
                out.preempted.append(victim)
                if victim is seq:
                    break
            else:
                bm.append_slots(seq, num_tokens)
                out.scheduled.append(ScheduledSeq(seq, num_tokens))
                budget -= num_tokens
                i += 1

        if out.preempted:
            return out

        while self.waiting and budget > 0 and len(self.running) < self.config.max_num_seqs:
            seq = self.waiting[0]
            if not seq.block_table:
                bm.allocate_prefix(seq)
            num_tokens = min(seq.num_uncomputed_tokens, budget)
            if not bm.can_append(seq, num_tokens):
                bm.free(seq)
                break
            bm.append_slots(seq, num_tokens)
            self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            out.scheduled.append(ScheduledSeq(seq, num_tokens))
            budget -= num_tokens
        return out

    def _preempt(self, seq: Sequence) -> None:
        self.block_manager.free(seq)
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)
        self.num_preemptions += 1

    def finish(self, seq: Sequence, reason: FinishReason) -> None:
        if seq.status is SequenceStatus.RUNNING:
            self.running.remove(seq)
        elif seq.status is SequenceStatus.WAITING:
            self.waiting.remove(seq)
        self.block_manager.free(seq)
        seq.status = SequenceStatus.FINISHED
        seq.finish_reason = reason
        seq.finish_time = time.perf_counter()
