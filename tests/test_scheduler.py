from __future__ import annotations

from nanoserve.block_manager import BlockManager
from nanoserve.config import EngineConfig
from nanoserve.sampling import SamplingParams
from nanoserve.scheduler import Scheduler
from nanoserve.sequence import Sequence, SequenceStatus


def make(num_blocks=64, budget=16, max_seqs=4, block_size=4):
    cfg = EngineConfig(max_num_batched_tokens=budget, max_num_seqs=max_seqs, block_size=block_size)
    return Scheduler(cfg, BlockManager(num_blocks, block_size))


def advance(sched, out, sample=True):
    """Pretend the model ran: mark tokens computed and append a token where one is sampled."""
    for item in out.scheduled:
        item.seq.num_computed_tokens += item.num_tokens
        if sample and item.seq.num_computed_tokens == item.seq.num_tokens:
            item.seq.append_token(7)


def test_long_prompt_is_chunked_to_the_budget():
    sched = make(budget=16)
    s = Sequence("a", list(range(40)), SamplingParams())
    sched.add(s)
    sizes = []
    for _ in range(3):
        out = sched.schedule()
        sizes.append(out.num_batched_tokens)
        advance(sched, out)
    assert sizes == [16, 16, 8]
    assert s.num_output_tokens == 1


def test_decodes_go_before_new_prefills():
    sched = make(budget=8)
    a = Sequence("a", [1, 2, 3], SamplingParams())
    sched.add(a)
    advance(sched, sched.schedule())  # a finishes prefill
    b = Sequence("b", list(range(20)), SamplingParams())
    sched.add(b)
    out = sched.schedule()
    assert [(i.seq.request_id, i.num_tokens) for i in out.scheduled] == [("a", 1), ("b", 7)]


def test_respects_max_num_seqs():
    sched = make(budget=64, max_seqs=2)
    for i in range(3):
        sched.add(Sequence(str(i), [1, 2], SamplingParams()))
    out = sched.schedule()
    assert len(out.scheduled) == 2
    assert len(sched.waiting) == 1


def test_preempts_newest_when_out_of_blocks():
    sched = make(num_blocks=4, budget=64, block_size=4)
    a = Sequence("a", [1, 2, 3, 4], SamplingParams())
    b = Sequence("b", [5, 6, 7, 8], SamplingParams())
    sched.add(a)
    sched.add(b)
    advance(sched, sched.schedule())  # each holds one full block and sampled a 5th token
    sched.block_manager.append_slots(Sequence("x", [0] * 8, SamplingParams()), 8)  # occupy 2 more
    out = sched.schedule()
    assert out.preempted == [b]
    assert b.status is SequenceStatus.WAITING and sched.waiting[0] is b
    assert b.block_table == [] and b.num_computed_tokens == 0
    assert [i.seq for i in out.scheduled] == [a]
