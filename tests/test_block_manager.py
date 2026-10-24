from __future__ import annotations

from nanoserve.block_manager import BlockManager
from nanoserve.sampling import SamplingParams
from nanoserve.sequence import Sequence


def seq(tokens: list[int], rid: str = "r") -> Sequence:
    return Sequence(rid, tokens, SamplingParams())


def compute_all(bm: BlockManager, s: Sequence) -> None:
    bm.append_slots(s, s.num_uncomputed_tokens)
    s.num_computed_tokens = s.num_tokens
    bm.register_full_blocks(s)


def test_allocates_lazily_per_block():
    bm = BlockManager(num_blocks=8, block_size=4)
    s = seq(list(range(10)))
    assert bm.num_blocks_needed(s, 10) == 3
    bm.append_slots(s, 5)
    assert len(s.block_table) == 2
    s.num_computed_tokens = 5
    bm.append_slots(s, 3)
    assert len(s.block_table) == 2  # positions 5..7 still fit in block 2
    bm.append_slots(s, 4)
    assert len(s.block_table) == 3


def test_prefix_hit_shares_blocks_and_counts_refs():
    bm = BlockManager(num_blocks=8, block_size=4)
    a = seq(list(range(12)), "a")
    compute_all(bm, a)
    b = seq(list(range(8)) + [99, 98, 97], "b")
    bm.allocate_prefix(b)
    assert b.block_table == a.block_table[:2]
    assert b.num_computed_tokens == 8
    assert [bm.ref_counts[i] for i in a.block_table] == [2, 2, 1]


def test_prefix_requires_whole_chain_to_match():
    bm = BlockManager(num_blocks=8, block_size=4)
    compute_all(bm, seq([1, 2, 3, 4, 5, 6, 7, 8, 9], "a"))
    b = seq([0, 2, 3, 4, 5, 6, 7, 8, 9], "b")  # same second block, different first block
    bm.allocate_prefix(b)
    assert b.block_table == []


def test_freed_blocks_stay_cached_until_evicted():
    bm = BlockManager(num_blocks=4, block_size=4)
    a = seq(list(range(9)), "a")
    compute_all(bm, a)
    cached = a.block_table[:2]
    bm.free(a)
    assert bm.num_free_blocks == 4
    b = seq(list(range(9)), "b")
    bm.allocate_prefix(b)
    assert b.block_table == cached


def test_uncached_blocks_are_reused_before_cached_ones():
    bm = BlockManager(num_blocks=4, block_size=4)
    a = seq(list(range(8)), "a")
    compute_all(bm, a)
    bm.free(a)
    c = seq([50, 51], "c")
    bm.append_slots(c, 2)
    assert c.block_table[0] not in a.block_table


def test_lru_evicts_tail_blocks_first():
    bm = BlockManager(num_blocks=3, block_size=2)
    a = seq([1, 2, 3, 4, 5, 6], "a")
    compute_all(bm, a)
    head, _, tail = a.block_table
    bm.free(a)
    c = seq([9], "c")
    bm.append_slots(c, 1)
    assert c.block_table == [tail]
    b = seq([1, 2, 3, 4, 7], "b")
    bm.allocate_prefix(b)
    assert b.block_table[0] == head


def test_hit_rate_counts_prompt_tokens():
    bm = BlockManager(num_blocks=8, block_size=4)
    compute_all(bm, seq(list(range(8)), "a"))
    b = seq(list(range(8)) + [1], "b")
    bm.allocate_prefix(b)
    assert bm.prefix_hit_tokens == 8
    assert bm.prefix_query_tokens == 9
