from __future__ import annotations

from collections import OrderedDict, deque

from nanoserve.sequence import Sequence


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


class BlockManager:
    """Allocates fixed-size KV cache blocks and shares full blocks across requests.

    A full block is identified by a chained hash: hash(parent block hash, its tokens), so two
    requests share a block only when their whole prefix up to and including that block is
    identical. When the last reference to a hashed block goes away the block is not wiped;
    it moves to an LRU pool and can still be hit until something needs the memory.
    Blocks that were never hashed are handed out first, so cached prefixes live as long as
    possible.
    """

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.ref_counts = [0] * num_blocks
        self.block_hash: list[int | None] = [None] * num_blocks
        self.block_tokens: list[tuple[int, ...] | None] = [None] * num_blocks
        self.free_uncached: deque[int] = deque(range(num_blocks))
        self.free_cached: OrderedDict[int, None] = OrderedDict()  # LRU order, oldest first
        self.hash_to_block: dict[int, int] = {}
        self.prefix_query_tokens = 0
        self.prefix_hit_tokens = 0

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_uncached) + len(self.free_cached)

    @property
    def usage(self) -> float:
        return 1.0 - self.num_free_blocks / self.num_blocks

    @property
    def prefix_hit_rate(self) -> float:
        if self.prefix_query_tokens == 0:
            return 0.0
        return self.prefix_hit_tokens / self.prefix_query_tokens

    @staticmethod
    def hash_block(parent: int | None, tokens: tuple[int, ...]) -> int:
        return hash((parent, tokens))

    def _take_free_block(self) -> int:
        if self.free_uncached:
            block_id = self.free_uncached.popleft()
        else:
            block_id, _ = self.free_cached.popitem(last=False)
            block_hash = self.block_hash[block_id]
            if self.hash_to_block.get(block_hash) == block_id:
                del self.hash_to_block[block_hash]
            self.block_hash[block_id] = None
            self.block_tokens[block_id] = None
        self.ref_counts[block_id] = 1
        return block_id

    def allocate_prefix(self, seq: Sequence) -> None:
        """Attach cached blocks for the longest matching prefix of a newly admitted sequence.

        At least one token is always left uncomputed, because the model has to run on the
        last prompt token to produce logits for the first output token.
        """
        assert not seq.block_table, "sequence already holds blocks"
        if not self.enable_prefix_caching:
            return
        bs = self.block_size
        hits: list[int] = []
        parent: int | None = None
        for i in range((seq.num_tokens - 1) // bs):
            tokens = tuple(seq.token_ids[i * bs : (i + 1) * bs])
            block_hash = self.hash_block(parent, tokens)
            block_id = self.hash_to_block.get(block_hash)
            if block_id is None or self.block_tokens[block_id] != tokens:
                break
            hits.append(block_id)
            parent = block_hash
        for block_id in hits:
            if self.ref_counts[block_id] == 0:
                del self.free_cached[block_id]
            self.ref_counts[block_id] += 1
        seq.block_table = hits
        seq.num_computed_tokens = len(hits) * bs
        seq.num_registered_blocks = len(hits)
        if seq.num_output_tokens == 0:
            seq.num_cached_tokens = seq.num_computed_tokens
            self.prefix_query_tokens += seq.num_prompt_tokens
            self.prefix_hit_tokens += seq.num_computed_tokens

    def num_blocks_needed(self, seq: Sequence, num_new_tokens: int) -> int:
        total = cdiv(seq.num_computed_tokens + num_new_tokens, self.block_size)
        return max(0, total - len(seq.block_table))

    def can_append(self, seq: Sequence, num_new_tokens: int) -> bool:
        return self.num_blocks_needed(seq, num_new_tokens) <= self.num_free_blocks

    def append_slots(self, seq: Sequence, num_new_tokens: int) -> None:
        for _ in range(self.num_blocks_needed(seq, num_new_tokens)):
            seq.block_table.append(self._take_free_block())

    def register_full_blocks(self, seq: Sequence) -> None:
        """Publish blocks that became full (and computed) so later requests can reuse them."""
        if not self.enable_prefix_caching:
            return
        bs = self.block_size
        num_full = seq.num_computed_tokens // bs
        for i in range(seq.num_registered_blocks, num_full):
            block_id = seq.block_table[i]
            parent = self.block_hash[seq.block_table[i - 1]] if i > 0 else None
            tokens = tuple(seq.token_ids[i * bs : (i + 1) * bs])
            block_hash = self.hash_block(parent, tokens)
            self.block_hash[block_id] = block_hash
            self.block_tokens[block_id] = tokens
            # Two requests can compute the same prefix at once; the first block wins the
            # lookup entry, but both keep the hash so their own chains stay intact.
            self.hash_to_block.setdefault(block_hash, block_id)
        seq.num_registered_blocks = max(seq.num_registered_blocks, num_full)

    def free(self, seq: Sequence) -> None:
        # Release tail blocks first so they are evicted before the shared prefix blocks.
        for block_id in reversed(seq.block_table):
            self.ref_counts[block_id] -= 1
            assert self.ref_counts[block_id] >= 0
            if self.ref_counts[block_id] == 0:
                if self.block_hash[block_id] is None:
                    self.free_uncached.append(block_id)
                else:
                    self.free_cached[block_id] = None
        seq.block_table = []
        seq.num_registered_blocks = 0
        seq.num_computed_tokens = 0
