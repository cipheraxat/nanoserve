# Decisions

Architectural decision log. Add an entry for every major choice; don't reverse one without a new entry that says why.

## ADR-001: Write the model instead of wrapping Hugging Face modules

- **Why:** Paged attention needs control of the attention call and a flat `[T]` token batch. HF modules expect padded
  `[B, L]` inputs and their own cache classes.
- **Trade-offs:** One model file per architecture (only Qwen3 today) and the risk of numerical drift from HF.
- **Outcome:** `models/qwen3.py`, with parameter names that mirror the HF checkpoint so weights load directly. Drift is
  caught by tests that require token-exact greedy output vs HF; on real Qwen3-0.6B in fp32 the outputs match exactly.

## ADR-002: Packed `[T]` batches; attention is the only per-sequence code

- **Why:** No padding waste in the projections and MLP, and prefill chunks and decodes mix freely in one step.
- **Trade-offs:** Attention needs index metadata (`token_index`, `output_index`, `key_slots`, masks).
- **Outcome:** `ModelRunner.prepare` builds the metadata once per step; all layers share it.

## ADR-003: Paged KV cache with 16-token blocks

- **Why:** Allocate memory per block as sequences grow, instead of reserving `max_model_len` per request.
- **Trade-offs:** Up to 15 tokens of waste per sequence, and gathers or a custom kernel to read scattered blocks.
- **Outcome:** Default `block_size=16` (power of two, enforced in `EngineConfig`). Tests also run with block sizes 1, 4
  and 16.

## ADR-004: Prefix caching with chained block hashes and an LRU pool

- **Why:** Share prompt prefixes (system prompts, few-shot examples, chat history) without a radix tree.
- **Trade-offs:** Only full blocks are shared, so up to one block of a matching prefix is recomputed. Python `hash` on
  token tuples can collide, so the block's tokens are stored and compared on every hit.
- **Outcome:** `hash((parent_hash, tokens))` per full block. Freed hashed blocks stay cached in LRU order; blocks that
  were never hashed are reused first. A match always leaves one prompt token to compute. 94% hit rate on the shared
  system prompt benchmark.

## ADR-005: Preemption by recompute, not swap

- **Why:** Simpler than swapping KV to CPU memory (no host buffers, no copy streams), and the prefix cache often still
  holds the preempted sequence's blocks when it returns.
- **Trade-offs:** Recompute costs prefill compute under memory pressure.
- **Outcome:** The newest running sequence is preempted first and goes back to the front of the queue. Nothing new is
  admitted in a step that preempted. Covered by a test that forces preemption and still matches HF.

## ADR-006: Scheduler with a per-step token budget (chunked prefill)

- **Why:** Bound step time so long prompts don't stall running decodes.
- **Trade-offs:** Prompts take several steps, so each prompt's TTFT rises slightly.
- **Outcome:** `max_num_batched_tokens` (default 512). Running sequences go first, then waiting ones FCFS. On an L4 a
  256-token budget cut p99 inter-token latency from 473 ms to 65 ms.

## ADR-007: Two attention paths, split per step

- **Why:** The torch gather + SDPA path works everywhere and handles any query length; a Triton kernel that reads K/V
  through the block table is much faster for decodes on CUDA.
- **Trade-offs:** Two implementations to keep equivalent, plus the scatter-back logic.
- **Outcome:** The first version used Triton only when a step was decode-only. Under continuous batching most steps are
  mixed, so throughput on an L4 was 168 tok/s, below HF's 233. Splitting every step into a decode group (Triton) and a
  gather group took it to 915 tok/s. A CPU test swaps the kernel for a torch reference to cover the split without a
  GPU. See `lessons.md`.

## ADR-008: Fuse QKV and gate/up projections at load time

- **Why:** Halves matmul launches per layer, which matters when steps are launch-bound.
- **Trade-offs:** Parameter names differ from HF for those layers; `load_hf_state_dict` concatenates them.
- **Outcome:** Implemented. The gain on MPS was small; it is the standard approach in vLLM and SGLang.

## ADR-009: Exponential-race sampling with per-request CPU generators

- **Why:** One batched argmax for all rows, and seeded requests that reproduce regardless of batch composition.
- **Trade-offs:** Seeded rows draw noise on the CPU and copy it over.
- **Outcome:** `sampler.py`. Top-k and top-p share one sort. Tested against a brute-force reference.

## ADR-010: The engine runs on one background thread

- **Why:** The engine isn't thread-safe and its step loop is synchronous; FastAPI handlers are async.
- **Trade-offs:** Requests pass through a queue (small latency); the GIL is shared with the event loop.
- **Outcome:** `AsyncEngine` uses a thread-safe inbox and per-request asyncio queues fed with `call_soon_threadsafe`.
  Request validation happens in the handler so bad requests return 400 before streaming starts. A client disconnect
  aborts its request.

## ADR-011: Test against Hugging Face on a tiny random model

- **Why:** Token-exact comparison catches any bug in caching, chunking or indexing, needs no download, and runs on CPU
  in CI.
- **Trade-offs:** A random model is not a real one, so the real Qwen3-0.6B was also checked by hand (exact match in
  fp32 on CPU).
- **Outcome:** `tests/conftest.py` builds a 2-layer Qwen3 with untied embeddings and `initializer_range=0.1`, which
  gives varied tokens.

## ADR-012: Loose dependency lower bounds

- **Why:** On Colab, `torch>=2.14` (what `uv add` wrote) would reinstall torch over the preinstalled CUDA build.
- **Trade-offs:** Less guarantee about old versions.
- **Outcome:** `torch>=2.5`, `transformers>=4.51`, and so on. GPU jobs upgrade `transformers` to 4.56+ for the
  benchmarks' `dtype=` argument.

## ADR-013: GPU runs through the mcp-colab-gpu runner

- **Why:** The author's Mac has no CUDA GPU. The plugin's login is cached; the `colab` CLI isn't signed in on that Mac.
- **Trade-offs:** Every run gets a fresh VM, so each job clones, installs and runs in one go.
- **Outcome:** See `runbook.md`. Default GPU is an L4.
