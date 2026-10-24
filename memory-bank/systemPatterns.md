# System patterns

## Architecture

```
HTTP client -> server.py (FastAPI, SSE)
            -> async_engine.py (one background engine thread, per-request asyncio queues)
            -> engine.py LLMEngine.step()
                 scheduler.py      which sequences run this step, and how many tokens each
                 block_manager.py  KV blocks, ref counts, prefix-cache hashes, LRU eviction
                 model_runner.py   pack the step into flat tensors, build attention metadata, run the model
                 models/qwen3.py   the decoder, written for packed and paged inputs
                 attention.py      write K/V into the cache; gather path (torch) and decode path (Triton)
                 triton_attention.py  paged decode kernel
                 sampler.py        temperature, top-k, top-p, seeds
```

One engine step: schedule, prepare, forward, sample, update (append tokens, publish full blocks to the prefix cache,
finish requests and free their blocks).

## Component relationships and ownership

- `LLMEngine` owns the `Scheduler`, `BlockManager`, `ModelRunner` and `Sampler`. Nothing else mutates them.
- In the server, only the engine thread touches the engine. Handlers talk to it through `AsyncEngine`'s thread-safe
  inbox. The one exception is `LLMEngine.validate_request`, which reads config only and can be called from any thread.
- `Scheduler` decides what runs; `BlockManager` decides where K/V lives. The scheduler asks the block manager
  (`can_append`, `append_slots`, `allocate_prefix`, `free`) and never edits block tables itself.
- `ModelRunner` is stateless across steps apart from the KV cache tensor and the weights.

## Invariants (break one and outputs silently go wrong)

- `Sequence.token_ids` is prompt + output. `num_computed_tokens` counts tokens whose K/V is already in the cache.
  Everything after that index still has to go through the model.
- A sequence samples a token only in the step where `num_computed_tokens` reaches `num_tokens`. Mid-prefill chunks
  produce no logits.
- A prefix-cache match always leaves at least one prompt token uncomputed, because the model must run on the last
  prompt token to produce the first output token.
- Shared (prefix-cached) blocks are always full and never written again. A sequence only writes positions at or after
  its `num_computed_tokens`.
- Packed order is sequence by sequence, tokens in order. Every index tensor (`token_index`, `output_index`,
  `logits_index`) assumes this.
- Preemption frees all of a sequence's blocks and resets `num_computed_tokens` to 0. The sequence keeps its generated
  tokens and is recomputed later.

## Design patterns

- **Packed batches.** All new tokens of a step form one `[T]` dimension. Projections, MLP and norms run on `[T, ...]`
  with no padding; only attention needs per-sequence structure, via metadata built once per step and shared by all
  layers.
- **Two attention groups per step.** Sequences with one new token (decodes) go to the Triton kernel on CUDA; prefill
  chunks go to the torch gather path. Each group's output is scattered back with `index_copy_`. Without Triton,
  everything goes to the gather path.
- **Chained block hashing.** A full block's identity is `hash((parent_hash, tokens))`, so a block matches only when its
  whole prefix matches. Freed hashed blocks stay in an LRU pool; never-hashed free blocks are handed out first.
- **HF-compatible weights.** Parameter names mirror the Hugging Face checkpoint, except that q/k/v and gate/up are fused
  at load time (`load_hf_state_dict`). The model is built on the `meta` device and filled from safetensors.
- **Deterministic sampling.** Sampling uses the exponential race (argmax of p / Exp(1)). Seeded requests draw noise from
  their own CPU generator, so they are reproducible whatever else is in the batch.

## Technology choices

- Python 3.10+, PyTorch 2.5+ (uses SDPA `enable_gqa`), Triton (CUDA only, optional), Hugging Face `transformers` 4.51+
  for config and tokenizer only, `safetensors`, FastAPI + uvicorn, httpx.
- `uv` for environments and the lockfile. Dependencies carry low lower bounds on purpose so the package installs on
  top of Colab's preinstalled CUDA torch without reinstalling it.
- pytest for tests, ruff for lint and format, GitHub Actions for CI (CPU torch).

## Coding standards

- `from __future__ import annotations` in every module; type hints on public functions; dataclasses for plain state.
- ruff with `E, F, I, B, UP, SIM`, line length 120, `ruff format`. Use `zip(..., strict=True)` when lengths must match.
- Comments explain why (a constraint, a measured number, a non-obvious invariant), not what the next line does.
  Module docstrings describe the component's job and design.
- Prefer `index_select` / `index_copy_` over advanced indexing on hot paths (much faster on MPS).
- Every behavior change gets a test. Engine-level changes are tested by comparing greedy output with Hugging Face on a
  tiny random Qwen3 (see `tests/conftest.py`). Unit tests cover the block manager, scheduler and sampler directly.
- Performance claims need a benchmark run and its JSON in `benchmarks/results/`, labelled with the hardware.
