# Progress

Last updated: 2026-10-25 (after the NVIDIA L4 results, commit `f548bb2`).

## Done

- **Engine core.** Qwen3 decoder for packed and paged inputs with fused QKV and gate/up; paged KV cache; block manager
  with ref counts, chained-hash prefix caching and LRU eviction; continuous-batching scheduler with a token budget,
  chunked prefill and preemption by recompute; batched sampler (temperature, top-k, top-p, seeds); offline
  `LLMEngine.generate`.
- **Attention.** Torch gather + SDPA path for any mix of prefill and decode; Triton paged decode kernel (online
  softmax, GQA); per-step split so decodes always use the kernel on CUDA.
- **Server.** OpenAI-compatible `/v1/completions` and `/v1/chat/completions` with SSE streaming, `cached_tokens` in
  usage, client-disconnect aborts, `/v1/models`, `/health`, Prometheus `/metrics`, `nanoserve` CLI.
- **Tests.** 35 on CPU plus 5 CUDA-only (40 pass on an L4): token-exact greedy output vs Hugging Face under batching,
  chunking, prefix caching, preemption and the decode/prefill split; block manager, scheduler, sampler and server unit
  tests; Triton kernel vs torch reference.
- **Benchmarks.** `bench_offline`, `bench_prefix`, `bench_chunked`, `bench_attention`, `bench_serving`. Results for
  Apple M4 (MPS) and NVIDIA L4 are in `benchmarks/results/` and the README.
- **Repo.** Public at github.com/cipheraxat/nanoserve, MIT, CI on every push (ruff + pytest on CPU).

## In progress

Nothing.

## To do (ordered by value)

1. Re-run the Apple M4 benchmarks. They predate the decode/prefill split (which only changes CUDA behavior, so the M4
   numbers are still valid, but the README notes the gap).
2. Record a `bench_serving.py` run (online Poisson load: TTFT, ITL, end-to-end percentiles) on the L4 and add it to the
   README. The script exists but has no published results.
3. Faster decode kernel: process several KV blocks per loop iteration and share K/V loads across the query heads of a
   GQA group. Measure with `bench_attention.py`.
4. Faster prefill: the gather path passes an explicit boolean mask to SDPA, which rules out FlashAttention. A varlen
   prefill path (or a Triton prefill kernel) would remove the gather for long prompts.
5. CUDA graphs for decode-only steps (needs padded batch-size buckets and fixed-shape metadata).
6. Another model family (for example Llama 3.2 1B) to prove the model interface.
7. Out of scope unless the brief changes: tensor parallelism, quantization, speculative decoding.
