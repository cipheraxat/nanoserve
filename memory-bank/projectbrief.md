# Project brief

## What it is

nanoserve is a small LLM inference engine in the style of vLLM, written from scratch in PyTorch. It serves Qwen3
models through an OpenAI-compatible HTTP API and implements the core techniques of a production serving engine: paged
KV cache, continuous batching, chunked prefill, prefix caching, preemption, and a Triton paged-attention decode kernel.

## Who it is for

- **The author.** A portfolio project for AI inference and AI platform engineering roles. It sits alongside the
  author's upstream work in vLLM, SGLang and NVIDIA Dynamo, and every part of it should hold up in an interview.
- **Readers learning how serving engines work.** The code is small enough to read end to end (about 1,600 lines in
  `src/`), and each technique lives in one file.

## Core requirements

1. **Correct.** Greedy decoding must produce exactly the tokens Hugging Face `transformers` produces for each request
   on its own, under any batching, chunking, prefix-cache hit or preemption. Tests enforce this.
2. **Real techniques, not mocks.** Paged KV cache with block tables, a scheduler with a per-step token budget,
   content-addressed prefix caching, preemption by recompute, a custom GPU kernel.
3. **Measured.** Every performance claim comes from a benchmark script in `benchmarks/` with raw JSON in
   `benchmarks/results/`, run on named hardware (Apple M4 and NVIDIA L4 so far).
4. **Usable.** `uv sync && uv run nanoserve` starts an OpenAI-compatible server with streaming and Prometheus metrics.
5. **Runs anywhere.** CPU, Apple Silicon (MPS) and CUDA. The test suite runs on CPU without downloads; CUDA-only tests
   skip cleanly.

## Out of scope

Tensor or pipeline parallelism, multi-node serving, quantization, LoRA, speculative decoding, and model families other
than Qwen3 dense. These are fine as future work but must not complicate the core.

## Definition of success

- Every test passes on CPU in CI and on an NVIDIA GPU (40 on an L4).
- On an NVIDIA L4, nanoserve clearly beats Hugging Face `generate` with static batching on a mixed workload (currently
  3.9x: 915 vs 233 output tok/s), the Triton decode kernel beats the gather path, and prefix caching and chunked
  prefill show their expected effects.
- The README explains the design, the results and how to reproduce them, and the numbers on the author's resume match
  the README exactly.
