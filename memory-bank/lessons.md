# Lessons

Problems that took more than one attempt, and how they were solved. Read this before touching the area involved.

## A fast kernel that never ran: Triton only on decode-only steps

- **Symptom:** On an L4, the Triton kernel was 14-19x faster than the gather path in isolation, yet end-to-end
  throughput was 168 tok/s, below Hugging Face static batching (233).
- **Cause:** The kernel was used only when every sequence in a step had one new token. Under continuous batching a
  new request's prefill chunk joins most steps, so those steps sent all 64 decodes down the gather path, which gathers
  each decode's whole context on every layer.
- **Fix:** Split each step into a decode group and a gather group (`AttentionMetadata.decode` / `.gather`) and scatter
  outputs back: 915 tok/s. A step profiler showed 397 decode-only steps (median 37 ms) and 77 mixed steps (median
  59 ms).
- **Rule:** Benchmark a kernel end to end, not just in isolation, and count which path each step actually takes.

## MPS slow paths

- Advanced indexing (`cache[slots]`) for the K/V gather was about 6x slower than `index_select` on MPS. Use
  `index_select` / `index_copy_` on hot paths.
- A `[>8, 1024] x [1024, 151936]` bf16 matmul (the LM head) falls off a fast path on MPS: about 26 ms at 32 rows vs
  3.5 ms at 8. Slicing the vocabulary into 16 pieces roughly halves it (`Qwen3ForCausalLM.compute_logits`).
- After those fixes, MPS is launch-bound (about half the step time is outside attention). Don't spend more time tuning
  for MPS; the CUDA path is what matters.

## Mac benchmarks under memory pressure

- The 16 GB M4 had 7 GB in swap during the first benchmark run, so absolute numbers were low and noisy. Prefill speed
  matched HF (about 3 s for 2,048 tokens in both), which showed the engine wasn't the cause. Report ratios from Mac
  runs and say what the machine was doing.

## transformers 5 chat templates

- `apply_chat_template(..., tokenize=True)` returns a `BatchEncoding`, which is not a `dict` subclass, so an
  `isinstance(x, dict)` check missed it and the server failed with "too many dimensions 'str'". Use `tokenize=False`,
  then `tokenizer.encode(text, add_special_tokens=False)`.
- transformers 5 moved `rope_theta` into `config.rope_parameters`. `rope_theta_of()` handles both layouts.

## Tiny random test models

- With tied embeddings and the default `initializer_range=0.02`, a random 2-layer Qwen3 repeats the same token, so
  "matches HF" tests prove almost nothing. Use untied embeddings and `initializer_range=0.1`.

## Model weights

- The local Hugging Face cache had Qwen3-0.6B's tokenizer and config but no `.safetensors`. `snapshot_download` with
  `allow_patterns=["*.safetensors", "*.json"]` fetches them (about 1.2 GB).

## Colab GPU runs

- The `colab` CLI on the author's Mac has no credentials and `gcloud` isn't installed, so it keeps asking for browser
  auth codes. Use the mcp-colab-gpu runner from `runbook.md`, whose token is cached.
- Colab's preinstalled `transformers` can be too old for `from_pretrained(..., dtype=...)`; the job upgrades it.
- In a `cmd | tail` pipeline the exit code is `tail`'s. Read the output, not the exit code.

## Tooling

- zsh errors on a glob with no matches (`no matches found`), which killed a monitoring loop. Use `ls dir` or
  `setopt nullglob` instead of a bare `dir/*.json` glob.
- ruff's `B905` wants `zip(..., strict=...)`; use `strict=True` where lengths must match and `strict=False` for
  pairwise `zip(xs, xs[1:])`.
