# Active context

Last updated: 2026-10-25.

## Current focus

None. The last milestone is complete: the NVIDIA L4 results are in the README, CI is green on `main`, and the numbers on
the author's resume match the README.

## State of the world

- `main` at github.com/cipheraxat/nanoserve is the only branch. The local checkout is clean and in sync.
- Last GPU run: one Colab L4 job at commit `76685e6` (40 tests passed, all four benchmarks), results committed in
  `f548bb2`.
- The resume quotes these numbers, so changing any of them means updating the resume too: 3.9x vs Hugging Face (915 vs
  233 tok/s), Triton kernel 14-19x faster than gather + SDPA, throughput 168 to 915 tok/s after the split, prefix
  caching 94% hits and 11x less prefill, chunked prefill p99 ITL 473 to 65 ms (7x).

## Immediate next steps

Pick from `progress.md` "To do". The cheapest useful one is item 2 (an L4 `bench_serving.py` run for online latency
percentiles).

## Blockers

None. GPU runs go through the mcp-colab-gpu runner described in `runbook.md`; its login is cached on the author's Mac.
