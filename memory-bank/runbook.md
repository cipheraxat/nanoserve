# Runbook

Exact commands for this repo. Run them from the repo root unless a step says otherwise.

## Setup

```bash
uv sync
```

Needs Python 3.10+ and `uv`. On CUDA machines with a preinstalled torch (Colab), use `pip install -e . pytest`
instead so torch isn't reinstalled.

## Test, lint, format

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format .
```

- CPU: 35 pass, 1 skipped (the CUDA-only module). On a CUDA GPU with Triton: 40 pass.
- One file or test: `uv run pytest -q tests/test_scheduler.py -k preempts`.
- CI (`.github/workflows/ci.yml`) runs `ruff check` and `pytest` with CPU torch on every push to `main` and on pull
  requests. Check it with `gh run list -R cipheraxat/nanoserve --limit 1`.

## Run the server

```bash
uv run nanoserve --model Qwen/Qwen3-0.6B --port 8000
```

First start downloads about 1.2 GB of weights. Flags: `--max-num-seqs`, `--max-num-batched-tokens`,
`--kv-cache-memory-gb`, `--block-size`, `--no-prefix-caching`, `--attention-backend {auto,torch,triton}`,
`--device`, `--dtype`.

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":32,"temperature":0}'
curl -s localhost:8000/metrics
```

## Benchmarks (local)

Run from `benchmarks/`; each writes `results/<name>-<device type>.json`.

```bash
cd benchmarks
uv run python bench_offline.py
uv run python bench_prefix.py
uv run python bench_chunked.py
uv run python bench_serving.py --url http://127.0.0.1:8000 --rate 2
```

`bench_attention.py` needs CUDA. `bench_offline.py` loads a Hugging Face copy of the model too; on a 16 GB Mac close
other apps first.

## GPU tests and benchmarks (Colab L4)

The author's Mac has no CUDA GPU. Use the mcp-colab-gpu runner (pinned commit, login cached in
`~/.config/colab-exec/token.json`). Don't use the `colab` CLI; it isn't signed in there.

1. Push the commit to test to `main` (the job clones from GitHub).
2. Write a runner, `colab_run.py`, somewhere outside the repo:

   ```python
   import json, sys
   from mcp_colab_gpu.server import _run_on_colab
   code = open(sys.argv[2]).read()
   out, err, rc, t = _run_on_colab(code, sys.argv[1], False, int(sys.argv[3]) if len(sys.argv) > 3 else 3600)
   print(json.dumps({"exit_code": rc, "elapsed_s": t})); print("--- stdout ---\n" + out); print("--- stderr ---\n" + err[-5000:])
   ```

3. Write the job, `job.py`, next to it. It clones the repo, installs it over Colab's torch, runs the tests and the
   benchmarks, and prints the result JSON:

   ```python
   import glob, subprocess
   def sh(c, cwd="/content"):
       r = subprocess.run(c, shell=True, cwd=cwd, executable="/bin/bash", capture_output=True, text=True)
       print(f"$ {c}\n[exit {r.returncode}]\n{(r.stdout + r.stderr)[-8000:]}", flush=True)
   sh("nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv")
   sh("git clone -q --depth 1 https://github.com/cipheraxat/nanoserve /content/nanoserve")
   sh("pip -q install -e /content/nanoserve pytest && pip -q install -U 'transformers>=4.56'")
   sh("python -m pytest -q -rs 2>&1 | tail -25", cwd="/content/nanoserve")
   for b in ["bench_attention.py", "bench_offline.py --num-requests 128 --batch 64", "bench_prefix.py", "bench_chunked.py"]:
       sh(f"python {b}", cwd="/content/nanoserve/benchmarks")
   for p in sorted(glob.glob("/content/nanoserve/benchmarks/results/*-cuda.json")):
       print("===== " + p.rsplit("/", 1)[1]); print(open(p).read())
   ```

4. Run it (about 5 to 8 minutes on an L4; the VM is released afterwards):

   ```bash
   uvx --with "mcp[cli]<2" --from "git+https://github.com/mio-github/mcp-colab-gpu@f00acc2de858f60f5a4918edac15af568230c920" python -I colab_run.py L4 job.py 3600 > colab_out.txt 2>&1
   ```

5. Copy each `===== <name>.json` block from the output into `benchmarks/results/<name>.json`, update the README tables,
   and commit.

Rules: never mount Drive or call `google.colab.auth` in a job. T4 has no native bf16; use an L4 or better.
`scripts/colab_gpu_bench.py` is the equivalent job for anyone using the `colab` CLI instead
(`colab run --gpu L4 scripts/colab_gpu_bench.py`).

## Release (push to GitHub)

```bash
uv run ruff check . && uv run pytest -q
git add -A && git commit
git push origin main
gh run list -R cipheraxat/nanoserve --limit 1
```

There is no package release; `main` is the deliverable. If benchmark numbers change, update the README and the
author's resume together.
