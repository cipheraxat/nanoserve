"""Run the test suite and every benchmark on a Colab GPU.

    colab run --gpu L4 scripts/colab_gpu_bench.py

Clones the public repo, installs it on top of Colab's preinstalled CUDA torch and triton,
runs the tests (including the Triton kernel test) and the benchmarks, and prints the JSON
results so they can be copied into benchmarks/results/.
"""

import glob
import subprocess
import sys

REPO = "https://github.com/cipheraxat/nanoserve"
failures = []


def sh(cmd: str, required: bool = False) -> None:
    print(f"\n$ {cmd}", flush=True)
    code = subprocess.run(cmd, shell=True, check=False).returncode
    if code:
        failures.append(cmd)
        if required:
            sys.exit(code)


sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv", required=True)
sh(f"rm -rf /content/nanoserve && git clone --depth 1 {REPO} /content/nanoserve", required=True)
sh("cd /content/nanoserve && pip install -q -e . pytest", required=True)
sh("cd /content/nanoserve && python -m pytest -q")
bench = "cd /content/nanoserve/benchmarks && python"
sh(f"{bench} bench_attention.py")
sh(f"{bench} bench_offline.py --num-requests 128 --batch 64")
sh(f"{bench} bench_prefix.py")
sh(f"{bench} bench_chunked.py")
for path in sorted(glob.glob("/content/nanoserve/benchmarks/results/*-cuda.json")):
    print(f"\n===== {path}")
    with open(path) as f:
        print(f.read())
if failures:
    print("\nFAILED:", *failures, sep="\n  ")
    sys.exit(1)
