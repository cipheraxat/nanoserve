from __future__ import annotations

import json
import os
import platform
import random
import statistics

import torch


def sync(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def device_name(device: torch.device | str) -> str:
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.get_device_name()
    if device.type == "mps":
        return f"Apple {platform.processor() or 'Silicon'} (MPS)"
    return f"CPU ({platform.processor()})"


def random_prompts(n: int, lo: int, hi: int, seed: int = 0, vocab: tuple[int, int] = (1000, 30000)) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(*vocab) for _ in range(rng.randint(lo, hi))] for _ in range(n)]


def output_lengths(n: int, lo: int, hi: int, seed: int = 1) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(lo, hi) for _ in range(n)]


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def summary_ms(values: list[float]) -> dict:
    return {
        "mean_ms": round(statistics.fmean(values) * 1000, 1) if values else None,
        "p50_ms": round(pct(values, 50) * 1000, 1),
        "p99_ms": round(pct(values, 99) * 1000, 1),
        "max_ms": round(max(values) * 1000, 1) if values else None,
    }


def save(name: str, device: str, payload: dict) -> str:
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    tag = torch.device(device).type
    path = os.path.join(out_dir, f"{name}-{tag}.json")
    with open(path, "w") as f:
        json.dump({"device": device_name(device), "torch": torch.__version__, **payload}, f, indent=2)
    return path


def table(rows: list[dict], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(lines)
