"""Throughput / latency benchmark of the engine across batch sizes.

Reports tokens per second and p50/p95 batch latency so the batching trade-off is visible
in numbers, not just argued.
"""

from __future__ import annotations

import platform
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from llmserve.engine import GenerationEngine, GenerationRequest
from llmserve.sampling import SamplingParams


@dataclass(frozen=True)
class LatencyStats:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def summarize_latencies(latencies_ms: Sequence[float]) -> LatencyStats:
    if not latencies_ms:
        raise ValueError("no latencies")
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return LatencyStats(
        mean_ms=float(arr.mean()),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
    )


@dataclass(frozen=True)
class BenchmarkRow:
    batch_size: int
    prompt_tokens: int
    new_tokens_per_request: int
    repeats: int
    latency: LatencyStats
    tokens_per_second: float
    """Completion tokens per second across the whole batch (throughput)."""
    ms_per_token_per_request: float
    """Decode latency as seen by one request (p50 batch latency / new tokens)."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_benchmark(
    engine: GenerationEngine,
    *,
    prompt: str,
    batch_sizes: Sequence[int],
    max_new_tokens: int,
    repeats: int = 3,
    warmup: int = 1,
) -> list[BenchmarkRow]:
    if repeats <= 0 or max_new_tokens <= 0:
        raise ValueError("repeats and max_new_tokens must be positive")
    rows: list[BenchmarkRow] = []
    greedy = SamplingParams(temperature=0.0)
    for bs in batch_sizes:
        if bs <= 0:
            raise ValueError("batch sizes must be positive")
        requests = [
            GenerationRequest(
                request_id=f"bench-{bs}-{i}",
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                sampling=greedy,
                stop_on_eos=False,  # fixed token count => comparable numbers
            )
            for i in range(bs)
        ]
        for _ in range(warmup):
            engine.generate_batch(requests)
        latencies: list[float] = []
        total_tokens = 0
        total_seconds = 0.0
        prompt_tokens = 0
        for _ in range(repeats):
            if engine.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            results = engine.generate_batch(requests)
            if engine.device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            latencies.append(dt * 1000.0)
            total_seconds += dt
            total_tokens += sum(r.completion_tokens for r in results)
            prompt_tokens = results[0].prompt_tokens
        stats = summarize_latencies(latencies)
        rows.append(
            BenchmarkRow(
                batch_size=bs,
                prompt_tokens=prompt_tokens,
                new_tokens_per_request=max_new_tokens,
                repeats=repeats,
                latency=stats,
                tokens_per_second=total_tokens / total_seconds,
                ms_per_token_per_request=stats.p50_ms / max_new_tokens,
            )
        )
    return rows


def environment_info(engine: GenerationEngine) -> dict[str, str]:
    info = {
        "model": engine.model_name,
        "device": str(engine.device),
        "dtype": str(engine.dtype).replace("torch.", ""),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if engine.device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(engine.device)
    return info


def render_markdown(rows: Sequence[BenchmarkRow], title: str, env: dict[str, str]) -> str:
    lines = [f"## {title}", ""]
    lines.append(", ".join(f"{k}: {v}" for k, v in env.items()))
    lines.append("")
    lines.append(
        "| Batch | Prompt tok | New tok/req | Tokens/s | Batch p50 (ms) | Batch p95 (ms) "
        "| ms/token/request | Speed-up vs batch 1 |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    base = rows[0].tokens_per_second if rows else 1.0
    for r in rows:
        lines.append(
            f"| {r.batch_size} | {r.prompt_tokens} | {r.new_tokens_per_request} | "
            f"{r.tokens_per_second:,.1f} | {r.latency.p50_ms:,.1f} | {r.latency.p95_ms:,.1f} | "
            f"{r.ms_per_token_per_request:.2f} | {r.tokens_per_second / base:.2f}x |"
        )
    return "\n".join(lines) + "\n"


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values))
