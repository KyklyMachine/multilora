"""
LoRA adapter benchmark for vLLM.

Sends one request per (sample, lora_adapter) pair concurrently — vLLM batches
them internally. Measures RPS and latency across different adapter counts.

Requirements:
    vLLM must be started with:
        --enable-lora
        --max-loras <N>
        --lora-modules name1=/path/to/lora1 name2=/path/to/lora2 ...

Usage:
    from openai import AsyncOpenAI
    from lora_benchmark import benchmark_lora_sweep

    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="token")
    inputs = list(zip(texts, image_urls))   # image_urls can be None

    summary_df, per_lora_df = benchmark_lora_sweep(
        client=client,
        all_lora_names=["lora_a", "lora_b", "lora_c", "lora_d"],
        system_prompt="You are a content classifier.",
        user_prompt="Classify the following: ",
        inputs=inputs,
        lora_counts=[1, 2, 4],
        n_repeats=10,
        max_tokens=16,
    )
"""

import asyncio
import logging
import statistics
import time
from asyncio import Semaphore
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from utils.async_openai_utils import create_request

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    """Settings for a single benchmark run."""

    lora_names: List[str]       # adapter names as passed to --lora-modules
    system_prompt: str
    user_prompt: str
    max_tokens: int = 16
    temperature: float = 0.0    # deterministic → stable latency measurements
    concurrency: int = 512      # semaphore cap; keep >= n_loras * n_samples
    n_repeats: int = 1          # repeats per (sample, lora) pair


@dataclass
class _Result:
    lora_name: str
    latency: float
    success: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ---------------------------------------------------------------------------
# Single request
# ---------------------------------------------------------------------------


async def _send_one(
    client: AsyncOpenAI,
    sem: Semaphore,
    lora_name: str,
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
) -> _Result:
    """Send one request to vLLM using the given LoRA adapter as the model name."""
    async with sem:
        t0 = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=lora_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            latency = time.perf_counter() - t0
            usage = resp.usage
            return _Result(
                lora_name=lora_name,
                latency=latency,
                success=True,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
            )
        except Exception as e:
            logger.warning(f"[{lora_name}] request failed: {e}")
            return _Result(
                lora_name=lora_name,
                latency=time.perf_counter() - t0,
                success=False,
            )


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _pct(sorted_values: List[float], p: float) -> float:
    idx = min(int(p * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[idx]


def _stats_row(results: List[_Result], elapsed: float) -> Dict:
    successful = [r for r in results if r.success]
    latencies = sorted(r.latency for r in successful)
    n_total, n_ok = len(results), len(successful)

    row: Dict = {
        "n_requests": n_total,
        "n_success": n_ok,
        "error_rate": round((n_total - n_ok) / n_total, 4) if n_total else 0,
        "rps": round(n_ok / elapsed, 2) if elapsed > 0 else 0,
    }

    if latencies:
        row.update(
            {
                "mean_lat_s": round(statistics.mean(latencies), 3),
                "p50_lat_s": round(_pct(latencies, 0.50), 3),
                "p95_lat_s": round(_pct(latencies, 0.95), 3),
                "p99_lat_s": round(_pct(latencies, 0.99), 3),
                "mean_prompt_tok": round(
                    statistics.mean(r.prompt_tokens for r in successful), 1
                ),
                "mean_compl_tok": round(
                    statistics.mean(r.completion_tokens for r in successful), 1
                ),
            }
        )
    else:
        row.update(
            {k: None for k in
             ["mean_lat_s", "p50_lat_s", "p95_lat_s", "p99_lat_s",
              "mean_prompt_tok", "mean_compl_tok"]}
        )

    return row


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------


async def run_benchmark(
    client: AsyncOpenAI,
    config: BenchmarkConfig,
    inputs: List[Tuple[str, Optional[str]]],
) -> Tuple[Dict, pd.DataFrame]:
    """
    Fire all (sample × lora × repeat) requests concurrently.

    vLLM sees the full queue at once and schedules its own batches internally.

    Args:
        client:  AsyncOpenAI pointing at vLLM (base_url="http://localhost:8000/v1").
        config:  Benchmark settings.
        inputs:  List of (text, img_url) pairs. img_url can be None for text-only.
                 For video inputs pass the video URL — vLLM routes it accordingly.

    Returns:
        global_stats: Dict with aggregate metrics for this run.
        per_lora_df:  DataFrame with one row per LoRA adapter.
    """
    sem = Semaphore(config.concurrency)

    tasks = [
        _send_one(
            client=client,
            sem=sem,
            lora_name=lora_name,
            messages=create_request(
                system_prompt=config.system_prompt,
                user_prompt=config.user_prompt,
                text=text,
                img_url=img_url,
            ),
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        for text, img_url in inputs
        for lora_name in config.lora_names
        for _ in range(config.n_repeats)
    ]

    n_loras = len(config.lora_names)
    n_samples = len(inputs)
    logger.info(
        f"Firing {len(tasks)} requests "
        f"({n_samples} samples × {n_loras} LoRAs × {config.n_repeats} repeats)"
    )

    t0 = time.perf_counter()
    results: List[_Result] = await tqdm_asyncio.gather(
        *tasks, desc=f"n_loras={n_loras}"
    )
    elapsed = time.perf_counter() - t0

    # Aggregate stats
    global_stats = _stats_row(results, elapsed)
    global_stats["n_loras"] = n_loras
    global_stats["elapsed_s"] = round(elapsed, 2)
    global_stats["lora_names"] = ", ".join(config.lora_names)

    # Per-adapter stats (each adapter's share of the same elapsed window)
    per_lora_rows = []
    for name in config.lora_names:
        row = _stats_row([r for r in results if r.lora_name == name], elapsed)
        row["lora_name"] = name
        per_lora_rows.append(row)

    return global_stats, pd.DataFrame(per_lora_rows)


# ---------------------------------------------------------------------------
# Sweep: run benchmark for different LoRA counts and collect results
# ---------------------------------------------------------------------------


def benchmark_lora_sweep(
    client: AsyncOpenAI,
    all_lora_names: List[str],
    system_prompt: str,
    user_prompt: str,
    inputs: List[Tuple[str, Optional[str]]],
    lora_counts: Optional[List[int]] = None,
    n_repeats: int = 1,
    max_tokens: int = 16,
    temperature: float = 0.0,
    concurrency: int = 512,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run benchmark for each count in `lora_counts`, using the first N adapters
    from `all_lora_names` each time.

    Args:
        client:          AsyncOpenAI client.
        all_lora_names:  All available LoRA names (registered via --lora-modules).
        system_prompt:   Fixed system prompt for all requests.
        user_prompt:     Fixed user prompt prefix (text from inputs is appended).
        inputs:          List of (text, img_url) pairs — the benchmark dataset.
        lora_counts:     List of adapter counts to test, e.g. [1, 2, 4, 8].
                         Defaults to [1, 2, ..., len(all_lora_names)].
        n_repeats:       Repeats per (sample, lora) pair. Increase for stable RPS.
        max_tokens:      Max tokens per response.
        temperature:     Sampling temperature (0.0 = deterministic, recommended).
        concurrency:     Max simultaneous in-flight requests.

    Returns:
        summary_df:   One row per benchmark run; columns include n_loras, rps,
                      mean/p50/p95/p99 latency, elapsed_s, error_rate.
        per_lora_df:  Per-adapter breakdown for every run; includes n_loras_in_run.
    """
    if lora_counts is None:
        lora_counts = list(range(1, len(all_lora_names) + 1))

    summary_rows: List[Dict] = []
    per_lora_frames: List[pd.DataFrame] = []

    for n in lora_counts:
        if n > len(all_lora_names):
            logger.warning(
                f"Skipping n_loras={n}: only {len(all_lora_names)} adapters available."
            )
            continue

        loras = all_lora_names[:n]
        config = BenchmarkConfig(
            lora_names=loras,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            concurrency=concurrency,
            n_repeats=n_repeats,
        )

        print(f"\n{'─' * 64}")
        print(f"  n_loras={n}   adapters: {loras}")
        print(f"{'─' * 64}")

        global_stats, per_lora_df = asyncio.run(
            run_benchmark(client=client, config=config, inputs=inputs)
        )
        summary_rows.append(global_stats)

        per_lora_df.insert(0, "n_loras_in_run", n)
        per_lora_frames.append(per_lora_df)

    summary_df = pd.DataFrame(summary_rows)
    per_lora_df = pd.concat(per_lora_frames, ignore_index=True)

    _print_results(summary_df, per_lora_df)
    return summary_df, per_lora_df


def _print_results(summary_df: pd.DataFrame, per_lora_df: pd.DataFrame) -> None:
    _SUMMARY_COLS = [
        "n_loras", "n_requests", "n_success", "error_rate",
        "rps", "mean_lat_s", "p50_lat_s", "p95_lat_s", "p99_lat_s", "elapsed_s",
    ]
    _PER_LORA_COLS = [
        "n_loras_in_run", "lora_name", "n_requests", "n_success",
        "error_rate", "rps", "mean_lat_s", "p50_lat_s", "p95_lat_s",
    ]

    def _avail(df, cols):
        return [c for c in cols if c in df.columns]

    print(f"\n{'═' * 64}")
    print("  BENCHMARK SUMMARY")
    print(f"{'═' * 64}")
    print(summary_df[_avail(summary_df, _SUMMARY_COLS)].to_string(index=False))

    print(f"\n{'═' * 64}")
    print("  PER-LORA BREAKDOWN")
    print(f"{'═' * 64}")
    print(per_lora_df[_avail(per_lora_df, _PER_LORA_COLS)].to_string(index=False))
