"""
Multi-LoRA benchmark for a vLLM-served VLM.

Run once per server configuration (different number of adapters, rank, target
modules, etc.), stamp the config into `meta`, then concatenate the returned
DataFrames across runs into a single comparison table.

Example:
    inputs = list(zip(df["text_value"], df["local_image_path"]))

    df_small = run_benchmark(
        client=vllm_client,
        adapters=["a1", "a2"],
        inputs=inputs,
        meta={"n_loras": 2, "rank": 8, "target_modules": "qkv"},
        concurrency=64,
    )
    df_large = run_benchmark(
        client=vllm_client,
        adapters=["a1", "a2", "a3", "a4"],
        inputs=inputs,
        meta={"n_loras": 4, "rank": 16, "target_modules": "qkv"},
        concurrency=64,
    )

    summary = pd.concat([df_small, df_large], ignore_index=True)
"""

import asyncio
import logging
import time
from asyncio import Semaphore, gather
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


logger = logging.getLogger(__name__)


def default_build_prompt(
    text: str = "", img_url: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Minimal OpenAI chat message: text plus optional image as one user turn."""
    user_content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    if img_url:
        user_content.append({"type": "image_url", "image_url": {"url": img_url}})
    return [{"role": "user", "content": user_content}]


async def _call(
    client: AsyncOpenAI,
    adapter: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    semaphore: Semaphore,
) -> Dict[str, Any]:
    """Send one request to one adapter; record latency and success."""
    t0 = time.perf_counter()
    async with semaphore:
        try:
            await client.chat.completions.create(
                model=adapter,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            success, err = True, None
        except Exception as e:
            success, err = False, f"{type(e).__name__}: {e}"
            logger.warning(f"[{adapter}] {err}")
    return {
        "adapter": adapter,
        "success": success,
        "latency": time.perf_counter() - t0,
        "error": err,
    }


async def _process_input(
    client: AsyncOpenAI,
    adapters: List[str],
    messages: List[Dict[str, Any]],
    max_tokens: int,
    semaphore: Semaphore,
) -> Tuple[List[Dict[str, Any]], float]:
    """Fire one request per adapter concurrently for a single input."""
    t0 = time.perf_counter()
    per_adapter = await gather(
        *[_call(client, a, messages, max_tokens, semaphore) for a in adapters]
    )
    return per_adapter, time.perf_counter() - t0


def _stats(latencies: List[float]) -> Dict[str, float]:
    """n, mean, p50, p95, p99, max for a list of latencies (seconds)."""
    if not latencies:
        return {"n": 0, "mean_s": 0.0, "p50_s": 0.0, "p95_s": 0.0, "p99_s": 0.0, "max_s": 0.0}
    arr = np.asarray(latencies, dtype=float)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    return {
        "n": int(arr.size),
        "mean_s": float(arr.mean()),
        "p50_s": float(p50),
        "p95_s": float(p95),
        "p99_s": float(p99),
        "max_s": float(arr.max()),
    }


async def benchmark_loras(
    client: AsyncOpenAI,
    adapters: List[str],
    inputs: List[Tuple[str, Optional[str]]],
    build_prompt: Callable[..., List[Dict[str, Any]]] = default_build_prompt,
    concurrency: int = 64,
    max_tokens: int = 1,
    meta: Optional[Dict[str, Any]] = None,
    desc: str = "lora-bench",
) -> pd.DataFrame:
    """Benchmark a set of LoRA adapters served by vLLM.

    For every input, fires one request per adapter concurrently; vLLM batches
    them on top of shared base-model compute. Latency / RPS are measured at
    the OpenAI-API level.

    Args:
        client: AsyncOpenAI client targeting the vLLM server.
        adapters: LoRA names registered in vLLM (`--lora-modules name=path ...`).
        inputs: List of `(text, img_url)` pairs to use as load.
        build_prompt: Callable returning OpenAI chat messages from `text=` and
            `img_url=` kwargs. Defaults to a minimal user-turn builder.
        concurrency: Max in-flight requests across all adapters and inputs.
        max_tokens: Generation cap. Use 1 if only the first decoder hidden
            state / embedding is consumed downstream.
        meta: Server-config attributes (e.g.
            `{"n_loras": 4, "rank": 16, "target_modules": "qkv"}`). Every key
            becomes a column on every output row, so prefer flat scalar values.
        desc: tqdm description.

    Returns:
        Long-format DataFrame: one row per adapter (`scope="per_adapter"`) plus
        one aggregate row (`scope="end_to_end"`, `adapter="all"`). Columns:
        `<meta keys>, wall_time_s, concurrency, max_tokens, n_inputs,
        n_adapters, n_failed, scope, adapter, rps, n, mean_s, p50_s, p95_s,
        p99_s, max_s`.
    """
    semaphore = Semaphore(concurrency)
    prebuilt = [build_prompt(text=t, img_url=u) for t, u in inputs]
    progress = tqdm_asyncio(total=len(inputs), desc=desc)

    async def _wrapped(messages: List[Dict[str, Any]]):
        out = await _process_input(client, adapters, messages, max_tokens, semaphore)
        progress.update(1)
        return out

    wall_t0 = time.perf_counter()
    per_input = await gather(*[_wrapped(m) for m in prebuilt])
    wall_time = time.perf_counter() - wall_t0
    progress.close()

    records: List[Dict[str, Any]] = [r for per_adapter, _ in per_input for r in per_adapter]
    e2e: List[float] = [t for _, t in per_input]

    common: Dict[str, Any] = {
        **(meta or {}),
        "wall_time_s": wall_time,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "n_inputs": len(inputs),
        "n_adapters": len(adapters),
        "n_failed": sum(1 for r in records if not r["success"]),
    }

    rows: List[Dict[str, Any]] = []
    for adapter in adapters:
        lats = [r["latency"] for r in records if r["adapter"] == adapter and r["success"]]
        s = _stats(lats)
        rows.append(
            {
                **common,
                "scope": "per_adapter",
                "adapter": adapter,
                "rps": s["n"] / wall_time if wall_time > 0 else 0.0,
                **s,
            }
        )
    s = _stats(e2e)
    rows.append(
        {
            **common,
            "scope": "end_to_end",
            "adapter": "all",
            "rps": len(inputs) / wall_time if wall_time > 0 else 0.0,
            **s,
        }
    )
    return pd.DataFrame(rows)


def run_benchmark(
    client: AsyncOpenAI,
    adapters: List[str],
    inputs: List[Tuple[str, Optional[str]]],
    build_prompt: Callable[..., List[Dict[str, Any]]] = default_build_prompt,
    concurrency: int = 64,
    max_tokens: int = 1,
    meta: Optional[Dict[str, Any]] = None,
    desc: str = "lora-bench",
) -> pd.DataFrame:
    """Sync wrapper over `benchmark_loras` (uses `asyncio.run`).

    See `benchmark_loras` for argument and return-value documentation.
    """
    return asyncio.run(
        benchmark_loras(
            client=client,
            adapters=adapters,
            inputs=inputs,
            build_prompt=build_prompt,
            concurrency=concurrency,
            max_tokens=max_tokens,
            meta=meta,
            desc=desc,
        )
    )
