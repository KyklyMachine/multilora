"""
Client-side batched multi-LoRA benchmark for a running vLLM HTTP server.

Forms batches on the client via `/v1/completions` with an array `prompt`
field: one HTTP request carries `batch_size` prompts to one adapter. Uses
raw httpx (no openai SDK). The vLLM server must be started separately
(`vllm serve ... --enable-lora --max-loras N ...`).

LIMITATION: `/v1/completions` is text-only. If your real workload includes
images, this benchmark cannot reproduce it exactly — use plain text
prompts here to isolate the question "does client-side batching help RPS?".
For the multimodal path see `lora_benchmark.py` (HTTP, per-request) or
`lora_benchmark_offline.py` (embedded LLM, one big batch).

Schema of the returned DataFrame is kept compatible with
`lora_benchmark.run_benchmark` so results from all three modes can be
concatenated for comparison.

Example:
    from lora_benchmark_batched import run_self_batched_benchmark

    text_prompts = ["...", "...", ...]  # already chat-templated strings
    df_result = run_self_batched_benchmark(
        base_url="http://localhost:8000",
        adapters=["a1", "a2", "a3", "a4"],
        prompts=text_prompts,
        batch_size=64,
        concurrency=4,
        meta={"n_loras": 4, "rank": 16, "mode": "client_batched"},
    )
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm_asyncio


logger = logging.getLogger(__name__)


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


def _chunk(seq: List[str], size: int) -> List[List[str]]:
    """Split `seq` into consecutive chunks of length `size`."""
    return [seq[i : i + size] for i in range(0, len(seq), size)]


async def _post_batch(
    client: httpx.AsyncClient,
    adapter: str,
    prompts_chunk: List[str],
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """One HTTP POST to `/v1/completions` carrying multiple prompts for one adapter."""
    payload = {
        "model": adapter,
        "prompt": prompts_chunk,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    async with semaphore:
        t0 = time.perf_counter()
        try:
            resp = await client.post("/v1/completions", json=payload)
            resp.raise_for_status()
            n_returned = len(resp.json().get("choices", []))
            success, err = True, None
        except Exception as e:
            success, err = False, f"{type(e).__name__}: {e}"
            n_returned = 0
            logger.warning(f"[{adapter}] batch failed: {err}")
        t1 = time.perf_counter()
    return {
        "adapter": adapter,
        "success": success,
        "latency": t1 - t0,
        "batch_size": len(prompts_chunk),
        "n_returned": n_returned,
        "error": err,
    }


async def benchmark_self_batched(
    base_url: str,
    adapters: List[str],
    prompts: List[str],
    batch_size: Optional[int] = None,
    concurrency: int = 4,
    max_tokens: int = 1,
    request_timeout: float = 600.0,
    meta: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Send batched `/v1/completions` requests, one HTTP call per chunk per adapter.

    Args:
        base_url: vLLM server base URL (e.g. "http://localhost:8000").
        adapters: LoRA names registered in the server (`--lora-modules`).
        prompts: Plain text prompts. Pre-render any chat template yourself —
            `/v1/completions` does not apply one.
        batch_size: Prompts per HTTP request. `None` packs all prompts into
            one request per adapter (maximum client-side batching).
        concurrency: Max HTTP requests in flight across all chunks/adapters.
        max_tokens: Generation cap. Keep at 1 for embedding-style benchmarks.
        request_timeout: Per-request httpx timeout (seconds). Big batches +
            small concurrency may need this higher.
        meta: Server-config attributes stamped into every output row.

    Returns:
        Long-format DataFrame: one row per adapter (`scope="per_adapter"`)
        plus one aggregate row (`scope="end_to_end"`). Column schema matches
        `lora_benchmark.run_benchmark`, with extra columns `batch_size` and
        `n_completions`. Latency percentiles are over HTTP-request (chunk)
        times — `/v1/completions` does not expose per-prompt timing.
    """
    effective_batch = batch_size if batch_size and batch_size > 0 else len(prompts)
    chunks = _chunk(prompts, effective_batch)
    n_requests = len(adapters) * len(chunks)

    logger.info(
        f"Self-batched: {len(adapters)} adapters x {len(chunks)} chunks of "
        f"{effective_batch} = {n_requests} HTTP requests; "
        f"{len(prompts) * len(adapters)} total completions"
    )

    semaphore = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(request_timeout, connect=30.0)
    limits = httpx.Limits(
        max_connections=concurrency + 10,
        max_keepalive_connections=concurrency,
    )
    progress = tqdm_asyncio(total=n_requests, desc="self-batched")

    wall_t0 = time.perf_counter()
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout, limits=limits) as client:

        async def _wrapped(adapter: str, chunk: List[str]) -> Dict[str, Any]:
            out = await _post_batch(client, adapter, chunk, max_tokens, semaphore)
            progress.update(1)
            return out

        records = await asyncio.gather(
            *[_wrapped(adapter, chunk) for adapter in adapters for chunk in chunks]
        )
    wall_time = time.perf_counter() - wall_t0
    progress.close()

    common: Dict[str, Any] = {
        **(meta or {}),
        "wall_time_s": wall_time,
        "concurrency": concurrency,
        "batch_size": effective_batch,
        "max_tokens": max_tokens,
        "n_inputs": len(prompts),
        "n_adapters": len(adapters),
        "n_failed": sum(1 for r in records if not r["success"]),
    }

    rows: List[Dict[str, Any]] = []
    for adapter in adapters:
        ok = [r for r in records if r["adapter"] == adapter and r["success"]]
        lats = [r["latency"] for r in ok]
        n_completions = sum(r["n_returned"] for r in ok)
        s = _stats(lats)
        rows.append(
            {
                **common,
                "scope": "per_adapter",
                "adapter": adapter,
                "rps": n_completions / wall_time if wall_time > 0 else 0.0,
                "n_completions": n_completions,
                **s,
            }
        )

    total_completions = sum(r["n_returned"] for r in records if r["success"])
    rows.append(
        {
            **common,
            "scope": "end_to_end",
            "adapter": "all",
            "rps": total_completions / wall_time if wall_time > 0 else 0.0,
            "n_completions": total_completions,
            "n": len(records),
            "mean_s": float("nan"),
            "p50_s": float("nan"),
            "p95_s": float("nan"),
            "p99_s": float("nan"),
            "max_s": wall_time,
        }
    )

    return pd.DataFrame(rows)


def run_self_batched_benchmark(
    base_url: str,
    adapters: List[str],
    prompts: List[str],
    batch_size: Optional[int] = None,
    concurrency: int = 4,
    max_tokens: int = 1,
    request_timeout: float = 600.0,
    meta: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Sync wrapper over `benchmark_self_batched` (uses `asyncio.run`)."""
    return asyncio.run(
        benchmark_self_batched(
            base_url=base_url,
            adapters=adapters,
            prompts=prompts,
            batch_size=batch_size,
            concurrency=concurrency,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
            meta=meta,
        )
    )
