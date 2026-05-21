"""
Asynchronous multi-LoRA benchmark utilities for vLLM-served VLMs.

Mirrors the style of `async_openai_utils.py`: each input goes through every
configured LoRA adapter concurrently, per-adapter and aggregate (per-input,
"all adapters in parallel") latency / throughput stats are collected and
optionally written back to the source DataFrame for downstream analysis.
"""

import asyncio
import logging
import statistics
import time
from asyncio import Semaphore, gather
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


logger = logging.getLogger(__name__)


async def call_adapter(
    create_request: Callable,
    client: AsyncOpenAI,
    adapter: str,
    semaphore: Semaphore,
    text: str = "",
    img_url: Optional[str] = None,
    max_tokens: int = 1,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Send a single request to one LoRA adapter and time it.

    Unlike `async_openai_utils.process_row`, this helper does NOT retry —
    retried requests would skew latency percentiles and RPS. Failures are
    recorded explicitly so the report shows them.

    Args:
        create_request: Callable that builds OpenAI-style chat messages.
            Must accept `text=` and `img_url=` kwargs.
        client: AsyncOpenAI client pointing at the vLLM server.
        adapter: LoRA adapter name registered in vLLM (see `--lora-modules`).
        semaphore: Concurrency limiter (passed in so it is per-benchmark-run,
            not module-global).
        text: Optional text input.
        img_url: Optional image URL.
        max_tokens: Generation cap. Use 1 if only the first decoder hidden
            state / embedding is needed — prefill dominates anyway.
        temperature: Sampling temperature (0.0 for deterministic latency).

    Returns:
        Dict with adapter, success, latency, prompt_tokens, completion_tokens,
        error, started_at, finished_at.
    """
    messages = create_request(text=text, img_url=img_url)
    started_at = time.perf_counter()

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=adapter,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            finished_at = time.perf_counter()
            usage = response.usage
            return {
                "adapter": adapter,
                "success": True,
                "latency": finished_at - started_at,
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "error": None,
                "started_at": started_at,
                "finished_at": finished_at,
            }
        except Exception as e:
            finished_at = time.perf_counter()
            logger.warning(f"[{adapter}] request failed: {type(e).__name__} - {e}")
            return {
                "adapter": adapter,
                "success": False,
                "latency": finished_at - started_at,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "error": f"{type(e).__name__}: {e}",
                "started_at": started_at,
                "finished_at": finished_at,
            }


async def process_row_multilora(
    create_request: Callable,
    client: AsyncOpenAI,
    adapters: List[str],
    semaphore: Semaphore,
    row_idx: int,
    text: str,
    img_url: Optional[str],
    max_tokens: int,
    records: List[Dict[str, Any]],
    progress_bar: tqdm_asyncio,
) -> Dict[str, Any]:
    """Fire all adapters for a single input concurrently.

    The end-to-end latency reported here is "time to get embeddings from all N
    adapters for this row" — the metric you actually care about in production
    when one user input needs to be scored by every adapter.

    Args:
        create_request: Message builder.
        client: AsyncOpenAI client.
        adapters: LoRA adapter names to query.
        semaphore: Shared concurrency limiter.
        row_idx: Index of this input in the source list / DataFrame.
        text: Text input.
        img_url: Optional image URL.
        max_tokens: Generation cap per adapter.
        records: Mutable list to append per-request records to.
        progress_bar: Async tqdm bar.

    Returns:
        {"row_idx": int, "end_to_end": float, "per_adapter": [...]}.
    """
    t0 = time.perf_counter()
    per_adapter = await gather(
        *[
            call_adapter(
                create_request=create_request,
                client=client,
                adapter=adapter,
                semaphore=semaphore,
                text=text,
                img_url=img_url,
                max_tokens=max_tokens,
            )
            for adapter in adapters
        ]
    )
    end_to_end = time.perf_counter() - t0

    for r in per_adapter:
        r["row_idx"] = row_idx
        records.append(r)
    progress_bar.update(1)

    return {
        "row_idx": row_idx,
        "end_to_end": end_to_end,
        "per_adapter": per_adapter,
    }


async def run_multilora_benchmark(
    create_request: Callable,
    client: AsyncOpenAI,
    adapters: List[str],
    inputs: List[Tuple[str, Optional[str]]],
    max_tokens: int = 1,
    concurrency: int = 64,
    desc: str = "multilora",
) -> Dict[str, Any]:
    """Run the benchmark over a list of (text, img_url) inputs.

    Args:
        create_request: Message builder (e.g. `VLM_Runner.build_prompt`).
        client: AsyncOpenAI client targeting the vLLM server.
        adapters: LoRA adapter names registered in vLLM.
        inputs: List of (text, img_url) pairs to benchmark on.
        max_tokens: Generation cap (1 if only embedding is consumed).
        concurrency: Max in-flight requests across all adapters and inputs.
            This is the knob you sweep to find the saturation point.
        desc: tqdm description.

    Returns:
        Dict with keys: records, end_to_end, wall_time, adapters,
        concurrency, max_tokens.
    """
    semaphore = Semaphore(concurrency)
    records: List[Dict[str, Any]] = []
    progress_bar = tqdm_asyncio(total=len(inputs), desc=desc)

    wall_t0 = time.perf_counter()
    per_row = await gather(
        *[
            process_row_multilora(
                create_request=create_request,
                client=client,
                adapters=adapters,
                semaphore=semaphore,
                row_idx=i,
                text=text,
                img_url=img_url,
                max_tokens=max_tokens,
                records=records,
                progress_bar=progress_bar,
            )
            for i, (text, img_url) in enumerate(inputs)
        ]
    )
    wall_time = time.perf_counter() - wall_t0
    progress_bar.close()

    return {
        "records": records,
        "end_to_end": [r["end_to_end"] for r in per_row],
        "wall_time": wall_time,
        "adapters": adapters,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
    }


def _percentile(sorted_values: List[float], p: float) -> float:
    """Nearest-rank percentile on a pre-sorted list."""
    if not sorted_values:
        return 0.0
    k = max(0, min(len(sorted_values) - 1, int(round(p / 100 * (len(sorted_values) - 1)))))
    return sorted_values[k]


def summarize(values: List[float]) -> Dict[str, float]:
    """Mean / min / max / p50 / p95 / p99 of a list of latencies (seconds)."""
    if not values:
        return {"n": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    s = sorted(values)
    return {
        "n": len(s),
        "mean": statistics.fmean(s),
        "min": s[0],
        "max": s[-1],
        "p50": _percentile(s, 50),
        "p95": _percentile(s, 95),
        "p99": _percentile(s, 99),
    }


def generate_multilora_report(results: Dict[str, Any]) -> str:
    """Build a human-readable benchmark report from `run_multilora_benchmark` output.

    Reports two views:
        1. Per-adapter latency / throughput — how each adapter performs in isolation.
        2. End-to-end per-input — how long it takes to score one input across
           every adapter concurrently. This is the SLO-relevant number.
    """
    records: List[Dict[str, Any]] = results["records"]
    e2e: List[float] = results["end_to_end"]
    wall_time: float = results["wall_time"]
    adapters: List[str] = results["adapters"]
    n_inputs = len(e2e)

    ok = [r for r in records if r["success"]]
    failed = [r for r in records if not r["success"]]

    lines: List[str] = []
    add = lines.append
    add("=" * 78)
    add(f"Multi-LoRA benchmark report")
    add("=" * 78)
    add(f"Adapters      : {len(adapters)}  [{', '.join(adapters)}]")
    add(f"Inputs        : {n_inputs}")
    add(f"Concurrency   : {results['concurrency']}")
    add(f"max_tokens    : {results['max_tokens']}")
    add(f"Wall time     : {wall_time:.2f} s")
    add(
        f"Requests      : {len(records)} total  |  "
        f"{len(ok)} ok  |  {len(failed)} failed"
    )
    add(f"Request RPS   : {len(ok) / wall_time:.2f} req/s (successful only)")
    add(f"Input RPS     : {n_inputs / wall_time:.2f} input/s "
        f"(one input = {len(adapters)} requests)")
    add("")
    add("Per-adapter latency (successful requests, seconds):")
    header = f"  {'adapter':<22}{'n':>6}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}{'rps':>9}"
    add(header)
    add("  " + "-" * (len(header) - 2))
    for adapter in adapters:
        lats = [r["latency"] for r in ok if r["adapter"] == adapter]
        s = summarize(lats)
        rps = s["n"] / wall_time if wall_time > 0 else 0.0
        add(
            f"  {adapter:<22}{s['n']:>6}{s['mean']:>9.3f}{s['p50']:>9.3f}"
            f"{s['p95']:>9.3f}{s['p99']:>9.3f}{s['max']:>9.3f}{rps:>9.2f}"
        )

    add("")
    add("End-to-end latency per input (all adapters in parallel, seconds):")
    s = summarize(e2e)
    add(f"  mean={s['mean']:.3f}  p50={s['p50']:.3f}  "
        f"p95={s['p95']:.3f}  p99={s['p99']:.3f}  max={s['max']:.3f}")

    if failed:
        add("")
        add(f"Failures ({len(failed)}):")
        # Show up to 5 distinct error signatures
        seen: Dict[str, int] = {}
        for r in failed:
            seen[r["error"]] = seen.get(r["error"], 0) + 1
        for err, n in list(seen.items())[:5]:
            add(f"  x{n:<4} {err}")
    add("=" * 78)
    return "\n".join(lines)


def records_to_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """Flatten per-request records into a DataFrame for further analysis / plotting."""
    df = pd.DataFrame(results["records"])
    return df[
        [
            "row_idx",
            "adapter",
            "success",
            "latency",
            "prompt_tokens",
            "completion_tokens",
            "started_at",
            "finished_at",
            "error",
        ]
    ]


def inputs_from_dataframe(
    df: pd.DataFrame,
    text_column: Optional[str] = "text_value",
    image_column: Optional[str] = "local_image_path",
) -> List[Tuple[str, Optional[str]]]:
    """Pull (text, img_url) pairs out of a DataFrame in the project format."""
    texts = df[text_column].fillna("").astype(str).tolist() if text_column else [""] * len(df)
    imgs = df[image_column].tolist() if image_column else [None] * len(df)
    return list(zip(texts, imgs))


async def sweep_concurrency(
    create_request: Callable,
    client: AsyncOpenAI,
    adapters: List[str],
    inputs: List[Tuple[str, Optional[str]]],
    concurrencies: List[int],
    max_tokens: int = 1,
) -> List[Dict[str, Any]]:
    """Re-run the benchmark at several concurrency levels to find saturation.

    Useful for the "at what RPS does p95 start to blow up" question.

    Returns:
        One result dict per concurrency level, each enriched with the summary
        of end-to-end latency for convenience.
    """
    out: List[Dict[str, Any]] = []
    for c in concurrencies:
        logger.info(f"sweep: concurrency={c}")
        res = await run_multilora_benchmark(
            create_request=create_request,
            client=client,
            adapters=adapters,
            inputs=inputs,
            max_tokens=max_tokens,
            concurrency=c,
            desc=f"multilora c={c}",
        )
        res["e2e_summary"] = summarize(res["end_to_end"])
        out.append(res)
    return out
