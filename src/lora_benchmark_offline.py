"""
Multi-LoRA benchmark against a running vLLM server (OpenAI-compatible API).

All ``n_inputs × n_adapters`` requests are submitted concurrently via
``asyncio.gather``, so vLLM's scheduler sees the full workload at once.
The server handles chat-template formatting and tokenisation internally —
no local tokenizer or engine required.

Prerequisites
-------------
The vLLM server must be started with LoRA support and the adapters registered::

    vllm serve Qwen/Qwen3-VL-8B-Instruct \\
        --enable-lora \\
        --lora-modules a1=/path/a1 a2=/path/a2 \\
        --max-loras 8 \\
        --max-lora-rank 64

Reported metrics
----------------
rps             — aggregate inputs per second over the full run.
rps_p50/p95/p99 — percentiles of per-input instantaneous RPS
                  (``1 / per_input_time``).
n_inputs        — total number of input pairs submitted.
n_failed        — requests that raised an exception or returned no choices.
wall_time_s     — total elapsed wall-clock time in seconds.

Example
-------
::

    from openai import AsyncOpenAI
    from vlm_inference import VLM_Runner
    from lora_benchmark_server import benchmark_loras_server, make_prompt_builder

    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="token")

    runner = VLM_Runner(
        model_name="qwen_inference",
        exp_name="_lora_bench",
        work_dir=WORK_DIR,
        is_rus=True,
        continue_if_exists=False,
        model_path=LLM_PATH,
        save_results=False,
    )
    build_prompt = make_prompt_builder(runner, "ml_audit_sgc_photo_title_mismatch")
    inputs = list(zip(df["text_value"], df["local_image_path"]))

    row = benchmark_loras_server(
        client=client,
        adapter_names=["a1", "a2"],          # must match --lora-modules names
        inputs=inputs,
        build_prompt=build_prompt,
        meta={"n_loras": 2, "rank": 16, "target_modules": "qkv"},
        verbose=True,
    )
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm
from vlm_inference import VLM_Runner

__all__ = ["benchmark_loras_server", "make_prompt_builder"]

logger = logging.getLogger(__name__)


def make_prompt_builder(
    runner: VLM_Runner,
    project_tag: str,
) -> Callable[..., List[Dict[str, Any]]]:
    """Configure *runner* for *project_tag* and return its prompt builder.

    The returned callable has the signature ``(text=..., img_url=...) ->
    messages`` and reuses the project's prompt-construction utilities, so
    benchmark requests are formed identically to production inference.
    """
    runner.set_project_tag(project_tag)
    return runner.build_prompt


def _rps_percentiles(per_input_times: List[float]) -> Dict[str, Optional[float]]:
    """Compute RPS percentiles from per-input wall-clock times.

    Instantaneous RPS for each input is ``1 / per_input_time``. Returns
    ``None`` values when the list is empty.
    """
    if not per_input_times:
        return {"rps_p50": None, "rps_p95": None, "rps_p99": None}

    instant_rps = 1.0 / np.asarray(per_input_times, dtype=float)
    p50, p95, p99 = np.percentile(instant_rps, [50, 95, 99])
    return {
        "rps_p50": float(p50),
        "rps_p95": float(p95),
        "rps_p99": float(p99),
    }


async def _run_benchmark(
    client: AsyncOpenAI,
    inputs: List[Tuple[str, Optional[str]]],
    adapter_names: List[str],
    build_prompt: Callable[..., List[Dict[str, Any]]],
    max_tokens: int,
    temperature: float,
    verbose: bool,
) -> Tuple[List[float], float, int]:
    """Submit all requests concurrently and collect per-input finish times.

    Returns:
        ``(per_input_times, wall_time, n_failed)`` where ``per_input_times[i]``
        is the elapsed seconds from ``wall_start`` until input ``i``'s last
        adapter response arrived.
    """
    n_inputs = len(inputs)
    messages_list = [
        build_prompt(text=text, img_url=img_url) for text, img_url in inputs
    ]

    input_finish: Dict[int, float] = {}
    n_failed = 0
    lock = asyncio.Lock()
    pbar = tqdm(
        total=n_inputs * len(adapter_names),
        desc="Requests",
        unit="req",
        disable=not verbose,
    )

    async def process_one(input_idx: int, adapter_name: str) -> None:
        nonlocal n_failed
        try:
            response = await client.chat.completions.create(
                model=adapter_name,
                messages=messages_list[input_idx],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            finish_time = time.perf_counter()
            has_output = bool(response.choices)
        except Exception as exc:
            logger.debug("Request failed (input=%d, adapter=%s): %s", input_idx, adapter_name, exc)
            finish_time = time.perf_counter()
            has_output = False

        async with lock:
            if not has_output:
                n_failed += 1
            else:
                input_finish[input_idx] = max(
                    input_finish.get(input_idx, 0.0), finish_time
                )
            pbar.update(1)

    wall_start = time.perf_counter()
    await asyncio.gather(
        *(
            process_one(idx, adapter_name)
            for idx in range(n_inputs)
            for adapter_name in adapter_names
        )
    )
    wall_time = time.perf_counter() - wall_start
    pbar.close()

    per_input_times = [
        input_finish[i] - wall_start
        for i in range(n_inputs)
        if i in input_finish
    ]
    return per_input_times, wall_time, n_failed


def benchmark_loras_server(
    client: AsyncOpenAI,
    adapter_names: List[str],
    inputs: List[Tuple[str, Optional[str]]],
    build_prompt: Callable[..., List[Dict[str, Any]]],
    max_tokens: int = 1,
    temperature: float = 0.0,
    meta: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Benchmark a multi-LoRA workload against a running vLLM server.

    Submits all ``len(inputs) × len(adapter_names)`` requests concurrently
    so vLLM's scheduler sees the entire workload at once. Chat-template
    formatting is handled server-side — no local tokenizer needed.

    Args:
        client: ``AsyncOpenAI`` pointed at the vLLM server
            (e.g. ``base_url="http://localhost:8000/v1"``).
        adapter_names: Adapter names as registered via ``--lora-modules``
            when the server was started.
        inputs: List of ``(text, img_url)`` pairs.
        build_prompt: Callable ``(text=..., img_url=...) -> messages``. Use
            ``make_prompt_builder(runner, project_tag)`` to obtain one
            consistent with the project's production prompt format.
        max_tokens: Token budget per response. Defaults to ``1`` — only the
            prefill pass is needed for scoring.
        temperature: Sampling temperature. Defaults to ``0.0``.
        meta: Arbitrary key-value pairs stamped into the output row
            (e.g. ``{"n_loras": 4, "rank": 16, "target_modules": "qkv"}``).
        verbose: Emit INFO-level progress via the module logger and show a
            tqdm progress bar over requests.

    Returns:
        Single-row :class:`pandas.DataFrame` with columns:
        ``<meta keys>``, ``rps``, ``rps_p50``, ``rps_p95``, ``rps_p99``,
        ``n_inputs``, ``n_failed``, ``wall_time_s``.
    """
    if verbose:
        logger.info(
            "Server benchmark: %d inputs × %d adapters = %d concurrent requests.",
            len(inputs),
            len(adapter_names),
            len(inputs) * len(adapter_names),
        )

    per_input_times, wall_time, n_failed = asyncio.run(
        _run_benchmark(
            client, inputs, adapter_names, build_prompt,
            max_tokens, temperature, verbose,
        )
    )

    rps = len(inputs) / wall_time if wall_time > 0 else 0.0

    if verbose:
        logger.info(
            "Done in %.2fs — %.2f inputs/s | %d failed / %d inputs.",
            wall_time,
            rps,
            n_failed,
            len(inputs),
        )

    row: Dict[str, Any] = {
        **(meta or {}),
        "rps": rps,
        **_rps_percentiles(per_input_times),
        "n_inputs": len(inputs),
        "n_failed": n_failed,
        "wall_time_s": wall_time,
    }
    return pd.DataFrame([row])
