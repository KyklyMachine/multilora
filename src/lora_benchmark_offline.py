"""
Offline multi-LoRA benchmark for vLLM (embedded `LLM` mode, no HTTP).

Submits every `(input, adapter)` pair to a single batched `LLM.chat` call,
which fills vLLM's scheduler queue at once and lets its continuous-batching
scheduler form the widest possible per-step batches. Returns one aggregate
row per call — concatenate rows across runs to compare configurations.

Reported metrics:
    rps         — inputs per second (one input = one logical scoring,
                  which internally produces `n_adapters` embeddings).
    latency_*   — per-input end-to-end seconds: time from the first of an
                  input's adapter requests arriving in the scheduler to
                  the last one finishing. This is the wall-clock an
                  upstream caller would observe.

Example:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vlm_inference import VLM_Runner
    from lora_benchmark_offline import (
        benchmark_loras_offline,
        make_prompt_builder,
    )

    llm = LLM(
        model="Qwen/Qwen3-VL-8B-Instruct",
        enable_lora=True,
        max_loras=8,
        max_lora_rank=64,
        max_model_len=4096,
        dtype="bfloat16",
    )
    adapters = [
        LoRARequest("a1", 1, "/path/a1"),
        LoRARequest("a2", 2, "/path/a2"),
    ]
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

    row = benchmark_loras_offline(
        llm=llm,
        adapters=adapters,
        inputs=inputs,
        build_prompt=build_prompt,
        meta={"n_loras": len(adapters), "rank": 16, "target_modules": "qkv"},
    )
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from vlm_inference import VLM_Runner


__all__ = ["benchmark_loras_offline", "make_prompt_builder"]

logger = logging.getLogger(__name__)


def make_prompt_builder(
    runner: VLM_Runner,
    project_tag: str,
) -> Callable[..., List[Dict[str, Any]]]:
    """Configure `runner` for `project_tag` and return its prompt builder.

    The returned callable has signature `(text=..., img_url=...) -> messages`
    and reuses the project's existing prompt-construction utilities, so
    benchmark requests are formed identically to production inference.
    """
    runner.set_project_tag(project_tag)
    return runner.build_prompt


def _latency_stats(values: List[float]) -> Dict[str, float]:
    """Return `latency_{mean,p50,p95,p99,max}_s` from a list of seconds."""
    if not values:
        return {
            "latency_mean_s": 0.0,
            "latency_p50_s": 0.0,
            "latency_p95_s": 0.0,
            "latency_p99_s": 0.0,
            "latency_max_s": 0.0,
        }
    arr = np.asarray(values, dtype=float)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    return {
        "latency_mean_s": float(arr.mean()),
        "latency_p50_s": float(p50),
        "latency_p95_s": float(p95),
        "latency_p99_s": float(p99),
        "latency_max_s": float(arr.max()),
    }


def _expand_requests(
    inputs: List[Tuple[str, Optional[str]]],
    adapters: List[LoRARequest],
    build_prompt: Callable[..., List[Dict[str, Any]]],
) -> Tuple[List[List[Dict[str, Any]]], List[LoRARequest], List[int]]:
    """Cross-product `inputs × adapters` into three positionally-aligned lists.

    Returns:
        `(messages, lora_requests, input_index)` of length
        `len(inputs) * len(adapters)`. `input_index[k]` records which input
        request `k` came from, used later to group adapter responses by input.
    """
    per_input_messages = [
        build_prompt(text=text, img_url=url) for text, url in inputs
    ]
    messages: List[List[Dict[str, Any]]] = []
    lora_requests: List[LoRARequest] = []
    input_index: List[int] = []
    for idx, msgs in enumerate(per_input_messages):
        for adapter in adapters:
            messages.append(msgs)
            lora_requests.append(adapter)
            input_index.append(idx)
    return messages, lora_requests, input_index


def _collect_end_to_end(
    outputs: List[Any],
    input_index: List[int],
) -> Tuple[List[float], int]:
    """Per-input end-to-end latencies and the failure count.

    End-to-end per input = `max(finished_time) - min(arrival_time)` across
    that input's adapter requests, taken from vLLM's `RequestMetrics`. This
    is the wall clock an upstream caller would observe waiting for all
    adapter embeddings of one input.
    """
    grouped: Dict[int, List[Any]] = defaultdict(list)
    n_failed = 0
    for output, idx in zip(outputs, input_index):
        if not output.outputs:
            n_failed += 1
            continue
        grouped[idx].append(output)

    end_to_end: List[float] = []
    for outs in grouped.values():
        usable = [
            o.metrics
            for o in outs
            if o.metrics
            and o.metrics.arrival_time is not None
            and o.metrics.finished_time is not None
        ]
        if usable:
            end_to_end.append(
                max(m.finished_time for m in usable)
                - min(m.arrival_time for m in usable)
            )
    return end_to_end, n_failed


def benchmark_loras_offline(
    llm: LLM,
    adapters: List[LoRARequest],
    inputs: List[Tuple[str, Optional[str]]],
    build_prompt: Callable[..., List[Dict[str, Any]]],
    sampling_params: Optional[SamplingParams] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Benchmark a multi-LoRA workload via vLLM's offline `LLM.chat`.

    Submits `len(inputs) * len(adapters)` chat requests in a single call so
    vLLM's continuous-batching scheduler sees the entire workload at once.

    Args:
        llm: vLLM `LLM` started with `enable_lora=True` and
            `max_loras >= len(adapters)`.
        adapters: One `LoRARequest` per adapter; each must have a unique
            `lora_int_id`.
        inputs: List of `(text, img_url)` pairs.
        build_prompt: Callable `(text=..., img_url=...) -> messages`. Use
            `make_prompt_builder(runner, project_tag)` to obtain one
            consistent with the project's production prompt format.
        sampling_params: Defaults to `SamplingParams(max_tokens=1,
            temperature=0.0)` — only the prefill is needed for an embedding.
        meta: Server-config attributes stamped into the output row
            (e.g. `{"n_loras": 4, "rank": 16, "target_modules": "qkv"}`).

    Returns:
        Single-row DataFrame. Columns:
            <meta keys you passed>,
            rps,
            latency_mean_s, latency_p50_s, latency_p95_s, latency_p99_s,
            latency_max_s,
            n_inputs, n_failed, wall_time_s.
    """
    sampling = sampling_params or SamplingParams(max_tokens=1, temperature=0.0)
    messages, lora_requests, input_index = _expand_requests(
        inputs, adapters, build_prompt
    )

    logger.info(
        "Offline batch: %d inputs x %d adapters = %d requests in a single LLM.chat call",
        len(inputs),
        len(adapters),
        len(messages),
    )

    wall_start = time.perf_counter()
    outputs = llm.chat(
        messages=messages,
        sampling_params=sampling,
        lora_request=lora_requests,
        use_tqdm=True,
    )
    wall_time = time.perf_counter() - wall_start

    end_to_end, n_failed = _collect_end_to_end(outputs, input_index)
    rps = len(inputs) / wall_time if wall_time > 0 else 0.0

    row: Dict[str, Any] = {
        **(meta or {}),
        "rps": rps,
        **_latency_stats(end_to_end),
        "n_inputs": len(inputs),
        "n_failed": n_failed,
        "wall_time_s": wall_time,
    }
    return pd.DataFrame([row])
    
