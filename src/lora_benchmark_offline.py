"""
Offline multi-LoRA benchmark for vLLM (embedded `LLM` mode, no HTTP).

Submits every `(input, adapter)` pair as a single batched `LLM.chat` call so
vLLM's scheduler sees the entire workload at once and packs prefill across
adapters using BGMV kernels. Returns one aggregate row per call, ready to
be concatenated across runs into a single comparison table.

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
        meta={"n_loras": len(adapters), "rank": 16, "mode": "offline_batch"},
    )
    # `row` is a single-row DataFrame; pd.concat([...]) across runs.
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
    and reuses the project's existing prompt-construction utilities
    (`get_openai_prompt` / `get_internvl_prompt`), so benchmark requests are
    formed identically to production inference requests.
    """
    runner.set_project_tag(project_tag)
    return runner.build_prompt


def _latency_summary(values: List[float], prefix: str) -> Dict[str, float]:
    """Aggregate latencies into `{prefix}_{mean,p50,p95,p99,max}_s` columns."""
    if not values:
        agg = {"mean_s": 0.0, "p50_s": 0.0, "p95_s": 0.0, "p99_s": 0.0, "max_s": 0.0}
    else:
        arr = np.asarray(values, dtype=float)
        p50, p95, p99 = np.percentile(arr, [50, 95, 99])
        agg = {
            "mean_s": float(arr.mean()),
            "p50_s": float(p50),
            "p95_s": float(p95),
            "p99_s": float(p99),
            "max_s": float(arr.max()),
        }
    return {f"{prefix}_{key}": value for key, value in agg.items()}


def _expand_requests(
    inputs: List[Tuple[str, Optional[str]]],
    adapters: List[LoRARequest],
    build_prompt: Callable[..., List[Dict[str, Any]]],
) -> Tuple[List[List[Dict[str, Any]]], List[LoRARequest], List[int]]:
    """Cross-product `inputs × adapters` into three positionally-aligned lists.

    Returns:
        `(messages, lora_requests, input_index)` of length
        `len(inputs) * len(adapters)`. `input_index[k]` tells which input
        request `k` came from, used later to group end-to-end latencies.
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


def _collect_timings(
    outputs: List[Any],
    input_index: List[int],
) -> Tuple[List[float], List[float], int]:
    """Pull per-request and per-input timings out of vLLM `RequestOutput`s.

    Per-request latency = `finished_time - arrival_time` from `RequestMetrics`.
    Per-input end-to-end = wall clock between the earliest arrival and the
    latest finish among that input's adapter requests — the time an upstream
    caller would actually observe waiting for all adapter embeddings of one
    input.

    Returns:
        `(request_latencies, end_to_end_per_input, n_failed)`.
    """
    request_latencies: List[float] = []
    n_failed = 0
    grouped: Dict[int, List[Any]] = defaultdict(list)

    for output, idx in zip(outputs, input_index):
        if not output.outputs:
            n_failed += 1
            continue
        metrics = output.metrics
        if (
            metrics
            and metrics.arrival_time is not None
            and metrics.finished_time is not None
        ):
            request_latencies.append(
                float(metrics.finished_time - metrics.arrival_time)
            )
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

    return request_latencies, end_to_end, n_failed


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
    vLLM's scheduler sees the entire workload at once.

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
        Single-row DataFrame ready for `pd.concat` across runs. Columns:
        `<meta keys>, wall_time_s, n_inputs, n_adapters, n_requests,
        n_failed, max_tokens, request_rps, input_rps,
        req_{mean,p50,p95,p99,max}_s, e2e_{mean,p50,p95,p99,max}_s`.
    """
    sampling = sampling_params or SamplingParams(max_tokens=1, temperature=0.0)

    messages, lora_requests, input_index = _expand_requests(
        inputs, adapters, build_prompt
    )
    n_requests = len(messages)

    logger.info(
        "Offline batch: %d inputs x %d adapters = %d requests in a single LLM.chat call",
        len(inputs),
        len(adapters),
        n_requests,
    )

    wall_start = time.perf_counter()
    outputs = llm.chat(
        messages=messages,
        sampling_params=sampling,
        lora_request=lora_requests,
        use_tqdm=True,
    )
    wall_time = time.perf_counter() - wall_start

    request_latencies, end_to_end, n_failed = _collect_timings(outputs, input_index)
    n_ok = n_requests - n_failed

    row: Dict[str, Any] = {
        **(meta or {}),
        "wall_time_s": wall_time,
        "n_inputs": len(inputs),
        "n_adapters": len(adapters),
        "n_requests": n_requests,
        "n_failed": n_failed,
        "max_tokens": sampling.max_tokens,
        "request_rps": n_ok / wall_time if wall_time > 0 else 0.0,
        "input_rps": len(inputs) / wall_time if wall_time > 0 else 0.0,
        **_latency_summary(request_latencies, "req"),
        **_latency_summary(end_to_end, "e2e"),
    }
    return pd.DataFrame([row])
