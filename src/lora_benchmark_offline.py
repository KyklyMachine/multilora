"""
Offline multi-LoRA benchmark for vLLM (no OpenAI API).

Builds one large batch of (input x adapter) requests and submits it to a vLLM
offline `LLM` instance in a single `LLM.chat` call, so the scheduler has the
widest possible batching window. Returns the same long-format DataFrame as
`lora_benchmark.run_benchmark`, so results from both modes can be concatenated
into one comparison table.

Example:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vlm_inference import VLM_Runner
    from lora_benchmark_offline import (
        build_prompt_for_project,
        run_offline_benchmark,
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
        LoRARequest("a3", 3, "/path/a3"),
        LoRARequest("a4", 4, "/path/a4"),
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
    build_prompt = build_prompt_for_project(runner, "ml_audit_sgc_photo_title_mismatch")
    inputs = list(zip(df["text_value"], df["local_image_path"]))

    df_result = run_offline_benchmark(
        llm=llm,
        adapters=adapters,
        inputs=inputs,
        build_prompt=build_prompt,
        meta={"n_loras": 4, "rank": 16, "mode": "offline_batch"},
    )
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from vlm_inference import VLM_Runner


logger = logging.getLogger(__name__)


def build_prompt_for_project(
    runner: VLM_Runner,
    project_tag: str,
) -> Callable[..., List[Dict[str, Any]]]:
    """Configure `runner` for `project_tag` and return its prompt builder.

    Same helper as in `lora_benchmark.py`, duplicated here so this script is
    self-contained.
    """
    runner.set_project_tag(project_tag)
    return runner.build_prompt


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


def run_offline_benchmark(
    llm: LLM,
    adapters: List[LoRARequest],
    inputs: List[Tuple[str, Optional[str]]],
    build_prompt: Callable[..., List[Dict[str, Any]]],
    sampling_params: Optional[SamplingParams] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Benchmark LoRA adapters via vLLM offline `LLM.chat`, one big batch.

    Builds `len(inputs) * len(adapters)` chat requests and submits them in a
    single call. vLLM packs tokens from all adapters and inputs into the same
    scheduler step (BGMV applies the per-position LoRA dropdowns), which is
    the widest batching window the engine is capable of exposing.

    Args:
        llm: vLLM `LLM` started with `enable_lora=True` and `max_loras` >=
            `len(adapters)`.
        adapters: One `LoRARequest` per adapter to benchmark. Each must have
            a unique `lora_int_id`.
        inputs: List of `(text, img_url)` pairs.
        build_prompt: Callable `(text=..., img_url=...) -> messages`. Use
            `build_prompt_for_project(runner, project_tag)` to obtain a
            builder consistent with the rest of the project.
        sampling_params: Defaults to `SamplingParams(max_tokens=1,
            temperature=0.0)` — only the prefill is needed for an embedding.
        meta: Server-config attributes stamped into every output row.

    Returns:
        Long-format DataFrame identical in schema to
        `lora_benchmark.run_benchmark`: one row per adapter plus one
        `end_to_end` row. Per-request latency is taken from vLLM's
        `RequestMetrics` (`finished_time - arrival_time`); end-to-end per
        input is `max(finished_time) - min(arrival_time)` across that
        input's adapter requests.
    """
    if sampling_params is None:
        sampling_params = SamplingParams(max_tokens=1, temperature=0.0)

    per_input_messages: List[List[Dict[str, Any]]] = [
        build_prompt(text=t, img_url=u) for t, u in inputs
    ]

    all_messages: List[List[Dict[str, Any]]] = []
    all_lora: List[LoRARequest] = []
    tags: List[Tuple[int, str]] = []
    for input_idx, msgs in enumerate(per_input_messages):
        for adapter in adapters:
            all_messages.append(msgs)
            all_lora.append(adapter)
            tags.append((input_idx, adapter.lora_name))

    logger.info(
        f"Offline batch: {len(inputs)} inputs x {len(adapters)} adapters "
        f"= {len(all_messages)} requests in a single LLM.chat call"
    )

    wall_t0 = time.perf_counter()
    outputs = llm.chat(
        messages=all_messages,
        sampling_params=sampling_params,
        lora_request=all_lora,
        use_tqdm=True,
    )
    wall_time = time.perf_counter() - wall_t0

    per_request: List[Dict[str, Any]] = []
    for (input_idx, adapter_name), out in zip(tags, outputs):
        m = out.metrics
        if (
            m is not None
            and m.finished_time is not None
            and m.arrival_time is not None
        ):
            latency = float(m.finished_time - m.arrival_time)
        else:
            latency = float("nan")
        per_request.append(
            {
                "input_idx": input_idx,
                "adapter": adapter_name,
                "latency": latency,
                "success": bool(out.outputs),
            }
        )

    by_input: Dict[int, List[Any]] = {}
    for tag, out in zip(tags, outputs):
        by_input.setdefault(tag[0], []).append(out)
    e2e: List[float] = []
    for outs in by_input.values():
        ms = [o.metrics for o in outs if o.metrics is not None]
        if ms and all(
            m.arrival_time is not None and m.finished_time is not None for m in ms
        ):
            e2e.append(
                max(m.finished_time for m in ms) - min(m.arrival_time for m in ms)
            )

    common: Dict[str, Any] = {
        **(meta or {}),
        "wall_time_s": wall_time,
        "concurrency": len(all_messages),
        "max_tokens": sampling_params.max_tokens,
        "n_inputs": len(inputs),
        "n_adapters": len(adapters),
        "n_failed": sum(1 for r in per_request if not r["success"]),
    }

    rows: List[Dict[str, Any]] = []
    for adapter in adapters:
        lats = [
            r["latency"]
            for r in per_request
            if r["adapter"] == adapter.lora_name and r["success"]
        ]
        lats = [x for x in lats if not np.isnan(x)]
        s = _stats(lats)
        rows.append(
            {
                **common,
                "scope": "per_adapter",
                "adapter": adapter.lora_name,
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
