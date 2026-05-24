"""
Offline multi-LoRA benchmark for vLLM (embedded ``LLM`` mode, no HTTP).

For each input, submits one ``LLM.chat`` call containing ``n_adapters``
requests (same prompt, each with a different LoRA). This gives every input
its own wall-clock time, from which per-input RPS is computed directly —
no vLLM internal metrics needed.

Reported metrics
----------------
rps          — aggregate inputs per second over the full run.
rps_p50/p95/p99 — percentiles of per-input instantaneous RPS
               (``1 / per_input_wall_time``).
n_inputs     — total number of input pairs submitted.
n_failed     — adapter responses that returned no output.
wall_time_s  — total elapsed wall-clock time in seconds.

Example
-------
::

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vlm_inference import VLM_Runner
    from lora_benchmark_offline import benchmark_loras_offline, make_prompt_builder

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
        verbose=True,
    )
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vlm_inference import VLM_Runner

__all__ = ["benchmark_loras_offline", "make_prompt_builder"]

logger = logging.getLogger(__name__)

# Reused across calls — only the prefill pass is needed for scoring.
_DEFAULT_SAMPLING = SamplingParams(max_tokens=1, temperature=0.0)


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

    Instantaneous RPS for each input is ``1 / wall_time``. Returns ``None``
    values when the list is empty.
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


def benchmark_loras_offline(
    llm: LLM,
    adapters: List[LoRARequest],
    inputs: List[Tuple[str, Optional[str]]],
    build_prompt: Callable[..., List[Dict[str, Any]]],
    sampling_params: Optional[SamplingParams] = None,
    meta: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Benchmark a multi-LoRA workload via vLLM's offline ``LLM.chat``.

    For each input, issues one ``LLM.chat`` call with ``len(adapters)``
    requests (same prompt, each paired with a different LoRA adapter).
    Per-input wall-clock times are recorded directly, enabling RPS percentiles
    without relying on vLLM's internal request metrics.

    Args:
        llm: vLLM ``LLM`` started with ``enable_lora=True`` and
            ``max_loras >= len(adapters)``.
        adapters: One ``LoRARequest`` per adapter; each must have a unique
            ``lora_int_id``.
        inputs: List of ``(text, img_url)`` pairs.
        build_prompt: Callable ``(text=..., img_url=...) -> messages``. Use
            ``make_prompt_builder(runner, project_tag)`` to obtain one
            consistent with the project's production prompt format.
        sampling_params: Defaults to ``SamplingParams(max_tokens=1,
            temperature=0.0)`` — only the prefill pass is needed for scoring.
        meta: Arbitrary key-value pairs stamped into the output row
            (e.g. ``{"n_loras": 4, "rank": 16, "target_modules": "qkv"}``).
        verbose: Emit INFO-level progress via the module logger and show a
            tqdm progress bar over inputs.

    Returns:
        Single-row :class:`pandas.DataFrame` with columns:
        ``<meta keys>``, ``rps``, ``rps_p50``, ``rps_p95``, ``rps_p99``,
        ``n_inputs``, ``n_failed``, ``wall_time_s``.
    """
    sampling = sampling_params or _DEFAULT_SAMPLING

    if verbose:
        logger.info(
            "Offline benchmark: %d inputs × %d adapters, %d calls total.",
            len(inputs),
            len(adapters),
            len(inputs),
        )

    per_input_times: List[float] = []
    n_failed = 0

    wall_start = time.perf_counter()
    for text, img_url in tqdm(inputs, desc="Inputs", disable=not verbose):
        prompt = build_prompt(text=text, img_url=img_url)
        messages = [prompt] * len(adapters)

        t0 = time.perf_counter()
        outputs = llm.chat(
            messages=messages,
            sampling_params=sampling,
            lora_request=adapters,
            use_tqdm=False,
        )
        per_input_times.append(time.perf_counter() - t0)
        n_failed += sum(1 for o in outputs if not o.outputs)

    wall_time = time.perf_counter() - wall_start
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
