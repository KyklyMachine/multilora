"""
Multi-LoRA benchmark runner — sits next to `vlm_inference.VLM_Runner` and
re-uses the same prompt-building / dataframe conventions, but instead of
producing model answers it fires every configured LoRA adapter against each
input and reports per-adapter / aggregate latency and RPS.
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from openai import AsyncOpenAI

from utils.async_multilora_utils import (
    generate_multilora_report,
    inputs_from_dataframe,
    records_to_dataframe,
    run_multilora_benchmark,
    summarize,
    sweep_concurrency,
)
from vlm_inference import VLM_Runner


logger = logging.getLogger(__name__)


class MultiLoRABenchmarkRunner:
    """Benchmark a vLLM-served VLM across multiple LoRA adapters.

    Re-uses an existing `VLM_Runner` for prompt construction so the same yaml
    prompt files and text/image handling apply. The runner does not write a
    `_answer` column — it writes per-adapter latency records (one row per
    (input, adapter) request) and an aggregated end-to-end CSV.

    Example:
        base_runner = VLM_Runner(
            model_name="qwen_inference",
            exp_name="_lora_bench",
            work_dir=WORK_DIR,
            is_rus=True,
            continue_if_exists=False,
            save_results=False,
        )
        bench = MultiLoRABenchmarkRunner(
            base_runner=base_runner,
            adapters=["lora_a", "lora_b", "lora_c", "lora_d"],
            exp_name="lora_bench_v1",
            work_dir=WORK_DIR,
            save_results=True,
        )
        results = bench.run(
            df=test_dataset,
            project_tag="ml_audit_sgc_photo_title_mismatch",
            client=vllm_client,
            concurrency=64,
            max_tokens=1,
        )
    """

    def __init__(
        self,
        base_runner: VLM_Runner,
        adapters: List[str],
        exp_name: str,
        work_dir: str = ".",
        save_results: bool = True,
    ):
        """Initialize the multi-LoRA runner.

        Args:
            base_runner: Existing `VLM_Runner` used solely for its
                `set_project_tag` / `build_prompt` plumbing.
            adapters: LoRA adapter names registered in vLLM (must match
                whatever you passed to `--lora-modules`).
            exp_name: Experiment identifier for output files.
            work_dir: Working directory; reports/CSVs go to `{work_dir}/exps`.
            save_results: If True, dumps per-request CSV + report .txt.
        """
        self.base_runner = base_runner
        self.adapters = adapters
        self.exp_name = exp_name
        self.work_dir = work_dir
        self.save_results = save_results

    def _output_paths(self, project_tag: str) -> Dict[str, str]:
        exps_dir = f"{self.work_dir}/exps"
        os.makedirs(exps_dir, exist_ok=True)
        stem = f"{exps_dir}/{project_tag}_multilora_exp{self.exp_name}"
        return {
            "records_csv": f"{stem}_records.csv",
            "summary_csv": f"{stem}_summary.csv",
            "report_txt": f"{stem}_report.txt",
        }

    def run(
        self,
        df: pd.DataFrame,
        project_tag: str,
        client: AsyncOpenAI,
        concurrency: int = 64,
        max_tokens: int = 1,
        text_column: Optional[str] = "text_value",
        image_column: Optional[str] = "local_image_path",
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run the benchmark and write per-request CSV + report.

        Args:
            df: Input DataFrame (same schema as VLM_Runner expects).
            project_tag: Task identifier — used to load the right prompt yaml.
            client: AsyncOpenAI client targeting the vLLM server.
            concurrency: Max in-flight requests across all adapters.
            max_tokens: Generation cap per request. Keep at 1 for embeddings.
            text_column: Column with text inputs.
            image_column: Column with image URLs / local paths.
            limit: Optional cap on number of rows (handy for smoke tests).

        Returns:
            The raw benchmark result dict (`records`, `end_to_end`,
            `wall_time`, ...). The report string is also returned under
            `report` and printed to stdout.
        """
        self.base_runner.set_project_tag(project_tag)
        build_prompt = self.base_runner.build_prompt

        work_df = df.reset_index(drop=True)
        if limit is not None:
            work_df = work_df.head(limit).reset_index(drop=True)

        inputs = inputs_from_dataframe(
            work_df, text_column=text_column, image_column=image_column
        )
        logger.info(
            f"Benchmarking {len(self.adapters)} adapters × {len(inputs)} inputs "
            f"= {len(self.adapters) * len(inputs)} requests at concurrency={concurrency}"
        )

        results = asyncio.run(
            run_multilora_benchmark(
                create_request=build_prompt,
                client=client,
                adapters=self.adapters,
                inputs=inputs,
                max_tokens=max_tokens,
                concurrency=concurrency,
                desc=f"{project_tag} multilora",
            )
        )
        report = generate_multilora_report(results)
        results["report"] = report
        logger.info(report)
        print(report)

        if self.save_results:
            paths = self._output_paths(project_tag)
            records_to_dataframe(results).to_csv(paths["records_csv"], index=False)
            self._summary_dataframe(results).to_csv(paths["summary_csv"], index=False)
            with open(paths["report_txt"], "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(
                f"Saved: {paths['records_csv']}, {paths['summary_csv']}, "
                f"{paths['report_txt']}"
            )
            results["paths"] = paths

        return results

    def run_concurrency_sweep(
        self,
        df: pd.DataFrame,
        project_tag: str,
        client: AsyncOpenAI,
        concurrencies: List[int],
        max_tokens: int = 1,
        text_column: Optional[str] = "text_value",
        image_column: Optional[str] = "local_image_path",
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Sweep concurrency levels to locate the saturation point.

        Returns:
            DataFrame with one row per concurrency level: rps, p50/p95/p99
            of end-to-end latency. Useful for picking the right
            `--max-num-seqs` for production.
        """
        self.base_runner.set_project_tag(project_tag)
        build_prompt = self.base_runner.build_prompt

        work_df = df.reset_index(drop=True)
        if limit is not None:
            work_df = work_df.head(limit).reset_index(drop=True)
        inputs = inputs_from_dataframe(
            work_df, text_column=text_column, image_column=image_column
        )

        sweep = asyncio.run(
            sweep_concurrency(
                create_request=build_prompt,
                client=client,
                adapters=self.adapters,
                inputs=inputs,
                concurrencies=concurrencies,
                max_tokens=max_tokens,
            )
        )

        rows = []
        for res in sweep:
            ok = sum(1 for r in res["records"] if r["success"])
            s = res["e2e_summary"]
            rows.append(
                {
                    "concurrency": res["concurrency"],
                    "wall_time_s": res["wall_time"],
                    "successful_requests": ok,
                    "request_rps": ok / res["wall_time"] if res["wall_time"] > 0 else 0.0,
                    "input_rps": len(inputs) / res["wall_time"] if res["wall_time"] > 0 else 0.0,
                    "e2e_mean_s": s["mean"],
                    "e2e_p50_s": s["p50"],
                    "e2e_p95_s": s["p95"],
                    "e2e_p99_s": s["p99"],
                }
            )
        sweep_df = pd.DataFrame(rows)

        if self.save_results:
            paths = self._output_paths(project_tag)
            out = paths["summary_csv"].replace("_summary.csv", "_sweep.csv")
            sweep_df.to_csv(out, index=False)
            logger.info(f"Saved sweep: {out}")
        print(sweep_df.to_string(index=False))
        return sweep_df

    @staticmethod
    def _summary_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
        """Per-adapter and end-to-end summary as a tidy DataFrame."""
        records = results["records"]
        adapters = results["adapters"]
        wall_time = results["wall_time"]
        ok = [r for r in records if r["success"]]

        rows = []
        for adapter in adapters:
            lats = [r["latency"] for r in ok if r["adapter"] == adapter]
            s = summarize(lats)
            rows.append(
                {
                    "scope": "per_adapter",
                    "name": adapter,
                    "n": s["n"],
                    "rps": s["n"] / wall_time if wall_time > 0 else 0.0,
                    "mean_s": s["mean"],
                    "p50_s": s["p50"],
                    "p95_s": s["p95"],
                    "p99_s": s["p99"],
                    "max_s": s["max"],
                }
            )
        s = summarize(results["end_to_end"])
        rows.append(
            {
                "scope": "end_to_end",
                "name": "all_adapters",
                "n": s["n"],
                "rps": s["n"] / wall_time if wall_time > 0 else 0.0,
                "mean_s": s["mean"],
                "p50_s": s["p50"],
                "p95_s": s["p95"],
                "p99_s": s["p99"],
                "max_s": s["max"],
            }
        )
        return pd.DataFrame(rows)


def benchmark_multilora(
    df: pd.DataFrame,
    project_tag: str,
    base_runner: VLM_Runner,
    adapters: List[str],
    client: AsyncOpenAI,
    exp_name: str,
    concurrency: int = 64,
    max_tokens: int = 1,
    work_dir: str = ".",
    save_results: bool = True,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Thin functional wrapper mirroring `get_openai_metrics` from vlm_inference."""
    runner = MultiLoRABenchmarkRunner(
        base_runner=base_runner,
        adapters=adapters,
        exp_name=exp_name,
        work_dir=work_dir,
        save_results=save_results,
    )
    return runner.run(
        df=df,
        project_tag=project_tag,
        client=client,
        concurrency=concurrency,
        max_tokens=max_tokens,
        limit=limit,
    )
