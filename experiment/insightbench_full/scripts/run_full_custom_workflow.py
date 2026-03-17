#!/usr/bin/env python3
"""
InsightBench Full Custom Workflow Experiment Runner

Runs the complete custom workflow (profiler → router → modeling_custom → summary)
on all InsightBench samples using bounded local parallel execution.
"""
from __future__ import annotations

import asyncio
import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiment/insightbench_full"))

from src.utils import load_all_samples, write_jsonl
from src.runner import run_single_sample


def _default_max_concurrent() -> int:
    # Local kernels are memory-heavy; keep a safe default.
    cpu = os.cpu_count() or 8
    return max(4, min(12, cpu // 2))


async def main(max_concurrent: int):
    """Main entry point for the experiment"""
    print("=" * 80)
    print("InsightBench Full Custom Workflow Experiment")
    print("=" * 80)

    # 1. Load all samples
    data_dir = project_root / "experiment/insightbench_full/data"
    print(f"\n[1/4] Loading samples from: {data_dir}")
    all_samples = load_all_samples(str(data_dir))
    samples = all_samples
    print(f"      Loaded {len(samples)} samples, running all samples")

    # 2. Initialize local concurrency and output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = project_root / f"experiment/insightbench_full/results/full_custom_responses_{timestamp}.jsonl"

    print(f"\n[2/4] Initializing local worker pool (max_concurrent={max_concurrent})")
    print(f"      Output file: {output_file.name}")

    # 3. Run all tasks with bounded local concurrency
    print(f"\n[3/4] Running {len(samples)} samples in bounded parallel mode...")
    print("      Note: this runs all 100 samples, but only max_concurrent samples at once.")

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_with_limit(sample):
        async with semaphore:
            return await run_single_sample(sample)

    tasks = [asyncio.create_task(_run_with_limit(sample)) for sample in samples]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 4. Process results and write JSONL
    print(f"\n[4/4] Processing results...")

    processed_results = []
    success_count = 0
    error_count = 0

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Handle exceptions from asyncio.gather
            processed_results.append({
                "sample_id": samples[i]["sample_id"],
                "goal": samples[i]["goal"],
                "user_input": samples[i]["user_input"],
                "insights": [],
                "summary": "",
                "final_summary_raw": "",
                "modeling_summary": None,
                "success": False,
                "error": f"{type(result).__name__}: {str(result)}",
            })
            error_count += 1
        else:
            processed_results.append(result)
            if result.get("success"):
                success_count += 1
            else:
                error_count += 1

    write_jsonl(str(output_file), processed_results)

    # 5. Summary
    print(f"\n{'=' * 80}")
    print(f"Experiment Complete!")
    print(f"{'=' * 80}")
    print(f"Total samples:    {len(samples)}")
    print(f"Successful:       {success_count}")
    print(f"Failed:           {error_count}")
    print(f"Output file:      {output_file}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Run full InsightBench custom workflow on local runtime.")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=_default_max_concurrent(),
        help="Maximum number of samples to run in parallel (default: safe local value).",
    )
    args = parser.parse_args()

    if args.max_concurrent < 1:
        raise SystemExit("--max-concurrent must be >= 1")

    asyncio.run(main(max_concurrent=args.max_concurrent))
