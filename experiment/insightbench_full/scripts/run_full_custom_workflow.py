#!/usr/bin/env python3
"""
InsightBench Full Custom Workflow Experiment Runner

Runs the complete custom workflow (profiler → router → modeling_custom → viz_custom → summary)
on 100 InsightBench samples using parallel sandbox execution.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add project root and da-bench src to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiment/modeling_custom/da-bench/src"))
sys.path.insert(0, str(project_root / "experiment/insightbench_full"))

from src.utils import load_all_samples, write_jsonl
from src.runner import run_single_sample
from sandbox_manager import SandboxPool


async def main():
    """Main entry point for the experiment"""
    print("="*80)
    print("InsightBench Full Custom Workflow Experiment")
    print("="*80)

    # 1. Load all samples
    data_dir = project_root / "experiment/insightbench_full/data"
    print(f"\n[1/4] Loading samples from: {data_dir}")
    all_samples = load_all_samples(str(data_dir))
    samples = all_samples[:42]  # Only run first 42 samples
    print(f"      Loaded {len(all_samples)} samples, running first {len(samples)}")

    # 2. Initialize sandbox pool and output file
    max_concurrent = 45
    pool = SandboxPool(max_concurrent=max_concurrent)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = project_root / f"experiment/insightbench_full/results/full_custom_responses_{timestamp}.jsonl"

    print(f"\n[2/4] Initializing SandboxPool (max_concurrent={max_concurrent})")
    print(f"      Output file: {output_file.name}")

    # 3. Run all tasks concurrently
    print(f"\n[3/4] Running {len(samples)} samples concurrently...")
    print(f"      This may take 15-25 minutes...")

    tasks = [
        pool.run_with_sandbox(run_single_sample, sample)
        for sample in samples
    ]

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
                "viz_data": None,
                "viz_config": None,
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
    print(f"\n{'='*80}")
    print(f"Experiment Complete!")
    print(f"{'='*80}")
    print(f"Total samples:    {len(samples)}")
    print(f"Successful:       {success_count}")
    print(f"Failed:           {error_count}")
    print(f"Output file:      {output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())
