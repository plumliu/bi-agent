#!/usr/bin/env python3
"""
Test script to verify the workflow on 2-3 samples
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add project root and da-bench src to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiment/modeling_custom/da-bench/src"))
sys.path.insert(0, str(project_root / "experiment/insightbench_full"))

from src.utils import load_all_samples
from src.runner import run_single_sample
from sandbox_manager import SandboxPool


async def main():
    """Test on 2-3 samples"""
    print("="*80)
    print("Testing InsightBench Full Custom Workflow (2-3 samples)")
    print("="*80)

    # Load samples
    data_dir = project_root / "experiment/insightbench_full/data"
    print(f"\nLoading samples from: {data_dir}")
    all_samples = load_all_samples(str(data_dir))
    print(f"Total samples available: {len(all_samples)}")

    # Take first 2 samples
    test_samples = all_samples[:2]
    print(f"Testing with {len(test_samples)} samples: {[s['sample_id'] for s in test_samples]}")

    # Initialize pool
    pool = SandboxPool(max_concurrent=2)

    # Run tasks
    print("\nRunning workflows...")
    tasks = [
        pool.run_with_sandbox(run_single_sample, sample)
        for sample in test_samples
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Display results
    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"\nSample {test_samples[i]['sample_id']}: FAILED")
            print(f"  Error: {type(result).__name__}: {str(result)}")
        else:
            print(f"\nSample {result['sample_id']}: {'SUCCESS' if result.get('success') else 'FAILED'}")
            print(f"  Goal: {result['goal'][:80]}...")
            print(f"  Insights count: {len(result.get('insights', []))}")
            print(f"  Summary length: {len(result.get('summary', ''))} chars")
            if result.get('error'):
                print(f"  Error: {result['error']}")
            if result.get('parse_error'):
                print(f"  Parse error: {result['parse_error']}")

    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(main())
