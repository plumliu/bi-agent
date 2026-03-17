#!/usr/bin/env python3
"""Extract successful results to per-sample JSON files."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    input_path = project_root / "experiment/insightbench_full/results/full_custom_responses_20260317_005705.jsonl"
    output_dir = project_root / "experiment/insightbench_full/result5"
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("success"):
                continue

            sample_id = record.get("sample_id")
            if sample_id is None:
                continue

            payload = {
                "insights": record.get("insights", []),
                "summary": record.get("summary", ""),
            }

            output_path = output_dir / f"flag-{sample_id}.json"
            with output_path.open("w", encoding="utf-8") as out:
                json.dump(payload, out, ensure_ascii=False, indent=2)
            success_count += 1

    print(f"Saved {success_count} files to {output_dir}")


if __name__ == "__main__":
    main()
