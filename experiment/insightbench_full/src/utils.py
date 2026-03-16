from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_all_samples(data_dir: str) -> List[Dict[str, Any]]:
    data_path = Path(data_dir)
    json_files = sorted(data_path.glob("flag-*.json"))
    samples: List[Dict[str, Any]] = []

    for json_path in json_files:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        metadata = payload.get("metadata", {})
        goal = metadata.get("goal", "").strip()
        user_input = f"**Goal**: {goal}" if goal else "**Goal**:"

        stem = json_path.stem  # e.g. flag-14
        # Match exact file (flag-14.csv) and supplementary files (flag-14-*.csv)
        csv_files = sorted(
            list(data_path.glob(f"{stem}.csv")) +
            list(data_path.glob(f"{stem}-*.csv"))
        )
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found for {stem}")

        samples.append(
            {
                "sample_id": stem.replace("flag-", ""),
                "goal": goal,
                "user_input": user_input,
                "json_path": str(json_path),
                "csv_files": [str(p) for p in csv_files],
                "original_filenames": [p.name for p in csv_files],
            }
        )

    return samples


def write_jsonl(output_path: str, results: List[Dict[str, Any]]) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
