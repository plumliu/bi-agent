"""Workflow runner for single sample execution"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from ppio_sandbox.code_interpreter import Sandbox
from app.core.state import WorkflowState
from app.graph.workflow import build_graph


async def run_single_sample(sandbox: Sandbox, sample_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run full custom workflow for a single sample

    Args:
        sandbox: PPIO sandbox instance
        sample_data: Sample metadata from utils.load_all_samples

    Returns:
        Result dict with insights, summary, and metadata
    """
    sample_id = sample_data["sample_id"]
    print(f"\n{'='*60}")
    print(f"[Sample {sample_id}] Starting workflow")
    print(f"{'='*60}")

    try:
        # 1. Upload CSV files to sandbox
        csv_files = sample_data["csv_files"]
        raw_file_paths = []

        for idx, csv_path in enumerate(csv_files):
            remote_path = f"/home/user/raw_{idx}.csv"
            with open(csv_path, "rb") as f:
                sandbox.files.write(remote_path, f)
            raw_file_paths.append(remote_path)
            print(f"[Sample {sample_id}] Uploaded: {Path(csv_path).name} -> {remote_path}")

        # 2. Build initial state (matching main.py:144-159)
        initial_state: WorkflowState = {
            "user_input": sample_data["user_input"],
            "raw_file_paths": raw_file_paths,
            "original_filenames": sample_data["original_filenames"],
            "local_file_paths": csv_files,
            "files_metadata": [],
            "merge_recommendations": None,
            "remote_file_path": None,
            "data_schema": {},
            "scenario": "custom",
            "reasoning": None,
            "modeling_artifacts": None,
            "modeling_summary": None,
            "generated_data_files": None,
            "file_metadata": None,
            "viz_config": None,
            "viz_success": False,
            "viz_data": None,
            "final_summary": None,
            "error_count": 0,
        }

        print(f"[Sample {sample_id}] Built initial state")

        # 3. Build and invoke workflow
        workflow_app = build_graph(sandbox=sandbox)
        print(f"[Sample {sample_id}] Invoking workflow...")

        final_state = await workflow_app.ainvoke(initial_state)
        print(f"[Sample {sample_id}] Workflow completed")

        # 4. Extract and parse results
        final_summary_raw = final_state.get("final_summary", "")

        # Parse JSON from summary_custom.yaml output
        insights = []
        summary = ""
        parse_error = None

        if final_summary_raw:
            try:
                summary_json = json.loads(final_summary_raw)
                insights = summary_json.get("insights", [])
                summary = summary_json.get("summary", "")
            except json.JSONDecodeError as e:
                parse_error = str(e)
                summary = final_summary_raw
                print(f"[Sample {sample_id}] Warning: Failed to parse final_summary as JSON: {e}")

        result = {
            "sample_id": sample_id,
            "goal": sample_data["goal"],
            "user_input": sample_data["user_input"],
            "insights": insights,
            "summary": summary,
            "final_summary_raw": final_summary_raw,
            "modeling_summary": final_state.get("modeling_summary"),
            "viz_data": final_state.get("viz_data"),
            "viz_config": final_state.get("viz_config"),
            "success": True,
            "error": None,
            "parse_error": parse_error,
        }

        print(f"[Sample {sample_id}] ✓ Success")
        return result

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        traceback_str = traceback.format_exc()

        print(f"[Sample {sample_id}] ✗ Error: {error_msg}")
        print(traceback_str)

        return {
            "sample_id": sample_id,
            "goal": sample_data["goal"],
            "user_input": sample_data["user_input"],
            "insights": [],
            "summary": "",
            "final_summary_raw": "",
            "modeling_summary": None,
            "viz_data": None,
            "viz_config": None,
            "success": False,
            "error": error_msg,
            "traceback": traceback_str,
        }
