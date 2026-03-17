"""Workflow runner for single sample execution (local runtime only)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from app.core.config import settings
from app.core.state import WorkflowState
from app.graph.workflow import build_graph
from app.tools.local_kernel_runtime import LocalKernelRuntime

DEFAULT_WORKSPACE_SESSIONS_DIR = "/Users/plumliu/Desktop/python_workspace/agent_workspace/sessions"
DEFAULT_WORKSPACE_PYTHON = "/Users/plumliu/Desktop/python_workspace/agent_workspace/.venv/bin/python"


def _workspace_sessions_dir() -> Path:
    configured = settings.AGENT_WORKSPACE_SESSIONS_DIR or DEFAULT_WORKSPACE_SESSIONS_DIR
    return Path(configured)


def _workspace_python() -> str:
    configured = settings.AGENT_WORKSPACE_PYTHON or DEFAULT_WORKSPACE_PYTHON
    return configured


def _ensure_runtime_ready() -> str:
    python_exec = _workspace_python()
    python_path = Path(python_exec)
    if not python_path.exists():
        raise RuntimeError(
            f"Workspace python not found: {python_exec}. "
            "Please create/activate /agent_workspace/.venv first."
        )

    check = subprocess.run(
        [python_exec, "-c", "import ipykernel"],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        raise RuntimeError(
            "ipykernel is missing in workspace python. "
            f"Python: {python_exec}. stderr: {check.stderr.strip()}"
        )
    return python_exec


def _extract_summary_payload(final_summary_raw: str) -> tuple[list[str], str, str | None]:
    insights: list[str] = []
    summary = ""
    parse_error = None

    if not final_summary_raw:
        return insights, summary, parse_error

    payload = final_summary_raw.strip()
    if payload.startswith("```json"):
        payload = payload[7:].strip()
    if payload.startswith("```"):
        payload = payload[3:].strip()
    if payload.endswith("```"):
        payload = payload[:-3].strip()

    try:
        summary_json = json.loads(payload)
        insights = summary_json.get("insights", [])
        summary = summary_json.get("summary", "")
    except json.JSONDecodeError as e:
        parse_error = str(e)
        summary = final_summary_raw

    return insights, summary, parse_error


async def run_single_sample(sample_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run full custom workflow for a single sample on local runtime."""
    sample_id = sample_data["sample_id"]
    print(f"\n{'=' * 60}")
    print(f"[Sample {sample_id}] Starting workflow")
    print(f"{'=' * 60}")

    runtime: LocalKernelRuntime | None = None
    session_dir: Path | None = None

    try:
        python_exec = _ensure_runtime_ready()

        session_id = f"insightbench_{sample_id}_{uuid.uuid4().hex[:8]}"
        session_dir = _workspace_sessions_dir() / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Sample {sample_id}] Session dir: {session_dir}")

        csv_files = sample_data["csv_files"]
        raw_file_paths: list[str] = []
        for idx, csv_path in enumerate(csv_files):
            dst = session_dir / f"raw_{idx}.csv"
            shutil.copy2(csv_path, dst)
            raw_file_paths.append(str(dst))
            print(f"[Sample {sample_id}] Copied: {Path(csv_path).name} -> {dst.name}")

        initial_state: WorkflowState = {
            "user_input": sample_data["user_input"],
            "raw_file_paths": raw_file_paths,
            "original_filenames": sample_data["original_filenames"],
            "local_file_paths": raw_file_paths,
            "files_metadata": [],
            "merge_recommendations": None,
            "scenario": "custom",
            "reasoning": None,
            "modeling_summary": None,
            "final_summary": None,
            "error_count": 0,
        }

        runtime = LocalKernelRuntime(
            session_dir=str(session_dir),
            python_executable=python_exec,
        )
        runtime.start()

        workflow_app = build_graph(
            runtime=runtime,
            session_dir=str(session_dir),
            session_id=session_id,
        )
        print(f"[Sample {sample_id}] Invoking workflow...")

        final_state = await workflow_app.ainvoke(initial_state)
        print(f"[Sample {sample_id}] Workflow completed")

        final_summary_raw = final_state.get("final_summary", "") or ""
        insights, summary, parse_error = _extract_summary_payload(final_summary_raw)
        if parse_error:
            print(f"[Sample {sample_id}] Warning: Failed to parse final_summary as JSON: {parse_error}")

        result = {
            "sample_id": sample_id,
            "goal": sample_data["goal"],
            "user_input": sample_data["user_input"],
            "insights": insights,
            "summary": summary,
            "final_summary_raw": final_summary_raw,
            "modeling_summary": final_state.get("modeling_summary"),
            "success": True,
            "error": None,
            "parse_error": parse_error,
            "session_dir": str(session_dir),
        }

        print(f"[Sample {sample_id}] ✓ Success")
        return result

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        traceback_str = traceback.format_exc()

        print(f"[Sample {sample_id}] ✗ Error: {error_msg}")
        print(traceback_str)

        return {
            "sample_id": sample_id,
            "goal": sample_data.get("goal", ""),
            "user_input": sample_data.get("user_input", ""),
            "insights": [],
            "summary": "",
            "final_summary_raw": "",
            "modeling_summary": None,
            "success": False,
            "error": error_msg,
            "traceback": traceback_str,
            "session_dir": str(session_dir) if session_dir else None,
        }
    finally:
        if runtime is not None:
            runtime.shutdown()
