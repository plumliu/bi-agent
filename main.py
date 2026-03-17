import json
import os
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Ensure app package import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.config import settings
from app.core.state import WorkflowState
from app.graph.workflow import build_graph
from app.tools.local_kernel_runtime import LocalKernelRuntime
from app.utils.alias_generator import generate_semantic_alias
from app.utils.extract_text_from_content import extract_text_from_content
from app.utils.file_parser import parse_file_content


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Launching] BI agent (modeling-only) starting...")
    yield
    print("[Shutdown] BI agent stopped.")


app = FastAPI(title="BI Agent API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_sse(event_type: str, data: Any = None) -> str:
    payload = {"type": event_type, "data": data}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _ensure_workspace_runtime_ready() -> None:
    python_exec = settings.AGENT_WORKSPACE_PYTHON
    if not os.path.exists(python_exec):
        raise RuntimeError(
            f"Workspace python not found: {python_exec}. "
            "Please create/activate /agent_workspace/.venv first."
        )

    check_cmd = [python_exec, "-c", "import ipykernel"]
    result = subprocess.run(check_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ipykernel is missing in workspace python. "
            f"Python: {python_exec}. stderr: {result.stderr.strip()}"
        )


def _get_session_dir(session_id: str) -> str:
    sessions_root = settings.AGENT_WORKSPACE_SESSIONS_DIR
    os.makedirs(sessions_root, exist_ok=True)
    session_dir = os.path.join(sessions_root, session_id)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


async def run_workflow_stream(
    user_query: str,
    scenario: Optional[str],
    upload_queue: List[Dict[str, str]],
    original_filenames: List[str],
    session_id: str,
    session_dir: str,
):
    runtime = None

    try:
        yield format_sse("log", "初始化本地建模运行环境...")
        _ensure_workspace_runtime_ready()

        runtime = LocalKernelRuntime(
            session_dir=session_dir,
            python_executable=settings.AGENT_WORKSPACE_PYTHON,
        )
        runtime.start()

        yield format_sse("log", f"本地会话内核已启动: {session_dir}")

        workflow_app = build_graph(runtime=runtime, session_dir=session_dir, session_id=session_id)

        raw_file_paths = [item["path"] for item in upload_queue]
        initial_state = WorkflowState(
            user_input=user_query,
            raw_file_paths=raw_file_paths,
            original_filenames=original_filenames,
            local_file_paths=raw_file_paths,
            files_metadata=[],
            merge_recommendations=None,
            scenario=scenario,
            reasoning=None,
            modeling_summary=None,
            final_summary=None,
            error_count=0,
        )

        yield format_sse("log", "智能体工作流启动...")

        profiler_done = False
        final_summary_cache = ""
        last_execution_signature = None

        async for namespace, mode, payload in workflow_app.astream(
            initial_state,
            stream_mode=["messages", "values"],
            subgraphs=True,
        ):
            _ = namespace

            if mode == "messages":
                chunk, metadata = payload
                _ = metadata

                if chunk.type == "AIMessageChunk":
                    if chunk.content:
                        content = extract_text_from_content(chunk.content)
                        if content:
                            yield format_sse("log_stream", content)

                    if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                        for tool_call in chunk.tool_calls:
                            tool_name = tool_call.get("name", "") or "unknown_tool"
                            if tool_name and tool_name != "unknown_tool":
                                yield format_sse(
                                    "tool_calling",
                                    {
                                        "tool_name": tool_name,
                                        "message": f"正在调用工具: {tool_name}",
                                    },
                                )

            elif mode == "values":
                event = payload

                if not profiler_done and event.get("files_metadata"):
                    profiler_done = True
                    yield format_sse("log", "文件元信息收集完成。")

                latest_execution = event.get("latest_execution")
                if latest_execution:
                    signature = (
                        latest_execution.get("code", ""),
                        latest_execution.get("stdout", ""),
                        latest_execution.get("stderr", ""),
                        json.dumps(latest_execution.get("error"), ensure_ascii=False),
                    )
                    if signature != last_execution_signature:
                        last_execution_signature = signature
                        tool_lines = []
                        if latest_execution.get("stdout"):
                            tool_lines.append(str(latest_execution["stdout"]))
                        if latest_execution.get("result_text"):
                            tool_lines.append(str(latest_execution["result_text"]))
                        if latest_execution.get("stderr"):
                            tool_lines.append("[STDERR]\n" + str(latest_execution["stderr"]))
                        if latest_execution.get("error"):
                            tool_lines.append("[ERROR]\n" + json.dumps(latest_execution["error"], ensure_ascii=False))
                        if tool_lines:
                            yield format_sse("tool_log", "\n".join(tool_lines))

                if event.get("final_summary"):
                    final_summary_cache = event["final_summary"]
                    yield format_sse("summary", final_summary_cache)

        yield format_sse(
            "done",
            {
                "success": True,
                "message": final_summary_cache,
                "echarts": [],
            },
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        yield format_sse("error", f"工作流异常: {str(e)}")

    finally:
        if runtime is not None:
            runtime.shutdown()
            yield format_sse("log", "本地会话内核已关闭。")


@app.post("/query_agents_stream")
async def query_agents_stream(
    file: List[UploadFile] = File(...),
    request_data: str = Form(...),
):
    session_id = str(uuid.uuid4())[:8]
    session_dir = _get_session_dir(session_id)
    print(f"--- [Req {session_id}] New Request @ {session_dir} ---")

    try:
        data = json.loads(request_data)
        user_query = data.get("query")
        scenario = data.get("scenario", None)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if not isinstance(user_query, str):
        raise HTTPException(status_code=400, detail="query must be a string")

    upload_queue: List[Dict[str, str]] = []
    original_filenames: List[str] = []

    try:
        fragment_count = 0
        for file_index, f in enumerate(file):
            content = await f.read()
            base_alias = generate_semantic_alias(f.filename, file_index)

            extracted_items = parse_file_content(content, f.filename)

            for sheet_name, df in extracted_items:
                raw_filename = f"raw_{fragment_count}.csv"
                raw_path = os.path.join(session_dir, raw_filename)
                df.to_csv(raw_path, index=False)

                if sheet_name == "CSV_Content":
                    final_name = base_alias
                else:
                    final_name = f"{base_alias} (Sheet: {sheet_name})"

                upload_queue.append({"path": raw_path})
                original_filenames.append(final_name)
                fragment_count += 1

        if not upload_queue:
            raise HTTPException(status_code=400, detail="No valid data parsed")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {e}")

    return StreamingResponse(
        run_workflow_stream(
            user_query=user_query,
            scenario=scenario,
            upload_queue=upload_queue,
            original_filenames=original_filenames,
            session_id=session_id,
            session_dir=session_dir,
        ),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8009, reload=True)
