import io
import os
import sys
import json
import shutil
import uuid

import chardet

import pandas as pd
import uvicorn
import asyncio
from typing import Any, List, Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from app.utils.alias_generator import generate_semantic_alias
from app.utils.clean_log_content import clean_log_content
from app.utils.file_parser import parse_file_content

# ==========================================
# 0. 环境路径配置
# ==========================================
# 确保能导入 app 模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from e2b_code_interpreter import Sandbox
from ppio_sandbox.code_interpreter import Sandbox
from app.utils.csv_reader import get_csv_schema
from app.core.state import AgentState
from app.graph.workflow import build_graph
from app.core.config import settings
from app.tools.sandbox import create_code_interpreter_tool

# ==========================================
# 1. Config & Setup (No Globals)
# ==========================================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="BI Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


# ==========================================
# 2. Helper Functions
# ==========================================
def format_sse(event_type: str, data: Any = None) -> str:
    payload = {"type": event_type, "data": data}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ==========================================
# 3. Core Logic: Request-Scoped Generator
# ==========================================

async def run_workflow_stream(
        user_query: str,
        scenario: str,
        upload_queue: List[Dict[str, str]],  # List of {local_path, remote_path}
        original_filenames: List[str],
        session_id: str
):
    """
    Lifecycle-bound Generator:
    1. Creates Sandbox
    2. Uploads Files
    3. Runs Agent
    4. Destroys Sandbox (Finally block)
    """
    sandbox = None
    # Use a session-specific path for viz data to support concurrency
    local_viz_path = os.path.join(TEMP_DIR, f"viz_{session_id}.json")

    try:
        # --- A. Environment Init ---
        yield format_sse("log", "正在动态创建 PPIO 沙盒环境...")

        # [Lifecycle Start] Create Sandbox
        sandbox = Sandbox.create(
            settings.PPIO_TEMPLATE,
            api_key=settings.PPIO_API_KEY,
            timeout=3600
        )
        yield format_sse("log", f"沙盒创建成功")

        # --- B. File Upload (Consume Queue) ---
        yield format_sse("log", "正在同步数据到隔离环境...")

        raw_file_paths_in_sandbox = []
        for item in upload_queue:
            local_p = item["local_path"]
            remote_p = item["remote_path"]

            # Upload to the specific sandbox instance
            with open(local_p, "rb") as f:
                sandbox.files.write(remote_p, f)

            raw_file_paths_in_sandbox.append(remote_p)

            # Optional: Clean up local CSV fragment immediately to save space?
            # os.remove(local_p)

        # --- C. Workflow Init ---
        # Pass the specific sandbox instance to tools
        code_tool = create_code_interpreter_tool(sandbox)
        workflow_app = build_graph(tools=[code_tool], sandbox=sandbox)

        # Initialize State
        initial_state = AgentState(
            messages=[],
            user_input=user_query,
            raw_file_paths=raw_file_paths_in_sandbox,
            original_filenames=original_filenames,
            # Pass session specific paths if your nodes support it
            # viz_file_path=local_viz_path,
            remote_file_path=None,
            data_schema={},
            scenario=scenario,
            modeling_artifacts=None,
            viz_config=None,
            viz_success=False,
            final_summary=None,
            error_count=0
        )

        # --- D. Execution Loop ---
        yield format_sse("log", "智能体工作流启动...")

        etl_completed_flag = False
        last_msg_id = None
        final_viz_data_cache = []
        final_summary_cache = ""

        async for event in workflow_app.astream(initial_state, stream_mode="values"):

            # [Logic identical to before, just ensured context safety]
            if not etl_completed_flag and event.get("data_schema"):
                yield format_sse("log", "数据自动合并完成，Schema 已生成。")
                etl_completed_flag = True

            # Capture Logs
            if "messages" in event and event["messages"]:
                last_msg = event["messages"][-1]
                current_id = id(last_msg)
                if current_id != last_msg_id:
                    content = str(last_msg.content)
                    if not content.strip():
                        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            yield format_sse("log", "正在构建代码指令...")
                    else:
                        if len(content) < 2000:
                            yield format_sse("log", clean_log_content(content))
                        else:
                            yield format_sse("log", "正在生成复杂配置...")
                    last_msg_id = current_id

            # Capture Viz
            # Note: You should update your Viz Node to write to `local_viz_path`
            # OR default to looking for the file generated by this specific workflow
            if event.get("viz_success") is True:
                # Check specific session path first, fallback to generic (race condition risk)
                target_viz_path = local_viz_path if os.path.exists(local_viz_path) else os.path.join(TEMP_DIR,
                                                                                                     "viz_data.json")

                if os.path.exists(target_viz_path):
                    try:
                        with open(target_viz_path, "r", encoding="utf-8") as f:
                            data_content = json.load(f)
                            if "echarts" in data_content:
                                final_viz_data_cache = data_content["echarts"]
                                yield format_sse("viz_data", final_viz_data_cache)
                    except Exception as e:
                        print(f"Viz read error: {e}")

            # Capture Summary
            if "final_summary" in event and event["final_summary"]:
                final_summary_cache = event["final_summary"]
                yield format_sse("summary", final_summary_cache)

        # Done
        full_result = {
            "success": True,
            "message": final_summary_cache,
            "echarts": final_viz_data_cache
        }
        yield format_sse("done", full_result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield format_sse("error", f"工作流异常: {str(e)}")

    finally:
        # --- E. Cleanup (Crucial) ---
        yield format_sse("log", "正在清理计算资源...")

        # 1. Kill Sandbox
        if sandbox:
            try:
                sandbox.kill()
                print(f"--- [Session {session_id}] Sandbox killed. ---")
            except Exception as e:
                print(f"Error killing sandbox: {e}")

        # 2. Cleanup Local Temp Files for this session
        # (This assumes you tracked which local files belong to this session)
        for item in upload_queue:
            if os.path.exists(item["local_path"]):
                try:
                    os.remove(item["local_path"])
                except:
                    pass
        if os.path.exists(local_viz_path):
            try:
                os.remove(local_viz_path)
            except:
                pass


# ==========================================
# 4. Stream Endpoint
# ==========================================

@app.post("/query_agents_stream")
async def query_agents_stream(
        file: List[UploadFile] = File(...),
        request_data: str = Form(...)
):
    """
    1. Parse Params
    2. Local ETL (Parse to CSV)
    3. Handover to Stream Generator
    """
    # 1. Session ID for concurrency safety
    session_id = str(uuid.uuid4())[:8]
    print(f"--- [Req {session_id}] New Request ---")

    try:
        data = json.loads(request_data)
        user_query = data.get("query")
        scenario = data.get("scenario", None)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # 2. Prepare Upload Queue (Do not upload yet, just prep)
    upload_queue = []  # [{"local_path":..., "remote_path":...}]
    original_filenames = []

    try:
        fragment_count = 0
        for file_index, f in enumerate(file):
            content = await f.read()
            base_alias = generate_semantic_alias(f.filename, file_index)

            # Parse (CPU Bound - OK to do here)
            extracted_items = parse_file_content(content, f.filename)

            for sheet_name, df in extracted_items:
                # Save to local temp with session ID to avoid collisions
                local_filename = f"raw_{session_id}_{fragment_count}.csv"
                local_path = os.path.join(TEMP_DIR, local_filename)
                df.to_csv(local_path, index=False)

                remote_path = f"/home/user/raw_{fragment_count}.csv"

                # Semantic Name
                if sheet_name == "CSV_Content":
                    final_name = base_alias
                else:
                    final_name = f"{base_alias} (Sheet: {sheet_name})"

                upload_queue.append({
                    "local_path": local_path,
                    "remote_path": remote_path
                })
                original_filenames.append(final_name)
                fragment_count += 1

        if not upload_queue:
            raise HTTPException(status_code=400, detail="No valid data parsed")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {e}")

    # 3. Delegate to Stream
    # We pass the 'upload_queue' so the generator can handle the Sandbox interaction
    return StreamingResponse(
        run_workflow_stream(
            user_query,
            scenario,
            upload_queue,
            original_filenames,
            session_id
        ),
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8009, reload=True)