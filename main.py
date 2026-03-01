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
from app.utils.extract_text_from_content import extract_text_from_content
from app.utils.file_parser import parse_file_content

from langchain_core.messages import SystemMessage, HumanMessage

# ==========================================
# 0. 环境路径配置
# ==========================================
# 确保能导入 app 模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ppio_sandbox.code_interpreter import Sandbox, SandboxQuery, SandboxState
from app.core.state import AgentState
from app.graph.workflow import build_graph
from app.core.config import settings
from app.tools.python_interpreter import create_code_interpreter_tool

# ==========================================
# 1. Config & Setup
# ==========================================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 【启动阶段】: 这里可以放置应用启动时需要执行的代码
    print("[Launching] 启动BI agent中...")
    yield
    # 【关闭阶段】: 当 Ctrl + C 终止时执行以下逻辑
    print("\n[Shutdown] 正在检测并关闭所有运行中的沙箱...")
    try:
        # 1. 查找所有状态为 RUNNING 的沙箱
        query = SandboxQuery(state=[SandboxState.RUNNING])
        paginator = Sandbox.list(query=query)
        running_sandboxes = paginator.next_items()

        if not running_sandboxes:
            print("[Shutdown] 未发现运行中的沙箱。")
        else:
            print(f"[Shutdown] 发现 {len(running_sandboxes)} 个运行中的沙箱，准备关闭...")
            for sb_info in running_sandboxes:
                try:
                    # 2. 连接并杀死沙箱
                    sb = Sandbox.connect(sb_info.sandbox_id)
                    sb.kill()
                    print(f"成功关闭沙箱 ID: {sb_info.sandbox_id}")
                except Exception as e:
                    print(f"关闭沙箱 {sb_info.sandbox_id} 失败: {e}")
            print("[Shutdown] 所有沙箱清理完毕。")
    except Exception as e:
        print(f"[Shutdown] 获取沙箱列表失败: {e}")

app = FastAPI(title="BI Agent API", version="1.0.0", lifespan=lifespan)

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

        # --- C. Workflow Init ---
        # Pass the specific sandbox instance to tools
        code_tool = create_code_interpreter_tool(sandbox)
        workflow_app = build_graph(tools=[code_tool], sandbox=sandbox)

        # Initialize State
        initial_state = AgentState(
            messages=[HumanMessage(content=user_query)],
            user_input=user_query,
            raw_file_paths=raw_file_paths_in_sandbox,
            original_filenames=original_filenames,
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

        # 返回结构为 (namespace, mode, data)，直接解包即可
        async for namespace, mode, payload in workflow_app.astream(
                initial_state,
                stream_mode=["messages", "values"],
                subgraphs=True  # 开启子图事件穿透
        ):
            # namespace: 元组类型，标识事件来源（主图或子图路径），如需调试可打印
            # print(f"当前作用域：{namespace}")

            # payload 对应文档中的 data，原有业务逻辑保持不变
            # ==========================================
            # 模态 1：实时捕获大模型的“打字机”输出
            # ==========================================
            if mode == "messages":
                chunk, metadata = payload

                # 【核心拦截】：获取当前正在发声的节点名称，屏蔽后台节点
                current_node = metadata.get("langgraph_node", "")
                if current_node == "viz":
                    continue

                # 只拦截 AI 生成的增量消息
                if chunk.type == "AIMessageChunk" and chunk.content:

                    # 直接使用提炼好的通用解析工具！
                    content_to_yield = extract_text_from_content(chunk.content)

                    # 推送 log_stream 事件给前端
                    if content_to_yield:
                        yield format_sse("log_stream", content_to_yield)

            # ==========================================
            # 模态 2：捕获节点执行完毕后的全局状态更新
            # ==========================================
            elif mode == "values":
                event = payload

                # 捕获 ETL 状态
                if not etl_completed_flag and event.get("data_schema"):
                    # 这里的 log 事件前端可以新起一行显示
                    yield format_sse("log", "数据自动合并完成，Schema 已生成。")
                    etl_completed_flag = True

                # 捕获日志 (此时专注于拦截沙盒的执行结果)
                if "messages" in event and event["messages"]:
                    last_msg = event["messages"][-1]
                    current_id = id(last_msg)

                    if current_id != last_msg_id:
                        # 场景 1：沙盒工具执行完毕，输出的 STDOUT (Pandas 表格等)
                        if last_msg.type == "tool":
                            content = str(last_msg.content)
                            yield format_sse("tool_log", content)

                        last_msg_id = current_id

                # 捕获 Viz 结果 (已清理原代码中的冗余重复块)
                if event.get("viz_success") is True:
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

                # 捕获全局总结
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