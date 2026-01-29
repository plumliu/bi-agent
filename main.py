import io
import os
import sys
import json
import shutil
import chardet

import pandas as pd
import uvicorn
import asyncio
from typing import Any, List
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
# 1. 全局变量与生命周期管理
# ==========================================

# 全局沙箱实例
GLOBAL_SANDBOX: Sandbox = None
# 异步锁，防止并发请求同时操作沙箱导致数据错乱
SANDBOX_LOCK = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期管理器：
    1. 启动时：创建 PPIO 沙盒（已从 E2B 更换）。
    2. 运行时：保持沙盒存活。
    3. 关闭时：销毁沙盒。
    """
    global GLOBAL_SANDBOX
    print("--- [Lifespan] 创建沙盒环境中... ---")


    GLOBAL_SANDBOX = Sandbox.create(
        settings.PPIO_TEMPLATE,
        api_key=settings.PPIO_API_KEY,
        timeout=3600
    )

    print("--- [Lifespan] 沙盒创建成功！---")

    yield

    print("--- [Lifespan] 关闭沙盒中... ---")
    GLOBAL_SANDBOX.kill()
    print("--- [Lifespan] 沙盒已成功关闭！ ---")


# ==========================================
# 2. 初始化 FastAPI
# ==========================================
app = FastAPI(title="BI Agent API", version="1.0.0", lifespan=lifespan)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 确保临时目录存在
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)
LOCAL_CSV_PATH = os.path.join(TEMP_DIR, "temp_data.csv")
LOCAL_VIZ_DATA_PATH = os.path.join(TEMP_DIR, "viz_data.json")


# ==========================================
# 3. 辅助函数
# ==========================================

def format_sse(event_type: str, data: Any = None) -> str:
    """
    构造 SSE 格式的消息。
    格式: data: {"type": "...", "data": ...}\n\n
    """
    payload = {
        "type": event_type,
        "data": data
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ==========================================
# 4. 核心逻辑：流式生成器 (已适配 Auto-ETL)
# ==========================================

async def run_workflow_stream(
        user_query: str,
        scenario: str,
        raw_file_paths: List[str],  # [新参数] 沙盒内的物理路径列表
        original_filenames: List[str]  # [新参数] 给 LLM 看的语义化文件名列表
):
    """
    异步生成器，用于 SSE 流式输出。
    负责初始化 LangGraph 状态并流式传输执行结果。
    """
    if GLOBAL_SANDBOX is None:
        yield format_sse("error", "Sandbox not initialized")
        return

    # [关键] 获取锁，确保本流程独占沙箱
    # 注意：在单例模式下，虽然文件已经在 API 层上传了，但执行期间仍需加锁防止冲突
    async with SANDBOX_LOCK:
        try:
            yield format_sse("log", "正在初始化智能体工作流...")

            # 1. 构建图
            # 这里的 tools 需要包含代码解释器
            code_tool = create_code_interpreter_tool(GLOBAL_SANDBOX)
            workflow_app = build_graph(tools=[code_tool], sandbox=GLOBAL_SANDBOX)

            # 2. 初始化状态 (AgentState)
            # 注意：data_schema 和 remote_file_path 此时为空，等待 Auto-ETL 节点填充
            initial_state = AgentState(
                messages=[],
                user_input=user_query,

                # --- Auto-ETL 输入 ---
                raw_file_paths=raw_file_paths,
                original_filenames=original_filenames,

                # --- 待产出 (Output) ---
                remote_file_path=None,  # 等待 Auto-ETL 生成 /home/user/data.csv
                data_schema={},  # 等待 Auto-ETL 生成 Schema

                # --- 其他上下文 ---
                scenario=scenario,
                modeling_artifacts=None,
                viz_config=None,
                viz_success=False,
                final_summary=None,
                error_count=0
            )

            yield format_sse("log", "多文件智能合并 (Auto-ETL) 启动中...")

            final_viz_data_cache = []
            final_summary_cache = ""

            # 用于记录 Auto-ETL 是否已完成的标志位，防止重复打日志
            etl_completed_flag = False

            # 3. 监听 LangGraph 执行过程 (使用 astream 异步流)
            last_msg_id = None

            async for event in workflow_app.astream(initial_state, stream_mode="values"):

                # --- [新增] 监听 Auto-ETL 进度 ---
                # 如果发现 data_schema 从空变为了有值，说明 ETL 完成了
                if not etl_completed_flag and event.get("data_schema"):
                    yield format_sse("log", "数据自动合并完成，Schema 已生成。")
                    etl_completed_flag = True

                # A. 捕获最新的 Log 消息
                if "messages" in event and event["messages"]:
                    last_msg = event["messages"][-1]
                    current_id = id(last_msg)

                    if current_id != last_msg_id:
                        content = str(last_msg.content)
                        if not content.strip():
                            # 处理工具调用的中间状态
                            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                                yield format_sse("log", "正在构建代码指令...")
                        else:
                            # 过滤过长的日志，避免前端卡顿
                            if len(content) < 2000:
                                clean_content = clean_log_content(content)
                                yield format_sse("log", clean_content)
                            else:
                                yield format_sse("log", "正在生成复杂配置...")

                        last_msg_id = current_id

                # B. 捕获可视化数据 (当 Viz 成功生成文件后)
                # 注意：这里逻辑假设 Viz 节点生成的 JSON 依然保存在本地 LOCAL_VIZ_DATA_PATH
                # 如果 Viz 节点是在沙盒内生成并下载的，请确保 viz_execution_node 里有下载逻辑
                if event.get("viz_success") is True:
                    if os.path.exists(LOCAL_VIZ_DATA_PATH):
                        try:
                            with open(LOCAL_VIZ_DATA_PATH, "r", encoding="utf-8") as f:
                                data_content = json.load(f)
                                if "echarts" in data_content:
                                    echarts_data = data_content["echarts"]
                                    final_viz_data_cache = echarts_data
                                    yield format_sse("viz_data", echarts_data)
                                    yield format_sse("log", "图表数据已生成完毕。")
                        except Exception as e:
                            print(f"Error reading viz data: {e}")

                # C. 捕获最终总结
                if "final_summary" in event and event["final_summary"]:
                    summary_text = event["final_summary"]
                    final_summary_cache = summary_text
                    yield format_sse("summary", summary_text)

            # 4. 工作流结束
            full_result = {
                "success": True,
                "message": final_summary_cache,
                "echarts": final_viz_data_cache
            }

            yield format_sse("done", full_result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield format_sse("error", f"工作流执行错误: {str(e)}")


# ==========================================
# 5. 接口定义
# ==========================================

@app.post("/query_agents_stream")
async def query_agents_stream(
        file: List[UploadFile] = File(...),  # 保持变量名为 file
        request_data: str = Form(...)
):
    """
    【流式接口】Server-Sent Events (SSE)
    多文件上传 -> 解析拆解(含多Sheet) -> 上传碎片到沙盒 -> 启动 LangGraph (Auto-ETL)
    """

    # 0. 沙盒检查 (单例模式)
    if GLOBAL_SANDBOX is None:
        raise HTTPException(status_code=503, detail="计算环境(Sandbox)尚未初始化，请稍后重试")

    print(f"--- [Stream API] 已接收请求，文件数量: {len(file)} ---")

    # 1. 解析 JSON 参数
    try:
        data = json.loads(request_data)
        user_query = data.get("query")
        scenario = data.get("scenario", None)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="非法的JSON")

    # 2. 预处理：解析 -> 转 CSV -> 上传到沙盒
    raw_file_paths_in_sandbox = []  # [物理路径] 给代码用: /home/user/raw_0.csv
    original_filenames = []  # [逻辑名称] 给 LLM 用: File_A.xlsx (Sheet: Jan)

    try:
        # A. 清理沙盒旧数据 (防止单例模式下上一次请求的文件残留)
        GLOBAL_SANDBOX.run_code("%reset -f") # 添加了这个
        GLOBAL_SANDBOX.commands.run("rm -f /home/user/*.csv /home/user/*.json")

        global_fragment_count = 0  # 全局碎片计数器

        # 遍历用户上传的每一个文件
        for file_index, f in enumerate(file):
            # 读取内容
            content = await f.read()

            # B. 生成基础别名 (处理哈希文件名)
            base_alias = generate_semantic_alias(f.filename, file_index)

            try:
                # C. 解析提取 (返回 [(sheet_name, df), ...])
                extracted_items = parse_file_content(content, f.filename)

                for sheet_name, df in extracted_items:
                    # --- 物理层操作 ---

                    # D. 本地暂存 (使用最安全的物理命名 raw_0.csv)
                    local_filename = f"raw_{global_fragment_count}.csv"
                    local_path = os.path.join(TEMP_DIR, local_filename)
                    df.to_csv(local_path, index=False)

                    # E. 上传到沙盒
                    remote_path = f"/home/user/{local_filename}"
                    with open(local_path, "rb") as file_obj:
                        GLOBAL_SANDBOX.files.write(remote_path, file_obj)

                    # --- 逻辑层操作 ---

                    # F. 构建语义化全名 (用于 Prompt)
                    if sheet_name == "CSV_Content":
                        final_semantic_name = base_alias
                    else:
                        # 例如: "File_A.xlsx (Sheet: 一月)"
                        final_semantic_name = f"{base_alias} (Sheet: {sheet_name})"

                    # G. 存入列表
                    raw_file_paths_in_sandbox.append(remote_path)
                    original_filenames.append(final_semantic_name)

                    global_fragment_count += 1

            except ValueError as ve:
                raise HTTPException(status_code=400, detail=str(ve))

        # 检查是否成功上传了数据
        if not raw_file_paths_in_sandbox:
            raise HTTPException(status_code=400, detail="未解析出有效数据(可能是空文件)")

        print(f"预处理完成: 已上传 {len(raw_file_paths_in_sandbox)} 个数据碎片到沙盒。")

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件预处理或上传失败: {e}")

    # 3. 启动流式响应
    # 注意：这里我们移除了 schema 参数，改传文件列表
    # 下一步你需要修改 run_workflow_stream 的定义来接收这两个参数
    return StreamingResponse(
        run_workflow_stream(
            user_query,
            scenario,
            raw_file_paths_in_sandbox,  # 新参数 1
            original_filenames  # 新参数 2
        ),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    print("--- BI Agent 服务在端口 8009 启动中 ---")
    uvicorn.run("main:app", host="0.0.0.0", port=8009, reload=True)