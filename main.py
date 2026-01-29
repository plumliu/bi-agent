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
    1. 启动时：创建 E2B 沙箱并安装依赖 (预热)。
    2. 运行时：保持沙箱存活。
    3. 关闭时：销毁沙箱。
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
# 4. 核心逻辑：流式生成器
# ==========================================

async def run_workflow_stream(user_query: str, scenario: str, schema: dict):
    """
    异步生成器，用于 SSE 流式输出
    """
    if GLOBAL_SANDBOX is None:
        yield format_sse("error", "Sandbox not initialized")
        return

    # [关键] 获取锁，确保本流程独占沙箱
    async with SANDBOX_LOCK:
        try:
            yield format_sse("log", "正在初始化计算环境...")

            # 1. 清理沙箱
            GLOBAL_SANDBOX.run_code("%reset -f")
            GLOBAL_SANDBOX.commands.run("rm -f /home/user/*.csv /home/user/*.json /home/user/*.feather")

            # 2. 上传数据
            yield format_sse("log", "正在上传数据...")
            remote_path = "/home/user/data.csv"
            with open(LOCAL_CSV_PATH, "rb") as f:
                GLOBAL_SANDBOX.files.write(remote_path, f)

            # 3. 构建图
            code_tool = create_code_interpreter_tool(GLOBAL_SANDBOX)
            workflow_app = build_graph(tools=[code_tool], sandbox=GLOBAL_SANDBOX)

            # 4. 初始化状态
            initial_state = AgentState(
                messages=[],
                user_input=user_query,
                data_schema=schema,
                remote_file_path=remote_path,
                scenario=scenario,
                modeling_artifacts=None,
                viz_config=None,
                viz_success=False
            )

            yield format_sse("log", "开始执行多智能体工作流...")

            final_viz_data_cache = []
            final_summary_cache = ""

            # 5. 监听 LangGraph 执行过程 (使用 astream 异步流)
            last_msg_id = None

            async for event in workflow_app.astream(initial_state, stream_mode="values"):

                # A. 捕获最新的 Log 消息
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
                                clean_content = clean_log_content(content)
                                yield format_sse("log", clean_content)
                            else:
                                yield format_sse("log", "正在生成复杂配置...")

                        last_msg_id = current_id

                # B. 捕获可视化数据 (当 Viz 成功生成文件后)
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

            # 6. 工作流结束，组装并发送完整的 done 对象
            full_result = {
                "success": True,
                "message": final_summary_cache,
                "echarts": final_viz_data_cache
            }

            # 发送最终的大对象
            yield format_sse("done", full_result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield format_sse("error", f"内网错误: {str(e)}")


# ==========================================
# 5. 接口定义
# ==========================================

@app.post("/query_agents_stream")
async def query_agents_stream(
        files: List[UploadFile] = File(...),
        request_data: str = Form(...)
):
    """
    【流式接口】Server-Sent Events (SSE)
    支持多文件上传 -> 解析(含多Sheet) -> 扁平化处理 -> 自动合并 -> 生成唯一 CSV
    """
    print(f"--- [Stream API] 已接收请求，文件数量: {len(files)} ---")

    # 1. 解析 JSON 参数
    try:
        data = json.loads(request_data)
        user_query = data.get("query", "")
        scenario = data.get("scenario", None)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="非法的JSON")

    # 2. 内存处理：遍历文件并解析
    data_frames = []  # 用于存放所有解析出来的 DataFrame (扁平列表)

    try:
        for file in files:
            # 在内存中读取二进制内容
            content = await file.read()

            try:
                # file_dfs 是一个 list，可能包含 1 个(CSV) 或 多个(Excel多Sheet) DataFrame
                file_dfs = parse_file_content(content, file.filename)

                # 如果 file_dfs 是 [df1, df2]，extend 后 data_frames 增加两个元素，而不是由一个列表元素
                data_frames.extend(file_dfs)

            except ValueError as ve:
                raise HTTPException(status_code=400, detail=str(ve))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"处理文件 {file.filename} 时发生未知错误: {e}")

        # 3. 合并数据
        if not data_frames:
            raise HTTPException(status_code=400, detail="未解析出有效数据(可能是空文件)")

        print(f"正在合并数据表，共 {len(data_frames)} 个表格片段...")

        # pd.concat 接收一个 DataFrame 的列表 [df1, df2, df3...]
        # ignore_index=True 确保行号重新排列 (0, 1, 2...)
        final_df = pd.concat(data_frames, ignore_index=True, sort=False)

        # 4. 最终落盘
        final_df.to_csv(LOCAL_CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"合并完成，总行数: {len(final_df)}，已保存至: {LOCAL_CSV_PATH}")

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件流处理或合并失败: {e}")

    # 5. 读取 Schema 并运行工作流
    try:
        schema = get_csv_schema(LOCAL_CSV_PATH)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Schema 转换失败: {e}")

    # 6. 返回流式响应
    return StreamingResponse(
        run_workflow_stream(user_query, scenario, schema),
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    print("--- BI Agent 服务在端口 8009 启动中 ---")
    uvicorn.run("main:app", host="0.0.0.0", port=8009, reload=True)