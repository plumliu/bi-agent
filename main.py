import os
import sys
import json
import shutil
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ==========================================
# 0. 环境路径配置
# ==========================================
# 确保能导入 app 模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from e2b_code_interpreter import Sandbox
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
    print("--- [Lifespan] Creating Global Sandbox... ---")

    try:
        # 创建沙箱 (设置较长的超时时间，避免空闲断开)
        GLOBAL_SANDBOX = Sandbox.create(api_key=settings.E2B_API_KEY, timeout=3600)

        print("--- [Lifespan] Pre-warming: Installing Dependencies... ---")
        GLOBAL_SANDBOX.run_code("!pip install pyarrow scikit-learn pandas numpy")
        print("--- [Lifespan] Sandbox Ready! ---")

        yield  # 服务运行中

    finally:
        print("--- [Lifespan] Closing Global Sandbox... ---")
        if GLOBAL_SANDBOX:
            GLOBAL_SANDBOX.kill()
            print("FastAPI 退出，Sandbox 将随进程销毁")


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
# 3. 核心接口逻辑
# ==========================================

@app.post("/query_agents")
async def query_agents(
        file: UploadFile = File(...),
        request_data: str = Form(...)
):
    """
    接收 CSV 文件和用户 Query，运行 BI Multi-Agent Workflow。
    """
    if GLOBAL_SANDBOX is None:
        raise HTTPException(status_code=503, detail="Sandbox is not initialized")

    print(f"--- [API Request] Received request. Filename: {file.filename} ---")

    # 1. 解析请求参数
    try:
        data = json.loads(request_data)
        user_query = data.get("query", "")
        scenario = data.get("scenario", None)
        print(f"--- [API Request] Query: {user_query} ---")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request_data")

    # 2. 保存上传的 CSV 到本地
    try:
        with open(LOCAL_CSV_PATH, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"--- [System] CSV saved to {LOCAL_CSV_PATH} ---")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # 3. 读取 Schema
    try:
        schema = get_csv_schema(LOCAL_CSV_PATH)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV schema: {str(e)}")

    # ==========================================
    # 4. 运行 Agent Workflow (使用全局沙箱)
    # ==========================================

    # 使用 Lock 确保同一时间只有一个请求在使用沙箱，防止文件覆盖或变量污染
    async with SANDBOX_LOCK:
        print("--- [System] Acquired Sandbox Lock ---")

        try:
            print("--- [System] Cleaning Sandbox Memory & Files... ---")
            GLOBAL_SANDBOX.run_code("%reset -f")
            GLOBAL_SANDBOX.commands.run("rm -f /home/user/*.csv /home/user/*.json /home/user/*.feather")

            # B. 上传新数据
            remote_path = "/home/user/data.csv"
            print(f"--- [System] Uploading Data to Sandbox... ---")
            with open(LOCAL_CSV_PATH, "rb") as f:
                GLOBAL_SANDBOX.files.write(remote_path, f)

            # C. 构建图 (注入全局 Sandbox)
            code_tool = create_code_interpreter_tool(GLOBAL_SANDBOX)
            workflow_app = build_graph(tools=[code_tool], sandbox=GLOBAL_SANDBOX)

            # D. 初始化状态
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

            print("================ WORKFLOW START ================")
            final_state = workflow_app.invoke(initial_state)
            print("================ WORKFLOW FINISHED ================")

            # E. 提取结果

            # 1. 提取自然语言总结
            summary = final_state.get("final_summary", "分析完成，但未生成总结。")

            # 2. 提取可视化数据 (从本地文件读取)
            viz_data = []
            if os.path.exists(LOCAL_VIZ_DATA_PATH) and final_state.get("viz_success"):
                try:
                    with open(LOCAL_VIZ_DATA_PATH, "r", encoding="utf-8") as f:
                        data_content = json.load(f)
                        # 确保只返回 echarts 数组
                        if "echarts" in data_content:
                            viz_data = data_content["echarts"]
                except Exception as e:
                    print(f"[Warning] Failed to read viz_data.json: {e}")

            # F. 构造最终响应 (符合你的要求)
            return {
                "success": True,
                "message": summary,
                "echarts": viz_data
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")


# ==========================================
# 3. 启动入口
# ==========================================
if __name__ == "__main__":
    print("--- Starting BI Agent Server on Port 8011 ---")
    uvicorn.run("main:app", host="0.0.0.0", port=8011, reload=True)