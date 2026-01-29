import os
import sys
import asyncio
import json
from contextlib import asynccontextmanager

# 1. 环境路径配置 (确保能导入 app)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ppio_sandbox.code_interpreter import Sandbox
from app.core.config import settings
from app.core.subgraph.state import CustomModelingState
from app.graph.modeling_custom_workflow import build_modeling_custom_subgraph
from app.utils.csv_reader import get_csv_schema

# 模拟全局变量
GLOBAL_SANDBOX = None
TEMP_DIR = "temp"
LOCAL_CSV_PATH = os.path.join(TEMP_DIR, "temp_data.csv")

# ==========================================
# Mock 全局上下文
# ==========================================


import app.nodes.subgraph.executor as executor_module


async def setup_sandbox():
    """初始化沙盒并注入到 Executor 模块"""
    global GLOBAL_SANDBOX
    print("--- [Test] Creating PPIO Sandbox... ---")

    GLOBAL_SANDBOX = Sandbox.create(
        settings.PPIO_TEMPLATE,
        api_key=settings.PPIO_API_KEY,
        timeout=3600
    )

    # 将沙盒实例注入到 executor 模块中
    # 这样 executor_node 运行时就能用 GLOBAL_SANDBOX 了
    executor_module.GLOBAL_SANDBOX = GLOBAL_SANDBOX

    print("--- [Test] Sandbox Ready & Injected! ---")
    return GLOBAL_SANDBOX


async def main():
    # 1. 检查数据文件
    if not os.path.exists(LOCAL_CSV_PATH):
        print(f"[Error] 数据文件不存在: {LOCAL_CSV_PATH}")
        print("请先在 temp/ 目录下放入一个 temp_data.csv 文件用于测试。")
        return

    # 2. 启动沙盒
    sandbox = await setup_sandbox()

    try:
        # 3. 准备环境 (清理 & 上传)
        print("--- [Test] Uploading Data... ---")
        sandbox.run_code("%reset -f")
        remote_path = "/home/user/data.csv"

        with open(LOCAL_CSV_PATH, "rb") as f:
            sandbox.files.write(remote_path, f)

        # 4. 获取 Schema
        print("--- [Test] Reading Schema... ---")
        schema = get_csv_schema(LOCAL_CSV_PATH)
        print(f"Schema Detected: {list(schema.keys())}")

        # 5. 构建子图
        print("--- [Test] Building Subgraph... ---")
        subgraph = build_modeling_custom_subgraph()

        # 6. 构造初始状态
        # 注意：这里 user_input 是硬编码的测试问题
        initial_state = CustomModelingState(
            messages=[],
            user_input="为我找出这批机器中的异常机器",
            data_schema=schema,
            remote_file_path=remote_path,
            scenario="custom",  # 必须是 custom，以便加载正确的 prompt

            # 子图专用字段初始化
            plan=[],
            current_task_index=0,
            retry_count=0,
            scratchpad=[]
        )

        print("--- [Test] Starting Workflow... ---")
        print("=" * 60)

        # 7. 运行子图
        # 使用 astream 观察每一步的输出
        async for event in subgraph.astream(initial_state):
            for node_name, state_update in event.items():
                print(f"\n>>> Node Finished: [{node_name}]")

                # 打印 Plan 的变化
                if "plan" in state_update:
                    plan = state_update["plan"]
                    print("Current Plan:")
                    for task in plan:
                        status_icon = {
                            "pending": "[ ]",
                            "running": "[...]",
                            "completed": "[x]",
                            "failed": "[!]"
                        }.get(task.status, "[?]")
                        print(f"  {status_icon} ID {task.id}: {task.description}")

                # 打印 Scratchpad 的变化
                if "scratchpad" in state_update and state_update["scratchpad"]:
                    print("Latest Scratchpad:")
                    print(f"  {state_update['scratchpad'][-1]}")

                print("-" * 30)

        print("=" * 60)
        print("--- [Test] Workflow Completed Successfully ---")

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        # 8. 清理沙盒
        print("--- [Test] Killing Sandbox... ---")
        sandbox.kill()
        print("--- [Test] Sandbox Has Been Killed... ---")


if __name__ == "__main__":
    asyncio.run(main())