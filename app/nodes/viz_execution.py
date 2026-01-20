import json
import os
from langchain_core.messages import SystemMessage
from app.core.state import AgentState
from app.tools.viz.generator import generate_viz_data


def viz_execution_node(state: AgentState):
    print("--- [Step 3] Viz Execution Node (Local) ---")

    # 1. 准备路径 (对应 Fetch Artifacts 的下载路径)
    local_dir = "temp"
    local_feather = os.path.join(local_dir, "processed_data.feather")
    local_artifacts = os.path.join(local_dir, "analysis_artifacts.json")
    local_output = os.path.join(local_dir, "viz_data.json")

    # 2. 获取配置
    viz_config = state.get("viz_config")
    if not viz_config:
        return {
            "viz_success": False,
            "messages": [SystemMessage(content="[System Error] No 'viz_config' found in state.")]
        }

    # 3. 调用本地生成器 (核心动作)
    # 这一步是在本地 Python 环境运行，速度极快
    result = generate_viz_data(local_feather, local_artifacts, viz_config)

    if result["success"]:
        print(f"--- [Viz Exec] Success! Generated: {list(result['viz_data'].keys())} ---")

        # 保存最终 JSON 供前端读取
        with open(local_output, "w", encoding='utf-8') as f:
            json.dump(result["viz_data"], f, ensure_ascii=False, indent=2)

        return {
            "viz_success": True,
            # 可以在这里添加一条总结消息，或者什么都不加直接结束
            "messages": [SystemMessage(content="Visualization data generated successfully.")]
        }
    else:
        # 4. 失败处理 (反馈机制)
        errors = result.get("errors", {})
        error_msg = json.dumps(errors, ensure_ascii=False, indent=2)
        print(f"--- [Viz Exec] Failed! Errors:\n{error_msg} ---")

        # 构造反馈消息，让 Viz Agent 看到错误并修正
        feedback_msg = (
            f"[System Alert] Visualization Generation Failed.\n"
            f"The Python generator reported the following errors based on your config:\n"
            f"{error_msg}\n\n"
            f"Please analyze these errors (e.g., wrong column names, missing data keys) "
            f"and output a CORRECTED JSON configuration."
        )

        return {
            "viz_success": False,
            "messages": [SystemMessage(content=feedback_msg)]
        }