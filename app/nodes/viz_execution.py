import json
import os
from langchain_core.messages import HumanMessage
from app.core.state import WorkflowState
from app.tools.viz.generator import generate_viz_data


def viz_execution_node(state: WorkflowState):
    print("--- [Viz Exec] 开始执行Viz阶段的配置生成图表数据 ---")

    # 1. 准备路径 (对应 Fetch Artifacts 的下载路径)
    local_dir = "temp"
    local_feather = os.path.join(local_dir, "processed_data.feather")
    local_artifacts = os.path.join(local_dir, "analysis_artifacts.json")
    local_output = os.path.join(local_dir, "viz_data.json")

    # 2. 获取配置
    viz_config = state.get("viz_config")
    if not viz_config:
        return {
            "viz_success": False
        }

    # 3. 调用本地生成器 (核心动作)
    result = generate_viz_data(local_feather, local_artifacts, viz_config)

    if result["success"]:
        print(f"--- [Viz Exec] 成功! 生成出了: {list(result['viz_data'].keys())} ---")

        # 保存最终 JSON 供前端读取
        with open(local_output, "w", encoding='utf-8') as f:
            json.dump(result["viz_data"], f, ensure_ascii=False, indent=2)

        return {
            "viz_success": True
        }
    else:
        # 4. 失败处理 (反馈机制 - Reflexion Loop)
        errors = result.get("errors", {})
        error_msg = json.dumps(errors, ensure_ascii=False, indent=2)
        print(f"--- [Viz Exec] 失败! 错误:\n{error_msg} ---")

        return {
            "viz_success": False
        }