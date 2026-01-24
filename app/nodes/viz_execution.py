import json
import os
from langchain_core.messages import SystemMessage
from app.core.state import AgentState
from app.tools.viz.generator import generate_viz_data


def viz_execution_node(state: AgentState):
    print("--- [Viz Exec] 开始执行Viz阶段是的配置生成图表数据 ---")

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
            "messages": [SystemMessage(content="[System Error] 在state中没有找到'viz_config'！")]
        }

    # 3. 调用本地生成器 (核心动作)
    # 这一步是在本地 Python 环境运行，速度极快
    result = generate_viz_data(local_feather, local_artifacts, viz_config)

    if result["success"]:
        print(f"--- [Viz Exec] 成功! 生成出了: {list(result['viz_data'].keys())} ---")

        # 保存最终 JSON 供前端读取
        with open(local_output, "w", encoding='utf-8') as f:
            json.dump(result["viz_data"], f, ensure_ascii=False, indent=2)

        return {
            "viz_success": True,
            "messages": [SystemMessage(content="可视化信息生成成功！")]
        }
    else:
        # 4. 失败处理 (反馈机制)
        errors = result.get("errors", {})
        error_msg = json.dumps(errors, ensure_ascii=False, indent=2)
        print(f"--- [Viz Exec] 失败! 错误:\n{error_msg} ---")

        # 构造反馈消息，让 Viz Agent 看到错误并修正
        feedback_msg = (
            f"[System Alert] 可视化数据生成失败.\n"
            f"可视化数据 Python 生成器基于你的配置给出了下述错误信息:\n"
            f"{error_msg}\n\n"
            f"请分析这些错误 (比如 wrong column names, missing data keys) "
            f"修复错误，并且给出一个正确的配置"
        )

        return {
            "viz_success": False,
            "messages": [SystemMessage(content=feedback_msg)]
        }