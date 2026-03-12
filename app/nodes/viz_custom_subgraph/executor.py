import os
import json
import yaml
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from ppio_sandbox.code_interpreter import Sandbox

from app.core.viz_custom_subgraph.state import CustomVizState
from app.agents.viz_custom_agent import create_viz_custom_agent


def load_chart_guide(chart_type: str) -> str:
    """加载图表使用说明"""
    guide_path = f"app/prompts/viz_charts/{chart_type}.yaml"
    try:
        with open(guide_path, 'r', encoding='utf-8') as f:
            guide = yaml.safe_load(f)
            return f"""
【图表类型】: {guide['chart_type']}
【描述】: {guide['description']}

【数据协议】
{guide['data_protocol']}

【示例】
{guide['example']}

【常见错误】
{guide['common_errors']}
"""
    except Exception as e:
        print(f"--- [Viz Executor] 警告: 无法加载图表说明 {chart_type}: {e} ---")
        return f"图表类型: {chart_type}（无详细说明）"


def create_viz_executor_node(sandbox: Sandbox):
    """工厂函数：注入 Sandbox 依赖"""

    def viz_executor_node(state: CustomVizState) -> Dict[str, Any]:
        """
        【Viz Executor 节点】
        使用 create_agent 处理单个可视化任务（并发执行）
        """
        # 获取当前任务
        viz_task = state.get("viz_task")
        if not viz_task:
            raise RuntimeError("viz_task 未找到")

        task_id = viz_task["task_id"]
        chart_type = viz_task["chart_type"]

        print(f"--- [Viz Executor] {task_id}: 开始处理 {chart_type} 图表 ---")

        # 加载图表指南
        chart_guide = load_chart_guide(chart_type)

        # 获取文件元信息和 JSON 产物
        file_metadata = state.get("file_metadata", [])
        modeling_artifacts = state.get("modeling_artifacts", {})

        # 根据 data_requirements 按需提取信息
        data_requirements = viz_task.get("data_requirements", [])
        context_parts = []

        for req in data_requirements:
            file_name = req["file_name"]
            required_columns = req.get("required_columns")

            # 从 file_metadata 中查找该文件
            file_info = next((f for f in file_metadata if f["file_name"] == file_name), None)
            if not file_info:
                print(f"--- [Viz Executor] 警告: 文件 {file_name} 未在 file_metadata 中找到 ---")
                continue

            if file_info["type"] == "feather":
                # feather 主表：只提取需要的列
                columns_desc = file_info.get("columns_desc", {})
                if required_columns:
                    selected_cols = {col: columns_desc.get(col, "") for col in required_columns if col in columns_desc}
                else:
                    selected_cols = columns_desc

                context_parts.append(f"\n【文件: {file_name}】")
                context_parts.append(f"描述: {file_info.get('description', '')}")
                context_parts.append("可用列:")
                for col_name, col_desc in selected_cols.items():
                    context_parts.append(f"  - {col_name}: {col_desc}")

            elif file_info["type"] == "json":
                # JSON 产物：提供完整内容和键描述
                artifact_key = file_name.replace('.json', '')
                artifact_data = modeling_artifacts.get(artifact_key, {})
                columns_desc = file_info.get("columns_desc", {})

                context_parts.append(f"\n【JSON产物: {file_name}】")
                context_parts.append(f"描述: {file_info.get('description', '')}")
                context_parts.append("键说明:")
                for key, desc in columns_desc.items():
                    context_parts.append(f"  - {key}: {desc}")
                context_parts.append(f"完整数据: {json.dumps(artifact_data, ensure_ascii=False)}")

        file_info_text = "\n".join(context_parts)

        # 创建 agent
        agent = create_viz_custom_agent(sandbox, task_id, chart_guide)

        # 构建动态上下文
        context_content = f"""
【任务ID】: {task_id}
【图表类型】: {chart_type}
【标题】: {viz_task['title']}
【描述】: {viz_task['description']}

【可用数据】
{file_info_text}
"""

        # 调用 agent
        agent_result = agent.invoke({
            "viz_task": viz_task,
            "messages": [HumanMessage(content=context_content)]
        })

        # 检查产物是否生成
        try:
            check_result = sandbox.commands.run(f"test -f /home/user/viz_output_{task_id}.json && echo 'EXISTS' || echo 'NOT_FOUND'")
            if "EXISTS" not in check_result.stdout:
                raise RuntimeError("输出文件未生成")
        except Exception as e:
            print(f"--- [Viz Executor] {task_id}: 失败 - {e} ---")
            return {}

        # 下载产物到本地
        try:
            print(f"--- [Viz Executor] {task_id}: 正在下载产物... ---")
            file_content = sandbox.files.read(f"/home/user/viz_output_{task_id}.json")

            local_path = f"temp/viz_{task_id}.json"
            os.makedirs("temp", exist_ok=True)

            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(file_content)

            print(f"--- [Viz Executor] {task_id}: 产物已下载到 {local_path} ---")
            return {}

        except Exception as e:
            print(f"--- [Viz Executor] {task_id}: 下载失败 - {e} ---")
            return {}

    return viz_executor_node
