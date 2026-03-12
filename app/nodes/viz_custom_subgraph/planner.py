from typing import List, Dict, Any
import json
import re
import time
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.prompts_config import load_prompts_config
from app.core.viz_custom_subgraph.state import CustomVizState
from app.core.viz_custom_subgraph.schemas import VizTaskSimple, VizPlannerOutput
from app.utils.extract_text_from_content import extract_text_from_content
from app.utils.llm_factory import create_llm, apply_retry

# 初始化 LLM (主模型使用Anthropic)
llm = apply_retry(create_llm(use_flash=False))


def viz_planner_node(state: CustomVizState) -> Dict[str, Any]:
    """
    【Viz Planner 节点】
    职责：根据 modeling_summary 和用户需求，规划可视化任务列表
    """
    print("--- [Viz Subgraph] Planner: 正在规划可视化任务... ---")

    # 1. 检查是否已有计划
    if state.get("viz_tasks") and len(state["viz_tasks"]) > 0:
        print("--- [Viz Subgraph] Planner: 检测到已有计划，跳过规划 ---")
        return {}

    # 2. 获取上下文
    modeling_summary = state.get("modeling_summary", "")
    user_input = state.get("user_input", "")
    file_metadata = state.get("file_metadata", [])

    # 调试：打印接收到的 state
    print(f"--- [Debug] Viz Planner 接收到的 state ---")
    print(f"  modeling_summary: {modeling_summary[:100] if modeling_summary else 'EMPTY'}...")
    print(f"  file_metadata count: {len(file_metadata)}")
    print(f"  state keys: {list(state.keys())}")

    if not modeling_summary:
        raise RuntimeError("--- [Viz Subgraph] Planner: 错误! modeling_summary 为空 ---")

    # 3. 构造文件元信息文本
    file_metadata_parts = []
    for file_info in file_metadata:
        file_name = file_info.get("file_name", "")
        description = file_info.get("description", "")
        file_type = file_info.get("type", "")
        columns_desc = file_info.get("columns_desc", {})

        file_metadata_parts.append(f"\n文件: {file_name}")
        file_metadata_parts.append(f"类型: {file_type}")
        file_metadata_parts.append(f"描述: {description}")

        if file_type == "feather" and columns_desc:
            file_metadata_parts.append("可用列:")
            for col_name, col_desc in columns_desc.items():
                file_metadata_parts.append(f"  - {col_name}: {col_desc}")
        elif file_type == "json" and columns_desc:
            file_metadata_parts.append("数据键:")
            for key, desc in columns_desc.items():
                file_metadata_parts.append(f"  - {key}: {desc}")

    file_metadata_text = "\n".join(file_metadata_parts)

    # 4. 加载提示词配置
    prompts = load_prompts_config("viz", "custom")
    planner_instruction = prompts["planner_instruction"]
    context_template = prompts["planner_context_template"]

    # 5. SystemMessage 完全静态
    system_message = SystemMessage(content=planner_instruction)

    # 6. HumanMessage 包含动态上下文
    context_content = context_template.format(
        modeling_summary=modeling_summary,
        user_input=user_input,
        file_metadata_text=file_metadata_text
    )
    context_message = HumanMessage(content=context_content)

    print("--- [Viz Subgraph] Planner: 调用大模型中... ---")
    start_time = time.perf_counter()

    # 6. 调用 LLM
    messages = [system_message, context_message]
    response = llm.invoke(messages)
    llm_duration = time.perf_counter() - start_time

    print(f"--- [Time] Planner LLM 耗时: {llm_duration:.2f} 秒")

    # 6. 解析 JSON 输出
    try:
        # 尝试提取 JSON（支持 Markdown 代码块）
        content = extract_text_from_content(response.content)
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        json_str = m.group(1) if m else content
        parsed = json.loads(json_str)
        planner_output = VizPlannerOutput(**parsed)

        print(f"--- [Viz Subgraph] Planner: 生成了 {len(planner_output.tasks)} 个可视化任务 ---")

        # 6. 组装完整的 viz_tasks（添加 task_id）
        viz_tasks = []

        for idx, simple_task in enumerate(planner_output.tasks, start=1):
            task_id = f"viz_task_{idx}"

            # 组装完整的 VizTask
            full_task = {
                "task_id": task_id,
                "chart_type": simple_task.chart_type,
                "title": simple_task.title,
                "description": simple_task.description,
                "data_requirements": [req.model_dump() for req in simple_task.data_requirements]
            }

            viz_tasks.append(full_task)
            print(f"  - {task_id}: {simple_task.chart_type} - {simple_task.title}")
            for req in simple_task.data_requirements:
                cols_info = f"列: {req.required_columns}" if req.required_columns else "全部数据"
                print(f"    数据源: {req.file_name} ({cols_info})")

        # 初始化 metrics
        viz_metrics = {"planner_llm_duration": llm_duration}

        return {
            "viz_tasks": viz_tasks,
            "viz_metrics": viz_metrics
        }

    except Exception as e:
        error_msg = f"--- [Viz Subgraph] Planner: JSON 解析失败: {e} ---"
        print(error_msg)
        print(f"LLM 输出: {response.content}")
        raise RuntimeError(error_msg)
