from typing import List, Dict, Any
import json
import re
import time
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.prompts_config import load_prompts_config
from app.core.viz_custom_subgraph.state import CustomVizState
from app.core.viz_custom_subgraph.schemas import VizTaskSimple, VizPlannerOutput
from app.utils.extract_text_from_content import extract_text_from_content

# 初始化 LLM
llm = ChatOpenAI(
    model=settings.LLM_MODEL_NAME,
    temperature=0,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_API_BASE,
    use_responses_api=settings.USE_RESPONSES_API,
    streaming=True,  # responses API 需要流式调用
)


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
    generated_files = state.get("generated_data_files", [])

    # 调试：打印接收到的 state
    print(f"--- [Debug] Viz Planner 接收到的 state ---")
    print(f"  modeling_summary: {modeling_summary[:100] if modeling_summary else 'EMPTY'}...")
    print(f"  generated_files: {generated_files}")
    print(f"  state keys: {list(state.keys())}")

    if not modeling_summary:
        raise RuntimeError("--- [Viz Subgraph] Planner: 错误! modeling_summary 为空 ---")

    # 3. 加载提示词配置
    prompts = load_prompts_config("viz", "custom")
    planner_instruction = prompts["planner_instruction"]
    context_template = prompts["planner_context_template"]

    # 4. SystemMessage 完全静态
    system_message = SystemMessage(content=planner_instruction)

    # 5. HumanMessage 包含动态上下文
    context_content = context_template.format(
        modeling_summary=modeling_summary,
        user_input=user_input,
        generated_files=generated_files
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

        # 6. 组装完整的 viz_tasks（添加 task_id 和文件元信息）
        file_metadata = state.get("file_metadata", [])
        viz_tasks = []

        for idx, simple_task in enumerate(planner_output.tasks, start=1):
            task_id = f"viz_task_{idx}"

            # 提取该任务所需文件的列信息
            file_columns = []
            for source_file in simple_task.source_files:
                # 从 file_metadata 中查找对应的文件
                for file_meta in file_metadata:
                    if file_meta["file_name"] == source_file:
                        file_columns.append({
                            "file_name": source_file,
                            "columns": file_meta["columns"]
                        })
                        break

            # 组装完整的 VizTask
            full_task = {
                "task_id": task_id,
                "chart_type": simple_task.chart_type,
                "title": simple_task.title,
                "description": simple_task.description,
                "source_files": simple_task.source_files,
                "file_columns": file_columns
            }

            viz_tasks.append(full_task)
            print(f"  - {task_id}: {simple_task.chart_type} - {simple_task.title}")
            print(f"    数据源: {simple_task.source_files}")
            print(f"    透传列信息: {len(file_columns)} 个文件")

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
