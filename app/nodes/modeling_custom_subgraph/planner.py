import json
import time
import re

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from app.core.config import settings
from app.core.prompts_config import load_prompts_config
from app.core.modeling_custom_subgraph.state import CustomModelingState
from app.core.modeling_custom_subgraph.task import Task
from app.utils.extract_text_from_content import extract_text_from_content

step = "modeling"

# 1. 初始化 LLM
llm = ChatOpenAI(
    model=settings.LLM_MODEL_NAME,
    temperature=0,
    api_key=settings.OPENAI_API_KEY,
    use_responses_api=settings.USE_RESPONSES_API,
)


def planner_node(state: CustomModelingState):
    """
    【Planner 节点】
    职责：根据用户需求和数据 Schema，初始化任务列表 (Plan)。
    触发条件：仅在 plan 为空时执行。如果已有 plan，通常意味着是循环回来的，直接跳过。
    """
    print("--- [Modeling Subgraph] Planner: 正在规划任务列表... ---")

    # 1. 检查是否需要规划
    if state.get("plan") and len(state["plan"]) > 0:
        print("--- [Modeling Subgraph] Planner: 检测到已有计划，跳过规划步骤 ---")
        return {}

    # 安全地获取并初始化 metrics
    metrics = state.get("metrics") or {}
    metrics.setdefault("llm_duration", 0.0)
    metrics.setdefault("sandbox_duration", 0.0)

    # 2. 从 State 中获取上下文
    scenario = state.get("scenario")
    remote_file_path = state.get("remote_file_path")
    user_input = state.get("user_input", "")

    # 将 data_schema 字典转换为 JSON 字符串，方便 LLM 理解结构
    data_schema_obj = state.get("data_schema")
    data_schema_str = json.dumps(data_schema_obj, ensure_ascii=False, indent=2)

    # 3. 动态加载提示词配置
    config = load_prompts_config(step, scenario)
    instruction_template = config.get('planner_instruction')
    context_template = config.get('planner_context_template')

    if not instruction_template:
        raise ValueError(f"在 modeling_{scenario}.yaml 中未找到 'planner_instruction' 配置")

    # 4. SystemMessage 完全静态
    system_message = SystemMessage(content=instruction_template)

    # 5. HumanMessage 包含动态上下文
    context_content = context_template.format(
        remote_file_path=remote_file_path,
        data_schema=data_schema_str,
        user_input=user_input
    )
    context_message = HumanMessage(content=context_content)

    # 6. 调用原生 LLM
    messages = [system_message, context_message]

    llm_start_time = time.perf_counter()

    response = llm.invoke(messages, config=config)

    llm_end_time = time.perf_counter()
    llm_duration = llm_end_time - llm_start_time
    metrics["llm_duration"] += llm_duration
    print(f"--- [Time] Planner LLM 规划耗时: {llm_duration:.2f} 秒")

    # 6. 手动暴力解析 JSON 任务列表
    parsed_tasks = []
    try:
        # 第一层防护：多模态内容安全提取为纯文本
        raw_text = extract_text_from_content(response.content)

        # 第二层防护：极其强悍的正则与截取
        match = re.search(r"```(?:json)?(.*?)```", raw_text, re.DOTALL)
        if match:
            content_str = match.group(1).strip()
        else:
            start_idx = raw_text.find('{')
            end_idx = raw_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                content_str = raw_text[start_idx:end_idx + 1]
            else:
                content_str = raw_text.strip()

        parsed_json = json.loads(content_str)
        # 兼容处理，确保拿到的是列表，即使模型套了一层 {"tasks": [...]}
        parsed_tasks = parsed_json.get("tasks", [])

    except Exception as e:
        print(f"--- [Modeling Subgraph Error] Planner JSON 解析失败: {e} ---")
        print(f"--- [Modeling Subgraph Debug] 原始模型输出: {response.content} ---")
        raise RuntimeError(f"Planner 无法从文本中提取合法的 JSON 任务列表: {e}")

    actual_task_count = len(parsed_tasks)
    print(f"--- [Modeling Subgraph] Planner: 生成了 {actual_task_count} 个初始任务 ---")

    # 7. 将提取出的字典列表转换为内部 Task 对象
    initial_plan = []
    for i, item_dict in enumerate(parsed_tasks):
        desc = item_dict.get("description", "未知任务")
        print(f"  [Task {i + 1}/{actual_task_count}] {desc}")

        new_task = Task(
            id=i + 1,  # 自动分配 ID: 1, 2, 3...
            description=desc,
            status="pending"
        )
        initial_plan.append(new_task)

    # 8. 返回状态更新
    return {
        "plan": initial_plan,
        "metrics": metrics
    }