import json
import time  # [新增] 引入时间模块

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.prompts_config import load_prompts_config
from app.core.subgraph.state import CustomModelingState
from app.core.subgraph.task import Task
from app.core.subgraph.schemas import PlannerOutput

step = "modeling"

# 1. 初始化 LLM
# Planner 不需要写代码，只需要极强的逻辑规划能力，建议使用配置中的主模型
llm = ChatOpenAI(
    model=settings.LLM_MODEL_NAME,
    temperature=0,
    api_key=settings.OPENAI_API_KEY
)


def planner_node(state: CustomModelingState):
    """
    【Planner 节点】
    职责：根据用户需求和数据 Schema，初始化任务列表 (Plan)。
    触发条件：仅在 plan 为空时执行。如果已有 plan，通常意味着是循环回来的，直接跳过。
    """
    print("--- [Subgraph] Planner: 正在规划任务列表... ---")

    # 1. 检查是否需要规划
    if state.get("plan") and len(state["plan"]) > 0:
        print("--- [Subgraph] Planner: 检测到已有计划，跳过规划步骤 ---")
        return {}

    # [新增] 安全地获取并初始化 metrics
    metrics = state.get("metrics") or {}
    metrics.setdefault("llm_duration", 0.0)
    metrics.setdefault("sandbox_duration", 0.0)

    # 2. 从 State 中获取上下文
    scenario = state.get("scenario")
    remote_file_path = state.get("remote_file_path")
    user_input = state.get("user_input")

    # 将 data_schema 字典转换为 JSON 字符串，方便 LLM 理解结构
    data_schema_obj = state.get("data_schema")
    data_schema_str = json.dumps(data_schema_obj, ensure_ascii=False, indent=2)

    # 3. 动态加载提示词配置
    config = load_prompts_config(step, scenario)
    instruction_template = config.get('planner_instruction')

    if not instruction_template:
        raise ValueError(f"在 modeling_{scenario}.yaml 中未找到 'planner_instruction' 配置")

    # 4. 格式化 Prompt
    system_content = instruction_template.format(
        remote_file_path=remote_file_path,
        data_schema=data_schema_str,
        user_input=user_input
    )

    # 5. 调用 LLM (Structured Output)
    structured_llm = llm.with_structured_output(PlannerOutput)
    messages = [SystemMessage(content=system_content)]

    # [新增] 掐表开始：记录 LLM 调用耗时
    llm_start_time = time.perf_counter()

    output: PlannerOutput = structured_llm.invoke(messages)

    # [新增] 掐表结束：计算并累加耗时
    llm_end_time = time.perf_counter()
    llm_duration = llm_end_time - llm_start_time
    metrics["llm_duration"] += llm_duration
    print(f"--- [Time] Planner LLM 规划耗时: {llm_duration:.2f} 秒")

    actual_task_count = len(output.tasks)
    print(f"--- [Subgraph] Planner: 生成了 {actual_task_count} 个初始任务 ---")

    # 逐行遍历打印，彻底防止控制台截断，也绝不可能数错
    for idx, item in enumerate(output.tasks, start=1):
        print(f"  [Task {idx}/{actual_task_count}] {item.description}")

    # 6. 将 Output 转换为内部 Task 对象
    initial_plan = []
    for i, item in enumerate(output.tasks):
        new_task = Task(
            id=i + 1,  # 自动分配 ID: 1, 2, 3...
            description=item.description,
            status="pending"
        )
        initial_plan.append(new_task)

    # 7. 返回状态更新
    return {
        "plan": initial_plan,
        "current_task_index": 0,  # 指针归零
        "retry_count": 0,  # 重试计数归零
        "scratchpad": [],  # 初始化为空列表
        "metrics": metrics  #  将更新后的统计数据传回状态机
    }