from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.prompts_config import load_prompts_config
from app.core.subgraph.state import CustomModelingState
from app.core.subgraph.task import Task
from app.core.subgraph.schemas import ReflectorOutput

# 1. 初始化 LLM
llm = ChatOpenAI(
    model=settings.LLM_MODEL_NAME,
    temperature=0,
    api_key=settings.OPENAI_API_KEY
)

step = "modeling"

def reflector_node(state: CustomModelingState):
    """
    【Reflector 节点】
    职责：
    1. 审查 Executor 的执行结果 (Task.result)。
    2. 决定下一步动作 (Finish / Retry / Insert)。
    3. 维护 Plan 列表 (包括插入任务和重排 ID) 和 Scratchpad。
    """
    plan = state["plan"]
    current_idx = state["current_task_index"]

    # 安全检查
    if current_idx >= len(plan):
        return {"error": "Reflector索引越界"}

    task = plan[current_idx]
    print(f"--- [Subgraph] Reflector: 正在审查 Task {task.id} 的结果... ---")

    # 1. 准备 Context
    execution_result = task.result

    # 2. 加载 Prompt
    scenario = state.get("scenario")

    config = load_prompts_config(step, scenario)
    instruction_template = config.get('reflector_instruction')

    remaining_tasks = []
    for t in plan[current_idx + 1:]:
        remaining_tasks.append(f"ID {t.id}: {t.description} ({t.status})")

    remaining_plan_str = "\n".join(remaining_tasks) if remaining_tasks else "无后续任务"

    # 3. 构造 Prompt
    system_content = instruction_template.format(
        task_description=task.description,
        execution_result=execution_result,
        remaining_plan=remaining_plan_str
    )

    # 4. 调用 LLM (Structured Output)
    structured_llm = llm.with_structured_output(ReflectorOutput)
    decision: ReflectorOutput = structured_llm.invoke([SystemMessage(content=system_content)])

    print(f"--- [Subgraph] Reflector 决策: {decision.action} ---")

    # 5. 处理决策逻辑 (State Updates)
    updates = {}
    scratchpad = state.get("scratchpad")

    # === 分支 A: 任务完成 ===
    if decision.action == "finish_task":
        # 标记当前任务为完成
        task.mark_completed(task.result)

        # 更新 Scratchpad
        if decision.new_scratchpad:
            note = f"[Task {task.id} 成功]: {decision.new_scratchpad}"
            scratchpad.append(note)

        # 准备返回更新
        updates = {
            "current_task_index": current_idx + 1,  # 指针移向下一个任务
            "retry_count": 0,  # 重置重试计数
            "scratchpad": scratchpad,
            # 注意：plan 已经在内存中通过 task.mark_completed 修改了，LangGraph 会自动合并
            "plan": plan
        }

    # === 分支 B: 任务失败，重试 ===
    elif decision.action == "retry_task":
        current_retry = state.get("retry_count", 0)

        # 检查是否超过最大重试次数 (例如 3 次)
        if current_retry >= 3:
            print(f"--- [Subgraph] Task {task.id} 重试次数过多，标记失败 ---")
            task.mark_failed(f"达到了最大重试次数，Reason: {decision.retry_reason}")

            scratchpad.append(f"[Task {task.id} 执行失败]: 达到了最大重试次数！")

            raise RuntimeError(f"[Task {task.id} 执行失败]: 达到了最大重试次数！")
        else:
            # 还在重试机会内
            print(f"--- [Subgraph] Task {task.id} 将进行第 {current_retry + 1} 次重试 ---")
            task.status = "pending"  # 确保状态是 pending，Executor 才能识别

            if decision.retry_suggestion:
                note = f"[Task {task.id} 重试建议]: {decision.retry_suggestion}"
                scratchpad.append(note)

            updates = {
                "retry_count": current_retry + 1,
                "current_task_index": current_idx,
                "scratchpad": scratchpad,
                "plan": plan
            }

    # === 分支 C: 插入新任务 ===
    elif decision.action == "insert_task":
        # 逻辑：Reflector 认为必须先做个新任务（例如清洗数据），才能继续
        print(f"--- [Subgraph] 插入新任务: {decision.new_task_description} ---")

        task.mark_completed(task.result)
        if decision.insert_reason:
            scratchpad.append(f"[Task {task.id} 完成并且需要在下一步新增任务]: 发现了这样的问题: {decision.insert_reason}")

        new_task = Task(
            id=0,
            description=decision.new_task_description,
            status="pending"
        )

        # 插入到当前任务之后 (Index + 1)
        plan.insert(current_idx + 1, new_task)

        # 重排 ID
        for i, t in enumerate(plan):
            t.id = i + 1

        updates = {
            "plan": plan,
            "scratchpad": scratchpad,
            "retry_count": 0,
            "current_task_index": current_idx + 1
        }

    return updates