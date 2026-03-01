import time
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.prompts_config import load_prompts_config
from app.core.subgraph.state import CustomModelingState
from app.tools.sandbox import create_code_interpreter_tool
from ppio_sandbox.code_interpreter import Sandbox

# 1. 初始化基础 LLM
llm = ChatOpenAI(
    model=settings.LLM_MODEL_NAME,
    temperature=0,
    api_key=settings.OPENAI_API_KEY
)

step = "modeling"


def executor_node(state: CustomModelingState, sandbox: Sandbox):
    """
    【Executor 节点】(采用 partial 依赖注入模式)
    职责：看全局计划 -> 思考 -> 决定调用工具写代码 (或给出最终结论)
    """
    print("--- [Subgraph] Executor: 正在思考与规划行动... ---")

    # 1. 动态生成并绑定真实的沙盒工具
    python_tool = create_code_interpreter_tool(sandbox)
    llm_with_tools = llm.bind_tools([python_tool])

    # 2. 初始化时间统计
    metrics = state.get("metrics") or {}
    metrics.setdefault("llm_duration", 0.0)

    # 3. 获取上下文与 Planner 制定的战略计划
    plan = state.get("plan", [])
    remote_file_path = state.get("remote_file_path", "")

    # 将结构化的 Plan 转化为文本指南
    plan_str = "\n".join([f"任务 {i + 1}: {t.description}" for i, t in enumerate(plan)])

    # 4. 加载 Prompt 模板
    scenario = state.get("scenario")
    config = load_prompts_config(step, scenario)
    instruction_template = config.get('executor_instruction')

    # 构造核心 System Prompt
    system_content = instruction_template.format(
        remote_file_path=remote_file_path,
        plan_str=plan_str
    )

    # 5. 组装消息列表 (系统指令 + 对话历史记忆)
    messages = [SystemMessage(content=system_content)] + state.get("messages", [])

    # 6. 调用 LLM (大脑决策阶段)
    print("--- [Subgraph] Executor: 调用大模型中... ---")
    llm_start_time = time.perf_counter()

    # 让大模型自己决定是输出 tool_calls(写代码)，还是直接输出文本(宣布完工)
    response = llm_with_tools.invoke(messages)

    current_llm_time = time.perf_counter() - llm_start_time
    metrics["llm_duration"] += current_llm_time
    print(f"  [Time] Executor LLM 决策耗时: {current_llm_time:.2f} 秒")

    # 7. 返回状态 (直接把 LLM 的回复追加到主 messages 列表中)
    return {
        "messages": [response],
        "metrics": metrics
    }