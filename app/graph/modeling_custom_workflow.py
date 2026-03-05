from typing import Literal
from functools import partial

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.core.modeling_custom_subgraph.state import CustomModelingState
from ppio_sandbox.code_interpreter import Sandbox

from app.nodes.modeling_custom_subgraph.planner import planner_node
from app.nodes.modeling_custom_subgraph.executor import executor_node
from app.tools.python_interpreter import create_code_interpreter_tool


def should_continue(state: CustomModelingState) -> Literal["tools", END]:
    """
    【条件边逻辑】
    判断 Agent 是否还需要调用工具。
    """
    messages = state.get("messages", [])
    if not messages:
        return END  # 防御性编程

    last_message = messages[-1]

    # 如果大模型在最新回复中要求调用沙盒工具写代码
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("--- [Subgraph] Router: 发现工具调用请求，进入沙盒执行 ---")
        return "tools"

    # 如果大模型没有输出工具调用，说明它认为所有的计划任务都已解决，直接结束
    print("--- [Subgraph] Router: 任务全部完成，子图结束 ---")
    return END


def build_modeling_custom_subgraph(sandbox: Sandbox):
    """
    构建通用建模场景的子图 (纯正 Graph-level ReAct 架构)
    """
    workflow = StateGraph(CustomModelingState)

    # 1. 实例化真实的工具 (供官方 ToolNode 使用)
    python_tool = create_code_interpreter_tool(sandbox)

    # 2. 注册节点 (使用 partial 注入 sandbox 依赖)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", partial(executor_node, sandbox=sandbox))

    # 【高光时刻】：直接使用官方 ToolNode 接管沙盒执行！它会自动处理错误并将结果封装为 ToolMessage 传回给 LLM
    workflow.add_node("tools", ToolNode([python_tool]))

    # 3. 设置入口点
    workflow.set_entry_point("planner")

    # 4. 规划完毕后，进入执行器
    workflow.add_edge("planner", "executor")

    # 大脑思考后，必须做路由判断
    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "tools": "tools",  # 分支 A: 发现要写代码，去沙盒
            END: END  # 分支 B: 任务全部搞定，直接结束
        }
    )

    # 工具执行完毕后，必须无条件回到大脑，让大脑看执行结果（形成 ReAct 闭环）
    workflow.add_edge("tools", "executor")

    return workflow.compile()