from langgraph.graph import StateGraph, END
from ppio_sandbox.code_interpreter import Sandbox

from app.core.modeling_custom_subgraph.state import CustomModelingState
from app.nodes.modeling_custom_subgraph.planner import planner_node
from app.nodes.modeling_custom_subgraph.executor import create_executor_node


def build_modeling_custom_subgraph(sandbox: Sandbox):
    """
    构建 Custom Modeling 子图（使用 create_agent）
    """
    workflow = StateGraph(CustomModelingState)

    # 注册节点
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", create_executor_node(sandbox))

    # 设置入口点
    workflow.set_entry_point("planner")

    # 简化流程：planner → executor → END
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", END)

    return workflow.compile()