from pathlib import Path

from langgraph.graph import StateGraph, END
from ppio_sandbox.code_interpreter import Sandbox

from app.core.modeling_custom_subgraph.state import CustomModelingState
from app.nodes.modeling_custom_subgraph.planner import planner_node
from app.nodes.modeling_custom_subgraph.executor import create_executor_node
from app.nodes.modeling_custom_subgraph.tool import create_tool_node, tool_router
from app.nodes.modeling_custom_subgraph.observer import observer_node, observer_router
from app.nodes.modeling_custom_subgraph.replanner import replanner_node
from app.nodes.modeling_custom_subgraph.aggregator import create_modeling_aggregator_node


def build_modeling_custom_subgraph(sandbox: Sandbox):
    """
    构建 Custom Modeling 子图（新架构）
    拓扑: planner → executor → tool → observer → {executor, replanner, aggregator}
    """
    # Upload helper functions to sandbox
    helpers_path = Path(__file__).parent.parent / "helpers" / "modeling_helpers.py"
    helpers_code = helpers_path.read_text(encoding='utf-8')
    sandbox.files.write("/home/user/helpers.py", helpers_code)
    sandbox.run_code("from helpers import *")
    print("--- [Workflow] Helper 函数已注入到 sandbox ---")

    # Build graph
    workflow = StateGraph(CustomModelingState)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", create_executor_node(sandbox))
    workflow.add_node("tool", create_tool_node(sandbox))
    workflow.add_node("observer", observer_node)
    workflow.add_node("replanner", replanner_node)
    workflow.add_node("aggregator", create_modeling_aggregator_node(sandbox))

    # Set entry point
    workflow.set_entry_point("planner")

    # Add edges
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "tool")
    workflow.add_conditional_edges("tool", tool_router, ["executor", "observer"])
    workflow.add_conditional_edges("observer", observer_router, ["executor", "replanner", "aggregator"])
    workflow.add_edge("replanner", "executor")
    workflow.add_edge("aggregator", END)

    return workflow.compile()
