from langgraph.graph import StateGraph, END, START

from app.core.modeling_custom_subgraph.state import CustomModelingState
from app.nodes.modeling_custom_subgraph.aggregator import create_modeling_aggregator_node
from app.nodes.modeling_custom_subgraph.executor import create_executor_node
from app.nodes.modeling_custom_subgraph.observer import observer_node, observer_router
from app.nodes.modeling_custom_subgraph.planner import planner_node
from app.nodes.modeling_custom_subgraph.replanner import replanner_node
from app.nodes.modeling_custom_subgraph.tool import create_tool_node, tool_router


def build_modeling_custom_subgraph(runtime):
    """Build custom modeling subgraph for local runtime only."""
    workflow = StateGraph(CustomModelingState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", create_executor_node())
    workflow.add_node("tool", create_tool_node(runtime))
    workflow.add_node("observer", observer_node)
    workflow.add_node("replanner", replanner_node)
    workflow.add_node("aggregator", create_modeling_aggregator_node())

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "tool")
    workflow.add_conditional_edges("tool", tool_router, ["executor", "observer"])
    workflow.add_conditional_edges("observer", observer_router, ["executor", "replanner", "aggregator"])
    workflow.add_edge("replanner", "executor")
    workflow.add_edge("aggregator", END)

    return workflow.compile()
