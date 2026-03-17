from typing import Literal

from langgraph.graph import StateGraph, END, START

from app.core.state import WorkflowState
from app.graph.modeling_custom_workflow import build_modeling_custom_subgraph
from app.nodes.profiler import profiler_node
from app.nodes.router import router_node
from app.nodes.summary import summary_node


def route_after_router(state: WorkflowState) -> Literal["modeling_custom", END]:
    scenario = state.get("scenario")
    if scenario == "custom":
        return "modeling_custom"
    return END


def build_graph(runtime, session_dir: str, session_id: str):
    """Build modeling-only graph: profiler -> router -> modeling_custom -> summary."""
    _ = (session_dir, session_id)

    workflow = StateGraph(WorkflowState)

    workflow.add_node("profiler", profiler_node)
    workflow.add_node("router", router_node)
    workflow.add_node("modeling_custom", build_modeling_custom_subgraph(runtime))
    workflow.add_node("summary", summary_node)

    workflow.add_edge(START, "profiler")
    workflow.add_edge("profiler", "router")

    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "modeling_custom": "modeling_custom",
            END: END,
        },
    )

    workflow.add_edge("modeling_custom", "summary")
    workflow.add_edge("summary", END)

    return workflow.compile()
