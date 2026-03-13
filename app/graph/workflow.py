from typing import Literal
from functools import partial

from langgraph.graph import StateGraph, END, START
from ppio_sandbox.code_interpreter import Sandbox

from app.core.state import WorkflowState

# 导入节点
from app.nodes.profiler import profiler_node
from app.nodes.router import router_node
from app.nodes.modeling import create_modeling_node  # SOP Modeling
from app.nodes.fetch_artifacts import create_fetch_artifacts_node
from app.nodes.viz import create_viz_node
from app.nodes.summary import summary_node

#    导入子图构建器
from app.graph.modeling_custom_workflow import build_modeling_custom_subgraph
from app.graph.viz_custom_workflow import build_viz_custom_subgraph


# --- 路由逻辑 ---

def route_after_router(state: WorkflowState) -> Literal["modeling", "modeling_custom", END]:
    """
    Router 之后的路由：决定走 SOP 流程还是 Custom 流程
    """
    scenario = state.get("scenario")

    if scenario == "custom":
        return "modeling_custom"
    elif scenario == "unknown":
        # 未知场景，可以直接结束或者走默认
        return END

    # 其他情况 (cluster, rfm 等) 走 SOP Modeling
    return "modeling"


def route_after_modeling_custom(state: WorkflowState) -> Literal["viz_custom"]:
    """
    Custom Modeling 之后的路由：直接进入 viz_custom 子图
    """
    print("--- [Router] Custom Modeling 完成，进入 viz_custom 子图 ---")
    return "viz_custom"


# --- 构建图 ---

def build_graph(sandbox: Sandbox):
    """
    构建完整的工作流图 (SOP + Custom 双模态)。
    """
    workflow = StateGraph(WorkflowState)

    # ============================================================
    # 1. 注册节点
    # ============================================================
    workflow.add_node("profiler", partial(profiler_node, sandbox=sandbox))
    workflow.add_node("router", router_node)

    # 分支 A: SOP Modeling 节点（使用 create_agent）
    workflow.add_node("modeling", create_modeling_node(sandbox))

    workflow.add_node("fetch_artifacts", create_fetch_artifacts_node(sandbox))
    workflow.add_node("viz", create_viz_node())

    # 分支 B: Custom Modeling 子图
    # 注意：build_modeling_custom_subgraph 返回的是一个 CompiledGraph，可以直接作为节点
    workflow.add_node("modeling_custom", build_modeling_custom_subgraph(sandbox))
    workflow.add_node("viz_custom", build_viz_custom_subgraph(sandbox))

    # 公共节点

    workflow.add_node("summary", summary_node)

    # ============================================================
    # 2. 定义边
    # ============================================================

    # 1. 预处理: Start -> Profiler -> Router
    workflow.add_edge(START, "profiler")
    workflow.add_edge("profiler", "router")

    # 2. 核心分流: Router -> (SOP) / (Custom)
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "modeling": "modeling",  # SOP 路径
            "modeling_custom": "modeling_custom",  # Custom 路径
            END: END
        }
    )

    # --- 路径 A: SOP 流程 ---
    # create_agent 自动处理 ReAct 循环，直接连接到 fetch_artifacts
    workflow.add_edge("modeling", "fetch_artifacts")

    # --- 路径 B: Custom 流程 ---
    # 子图内部有自己的循环，当子图返回时(END)，说明建模完成
    workflow.add_conditional_edges(
        "modeling_custom",
        route_after_modeling_custom,
        {
            "viz_custom": "viz_custom"  # Custom 路径：建模 → 可视化
        }
    )

    # viz_custom 子图完成后，直接去 summary
    workflow.add_edge("viz_custom", "summary")

    # 3. SOP 路径：Fetch -> Viz -> Summary
    workflow.add_edge("fetch_artifacts", "viz")
    workflow.add_edge("viz", "summary")

    # 5. 终点
    workflow.add_edge("summary", END)

    app = workflow.compile()
    return app