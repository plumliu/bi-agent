import operator
from typing import Literal, List
from functools import partial

from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
# from e2b_code_interpreter import Sandbox
from ppio_sandbox.code_interpreter import Sandbox

from app.core.state import AgentState

# 导入节点
from app.nodes.auto_etl import auto_etl_node
from app.nodes.router import router_node
from app.nodes.modeling import modeling_node  # SOP Modeling
from app.nodes.fetch_artifacts import create_fetch_artifacts_node
from app.nodes.viz import viz_node
from app.nodes.viz_execution import viz_execution_node
from app.nodes.summary import summary_node

#    导入子图构建器
from app.graph.modeling_custom_workflow import build_modeling_custom_subgraph
from app.graph.viz_custom_workflow import build_viz_custom_subgraph


# --- 路由逻辑 ---

def route_after_router(state: AgentState) -> Literal["modeling", "modeling_custom", END]:
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


def route_after_modeling_sop(state: AgentState) -> Literal["tools", "fetch_artifacts"]:
    """
    SOP Modeling 之后的路由
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "fetch_artifacts"


def route_after_modeling_custom(state: AgentState) -> Literal["viz_custom"]:
    """
    Custom Modeling 之后的路由：直接进入 viz_custom 子图
    """
    print("--- [Router] Custom Modeling 完成，进入 viz_custom 子图 ---")
    return "viz_custom"


def route_after_fetch(state: AgentState) -> Literal["viz", "summary"]:
    """
       Fetch Artifacts 之后的路由（仅用于 SOP 路径）
    """
    scenario = state.get("scenario")

    # SOP 场景：继续走 Viz 流程
    return "viz"


def route_after_viz_execution(state: AgentState) -> Literal["viz", "summary"]:
    if state.get("viz_success"):
        return "summary"
    else:
        print("--- [Router] Viz execute failed, looping back to Config Agent ---")
        return "viz"


# --- 构建图 ---

def build_graph(tools: List[BaseTool], sandbox: Sandbox):
    """
    构建完整的工作流图 (SOP + Custom 双模态)。
    """
    workflow = StateGraph(AgentState)

    # ============================================================
    # 1. 注册节点
    # ============================================================
    workflow.add_node("auto_etl", partial(auto_etl_node, sandbox=sandbox))
    workflow.add_node("router", router_node)

    # 分支 A: SOP Modeling 节点
    workflow.add_node("modeling", partial(modeling_node, tools=tools))
    workflow.add_node("tools", ToolNode(tools))  # SOP 专用工具节点

    # 分支 B: Custom Modeling 子图
    # 注意：build_modeling_custom_subgraph 返回的是一个 CompiledGraph，可以直接作为节点
    workflow.add_node("modeling_custom", build_modeling_custom_subgraph(sandbox))

    # 分支 C: Custom Viz 子图
    workflow.add_node("viz_custom", build_viz_custom_subgraph(sandbox))

    # 公共节点
    workflow.add_node("fetch_artifacts", create_fetch_artifacts_node(sandbox))
    workflow.add_node("viz", viz_node)
    workflow.add_node("viz_execution", viz_execution_node)
    workflow.add_node("summary", summary_node)

    # ============================================================
    # 2. 定义边
    # ============================================================

    # 1. 预处理: Start -> Auto-ETL -> Router
    workflow.add_edge(START, "auto_etl")
    workflow.add_edge("auto_etl", "router")

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
    workflow.add_conditional_edges(
        "modeling",
        route_after_modeling_sop,
        {
            "tools": "tools",
            "fetch_artifacts": "fetch_artifacts"
        }
    )
    workflow.add_edge("tools", "modeling")

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

    # 3. 汇聚与再次分流: Fetch -> Viz (仅 SOP 路径)
    workflow.add_conditional_edges(
        "fetch_artifacts",
        route_after_fetch,
        {
            "viz": "viz"  # SOP 去绘图
        }
    )

    # 4. SOP 后半程: Viz -> Execution -> Summary
    workflow.add_edge("viz", "viz_execution")
    workflow.add_conditional_edges(
        "viz_execution",
        route_after_viz_execution,
        {
            "viz": "viz",
            "summary": "summary"
        }
    )

    # 5. 终点
    workflow.add_edge("summary", END)

    app = workflow.compile()
    return app