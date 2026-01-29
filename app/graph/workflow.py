import operator
from typing import Literal, List
from functools import partial  # [新增] 用于绑定 sandbox 参数

from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
# from e2b_code_interpreter import Sandbox
from ppio_sandbox.code_interpreter import Sandbox

from app.core.state import AgentState
from app.nodes.auto_etl import auto_etl_node  # [新增] 导入 Auto-ETL 节点
from app.nodes.router import router_node
from app.nodes.modeling import modeling_node
from app.nodes.fetch_artifacts import create_fetch_artifacts_node
from app.nodes.viz import viz_node
from app.nodes.viz_execution import viz_execution_node
from app.nodes.summary import summary_node


# --- 路由逻辑 ---

def route_after_router(state: AgentState) -> Literal["modeling", END]:
    scenario = state.get("scenario")
    if scenario == "unknown":
        return END
    return "modeling"


def route_after_modeling(state: AgentState) -> Literal["tools", "fetch_artifacts"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "fetch_artifacts"


def route_after_viz_execution(state: AgentState) -> Literal["viz", "summary"]:
    if state.get("viz_success"):
        # 成功 -> 进入总结报告阶段
        return "summary"
    else:
        # 失败 -> 回退到 viz 节点重试
        print("--- [Router] Viz execute failed, looping back to Config Agent ---")
        return "viz"


# --- 构建图 ---

def build_graph(tools: List[BaseTool], sandbox: Sandbox):
    """
    构建工作流图。
    """
    workflow = StateGraph(AgentState)

    # ============================================================
    # 1. 注册节点
    # ============================================================

    workflow.add_node("auto_etl", partial(auto_etl_node, sandbox=sandbox))
    workflow.add_node("router", router_node)
    workflow.add_node("modeling", modeling_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("fetch_artifacts", create_fetch_artifacts_node(sandbox))
    workflow.add_node("viz", viz_node)
    workflow.add_node("viz_execution", viz_execution_node)
    workflow.add_node("summary", summary_node)

    # ============================================================
    # 2. 定义边 (TEST MODE: 仅测试 Auto-ETL)
    # ============================================================

    # [修改] Start -> Auto-ETL
    workflow.add_edge(START, "auto_etl")

    # [修改] Auto-ETL -> END (测试阻断，暂时不进入 Router)
    workflow.add_edge("auto_etl", END)

    # ============================================================
    # 下方是完整的生产链路，暂时注释掉以进行隔离测试
    # ============================================================

    """
    # Start -> Router (原逻辑)
    # workflow.add_edge(START, "router") # 注释掉

    # Router -> Modeling / End
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "modeling": "modeling",
            END: END
        }
    )

    # Modeling -> Tools / Fetch
    workflow.add_conditional_edges(
        "modeling",
        route_after_modeling,
        {
            "tools": "tools",
            "fetch_artifacts": "fetch_artifacts"
        }
    )

    workflow.add_edge("tools", "modeling")

    # Fetch -> Viz (Config)
    workflow.add_edge("fetch_artifacts", "viz")

    # Viz -> Viz Execution
    workflow.add_edge("viz", "viz_execution")

    # Viz Execution -> Summary or Viz (Loop)
    workflow.add_conditional_edges(
        "viz_execution",
        route_after_viz_execution,
        {
            "viz": "viz",          # 失败重试
            "summary": "summary"   # 成功则去总结
        }
    )

    # Summary -> END
    workflow.add_edge("summary", END)
    """

    app = workflow.compile()
    return app