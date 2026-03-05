from typing import List
from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from ppio_sandbox.code_interpreter import Sandbox
from app.core.viz_custom_subgraph.state import CustomVizState
from app.nodes.viz_custom_subgraph.planner import viz_planner_node
from app.nodes.viz_custom_subgraph.executor import viz_executor_node
from app.nodes.viz_custom_subgraph.aggregator import viz_aggregator_node


def create_executor_sends(state: CustomVizState) -> List[Send]:
    """
    【路由函数】
    为每个 viz_task 创建一个独立的 executor 实例
    使用 Send API 实现并发执行
    """
    viz_tasks = state.get("viz_tasks", [])

    if not viz_tasks:
        # 没有任务，直接跳到 aggregator
        print("--- [Viz Subgraph] Router: 没有 viz_tasks，跳过 executor ---")
        return []

    print(f"--- [Viz Subgraph] Router: 创建 {len(viz_tasks)} 个并发 executor ---")

    # 为每个 task 创建一个 Send 对象
    # Send 会创建独立的执行分支，每个分支接收不同的 viz_task
    sends = []
    for task in viz_tasks:
        # 创建一个新的状态副本，注入当前的 viz_task
        send_state = {
            **state,  # 继承所有父状态
            "viz_task": task  # 注入当前任务
        }
        sends.append(Send("viz_executor", send_state))

    return sends


def build_viz_custom_subgraph(sandbox: Sandbox):
    """
    构建 viz_custom 子图（支持并发执行）

    流程：
    planner → [executor_1, executor_2, ...] → aggregator → END
              (并发执行，使用 Send API)
    """
    workflow = StateGraph(CustomVizState)

    # 1. 注册节点
    workflow.add_node("viz_planner", viz_planner_node)
    workflow.add_node("viz_executor", partial(viz_executor_node, sandbox=sandbox))
    workflow.add_node("viz_aggregator", viz_aggregator_node)

    # 2. 设置入口
    workflow.set_entry_point("viz_planner")

    # 3. Planner → Executors (并发分发)
    workflow.add_conditional_edges(
        "viz_planner",
        create_executor_sends,
        ["viz_executor"]  # 目标节点
    )

    # 4. Executors → Aggregator (自动汇聚)
    # 当所有 executor 完成后，自动进入 aggregator
    workflow.add_edge("viz_executor", "viz_aggregator")

    # 5. Aggregator → END
    workflow.add_edge("viz_aggregator", END)

    return workflow.compile()
