from typing import Literal

from langgraph.graph import StateGraph, END
from app.core.subgraph.state import CustomModelingState

# from e2b_code_interpreter import Sandbox
from ppio_sandbox.code_interpreter import Sandbox

from app.nodes.subgraph.planner import planner_node
from app.nodes.subgraph.executor import create_executor_node
from app.nodes.subgraph.reflector import reflector_node
from app.nodes.subgraph.finalizer import create_finalizer_node


def should_continue(state: CustomModelingState) -> Literal["executor", "finalizer"]:
    """
    【条件边逻辑】
    判断子图循环是否结束。
    """
    plan = state.get("plan", [])
    current_idx = state.get("current_task_index", 0)

    # 如果所有任务执行完毕
    if current_idx >= len(plan):
        print(f"--- [Subgraph] 所有任务执行完毕 ({len(plan)}/{len(plan)}), 进入收尾阶段 ---")
        return "finalizer"

    # 否则继续循环
    return "executor"


def build_modeling_custom_subgraph(sandbox: Sandbox):
    """
    构建通用建模场景的子图 (Subgraph)。
    Args:
        sandbox: 全局沙盒实例，需注入到 executor 和 finalizer
    """
    # 1. 初始化子图状态
    workflow = StateGraph(CustomModelingState)

    # 2. 添加节点
    workflow.add_node("planner", planner_node)

    # [修改] 使用工厂函数注入 sandbox
    workflow.add_node("executor", create_executor_node(sandbox))

    workflow.add_node("reflector", reflector_node)

    # [修改] 使用工厂函数注入 sandbox
    workflow.add_node("finalizer", create_finalizer_node(sandbox))

    # 3. 设置入口点
    workflow.set_entry_point("planner")

    # 4. 添加标准边
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "reflector")

    # 5. 添加条件边
    # Reflector -> Executor (循环) OR Finalizer (收尾)
    workflow.add_conditional_edges(
        "reflector",  # 上游节点
        should_continue,  # 路由函数
        {
            "executor": "executor",  # 继续循环
            "finalizer": "finalizer"  # 收尾
        }
    )

    # 6. 结束边
    # Finalizer -> END (退出子图)
    workflow.add_edge("finalizer", END)

    # 7. 编译子图
    return workflow.compile()