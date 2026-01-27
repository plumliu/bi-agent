from typing import Literal

from langgraph.graph import StateGraph, END
from app.core.subgraph.state import CustomModelingState

# 导入具体的节点实现
from app.nodes.subgraph.planner import planner_node
from app.nodes.subgraph.executor import executor_node
from app.nodes.subgraph.reflector import reflector_node


def should_continue(state: CustomModelingState) -> Literal["continue", END]:
    """
    【条件边逻辑】
    判断子图循环是否结束。
    """
    plan = state.get("plan", [])
    current_idx = state.get("current_task_index", 0)

    if current_idx >= len(plan):
        print(f"--- [Subgraph] 所有任务执行完毕 ({len(plan)}/{len(plan)}), 退出子图 ---")
        return END

    return "executor"


def build_modeling_custom_subgraph():
    """
    构建通用建模场景的子图 (Subgraph)。

    流程架构:
    [Start] -> Planner -> Executor -> Reflector --(check)--> [End]
                             ^           |
                             |___________|
    """
    # 1. 初始化子图状态
    workflow = StateGraph(CustomModelingState)

    # 2. 添加节点
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("reflector", reflector_node)

    # 3. 设置入口点
    # 注意：即使是主图第二次调用这个子图，入口依然是 planner
    # 但我们在 planner_node 内部做了 "if plan exists return" 的静默处理
    workflow.set_entry_point("planner")

    # 4. 添加标准边 (Standard Edges)
    # Planner 规划完成后，直接交给 Executor 开始执行第一个任务
    workflow.add_edge("planner", "executor")

    # Executor 执行完代码后，必须交给 Reflector 进行审查
    workflow.add_edge("executor", "reflector")

    # 5. 添加条件边 (Conditional Edges)
    # Reflector 审查结束后，决定是继续循环还是退出
    workflow.add_conditional_edges(
        "reflector",  # 上游节点
        should_continue,  # 路由函数
        {
            "executor": "executor",  # 继续循环：回到 Executor 执行 plan[current_index]
            END: END  # 结束：退出子图，返回主图的下一节点 (Connecting)
        }
    )

    # 6. 编译子图
    return workflow.compile()