from typing import List, Dict
from app.core.state import AgentState
from app.core.modeling_custom_subgraph.task import Task

class CustomModelingState(AgentState):
    """
    通用建模子图 (Modeling Custom Subgraph) 的专用状态。
    继承自 AgentState (这意味着它天生拥有 messages 列表，用于记忆对话历史)
    """

    # 战略任务指南 (由 Planner 生成，供 Executor 全局参考)
    plan: List[Task]

    # 时间统计 (用于性能监控)
    metrics: Dict[str, float]