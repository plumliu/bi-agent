from typing import List, Dict, Any, Optional
from app.core.state import AgentState


class CustomVizState(AgentState):
    """
    viz_custom 子图的专用状态
    继承自 AgentState，拥有所有父图字段
    """

    # viz 任务列表（由 planner 生成）
    viz_tasks: Optional[List[Dict[str, Any]]]

    # 收集所有 executor 的输出结果
    # 格式: {"viz_task_1": {"type": "scatter", "data": {...}}, ...}
    viz_results: Optional[Dict[str, Dict[str, Any]]]

    # 时间统计
    viz_metrics: Optional[Dict[str, float]]
