from typing import List, Optional
from app.core.state import AgentState  # 导入主图状态
from app.core.subgraph.task import Task


class CustomModelingState(AgentState):
    """
    通用建模子图 (Modeling Custom Subgraph) 的专用状态。
    继承自 AgentState
    """

    # 动态任务列表
    # Agent 会根据执行结果不断更新这个列表
    plan: List[Task]

    # 当前正在执行的任务索引 (0-based)
    # Python 使用: plan[current_task_index]
    # Prompt 使用: plan[current_task_index].id
    current_task_index: int

    # 思考草稿本 (Scratchpad)
    # 例如："上一步代码报错了，可能是列名不对，下一步我需要先打印列名确认。"
    scratchpad: List[str]

    # 重试计数器
    # 针对同一个 Task，如果连续失败超过 3 次，可能需要人工介入或跳过
    retry_count: int