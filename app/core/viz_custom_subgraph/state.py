from typing import TypedDict, List, Optional, Dict, Any


class CustomVizState(TypedDict):
    """
    Custom Viz 子图状态（无 messages）
    """
    # 从主图传入
    scenario: str
    file_metadata: List[Dict[str, Any]]

    # Planner 生成
    viz_tasks: Optional[List[Dict[str, Any]]]

    # Send API 注入（每个并发分支）
    viz_task: Optional[Dict[str, Any]]

    # Aggregator 汇总
    viz_results: Optional[List[Dict[str, Any]]]
