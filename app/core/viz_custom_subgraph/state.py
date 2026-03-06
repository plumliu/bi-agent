from typing import TypedDict, List, Optional, Dict, Any


class CustomVizState(TypedDict):
    """
    Custom Viz 子图状态（无 messages）
    """
    # 从主图传入
    scenario: str
    user_input: str  # 添加：用户原始需求
    modeling_summary: str  # 添加：建模摘要
    generated_data_files: List[str]  # 添加：生成的数据文件列表
    file_metadata: List[Dict[str, Any]]

    # Planner 生成
    viz_tasks: Optional[List[Dict[str, Any]]]
    viz_metrics: Optional[Dict[str, Any]]  # 添加：可视化指标

    # Send API 注入（每个并发分支）
    viz_task: Optional[Dict[str, Any]]

    # Aggregator 汇总
    viz_results: Optional[List[Dict[str, Any]]]
    viz_data: Optional[Dict[str, Any]]  # 添加：最终可视化数据
