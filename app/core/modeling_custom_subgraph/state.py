from typing import TypedDict, List, Optional, Dict, Any


class CustomModelingState(TypedDict):
    """
    Custom Modeling 子图状态（无 messages）
    """
    # 从主图传入
    scenario: str
    remote_file_path: str
    data_schema: Dict[str, Any]
    user_input: str

    # Planner 生成
    plan: Optional[List[Dict[str, Any]]]
    metrics: Optional[Dict[str, Any]]

    # Executor 生成（返回给主图）
    modeling_summary: Optional[str]
    generated_data_files: Optional[List[str]]
    file_metadata: Optional[List[Dict[str, Any]]]
