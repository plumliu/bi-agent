from typing import TypedDict, List, Optional, Dict, Any


class CustomModelingState(TypedDict):
    """
    Custom Modeling 子图状态
    包含 InputState, InternalState, OutputState 三层结构
    """
    # ========== InputState (从主图传入) ==========
    scenario: str
    user_input: str

    # 原始文件路径列表（沙盒内）
    raw_file_paths: List[str]
    # 原始文件名列表
    original_filenames: List[str]

    # 文件元信息列表 (Profiler 输出)
    files_metadata: List[Dict[str, Any]]
    # 合并建议列表 (Profiler 输出，仅多文件时存在)
    merge_recommendations: Optional[List[Dict[str, Any]]]

    # [已废弃] 保留以向后兼容
    remote_file_path: Optional[str]
    data_schema: Optional[Dict[str, Any]]

    # ========== InternalState - Planning ==========
    initial_plan: Optional[Dict[str, Any]]
    remaining_tasks: Optional[List[Dict[str, str]]]
    completed_tasks: Optional[List[Dict[str, str]]]
    current_task: Optional[str]
    followup_playbook: Optional[List[Dict[str, Any]]]

    # ========== InternalState - Execution ==========
    latest_ai_message: Optional[Any]
    latest_execution: Optional[Dict[str, Any]]
    last_error: Optional[Dict[str, Any]]
    execution_trace: Optional[List[Dict[str, Any]]]

    # ========== InternalState - Observer ==========
    latest_control_signal: Optional[str]
    confirmed_findings: Optional[List[str]]
    working_hypotheses: Optional[List[str]]
    open_questions: Optional[List[str]]
    observer_history: Optional[List[str]]
    replan_reason: Optional[str]
    stop_reason: Optional[str]

    # ========== InternalState - Files ==========
    generated_files: Optional[Dict[str, Any]]

    # ========== OutputState (返回给主图) ==========
    modeling_summary: Optional[str]
    generated_data_files: Optional[List[str]]
    file_metadata: Optional[List[Dict[str, Any]]]
    modeling_artifacts: Optional[Dict[str, Any]]
