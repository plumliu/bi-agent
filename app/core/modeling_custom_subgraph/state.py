from typing import TypedDict, List, Optional, Dict, Any


class CustomModelingState(TypedDict):
    """
    Custom Modeling 子图状态
    包含 InputState, InternalState, OutputState 三层结构
    """
    # ========== InputState (从主图传入) ==========
    scenario: str
    remote_file_path: str
    data_schema: Dict[str, Any]
    user_input: str

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
