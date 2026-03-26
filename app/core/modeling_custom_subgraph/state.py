from typing import TypedDict, List, Optional, Dict, Any, Annotated

from langchain_core.messages import AnyMessage, AIMessage
from langgraph.graph.message import add_messages


class CustomModelingState(TypedDict):
    """Custom modeling subgraph state."""

    # Input from parent graph
    scenario: str
    user_input: str
    raw_file_paths: List[str]
    original_filenames: List[str]
    files_metadata: List[Dict[str, Any]]
    merge_recommendations: Optional[List[Dict[str, Any]]]

    # Planning
    initial_plan: Optional[Dict[str, Any]]
    remaining_tasks: Optional[List[Dict[str, str]]]
    completed_tasks: Optional[List[Dict[str, str]]]
    current_task: Optional[str]
    followup_playbook: Optional[List[Dict[str, Any]]]

    # Execution
    latest_ai_message: Optional[AIMessage]
    latest_execution: Optional[Dict[str, Any]]
    last_error: Optional[Dict[str, Any]]
    execution_trace: Annotated[list[AnyMessage], add_messages]

    # Observation control
    latest_control_signal: Optional[str]
    confirmed_findings: Optional[List[str]]
    observer_history: Optional[List[str]]
    replan_reason: Optional[str]
    stop_reason: Optional[str]

    # Output back to parent graph
    modeling_summary: Optional[str]
