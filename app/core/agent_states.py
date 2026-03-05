"""
Agent 状态定义
继承自 LangChain v1.0 官方 AgentState
"""
from typing import Dict, Any, List
from typing_extensions import NotRequired
from langchain.agents.middleware.types import AgentState


class ModelingAgentState(AgentState[Any]):
    """SOP Modeling Agent 状态"""
    remote_file_path: NotRequired[str]
    data_schema: NotRequired[Dict[str, Any]]
    scenario: NotRequired[str]
    modeling_summary: NotRequired[str]


class ModelingCustomAgentState(AgentState[Any]):
    """Custom Modeling Executor Agent 状态"""
    remote_file_path: NotRequired[str]
    plan: NotRequired[List[Dict[str, Any]]]
    current_task_index: NotRequired[int]
    task_results: NotRequired[List[Dict[str, Any]]]
    modeling_summary: NotRequired[str]
    generated_data_files: NotRequired[List[str]]
    file_metadata: NotRequired[List[Dict[str, Any]]]


class VizAgentState(AgentState[Any]):
    """SOP Viz Agent 状态"""
    scenario: NotRequired[str]
    data_schema: NotRequired[Dict[str, Any]]
    modeling_artifacts: NotRequired[Dict[str, Any]]
    modeling_summary: NotRequired[str]
    viz_config: NotRequired[Dict[str, Any]]


class VizCustomAgentState(AgentState[Any]):
    """Custom Viz Executor Agent 状态"""
    viz_task: NotRequired[Dict[str, Any]]
    file_metadata: NotRequired[List[Dict[str, Any]]]
    viz_result: NotRequired[Dict[str, Any]]
