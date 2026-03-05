import json
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from ppio_sandbox.code_interpreter import Sandbox

from app.core.prompts_config import load_prompts_config
from app.core.state import WorkflowState
from app.agents.modeling_agent import create_modeling_agent


def create_modeling_node(sandbox: Sandbox):
    """创建 modeling 节点（工厂函数）"""

    def modeling_node(state: WorkflowState) -> Dict[str, Any]:
        print("--- [Modeling] 思考中... ---")

        scenario = state.get("scenario")
        data_schema = state.get("data_schema")
        remote_file_path = state.get("remote_file_path")

        # 动态创建 agent
        agent = create_modeling_agent(sandbox, scenario)

        # 构建动态上下文
        config = load_prompts_config("modeling", scenario)
        context_template = config.get('context_template')
        context_content = context_template.format(
            remote_file_path=remote_file_path,
            data_schema=json.dumps(data_schema, ensure_ascii=False),
            scenario=scenario
        )

        # 调用 agent
        agent_result = agent.invoke({
            "remote_file_path": remote_file_path,
            "data_schema": data_schema,
            "scenario": scenario,
            "messages": [HumanMessage(content=context_content)]
        })

        # 提取结果
        return {
            "modeling_summary": agent_result.get("modeling_summary", "")
        }

    return modeling_node