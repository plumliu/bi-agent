import json
from typing import Dict, Any
from langchain_core.messages import HumanMessage

from app.core.state import WorkflowState
from app.core.prompts_config import load_prompts_config
from app.agents.viz_agent import create_viz_agent


def create_viz_node():
    """创建 viz 节点（工厂函数）"""

    def viz_node(state: WorkflowState) -> Dict[str, Any]:
        print("--- [Viz] 生成可视化配置 ---")

        scenario = state.get("scenario", "clustering")
        data_schema = state.get("data_schema")
        artifacts = state.get("modeling_artifacts", {})
        modeling_summary = state.get("modeling_summary", "")

        # 动态创建 agent
        agent = create_viz_agent(scenario)

        # 构建动态上下文
        config = load_prompts_config("viz", scenario)
        context_template = config.get('context_template')
        context_content = context_template.format(
            modeling_summary=modeling_summary,
            columns=data_schema,
            artifacts=json.dumps(artifacts, ensure_ascii=False)
        )

        # 调用 agent
        agent_result = agent.invoke({
            "scenario": scenario,
            "data_schema": data_schema,
            "modeling_artifacts": artifacts,
            "modeling_summary": modeling_summary,
            "messages": [HumanMessage(content=context_content)]
        })

        # Agent 成功完成（viz_execution_tool 已保存 viz_data.json）
        print("--- [Viz] 可视化配置已完成 ---")
        return {
            "viz_success": True
        }

    return viz_node