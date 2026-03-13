from langchain.agents import create_agent

from app.core.agent_states import VizAgentState
from app.core.prompts_config import load_prompts_config
from app.tools.viz_execution_tool import viz_execution_tool
from app.utils.llm_factory import create_llm, apply_retry


def create_viz_agent(scenario: str):
    """创建 SOP Viz Agent"""

    # 加载配置
    config = load_prompts_config("viz", scenario)

    role_definition = config.get('role_definition')
    instruction = config.get('viz_instruction')

    # 构建 system prompt
    system_prompt = f"""
{role_definition}

【工作流程】
1. 分析建模产物和数据schema
2. 生成可视化配置（JSON格式）
3. 调用 viz_execution 工具验证配置
4. 如果失败，根据错误信息调整配置并重试
5. 成功后输出"可视化配置已完成。"

【任务指南】
{instruction}

【重要提示】
- 配置必须是有效的JSON格式
- 调用工具前，先输出你的配置思路
- 如果工具返回错误，仔细分析错误原因并调整
"""

    # 创建 LLM（FLASH 模型）
    llm = create_llm(use_flash=True)

    # 创建 agent（先创建 agent，再 apply_retry）
    agent = create_agent(
        model=llm,
        tools=[viz_execution_tool],
        system_prompt=system_prompt,
        state_schema=VizAgentState,
        name=f"viz_{scenario}"
    )

    # 添加智能重试机制
    agent = apply_retry(agent)

    return agent
