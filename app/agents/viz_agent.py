from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from app.core.agent_states import VizAgentState
from app.core.config import settings
from app.core.prompts_config import load_prompts_config
from app.tools.viz_execution_tool import viz_execution_tool


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

    # 创建 agent
    agent = create_agent(
        model=ChatOpenAI(
            model=settings.LLM_FLASH_MODEL_NAME,
            temperature=0,
            api_key=settings.OPENAI_API_KEY_FLASH,
            use_responses_api=settings.USE_RESPONSES_API_FLASH,
            base_url=settings.OPENAI_API_BASE_FLASH,
        ),
        tools=[viz_execution_tool],
        system_prompt=system_prompt,
        state_schema=VizAgentState,
        name=f"viz_{scenario}"
    )

    return agent
