from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from openai import APIError
from ppio_sandbox.code_interpreter import Sandbox

from app.core.agent_states import ModelingCustomAgentState
from app.core.config import settings
from app.core.prompts_config import load_prompts_config
from app.tools.python_interpreter import create_code_interpreter_tool


def create_modeling_custom_agent(sandbox: Sandbox, scenario: str):
    """创建 Custom Modeling Executor Agent"""

    # 加载配置
    config = load_prompts_config("modeling", scenario)
    instruction_template = config.get('executor_instruction')

    # 构建 system prompt
    system_prompt = instruction_template

    # 创建工具
    code_tool = create_code_interpreter_tool(sandbox)

    # 创建 agent
    agent = create_agent(
        model=ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            temperature=0,
            api_key=settings.OPENAI_API_KEY,
            use_responses_api=settings.USE_RESPONSES_API,
            max_retries=5,
            timeout=120,
        ),
        tools=[code_tool],
        system_prompt=system_prompt,
        state_schema=ModelingCustomAgentState,
        name=f"modeling_custom_{scenario}"
    )

    # 添加智能重试机制
    agent = agent.with_retry(
        stop_after_attempt=5,
        retry_if_exception_type=(APIError,),
        wait_exponential_jitter=True,
    )

    return agent
