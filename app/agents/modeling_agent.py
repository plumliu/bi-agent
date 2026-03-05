from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from ppio_sandbox.code_interpreter import Sandbox

from app.core.agent_states import ModelingAgentState
from app.core.config import settings
from app.core.prompts_config import load_prompts_config
from app.tools.python_interpreter import create_code_interpreter_tool


def create_modeling_agent(sandbox: Sandbox, scenario: str):
    """创建 SOP Modeling Agent"""

    # 加载配置
    config = load_prompts_config("modeling", scenario)

    role_definition = config.get('role_definition')
    instruction = config.get('modeling_instruction')
    code_example = config.get('code_example')

    # 构建 system prompt
    system_prompt = f"""
{role_definition}

【任务指南】
{instruction}

【协议与代码范式 (Protocol)】
在生成最终产物时，必须严格遵守以下代码结构（尤其是文件保存部分）
在执行完 SDK 中的方法得到 result 对象后，不需要再次打印查看 result 对象的内容，直接保存即可。

```python
{code_example}
```

【交互规范：操作前播报】
在你决定调用 `python_interpreter` 工具编写代码之前，你**必须**先在回复中输出一句简短的自然语言（1 到 2 句话即可），告诉用户你正在做什么。
示例："正在为您加载数据并进行初步探查..." 或 "我正在编写代码训练 RFM 聚类模型..."
输出这句话后，再附带你的工具调用指令。

【结束与交付】
你需要足够相信沙盒中的bi_sandbox_sdk！
当你完成所有代码执行并获得满意的分析结果后，只需要输出一句"建模过程结束。"即可。
"""

    # 创建工具
    code_tool = create_code_interpreter_tool(sandbox)

    # 创建 agent
    agent = create_agent(
        model=ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            temperature=0,
            api_key=settings.OPENAI_API_KEY,
            use_responses_api=settings.USE_RESPONSES_API,
        ),
        tools=[code_tool],
        system_prompt=system_prompt,
        state_schema=ModelingAgentState,
        name=f"modeling_{scenario}"
    )

    return agent
