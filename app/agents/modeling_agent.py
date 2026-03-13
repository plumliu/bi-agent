from langchain.agents import create_agent
from ppio_sandbox.code_interpreter import Sandbox

from app.core.agent_states import ModelingAgentState
from app.core.prompts_config import load_prompts_config
from app.tools.python_interpreter import create_code_interpreter_tool
from app.utils.llm_factory import create_llm, apply_retry


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

【结束与交付】
你需要足够相信沙盒中的bi_sandbox_sdk！
当你完成所有代码执行并获得满意的分析结果后，只需要输出一句"建模过程结束。"即可。
"""

    # 创建工具
    code_tool = create_code_interpreter_tool(sandbox)

    # 创建 LLM（主模型）
  llm = create_llm(use_flash=False)

    # 创建 agent（先创建 agent，再 apply_retry）
    agent = create_agent(
        model=llm,
        tools=[code_tool],
        system_prompt=system_prompt,
        state_schema=ModelingAgentState,
        name=f"modeling_{scenario}"
    )

    # 添加智能重试机制
    agent = apply_retry(agent)

    return agent
