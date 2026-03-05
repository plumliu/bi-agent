from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from ppio_sandbox.code_interpreter import Sandbox

from app.core.agent_states import VizCustomAgentState
from app.core.config import settings
from app.tools.python_script_runner import create_python_script_runner


def create_viz_custom_agent(sandbox: Sandbox, task_id: str, chart_guide: str):
    """创建 Custom Viz Executor Agent"""

    # 构建 system prompt
    system_prompt = f"""
你是一个专业的数据可视化工程师，负责生成单个图表的数据。

【工作流程】
1. 分析任务需求和数据文件
2. 编写 Python 脚本生成图表数据
3. 调用 python_script_runner 工具执行脚本
4. 如果失败，根据错误信息调整脚本并重试
5. 成功后输出"任务完成"

【图表规范】
{chart_guide}

【重要提示】
- 脚本必须将结果保存到 /home/user/viz_output_{task_id}.json
- 数据格式必须严格遵守图表规范
- 如果遇到错误，仔细分析并修正
"""

    # 创建工具
    script_runner = create_python_script_runner(sandbox, task_id)

    # 创建 agent
    agent = create_agent(
        model=ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            temperature=0,
            api_key=settings.OPENAI_API_KEY,
            use_responses_api=settings.USE_RESPONSES_API,
            base_url=settings.OPENAI_API_BASE,
        ),
        tools=[script_runner],
        system_prompt=system_prompt,
        state_schema=VizCustomAgentState,
        name=f"viz_custom_{task_id}"
    )

    return agent
