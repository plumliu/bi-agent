import os
import yaml
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.prompts_config import load_prompts_config
from app.core.state import AgentState
from app.core.config import settings
from app.utils.extract_text_from_content import extract_text_from_content

# 1. 当前阶段
step = "modeling"

# 2. 初始化 Modeling LLM (不再全局绑定工具)
llm = ChatOpenAI(
    model=settings.LLM_MODEL_NAME,
    temperature=0,
    api_key=settings.OPENAI_API_KEY,
    use_responses_api=settings.USE_RESPONSES_API,
)


# 3. Modeling Node 核心逻辑 (新增 tools 依赖注入参数)
def modeling_node(state: AgentState, tools: list):
    print("--- [Modeling] 思考中... ---")

    # 【核心升级】：动态绑定从主图注入进来的真实 tools
    llm_with_tools = llm.bind_tools(tools)

    scenario = state.get("scenario")
    data_schema = state.get("data_schema")
    remote_file_path = state.get("remote_file_path")

    # A. 动态加载配置
    config = load_prompts_config(step, scenario)

    role_definition = config.get('role_definition')
    instruction = config.get('modeling_instruction')
    code_example = config.get('code_example')
    context_template = config.get('context_template')

    # B. 构建 System Prompt (完全静态，跨请求 KV Cache 命中)
    system_content = f"""
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
    system_message = SystemMessage(content=system_content)

    # C. HumanMessage 包含动态上下文
    context_content = context_template.format(
        remote_file_path=remote_file_path,
        data_schema=data_schema,
        scenario=scenario
    )
    context_message = HumanMessage(content=context_content)

    # 构造消息列表：静态系统规则 + 动态上下文 + 原生对话历史
    messages = [system_message, context_message] + state.get("messages", [])

    # C. 调用 LLM
    response = llm_with_tools.invoke(messages)

    # D. 返回更新 (将 LLM 的包含 content 和 tool_calls 的原生消息追加到记录中)
    updates = {"messages": [response]}

    # 判断：如果这次响应没有调用工具，说明 Agent 认为任务结束了，正在输出最终结论
    if not response.tool_calls:
        print("--- [Modeling] 建模成功，正在输出结论... ---")
        updates["modeling_summary"] = extract_text_from_content(response.content)

    return updates