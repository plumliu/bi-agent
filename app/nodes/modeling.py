import os
import yaml
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from app.core.prompts_config import load_prompts_config
from app.core.state import AgentState
from app.core.config import settings


# 1. 当前阶段
step = "modeling"

# 2. 定义工具存根
@tool("python_interpreter")
def python_interpreter(code: str):
    """
    Python 代码执行环境。具备数据分析库 (pandas, numpy, scikit-learn 等)，还有一个封装好的强大算法库 bi_sandbox_sdk。
    使用 print() 输出结果。
    """
    pass


# 3. 初始化 Modeling LLM
llm = ChatOpenAI(
    model=settings.LLM_MODEL_NAME,
    temperature=0,
    api_key=settings.OPENAI_API_KEY
)

llm_with_tools = llm.bind_tools([python_interpreter])


# 4. Modeling Node 核心逻辑
def modeling_node(state: AgentState):
    print("--- [Modeling] 思考中... ---")

    scenario = state.get("scenario")
    data_schema = state.get("data_schema")
    remote_file_path = state.get("remote_file_path")
    user_input = state.get("user_input")

    # A. 动态加载配置
    config = load_prompts_config(step, scenario)

    instruction = config.get('modeling_instruction')
    code_example = config.get('code_example')


    # B. 构建 System Prompt
    system_message_content = f"""
    {config.get('role_definition')}

    你当前的任务场景是: {str(scenario).upper()}

    【环境信息】
    数据路径: '{remote_file_path}' (已上传，请直接使用 pd.read_csv 读取此路径)
    数据结构: 
    {data_schema}
    
    用户问题:
    {user_input}
    
    【任务指南】
    {instruction}

    【协议与代码范式 (Protocol)】
    在生成最终产物时，必须严格遵守以下代码结构（尤其是文件保存部分）
    在执行完 SDK 中的方法拿到 result 对象后，不需要再次打印查看 result 对象的内容，直接保存即可。

    ```python
    {code_example}
    ```

    【结束与交付】
    当你完成所有代码执行并获得满意的分析结果后，你需要输出最后一条自然语言回复。
    这条回复将作为【分析摘要】传递给可视化专家和决策者，注意这部分篇幅不要太长，1到2句话。
    """

    # 构造消息列表
    messages = [SystemMessage(content=system_message_content)] + state["messages"]

    # C. 调用 LLM
    response = llm_with_tools.invoke(messages)

    # D. 返回更新
    updates = {"messages": [response]}

    # 判断：如果这次响应没有调用工具，说明 Agent 认为任务结束了，正在输出最终结论
    if not response.tool_calls:
        print("--- [Modeling] 建模成功，正在输出结论... ---")
        updates["modeling_summary"] = response.content

    return updates