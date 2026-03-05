import json
import os
import re

import yaml
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.core.state import AgentState
from app.core.config import settings
from app.core.prompts_config import load_prompts_config
from app.utils.extract_text_from_content import extract_text_from_content

step = "viz"

llm = ChatOpenAI(
    model=settings.LLM_FLASH_MODEL_NAME,
    temperature=0,
    api_key=settings.OPENAI_API_KEY_FLASH,
    use_responses_api=settings.USE_RESPONSES_API_FLASH,
    base_url=settings.OPENAI_API_BASE_FLASH,
)

def viz_node(state: AgentState):
    print("--- [Viz] 生成式的配置信息 ---")

    # 1. 获取上下文信息
    scenario = state.get("scenario", "clustering")
    config = load_prompts_config(step, scenario)

    # 获取最新的 Schema 和产物
    data_schema = state.get("data_schema")
    artifacts = state.get("modeling_artifacts", {})
    modeling_summary = state.get("modeling_summary", "")

    # 2. 构造静态 System Prompt
    prompt_template = config.get("viz_instruction")
    context_template = config.get("context_template")

    system_content = config.get("role_definition") + "\n\n" + prompt_template
    system_message = SystemMessage(content=system_content)

    # 3. HumanMessage 包含动态上下文
    context_content = context_template.format(
        modeling_summary=modeling_summary,
        columns=data_schema,
        artifacts=json.dumps(artifacts, ensure_ascii=False)
    )
    context_message = HumanMessage(content=context_content)

    # 4. 组装 Messages：静态规则 + 动态上下文 + 全局对话历史
    messages = [system_message, context_message] + state.get("messages", [])

    print("--- [Viz] 思考中... ---")
    response = llm.invoke(messages)

    # 4. 解析结果
    viz_config = {}
    try:
        raw_text = extract_text_from_content(response.content)

        # 策略 1：使用正则表达式提取 Markdown 代码块中的内容
        match = re.search(r"```(?:json)?(.*?)```", raw_text, re.DOTALL)
        if match:
            content = match.group(1).strip()
        else:
            # 策略 2：如果没有使用 Markdown 代码块，尝试暴力截取第一个 { 和最后一个 } 之间的内容
            start_idx = raw_text.find('{')
            end_idx = raw_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                content = raw_text[start_idx:end_idx + 1]
            else:
                content = raw_text.strip()

        viz_config = json.loads(content)
        print("--- [Viz] 图表配置生成成功 ---")

    except Exception as e:
        print(f"--- [Viz] JSON转换失败: {e} ---")
        print(f"Raw回答: {response.content}")
        return {
            "viz_config": None,  # 明确置为 None，触发下游错误拦截
            "messages": [AIMessage(content=f"[系统报错] Viz 节点未能生成合法的 JSON 配置。")]
        }

    # 5. 更新 State (修正为 AIMessage)
    return {
        "viz_config": viz_config,
        "messages": [AIMessage(content="[系统汇报] 图表配置生成成功，等待执行。")]
    }