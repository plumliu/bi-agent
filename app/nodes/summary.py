import os
import json
import yaml
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage

from app.core.prompts_config import load_prompts_config
from app.core.state import AgentState
from app.core.config import settings
from app.utils.extract_text_from_content import extract_text_from_content

step = "summary"

llm = ChatOpenAI(
    model=settings.LLM_FLASH_MODEL_NAME,
    temperature=0.25,
    api_key=settings.OPENAI_API_KEY_FLASH,
    use_responses_api=settings.USE_RESPONSES_API_FLASH,
    base_url=settings.OPENAI_API_BASE_FLASH,
)

def summary_node(state: AgentState):
    print("--- [Summary] 生成最终报告中 ---")

    scenario = state.get("scenario")
    config = load_prompts_config(step, scenario)

    if not config:
        raise RuntimeError(f"没有配置{scenario}算法场景的提示词")

    # 1. 提取信息
    modeling_summary = state.get("modeling_summary", "")
    artifacts = state.get("modeling_artifacts", {})
    artifacts_str = json.dumps(artifacts, ensure_ascii=False)

    # 2. 构造 Viz List String
    viz_config = state.get("viz_config")
    viz_list_str = ""
    example_chart_name = ""
    if viz_config:
        charts_map = viz_config.get("charts")
        viz_list_items = []
        if charts_map:
            for key, info in charts_map.items():
                title = info.get("title")
                viz_list_items.append(f"- **{key.capitalize()} Chart**: \"{title}\"")

        viz_list_str = "\n".join(viz_list_items) if viz_list_items else "无生成图表"
        example_chart_name = list(charts_map.keys())[0].capitalize() if charts_map else "Chart"

    # 3. 构造静态 System Prompt (移除 user_input)
    prompt_template = config.get("summary_instruction")
    system_prompt = config.get("role_definition") + "\n\n" + prompt_template.format(
        # user_input 被移除
        modeling_summary=modeling_summary,
        artifacts_summary=artifacts_str,
        viz_list_str=viz_list_str,
        example_chart_name=example_chart_name
    )

    # 4. 组装 Messages：静态规则 + 全局对话历史
    messages = [SystemMessage(content=system_prompt)] + state.get("messages", [])

    print("--- [Summary] 思考中... ---")
    response = llm.invoke(messages)

    # summary_node 直接返回 response (这就是一个自带 content 的 AIMessage)，逻辑是完美的
    return {
        "final_summary": extract_text_from_content(response.content),
        "messages": [response]
    }