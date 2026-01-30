import os
import json
import yaml
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage

from app.core.prompts_config import load_prompts_config
from app.core.state import AgentState
from app.core.config import settings

step = "summary"

llm = ChatOpenAI(
    model=settings.LLM_FLASH_MODEL_NAME,
    temperature=0.5,
    api_key=settings.OPENAI_API_KEY
)

def summary_node(state: AgentState):
    print("--- [Summary] 生成最终报告中 ---")

    scenario = state.get("scenario")
    config = load_prompts_config(step, scenario)

    if not config:
        raise RuntimeError(f"没有配置{scenario}算法场景的提示词")

    # 1. 提取 Modeling 结论
    modeling_summary = state["modeling_summary"]

    # 2. 构造 Artifacts
    artifacts = state.get("modeling_artifacts")
    artifacts_str = json.dumps(artifacts, ensure_ascii=False)

    # 3. 构造 Viz List String
    # 我们从 Config 里读，Config 里有标题和类型，这就够了
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

    # 4. 填充 Prompt
    prompt_template = config.get("summary_instruction")
    system_prompt = config.get("role_definition") + "\n\n" + prompt_template.format(
        user_input=state.get("user_input"),
        modeling_summary=modeling_summary,
        artifacts_summary=artifacts_str,
        viz_list_str=viz_list_str,
        example_chart_name=example_chart_name
    )

    # 5. 调用 LLM

    messages = [SystemMessage(content=system_prompt)]

    print("--- [Summary] 思考中... ---")
    response = llm.invoke(messages)

    return {
        "final_summary": response.content,
        "messages": [response]
    }