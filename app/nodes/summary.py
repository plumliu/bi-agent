import os
import json
import yaml
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.core.prompts_config import load_prompts_config
from app.core.state import WorkflowState
from app.utils.extract_text_from_content import extract_text_from_content
from app.utils.llm_factory import create_llm, apply_retry

step = "summary"

# 使用 FLASH 模型
llm = apply_retry(create_llm(use_flash=True))

def summary_node(state: WorkflowState):
    print("--- [Summary] 生成最终报告中 ---")

    scenario = state.get("scenario")
    config = load_prompts_config(step, scenario)

    if not config:
        raise RuntimeError(f"没有配置{scenario}算法场景的提示词")

    # 1. 提取信息
    modeling_summary = state.get("modeling_summary", "")
    user_input = state.get("user_input")
    artifacts = state.get("modeling_artifacts", {})
    artifacts_str = json.dumps(artifacts, ensure_ascii=False)

    # 2. 构造 Viz List String
    viz_config = state.get("viz_config")
    viz_list_str = ""
    if viz_config:
        charts_map = viz_config.get("charts")
        viz_list_items = []
        if charts_map:
            for key, info in charts_map.items():
                title = info.get("title")
                viz_list_items.append(f"- **{key.capitalize()} Chart**: \"{title}\"")

        viz_list_str = "\n".join(viz_list_items) if viz_list_items else "无生成图表"

    # 3. 构造静态 System Prompt
    prompt_template = config.get("summary_instruction")
    context_template = config.get("context_template")

    system_content = config.get("role_definition") + "\n\n" + prompt_template
    system_message = SystemMessage(content=system_content)

    # 4. HumanMessage 包含动态上下文
    context_content = context_template.format(
        user_input=user_input,
        modeling_summary=modeling_summary,
        artifacts_summary=artifacts_str,
        viz_list_str=viz_list_str
    )
    context_message = HumanMessage(content=context_content)

    # 5. 组装 Messages：静态规则 + 动态上下文
    messages = [system_message, context_message]

    print("--- [Summary] 思考中... ---")
    response = llm.invoke(messages)

    # summary_node 直接返回 response (这就是一个自带 content 的 AIMessage)，逻辑是完美的
    return {
        "final_summary": extract_text_from_content(response.content)
    }