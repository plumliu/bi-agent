import os
import json
import yaml
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from app.core.state import AgentState
from app.core.config import settings


def load_summary_config(scenario: str):
    # 动态加载对应场景的 yaml
    base_path = os.path.join(os.path.dirname(__file__), "../prompts/scenarios")
    path = os.path.join(base_path, f"summary_{scenario}.yaml")

    # 降级策略：如果没有特定 summary，可以用一个默认的
    if not os.path.exists(path):
        print(f"[Warning] Summary config for {scenario} not found.")
        return None

    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

llm = ChatOpenAI(
    model=settings.LLM_FLASH_MODEL_NAME,
    temperature=0.5,
    api_key=settings.OPENAI_API_KEY
)

def summary_node(state: AgentState):
    print("--- [Step 4] Summary Node: Generating Final Report ---")

    scenario = state.get("scenario")
    config = load_summary_config(scenario)

    if not config:
        # 如果没有配置，返回一个简单结束语
        return {"final_summary": "Analysis completed. Please check the charts."}

    # 1. 提取 Modeling 结论 (从历史消息中找)
    modeling_summary = state["modeling_summary"]

    # 2. [关键] 构造 Artifacts
    artifacts = state.get("modeling_artifacts")
    artifacts_str = json.dumps(artifacts, ensure_ascii=False)

    # 3. [关键] 构造 Viz List String (不读取 viz_data.json!)
    # 我们从 Config 里读，Config 里有标题和类型，这就够了
    viz_config = state.get("viz_config")
    charts_map = viz_config.get("charts")

    viz_list_items = []
    if charts_map:
        for key, info in charts_map.items():
            title = info.get("title")
            viz_list_items.append(f"- **{key.capitalize()} Chart**: \"{title}\"")

    viz_list_str = "\n".join(viz_list_items) if viz_list_items else "无生成图表"

    # 4. 填充 Prompt
    prompt_template = config.get("summary_instruction", "")
    system_prompt = config.get("role_definition", "") + "\n\n" + prompt_template.format(
        user_input=state.get("user_input"),
        modeling_summary=modeling_summary,
        artifacts_summary=artifacts_str,
        viz_list_str=viz_list_str,
        example_chart_name=list(charts_map.keys())[0].capitalize() if charts_map else "Chart"  # 动态示例
    )

    # 5. 调用 LLM

    messages = [SystemMessage(content=system_prompt)]

    print("--- [Summary] Thinking... ---")
    response = llm.invoke(messages)

    return {
        "final_summary": response.content,
        "messages": [response]
    }