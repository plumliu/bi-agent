import json
import os
import yaml
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from app.core.state import AgentState
from app.core.config import settings


def load_viz_config(scenario: str):
    base_path = os.path.join(os.path.dirname(__file__), "../prompts/scenarios")
    path = os.path.join(base_path, f"viz_{scenario}.yaml")
    if not os.path.exists(path):
        print(f"[Warning] Viz prompt not found: {path}")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

llm = ChatOpenAI(
    model=settings.LLM_FLASH_MODEL_NAME,
    temperature=0,
    api_key=settings.OPENAI_API_KEY
)

def viz_node(state: AgentState):
    print("--- [Step 2] Viz Config Node: Generative Configuration ---")

    # 1. 获取上下文信息
    scenario = state.get("scenario", "clustering")
    config = load_viz_config(scenario)

    # 获取最新的 Schema (由 fetch_artifacts 更新过的)
    data_schema = state.get("data_schema")

    # 获取 Modeling 阶段的产物 (artifacts)
    artifacts = state.get("modeling_artifacts", {})

    # 获取 Modeling 阶段的自然语言摘要
    # 策略：取最后一条 AI 消息的内容作为摘要
    modeling_summary = state["modeling_summary"]

    # 2. 构造 Prompt
    # 注意：使用 str(json.dumps(...)) 确保 Prompt 里是合法的 JSON 字符串表示，方便 LLM 阅读
    prompt_template = config.get("viz_instruction")
    system_prompt = config.get("role_definition") + "\n\n" + prompt_template.format(
        user_input=state.get("user_input"),
        modeling_summary=modeling_summary,
        columns=data_schema,
        artifacts=json.dumps(artifacts, ensure_ascii=False)  # 注入全量 Artifacts
    )

    # 3. 调用 LLM
    messages = [SystemMessage(content=system_prompt)]

    print("--- [Viz] Thinking... ---")
    response = llm.invoke(messages)

    # 4. 解析结果
    viz_config = {}
    try:
        content = response.content.replace("```json", "").replace("```", "").strip()
        viz_config = json.loads(content)
        print("--- [Viz] Configuration Generated Successfully ---")
    except Exception as e:
        print(f"--- [Viz] JSON Parsing Failed: {e} ---")
        print(f"Raw Output: {response.content}")

    # 5. 更新 State
    return {
        "viz_config": viz_config,
        "messages": [SystemMessage(content="Viz Configuration Generated Successfully, 等待viz_execution执行中")]
    }