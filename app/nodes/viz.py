import json
import os
import yaml
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from app.core.state import AgentState
from app.core.config import settings
from app.core.prompts_config import load_prompts_config

step = "viz"

llm = ChatOpenAI(
    model=settings.LLM_FLASH_MODEL_NAME,
    temperature=0,
    api_key=settings.OPENAI_API_KEY
)

def viz_node(state: AgentState):
    print("--- [Viz] 生成式的配置信息 ---")

    # 1. 获取上下文信息
    scenario = state.get("scenario", "clustering")
    config = load_prompts_config(step, scenario)

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

    print("--- [Viz] 思考中... ---")
    response = llm.invoke(messages)

    # 4. 解析结果
    viz_config = {}
    try:
        content = response.content.replace("```json", "").replace("```", "").strip()
        viz_config = json.loads(content)
        print("--- [Viz] 图表配置生成成功 ---")
    except Exception as e:
        print(f"--- [Viz] JSON转换失败: {e} ---")
        print(f"Raw回答: {response.content}")

    # 5. 更新 State
    return {
        "viz_config": viz_config,
        "messages": [SystemMessage(content="图表配置生成成功, 等待viz_execution节点执行中")]
    }