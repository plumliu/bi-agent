from langchain_core.messages import HumanMessage, SystemMessage

from app.core.prompts_config import load_prompts_config
from app.core.state import WorkflowState
from app.utils.extract_text_from_content import extract_text_from_content
from app.utils.llm_factory import apply_retry, create_llm

step = "summary"
llm = apply_retry(create_llm(use_flash=True))


def summary_node(state: WorkflowState):
    print("--- [Summary] 生成最终报告中 ---")

    scenario = state.get("scenario")
    config = load_prompts_config(step, scenario)
    if not config:
        raise RuntimeError(f"没有配置 {scenario} 场景的提示词")

    modeling_summary = state.get("modeling_summary", "")
    user_input = state.get("user_input", "")

    system_message = SystemMessage(
        content=f"{config.get('role_definition', '')}\n\n{config.get('summary_instruction', '')}".strip()
    )

    context_content = config.get("context_template", "").format(
        user_input=user_input,
        modeling_summary=modeling_summary,
    )
    context_message = HumanMessage(content=context_content)

    response = llm.invoke([system_message, context_message])
    return {"final_summary": extract_text_from_content(response.content)}
