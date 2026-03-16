import json
import re
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

from app.core.modeling_custom_subgraph.state import CustomModelingState
from app.core.prompts_config import load_prompts_config
from app.utils.extract_text_from_content import extract_text_from_content
from app.utils.llm_factory import create_llm, apply_retry

# Initialize LLM (主模型使用Anthropic)
llm = apply_retry(create_llm(use_flash=False))


def replanner_node(state: CustomModelingState) -> Dict[str, Any]:
    """Replanner 节点：重新规划剩余任务"""
    print("--- [Replanner] 重新规划剩余任务 ---")

    # Load replanner instruction
    prompts = load_prompts_config("modeling", "custom")
    replanner_instruction = prompts["replanner_instruction"]

    # Construct messages
    system_message = SystemMessage(content=replanner_instruction)

    # Extract latest execution context
    latest_execution = state.get('latest_execution')
    execution_context = ""
    if latest_execution:
        execution_context = f"""
The final execution in the round that triggered replanning (this may be key evidence for what caused the replanning):
code:
{latest_execution.get('code', '')}

stdout:
{latest_execution.get('stdout', '')}
"""

    context = f"""user_input: {state['user_input']}

initial_plan: {json.dumps(state.get('initial_plan'), ensure_ascii=False, indent=2)}

completed_tasks:
{json.dumps(state.get('completed_tasks'), ensure_ascii=False, indent=2)}

remaining_tasks:
{json.dumps(state.get('remaining_tasks'), ensure_ascii=False, indent=2)}

confirmed_findings:
{json.dumps(state.get('confirmed_findings'), ensure_ascii=False, indent=2)}

open_questions:
{json.dumps(state.get('open_questions'), ensure_ascii=False, indent=2)}

generated_files:
{json.dumps(state.get('generated_files'), ensure_ascii=False, indent=2)}

{execution_context}

replan_reason: {state.get('replan_reason', '')}
"""

    human_message = HumanMessage(content=context)

    # Call LLM with retry loop for JSON parse failures
    max_retries = 2
    messages = [system_message, human_message]
    phase_tasks = []
    followup_playbook = state.get("followup_playbook") or []

    for attempt in range(max_retries):
        response = llm.invoke(messages)
        raw_text = extract_text_from_content(response.content)

        match = re.search(r"```(?:json)?(.*?)```", raw_text, re.DOTALL)
        if match:
            content_str = match.group(1).strip()
        else:
            start_idx = raw_text.find('{')
            end_idx = raw_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                content_str = raw_text[start_idx:end_idx + 1]
            else:
                content_str = raw_text.strip()

        try:
            parsed = json.loads(content_str)
            phase_tasks = parsed["phase_tasks"]
            followup_playbook = parsed.get("followup_playbook", followup_playbook)
            break
        except (json.JSONDecodeError, KeyError) as e:
            print(f"--- [Replanner] JSON 解析失败 (尝试 {attempt + 1}/{max_retries}): {e} ---")
            if attempt < max_retries - 1:
                messages.append(response)
                messages.append(HumanMessage(content=f"Your previous output JSON format is invalid and failed to parse: {e} \n Please output strictly valid JSON only, without any extra text, Markdown, or code block markers, and ensure all strings are properly closed. Please output again:\n"))
            else:
                print("--- [Replanner] 多次重试失败，返回空任务列表 ---")
                return {
                    "remaining_tasks": [],
                    "current_task": None,
                    "replan_reason": None,
                }

    print(f"--- [Replanner] 重新规划了 {len(phase_tasks)} 个任务 ---")
    for i, task in enumerate(phase_tasks, 1):
        print(f"  [Task {i}] {task['description']}")

    # Update state
    return {
        "remaining_tasks": phase_tasks[1:] if len(phase_tasks) > 1 else [],
        "current_task": phase_tasks[0]["description"] if phase_tasks else None,
        "followup_playbook": followup_playbook,
        "replan_reason": None,
    }
