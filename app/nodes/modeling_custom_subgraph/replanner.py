import json
import re
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.modeling_custom_subgraph.state import CustomModelingState
from app.core.prompts_config import load_prompts_config
from app.utils.extract_text_from_content import extract_text_from_content
from app.utils.llm_factory import apply_retry, create_llm
from app.utils.terminal_logger import print_block, print_kv, print_list, print_subheader, preview_text

llm = apply_retry(create_llm(use_flash=False))


def replanner_node(state: CustomModelingState) -> Dict[str, Any]:
    print_block("Replanner")

    prompts = load_prompts_config("modeling", "custom")
    replanner_instruction = prompts["replanner_instruction"]

    system_message = SystemMessage(content=replanner_instruction)

    latest_execution = state.get("latest_execution")
    execution_context = ""
    if latest_execution:
        execution_context = f"""
latest_execution_code:
{latest_execution.get('code', '')}

latest_execution_stdout:
{latest_execution.get('stdout', '')}

latest_execution_stderr:
{latest_execution.get('stderr', '')}

latest_execution_result_text:
{latest_execution.get('result_text', '')}
"""
    print_kv("current_task", preview_text(state.get("current_task", ""), max_chars=220))
    print_kv("replan_reason", preview_text(state.get("replan_reason", ""), max_chars=260))
    print_kv("completed_tasks", len(state.get("completed_tasks") or []))
    print_kv("remaining_tasks", len(state.get("remaining_tasks") or []))
    print_kv("open_questions", len(state.get("open_questions") or []))
    print_kv("latest_stdout_chars", len(str((latest_execution or {}).get("stdout", ""))))

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

{execution_context}

replan_reason: {state.get('replan_reason', '')}
"""

    human_message = HumanMessage(content=context)

    max_retries = 2
    messages = [system_message, human_message]
    phase_tasks = []
    followup_playbook = state.get("followup_playbook") or []
    print_kv("messages_count", len(messages))

    for attempt in range(max_retries):
        print_subheader("Replanner / LLM Attempt")
        print_kv("attempt", f"{attempt + 1}/{max_retries}")
        response = llm.invoke(messages)
        raw_text = extract_text_from_content(response.content)
        print_kv("response_preview", preview_text(raw_text, max_chars=520))

        match = re.search(r"```(?:json)?(.*?)```", raw_text, re.DOTALL)
        if match:
            content_str = match.group(1).strip()
        else:
            start_idx = raw_text.find("{")
            end_idx = raw_text.rfind("}")
            if start_idx != -1 and end_idx != -1:
                content_str = raw_text[start_idx : end_idx + 1]
            else:
                content_str = raw_text.strip()

        try:
            parsed = json.loads(content_str)
            phase_tasks = parsed["phase_tasks"]
            followup_playbook = parsed.get("followup_playbook", followup_playbook)
            print_subheader("Replanner / Parsed Plan")
            print_kv("phase_tasks", len(phase_tasks))
            print_list(
                "phase_task_descriptions",
                [task.get("description", "") for task in phase_tasks],
                max_items=6,
            )
            print_kv("followup_playbook_items", len(followup_playbook))
            break
        except (json.JSONDecodeError, KeyError) as e:
            print_kv("parse_error", str(e))
            if attempt < max_retries - 1:
                messages.append(response)
                messages.append(
                    HumanMessage(
                        content=(
                            "Your previous output is not valid JSON with the required keys. "
                            f"Error: {e}. Output strict JSON only."
                        )
                    )
                )
            else:
                print_subheader("Replanner / Fallback")
                print("Failed to produce valid replanning JSON, returning empty remaining tasks.")
                return {
                    "remaining_tasks": [],
                    "current_task": None,
                    "replan_reason": None,
                }

    return {
        "remaining_tasks": phase_tasks[1:] if len(phase_tasks) > 1 else [],
        "current_task": phase_tasks[0]["description"] if phase_tasks else None,
        "followup_playbook": followup_playbook,
        "replan_reason": None,
    }
