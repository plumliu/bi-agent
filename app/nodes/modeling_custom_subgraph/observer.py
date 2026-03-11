import json
import re
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.core.modeling_custom_subgraph.state import CustomModelingState
from app.core.prompts_config import load_prompts_config
from app.utils.extract_text_from_content import extract_text_from_content
from app.utils.llm_factory import create_llm, apply_retry

# Initialize LLM (主模型使用Anthropic)
llm = apply_retry(create_llm(use_flash=False))


def observer_node(state: CustomModelingState) -> Dict[str, Any]:
    """Observer 节点：观察执行结果并决定下一步行动"""
    print("--- [Observer] 观察执行结果 ---")

    # Load observer instruction
    prompts = load_prompts_config("modeling", "custom")
    observer_instruction = prompts["observer_instruction"]

    # Construct messages array
    messages = [SystemMessage(content=observer_instruction)]

    # Add user input as first HumanMessage
    messages.append(HumanMessage(content=f"用户需求: {state['user_input']}"))

    # Add observer history
    observer_history = state.get("observer_history") or []
    for summary in observer_history:
        messages.append(AIMessage(content=summary))

    # Construct current observation context
    initial_plan = state.get("initial_plan") or {}
    completed_tasks = state.get("completed_tasks") or []
    remaining_tasks = state.get("remaining_tasks") or []
    followup_playbook = state.get("followup_playbook") or []
    open_questions = state.get("open_questions") or []
    confirmed_findings = state.get("confirmed_findings") or []
    generated_files = state.get("generated_files") or {}
    latest_execution = state.get("latest_execution") or {}

    context = f"""初始计划: {json.dumps(initial_plan, ensure_ascii=False)}

已完成任务:
{json.dumps(completed_tasks, ensure_ascii=False, indent=2)}

当前任务: {state['current_task']}

剩余任务:
{json.dumps(remaining_tasks, ensure_ascii=False, indent=2)}

追问手册:
{json.dumps(followup_playbook, ensure_ascii=False, indent=2)}

问题池:
{json.dumps(open_questions, ensure_ascii=False, indent=2)}

已确认发现:
{json.dumps(confirmed_findings, ensure_ascii=False, indent=2)}

当前文件:
{json.dumps(generated_files, ensure_ascii=False, indent=2)}

上一轮执行:
代码: {latest_execution.get('code', '')}
输出: {latest_execution.get('stdout', '')}
"""

    messages.append(HumanMessage(content=context))

    # Call LLM
    response = llm.invoke(messages)
    text = extract_text_from_content(response.content)

    # Print observer output
    print("=" * 80)
    print("[Observer] LLM原始输出:")
    print(text)
    print("=" * 80)

    # Parse structured output
    decision_match = re.search(r'\[DECISION\]\s*(\w+)', text)
    decision = decision_match.group(1) if decision_match else "STOP"

    task_summary_match = re.search(r'\[TASK_SUMMARY\](.*?)(?:\[|$)', text, re.DOTALL)
    task_summary = task_summary_match.group(1).strip() if task_summary_match else ""

    findings_match = re.search(r'\[CONFIRMED_FINDINGS_DELTA\](.*?)(?:\[|$)', text, re.DOTALL)
    findings_text = findings_match.group(1).strip() if findings_match else ""
    findings_delta = [line.strip('- ').strip() for line in findings_text.split('\n') if line.strip().startswith('-')]

    questions_match = re.search(r'\[OPEN_QUESTIONS\](.*?)(?:\[|$)', text, re.DOTALL)
    questions_text = questions_match.group(1).strip() if questions_match else ""
    new_open_questions = [line.strip('- ').strip() for line in questions_text.split('\n') if line.strip().startswith('-')]

    print(f"--- [Observer] 决策: {decision} ---")

    # Update state based on decision
    result = {
        "latest_control_signal": decision,
        "confirmed_findings": confirmed_findings + findings_delta,
        "open_questions": new_open_questions,
        "observer_history": observer_history + [task_summary]
    }

    if decision == "CONTINUE":
        # Move to next task
        result["completed_tasks"] = completed_tasks + [{"description": state['current_task']}]
        if remaining_tasks:
            result["current_task"] = remaining_tasks[0]["description"]
            result["remaining_tasks"] = remaining_tasks[1:]
        else:
            result["current_task"] = None

    elif decision == "FOLLOW_UP":
        # Insert follow-up task
        next_task_match = re.search(r'\[NEXT_TASK\](.*?)(?:\[|$)', text, re.DOTALL)
        next_task = next_task_match.group(1).strip() if next_task_match else None

        result["completed_tasks"] = completed_tasks + [{"description": state['current_task']}]
        result["current_task"] = next_task

    elif decision == "REPLAN":
        # Trigger replanning
        replan_reason_match = re.search(r'\[REPLAN_REASON\](.*?)(?:\[|$)', text, re.DOTALL)
        replan_reason = replan_reason_match.group(1).strip() if replan_reason_match else ""
        result["replan_reason"] = replan_reason

    elif decision == "STOP":
        # Stop execution
        stop_reason_match = re.search(r'\[STOP_REASON\](.*?)(?:\[|$)', text, re.DOTALL)
        stop_reason = stop_reason_match.group(1).strip() if stop_reason_match else ""
        result["stop_reason"] = stop_reason

    return result


def observer_router(state: CustomModelingState) -> str:
    """Observer 节点路由函数"""
    signal = state.get("latest_control_signal")
    if signal in ["CONTINUE", "FOLLOW_UP"]:
        return "executor"
    elif signal == "REPLAN":
        return "replanner"
    elif signal == "STOP":
        return "aggregator"
    return "aggregator"  # Default fallback
