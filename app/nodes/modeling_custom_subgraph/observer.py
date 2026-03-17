import json
import re
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.modeling_custom_subgraph.state import CustomModelingState
from app.core.prompts_config import load_prompts_config
from app.utils.extract_text_from_content import extract_text_from_content
from app.utils.llm_factory import apply_retry, create_llm
from app.utils.terminal_logger import print_block, print_kv, print_list, print_subheader, preview_text

llm = apply_retry(create_llm(use_flash=False))
MAX_REPAIR_ROUNDS = 2


def _parse_observer_output(text: str) -> Dict[str, Any]:
    decision_match = re.search(r"\[DECISION\]\s*(\w+)", text)
    decision = decision_match.group(1) if decision_match else "STOP"

    task_summary_match = re.search(r"\[TASK_SUMMARY\](.*?)(?:\[|$)", text, re.DOTALL)
    task_summary = task_summary_match.group(1).strip() if task_summary_match else ""

    findings_match = re.search(r"\[CONFIRMED_FINDINGS_DELTA\](.*?)(?:\[|$)", text, re.DOTALL)
    findings_text = findings_match.group(1).strip() if findings_match else ""
    findings_delta = [
        line.strip("- ").strip()
        for line in findings_text.split("\n")
        if line.strip().startswith("-")
    ]

    questions_match = re.search(r"\[OPEN_QUESTIONS\](.*?)(?:\[|$)", text, re.DOTALL)
    questions_text = questions_match.group(1).strip() if questions_match else ""
    new_open_questions = [
        line.strip("- ").strip()
        for line in questions_text.split("\n")
        if line.strip().startswith("-")
    ]

    hypotheses_match = re.search(r"\[WORKING_HYPOTHESES\](.*?)(?:\[|$)", text, re.DOTALL)
    hypotheses_text = hypotheses_match.group(1).strip() if hypotheses_match else ""
    new_working_hypotheses = [
        line.strip("- ").strip()
        for line in hypotheses_text.split("\n")
        if line.strip().startswith("-")
    ]

    next_task_match = re.search(r"\[NEXT_TASK\](.*?)(?:\[|$)", text, re.DOTALL)
    next_task = next_task_match.group(1).strip() if next_task_match else ""

    replan_reason_match = re.search(r"\[REPLAN_REASON\](.*?)(?:\[|$)", text, re.DOTALL)
    replan_reason = replan_reason_match.group(1).strip() if replan_reason_match else ""

    stop_reason_match = re.search(r"\[STOP_REASON\](.*?)(?:\[|$)", text, re.DOTALL)
    stop_reason = stop_reason_match.group(1).strip() if stop_reason_match else ""

    return {
        "decision": decision,
        "task_summary": task_summary,
        "findings_delta": findings_delta,
        "new_working_hypotheses": new_working_hypotheses,
        "new_open_questions": new_open_questions,
        "next_task": next_task,
        "replan_reason": replan_reason,
        "stop_reason": stop_reason,
    }


def _build_repair_message(parsed: Dict[str, Any]) -> str:
    decision = parsed["decision"]

    if decision == "FOLLOW_UP" and not parsed["next_task"]:
        return (
            "Your previous response selected FOLLOW_UP but did not include [NEXT_TASK].\n"
            "Reply with ONLY:\n\n"
            "[NEXT_TASK]\n"
            "<a specific executable task for the Executor>"
        )

    if decision == "REPLAN" and not parsed["replan_reason"]:
        return (
            "Your previous response selected REPLAN but did not include [REPLAN_REASON].\n"
            "Reply with ONLY:\n\n"
            "[REPLAN_REASON]\n"
            "<a clear, actionable reason>"
        )

    if decision == "STOP" and not parsed["stop_reason"]:
        return (
            "Your previous response selected STOP but did not include [STOP_REASON].\n"
            "Reply with ONLY:\n\n"
            "[STOP_REASON]\n"
            "<a clear reason explaining why no higher-value next step remains>"
        )

    return ""


def observer_node(state: CustomModelingState) -> Dict[str, Any]:
    print_block("Observer")

    prompts = load_prompts_config("modeling", "custom")
    observer_instruction = prompts["observer_instruction"]

    messages = [SystemMessage(content=observer_instruction)]
    messages.append(HumanMessage(content=f"user_input: {state['user_input']}"))

    observer_history = state.get("observer_history") or []
    for summary in observer_history:
        messages.append(AIMessage(content=summary))

    initial_plan = state.get("initial_plan") or {}
    completed_tasks = state.get("completed_tasks") or []
    remaining_tasks = state.get("remaining_tasks") or []
    followup_playbook = state.get("followup_playbook") or []
    open_questions = state.get("open_questions") or []
    confirmed_findings = state.get("confirmed_findings") or []
    working_hypotheses = state.get("working_hypotheses") or []
    latest_execution = state.get("latest_execution") or {}
    print_kv("current_task", preview_text(state.get("current_task", ""), max_chars=240))
    print_kv("completed_tasks", len(completed_tasks))
    print_kv("remaining_tasks", len(remaining_tasks))
    print_kv("confirmed_findings", len(confirmed_findings))
    print_kv("open_questions", len(open_questions))
    print_kv("observer_history", len(observer_history))
    print_kv("latest_stdout_chars", len(str(latest_execution.get("stdout", ""))))
    print_kv("latest_stderr_chars", len(str(latest_execution.get("stderr", ""))))

    context = f"""initial_plan: {json.dumps(initial_plan, ensure_ascii=False)}

completed_tasks:
{json.dumps(completed_tasks, ensure_ascii=False, indent=2)}

current_task: {state['current_task']}

remaining_tasks:
{json.dumps(remaining_tasks, ensure_ascii=False, indent=2)}

followup_playbook:
{json.dumps(followup_playbook, ensure_ascii=False, indent=2)}

open_questions:
{json.dumps(open_questions, ensure_ascii=False, indent=2)}

confirmed_findings:
{json.dumps(confirmed_findings, ensure_ascii=False, indent=2)}

working_hypotheses:
{json.dumps(working_hypotheses, ensure_ascii=False, indent=2)}

latest_execution:
code: {latest_execution.get('code', '')}
stdout: {latest_execution.get('stdout', '')}
stderr: {latest_execution.get('stderr', '')}
result_text: {latest_execution.get('result_text', '')}
"""

    messages.append(HumanMessage(content=context))
    print_kv("messages_count", len(messages))

    response = llm.invoke(messages)
    text = extract_text_from_content(response.content)
    print_subheader("Observer / Raw Output Preview")
    print(preview_text(text, max_chars=700))
    parsed = _parse_observer_output(text)
    print_subheader("Observer / Parsed Decision")
    print_kv("decision", parsed.get("decision"))
    print_kv("task_summary", preview_text(parsed.get("task_summary", ""), max_chars=260))
    print_list("findings_delta", parsed.get("findings_delta", []), max_items=5)
    print_list("new_open_questions", parsed.get("new_open_questions", []), max_items=5)
    print_list("new_working_hypotheses", parsed.get("new_working_hypotheses", []), max_items=5)

    repair_round = 0
    while repair_round < MAX_REPAIR_ROUNDS:
        repair_message = _build_repair_message(parsed)
        if not repair_message:
            break

        print_subheader("Observer / Repair Round")
        print_kv("round", repair_round + 1)
        print_kv("reason", preview_text(repair_message, max_chars=280))

        messages.append(AIMessage(content=text))
        messages.append(HumanMessage(content=repair_message))

        response = llm.invoke(messages)
        text = extract_text_from_content(response.content)

        if parsed["decision"] == "FOLLOW_UP":
            next_task_match = re.search(r"\[NEXT_TASK\](.*?)(?:\[|$)", text, re.DOTALL)
            if next_task_match:
                parsed["next_task"] = next_task_match.group(1).strip()
        elif parsed["decision"] == "REPLAN":
            replan_reason_match = re.search(r"\[REPLAN_REASON\](.*?)(?:\[|$)", text, re.DOTALL)
            if replan_reason_match:
                parsed["replan_reason"] = replan_reason_match.group(1).strip()

        repair_round += 1

    decision = parsed["decision"]
    print_subheader("Observer / Final Control")
    print_kv("decision", decision)
    if parsed.get("next_task"):
        print_kv("next_task", preview_text(parsed["next_task"], max_chars=240))
    if parsed.get("replan_reason"):
        print_kv("replan_reason", preview_text(parsed["replan_reason"], max_chars=260))
    if parsed.get("stop_reason"):
        print_kv("stop_reason", preview_text(parsed["stop_reason"], max_chars=260))

    result: Dict[str, Any] = {
        "latest_control_signal": decision,
        "confirmed_findings": confirmed_findings + parsed["findings_delta"],
        "working_hypotheses": parsed["new_working_hypotheses"],
        "open_questions": parsed["new_open_questions"],
        "observer_history": observer_history + [parsed["task_summary"]],
    }

    if decision == "CONTINUE":
        result["completed_tasks"] = completed_tasks + [{"description": state["current_task"]}]
        if remaining_tasks:
            result["current_task"] = remaining_tasks[0]["description"]
            result["remaining_tasks"] = remaining_tasks[1:]
        else:
            result["current_task"] = None
            result["remaining_tasks"] = []

    elif decision == "FOLLOW_UP":
        result["completed_tasks"] = completed_tasks + [{"description": state["current_task"]}]
        result["current_task"] = parsed["next_task"] or None

    elif decision == "REPLAN":
        result["replan_reason"] = parsed["replan_reason"]

    elif decision == "STOP":
        result["stop_reason"] = parsed["stop_reason"]

    return result


def observer_router(state: CustomModelingState) -> str:
    signal = state.get("latest_control_signal")
    if signal in ["CONTINUE", "FOLLOW_UP"]:
        return "executor"
    if signal == "REPLAN":
        return "replanner"
    return "aggregator"
