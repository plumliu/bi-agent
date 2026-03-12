import json
import re
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.core.modeling_custom_subgraph.state import CustomModelingState
from app.core.prompts_config import load_prompts_config
from app.utils.extract_text_from_content import extract_text_from_content
from app.utils.llm_factory import create_llm, apply_retry

llm = apply_retry(create_llm(use_flash=False))

# 最多补救 2 轮，属于轻量级修复，不做无限循环
MAX_REPAIR_ROUNDS = 2


def _parse_observer_output(text: str) -> Dict[str, Any]:
    """解析 Observer 的结构化输出"""
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
        "new_open_questions": new_open_questions,
        "next_task": next_task,
        "replan_reason": replan_reason,
        "stop_reason": stop_reason,
    }


def _build_repair_message(parsed: Dict[str, Any]) -> str:
    """根据缺失字段构造补救提示"""
    decision = parsed["decision"]

    if decision == "FOLLOW_UP" and not parsed["next_task"]:
        return (
            "你明明已经给出了控制信号为 FOLLOW_UP，但是你似乎忘记了给出 NEXT_TASK。\n"
            "请严格按照既定结构重新输出完整结果，并补充一个非空、具体、可直接执行的 [NEXT_TASK]。\n"
            "这次输出中，你不需要输出全量回答内容，只需要给出上个回答中遗漏的 [NEXT_TASK]"
        )

    if decision == "REPLAN" and not parsed["replan_reason"]:
        return (
            "你明明已经给出了控制信号为 REPLAN，但是你似乎忘记了给出 REPLAN_REASON。\n"
            "请严格按照既定结构重新输出完整结果，并补充一个非空、明确、具体、可操作的 [REPLAN_REASON]。\n"
            "这次输出中，你不需要输出全量回答内容，只需要给出上个回答中遗漏的 [REPLAN_REASON]"
        )

    return ""


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

    # First call
    response = llm.invoke(messages)
    text = extract_text_from_content(response.content)

    print("=" * 80)
    print("[Observer] LLM原始输出:")
    print(text)
    print("=" * 80)

    parsed = _parse_observer_output(text)

    # 轻量级补救：仅在 FOLLOW_UP 缺 NEXT_TASK 或 REPLAN 缺 REPLAN_REASON 时触发
    repair_round = 0
    while repair_round < MAX_REPAIR_ROUNDS:
        repair_message = _build_repair_message(parsed)
        if not repair_message:
            break

        print(f"--- [Observer] 触发补救回合 #{repair_round + 1}: {parsed['decision']} 缺少必填字段 ---")

        messages.append(AIMessage(content=text))
        messages.append(HumanMessage(content=repair_message))

        response = llm.invoke(messages)
        text = extract_text_from_content(response.content)

        print("=" * 80)
        print(f"[Observer] 补救回合 #{repair_round + 1} 输出:")
        print(text)
        print("=" * 80)

        parsed = _parse_observer_output(text)
        repair_round += 1

    decision = parsed["decision"]
    task_summary = parsed["task_summary"]
    findings_delta = parsed["findings_delta"]
    new_open_questions = parsed["new_open_questions"]

    print(f"--- [Observer] 最终决策: {decision} ---")

    # Update state based on decision
    result = {
        "latest_control_signal": decision,
        "confirmed_findings": confirmed_findings + findings_delta,
        "open_questions": new_open_questions,
        "observer_history": observer_history + [task_summary],
    }

    if decision == "CONTINUE":
        # Move to next task
        result["completed_tasks"] = completed_tasks + [{"description": state["current_task"]}]
        if remaining_tasks:
            result["current_task"] = remaining_tasks[0]["description"]
            result["remaining_tasks"] = remaining_tasks[1:]
        else:
            result["current_task"] = None

    elif decision == "FOLLOW_UP":
        # Insert follow-up task
        result["completed_tasks"] = completed_tasks + [{"description": state["current_task"]}]
        result["current_task"] = parsed["next_task"] or None

    elif decision == "REPLAN":
        # Trigger replanning
        result["replan_reason"] = parsed["replan_reason"]

    elif decision == "STOP":
        # Stop execution
        result["stop_reason"] = parsed["stop_reason"]

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