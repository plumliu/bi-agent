from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.core.modeling_custom_subgraph.state import CustomModelingState
from app.core.prompts_config import load_prompts_config
from app.tools.python_interpreter import create_code_interpreter_tool
from app.utils.llm_factory import apply_retry, create_llm
from app.utils.terminal_logger import (
    preview_code,
    preview_text,
    print_block,
    print_kv,
    print_list,
    print_subheader,
)


def create_executor_node():
    """Executor node for generating the next python_interpreter tool call."""
    llm_with_tools = apply_retry(
        create_llm(use_flash=False).bind_tools([create_code_interpreter_tool()])
    )

    def executor_node(state: CustomModelingState) -> Dict[str, Any]:
        print_block("Executor")

        prompts = load_prompts_config("modeling", "custom")
        executor_instruction = prompts["executor_instruction"]

        messages = [SystemMessage(content=executor_instruction)]

        history = _coerce_ai_tool_history(state.get("execution_trace") or [])
        messages.extend(history)
        print_kv("current_task", preview_text(state.get("current_task", ""), max_chars=240))
        print_kv("history_pairs", len(history) // 2)

        last_error = state.get("last_error")
        if last_error:
            ai_msg = state.get("latest_ai_message")
            if ai_msg:
                messages.append(ai_msg)
                error_tool_msg = last_error.get("tool_message")
                if error_tool_msg:
                    messages.append(error_tool_msg)
            print_subheader("Executor / Retry Context")
            print_kv("has_last_error", True)
            print_kv(
                "error_tool_message_preview",
                preview_text(getattr(last_error.get("tool_message"), "content", ""), max_chars=320),
            )
        else:
            print_kv("has_last_error", False)

        completed_tasks = state.get("completed_tasks") or []
        confirmed_findings = state.get("confirmed_findings") or []
        working_hypotheses = state.get("working_hypotheses") or []
        print_kv("completed_tasks", len(completed_tasks))
        print_kv("confirmed_findings", len(confirmed_findings))
        print_kv("working_hypotheses", len(working_hypotheses))
        if completed_tasks:
            print_list(
                "completed_task_preview",
                [t.get("description", "") for t in completed_tasks],
                max_items=4,
            )

        completed_str = "\n".join([f"- {t['description']}" for t in completed_tasks]) if completed_tasks else "none"
        findings_str = "\n".join([f"- {f}" for f in confirmed_findings]) if confirmed_findings else "none"
        hypotheses_str = "\n".join([f"- {h}" for h in working_hypotheses]) if working_hypotheses else "none"

        context = f"""current_task: {state['current_task']}

completed_tasks:
{completed_str}

confirmed_findings:
{findings_str}

working_hypotheses:
{hypotheses_str}
"""

        if last_error:
            context += "\n\nThe previous code cell failed. Rewrite code for the same current_task."

        messages.append(HumanMessage(content=context))
        print_kv("messages_count", len(messages))

        ai_response = llm_with_tools.invoke(messages)
        if not ai_response.tool_calls:
            raise RuntimeError("Executor: LLM did not return any tool call")

        tool_call = ai_response.tool_calls[0]
        if isinstance(tool_call, dict):
            args = tool_call.get("args", {}) or {}
            call_name = tool_call.get("name", "python_interpreter")
        else:
            args = getattr(tool_call, "args", {}) or {}
            call_name = getattr(tool_call, "name", "python_interpreter")

        code = str(args.get("code", ""))
        print_subheader("Executor / Generated Tool Call")
        print_kv("tool_name", call_name)
        print_kv("code_chars", len(code))
        print_kv("code_lines", len(code.splitlines()) if code else 0)
        if code:
            print(preview_code(code))

        return {
            "latest_ai_message": ai_response,
            "last_error": None,
        }

    return executor_node


def _coerce_ai_tool_history(history):
    if len(history) % 2 != 0:
        raise RuntimeError(
            f"Executor: execution_trace length must be even (AI/Tool pairs), current={len(history)}"
        )

    repaired = []
    for i in range(0, len(history), 2):
        ai_candidate = history[i]
        tool_candidate = history[i + 1]

        if not isinstance(ai_candidate, AIMessage):
            ai_candidate = AIMessage(
                content=getattr(ai_candidate, "content", None),
                additional_kwargs=getattr(ai_candidate, "additional_kwargs", {}) or {},
                response_metadata=getattr(ai_candidate, "response_metadata", {}) or {},
                tool_calls=getattr(ai_candidate, "tool_calls", []) or [],
                invalid_tool_calls=getattr(ai_candidate, "invalid_tool_calls", []) or [],
                usage_metadata=getattr(ai_candidate, "usage_metadata", None),
                id=getattr(ai_candidate, "id", None),
                name=getattr(ai_candidate, "name", None),
            )

        if not isinstance(tool_candidate, ToolMessage):
            status = getattr(tool_candidate, "status", "success") or "success"
            if status not in {"success", "error"}:
                status = "success"

            tool_call_id = getattr(tool_candidate, "tool_call_id", None)
            if not tool_call_id:
                raise RuntimeError(
                    f"Executor: history[{i + 1}] missing tool_call_id, cannot coerce to ToolMessage"
                )

            tool_candidate = ToolMessage(
                content=getattr(tool_candidate, "content", ""),
                tool_call_id=str(tool_call_id),
                status=status,
                artifact=getattr(tool_candidate, "artifact", None),
                additional_kwargs=getattr(tool_candidate, "additional_kwargs", {}) or {},
                response_metadata=getattr(tool_candidate, "response_metadata", {}) or {},
                id=getattr(tool_candidate, "id", None),
                name=getattr(tool_candidate, "name", None),
            )

        if not ai_candidate.tool_calls:
            raise RuntimeError(f"Executor: history[{i}] has no tool_calls")
        if not tool_candidate.tool_call_id:
            raise RuntimeError(f"Executor: history[{i + 1}] has no tool_call_id")

        repaired.extend([ai_candidate, tool_candidate])

    return repaired
