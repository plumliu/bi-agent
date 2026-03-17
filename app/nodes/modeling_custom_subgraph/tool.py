from typing import Any, Dict

from langchain_core.messages import AIMessage, ToolMessage

from app.core.modeling_custom_subgraph.state import CustomModelingState


def create_tool_node(runtime):
    """Execute python_interpreter code via local kernel runtime."""

    def tool_node(state: CustomModelingState) -> Dict[str, Any]:
        print("--- [Tool] 执行代码 ---")

        ai_message = state["latest_ai_message"]
        if not ai_message.tool_calls:
            raise RuntimeError("Tool: latest_ai_message has no tool_calls")

        tool_call = ai_message.tool_calls[0]
        if isinstance(tool_call, dict):
            args = tool_call.get("args", {})
            tool_call_id = tool_call["id"]
        else:
            args = getattr(tool_call, "args", {})
            tool_call_id = tool_call.id

        if "code" in args:
            code = args["code"]
        elif len(args) == 1:
            code = list(args.values())[0]
        else:
            tool_message = ToolMessage(
                content=(
                    f"Error: invalid tool call args={args}. "
                    "Please call python_interpreter with a single 'code' argument."
                ),
                tool_call_id=tool_call_id,
                status="error",
            )
            return {
                "latest_execution": None,
                "last_error": {"tool_message": tool_message},
            }

        try:
            execution = runtime.execute(code)
        except Exception as e:
            tool_message = ToolMessage(
                content=f"Error: local kernel runtime failure: {e}",
                tool_call_id=tool_call_id,
                status="error",
            )
            return {
                "latest_execution": None,
                "last_error": {"tool_message": tool_message},
            }
        latest_execution = {
            "code": code,
            "stdout": execution.get("stdout", ""),
            "stderr": execution.get("stderr", ""),
            "result_text": execution.get("result_text", ""),
            "error": execution.get("error"),
        }

        if latest_execution["error"]:
            err = latest_execution["error"]
            error_text = (
                f"Error: {err.get('name', 'ExecutionError')}: {err.get('value', '')}\n"
                f"Traceback:\n{err.get('traceback', '')}\n"
                f"Stderr:\n{latest_execution['stderr']}"
            )
            tool_message = ToolMessage(
                content=error_text,
                tool_call_id=tool_call_id,
                status="error",
            )
            return {
                "latest_execution": latest_execution,
                "last_error": {"tool_message": tool_message},
            }

        output_text = "".join(
            [
                latest_execution.get("stdout", ""),
                latest_execution.get("stderr", ""),
                latest_execution.get("result_text", ""),
            ]
        ).strip()
        if not output_text:
            output_text = "[SYSTEM] Cell executed successfully with no visible output."

        tool_message = ToolMessage(content=output_text, tool_call_id=tool_call_id)

        assert isinstance(ai_message, AIMessage), type(ai_message)
        return {
            "latest_execution": latest_execution,
            "last_error": None,
            "execution_trace": [ai_message, tool_message],
        }

    return tool_node


def tool_router(state: CustomModelingState) -> str:
    return "executor" if state.get("last_error") else "observer"
