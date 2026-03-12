import json
from typing import Dict, Any

from langchain_core.messages import ToolMessage
from ppio_sandbox.code_interpreter import Sandbox

from app.core.modeling_custom_subgraph.state import CustomModelingState


def create_tool_node(sandbox: Sandbox):
    """创建 Tool 节点（工厂函数）"""

    def tool_node(state: CustomModelingState) -> Dict[str, Any]:
        print("--- [Tool] 执行代码 ---")

        ai_message = state["latest_ai_message"]

        # Debug: print tool_calls structure
        print(f"--- [Tool DEBUG] tool_calls: {ai_message.tool_calls}")

        if not ai_message.tool_calls:
            raise RuntimeError("Tool: ai_message 没有 tool_calls")

        tool_call = ai_message.tool_calls[0]
        # Extract args and tool_call_id
        if isinstance(tool_call, dict):
            args = tool_call.get("args", {})
            tool_call_id = tool_call["id"]
        else:
            args = getattr(tool_call, "args", {})
            tool_call_id = tool_call.id

        # Smart code extraction: tool only has one param, match by key name with fallback
        if "code" in args:
            code = args["code"]
        elif len(args) == 1:
            actual_key = list(args.keys())[0]
            code = list(args.values())[0]
            print(f"--- [Tool] 警告: LLM 使用了非标准参数名 '{actual_key}'，已自动适配 ---")
        else:
            raise RuntimeError(f"Tool: 无法提取 code 参数，args={args}")

        # Print generated code
        print("=" * 80)
        print("[Tool] 生成的代码:")
        print(code)
        print("=" * 80)

        # Execute code in sandbox
        execution = sandbox.run_code(code)

        # Construct execution result
        latest_execution = {
            "code": code,
            "stdout": execution.logs.stdout if execution.logs else "",
            "stderr": execution.logs.stderr if execution.logs else "",
            "error": execution.error if execution.error else None
        }

        # Read registry from sandbox
        try:
            registry_content = sandbox.files.read("/home/user/registered_files.json")
            registry = json.loads(registry_content)
            generated_files = {**registry.get("main_tables", {}), **registry.get("artifacts", {})}
        except Exception:
            generated_files = state.get("generated_files") or {}

        # Check if execution failed
        if execution.error:
            print(f"--- [Tool] 执行失败: {execution.error} ---")

            # Construct error for retry
            tool_message = ToolMessage(
                content=f"Error: {execution.error}\nStderr: {execution.logs.stderr if execution.logs else ''}",
                tool_call_id=tool_call_id,
                status="error"
            )

            last_error = {
                "ai_message": ai_message,
                "tool_message": tool_message
            }

            return {
                "latest_execution": None,
                "last_error": last_error,
                "generated_files": generated_files
            }

        # Execution succeeded
        print("--- [Tool] 执行成功 ---")

        # Construct ToolMessage
        tool_message = ToolMessage(
            content=execution.logs.stdout if execution.logs else "",
            tool_call_id=tool_call_id
        )

        # Append to execution trace
        execution_trace = state.get("execution_trace") or []
        execution_trace.append({
            "ai_message": ai_message,
            "tool_message": tool_message
        })

        return {
            "latest_execution": latest_execution,
            "last_error": None,
            "execution_trace": execution_trace,
            "generated_files": generated_files
        }

    return tool_node


def tool_router(state: CustomModelingState) -> str:
    """Tool 节点路由函数"""
    return "executor" if state.get("last_error") else "observer"
