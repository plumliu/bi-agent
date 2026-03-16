import json
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from ppio_sandbox.code_interpreter import Sandbox

from app.core.modeling_custom_subgraph.state import CustomModelingState
from app.core.prompts_config import load_prompts_config
from app.tools.python_interpreter import create_code_interpreter_tool
from app.utils.llm_factory import create_llm, apply_retry

def create_executor_node(sandbox: Sandbox):
    """创建 Executor 节点（工厂函数）"""

    # Initialize LLM with tools (主模型使用Anthropic)
    # 先bind_tools，再apply_retry
    llm_with_tools = apply_retry(
        create_llm(use_flash=False).bind_tools([create_code_interpreter_tool(sandbox)])
    )

    def executor_node(state: CustomModelingState) -> Dict[str, Any]:
        print("--- [Executor] 开始执行当前任务 ---")

        # Load executor instruction
        prompts = load_prompts_config("modeling", "custom")
        executor_instruction = prompts["executor_instruction"]

        # Construct messages array
        messages = [SystemMessage(content=executor_instruction)]

        # Add successful execution history from execution_trace (reducer-managed)
        history = state.get("execution_trace") or []
        print(f"--- [Executor DEBUG] execution_trace长度: {len(history)} ---")
        for i, m in enumerate(history):
            print(i, type(m), repr(m)[:300])
        history = _coerce_ai_tool_history(history)
        messages.extend(history)

        # Add failed execution if retrying
        last_error = state.get("last_error")
        if last_error:
            ai_msg = state.get("latest_ai_message")
            if ai_msg:
                messages.append(ai_msg)
                error_tool_msg = last_error.get("tool_message")
                if error_tool_msg:
                    messages.append(error_tool_msg)

        # Construct current context HumanMessage
        completed_tasks = state.get("completed_tasks") or []
        confirmed_findings = state.get("confirmed_findings") or []
        working_hypotheses = state.get("working_hypotheses") or []
        generated_files = state.get("generated_files") or {}

        completed_str = "\n".join([f"- {t['description']}" for t in completed_tasks]) if completed_tasks else "无"
        findings_str = "\n".join([f"- {f}" for f in confirmed_findings]) if confirmed_findings else "无"
        hypotheses_str = "\n".join([f"- {h}" for h in working_hypotheses]) if working_hypotheses else "无"

        context = f"""current_task: {state['current_task']}

completed_tasks:
{completed_str}

confirmed_findings:
{findings_str}

working_hypotheses (the previous round’s Observer cognitive state, for reference only):
{hypotheses_str}

generated_files:
{json.dumps(generated_files, ensure_ascii=False, indent=2)}
"""

        if last_error:
            context += "\n\n The current Jupyter code cell execution failed! Please rewrite the code for the current task!"

        messages.append(HumanMessage(content=context))

        # Call LLM
        print(f"--- [Executor] 当前任务: {state['current_task']} ---")
        print(f"--- [Executor DEBUG] Messages数组长度: {len(messages)} ---")
        print(f"--- [Executor DEBUG] Messages类型: {[type(m).__name__ for m in messages]} ---")
        print(f"--- [Executor DEBUG] 准备调用LLM... ---")

        try:
            ai_response = llm_with_tools.invoke(messages)
            print(f"--- [Executor DEBUG] LLM调用成功 ---")
        except Exception as e:
            print(f"--- [Executor ERROR] LLM调用失败: {type(e).__name__}: {str(e)} ---")
            raise

        # Extract code from tool calls
        if not ai_response.tool_calls:
            raise RuntimeError("Executor: LLM 未返回工具调用")

        print(f"--- [Executor] 已生成代码 ---")

        return {
            "latest_ai_message": ai_response,
            "last_error": None
        }

    return executor_node


def _coerce_ai_tool_history(history):
    if len(history) % 2 != 0:
        raise RuntimeError(
            f"Executor: execution_trace 长度必须为偶数（AI/Tool 成对），当前={len(history)}"
        )

    repaired = []

    for i in range(0, len(history), 2):
        ai_candidate = history[i]
        tool_candidate = history[i + 1]

        if not isinstance(ai_candidate, AIMessage):
            print(f"--- [Executor WARN] history[{i}] 不是 AIMessage，而是 {type(ai_candidate).__name__}，尝试转换 ---")
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
            print(f"--- [Executor WARN] history[{i+1}] 不是 ToolMessage，而是 {type(tool_candidate).__name__}，尝试转换 ---")
            status = getattr(tool_candidate, "status", "success") or "success"
            if status not in {"success", "error"}:
                status = "success"

            tool_call_id = getattr(tool_candidate, "tool_call_id", None)
            if not tool_call_id:
                raise RuntimeError(
                    f"Executor: history[{i+1}] 缺少 tool_call_id，无法安全转换为 ToolMessage"
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
            raise RuntimeError(
                f"Executor: history[{i}] 转换后仍无 tool_calls，不能作为合法 AIMessage"
            )
        if not tool_candidate.tool_call_id:
            raise RuntimeError(
                f"Executor: history[{i+1}] 转换后仍无 tool_call_id，不能作为合法 ToolMessage"
            )

        repaired.extend([ai_candidate, tool_candidate])

    return repaired