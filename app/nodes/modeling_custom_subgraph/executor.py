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

        # Add successful execution history from execution_trace
        execution_trace = state.get("execution_trace") or []
        print(f"--- [Executor DEBUG] execution_trace长度: {len(execution_trace)} ---")
        for i, trace in enumerate(execution_trace):
            ai_msg = trace["ai_message"]
            tool_msg = trace["tool_message"]
            print(f"--- [Executor DEBUG] Trace {i}: ai_msg类型={type(ai_msg).__name__}, tool_msg类型={type(tool_msg).__name__} ---")
            # Reconstruct message objects if they were serialized to dicts
            if isinstance(ai_msg, dict):
                print(f"--- [Executor DEBUG] ai_msg是dict，keys={list(ai_msg.keys())} ---")
                ai_msg = AIMessage(**ai_msg)
            if isinstance(tool_msg, dict):
                print(f"--- [Executor DEBUG] tool_msg是dict，keys={list(tool_msg.keys())} ---")
                tool_msg = ToolMessage(**tool_msg)
            messages.append(ai_msg)
            messages.append(tool_msg)

        # Add failed execution if retrying
        last_error = state.get("last_error")
        if last_error:
            ai_msg = last_error["ai_message"]
            tool_msg = last_error["tool_message"]
            if isinstance(ai_msg, dict):
                ai_msg = AIMessage(**ai_msg)
            if isinstance(tool_msg, dict):
                tool_msg = ToolMessage(**tool_msg)
            messages.append(ai_msg)
            messages.append(tool_msg)

        # Construct current context HumanMessage
        completed_tasks = state.get("completed_tasks") or []
        confirmed_findings = state.get("confirmed_findings") or []
        generated_files = state.get("generated_files") or {}

        completed_str = "\n".join([f"- {t['description']}" for t in completed_tasks]) if completed_tasks else "无"
        findings_str = "\n".join([f"- {f}" for f in confirmed_findings]) if confirmed_findings else "无"

        context = f"""当前任务: {state['current_task']}

已完成任务:
{completed_str}

已确认发现:
{findings_str}

当前已登记文件:
{json.dumps(generated_files, ensure_ascii=False, indent=2)}
"""

        if last_error:
            context += "\n\n 当前的 jupyter code cell 执行失败！请重新编写当前任务的代码！"

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
