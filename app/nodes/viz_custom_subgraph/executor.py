import os
import json
import time
import yaml
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from ppio_sandbox.code_interpreter import Sandbox

from app.core.config import settings
from app.core.prompts_config import load_prompts_config
from app.core.viz_custom_subgraph.state import CustomVizState
from app.tools.python_script_runner import create_python_script_runner
from app.utils.extract_text_from_content import extract_text_from_content


# 初始化 LLM
llm = ChatOpenAI(
    model=settings.LLM_MODEL_NAME,
    temperature=0,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_API_BASE,
    use_responses_api=settings.USE_RESPONSES_API,
    streaming=True,  # responses API 需要流式调用
)


def load_chart_guide(chart_type: str) -> str:
    """加载图表使用说明"""
    guide_path = f"app/prompts/viz_charts/{chart_type}.yaml"
    try:
        with open(guide_path, 'r', encoding='utf-8') as f:
            guide = yaml.safe_load(f)
            return f"""
【图表类型】: {guide['chart_type']}
【描述】: {guide['description']}

【数据协议】
{guide['data_protocol']}

【示例】
{guide['example']}

【常见错误】
{guide['common_errors']}
"""
    except Exception as e:
        print(f"--- [Viz Executor] 警告: 无法加载图表说明 {chart_type}: {e} ---")
        return f"图表类型: {chart_type}（无详细说明）"


def viz_executor_node(state: CustomVizState, sandbox: Sandbox) -> Dict[str, Any]:
    """
    【Viz Executor 节点】
    职责：
    1. 接收单个 viz_task（通过 Send API 传入）
    2. 动态加载图表使用说明
    3. 使用 python_script_runner 在隔离目录生成数据
    4. 在 ReAct 循环中自我纠错
    5. 完成后直接从沙箱下载产物到本地
    """

    # 1. 获取当前任务（Send API 会将 viz_task 注入到 state 中）
    viz_task = state.get("viz_task")
    if not viz_task:
        raise RuntimeError("--- [Viz Executor] 错误: viz_task 未找到 ---")

    task_id = viz_task["task_id"]
    chart_type = viz_task["chart_type"]

    print(f"--- [Viz Executor] {task_id}: 开始处理 {chart_type} 图表 ---")

    # 2. 加载图表使用说明
    chart_guide = load_chart_guide(chart_type)

    # 3. 获取文件列信息（透传）
    file_columns = viz_task.get("file_columns", [])

    # 格式化文件列信息为可读文本
    file_columns_text = ""
    for file_info in file_columns:
        file_name = file_info["file_name"]
        columns = file_info["columns"]
        file_columns_text += f"\n【文件: {file_name}】\n"
        for col in columns:
            file_columns_text += f"  - {col['name']}: {col['description']}\n"

    # 4. 创建工具
    python_runner = create_python_script_runner(sandbox, task_id)
    llm_with_tools = llm.bind_tools([python_runner])

    # 5. 加载提示词配置
    prompts = load_prompts_config("viz", "custom")
    executor_instruction = prompts["executor_instruction"]
    context_template = prompts["executor_context_template"]

    # 6. SystemMessage 完全静态
    system_message = SystemMessage(content=executor_instruction)

    # 7. HumanMessage 包含动态上下文
    source_files_str = ", ".join(viz_task.get('source_files', []))
    context_content = context_template.format(
        task_id=task_id,
        chart_type=chart_type,
        title=viz_task['title'],
        description=viz_task['description'],
        source_files_str=source_files_str,
        file_columns_text=file_columns_text,
        chart_guide=chart_guide
    )
    context_message = HumanMessage(content=context_content)

    # 8. ReAct 循环
    messages = [system_message, context_message]
    max_iterations = 5
    iteration = 0

    start_time = time.perf_counter()

    while iteration < max_iterations:
        iteration += 1
        print(f"--- [Viz Executor] {task_id}: 迭代 {iteration}/{max_iterations} ---")

        # 调用 LLM
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # 检查是否有工具调用
        if not response.tool_calls:
            # LLM 认为任务完成
            print(f"--- [Viz Executor] {task_id}: LLM 宣布完成 ---")
            answer = extract_text_from_content(response.content)
            print(f"--- [Viz Executor] {task_id}: {answer} ---")
            break

        # 执行工具调用（python_script_runner）
        from langchain_core.messages import ToolMessage

        for tool_call in response.tool_calls:
            print(f"--- [Viz Executor] {task_id}: 执行脚本... ---")

            # 手动调用工具
            try:
                tool_result = python_runner.invoke(tool_call["args"])

                # 创建 ToolMessage
                tool_message = ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                )
                messages.append(tool_message)

                print(f"--- [Viz Executor] {task_id}: 脚本执行完成 ---")

            except Exception as e:
                error_msg = f"脚本执行失败: {e}"
                print(f"--- [Viz Executor] {task_id}: {error_msg} ---")

                tool_message = ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_call["id"]
                )
                messages.append(tool_message)

    executor_duration = time.perf_counter() - start_time

    # 7. 检查是否成功（通过检查沙箱中的文件）
    try:
        check_result = sandbox.commands.run(f"test -f /home/user/viz_output_{task_id}.json && echo 'EXISTS' || echo 'NOT_FOUND'")
        if "EXISTS" not in check_result.stdout:
            raise RuntimeError(f"输出文件未生成")
    except Exception as e:
        print(f"--- [Viz Executor] {task_id}: 失败 - {e} ---")
        return {
            "messages": [AIMessage(content=f"[Viz Executor] {task_id} 失败: {e}")]
        }

    # 8. 下载产物到本地
    try:
        print(f"--- [Viz Executor] {task_id}: 正在下载产物... ---")

        # 从沙箱读取文件
        file_content = sandbox.files.read(f"/home/user/viz_output_{task_id}.json")

        # 保存到本地 temp 目录
        local_path = f"temp/viz_{task_id}.json"
        os.makedirs("temp", exist_ok=True)

        with open(local_path, 'w', encoding='utf-8') as f:
            f.write(file_content)

        print(f"--- [Viz Executor] {task_id}: 产物已下载到 {local_path} ---")
        print(f"--- [Time] {task_id} 总耗时: {executor_duration:.2f} 秒 ---")

        # 9. 返回状态更新
        return {
            "messages": [AIMessage(content=f"[Viz Executor] {task_id} 完成，耗时 {executor_duration:.2f}s")]
        }

    except Exception as e:
        error_msg = f"--- [Viz Executor] {task_id}: 下载失败 - {e} ---"
        print(error_msg)
        return {
            "messages": [AIMessage(content=f"[Viz Executor] {task_id} 下载失败: {e}")]
        }
