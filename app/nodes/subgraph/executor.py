import json
import re
import time  # [新增] 引入时间模块

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.prompts_config import load_prompts_config
from app.core.subgraph.state import CustomModelingState
from app.tools.sandbox import create_code_interpreter_tool

# 1. 初始化 LLM
llm = ChatOpenAI(
    model=settings.LLM_MODEL_NAME,
    temperature=0,  # 保持 0，确保代码生成的确定性
    api_key=settings.OPENAI_API_KEY
)

step = "modeling"


def _extract_code(text: str) -> str:
    """
    【升级版】辅助函数：从 LLM 回复中提取 Python 代码块
    增强了对多代码块和非标准 markdown 格式的兼容性。
    """
    # 优先匹配所有标准的 ```python ... ``` 块
    matches = re.findall(r'```python\n(.*?)\n```', text, re.DOTALL)
    if matches:
        # 如果模型输出了多个代码块，将它们拼接起来（通常模型是为了分段解释，但执行时需要完整代码）
        return "\n\n".join(matches).strip()

    # 降级方案：匹配没有写 python 语言标识的 ``` ... ``` 块
    fallback_matches = re.findall(r'```(.*?)```', text, re.DOTALL)
    if fallback_matches:
        return "\n\n".join([m.strip() for m in fallback_matches]).strip()

    # 如果完全没有 markdown 标记，直接剔除首尾空白返回
    return text.strip()


def create_executor_node(sandbox):
    """
    工厂函数：注入 Sandbox 依赖，返回 Executor 节点函数
    """

    def executor_node(state: CustomModelingState):
        """
        【Executor 节点】
        职责：
        1. 读取当前 Task。
        2. 检查是否为重试状态，若是则提取上次的错误现场。
        3. 让 LLM 生成代码 (Text -> Code)。
        4. 调用 Tool 在沙盒中运行代码。
        5. 将运行结果写入 Task.result。
        """
        print("--- [Subgraph] Executor: 正在准备执行任务... ---")

        # [新增] 初始化或获取当前的 metrics
        metrics = state.get("metrics") or {}
        metrics.setdefault("llm_duration", 0.0)
        metrics.setdefault("sandbox_duration", 0.0)

        # 1. 获取上下文与当前任务
        plan = state["plan"]
        current_idx = state["current_task_index"]

        # 安全检查：防止索引越界
        if current_idx >= len(plan):
            return {"error": "任务索引越界"}

        task = plan[current_idx]

        # 将该任务标记为正在运行
        task.mark_running()
        remote_file_path = state.get("remote_file_path")
        current_retry_count = state.get("retry_count", 0)

        # 提取 Scratchpad
        scratchpad_list = state.get("scratchpad")
        scratchpad_str = "\n".join(scratchpad_list) if scratchpad_list else "无"

        # 2. 加载 Prompt 模板
        scenario = state.get("scenario")
        config = load_prompts_config(step, scenario)
        instruction_template = config.get('executor_instruction')

        # 3. 构造历史成功代码 (Context)
        # 严格只取 current_idx 之前的 completed 任务，防止污染
        previous_code_blocks = []
        cell_pattern = re.compile(r'^# \[Jupyter Code Cell\]: .+\n')

        for t in plan[:current_idx]:
            if t.status == "completed" and t.code:
                stripped = t.code.lstrip()
                if cell_pattern.match(stripped):
                    previous_code_blocks.append(stripped)
                else:
                    previous_code_blocks.append(f"# [Jupyter Code Cell]: {t.description}\n{stripped}")

        previous_code_str = "\n\n".join(previous_code_blocks) if previous_code_blocks else "无 (这是第一个任务)"

        # 4. 构造基础 Prompt
        system_content = instruction_template.format(
            task_description=task.description,
            scratchpad=scratchpad_str,
            remote_file_path=remote_file_path,
            previous_code=previous_code_str
        )

        messages = [SystemMessage(content=system_content)]

        # 5. 【核心修复】：注入重试的“错误现场”
        if current_retry_count > 0 and task.code and task.result:
            print(
                f"--- [Subgraph] Executor: 检测到 Task {task.id} 为第 {current_retry_count} 次重试，正在注入错误日志 ---")
            retry_warning = (
                f"【紧急修正：当前为第 {current_retry_count} 次重试】\n"
                f"你上一次生成的代码执行失败（或被 Reflector 判定为逻辑错误）。请仔细分析下方的错误现场，并在本次生成中彻底修复！\n\n"
                f"--- 你上次生成的错误代码 ---\n```python\n{task.code}\n```\n\n"
                f"--- 沙盒报错 / 执行日志 ---\n{task.result}\n\n"
                f"注意：你在 Jupyter 环境中，请注意全局变量（如 df）可能已被你上次的错误代码污染。建议使用防御性编程（如检查列是否存在，或重新拷贝数据）。"
            )
            # 作为一个显眼的 HumanMessage 追加，强迫大模型优先关注报错
            messages.append(HumanMessage(content=retry_warning))

        # 6. 调用 LLM 生成代码
        print(f"--- [Subgraph] Executor: 正在生成 Task {task.id} 的代码 ---")

        # [新增] 掐表开始：LLM 耗时
        llm_start_time = time.perf_counter()

        response = llm.invoke(messages)

        # [新增] 掐表结束：LLM 耗时
        llm_end_time = time.perf_counter()
        current_llm_time = llm_end_time - llm_start_time
        metrics["llm_duration"] += current_llm_time
        print(f"--- [Time] Executor LLM 生成代码耗时: {current_llm_time:.2f} 秒")

        raw_content = response.content

        # 7. 代码清洗与提取
        code_to_run = _extract_code(raw_content)

        # 8. 在沙盒中执行 (调用工具)
        print(f"--- [Subgraph] Executor: 代码生成完毕，正在沙盒中运行... ---")
        execution_result = ""

        if sandbox:
            python_tool = create_code_interpreter_tool(sandbox)

            # [新增] 掐表开始：沙盒执行耗时
            sandbox_start_time = time.perf_counter()

            try:
                execution_result = python_tool.invoke(code_to_run)
            except Exception as e:
                execution_result = f"[SYSTEM ERROR] 沙盒工具调用底层异常: {str(e)}"

            # [新增] 掐表结束：沙盒执行耗时
            sandbox_end_time = time.perf_counter()
            current_sb_time = sandbox_end_time - sandbox_start_time
            metrics["sandbox_duration"] += current_sb_time
            print(f"--- [Time] Executor Sandbox 代码执行耗时: {current_sb_time:.2f} 秒")

        else:
            execution_result = "[SYSTEM ERROR] Sandbox 未注入，无法执行代码。"

        # 9. 更新 Task 对象状态 (覆盖掉之前的错误代码和结果，供下一轮 Reflector 审查)
        task.code = code_to_run
        task.result = execution_result

        print(f"--- [Subgraph] Executor: Task {task.id} 的代码执行结果如下 ---")
        print(execution_result)
        print(f"--- [Subgraph] Executor: 执行完毕，结果已更新至 Task {task.id} ---")

        # [修改] 返回未修改的 plan 引用和更新后的 metrics
        return {
            "plan": plan,
            "metrics": metrics
        }

    return executor_node