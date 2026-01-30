import json
import re

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.prompts_config import load_prompts_config
from app.core.subgraph.state import CustomModelingState
from app.tools.sandbox import create_code_interpreter_tool


llm = ChatOpenAI(
    model=settings.LLM_MODEL_NAME,
    temperature=0,
    api_key=settings.OPENAI_API_KEY
)

step = "modeling"


def _extract_code(text: str) -> str:
    """
    辅助函数：从 LLM 回复中提取 Python 代码块
    """
    if "```python" in text:
        return text.split("```python")[1].split("```")[0].strip()
    elif "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    else:
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
        2. 让 LLM 生成代码 (Text -> Code)。
        3. 调用 Tool 在沙盒中运行代码。
        4. 将运行结果写入 Task.result。
        """
        print("--- [Subgraph] Executor: 正在执行任务... ---")

        # 1. 获取上下文
        plan = state["plan"]
        current_idx = state["current_task_index"]

        # 安全检查：防止索引越界
        if current_idx >= len(plan):
            return {"error": "任务索引越界"}

        task = plan[current_idx]

        # 将该任务标记为正在运行
        task.mark_running()
        remote_file_path = state.get("remote_file_path")

        # 将 scratchpad 列表转换为字符串
        scratchpad_list = state.get("scratchpad")
        scratchpad_str = "\n".join(scratchpad_list) if scratchpad_list else "无"

        # 2. 加载 Prompt
        scenario = state.get("scenario")

        config = load_prompts_config(step, scenario)
        instruction_template = config.get('executor_instruction')

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

        # 3. 构造 Prompt
        system_content = instruction_template.format(
            task_description=task.description,
            scratchpad=scratchpad_str,
            remote_file_path=remote_file_path,
            previous_code=previous_code_str
        )

        # 4. 调用 LLM 生成代码
        print(f"--- [Subgraph] Executor: 正在生成 Task {task.id} 的代码 ---")
        response = llm.invoke([SystemMessage(content=system_content)])
        raw_content = response.content

        # 5. 代码清洗
        code_to_run = _extract_code(raw_content)

        # 6. 在沙盒中执行 (调用工具)
        print(f"--- [Subgraph] Executor: 代码生成完毕，正在沙盒运行... ---")

        execution_result = ""

        if sandbox:
            # [修改] 使用注入的 sandbox 实例
            python_tool = create_code_interpreter_tool(sandbox)

            try:
                execution_result = python_tool.invoke(code_to_run)
            except Exception as e:
                execution_result = f"[SYSTEM ERROR] 工具调用失败: {str(e)}"
        else:
            execution_result = "[SYSTEM ERROR] Sandbox 未注入，无法执行代码。"

        # 7. 更新 Task 对象状态
        task.code = code_to_run
        task.result = execution_result

        print(f"--- [Subgraph] Executor: 执行完毕，结果已存入 Task {task.id} ---")

        return {"plan": plan}

    return executor_node