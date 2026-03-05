import os
import uuid
from langchain_core.tools import tool
from ppio_sandbox.code_interpreter import Sandbox


def create_python_script_runner(sandbox: Sandbox, task_id: str):
    """
    工厂函数：为并行的 Viz Agent 创建专属的 Python 脚本执行工具。
    每个任务输出到独立的文件路径以支持并发执行。
    """

    @tool("python_script_runner")
    def python_script_runner(code: str) -> str:
        """
        Python 脚本执行器。执行数据处理和可视化数据生成脚本。

        【环境说明】
        1. 数据文件位置: /home/user/ (读取 Feather/JSON 文件)
        2. 输出文件路径: /home/user/viz_output_{task_id}.json (固定路径，请在代码中使用此路径)

        Args:
            code: 完整的 Python 代码（必须包含所有必要的 import 语句）。

        Returns:
            代码执行的标准输出 (STDOUT)、标准错误 (STDERR) 以及进程退出码。
        """
        print(f"\n--- [Viz Tool] 执行任务 {task_id} 的脚本 ---")

        # 生成临时脚本文件名
        script_file = f"/tmp/viz_script_{task_id}_{uuid.uuid4().hex[:8]}.py"

        try:
            # 1. 写入代码到临时文件
            sandbox.files.write(script_file, code)

            # 2. 运行脚本，获取 CommandResult
            execution = sandbox.commands.run(f"cd /home/user && python {script_file}")

            output = []

            # 3. 解析 CommandResult (标准输出)
            if execution.stdout:
                output.append(f"[STDOUT]\n{execution.stdout.strip()}")

            # 解析标准错误 (Python的异常Traceback通常打印在stderr中)
            if execution.stderr:
                output.append(f"[STDERR]\n{execution.stderr.strip()}")

            # 解析系统级错误 (例如命令不存在、沙盒内存溢出被打断等)
            if execution.error:
                output.append(f"[SYSTEM ERROR]\n{execution.error.strip()}")

            # 强化反馈：通过 exit_code 明确告诉大模型执行状态
            if execution.exit_code == 0:
                output.append(f"\n[STATUS] 进程正常结束 (exit_code: 0)")
            else:
                output.append(
                    f"\n[STATUS] 进程异常崩溃 (exit_code: {execution.exit_code})！请检查上方的 [STDERR] 报错信息并修正代码。")

            # 如果没有任何输出且正常退出
            if not execution.stdout and not execution.stderr and execution.exit_code == 0:
                return "[SYSTEM] 代码执行成功，进程正常退出，但没有任何控制台输出 (STDOUT 为空)。如果你需要提取数据，请确保在代码中使用了 print()。"

            return "\n".join(output)

        except Exception as e:
            return f"[SANDBOX SDK ERROR] 沙盒底层调用失败: {str(e)}"

    # 动态注入 task_id 到 Docstring
    python_script_runner.__doc__ = python_script_runner.__doc__.format(task_id=task_id)

    return python_script_runner