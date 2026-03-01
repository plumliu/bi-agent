import os
import uuid
from langchain_core.tools import tool
from ppio_sandbox.code_interpreter import Sandbox


def create_python_script_runner(sandbox: Sandbox, task_id: str, base_data_dir: str = "/home/user"):
    """
    工厂函数：为并行的 Viz Agent 创建专属的、目录隔离的 Python 脚本执行工具。
    """

    # 定义该 Agent 的专属工作目录
    workspace_dir = f"/home/user/viz_workspaces/{task_id}"

    # 提前在沙盒中创建好隔离目录
    try:
        sandbox.commands.run(f"mkdir -p {workspace_dir}")
    except Exception as e:
        print(f"Warning: Failed to create workspace {workspace_dir}: {e}")

    @tool("python_script_runner")
    def python_script_runner(file_name: str, code: str) -> str:
        """
        独立的 Python 脚本执行器。你的代码将运行在一个专属的、隔离的目录下。

        【环境说明】
        1. 你的专属工作目录是: {workspace_dir}
        2. 上游数据存放目录是: {base_data_dir} (读取 Feather/JSON 时请使用绝对路径或正确的相对路径)

        Args:
            file_name: 你想保存的 Python 脚本文件名（例如 'process_data.py'）。
            code: 完整的 Python 代码（必须包含所有必要的 import 语句）。

        Returns:
            代码执行的标准输出 (STDOUT)、标准错误 (STDERR) 以及进程退出码。
        """
        print(f"\n--- [Viz Tool] 在专属隔离目录 ({task_id}) 中执行 {file_name} ---")

        # 拼接并安全化文件路径
        safe_file_name = os.path.basename(file_name)
        file_path = f"{workspace_dir}/{safe_file_name}"

        try:
            # 1. 写入代码到专属目录
            sandbox.files.write(file_path, code)

            # 2. 运行脚本，获取 CommandResult
            execution = sandbox.commands.run(f"cd {workspace_dir} && python {safe_file_name}")

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

    # 动态注入 Docstring，明确工作目录路径
    python_script_runner.__doc__ = python_script_runner.__doc__.format(
        workspace_dir=workspace_dir,
        base_data_dir=base_data_dir
    )

    return python_script_runner