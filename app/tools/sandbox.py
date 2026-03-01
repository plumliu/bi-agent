import os
from langchain_core.tools import tool
# from e2b_code_interpreter import Sandbox
from ppio_sandbox.code_interpreter import Sandbox

def create_code_interpreter_tool(sandbox: Sandbox):
    """
    使用工厂模式创建一个绑定了特定 Sandbox 实例的 Tool。
    这样沙箱的生命周期（创建、上传文件、安装包）完全由 main.py 控制。
    """

    @tool("python_interpreter")
    def python_interpreter(code: str) -> str:
        """
        一个强大的、状态持久化的 Python Jupyter Notebook 执行沙盒。

        【环境特性与使用准则】
        1. 状态持久化：这是一个 Stateful 环境。你在上一次调用中导入的库（如 pandas）、定义的变量（如 df）会一直保留在内存中，下次调用可直接使用，无需重复读取或导入。
        2. 必须显式输出：如果你想查看变量的内容、数据的预览或算法的结果，必须在代码中使用 `print()` 函数打印出来。工具只会输出标准输出 (STDOUT) 和报错 (STDERR)，仅仅计算而不 print 没有任何意义。

        Args:
            code: 需要在沙盒中执行的有效 Python 代码块。
        """
        print(f"\n--- [Tool] 在沙盒中执行代码中 ---")
        print(code)

        try:
            execution = sandbox.run_code(code)

            output = []

            if execution.logs.stdout:
                output.append(f"[STDOUT]\n{execution.logs.stdout}")
            if execution.logs.stderr:
                print(execution.logs.stderr)
                output.append(f"[STDERR]\n{execution.logs.stderr}")
            if execution.error:
                print(execution.error)
                output.append(
                    f"[EXECUTION ERROR]\n{execution.error.name}: {execution.error.value}\n{execution.error.traceback}")

            if execution.results:
                output.append(f"[SYSTEM] 生成了 {len(execution.results)} 结果.")

            if not output:
                return "[SYSTEM] 代码执行成功了，但并没有返回任何输出."

            return "\n".join(output)

        except Exception as e:
            return f"[SYSTEM ERROR] 沙盒出现错误: {str(e)}"

    return python_interpreter