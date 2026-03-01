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
        Python 代码执行环境。
        Args:
            code: Python 代码。
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