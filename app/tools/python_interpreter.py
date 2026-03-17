from langchain_core.tools import tool


def create_code_interpreter_tool():
    """Create the local python_interpreter tool schema for tool calling."""

    @tool("python_interpreter")
    def python_interpreter(code: str) -> str:
        """
        Stateful local IPython kernel executor.

        Args:
            code: Valid Python code for a single notebook cell.

        Returns:
            Execution output text. In this workflow, execution is handled by the tool node.
        """
        return "python_interpreter call received"

    return python_interpreter
