import ast

def clean_log_content(content: str) -> str:
    """
    清洗 Agent 的原始输出，使其对人类更友好
    """
    if "[STDOUT]" in content:
        content = content.replace("[STDOUT]", "").strip()

        try:
            parsed = ast.literal_eval(content)
            if isinstance(parsed, list):
                return "".join(parsed).strip()
        except:
            pass

    if "Traceback" in content:
        return "执行代码时遇到错误，正在重试..."

    return content