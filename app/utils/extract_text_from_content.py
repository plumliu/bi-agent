def extract_text_from_content(content) -> str:
    """
    兼容 OpenAI 新旧 API：从 response.content 中安全提取纯文本。
    支持 str 和 List[dict] 格式。
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts)

    return str(content)