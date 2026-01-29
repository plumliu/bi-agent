
def generate_semantic_alias(original_filename: str, file_index: int) -> str:

    base_name, ext = original_filename.rsplit('.', 1)[0], original_filename.rsplit('.', 1)[1]

    if len(base_name) >= 16:
        alias = f"file_{chr(file_index)}"
        return f"{alias}.{ext}"

    return original_filename