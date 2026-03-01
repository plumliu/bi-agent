import json
import logging
import os
import re
from typing import Dict, Any, List, Union

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.core.state import AgentState
from app.core.config import settings
from ppio_sandbox.code_interpreter import Sandbox
from app.prompts.auto_etl_prompt import AUTO_ETL_SYSTEM_TEMPLATE
from app.utils.csv_reader import get_csv_schema

# 1. 当前阶段标识
step = "auto_etl"
logger = logging.getLogger(__name__)

# 定义本地留档路径 (与 main.py 中的 LOCAL_CSV_PATH 保持一致)
LOCAL_CSV_PATH = "temp/temp_data.csv"

# 2. 初始化 LLM
llm = ChatOpenAI(
    model=settings.LLM_MODEL_NAME,
    temperature=0.1,
    api_key=settings.OPENAI_API_KEY,
    use_responses_api=settings.USE_RESPONSES_API,
)


# ==========================================
# 辅助函数
# ==========================================

def _parse_stdout(stdout_content: Union[str, List[str], None]) -> str:
    """
    兼容处理沙盒返回的 stdout。
    API 可能返回字符串，也可能返回字符串列表。
    """
    if stdout_content is None:
        return ""
    if isinstance(stdout_content, list):
        return "".join(stdout_content)
    return str(stdout_content)

def _generate_meta_extraction_code(raw_paths: List[str]) -> str:
    """生成用于提取原始文件列名的 Python 代码 (仅在多文件时使用)"""
    return """
    import pandas as pd
    import json

    paths = {raw_paths_json}
    results = []

    for p in paths:
        try:
            cols = pd.read_csv(p, nrows=0).columns.tolist()
            results.append(cols)
        except Exception as e:
            results.append([f"Error: {{str(e)}}"])

    print(json.dumps(results))
    """.format(raw_paths_json=json.dumps(raw_paths))


# ==========================================
# Auto-ETL Node 核心逻辑
# ==========================================

def auto_etl_node(state: AgentState, sandbox: Sandbox) -> Dict[str, Any]:
    print("--- [Auto-ETL] 分析文件关系中... ---")

    raw_paths = state.get("raw_file_paths", [])
    orig_names = state.get("original_filenames", [])

    # 沙盒内的目标路径
    TARGET_PATH = "/home/user/data.csv"

    # 确保本地目录存在
    os.makedirs(os.path.dirname(LOCAL_CSV_PATH), exist_ok=True)

    # --------------------------------------------------------
    # 分支 A: 单文件快速通道 (Fast Path)
    # --------------------------------------------------------
    if len(raw_paths) == 1:
        print("--- [Auto-ETL] 单文件检测，执行快速重命名... ---")
        src_path = raw_paths[0]

        # 1. 在沙盒内重命名
        cmd_exec = sandbox.commands.run(f"mv {src_path} {TARGET_PATH}")
        if cmd_exec.error:
            raise RuntimeError(f"文件重命名失败: {cmd_exec.error.value}")

        merge_msg = f"单文件 {orig_names[0]} 已就绪。"

    # --------------------------------------------------------
    # 分支 B: 多文件智能合并 (Smart Merge)
    # --------------------------------------------------------
    else:
        print(f"--- [Auto-ETL] 多文件检测 ({len(raw_paths)}个)，请求 LLM 生成合并策略... ---")

        # 1. 提取元数据 (沙盒内执行)
        meta_code = _generate_meta_extraction_code(raw_paths)
        exec_res = sandbox.run_code(meta_code)

        # 错误检查
        if exec_res.error:
            raise RuntimeError(f"无法读取文件表头 (Execution Error): {exec_res.error.name}: {exec_res.error.value}")
        elif exec_res.logs.stderr:
            raise RuntimeError(f"无法读取文件表头 (Stderr): {exec_res.logs.stderr}")

        # 解析输出
        try:
            stdout_str = _parse_stdout(exec_res.logs.stdout)
            if not stdout_str.strip():
                stdout_str = "[]"

            columns_list = json.loads(stdout_str)
        except Exception as e:
            raise RuntimeError(f"表头元数据解析失败: {e}, STDOUT: {exec_res.logs.stdout}")

        # 2. 构建 Prompt 上下文
        file_info_str = ""
        for name, path, cols in zip(orig_names, raw_paths, columns_list):
            file_info_str += f"- 原始名: {name}\n  沙盒路径: {path}\n  列名: {cols}\n\n"

        # 3. 调用 LLM
        system_content = AUTO_ETL_SYSTEM_TEMPLATE.format(file_info=file_info_str)
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content="请生成合并代码。")
        ]

        response = llm.invoke(messages)
        code_match = re.search(r"<code>(.*?)</code>", response.content, re.DOTALL)

        code = None

        if code_match:
            code = code_match.group(1).strip()
            # 打印分析过程供调试 (可选)
            analysis_match = re.search(r"<analysis>(.*?)</analysis>", response.content, re.DOTALL)
            if analysis_match:
                print(f"\n[Auto-ETL Analysis]:\n{analysis_match.group(1).strip()}\n")
        else:
            # 兼容性处理：如果没有标签，尝试直接清洗 markdown
            print("[Auto-ETL Warning] 未检测到 <code> 标签，尝试直接解析...")
            code = response.content.replace("```python", "").replace("```", "").strip()

        # 4. 检查 LLM 拒绝
        if "REJECT" in code:
            raise RuntimeError("错误：agent 判断上传的文件之间缺乏逻辑关联，拒绝合并。")

        # 5. 执行合并代码
        print("--- [Auto-ETL] 执行合并脚本... ---")
        merge_exec = sandbox.run_code(code)

        if merge_exec.error:
            error_detail = f"{merge_exec.error.name}: {merge_exec.error.value}"
            if merge_exec.error.traceback:
                error_detail += f"\nTraceback: {merge_exec.error.traceback}"
            raise RuntimeError(f"合并脚本执行失败: {error_detail}")
        elif merge_exec.logs.stderr:
            raise RuntimeError(f"合并脚本输出了错误信息: {merge_exec.logs.stderr}")

        merge_msg = f"已自动合并 {len(raw_paths)} 个文件片段。"

    # --------------------------------------------------------
    # 通用步骤: 下载文件 & 本地生成 Schema
    # --------------------------------------------------------
    print(f"--- [Auto-ETL] 正在下载合并后的文件至 {LOCAL_CSV_PATH}... ---")

    try:
        # 1. 从沙盒下载文件 (读取二进制流)
        file_content = sandbox.files.read(TARGET_PATH)

        # 2. 写入本地文件 (留档)
        with open(LOCAL_CSV_PATH, "w", encoding="utf-8") as f:
            f.write(file_content)

    except Exception as e:
        raise RuntimeError(f"无法从沙盒下载最终数据文件: {e}")

    print("--- [Auto-ETL] 正在本地生成 Data Schema... ---")

    # 3. 调用本地工具生成 Schema
    data_schema = get_csv_schema(LOCAL_CSV_PATH)

    # 4. 检查 Schema 生成结果
    if "error" in data_schema:
        raise RuntimeError(f"Schema 生成失败: {data_schema['error']}")

    # 返回 State 的增量更新
    return {
        "remote_file_path": TARGET_PATH,  # 沙盒里的路径，给后续节点用
        "data_schema": data_schema,  # Schema 给 Router 用
        "messages": [AIMessage(content=f"[系统汇报] {merge_msg}")]
    }