"""核心测试运行器模块"""
import traceback
from pathlib import Path
from typing import Dict, Any

from app.nodes.auto_etl import auto_etl_node
from app.graph.modeling_custom_workflow import build_modeling_custom_subgraph
from .utils import build_initial_state


async def run_single_question(sandbox, question_data: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
    """
    运行单个问题（完全参考 main.py 的实现）

    Args:
        sandbox: PPIO Sandbox 实例
        question_data: 问题数据（包含 question, constraints, format, file_name 等）
        file_path: 本地数据文件路径

    Returns:
        包含 output, error, success 的字典
    """
    try:
        # 1. 上传文件到 sandbox（参考 main.py:130-138）
        remote_path = "/home/user/raw_0.csv"
        with open(file_path, 'rb') as f:
            sandbox.files.write(remote_path, f)

        # 2. 构造 state（参考 main.py:144-156）
        initial_state = build_initial_state(question_data, remote_path)

        # 3. 运行 auto_etl（获取 data_schema）
        etl_updates = auto_etl_node(initial_state, sandbox)

        # 手动合并状态更新（模拟 LangGraph 的状态合并机制）
        state = {**initial_state, **etl_updates}

        # 4. 运行 modeling_custom 子图
        subgraph = build_modeling_custom_subgraph(sandbox)
        result = await subgraph.ainvoke(state)

        # 5. 提取输出
        output = result.get("modeling_summary", "")

        return {
            "output": output,
            "error": None,
            "success": True
        }

    except Exception as e:
        error_detail = traceback.format_exc()
        return {
            "output": "",
            "error": error_detail,
            "success": False
        }
