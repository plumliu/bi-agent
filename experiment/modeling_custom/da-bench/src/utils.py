"""辅助函数模块"""
import json
from pathlib import Path
from typing import Dict, Any, List


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def enhance_question(question_data: Dict[str, Any]) -> str:
    """拼接问题、约束条件和格式要求"""
    return (
        f"{question_data['question']}\n\n"
        f"约束条件：\n{question_data['constraints']}\n\n"
        f"输出格式要求：\n{question_data['format']}"
    )


def build_initial_state(question_data: Dict[str, Any], remote_path: str) -> Dict[str, Any]:
    """构造初始 WorkflowState（跳过 router）"""
    enhanced_question = enhance_question(question_data)

    return {
        "user_input": enhanced_question,
        "raw_file_paths": [remote_path],
        "original_filenames": [question_data['file_name']],
        "remote_file_path": None,  # auto_etl 会填充
        "data_schema": {},  # auto_etl 会填充
        "scenario": "custom",  # 直接设置，跳过 router
        "reasoning": "Direct routing to custom for benchmark evaluation",
        "modeling_artifacts": None,
        "modeling_summary": None,
        "generated_data_files": None,
        "file_metadata": None,
        "viz_config": None,
        "viz_success": False,
        "viz_data": None,
        "final_summary": None,
        "error_count": 0
    }
