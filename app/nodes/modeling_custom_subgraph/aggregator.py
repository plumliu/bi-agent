import json
from typing import Dict, Any

from ppio_sandbox.code_interpreter import Sandbox

from app.core.modeling_custom_subgraph.state import CustomModelingState


def create_modeling_aggregator_node(sandbox: Sandbox):
    """创建 Aggregator 节点（工厂函数）"""

    def modeling_aggregator_node(state: CustomModelingState) -> Dict[str, Any]:
        """收集并聚合所有产物"""
        print("--- [Aggregator] 聚合产物 ---")

        generated_files = state.get("generated_files") or {}
        completed_tasks = state.get("completed_tasks") or []
        confirmed_findings = state.get("confirmed_findings") or []
        observer_history = state.get("observer_history") or []
        stop_reason = state.get("stop_reason", "")

        # Collect JSON artifacts (exclude registered_files.json)
        modeling_artifacts = {}
        for file_name, file_info in generated_files.items():
            if file_name.endswith('.json') and file_name != 'registered_files.json':
                try:
                    content = sandbox.files.read(f"/home/user/{file_name}")
                    data = json.loads(content)
                    key = file_name.replace('.json', '')
                    modeling_artifacts[key] = data
                    print(f"  ✓ {file_name}")
                except Exception as e:
                    print(f"  ✗ 读取 {file_name} 失败: {e}")

        # Extract file names
        generated_data_files = list(generated_files.keys())

        # Use generated_files as file_metadata
        file_metadata = [
            {
                "file_name": fname,
                "description": finfo.get("description", ""),
                "columns_desc": finfo.get("columns_desc", {}),
                "type": finfo.get("type", "")
            }
            for fname, finfo in generated_files.items()
        ]

        # Construct modeling_summary
        summary_parts = []

        if stop_reason:
            summary_parts.append(f"停止原因: {stop_reason}\n")

        summary_parts.append("已完成任务:")
        for task in completed_tasks:
            summary_parts.append(f"- {task['description']}")

        summary_parts.append("\n已确认发现:")
        for finding in confirmed_findings:
            summary_parts.append(f"- {finding}")

        summary_parts.append("\n执行历史:")
        for history in observer_history:
            summary_parts.append(f"- {history}")

        summary_parts.append(f"\n生成文件: {len(generated_data_files)} 个")

        modeling_summary = "\n".join(summary_parts)

        print(f"--- [Aggregator] 完成: {len(modeling_artifacts)} 个 JSON 产物, {len(generated_data_files)} 个文件 ---")

        # Print detailed aggregation results
        print("=" * 80)
        print("[Aggregator] 汇总详情:")
        print(f"生成文件列表: {generated_data_files}")
        print(f"JSON产物: {list(modeling_artifacts.keys())}")
        print(f"文件元信息: {json.dumps(file_metadata, ensure_ascii=False, indent=2)}")
        print(f"已完成任务数: {len(completed_tasks)}")
        print(f"已确认发现数: {len(confirmed_findings)}")
        print(f"\n完整Summary:\n{modeling_summary}")
        print("=" * 80)

        return {
            "modeling_summary": modeling_summary,
            "generated_data_files": generated_data_files,
            "file_metadata": file_metadata,
            "modeling_artifacts": modeling_artifacts
        }

    return modeling_aggregator_node
