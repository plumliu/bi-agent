import os
import json
from typing import Dict, Any
from ppio_sandbox.code_interpreter import Sandbox
from app.core.modeling_custom_subgraph.state import CustomModelingState


def create_modeling_aggregator_node(sandbox: Sandbox):
    """创建 modeling aggregator 节点（工厂函数）"""

    def modeling_aggregator_node(state: CustomModelingState) -> Dict[str, Any]:
        """
        收集并合并 sandbox 中的所有 JSON 产物
        """
        print("--- [Modeling Subgraph] Aggregator: 正在收集产物... ---")

        try:
            # 扫描 sandbox 中的所有 JSON 文件
            ls_result = sandbox.commands.run("ls /home/user/*.json 2>/dev/null || true")

            if not ls_result.stdout or not ls_result.stdout.strip():
                print("--- [Modeling Subgraph] Aggregator: 未发现 JSON 文件 ---")
                return {"modeling_artifacts": {}}

            json_files = [f.strip() for f in ls_result.stdout.strip().split('\n') if f.strip()]
            print(f"--- [Modeling Subgraph] Aggregator: 发现 {len(json_files)} 个 JSON 文件 ---")

            # 合并所有 JSON 文件
            merged_artifacts = {}

            for json_path in json_files:
                try:
                    # 读取文件内容
                    content = sandbox.files.read(json_path)
                    data = json.loads(content)

                    # 使用文件名（不含扩展名）作为 key
                    file_name = os.path.basename(json_path).replace('.json', '')
                    merged_artifacts[file_name] = data

                    print(f"  ✓ {file_name}.json")

                except Exception as e:
                    print(f"  ✗ 读取 {json_path} 失败: {e}")

            print(f"--- [Modeling Subgraph] Aggregator: 成功合并 {len(merged_artifacts)} 个文件 ---")

            return {"modeling_artifacts": merged_artifacts}

        except Exception as e:
            print(f"--- [Modeling Subgraph] Aggregator: 错误 - {e} ---")
            return {"modeling_artifacts": {}}

    return modeling_aggregator_node
