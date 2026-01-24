import json
import os
import pandas as pd
# from e2b_code_interpreter import Sandbox
from ppio_sandbox.code_interpreter import Sandbox
from langchain_core.messages import SystemMessage
from app.core.state import AgentState


def create_fetch_artifacts_node(sandbox: Sandbox):
    """
    工厂函数：注入 Sandbox 依赖，返回节点函数
    """

    def fetch_artifacts_node(state: AgentState):
        print("--- [Middleware] 获取产物以及更新data_schema ---")

        # 定义路径 (与 Modeling 阶段的协议保持一致)
        remote_json = "/home/user/analysis_artifacts.json"
        remote_feather = "/home/user/processed_data.feather"

        # 本地缓存路径 (FastAPI 中通常使用临时目录或对象存储)
        local_dir = "temp"
        os.makedirs(local_dir, exist_ok=True)
        local_json_path = os.path.join(local_dir, "analysis_artifacts.json")
        local_feather_path = os.path.join(local_dir, "processed_data.feather")

        try:
            # 1. 下载 JSON (统计产物) 并保存到本地
            print(f"下载 {remote_json}...")
            json_str = sandbox.files.read(remote_json)
            artifacts = json.loads(json_str)

            # [修复] 必须保存到本地，供 Viz Execution 阶段读取
            with open(local_json_path, "w", encoding="utf-8") as f:
                json.dump(artifacts, f, ensure_ascii=False, indent=2)

            # 2. 下载 Feather (数据全集) 并保存到本地
            print(f"下载 {remote_feather} 中...")
            # 注意：必须用 format="bytes" 下载二进制
            feather_bytes = sandbox.files.read(remote_feather, format="bytes")

            with open(local_feather_path, "wb") as f:
                f.write(feather_bytes)

            # 3. 读取最新的 Schema
            df_full = pd.read_feather(local_feather_path)
            df_meta = df_full.head(2)

            columns = df_meta.columns.tolist()
            dtypes = df_meta.dtypes.astype(str).to_dict()
            samples = df_meta.to_dict(orient='records')

            # 获取旧的列名列表
            old_schema = state.get("data_schema")
            old_columns = old_schema.get("columns")

            new_columns = list(set(columns) - set(old_columns))

            result = {
                "columns": columns,
                "dtypes": dtypes,
                "samples": samples,
                "summary": f"建模阶段新增列: {new_columns}" if new_columns else "无新增列"
            }

            # 构造成功消息
            msg_content = f"已成功同步分析产物至本地。检测到新增列：{new_columns if new_columns else '无'}。"

            print(f"--- [Middleware] Schema已更新: 检测到了 {len(result['columns'])} 列 ---")
            print(f"--- [Middleware] {msg_content} ---")

            # 4. 更新 State
            # 这里返回的字典会合并到 State 中
            return {
                "data_schema": result,  # 更新元数据
                "modeling_artifacts": artifacts,
                "messages": [SystemMessage(content=msg_content)]
            }

        except Exception as e:
            error_msg = f"产物获取失败: {str(e)}"
            print(f"--- [Middleware Error] {error_msg} ---")
            # 如果失败，返回错误消息
            return {
                "messages": [
                    SystemMessage(content=f"[System Warning] 产物获取失败: {error_msg}！ Viz阶段可能会因此失败。")
                ]
            }

    return fetch_artifacts_node