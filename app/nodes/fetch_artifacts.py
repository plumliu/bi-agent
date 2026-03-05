import json
import os
import pandas as pd
# from e2b_code_interpreter import Sandbox
from ppio_sandbox.code_interpreter import Sandbox
from langchain_core.messages import SystemMessage, AIMessage
from app.core.state import WorkflowState


def create_fetch_artifacts_node(sandbox: Sandbox):
    """
    工厂函数：注入 Sandbox 依赖，返回节点函数
    """

    def fetch_artifacts_node(state: WorkflowState):
        print("--- [Middleware] 获取产物 (SOP 模式) ---")

        # 1. 基础配置
        remote_json = "/home/user/analysis_artifacts.json"

        # 本地缓存路径
        local_dir = "temp"
        os.makedirs(local_dir, exist_ok=True)
        local_json_path = os.path.join(local_dir, "analysis_artifacts.json")

        result_updates = {}

        try:
            # =========================================================
            # 第一步：总是同步 JSON (关键指标与元数据)
            # =========================================================
            print(f"正在下载指标数据: {remote_json}...")
            try:
                json_str = sandbox.files.read(remote_json)
                artifacts = json.loads(json_str)

                # 保存到本地 (供 Summary 或 前端 读取)
                with open(local_json_path, "w", encoding="utf-8") as f:
                    json.dump(artifacts, f, ensure_ascii=False, indent=2)

                result_updates["modeling_artifacts"] = artifacts
                print("指标数据同步成功。")

            except Exception as e:
                raise RuntimeError(f"无法下载/解析 analysis_artifacts.json: {e}")

            # =========================================================
            # 第二步：下载 processed_data.feather 并更新 Schema
            # =========================================================
            remote_feather = "/home/user/processed_data.feather"
            local_feather_path = os.path.join(local_dir, "processed_data.feather")

            print(f"正在下载全量数据: {remote_feather}...")

            try:
                # 下载
                feather_bytes = sandbox.files.read(remote_feather, format="bytes")
                with open(local_feather_path, "wb") as f:
                    f.write(feather_bytes)

                # 更新 Schema
                df_full = pd.read_feather(local_feather_path)
                df_meta = df_full.head(2)

                columns = df_meta.columns.tolist()
                dtypes = df_meta.dtypes.astype(str).to_dict()
                samples = df_meta.to_dict(orient='records')

                # 简单对比新增列
                old_schema = state.get("data_schema", {})
                old_columns = old_schema.get("columns", [])
                new_columns = list(set(columns) - set(old_columns))

                result_updates["data_schema"] = {
                    "columns": columns,
                    "dtypes": dtypes,
                    "samples": samples,
                    "summary": f"新增列: {new_columns}" if new_columns else "无新增列"
                }

                print("数据同步及 Schema 更新完成。")

            except Exception as e:
                print(f"[Warning] 数据同步失败 (可能未生成 Feather): {e}")

            # =========================================================
            # 第三步：返回更新
            # =========================================================
            return result_updates

        except Exception as e:
            error_msg = f"Fetch Artifacts Critical Error: {str(e)}"
            print(f"--- [Middleware Error] {error_msg} ---")

            return {}

    return fetch_artifacts_node