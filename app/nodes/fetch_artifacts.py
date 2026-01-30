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
        scenario = state.get("scenario")
        print(f"--- [Middleware] 获取产物 (模式: {scenario}) ---")

        # 1. 基础配置
        remote_json = "/home/user/analysis_artifacts.json"

        # 本地缓存路径
        local_dir = "temp"
        os.makedirs(local_dir, exist_ok=True)
        local_json_path = os.path.join(local_dir, "analysis_artifacts.json")

        result_updates = {}
        msg_content = ""

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
            # 第二步：基于 Scenario 的分支处理
            # =========================================================

            if scenario == "custom":
                # --- 分支 A: Custom 场景 ---
                # 策略：不下载 Feather 文件。
                # 理由：数据可能很大，且可视化将在沙盒内闭环完成。
                # 此时我们信任 state["generated_data_files"] 里记录的文件列表即可。

                generated_files = state.get("generated_data_files")
                file_count = len(generated_files)

                print(f"[Custom] 跳过 Feather 下载。沙盒内现有数据文件: {file_count} 个")
                msg_content = f"分析指标已同步。数据文件保留在云端环境 ({file_count} 个)，等待可视化处理。"

            else:
                # --- 分支 B: SOP / 通用场景 ---
                # 策略：下载 processed_data.feather 并更新 Schema。
                # 理由：SOP 流程通常依赖本地对最终数据的感知。

                remote_feather = "/home/user/processed_data.feather"
                local_feather_path = os.path.join(local_dir, "processed_data.feather")

                print(f"[SOP] 正在下载全量数据: {remote_feather}...")

                try:
                    # 下载
                    feather_bytes = sandbox.files.read(remote_feather, format="bytes")
                    with open(local_feather_path, "wb") as f:
                        f.write(feather_bytes)

                    # 更新 Schema (SOP 往往需要知道数据变成什么样了)
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

                    msg_content = f"分析产物已同步到本地。检测到新增列：{new_columns if new_columns else '无'}。"
                    print("[SOP] 数据同步及 Schema 更新完成。")

                except Exception as e:
                    print(f"[Warning] SOP 数据同步失败 (可能未生成 Feather): {e}")
                    msg_content = "分析指标已同步，但数据表文件未生成。"

            # =========================================================
            # 第三步：返回更新
            # =========================================================
            return {
                **result_updates,
                "messages": [SystemMessage(content=msg_content)]
            }

        except Exception as e:
            error_msg = f"Fetch Artifacts Critical Error: {str(e)}"
            print(f"--- [Middleware Error] {error_msg} ---")
            return {
                "messages": [
                    SystemMessage(content=f"[System Warning] 产物同步异常: {error_msg}")
                ]
            }

    return fetch_artifacts_node