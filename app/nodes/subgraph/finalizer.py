import json
import time  # [新增] 引入时间模块
# from e2b_code_interpreter import Sandbox
from ppio_sandbox.code_interpreter import Sandbox
from app.core.subgraph.state import CustomModelingState


def create_finalizer_node(sandbox: Sandbox):
    """
    工厂函数：注入 Sandbox 依赖，返回 Finalizer 节点函数。
    """

    def finalizer_node(state: CustomModelingState):
        """
        【Finalizer 节点】
        职责：
        1. 扫描并合并 JSON 产物。
        2. 登记 Feather 数据文件。
        3. 整理 Summary 并清理 State。
        4. 统计并打印全局执行耗时。
        """
        print("--- [Subgraph] Finalizer: 正在整理产物... ---")

        # [新增] 初始化或获取 metrics
        metrics = state.get("metrics") or {}
        metrics.setdefault("llm_duration", 0.0)
        metrics.setdefault("sandbox_duration", 0.0)

        # =========================================================
        # 1. 构造“收尾”脚本 (保持不变)
        # =========================================================
        cleanup_script = """
import os
import json
import glob

# 1. 扫描所有 JSON 文件
json_files = glob.glob('*.json')
merged_artifacts = {}

for f_path in json_files:
    try:
        with open(f_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 使用文件名(去后缀)作为 Key
            key = os.path.basename(f_path).replace('.json', '')
            merged_artifacts[key] = data
    except Exception as e:
        print(f"Warning: Failed to merge {f_path}: {e}")

# 2. 将合并后的字典写回 analysis_artifacts.json
target_file = 'analysis_artifacts.json'
try:
    with open(target_file, 'w', encoding='utf-8') as f:
        json.dump(merged_artifacts, f, ensure_ascii=False, indent=4)
except Exception as e:
    print(f"Error writing merged artifacts: {e}")

# 3. 扫描 Feather (数据文件)
feather_files = glob.glob('*.feather')

# 4. 输出结果
result = {
    "generated_json_files": json_files,
    "generated_feather_files": feather_files,
    "merged_keys": list(merged_artifacts.keys())
}

print("<<FINALIZER_OUTPUT>>")
print(json.dumps(result, ensure_ascii=False))
"""

        # =========================================================
        # 2. 执行脚本与输出处理
        # =========================================================
        try:
            # [新增] 掐表开始：Finalizer 沙盒执行耗时
            sandbox_start_time = time.perf_counter()

            execution = sandbox.run_code(cleanup_script)

            # [新增] 掐表结束：累加到 sandbox_duration
            sandbox_end_time = time.perf_counter()
            current_sb_time = sandbox_end_time - sandbox_start_time
            metrics["sandbox_duration"] += current_sb_time

            # Sandbox 的 stdout 是一个 List[str]，我们需要将其合并为一个完整的字符串
            stdout_list = execution.logs.stdout
            stdout_str = "\n".join(stdout_list) if stdout_list else ""

            files_info = {}

            if stdout_str and "<<FINALIZER_OUTPUT>>" in stdout_str:
                # 提取标记后的 JSON 字符串
                try:
                    json_part = stdout_str.split("<<FINALIZER_OUTPUT>>")[1].strip()
                    files_info = json.loads(json_part)
                except (IndexError, json.JSONDecodeError) as e:
                    raise RuntimeError(f"收尾脚本输出格式解析失败: {e}\n原始内容: {stdout_str}")
            else:
                error_details = "\n".join(execution.logs.stderr) if execution.logs.stderr else "无标准错误输出"
                raise RuntimeError(
                    f"Finalizer 脚本执行异常：未检测到预期的输出标记 '<<FINALIZER_OUTPUT>>'。\n"
                    f"标准输出: {stdout_str}\n"
                    f"标准错误: {error_details}"
                )

            # =========================================================
            # 3. 更新 State & 打印耗时
            # =========================================================

            # A. 整理 Modeling Summary
            scratchpad = state.get("scratchpad")
            summary_text = "\n".join(scratchpad) if scratchpad else "分析完成。"

            # B. 提取文件列表
            feather_files = files_info.get("generated_feather_files")
            merged_keys = files_info.get("merged_keys")

            print(f"--- [Subgraph] Finalizer: 指标合并完成，Key: {merged_keys}")
            print(f"--- [Subgraph] Finalizer: 登记数据文件: {feather_files}")

            # [新增] 打印全局性能统计看板
            print("\n" + "=" * 45)
            print("[Performance Metrics] 子图执行耗时统计")
            print("=" * 45)
            total_duration = 0.0
            for k, v in metrics.items():
                print(f"  ▶ {k:<20}: {v:>8.2f} 秒")
                total_duration += v
            print("-" * 45)
            print(f"  ▶ {'Total Duration':<20}: {total_duration:>8.2f} 秒")
            print("=" * 45 + "\n")

            return {
                "modeling_summary": summary_text,
                "generated_data_files": feather_files,
                "metrics": metrics  # [新增] 将最终指标返回
            }

        except Exception as e:
            print(f"[Error] Finalizer 收尾失败: {e}")
            raise RuntimeError(f"[Error] Finalizer 收尾失败: {e}")

    return finalizer_node