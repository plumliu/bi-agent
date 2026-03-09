import os
import json
import glob
from typing import Dict, Any

from app.core.viz_custom_subgraph.state import CustomVizState


def generate_viz_config_from_viz_data(viz_data: dict) -> dict:
    """从 viz_data 生成 viz_config"""
    charts_config = {}

    echarts_list = viz_data.get("echarts", [])

    for idx, chart_item in enumerate(echarts_list):
        chart_type = chart_item.get("type", "unknown")
        chart_data = chart_item.get("data", {})
        title = chart_data.get("title", f"图表 {idx+1}")

        # 生成唯一的 chart_key
        if idx == 0:
            chart_key = chart_type
        else:
            chart_key = f"{chart_type}_{idx+1}"

        # 构造基础配置
        charts_config[chart_key] = {
            "type": chart_type,
            "title": title
        }

    return {
        "charts": charts_config
    }


def viz_aggregator_node(state: CustomVizState) -> Dict[str, Any]:
    """
    【Viz Aggregator 节点】
    职责：
    1. 收集所有 viz_executor 下载的本地文件
    2. 合并为统一的 viz_data.json 格式
    3. 生成 viz_config（新增）
    4. 保存到 temp/viz_data.json

    注意：这是纯本地操作，不涉及沙箱交互
    """
    print("--- [Viz Subgraph] Aggregator: 正在合并可视化数据... ---")

    # 1. 获取 viz_tasks 列表
    viz_tasks = state.get("viz_tasks", [])
    if not viz_tasks:
        print("--- [Viz Subgraph] Aggregator: 警告 - 没有 viz_tasks ---")
        return {}

    # 2. 收集所有 executor 生成的文件
    echarts_data = []
    success_count = 0
    failed_tasks = []

    for task in viz_tasks:
        task_id = task["task_id"]
        local_path = f"temp/viz_{task_id}.json"

        try:
            # 读取文件
            if not os.path.exists(local_path):
                print(f"--- [Viz Subgraph] Aggregator: 警告 - {local_path} 不存在 ---")
                failed_tasks.append(task_id)
                continue

            with open(local_path, 'r', encoding='utf-8') as f:
                chart_data = json.load(f)

            # 验证格式
            if "type" not in chart_data or "data" not in chart_data:
                print(f"--- [Viz Subgraph] Aggregator: 警告 - {task_id} 格式不正确 ---")
                failed_tasks.append(task_id)
                continue

            # 添加到 echarts 数组
            echarts_data.append(chart_data)
            success_count += 1
            print(f"--- [Viz Subgraph] Aggregator: ✓ {task_id} ({chart_data['type']}) ---")

        except Exception as e:
            print(f"--- [Viz Subgraph] Aggregator: 错误 - {task_id}: {e} ---")
            failed_tasks.append(task_id)

    # 3. 合并为统一格式
    viz_data = {
        "echarts": echarts_data
    }

    # 4. 生成 viz_config（新增）
    viz_config = generate_viz_config_from_viz_data(viz_data)

    # 5. 保存到 temp/viz_data.json
    try:
        os.makedirs("temp", exist_ok=True)
        output_path = "temp/viz_data.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(viz_data, f, ensure_ascii=False, indent=2)

        print(f"--- [Viz Subgraph] Aggregator: 成功合并 {success_count}/{len(viz_tasks)} 个图表 ---")
        print(f"--- [Viz Subgraph] Aggregator: 数据已保存到 {output_path} ---")

        if failed_tasks:
            print(f"--- [Viz Subgraph] Aggregator: 失败任务: {failed_tasks} ---")

        # 6. 返回状态更新（新增 viz_config 和 viz_success）
        return {
            "viz_data": viz_data,
            "viz_config": viz_config,
            "viz_success": True
        }

    except Exception as e:
        error_msg = f"--- [Viz Subgraph] Aggregator: 保存失败 - {e} ---"
        print(error_msg)
        return {}
