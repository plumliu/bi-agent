import json
import os
import sys
from langchain_core.tools import tool
from app.tools.viz.generator import generate_viz_data


@tool("viz_execution")
def viz_execution_tool(viz_config_json: str) -> str:
    """
    执行可视化配置并生成图表数据。

    【功能说明】
    根据提供的可视化配置（JSON格式），调用图表生成器生成前端所需的图表数据。
    配置会被验证并执行，如果成功会保存到本地文件。

    【输入格式】
    viz_config_json: 完整的可视化配置，必须是有效的JSON字符串，包含 charts 字段（数组）。

    【返回说明】
    - 成功：返回生成的图表类型列表
    - 失败：返回详细的错误信息，帮助你调整配置

    Args:
        viz_config_json: 可视化配置的JSON字符串
    """
    msg = "--- [Tool: Viz Execution] 开始执行可视化配置 ---"
    print(msg, flush=True)
    sys.stdout.flush()

    # 1. 准备路径
    local_dir = "temp"
    local_feather = os.path.join(local_dir, "processed_data.feather")
    local_artifacts = os.path.join(local_dir, "analysis_artifacts.json")
    local_output = os.path.join(local_dir, "viz_data.json")

    # 2. 解析配置
    try:
        viz_config = json.loads(viz_config_json)
    except json.JSONDecodeError as e:
        return f"配置解析失败：JSON格式错误 - {str(e)}"

    if not viz_config:
        return "配置为空，无法生成图表"

    # 3. 调用生成器
    result = generate_viz_data(local_feather, local_artifacts, viz_config)

    # 4. 保存结果
    try:
        with open(local_output, "w", encoding='utf-8') as f:
            json.dump(result.get("viz_data", {}), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"--- [Tool: Viz Execution] 文件保存失败: {e} ---", flush=True)
        sys.stdout.flush()

    if result["success"]:
        chart_types = list(result['viz_data']['echarts'])
        msg = f"--- [Tool: Viz Execution] 成功生成 {len(chart_types)} 个图表 ---"
        print(msg, flush=True)
        sys.stdout.flush()
        return f"成功！已生成 {len(chart_types)} 个图表。"
    else:
        # 失败处理
        errors = result.get("errors", {})
        error_msg = json.dumps(errors, ensure_ascii=False, indent=2)
        msg = f"--- [Tool: Viz Execution] 失败，错误：\n{error_msg} ---"
        print(msg, flush=True)
        sys.stdout.flush()
        return f"执行失败，错误详情：\n{error_msg}\n\n请根据错误信息调整配置后重试。"
