import json
import pandas as pd
import numpy as np
import traceback
from typing import Dict, Any, List


# ==========================================
# 0. 辅助函数
# ==========================================

def _clean_value(val: Any) -> Any:
    """清洗数据，确保 JSON 可序列化"""
    if pd.isna(val):
        return None
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        if np.isinf(val):
            return None
        return float(val)
    return val


def _clean_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """清洗整行数据"""
    return {k: _clean_value(v) for k, v in record.items()}


# ==========================================
# 1. 图表处理逻辑
# ==========================================

def _handle_scatter(df: pd.DataFrame, config: dict, artifacts: dict = None) -> dict:
    """
    Scatter Handler
    逻辑：扁平化映射，只强制注入 value 坐标，其余字段原样透传。
    """
    x_col = config.get("x_axis")
    y_col = config.get("y_axis")
    cat_col = config.get("category_col")

    # 1. 强校验 (Dead Logic: 找不到列就报错，不瞎猜)
    if x_col not in df.columns or y_col not in df.columns:
        # 即使这里，也建议直接报错，强迫 Agent 修正配置，而不是由代码去猜前两列
        raise ValueError(f"Scatter columns missing: {x_col} or {y_col} not in {df.columns.tolist()}")

    series_data = []

    # 2. 分组逻辑
    if cat_col and cat_col in df.columns:
        groups = df.groupby(cat_col)
    else:
        groups = [("All", df)]

    for group_id, group in groups:
        data_points = []
        records = group.to_dict(orient='records')

        for record in records:
            # A. 数据清洗
            clean_rec = _clean_record(record)

            # B. 构造对象
            point = clean_rec.copy()

            # C. 注入核心坐标 (这是 Scatter 图唯一必须的契约)
            point['value'] = [clean_rec.get(x_col), clean_rec.get(y_col)]

            # [已删除] 强制注入 point['name'] 的逻辑

            data_points.append(point)

        series_data.append({
            "name": f"分组 {group_id}",  # 这里的 group_id 就是 cluster_label (0, 1, 2)
            "data": data_points
        })

    return {
        "title": config.get("title", "Scatter Chart"),
        "series": series_data
    }


def _handle_radar(df: pd.DataFrame, config: dict, artifacts: dict) -> dict:
    """
    Radar Handler
    目标格式: Series Data 为对象数组
    Item Data: [ { "value": 80, "name": "维度A" }, { "value": 90, "name": "维度B" } ]
    """
    dimensions = config.get("dimensions", [])

    # 1. 数据绑定 (Data Binding)
    # 根据 Config 中的 data_key 去 Artifacts 里找数据
    data_key = config.get("data_key", "centroids")
    raw_data = artifacts.get(data_key, {})

    if not raw_data:
        # 容错：如果找不到 centroids，看看有没有同名的 key
        if "centroids" in artifacts:
            raw_data = artifacts["centroids"]
        else:
            raise ValueError(f"Radar data source '{data_key}' is empty or missing in artifacts")

    # 2. 构造 Indicator (自动计算 Max)
    indicators = []
    for dim in dimensions:
        max_val = 100
        if dim in df.columns:
            # 取该列最大值 * 1.2，并转为 float 避免 numpy 类型错误
            col_max = df[dim].max()
            if pd.notna(col_max) and not np.isinf(col_max):
                max_val = float(col_max) * 1.2
        indicators.append({"name": dim, "max": int(max_val)})

    # 3. 构造 Series
    series_list = []

    # raw_data 结构预期: { "0": {"维度A": 10, ...}, "1": {...} }
    for label, metrics in raw_data.items():

        # 构造特殊的对象数组结构
        item_data_array = []
        for dim in dimensions:
            val = metrics.get(dim, 0)
            item_data_array.append({
                "name": dim,
                "value": _clean_value(val)  # 确保是 PyFloat
            })

        series_list.append({
            "name": f"Group {label}",
            "data": item_data_array
        })

    return {
        "title": config.get("title", "Radar Chart"),
        "indicator": indicators,
        "series": series_list
    }


def _handle_bar(df: pd.DataFrame, config: dict, artifacts: dict) -> dict:
    """
    Bar Handler (智能适配器模式)
    兼容两种数据协议：
    1. List Mode (Vega-Lite Style): 数据是对象数组，需通过 config 指定 x_axis/y_axis 映射。
    2. Dict Mode (Legacy Style): 数据是字典，保留原有的 '分组 {k}' 格式化逻辑。
    """
    # Data Binding
    data_key = config.get("data_key")
    raw_data = artifacts.get(data_key)

    if raw_data is None:
        raise ValueError(f"Bar data source '{data_key}' missing")

    categories = []
    values = []
    series_name = "数量"  # 默认系列名

    # ============================================================
    # 模式 A: List of Records (显式映射 - 异常分析使用)
    # 判定依据: Config 中明确指定了 x_axis 和 y_axis
    # ============================================================
    if config.get("x_axis") and config.get("y_axis"):
        x_col = config["x_axis"]
        y_col = config["y_axis"]
        series_name = y_col  # 使用 Y 轴列名作为系列名

        # 强校验：数据必须是列表
        if not isinstance(raw_data, list):
            raise ValueError(f"Config specified axes (List Mode) but data is {type(raw_data)}. Expected List[dict].")

        for record in raw_data:
            clean_rec = _clean_record(record)
            # 提取并转字符串
            categories.append(str(clean_rec.get(x_col, "Unknown")))
            values.append(clean_rec.get(y_col, 0))

    # ============================================================
    # 模式 B: Dict Mode (隐式聚合 - 聚类分析使用)
    # 判定依据: Config 没给轴，且数据是 Dict
    # ============================================================
    elif isinstance(raw_data, dict):

        # 排序保证顺序一致
        sorted_keys = sorted(list(raw_data.keys()))

        # 提取值
        values = [int(raw_data[k]) for k in sorted_keys]

        # 格式化 X 轴 Label
        categories = [f"分组 {k}" for k in sorted_keys]

    else:
        raise ValueError(f"Unsupported data format for Bar Chart: {type(raw_data)}")

    return {
        "title": config.get("title", "Bar Chart"),
        "categories": categories,
        "series": [{
            "name": series_name,
            "data": values
        }]
    }

def _handle_pie(df: pd.DataFrame, config: dict, artifacts: dict) -> dict:
    """Pie Handler"""
    # Data Binding
    data_key = config.get("data_key", "cluster_counts")
    counts_map = artifacts.get(data_key, {})

    if not counts_map:
        raise ValueError(f"Pie data source '{data_key}' missing")

    pie_data = []
    for label, count in counts_map.items():
        pie_data.append({
            "name": f"分组 {label}",
            "value": int(count)
        })

    return {
        "title": config.get("title", "Pie Chart"),
        "series": [{
            "data": pie_data
        }]
    }


def _handle_boxplot(df: pd.DataFrame, config: dict, artifacts: dict) -> dict:
    # 1. 解析 Config & Data Binding
    dimensions = config.get("dimensions")

    key_stats = config.get("data_key_stats")
    key_outliers = config.get("data_key_outliers")

    # 从 Artifacts 获取数据源
    stats_map = artifacts.get(key_stats)  # expected: {"销售额": [-1, -0.5, 0, 0.5, 1], ...}
    outlier_list = artifacts.get(key_outliers)  # expected: List[Record]

    # 2. 校验维度有效性
    # 如果 Config 没传 dimensions，尝试使用 box_stats 的所有 key 作为默认值
    if not dimensions:
        dimensions = list(stats_map.keys())

    if not dimensions:
        raise ValueError("BoxPlot dimensions are empty. No features to visualize.")

    # 3. 构造 Series.data (Layer 1: 箱线图背景)
    # 必须保证顺序与 dimensions 列表严格一致
    box_data = []
    valid_dimensions = []  # 记录有效维度，防止 Config 里写了不存在的 Key 导致报错

    for dim in dimensions:
        if dim in stats_map:
            # stats_map[dim] 应该是长度为 5 的数组 [min, Q1, median, Q3, max]
            # 这里的数值已经是 Modeling 阶段计算好的 Scaled 值
            stat_values = [_clean_value(v) for v in stats_map[dim]]
            box_data.append(stat_values)
            valid_dimensions.append(dim)
        else:
            # 记录日志或忽略，不要让整个图挂掉
            print(f"[Viz Warning] Dimension '{dim}' 在 config 中找到，但是并不在 box_stats.")

    # 4. 构造 Series.outliers (Layer 2: 异常点前景)
    # 逻辑：笛卡尔积 (异常点 * 有效维度)
    rich_outliers = []

    for record in outlier_list:
        # 清洗基础数据 (NaN -> None, Numpy -> Python)
        clean_rec = _clean_record(record)

        # 遍历每一个有效维度，为这个异常实体生成 N 个散点 (N = valid_dimensions 数量)
        for dim_idx, dim_name in enumerate(valid_dimensions):

            # [关键约定] Modeling 阶段生成的标准化列名必须是 "{原始名}_scaled"
            scaled_key = f"{dim_name}_scaled"

            # 检查数据完整性：如果没有对应的标准化值，该点无法在 Z-Score 坐标系中定位，跳过
            if scaled_key not in clean_rec:
                continue

            val_scaled = clean_rec[scaled_key]

            # 如果值为 None (空值)，跳过
            if val_scaled is None:
                continue

            # 构造“胖对象” (Rich Object)
            # 1. 复制所有原始属性 (id, name, 原始销售额, 描述...) 供 Tooltip 使用
            point = clean_rec.copy()

            # 2. 注入 ECharts 绘图核心坐标
            # value[0]: x轴索引 (对应 valid_dimensions 中的位置)
            # value[1]: y轴数值 (标准化后的 Z-Score)
            point["value"] = [dim_idx, val_scaled]

            # 3. (可选) 注入当前维度的上下文，方便 Tooltip 高亮当前看的是哪个特征
            point["_current_feature"] = dim_name
            point["_current_scaled_val"] = val_scaled

            rich_outliers.append(point)

    # 5. 返回最终配置
    return {
        "title": config.get("title"),
        # X轴分类标签
        "categories": valid_dimensions,
        # ECharts Series 结构
        "series": [
            {
                # 前端组件会读取这个 type 来决定渲染逻辑
                "type": "boxplot",
                "name": "特征分布",
                # Layer 1: 箱体数据
                "data": box_data,
                # Layer 2: 异常散点数据 (前端组件会将其拆分为独立的 scatter series)
                "outliers": rich_outliers
            }
        ]
    }

# ==========================================
# 2. 注册表 (Registry)
# ==========================================
CHART_HANDLERS = {
    "scatter": _handle_scatter,
    "radar": _handle_radar,
    "bar": _handle_bar,
    "pie": _handle_pie,
    "boxplot": _handle_boxplot,
}


# ==========================================
# 3. 主入口 (Generate Viz Data)
# ==========================================
def generate_viz_data(local_feather_path: str, local_artifacts_path: str, viz_config: dict) -> dict:
    """
    根据配置生成前端所需的 JSON 数据
    Target Output Format:
    {
        "echarts": [
            { "type": "bar", "data": { ... } },
            { "type": "scatter", "data": { ... } }
        ]
    }
    """
    print(f"--- [Viz Generator] Starting generation... ---")

    # 1. 加载数据
    try:
        if not viz_config:
            return {"success": False, "error": "Viz config is empty"}

        df = pd.read_feather(local_feather_path)
        with open(local_artifacts_path, 'r', encoding='utf-8') as f:
            artifacts = json.load(f)

    except Exception as e:
        return {"success": False, "error": f"Data load failed: {str(e)}"}

    # [修改点 1] 初始化一个列表，而不是字典
    echarts_list = []
    errors = {}

    # 2. 遍历配置，批量生成
    charts_config = viz_config.get("charts")

    if not charts_config:
        return {"success": False, "error": "No 'charts' config found"}

    for chart_key, config in charts_config.items():
        # 获取对应的处理函数
        handler = CHART_HANDLERS.get(chart_key)

        if not handler:
            errors[chart_key] = f"No handler registered for chart type: {chart_key}"
            continue

        try:
            print(f"  -> Generating {chart_key}...")

            # 动态调用 Handler 生成具体的 data 内容
            chart_data = handler(df, config, artifacts)

            # [修改点 2] 构造对象并添加到数组
            # 结构: { "type": "bar", "data": { ... } }
            chart_item = {
                "type": chart_key,  # 例如 "bar", "boxplot"
                "data": chart_data
            }
            echarts_list.append(chart_item)

        except Exception as e:
            err_msg = f"{str(e)}"
            traceback.print_exc()
            errors[chart_key] = err_msg

    # 3. 结果判定
    has_success = len(echarts_list) > 0

    # 提取生成的类型列表用于日志
    generated_types = [item["type"] for item in echarts_list]
    print(f"--- [Viz Generator] Finished. Success: {generated_types}, Failures: {list(errors.keys())} ---")

    # [修改点 3] 返回包含 'echarts' 数组的字典
    return {
        "success": has_success,
        "viz_data": {
            "echarts": echarts_list
        },
        "errors": errors
    }