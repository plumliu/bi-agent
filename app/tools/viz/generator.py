import json
import pandas as pd
import datetime
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

    if isinstance(val, (pd.Timestamp, datetime.datetime, datetime.date)):
        return str(val)

    if isinstance(val, (np.bool_, bool)):
        return bool(val)

    if isinstance(val, np.datetime64):
        return str(val)

    return val

def _clean_series(series: pd.Series) -> list:
    if series.empty:
        return []

    cleaned = series.replace([np.inf, -np.inf], np.nan)

    valid_mask = cleaned.notna()
    if not valid_mask.any():
        return []

    first_valid = valid_mask.idxmax()
    last_valid = valid_mask[::-1].idxmax()

    trimmed = cleaned.loc[first_valid:last_valid]

    if trimmed.isna().any():
        interpolated = trimmed.interpolate(method='linear')
        interpolated = interpolated.bfill().ffill()
        if interpolated.isna().any():
            interpolated = interpolated.fillna(0)
        trimmed = interpolated

    return trimmed.tolist()



def _clean_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """清洗整行数据"""
    return {k: _clean_value(v) for k, v in record.items()}


# ==========================================
# 1. 图表处理逻辑
# ==========================================

def _handle_scatter(df: pd.DataFrame, config: dict, artifacts: dict = None) -> dict:
    """
    Scatter Handler (Universal Version)
    支持2种模式：
    1. Artifacts Mode: 从统计产物读取
    3. DataFrame Mode (Grouped): 通过 y_axis + category_col 自动分组
    """
    title = config.get("title")
    series_data = []

    # =================================================
    # 模式 A: Artifacts Mode
    # =================================================
    if config.get("data_source") == "artifacts":
        pass

    # =================================================
    # 模式 B: DataFrame Mode
    # =================================================
    else:
        x_col = config.get("x_axis")
        if not x_col:
            raise ValueError("Scatter Chart 需要 'x_axis'.")
        if x_col not in df.columns:
            raise ValueError(f"Scatter Chart 缺少 X 轴列: '{x_col}'")

        else:
            y_col = config.get("y_axis")
            if not y_col:
                raise ValueError("散点图需要 'y_axis'")
            if y_col not in df.columns:
                raise ValueError(f"列 '{y_col}' 未找到！")

            cat_col = config.get("category_col", None)

            # 分组逻辑
            if cat_col and cat_col in df.columns:
                groups = df.groupby(cat_col)
            else:
                groups = [("默认分组", df)]

            for group_id, group in groups:
                data_points = []
                records = group.to_dict(orient='records')

                for record in records:
                    clean_rec = _clean_record(record)
                    point = clean_rec.copy()
                    # 构造 value: [x, y]
                    point['value'] = [clean_rec.get(x_col), clean_rec.get(y_col)]
                    data_points.append(point)

                s_name = f"{group_id}" if cat_col else config.get("series_name", "系列 1")

                series_data.append({
                    "name": s_name,
                    "data": data_points
                })

    return {
        "title": title,
        "series": series_data
    }

def _handle_radar(df: pd.DataFrame, config: dict, artifacts: dict) -> dict:
    """
    Radar Handler (Unified Architecture)
    支持两种模式：
    1. Artifacts Mode: 从统计结果(如聚类中心)读取，自动计算 max 值。
    2. DataFrame Mode: (预留)
    """
    title = config.get("title")
    dimensions = config.get("dimensions")
    series_list = []
    indicators = []

    # =================================================
    # 模式 A: Artifacts Mode (从统计产物直接读取)
    # =================================================
    if config.get("data_source") == "artifacts":
        data_key = config.get("data_key")
        raw_data = artifacts.get(data_key)  # Expecting Dict {label: {dim: val}}

        if not raw_data or not isinstance(raw_data, dict):
            raise ValueError(f"Radar Chart artifact source '{data_key}' is missing or not a dictionary.")

        # --- 1. 动态计算每个维度的 Max 值 (基于 Artifacts 数据) ---
        # 遍历所有分组，找出每个维度的最大值
        dim_max_map = {dim: 0.0 for dim in dimensions}

        for group_metrics in raw_data.values():
            for dim in dimensions:
                val = group_metrics.get(dim, 0)
                if val > dim_max_map.get(dim, 0):
                    dim_max_map[dim] = float(val)

        # 构建 Indicator 配置 (Max * 1.2 留出视觉余量)
        for dim in dimensions:
            current_max = dim_max_map.get(dim, 100.0)
            # 防止最大值为0导致图表压扁，给个默认底限
            final_max = current_max * 1.2 if current_max > 0 else 100.0
            indicators.append({
                "name": dim,
                "max": int(final_max)
            })

        # --- 2. 构造 Series ---
        # 支持自定义命名模板，例如 "Cluster {}"
        name_template = config.get("name_template", "分组 {}")

        # 排序 Key 保证颜色稳定性
        for label in sorted(raw_data.keys()):
            metrics = raw_data[label]

            # 构造 ECharts 雷达图所需的对象数组结构
            item_data_array = []
            for dim in dimensions:
                val = metrics.get(dim, 0)
                item_data_array.append({
                    "name": dim,
                    "value": _clean_value(val)
                })

            series_list.append({
                "name": name_template.format(label),
                "data": item_data_array
            })

    # =================================================
    # 模式 B: DataFrame Mode (从列读取)
    # =================================================
    else:
        # 这里暂时留空，或者未来实现 "按某列分组取均值后画雷达图" 的逻辑
        pass

    return {
        "title": title,
        "indicator": indicators,
        "series": series_list
    }

def _handle_bar(df: pd.DataFrame, config: dict, artifacts: dict) -> dict:
    """
    Bar Handler (智能适配器模式 - 统一架构版)
    逻辑分流：
    1. Artifacts Mode:
       - Sub-Case A (List): 对象数组 (如 Top Anomalies)，需显式指定 x/y 键。
       - Sub-Case B (Dict): 字典映射 (如 Clustering Counts)，自动解析 Key/Value。
    2. DataFrame Mode:
       - Standard: 从 df 列中读取 x/y。
    """
    title = config.get("title", "Bar Chart")
    categories = []
    values = []
    series_name = config.get("series_name", "数量")  # 默认系列名

    # =================================================
    # 模式 A: Artifacts Mode (从统计产物直接读取)
    # =================================================
    if config.get("data_source") == "artifacts":
        data_key = config.get("data_key")
        raw_data = artifacts.get(data_key)

        if raw_data is None:
            raise ValueError(f"Bar Chart data source '{data_key}' missing in artifacts.")

        # --- Sub-Case A: List of Records (显式映射 - 例如 Top 10 异常) ---
        if isinstance(raw_data, list):
            x_key = config.get("x_axis")
            y_key = config.get("y_axis")

            if not x_key or not y_key:
                raise ValueError("Bar Chart (Artifacts List Mode) requires 'x_axis' and 'y_axis' keys.")

            series_name = config.get("series_name", y_key)  # 如果没指定系列名，用字段名

            for record in raw_data:
                # 假设 _clean_record 是个辅助函数，处理 NaN 等
                clean_rec = _clean_record(record) if '_clean_record' in globals() else record
                categories.append(str(clean_rec.get(x_key, "Unknown")))
                values.append(clean_rec.get(y_key, 0))

        # --- Sub-Case B: Dict Mode (隐式聚合 - 例如聚类统计) ---
        elif isinstance(raw_data, dict):
            # 兼容旧逻辑：Clustering 产出的 counts 是 {"0": 50, "1": 30}
            sorted_keys = sorted(list(raw_data.keys()))

            values = [int(raw_data[k]) for k in sorted_keys]
            # 保留原有的 "分组 {k}" 格式化逻辑，或者由 Config 决定
            categories = [f"分组 {k}" for k in sorted_keys]

        else:
            raise ValueError(f"Unsupported data format for Bar Chart artifacts: {type(raw_data)}")

    # =================================================
    # 模式 B: DataFrame Mode (从列读取)
    # =================================================
    else:
        x_col = config.get("x_axis")
        y_col = config.get("y_axis")

        if not x_col or not y_col:
            raise ValueError("Bar Chart (DataFrame Mode) requires 'x_axis' and 'y_axis'.")

        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Bar columns missing: {x_col} or {y_col} not found.")

        # 提取数据
        categories = df[x_col].astype(str).tolist()
        # 简单清洗数据，防止 NaN 传给前端炸裂
        values = df[y_col].fillna(0).tolist()

        # 使用 Y 轴列名作为系列名
        series_name = config.get("series_name", y_col)

    return {
        "title": title,
        "categories": categories,
        "series": [{
            "name": series_name,
            "type": "bar",
            "data": values
        }]
    }

def _handle_pie(df: pd.DataFrame, config: dict, artifacts: dict) -> dict:
    """
    Pie Handler (Unified Architecture)
    支持两种模式：
    1. Artifacts Mode: 读取字典 {label: count} (如聚类结果)。
    2. DataFrame Mode: 指定一列，自动执行 value_counts() 进行聚合统计。
    """
    title = config.get("title", "Pie Chart")
    pie_data = []

    # =================================================
    # 模式 A: Artifacts Mode (从统计产物直接读取)
    # =================================================
    if config.get("data_source") == "artifacts":
        data_key = config.get("data_key")
        raw_data = artifacts.get(data_key)

        if not raw_data or not isinstance(raw_data, dict):
            raise ValueError(f"Pie Chart artifact source '{data_key}' is missing or not a dictionary.")

        # 如果是聚类场景，Viz Config 可以传入 name_template: "分组 {}"
        name_template = config.get("name_template", "分组 {}")

        # 排序 Key 以保证颜色分配稳定性
        for label in sorted(raw_data.keys()):
            count = raw_data[label]
            pie_data.append({
                "name": name_template.format(label),
                "value": int(count)
            })

    # =================================================
    # 模式 B: DataFrame Mode (自动聚合)
    # =================================================
    else:
        # 用户指定一个分类列，我们统计该列中各值的出现次数
        cat_col = config.get("category_col")

        if not cat_col:
            raise ValueError("Pie Chart (DataFrame Mode) requires 'category_col' to aggregate.")

        if cat_col not in df.columns:
            raise ValueError(f"Pie Chart column '{cat_col}' not found in DataFrame.")

        # 执行聚合统计
        counts = df[cat_col].value_counts().sort_index()

        for label, count in counts.items():
            pie_data.append({
                "name": str(label),
                "value": int(count)
            })

    return {
        "title": title,
        "series": [{
            "type": "pie",
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

def _handle_decomposition(df: pd.DataFrame, config: dict, artifacts: dict) -> dict:
    """
    Decomposition Handler (专用于 STL 分解结果展示)
    结构: Trend, Seasonal, Residual, Observed
    """
    x_col = config.get("x_axis")
    # 获取时间轴
    categories = df[x_col].astype(str).tolist() if x_col in df.columns else list(df.index.astype(str))

    # 定义固定的四个组件，Agent 只需要告诉我们将哪一列映射到这四个组件上
    # 映射关系: { 组件名: 列名 }
    component_map = {
        "Trend": config.get("trend_col"),
        "Seasonal": config.get("seasonal_col"),
        "Residual": config.get("residual_col"),
        "Observed": config.get("observed_col")
    }

    series_data = []

    # 按照特定顺序生成 (符合人类阅读习惯：Trend -> Seasonal -> Resid -> Raw)
    # 或者按照你要求的顺序：Trend, Seasonal, Resid, Observed
    order = ["Trend", "Seasonal", "Residual", "Observed"]

    for comp_name in order:
        col_name = component_map.get(comp_name)

        if col_name and col_name in df.columns:
            data_vals = df[col_name].fillna(0).tolist()
            cleaned_vals = [_clean_value(v) for v in data_vals]

            series_data.append({
                "name": comp_name,
                "data": cleaned_vals
            })

    return {
        "title": config.get("title", "Decomposition"),
        "categories": categories,
        "series": series_data
    }

def _handle_line_error(df: pd.DataFrame, config: dict, artifacts: dict) -> dict:
    """
    LineWithError Handler (通用误差棒折线图)
    支持两种模式：
    1. Artifacts Mode: (预留)
    2. DataFrame Mode: 根据 series_mapping 遍历生成多条带有 errors 的折线。
    """
    title = config.get("title", "Error Chart")
    categories = []
    series_data = []

    # =================================================
    # 模式 A: Artifacts Mode (从统计产物直接读取)
    # =================================================
    if config.get("data_source") == "artifacts":
        # 暂时留空，保持架构一致性
        pass

    # =================================================
    # 模式 B: DataFrame Mode (从列读取)
    # =================================================
    else:
        # 1. 通用 X 轴处理
        x_col = config.get("x_axis")
        if x_col not in df.columns:
            # 容错：如果没有指定 X 轴，使用索引
            categories = list(df.index.astype(str))
        else:
            categories = df[x_col].astype(str).tolist()

        # 2. 解析 Series Mapping
        series_mapping = config.get("series_mapping", [])
        if not series_mapping:
            raise ValueError("Config 'series_mapping' is empty for line_error chart.")

        for mapping in series_mapping:
            name = mapping.get("name")
            y_col = mapping.get("y_col")
            error_col = mapping.get("error_col")  # 可为 None

            # 获取 Y 轴数据 (Main Data)
            if y_col not in df.columns:
                # 记录警告或跳过，这里选择填充 0 以防崩坏
                main_data = [0] * len(df)
            else:
                main_data = df[y_col].fillna(0).tolist()
                main_data = [_clean_value(v) for v in main_data]

            # 获取 Error 数据
            if error_col and error_col in df.columns:
                errors_data = df[error_col].fillna(0).tolist()
                errors_data = [_clean_value(v) for v in errors_data]
            else:
                # 如果没指定 error_col，或者列不存在，则误差为 0
                errors_data = [0] * len(df)

            # 组装 Series 对象
            series_data.append({
                "name": name,
                "data": main_data,
                "errors": errors_data
            })

    return {
        "title": title,
        "categories": categories,
        "series": series_data
    }

def _handle_line(df: pd.DataFrame, config: dict, artifacts: dict) -> dict:
    title = config.get("title")
    series_list = []
    categories = []

    # =================================================
    # 模式 A: Artifacts Mode (从统计产物直接读取)
    # =================================================
    if config.get("data_source") == "artifacts":
        data_key = config.get("data_key")
        raw_data = artifacts.get(data_key)  # [1,2,3] 或 [[1,2], [3,4]]

        if not raw_data or not isinstance(raw_data, list):
            raise ValueError(f"数据源 '{data_key}' 本身不存在，无法绘制")

        # --- [1. 处理 X 轴 Categories] ---
        # 路径 A: Config 直接给定
        if "categories" in config and isinstance(config["categories"], list):
            categories = [str(x) for x in config["categories"]]
        # 路径 B: Config 指定 Key (推荐)
        elif "categories_key" in config:
            cat_key = config["categories_key"]
            cat_data = artifacts.get(cat_key)
            if isinstance(cat_data, list):
                categories = [str(x) for x in cat_data]
            else:
                raise ValueError(f"指定了 'categories_key'='{cat_key}'，但 artifacts 中不是列表。")

        if not categories:
            raise ValueError(f"LineChart (Artifacts Mode) 缺少 Category 数据，无法绘制 X 轴！")

        # --- [2. 处理 Series] ---
        is_multi_series = len(raw_data) > 0 and isinstance(raw_data[0], list)

        if is_multi_series:
            # === Case A1: 二维数组 (多系列) ===
            series_names = []
            if "series_names_key" in config:
                sn_key = config["series_names_key"]
                sn_data = artifacts.get(sn_key)
                if not isinstance(sn_data, list):
                    raise ValueError(f"在 artifacts 中通过 series_names_key='{sn_key}' 指定的系列名称数据不是列表类型")
                series_names = sn_data

            if len(series_names) != len(raw_data):
                raise ValueError( f"系列名称数量不匹配：共 {len(raw_data)} 个系列，但提供了 {len(series_names)} 个名称")

            for i, cycle_data in enumerate(raw_data):
                if len(cycle_data) != len(categories):
                    raise ValueError(f"系列 {i} 的长度（{len(cycle_data)}）与类别长度（{len(categories)}）不相等")

                series_list.append({
                    "name": series_names[i],
                    "data": cycle_data
                })

        else:
            # === Case A2: 一维数组 (单线) ===
            # 单线模式下，系列名通常由 Config 直接指定 (例如 "平均周期")
            series_name = config.get("series_name", "曲线")

            if len(raw_data) != len(categories):
                raise ValueError(f"系列长度（{len(raw_data)}）与类别长度（{len(categories)}）不相等")

            series_list.append({
                "name": series_name,
                "data": raw_data
            })

    # =================================================
    # 模式 B: DataFrame Mode (从列读取)
    # =================================================
    else:
        # (这部分逻辑保持不变，它非常稳定)
        x_col = config.get("x_axis")
        if not x_col:
            raise ValueError("DataFrame Mode 必须指定 'x_axis'")

        if x_col in df.columns:
            categories = df[x_col].astype(str).tolist()
        else:
            raise ValueError(f"x_axis column '{x_col}' not found")

        y_input = config.get("y_axis")
        y_cols = [y_input] if isinstance(y_input, str) else y_input

        if not y_cols:
            raise ValueError("DataFrame Mode 必须指定 'y_axis'")

        for col in y_cols:
            if col in df.columns:
                series_list.append({
                    "name": col,  # 使用列名
                    "data": _clean_series(df[col])
                })
            else:
                raise ValueError(f"y_axis column '{col}' not found")

    return {
        "title": title,
        "categories": categories,
        "series": series_list
    }

def _handle_heatmap(df: pd.DataFrame, config: dict, artifacts: dict) -> dict:
    """
    Heatmap Handler (轻量化协议版)

    功能:
    1. 解析 Artifacts 中的相关性矩阵。
    2. 转换为 [x, y, value] 数组。
    3. 仅传递 min/max 阈值，把颜色策略交还给前端组件。
    """
    title = config.get("title", "特征关联热力图")
    # Num-Num (Spearman): min=-1, max=1
    # Cat-Cat (CramerV) / Num-Cat (Eta): min=0, max=1
    visual_map = config.get("visual_map")

    x_categories = []
    y_categories = []
    data_points = []

    # =================================================
    # 模式 A: Artifacts Mode
    # =================================================
    if config.get("data_source") == "artifacts":
        data_key = config.get("data_key")
        raw_matrix = artifacts.get(data_key)

        if not raw_matrix or not isinstance(raw_matrix, dict):
            raise ValueError(f"Heatmap data source '{data_key}' error.")

        # 1. 解析 Y 轴 (行)
        y_categories = sorted(list(raw_matrix.keys()))
        if not y_categories:
            raise ValueError("Heatmap matrix is empty.")

        # 2. 解析 X 轴 (列)
        first_row_key = y_categories[0]
        x_categories = sorted(list(raw_matrix[first_row_key].keys()))

        # 3. 扁平化数据
        for y_idx, y_key in enumerate(y_categories):
            row_data = raw_matrix[y_key]
            for x_idx, x_key in enumerate(x_categories):
                val = row_data.get(x_key, 0)

                clean_val = 0.0
                if pd.notna(val) and not np.isinf(val):
                    clean_val = float(val)

                data_points.append([x_idx, y_idx, clean_val])

    else:
        pass

    return {
        "title": title,
        "categories": y_categories,  # Y轴
        "categories2": x_categories,  # X轴
        "series": [{
            "data": data_points
        }],
        "visualMap": visual_map
    }

def _handle_line_confidence(df: pd.DataFrame, config: dict, artifacts: dict = None) -> dict:
    """
    Line Forecast with Confidence Interval Handler
    支持两种模式：
    1. Artifacts Mode: (预留接口)
    2. DataFrame Mode: 标准列映射，支持 lowers/uppers 区间数据
    """
    title = config.get("title", "时序预测分析")
    series_list = []
    categories = []

    # =================================================
    # 模式 A: Artifacts Mode
    # =================================================
    if config.get("data_source") == "artifacts":
        pass

    # =================================================
    # 模式 B: DataFrame Mode
    # =================================================
    else:
        # 1. X 轴处理
        x_col = config.get("x_axis")
        if not x_col:
            raise ValueError("DataFrame Mode 必须指定 'x_axis'")

        if x_col not in df.columns:
            raise ValueError(f"Confidence Chart 缺少 X 轴列: '{x_col}'")

        # [修正点 1]: X 轴数据也需要清洗 (防止 Timestamp 报错)
        # 虽然 astype(str) 很有用，但用 _clean_value 更保险
        raw_categories = df[x_col].tolist()
        categories = [_clean_value(x) for x in raw_categories]

        # 2. 提取配置
        series_configs = config.get("series_config", [])
        if not series_configs:
            raise ValueError("Confidence Chart (DataFrame Mode) 必须提供 'series_config' 配置")

        # 3. 构建 Series
        for i, sc in enumerate(series_configs):
            s_name = sc.get("name", f"系列{i + 1}")
            d_col = sc.get("data_col")
            l_col = sc.get("lower_col")
            u_col = sc.get("upper_col")

            # 强校验：Data 列必须存在
            if not d_col or d_col not in df.columns:
                continue

            if l_col and l_col not in df.columns:
                raise ValueError(f"系列 '{s_name}' 配置了 lower_col='{l_col}' 但数据中不存在该列")

            if u_col and u_col not in df.columns:
                raise ValueError(f"系列 '{s_name}' 配置了 upper_col='{u_col}' 但数据中不存在该列")

            raw_data = df[d_col].tolist()
            clean_data = [_clean_value(v) for v in raw_data]

            series_item = {
                "name": s_name,
                "data": clean_data
            }

            if l_col:
                raw_lower = df[l_col]
                series_item["lowers"] = [_clean_value(v) for v in raw_lower]

            if u_col:
                raw_upper = df[u_col]
                series_item["uppers"] = [_clean_value(v) for v in raw_upper]

            series_list.append(series_item)

        if not series_list:
            raise ValueError("Confidence Chart 未提取到有效数据列")

    return {
        "title": title,
        "categories": categories,
        "series": series_list
    }

def _handle_table(df: pd.DataFrame, config: dict, artifacts: dict) -> dict:
    """
    Table Handler
    支持 Artifacts Mode (例如 top_anomalies 列表) 和 DataFrame Mode (预览原始数据)
    Protocol:
    {
      "title": "...",
      "categories": ["Col1", "Col2"],
      "series": [ [Val1, Val2], [Val3, Val4] ... ]
    }
    """
    title = config.get("title", "表格详情")
    categories = []  # 表头
    series = []  # 数据行

    # --- Mode A: Artifacts (List[Dict]) ---
    if config.get("data_source") == "artifacts":
        data_key = config.get("data_key")
        if data_key and artifacts and data_key in artifacts:
            raw_list = artifacts[data_key]

            if isinstance(raw_list, list) and len(raw_list) > 0:
                # 1. 确定表头 (Categories)
                # 优先使用配置的 columns，否则使用第一条数据的 keys
                target_cols = config.get("columns")
                if not target_cols:
                    target_cols = list(raw_list[0].keys())

                categories = target_cols

                # 2. 构造数据 (Series)
                for i, item in enumerate(raw_list):
                    row = []
                    for col in target_cols:
                        val = item.get(col)
                        row.append(_clean_value(val))
                    series.append(row)

    # --- Mode B: DataFrame ---
    else:
        # 1. 确定表头
        target_cols = config.get("columns")
        if not target_cols:
            target_cols = df.columns.tolist()
        else:
            # 过滤掉不存在的列
            target_cols = [c for c in target_cols if c in df.columns]

        categories = target_cols

        # 2. 构造数据
        for _, record in df[target_cols].iterrows():
            row = [_clean_value(record[c]) for c in target_cols]
            series.append(row)

    return {
        "title": title,
        "categories": categories,
        "series": series
    }

# ==========================================
# 2. 注册表 (Registry)
# ==========================================
CHART_HANDLERS = {
    "scatter": _handle_scatter,
    "radar": _handle_radar,
    "bar": _handle_bar,
    "line": _handle_line,
    "pie": _handle_pie,
    "boxplot": _handle_boxplot,
    "heatmap": _handle_heatmap,
    "decomposition": _handle_decomposition,
    "lineWithErrorBars": _handle_line_error,
    "lineWithConfidence": _handle_line_confidence,
    "table": _handle_table
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
    print(f"--- [Viz Generator] 开始生成图表数据... ---")

    # 1. 加载数据
    try:
        if not viz_config:
            return {"success": False, "error": "Viz config is empty"}

        df = pd.read_feather(local_feather_path)
        with open(local_artifacts_path, 'r', encoding='utf-8') as f:
            artifacts = json.load(f)

    except Exception as e:
        return {"success": False, "error": f"Data load failed: {str(e)}"}

    echarts_list = []
    errors = {}

    # 2. 遍历配置，批量生成
    charts_config = viz_config.get("charts")

    if not charts_config:
        return {"success": False, "error": "未找到任何图表配置"}

    for chart_key, config in charts_config.items():
        # 获取对应的处理函数
        chart_type = config.get("type")
        handler = CHART_HANDLERS.get(chart_type)

        if not handler:
            errors[chart_key] = f"图表类型未注册: '{chart_type}' (图表名称: {chart_key})"
            continue

        try:
            print(f"  -> 生成 {chart_key} 中...")

            chart_data = handler(df, config, artifacts)

            chart_item = {
                "type": chart_type,
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
    print(f"--- [Viz Generator] 根据配置信息生成图表数据成功！生成的图表: {generated_types}, 失败的图表: {list(errors.keys())} ---")
    if len(errors.keys()) > 0:
        print(errors)

    return {
        "success": has_success,
        "viz_data": {
            "echarts": echarts_list
        },
        "errors": errors
    }