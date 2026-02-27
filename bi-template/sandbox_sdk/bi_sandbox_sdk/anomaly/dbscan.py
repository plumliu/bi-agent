import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from ..core.schemas import SDKResult
from ..core.utils import clean_for_json


def run_dbscan(df: pd.DataFrame, num_cols: list, cat_cols: list = None,
               model_params: dict = None) -> SDKResult:
    """内部执行 DBSCAN 密度异常检测及前端图表所需的数据聚合操作"""

    cat_cols = cat_cols or []
    if df.empty or (not num_cols and not cat_cols):
        raise ValueError("DataFrame 为空，或未指定任何数值/类别特征列。")

    # 安全提取手动覆盖参数
    model_params = model_params or {}
    # eps: 邻域半径，DBSCAN 最核心参数。默认 0.5 (缩放后的空间)
    eps = model_params.get('eps', 0.5)
    # min_samples: 成为核心点所需的最小样本数
    min_samples = model_params.get('min_samples', 5)

    result_df = df.copy()
    model_features = []

    # ---------------------------------------------------------
    # 步骤 1: 预处理 - 数值型特征 (必须输出带 _scaled 后缀的列以支持 BoxPlot)
    # ---------------------------------------------------------
    scaled_cols = []
    if num_cols:
        scaler = RobustScaler()
        for col in num_cols:
            if result_df[col].isnull().any():
                result_df[col] = result_df[col].fillna(result_df[col].mean())
            scaled_cols.append(col + '_scaled')

        result_df[scaled_cols] = scaler.fit_transform(result_df[num_cols])
        model_features.extend(scaled_cols)

    # ---------------------------------------------------------
    # 步骤 2: 预处理 - 类别型特征 (填充与独热编码)
    # ---------------------------------------------------------
    if cat_cols:
        for col in cat_cols:
            if result_df[col].isnull().any():
                mode_val = result_df[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else 'Unknown'
                result_df[col] = result_df[col].fillna(fill_val)

        ohe_df = pd.get_dummies(result_df[cat_cols], dummy_na=False, drop_first=False)
        ohe_df = ohe_df.astype(int)
        ohe_cols = ohe_df.columns.tolist()

        result_df = pd.concat([result_df, ohe_df], axis=1)
        model_features.extend(ohe_cols)

    # ---------------------------------------------------------
    # 步骤 3: 异常检测 (DBSCAN 聚类与连续化打分)
    # ---------------------------------------------------------
    X = result_df[model_features]

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    preds = dbscan.fit_predict(X)

    # DBSCAN 中，-1 代表噪声(异常)
    is_anom = (preds == -1).astype(int)
    result_df['is_anomaly'] = is_anom

    # 构建连续的 anomaly_score
    scores = np.zeros(len(X))
    normal_idx = np.where(preds != -1)[0]
    anom_idx = np.where(preds == -1)[0]

    if len(normal_idx) > 0 and len(anom_idx) > 0:
        # 寻找每个异常点到最近的正常点的距离
        nn = NearestNeighbors(n_neighbors=1).fit(X.iloc[normal_idx])
        distances, _ = nn.kneighbors(X.iloc[anom_idx])
        scores[anom_idx] = distances.flatten()
    elif len(anom_idx) > 0:
        # 极端情况：全是异常点
        scores[anom_idx] = 1.0

    result_df['anomaly_score'] = scores

    # ---------------------------------------------------------
    # 步骤 4 & 5 PCA降维与产物固化 (严格对齐 Viz 协议)
    # ---------------------------------------------------------
    random_state = 42
    if len(model_features) >= 2:
        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(X)
    else:
        coords = np.zeros((len(result_df), 2))
        coords[:, 0] = X.iloc[:, 0]

    result_df['pca_x'] = coords[:, 0]
    result_df['pca_y'] = coords[:, 1]

    # 5.1 准备箱线图统计值 (供 BoxPlot 渲染背景箱体)
    box_stats = {}
    if num_cols:
        for col, s_col in zip(num_cols, scaled_cols):
            s_data = result_df[s_col]
            desc = s_data.describe()

            Q1 = desc['25%']
            Q3 = desc['75%']
            IQR = Q3 - Q1

            lower_fence = Q1 - 1.5 * IQR
            upper_fence = Q3 + 1.5 * IQR

            filtered_lower = s_data[s_data >= lower_fence]
            lower_whisker = filtered_lower.min() if not filtered_lower.empty else lower_fence

            filtered_upper = s_data[s_data <= upper_fence]
            upper_whisker = filtered_upper.max() if not filtered_upper.empty else upper_fence

            box_stats[col] = [lower_whisker, Q1, desc['50%'], Q3, upper_whisker]

    # 5.2 准备异常点数据 (供 TableChart 与 Scatter 前景展示)
    outliers_df = result_df[result_df['is_anomaly'] == 1].sort_values(by='anomaly_score', ascending=False)
    outliers = outliers_df.head(500).to_dict(orient='records')

    # 5.3 准备柱状图数据 (供 BarChart 排行榜)
    top_anomalies = result_df.nlargest(10, 'anomaly_score').to_dict(orient='records')

    anomaly_count = int(result_df['is_anomaly'].sum())

    artifacts = {
        "anomaly_count": anomaly_count,
        "n_anomalies": len(outliers),
        "box_stats": box_stats,
        "outliers": outliers,
        "top_anomalies": top_anomalies,

        "model_info": {
            "method": "dbscan",
            "eps": eps,
            "min_samples": min_samples
        }
    }

    return SDKResult(
        result_df=result_df,
        artifacts=clean_for_json(artifacts)
    )