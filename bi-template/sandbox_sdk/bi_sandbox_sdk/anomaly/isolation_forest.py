import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from ..core.schemas import SDKResult
from ..core.utils import clean_for_json


def run_isolation_forest(df: pd.DataFrame, num_cols: list, cat_cols: list = None,
                         model_params: dict = None) -> SDKResult:
    """内部执行孤立森林及前端图表所需的数据聚合操作"""

    cat_cols = cat_cols or []
    if df.empty or (not num_cols and not cat_cols):
        raise ValueError("DataFrame 为空，或未指定任何数值/类别特征列。")

    # 安全提取手动覆盖参数
    model_params = model_params or {}
    n_estimators = model_params.get('n_estimators', 100)  # 森林中树的数量
    override_contamination = model_params.get('contamination', None)  # 是否强制指定异常比例
    iqr_multiplier = model_params.get('iqr_multiplier', 1.5)  # IQR 阈值乘数，默认 1.5

    result_df = df.copy()
    model_features = []  # 最终喂给模型的全量特征列名

    # ---------------------------------------------------------
    # 步骤 1: 预处理 - 数值型特征 (填充与缩放)
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
    # 步骤 3: 异常检测 (Isolation Forest 策略分流)
    # ---------------------------------------------------------

    random_state = 42
    applied_strategy = "dynamic_iqr"

    if override_contamination is not None:
        # 策略 A: 业务方强行指定了异常比例 (例如 0.05 代表 5%)
        applied_strategy = "fixed_contamination"
        # 防呆：确保 contamination 在合法区间 (0, 0.5]
        cont = max(0.001, min(0.5, float(override_contamination)))
        model = IsolationForest(n_estimators=n_estimators, contamination=cont, random_state=random_state)
        model.fit(result_df[model_features])

        scores = -model.decision_function(result_df[model_features])
        preds = model.predict(result_df[model_features])

        result_df['异常分数'] = scores
        result_df['是否异常'] = (preds == -1).astype(int)  # sklearn 中 -1 为异常
    else:
        # 策略 B (默认): 动态 IQR 阈值计算
        model = IsolationForest(n_estimators=n_estimators, random_state=random_state)
        model.fit(result_df[model_features])

        scores = -model.decision_function(result_df[model_features])
        result_df['异常分数'] = scores

        # 计算自定义倍数的 IQR 阈值
        Q1_score = np.percentile(scores, 25)
        Q3_score = np.percentile(scores, 75)
        IQR_score = Q3_score - Q1_score
        threshold = Q3_score + iqr_multiplier * IQR_score

        result_df['是否异常'] = (scores > threshold).astype(int)

    # ---------------------------------------------------------
    # 步骤 4 & 5 PCA 降维与产物固化
    # ---------------------------------------------------------
    if len(model_features) >= 2:
        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(result_df[model_features])
    else:
        coords = np.zeros((len(result_df), 2))
        coords[:, 0] = result_df[model_features[0]]

    result_df['主成分 x'] = coords[:, 0]
    result_df['主成分 y'] = coords[:, 1]

    # 5.1 准备箱线图统计值 (严格限定仅对原 num_cols 计算)
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

    # 5.2 准备异常点数据
    outliers_df = result_df[result_df['是否异常'] == 1].sort_values(by='异常分数', ascending=False)
    outliers = outliers_df.head(500).to_dict(orient='records')

    # 5.3 准备柱状图数据
    top_anomalies = result_df.nlargest(10, '异常分数').to_dict(orient='records')

    anomaly_count = int(result_df['是否异常'].sum())

    artifacts = {
        "anomaly_count": anomaly_count,
        "n_anomalies": len(outliers),
        "box_stats": box_stats,
        "outliers": outliers,
        "top_anomalies": top_anomalies,

        "model_info": {
            "method": "isolation_forest",
            "strategy": applied_strategy,
            "n_estimators": n_estimators,
            "override_contamination": override_contamination,
            "iqr_multiplier": iqr_multiplier
        }
    }

    return SDKResult(
        result_df=result_df,
        artifacts=clean_for_json(artifacts)
    )
