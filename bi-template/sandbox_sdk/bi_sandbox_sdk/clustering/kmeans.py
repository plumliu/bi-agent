import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from ..core.schemas import SDKResult
from ..core.utils import clean_for_json


def run_kmeans(df: pd.DataFrame, num_cols: list, cat_cols: list = None, random_state: int = 42,
               model_params: dict = None) -> SDKResult:
    """内部执行 KMeans 最佳 K 值搜索及前端数据准备"""
    cat_cols = cat_cols or []
    if df.empty or (not num_cols and not cat_cols):
        raise ValueError("DataFrame 为空，或未指定任何特征列。")

    #  安全提取手动覆盖参数
    model_params = model_params or {}
    override_n_clusters = model_params.get('n_clusters', None)

    result_df = df.copy()
    model_features = []

    # ---------------------------------------------------------
    # 步骤 1: 预处理 (均值/众数填充，数值标准化，类别独热编码)
    # ---------------------------------------------------------
    scaled_cols = []
    if num_cols:
        scaler = StandardScaler()
        for col in num_cols:
            if result_df[col].isnull().any():
                result_df[col] = result_df[col].fillna(result_df[col].mean())
            scaled_cols.append(col + '_scaled')

        result_df[scaled_cols] = scaler.fit_transform(result_df[num_cols])
        model_features.extend(scaled_cols)

    if cat_cols:
        for col in cat_cols:
            if result_df[col].isnull().any():
                mode_val = result_df[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else 'Unknown'
                result_df[col] = result_df[col].fillna(fill_val)

        ohe_df = pd.get_dummies(result_df[cat_cols], dummy_na=False)
        ohe_df = ohe_df.astype(int)
        result_df = pd.concat([result_df, ohe_df], axis=1)
        model_features.extend(ohe_df.columns.tolist())

    X = result_df[model_features]

    # ---------------------------------------------------------
    # 步骤 2: 聚类 (SDK 自动寻优 vs Agent 手动接管)
    # ---------------------------------------------------------
    best_k = 2
    best_score = -1.0
    best_labels = None

    if override_n_clusters is not None:
        #  CASE A: 手动接管，直接使用指定的 K 值
        k = int(override_n_clusters)
        # 防呆：不能超过样本数减 1，也不能小于 1
        k = max(1, min(k, len(X) - 1))

        if k > 1:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
            best_labels = kmeans.fit_predict(X)
            # 计算一下得分，仅供产物展示
            best_score = silhouette_score(X, best_labels)
        else:
            # 极端情况降级
            best_labels = np.zeros(len(X))
            best_score = 0.0

        best_k = k
    else:
        #  CASE B: 自动寻优，使用轮廓系数
        max_clusters = 10
        max_k = min(max_clusters, len(X) - 1)
        if max_k >= 2:
            for k in range(2, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
                labels = kmeans.fit_predict(X)
                score = silhouette_score(X, labels)

                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels
        else:
            best_k = 1
            best_labels = np.zeros(len(X))
            best_score = 0.0

    # 【强制要求】：聚类标签从 1 开始
    result_df['聚类标签'] = best_labels + 1

    # ---------------------------------------------------------
    # 步骤 3 & 4 PCA 降维与产物固化
    # ---------------------------------------------------------
    if len(model_features) >= 2:
        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(X)
    else:
        coords = np.zeros((len(result_df), 2))
        coords[:, 0] = X.iloc[:, 0]

    result_df['主成分 x'] = coords[:, 0]
    result_df['主成分 y'] = coords[:, 1]

    centroids = {}
    if num_cols:
        centroids = result_df.groupby('聚类标签')[num_cols].mean().to_dict(orient='index')

    counts = result_df['聚类标签'].value_counts().to_dict()

    artifacts = {
        "k_value": int(best_k),
        "silhouette_score": float(best_score),
        "cluster_counts": counts,
        "centroids": centroids,

        "model_info": {
            "method": "kmeans",
            "override_n_clusters": override_n_clusters,
            "auto_optimized": override_n_clusters is None
        }
    }

    return SDKResult(
        result_df=result_df,
        artifacts=clean_for_json(artifacts)
    )
