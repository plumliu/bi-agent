from .kmeans import run_kmeans
from ..core.schemas import SDKResult


def run_clustering(df, num_cols, cat_cols=None, method='kmeans', model_params=None, **kwargs) -> SDKResult:
    model_params = model_params or {}

    if method == 'kmeans':
        return run_kmeans(df, num_cols, cat_cols=cat_cols, model_params=model_params, **kwargs)
    else:
        raise ValueError(f"不支持的聚类算法: {method}")