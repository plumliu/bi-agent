from .isolation_forest import run_isolation_forest
from .dbscan import run_dbscan

from ..core.schemas import SDKResult

def detect_anomalies(df, num_cols, cat_cols=None, method='isolation_forest', model_params=None) -> SDKResult:
    model_params = model_params or {}

    if method == 'isolation_forest':
        return run_isolation_forest(df, num_cols=num_cols, cat_cols=cat_cols, model_params=model_params)
    elif method == 'dbscan':
        return run_dbscan(df, num_cols=num_cols, cat_cols=cat_cols, model_params=model_params)
    else:
        raise ValueError(f"不支持的异常检测算法: {method}")