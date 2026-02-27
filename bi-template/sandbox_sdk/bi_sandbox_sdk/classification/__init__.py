from .lightgbm_classifier import run_lightgbm
from .logistic_regression import run_logistic_regression
from ..core.schemas import SDKResult

def run_classification(df, target_col, num_cols, cat_cols=None, method='lightgbm', model_params=None) -> SDKResult:

    model_params = model_params or {}

    if method == 'lightgbm':
        return run_lightgbm(df, target_col=target_col, num_cols=num_cols, cat_cols=cat_cols, model_params=model_params)
    elif method == 'logistic_regression':
        return run_logistic_regression(df, target_col=target_col, num_cols=num_cols, cat_cols=cat_cols, model_params=model_params)
    else:
        raise ValueError(f"不支持的分类算法: {method}")