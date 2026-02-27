from .generalized_correlation import run_generalized_correlation
from ..core.schemas import SDKResult

def analyze_association(df, num_cols, cat_cols=None, method='generalized', model_params=None) -> SDKResult:

    model_params = model_params or {}

    if method == 'generalized':
        return run_generalized_correlation(df, num_cols, cat_cols=cat_cols, model_params=model_params)
    else:
        raise ValueError(f"不支持的关联分析宏观策略: {method}")