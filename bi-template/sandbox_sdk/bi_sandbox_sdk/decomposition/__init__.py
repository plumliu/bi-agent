from .stl import run_stl
from ..core.schemas import SDKResult


def decompose_time_series(df, time_col, value_col, method='stl', model_params=None) -> SDKResult:
    model_params = model_params or {}

    if method == 'stl':
        return run_stl(df, time_col=time_col, value_col=value_col, model_params=model_params)
    else:
        raise ValueError(f"不支持的时序分解算法: {method}")