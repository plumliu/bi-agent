from .sarimax import run_sarimax_forecast
from .prophet_forecaster import run_prophet_forecast
from ..core.schemas import SDKResult


def run_forecast(df, value_col, time_col=None, forecast_steps=12, method='sarimax', model_params=None) -> SDKResult:

    model_params = model_params or {}

    if method == 'sarimax':
        return run_sarimax_forecast(df, value_col, time_col, forecast_steps, model_params)
    elif method == 'prophet':
        return run_prophet_forecast(df, value_col, time_col, forecast_steps, model_params)
    else:
        raise ValueError(f"不支持的预测算法: {method}")