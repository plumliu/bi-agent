import pandas as pd
import numpy as np
import logging
from prophet import Prophet

from ..core.schemas import SDKResult
from ..core.utils import clean_for_json

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)


def run_prophet_forecast(df: pd.DataFrame, value_col: str, time_col: str = None,
                         forecast_steps: int = 12, model_params: dict = None) -> SDKResult:
    """
    内部执行 Prophet 时序预测。
    """

    if df.empty or value_col not in df.columns:
        raise ValueError(f"DataFrame 为空或未找到数值列 '{value_col}'。")
    if len(df) < 5:
        raise ValueError("数据量过少 (不足 5 条)，无法进行可靠的时间序列预测。")

    # 解析模型参数
    model_params = model_params or {}
    changepoint_prior_scale = model_params.get('changepoint_prior_scale', 0.05)
    seasonality_mode = model_params.get('seasonality_mode', 'additive')
    yearly_seasonality = model_params.get('yearly_seasonality', 'auto')
    weekly_seasonality = model_params.get('weekly_seasonality', 'auto')

    prep_df = df.copy()

    # ---------------------------------------------------------
    # 步骤 1: 数据预处理与聚合
    # ---------------------------------------------------------
    if time_col and time_col in prep_df.columns:
        prep_df['ds'] = pd.to_datetime(prep_df[time_col])
    else:
        prep_df['ds'] = pd.date_range(start='2000-01-01', periods=len(prep_df), freq='D')

    prep_df['y'] = pd.to_numeric(prep_df[value_col], errors='coerce')

    # 1.1 精准聚合目标列
    # 在插值前，先按时间聚合（处理同一时间点多条数据的情况）
    train_df = prep_df.groupby('ds', as_index=False)['y'].sum()
    train_df = train_df.sort_values('ds').set_index('ds')

    # ---------------------------------------------------------
    # 步骤 2: 线性插值处理 (Linear Interpolation)
    # ---------------------------------------------------------
    # 2.1 补全缺失的时间步 (Resample)
    # 尝试推断频率，若无法推断则不强制 resample 以免引入过多噪声
    freq = pd.infer_freq(train_df.index)
    if freq:
        # 按推断频率对齐，缺失的时间步会产生 NaN
        train_df = train_df.asfreq(freq)

    # 2.2 执行线性插值
    # 使用 linear 方法填充聚合或重采样后的 NaN 值
    # limit_direction='both' 确保开头和结尾的缺失也能被处理
    if train_df['y'].isnull().any():
        train_df['y'] = train_df['y'].interpolate(method='linear', limit_direction='both')

    # 移除插值后依然可能存在的极少数无法填充的空值（如数据全为空）
    train_df = train_df.dropna().reset_index()

    if len(train_df) < 2:
        raise ValueError("经过插值预处理后有效数据点不足。")

    # ---------------------------------------------------------
    # 步骤 3: 模型训练与推理
    # ---------------------------------------------------------
    m = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality
    )
    m.fit(train_df)

    # 确定预测步长及频率
    actual_freq = freq or 'D'
    future = m.make_future_dataframe(periods=forecast_steps, freq=actual_freq)
    forecast = m.predict(future)

    # ---------------------------------------------------------
    # 步骤 4: 产物组装 (8列标准结构)
    # ---------------------------------------------------------
    final_df = pd.DataFrame(index=forecast.index)

    # 4.1 日期格式化
    if (forecast['ds'].dt.time == pd.Timestamp('00:00:00').time()).all():
        final_df['日期'] = forecast['ds'].dt.strftime('%Y-%m-%d')
    else:
        final_df['日期'] = forecast['ds'].dt.strftime('%Y-%m-%d %H:%M:%S')

    history_mask = forecast['ds'] <= train_df['ds'].max()
    future_mask = forecast['ds'] > train_df['ds'].max()

    # 4.2 历史系列填充
    # 注意：这里的历史值是经过插值处理后的“干净”值
    final_df.loc[history_mask, '历史值'] = train_df['y'].values
    final_df.loc[history_mask, '历史下限'] = train_df['y'].values
    final_df.loc[history_mask, '历史上限'] = train_df['y'].values
    final_df.loc[history_mask, '拟合值'] = forecast.loc[history_mask, 'yhat'].values

    # 4.3 未来系列填充
    final_df.loc[future_mask, '预测值'] = forecast.loc[future_mask, 'yhat'].values
    final_df.loc[future_mask, '预测下限'] = forecast.loc[future_mask, 'yhat_lower'].values
    final_df.loc[future_mask, '预测上限'] = forecast.loc[future_mask, 'yhat_upper'].values

    # 4.4 视觉缝合
    last_hist_idx = final_df[history_mask].index[-1]
    last_val = train_df['y'].iloc[-1]
    final_df.loc[last_hist_idx, ['预测值', '预测下限', '预测上限']] = [last_val, last_val, last_val]

    # ---------------------------------------------------------
    # 步骤 5: 统计指标与固化
    # ---------------------------------------------------------
    rmse = float(np.sqrt(np.mean((train_df['y'].values - forecast.loc[history_mask, 'yhat'].values) ** 2)))
    forecast_mean_value = float(forecast.loc[future_mask, 'yhat'].mean())

    out_time_col = time_col if time_col else "Index"
    artifacts = {
        "time_col": out_time_col,
        "value_col": value_col,
        "RMSE": rmse,
        "forecast_mean_value": forecast_mean_value,
        "forecast_steps": forecast_steps,
        "model_info": {
            "method": "prophet",
            "inferred_freq": actual_freq,
            "interpolation": "linear",
            "changepoint_prior_scale": changepoint_prior_scale
        }
    }

    return SDKResult(result_df=final_df.reset_index(drop=True), artifacts=clean_for_json(artifacts))