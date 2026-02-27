import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from ..core.schemas import SDKResult
from ..core.utils import clean_for_json

warnings.filterwarnings("ignore")


def _auto_tune_sarimax(train_series, seasonal_s):
    """内部辅助：受限空间的鲁棒网格搜索，寻找最小 AIC"""
    best_aic = float("inf")
    best_order = (1, 1, 1)

    p_params = [0, 1, 2]
    q_params = [0, 1, 2]
    d = 1

    seasonal_order = (1, 1, 1, seasonal_s) if seasonal_s > 1 else (0, 0, 0, 0)

    for p in p_params:
        for q in q_params:
            try:
                model = SARIMAX(train_series, order=(p, d, q),
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                res = model.fit(disp=False, maxiter=50)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (p, d, q)
            except Exception:
                continue

    return best_order, seasonal_order


def build_time_index(df: pd.DataFrame, value_col: str, time_col: str = None) -> pd.DataFrame:
    """修复版：精准针对 value_col 进行时间维度的聚合"""
    result_df = df.copy()

    if time_col and time_col in result_df.columns:
        try:
            result_df[time_col] = pd.to_datetime(result_df[time_col])

            # 显式地针对指定的 value_col 进行求和聚合，保留时间列
            result_df['__target_value__'] = pd.to_numeric(result_df[value_col], errors='coerce')
            result_df = result_df.groupby(time_col, as_index=False)['__target_value__'].sum()

            # 将列名还原回去
            result_df = result_df.rename(columns={'__target_value__': value_col})

            result_df = result_df.set_index(time_col)
            result_df = result_df.sort_index()

            freq = pd.infer_freq(result_df.index)
            if freq:
                result_df = result_df.asfreq(freq)

        except Exception:
            result_df = result_df.sort_values(by=time_col).reset_index(drop=True)
    else:
        result_df = result_df.reset_index(drop=True)

    return result_df


def run_sarimax_forecast(df, value_col, time_col=None, forecast_steps=12, model_params=None) -> SDKResult:
    if df.empty or value_col not in df.columns:
        raise ValueError("DataFrame 为空或未指定正确的数值列 value_col。")
    if len(df) < 5:
        raise ValueError("数据量过少 (不足 5 条)，无法进行可靠的时间序列预测。")

    model_params = model_params or {}
    order = model_params.get('order', None)
    seasonal_order = model_params.get('seasonal_order', None)

    # 1. 构建标准化的时间索引与数据清洗 (传入 value_col)
    result_df = build_time_index(df, value_col, time_col)

    if result_df[value_col].isnull().any():
        result_df[value_col] = result_df[value_col].interpolate(method='linear', limit_direction='both')

    # 2. 参数决断逻辑
    if order is None:
        s = 0
        if isinstance(result_df.index, pd.DatetimeIndex) and result_df.index.freq:
            freq_str = result_df.index.freqstr
            if freq_str.startswith('M') or freq_str.startswith('MS'):
                s = 12
            elif freq_str.startswith('D'):
                s = 7
            elif freq_str.startswith('H'):
                s = 24

        base_order, seasonal_order = _auto_tune_sarimax(result_df[value_col], seasonal_s=s)
    else:
        base_order = order
        seasonal_order = seasonal_order if seasonal_order else (0, 0, 0, 0)

    # 3. 训练测试集切分以计算 RMSE
    n_total = len(result_df)
    test_size = max(int(n_total * 0.1), min(12, n_total // 2))
    train_size = n_total - test_size
    train, test = result_df.iloc[:train_size], result_df.iloc[train_size:]

    try:
        eval_model = SARIMAX(train[value_col], order=base_order, seasonal_order=seasonal_order,
                             enforce_stationarity=False, enforce_invertibility=False)
        eval_res = eval_model.fit(disp=False)
        used_order = base_order
        used_seasonal = seasonal_order
    except Exception:
        used_order = (1, 1, 0)
        used_seasonal = (0, 0, 0, 0)
        eval_model = SARIMAX(train[value_col], order=used_order, seasonal_order=used_seasonal,
                             enforce_stationarity=False, enforce_invertibility=False)
        eval_res = eval_res.fit(disp=False)

    if len(test) > 0:
        test_pred = eval_res.get_forecast(steps=len(test)).predicted_mean
        rmse = float(np.sqrt(mean_squared_error(test[value_col], test_pred)))
    else:
        rmse = 0.0

    # 4. 全量数据重拟合
    full_model = SARIMAX(result_df[value_col], order=used_order, seasonal_order=used_seasonal,
                         enforce_stationarity=False, enforce_invertibility=False)
    full_res = full_model.fit(disp=False)

    forecast_obj = full_res.get_forecast(steps=forecast_steps)
    forecast_values = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int(alpha=0.05)
    fitted_values = full_res.fittedvalues

    # 5. 构建供前端渲染的 8 列标准结构
    if isinstance(result_df.index, pd.DatetimeIndex):
        freq = result_df.index.freq or pd.infer_freq(result_df.index)
        if freq:
            future_idx = pd.date_range(start=result_df.index[-1], periods=forecast_steps + 1, freq=freq)[1:]
        else:
            avg_diff = result_df.index.to_series().diff().median()
            future_idx = pd.date_range(start=result_df.index[-1] + avg_diff, periods=forecast_steps, freq=avg_diff)
    else:
        start_step = result_df.index[-1] + 1
        future_idx = pd.RangeIndex(start=start_step, stop=start_step + forecast_steps)

    full_idx = result_df.index.append(future_idx)
    final_df = pd.DataFrame(index=full_idx)

    if isinstance(final_df.index, pd.DatetimeIndex):
        if (final_df.index.time == pd.Timestamp('00:00:00').time()).all():
            final_df['日期'] = final_df.index.strftime('%Y-%m-%d')
        else:
            final_df['日期'] = final_df.index.strftime('%Y-%m-%d %H:%M:%S')
    else:
        final_df['日期'] = final_df.index.astype(str)

    final_df.loc[result_df.index, '历史值'] = result_df[value_col].values
    final_df.loc[result_df.index, '历史下限'] = result_df[value_col].values
    final_df.loc[result_df.index, '历史上限'] = result_df[value_col].values
    final_df.loc[result_df.index, '拟合值'] = fitted_values.values

    final_df.loc[future_idx, '预测值'] = forecast_values.values
    final_df.loc[future_idx, '预测下限'] = conf_int.iloc[:, 0].values
    final_df.loc[future_idx, '预测上限'] = conf_int.iloc[:, 1].values

    last_hist_idx = result_df.index[-1]
    last_val = result_df.iloc[-1][value_col]
    final_df.loc[last_hist_idx, ['预测值', '预测下限', '预测上限']] = [last_val, last_val, last_val]

    # 6. 产物固化
    artifacts = {
        "time_col": time_col if time_col else "Index",
        "value_col": value_col,
        "RMSE": rmse,
        "forecast_mean_value": float(forecast_values.mean()),
        "forecast_steps": forecast_steps,
        "model_info": {
            "method": "sarimax",
            "order": str(used_order),
            "seasonal_order": str(used_seasonal)
        }
    }

    final_df = final_df.reset_index(drop=True)

    return SDKResult(
        result_df=final_df,
        artifacts=clean_for_json(artifacts)
    )