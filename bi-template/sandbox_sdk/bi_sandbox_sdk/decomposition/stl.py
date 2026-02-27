import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.nonparametric.smoothers_lowess import lowess
from ..core.schemas import SDKResult
from ..core.utils import clean_for_json


def detect_periodicity(series: np.ndarray) -> tuple:
    """内部辅助函数：使用 FFT 检测主导周期和信噪比 (SNR)"""
    n = len(series)
    if n < 4:
        return 0, 0.0

    # 1. 一阶差分去趋势
    diff_series = np.diff(series)

    # 2. FFT 频谱分析
    fft_vals = np.fft.rfft(diff_series)
    fft_power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(len(diff_series))

    # 忽略直流分量 (freq=0)
    fft_power[0] = 0

    # 3. 寻找峰值频率
    peak_idx = np.argmax(fft_power)
    peak_freq = freqs[peak_idx]

    if peak_freq == 0:
        return 0, 0.0

    detected_period = int(round(1 / peak_freq))

    # 4. 计算信噪比 (SNR = 峰值能量 / 背景平均噪声)
    peak_energy = fft_power[peak_idx]
    bg_noise = np.mean(np.delete(fft_power, peak_idx))
    snr = peak_energy / bg_noise if bg_noise > 0 else 0.0

    return detected_period, snr


def run_stl(df: pd.DataFrame, time_col: str, value_col: str, model_params: dict = None) -> SDKResult:
    """内部执行时序分解、周期检测及前端可视化数据聚合"""
    if df.empty or time_col not in df.columns or value_col not in df.columns:
        raise ValueError("DataFrame 为空或未指定正确的时间列/数值列。")

    #  1. 安全提取手动覆盖参数 (提供智能默认值)
    model_params = model_params or {}
    override_period = model_params.get('period', None)
    snr_threshold = model_params.get('snr_threshold', 3.0)
    anomaly_sigma = model_params.get('anomaly_sigma', 3.0)
    trend_frac = model_params.get('trend_frac', 0.17)
    stl_robust = model_params.get('robust', True)

    result_df = df.copy()

    # ---------------------------------------------------------
    # 步骤 1: 预处理 (排序与插值)
    # ---------------------------------------------------------
    if result_df[value_col].isnull().any():
        result_df[value_col] = result_df[value_col].interpolate(method='linear', limit_direction='both')

    series = result_df[value_col].values
    n = len(series)

    # ---------------------------------------------------------
    # 步骤 2: 周期盲测与策略分流 (FFT -> STL vs LOESS)
    # ---------------------------------------------------------
    detected_period, snr = detect_periodicity(series)

    #  参数覆盖决断：如果用户强制指定了周期，直接绕过 SNR 限制
    if override_period is not None and override_period > 1:
        has_seasonality = True
        active_period = int(override_period)
    else:
        # 使用自定义的 snr_threshold
        has_seasonality = snr > snr_threshold and 2 <= detected_period < n / 2
        active_period = detected_period

    if has_seasonality:
        # CASE A: 强周期，使用 STL 分解
        method_used = "stl"
        stl = STL(series, period=active_period, robust=stl_robust)
        res = stl.fit()
        result_df['趋势'] = res.trend
        result_df['季节'] = res.seasonal
        result_df['残差'] = res.resid
    else:
        # CASE B: 无显著周期，使用 LOESS 平滑提取趋势
        method_used = "lowess"
        has_seasonality = False
        active_period = 0

        #  使用自定义的 trend_frac
        trend_np = lowess(series, np.arange(n), frac=trend_frac)[:, 1]
        result_df['趋势'] = trend_np
        result_df['季节'] = 0.0
        result_df['残差'] = series - trend_np

    # ---------------------------------------------------------
    # 步骤 3: 异常检测 (基于自定义的 k-Sigma 原则)
    # ---------------------------------------------------------
    residuals = result_df['残差'].values
    residual_std = np.nanstd(residuals, ddof=1) if len(residuals) > 1 else 0.0

    #  使用自定义的 anomaly_sigma
    result_df['是否异常'] = (np.abs(residuals) > anomaly_sigma * residual_std).astype(int)
    result_df['拟合值'] = result_df['趋势'] + result_df['季节']
    result_df['误差带'] = anomaly_sigma * residual_std  # 供带误差带的折线图使用

    # ---------------------------------------------------------
    # 步骤 4: 周期数据折叠 (供周期指纹图渲染)
    # ---------------------------------------------------------
    seasonal_cycles = []
    cycle_names = []
    step_labels = []

    if has_seasonality and active_period > 0:
        period = active_period
        step_labels = [str(i + 1) for i in range(period)]
        seq = result_df['季节'].values
        time_seq = result_df[time_col].astype(str).values
        num_complete_cycles = len(seq) // period
        for i in range(num_complete_cycles):
            start = i * period
            end = start + period
            seasonal_cycles.append(seq[start:end].tolist())
            cycle_names.append(f"{time_seq[start]}至{time_seq[end - 1]}")

    # ---------------------------------------------------------
    # 步骤 5: 产物固化 (分离业务指标与模型超参数)
    # ---------------------------------------------------------
    artifacts = {
        "time_col": time_col,
        "value_col": value_col,
        "period": int(active_period),
        "has_seasonality": bool(has_seasonality),
        "snr": float(snr),
        "residual_std": float(residual_std),
        "seasonal_cycles": seasonal_cycles,
        "seasonal_categories": step_labels,
        "seasonal_cycle_names": cycle_names,

        "model_info": {
            "method": method_used,
            "anomaly_sigma": anomaly_sigma,
            "snr_threshold": snr_threshold,
            "trend_frac": trend_frac,
            "robust": stl_robust,
            "override_period_used": override_period
        }
    }

    return SDKResult(
        result_df=result_df,
        artifacts=clean_for_json(artifacts)
    )
