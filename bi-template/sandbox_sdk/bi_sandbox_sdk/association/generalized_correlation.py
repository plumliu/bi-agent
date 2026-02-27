import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings

from ..core.schemas import SDKResult
from ..core.utils import clean_for_json

# 忽略 scipy 在计算卡方检验时可能产生的除 0 警告
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _cramers_v(x: pd.Series, y: pd.Series):
    """内部辅助函数: 计算 Cramér's V (类别 vs 类别)"""
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.empty:
        return 0.0

    chi2 = stats.chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    if n == 0:
        return 0.0

    phi2 = chi2 / n
    r, k = confusion_matrix.shape

    # 修正偏差 (Bias correction)
    with np.errstate(divide='ignore', invalid='ignore'):
        phi2corr = np.maximum(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)

        denominator = np.minimum((kcorr - 1), (rcorr - 1))
        if denominator <= 0:
            return 0.0

        return np.sqrt(phi2corr / denominator)


def _tschuprows_t(x: pd.Series, y: pd.Series):
    """内部辅助函数: 计算 Tschuprow's T (类别 vs 类别)"""
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.empty:
        return 0.0

    chi2 = stats.chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    if n == 0:
        return 0.0

    r, k = confusion_matrix.shape
    denominator = np.sqrt((r - 1) * (k - 1))

    if denominator == 0:
        return 0.0

    return np.sqrt((chi2 / n) / denominator)


def _correlation_ratio(categories: pd.Series, measurements: pd.Series):
    """内部辅助函数: 计算 Correlation Ratio / Eta (数值 vs 类别)"""
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    if cat_num <= 1:
        return 0.0

    y = np.array(measurements)
    n = len(y)

    if n == 0:
        return 0.0

    ss_total = np.var(y, ddof=0) * n
    if ss_total == 0:
        return 0.0

    ss_between = 0.0
    for i in range(cat_num):
        sub_y = y[fcat == i]
        if len(sub_y) > 0:
            ss_between += len(sub_y) * (np.mean(sub_y) - np.mean(y)) ** 2

    eta_squared = ss_between / ss_total
    # 确保数值在合法区间内
    return np.sqrt(np.clip(eta_squared, 0.0, 1.0))


def _point_biserial(categories: pd.Series, measurements: pd.Series):
    """内部辅助函数: 计算 Point-Biserial Correlation (仅限二分类 vs 数值)"""
    fcat, unique_cats = pd.factorize(categories)

    # 防呆设计：如果该分类变量不是二分类（例如有3个以上的状态），直接降级为 correlation_ratio
    if len(unique_cats) != 2:
        return _correlation_ratio(categories, measurements)

    # 计算点双列相关系数，取绝对值以保持与其他关联指标的方向一致性（0~1）
    res = stats.pointbiserialr(fcat, measurements)
    # 如果遇到全相同的数据导致返回 NaN，这里直接返回，后续由 clean_for_json 兜底
    return np.abs(res.statistic)


def run_generalized_correlation(df: pd.DataFrame, num_cols: list, cat_cols: list = None,
                                model_params: dict = None) -> SDKResult:
    """内部执行广义混合关联分析及前端热力图所需的三大矩阵准备"""

    cat_cols = cat_cols or []
    if df.empty or (not num_cols and not cat_cols):
        raise ValueError("DataFrame 为空，或未指定任何特征列。")

    # ---------------------------------------------------------
    # 步骤 1: 解析模型参数 (动态算法挂载)
    # ---------------------------------------------------------
    model_params = model_params or {}
    # 支持 'spearman', 'pearson', 'kendall'
    num_num_metric = model_params.get('num_num_metric', 'spearman')
    # 支持 'cramers_v', 'tschuprows_t'
    cat_cat_metric = model_params.get('cat_cat_metric', 'cramers_v')
    # 支持 'correlation_ratio', 'point_biserial'
    num_cat_metric = model_params.get('num_cat_metric', 'point_biserial')

    result_df = df.copy()

    # ---------------------------------------------------------
    # 步骤 2: 安全预处理 (隔离脏数据)
    # ---------------------------------------------------------
    if num_cols:
        for col in num_cols:
            if result_df[col].isnull().any():
                result_df[col] = result_df[col].fillna(result_df[col].mean())

    if cat_cols:
        for col in cat_cols:
            if result_df[col].isnull().any():
                mode_val = result_df[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else 'Unknown'
                result_df[col] = result_df[col].fillna(fill_val)

    # ---------------------------------------------------------
    # 步骤 3: 构建三大关联矩阵 (按需执行，剔除显式 float 转换)
    # ---------------------------------------------------------

    # A. Num vs Num (连续变量之间的相关性)
    matrix_num_num = {}
    if len(num_cols) >= 2:
        df_num = result_df[num_cols]
        # pandas 的 corr 返回的就是 np.float64，直接 to_dict
        matrix_num_num = df_num.corr(method=num_num_metric).to_dict(orient='index')

    # B. Cat vs Cat (类别变量之间的关联性)
    matrix_cat_cat = {}
    if len(cat_cols) >= 2:
        for col1 in cat_cols:
            matrix_cat_cat[col1] = {}
            for col2 in cat_cols:
                if col1 == col2:
                    val = 1.0
                elif cat_cat_metric == 'cramers_v':
                    val = _cramers_v(result_df[col1], result_df[col2])
                elif cat_cat_metric == 'tschuprows_t':
                    val = _tschuprows_t(result_df[col1], result_df[col2])
                else:
                    raise ValueError(f"不支持的 Cat-Cat 关联算法: {cat_cat_metric}")

                # 直接赋值 Numpy 类型，由外部 clean_for_json 序列化
                matrix_cat_cat[col1][col2] = val

    # C. Num vs Cat (连续变量与类别变量之间的关联性)
    matrix_num_cat = {}
    if len(num_cols) > 0 and len(cat_cols) > 0:
        for n_col in num_cols:
            matrix_num_cat[n_col] = {}
            for c_col in cat_cols:
                if num_cat_metric == 'correlation_ratio':
                    val = _correlation_ratio(result_df[c_col], result_df[n_col])
                elif num_cat_metric == 'point_biserial':
                    val = _point_biserial(result_df[c_col], result_df[n_col])
                else:
                    raise ValueError(f"不支持的 Num-Cat 关联算法: {num_cat_metric}")

                # 直接赋值
                matrix_num_cat[n_col][c_col] = val

    # ---------------------------------------------------------
    # 步骤 4: 产物固化 (分离业务数据与模型调度信息)
    # ---------------------------------------------------------
    artifacts = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "matrix_num_num": matrix_num_num,
        "matrix_cat_cat": matrix_cat_cat,
        "matrix_num_cat": matrix_num_cat,

        "model_info": {
            "method": "generalized",
            "num_num_metric": num_num_metric,
            "cat_cat_metric": cat_cat_metric,
            "num_cat_metric": num_cat_metric
        }
    }

    return SDKResult(
        result_df=result_df,
        artifacts=clean_for_json(artifacts)
    )