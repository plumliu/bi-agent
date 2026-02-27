import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

from ..core.schemas import SDKResult
from ..core.utils import clean_for_json


def run_logistic_regression(df: pd.DataFrame, target_col: str, num_cols: list, cat_cols: list = None,
                            model_params: dict = None) -> SDKResult:
    """内部执行逻辑回归分类训练及推理，输出带正负权重的特征重要性"""

    cat_cols = cat_cols or []
    if df.empty or target_col not in df.columns:
        raise ValueError(f"DataFrame 为空或未找到目标列 '{target_col}'。")
    if not num_cols and not cat_cols:
        raise ValueError("未指定任何数值或类别特征列。")

    model_params = model_params or {}
    C_param = model_params.get('C', 1.0)  # 正则化强度的倒数
    max_iter = model_params.get('max_iter', 1000)
    random_state = 42

    result_df = df.copy()

    # 1. 核心数据分流
    is_unknown_mask = result_df[target_col].isna()
    df_infer = result_df[is_unknown_mask].copy()
    df_model = result_df[~is_unknown_mask].copy()

    if df_model.empty:
        raise ValueError("建模集为空：目标列全部为缺失值，无法进行模型训练。")

    has_inference_data = not df_infer.empty

    # 2. 预处理 (Fit on Model, Transform Both)
    # 2.1 目标列编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(df_model[target_col])
    class_names = [str(c) for c in le.classes_]

    # 2.2 数值型特征 (均值填充 + 标准化)
    if num_cols:
        mean_vals = df_model[num_cols].mean()
        df_model[num_cols] = df_model[num_cols].fillna(mean_vals)
        if has_inference_data:
            df_infer[num_cols] = df_infer[num_cols].fillna(mean_vals)

        scaler = StandardScaler()
        df_model[num_cols] = scaler.fit_transform(df_model[num_cols])
        if has_inference_data:
            df_infer[num_cols] = scaler.transform(df_infer[num_cols])

    # 2.3 类别型特征 (众数填充 + 独热编码 OHE)
    # 逻辑回归不支持原生 category，必须转为 one-hot 向量
    ohe_features = []
    if cat_cols:
        for col in cat_cols:
            mode_val = df_model[col].mode()
            fill_val = mode_val[0] if not mode_val.empty else 'Unknown'
            df_model[col] = df_model[col].fillna(fill_val)
            if has_inference_data:
                df_infer[col] = df_infer[col].fillna(fill_val)

        # 对 df_model 和 df_infer 拼接后统一做 OHE，保证维度一致
        combined_cat = pd.concat(
            [df_model[cat_cols], df_infer[cat_cols]] if has_inference_data else [df_model[cat_cols]])
        ohe_combined = pd.get_dummies(combined_cat, dummy_na=False, drop_first=True).astype(int)

        # 拆分回 Model 和 Infer
        ohe_model = ohe_combined.iloc[:len(df_model)]
        df_model = pd.concat([df_model, ohe_model], axis=1)

        if has_inference_data:
            ohe_infer = ohe_combined.iloc[len(df_model):]
            df_infer = pd.concat([df_infer, ohe_infer], axis=1)

        ohe_features = ohe_combined.columns.tolist()

    # 3. 训练验证划分与逻辑回归训练
    features = num_cols + ohe_features
    X = df_model[features]
    y = y_encoded

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)

    clf = LogisticRegression(C=C_param, max_iter=max_iter, random_state=random_state)
    clf.fit(X_train, y_train)

    # 4. 产物生成
    # A. 验证集评估
    y_pred_val = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred_val)

    cm = confusion_matrix(y_val, y_pred_val, normalize='true')
    cm_dict = {}
    for i, true_label in enumerate(class_names):
        cm_dict[true_label] = {}
        for j, pred_label in enumerate(class_names):
            cm_dict[true_label][pred_label] = float(cm[i, j])

    # B. 特征重要性排行 (提取模型系数 Coefficients)
    # 对于二分类，coef_ 形状为 (1, n_features)，取 [0]
    coefs = clf.coef_[0] if clf.coef_.shape[0] == 1 else clf.coef_[0]

    imp_df = pd.DataFrame({
        '特征': features,
        '重要性': coefs,
        # 新增一列绝对值，专用于排序，确保正负向影响最大的特征都在最上面
        'abs_importance': np.abs(coefs)
    }).sort_values(by='abs_importance', ascending=False).head(20)

    # 转换为 Viz 需要的格式 (依然保留原本的正负 importance)
    feature_importance = imp_df.to_dict(orient='records')

    # C. 推理预测 (隐式透传业务元数据)
    predicted_samples = []
    if has_inference_data:
        X_infer = df_infer[features]
        y_pred_infer = clf.predict(X_infer)
        y_prob_infer = clf.predict_proba(X_infer)

        results_df = df_infer.copy()
        results_df['预测标签'] = le.inverse_transform(y_pred_infer)
        results_df['置信度'] = np.max(y_prob_infer, axis=1)
        results_df['真实标签'] = None

        result_df.loc[df_infer.index, '预测标签'] = results_df['预测标签']
        result_df.loc[df_infer.index, '置信度'] = results_df['置信度']

        used_cols = set(num_cols + cat_cols + [target_col])
        meta_cols = [c for c in df_infer.columns if c not in used_cols and c not in ohe_features]
        display_cols = meta_cols + ['真实标签', '预测标签', '置信度']
        predicted_samples = results_df[display_cols].head(500).to_dict(orient='records')

    artifacts = {
        "accuracy": float(acc),
        "has_inference_data": bool(has_inference_data),
        "predicted_samples": predicted_samples,
        "confusion_matrix": cm_dict,
        "feature_importance": feature_importance,

        "model_info": {
            "method": "logistic_regression",
            "C": C_param,
            "max_iter": max_iter
        }
    }

    return SDKResult(
        result_df=result_df,
        artifacts=clean_for_json(artifacts)
    )