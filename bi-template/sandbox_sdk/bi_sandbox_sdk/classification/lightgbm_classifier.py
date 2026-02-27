import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

from ..core.schemas import SDKResult
from ..core.utils import clean_for_json


def run_lightgbm(df: pd.DataFrame, target_col: str, num_cols: list, cat_cols: list = None,
                 model_params: dict = None) -> SDKResult:
    """内部执行 LightGBM 分类训练、验证及推理，并隐式透传所有业务元数据列"""

    cat_cols = cat_cols or []
    if df.empty or target_col not in df.columns:
        raise ValueError(f"DataFrame 为空或未找到目标列 '{target_col}'。")
    if not num_cols and not cat_cols:
        raise ValueError("未指定任何数值或类别特征列。")

    model_params = model_params or {}
    n_estimators = model_params.get('n_estimators', 100)
    learning_rate = model_params.get('learning_rate', 0.1)
    max_depth = model_params.get('max_depth', -1)
    random_state = 42

    result_df = df.copy()

    # ---------------------------------------------------------
    # 步骤 1: 核心数据分流 (Modeling Set vs Inference Set)
    # ---------------------------------------------------------
    is_unknown_mask = result_df[target_col].isna()
    df_infer = result_df[is_unknown_mask].copy()
    df_model = result_df[~is_unknown_mask].copy()

    if df_model.empty:
        raise ValueError("建模集为空：目标列全部为缺失值，无法进行模型训练。")

    has_inference_data = not df_infer.empty

    # ---------------------------------------------------------
    # 步骤 2: 严格防止数据泄露的预处理 (Fit on Model, Transform Both)
    # ---------------------------------------------------------
    # 2.1 目标列编码 (仅针对 df_model)
    le = LabelEncoder()
    y_encoded = le.fit_transform(df_model[target_col])
    class_names = [str(c) for c in le.classes_]

    # 2.2 数值型特征处理
    if num_cols:
        # 填充：使用建模集的均值
        mean_vals = df_model[num_cols].mean()
        df_model[num_cols] = df_model[num_cols].fillna(mean_vals)
        if has_inference_data:
            df_infer[num_cols] = df_infer[num_cols].fillna(mean_vals)

        # 标准化
        scaler = StandardScaler()
        df_model[num_cols] = scaler.fit_transform(df_model[num_cols])
        if has_inference_data:
            df_infer[num_cols] = scaler.transform(df_infer[num_cols])

    # 2.3 类别型特征处理
    if cat_cols:
        for col in cat_cols:
            # 填充：使用建模集的众数
            mode_val = df_model[col].mode()
            fill_val = mode_val[0] if not mode_val.empty else 'Unknown'

            df_model[col] = df_model[col].fillna(fill_val).astype('category')
            if has_inference_data:
                # 注意：待预测集中可能出现新类别，但 LightGBM 原生支持处理未见过的类别
                df_infer[col] = df_infer[col].fillna(fill_val).astype('category')

    # ---------------------------------------------------------
    # 步骤 3: 训练验证划分与 LightGBM 训练
    # ---------------------------------------------------------
    features = num_cols + cat_cols
    X = df_model[features]
    y = y_encoded

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)

    clf = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        verbose=-1  # 屏蔽繁琐的打印信息
    )
    clf.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 步骤 4: 产物生成
    # ---------------------------------------------------------

    # --- A. 验证集评估 (Accuracy & 归一化混淆矩阵) ---
    y_pred_val = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred_val)

    # 计算行归一化的混淆矩阵 (真实类别的召回比例)
    cm = confusion_matrix(y_val, y_pred_val, normalize='true')
    cm_dict = {}
    for i, true_label in enumerate(class_names):
        cm_dict[true_label] = {}
        for j, pred_label in enumerate(class_names):
            cm_dict[true_label][pred_label] = float(cm[i, j])

    # --- B. 特征重要性排行 ---
    imp_df = pd.DataFrame({
        '特征': features,
        '重要性': clf.feature_importances_
    }).sort_values(by='重要性', ascending=False).head(20)
    feature_importance = imp_df.to_dict(orient='records')

    # --- C. 推理预测 (如果有待预测数据) ---
    predicted_samples = []
    if has_inference_data:
        X_infer = df_infer[features]
        y_pred_infer = clf.predict(X_infer)
        y_prob_infer = clf.predict_proba(X_infer)

        results_df = df_infer.copy()
        # 反向解码得到真实的字符串标签
        results_df['预测标签'] = le.inverse_transform(y_pred_infer)
        results_df['置信度'] = np.max(y_prob_infer, axis=1)
        results_df['真实标签'] = None  # 前端占位

        # 将结果拼装回 result_df，方便用户获取完整数据
        result_df.loc[df_infer.index, '预测标签'] = results_df['预测标签']
        result_df.loc[df_infer.index, '置信度'] = results_df['置信度']

        # 识别所有的业务元数据列 (即不在特征和目标列中的所有列)
        used_cols = set(num_cols + cat_cols + [target_col])
        meta_cols = [c for c in df_infer.columns if c not in used_cols]

        # 组装明细表列名：先把业务元数据放在前面，最后拼上预测结果
        display_cols = meta_cols + ['真实标签', '预测标签', '置信度']

        predicted_samples = results_df[display_cols].head(500).to_dict(orient='records')

    artifacts = {
        "accuracy": float(acc),
        "has_inference_data": bool(has_inference_data),
        "predicted_samples": predicted_samples,
        "confusion_matrix": cm_dict,
        "feature_importance": feature_importance,
        "model_info": {
            "method": "lightgbm",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth
        }
    }

    return SDKResult(
        result_df=result_df,
        artifacts=clean_for_json(artifacts)
    )