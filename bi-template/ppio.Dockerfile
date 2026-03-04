FROM image.ppinfra.com/sandbox/code-interpreter:latest

USER root

# =========================================================
# 1. 基础环境与依赖安装
# =========================================================
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    pyarrow \
    statsmodels \
    scikit-learn \
    xgboost \
    lightgbm \
    openpyxl \
    joblib \
    pydantic \
    prophet

# =========================================================
# 2. 安装业务 SDK
# =========================================================
COPY ./sandbox_sdk /tmp/sandbox_sdk
RUN pip install /tmp/sandbox_sdk && \
    rm -rf /tmp/sandbox_sdk

# =========================================================
# 3. 系统权限收尾与降权
# =========================================================
RUN if [ -f "/root/.jupyter/start-up.sh" ]; then chmod +x /root/.jupyter/start-up.sh; fi

USER 1000