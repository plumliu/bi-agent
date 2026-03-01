FROM image.ppinfra.com/sandbox/code-interpreter:latest

# 确保所有的 pip 安装都在系统全局目录
USER root

# =========================================================
# 1. 物理级封印：限制底层 C/C++ 数学库的线程暴走 (2核最佳策略: 算力隔离)
# =========================================================
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    BLIS_NUM_THREADS=1

# =========================================================
# 2. 基础环境与依赖安装
# =========================================================
# 设置国内源加速
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装所有重型第三方依赖 (此时为全局安装)
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
# 3. 安装业务 SDK
# =========================================================
# 将本地的 SDK 拷贝到镜像并作为 Python 库安装
COPY ./sandbox_sdk /tmp/sandbox_sdk
RUN pip install /tmp/sandbox_sdk && \
    rm -rf /tmp/sandbox_sdk

# =========================================================
# 4. 极致优化：Jupyter Kernel 冷启动预热脚本
# =========================================================
RUN if [ -d "/home/user" ]; then \
        mkdir -p /home/user/.ipython/profile_default/startup/ && \
        touch /home/user/.ipython/profile_default/startup/00-init.py && \
        echo "import os, sys, warnings" >> /home/user/.ipython/profile_default/startup/00-init.py && \
        echo "warnings.filterwarnings('ignore')" >> /home/user/.ipython/profile_default/startup/00-init.py && \
        echo "import pandas as pd" >> /home/user/.ipython/profile_default/startup/00-init.py && \
        echo "import numpy as np" >> /home/user/.ipython/profile_default/startup/00-init.py && \
        echo "try:" >> /home/user/.ipython/profile_default/startup/00-init.py && \
        echo "    import scipy, sklearn" >> /home/user/.ipython/profile_default/startup/00-init.py && \
        echo "    from sklearn.cluster import KMeans" >> /home/user/.ipython/profile_default/startup/00-init.py && \
        echo "    from sklearn.ensemble import IsolationForest" >> /home/user/.ipython/profile_default/startup/00-init.py && \
        echo "    import statsmodels.api as sm" >> /home/user/.ipython/profile_default/startup/00-init.py && \
        echo "    import xgboost as xgb, lightgbm as lgb, prophet" >> /home/user/.ipython/profile_default/startup/00-init.py && \
        echo "except Exception as e:" >> /home/user/.ipython/profile_default/startup/00-init.py && \
        echo "    print('Pre-warm warning:', e)" >> /home/user/.ipython/profile_default/startup/00-init.py && \
        chmod 644 /home/user/.ipython/profile_default/startup/00-init.py && \
        chown -R 1000:1000 /home/user/.ipython; \
    fi

# =========================================================
# 5. 系统清理与降权
# =========================================================
# 修改启动脚本权限 (忽略找不到文件的情况)
RUN if [ -f "/root/.jupyter/start-up.sh" ]; then chmod +x /root/.jupyter/start-up.sh; fi

# 所有的构建工作完成后，把用户身份降级回 1000（通常对应 user），保障沙盒安全
USER 1000