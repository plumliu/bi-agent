FROM image.ppinfra.com/sandbox/code-interpreter:latest

# 确保所有的 pip 安装都在系统全局目录（如 /usr/local/lib），而不是隐藏在普通用户的 ~/.local 里
USER root

# 1. 设置国内源加速
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 安装所有重型第三方依赖 (此时为全局安装)
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

# 3. 将本地的 SDK 拷贝到镜像并作为 Python 库安装
COPY ./sandbox_sdk /tmp/sandbox_sdk
RUN pip install /tmp/sandbox_sdk && \
    rm -rf /tmp/sandbox_sdk

# 4. 修复 Jupyter 工作目录权限
RUN if [ -d "/home/user" ]; then \
        mkdir -p /home/user/.ipython/profile_default/startup/ && \
        echo "import os; import sys" > /home/user/.ipython/profile_default/startup/00-init.py && \
        chown -R 1000:1000 /home/user/.ipython; \
    fi

# 5. 修改启动脚本权限 (此时依然是 root，不会报 Permission denied)
RUN chmod +x /root/.jupyter/start-up.sh

# 所有的构建工作完成后，把用户身份降级回 1000（通常对应 user）
USER 1000