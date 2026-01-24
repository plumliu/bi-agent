FROM image.ppinfra.com/sandbox/code-interpreter:latest

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
    joblib

RUN echo 'import os, sys, copy, builtins; builtins.os = os; builtins.sys = sys; builtins.copy = copy; builtins.E2BEnviron = type(os.environ)' > \
    $(python -c "import site; print(site.getsitepackages()[0])")/sitecustomize.py

RUN if [ -d "/home/user" ]; then \
        mkdir -p /home/user/.ipython/profile_default/startup/ && \
        echo "import os; import sys" > /home/user/.ipython/profile_default/startup/00-init.py && \
        chown -R 1000:1000 /home/user/.ipython; \
    fi

RUN chmod +x /root/.jupyter/start-up.sh