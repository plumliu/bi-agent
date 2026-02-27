from setuptools import setup, find_packages

setup(
    name='bi_sandbox_sdk',
    version='0.1.0',
    packages=find_packages(),
    description='BI Agent Custom Sandbox Algorithms SDK',
    author='BI Team',
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'pyarrow',
        'statsmodels',
        'scikit-learn',
        'xgboost',
        'lightgbm',
        'openpyxl',
        'joblib',
    ],
)
