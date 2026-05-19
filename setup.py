from setuptools import setup, find_packages

setup(
    name="cold_plasma_ml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "mlflow>=2.10",
        "pandas>=2.0",
        "numpy>=1.25",
        "matplotlib>=3.7",
        "openpyxl>=3.1",
    ],
    description="Cold plasma seed priming ML: reproducible pipelines with MLflow.",
    author="you",
)
