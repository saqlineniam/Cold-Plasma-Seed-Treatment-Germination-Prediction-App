import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.pipeline import Pipeline
from src.config import EXPERIMENT_NAME, FEATURE_SET_NAME, MLFLOW_TRACKING_URI, OUTPUTS_DIR

def init_mlflow():
    """Initialize MLflow tracking."""
    mlflow.end_run()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

def log_core_params_for_compare(algo_name, estimator, n_features, cat_cols, feature_set_name=FEATURE_SET_NAME):
    """Log core parameters for model comparison."""
    mlflow.log_param("algo", algo_name)
    mlflow.log_param("feature_set", feature_set_name)
    mlflow.log_param("n_features", n_features)
    mlflow.log_param("categoricals", ",".join(cat_cols))
    
    keep = [
        "n_estimators", "max_depth", "learning_rate", "subsample",
        "colsample_bytree", "min_samples_split", "min_samples_leaf", "max_features"
    ]
    
    params = estimator.get_params()
    safe = {k: params[k] for k in keep if k in params}
    mlflow.log_params(safe)

def log_predictions_csv(preds_df: pd.DataFrame, model_name: str):
    """Log predictions as a CSV artifact."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    path = OUTPUTS_DIR / f"predictions_{model_name}.csv"
    preds_df.to_csv(path, index=False)
    mlflow.log_artifact(str(path))
    return str(path)

def log_permutation_plot(fig, model_name: str):
    """Log permutation importance plot as an artifact."""
    mlflow.log_figure(fig, f"permutation_importance_{model_name}.png")

def log_sklearn_model(fitted_pipe: Pipeline, model_name: str, X_train_sample):
    """Log a scikit-learn model to MLflow."""
    mlflow.sklearn.log_model(
        sk_model=fitted_pipe,
        artifact_path="model",
        input_example=X_train_sample,
        registered_model_name=f"SeedGerm_{FEATURE_SET_NAME}_{model_name}",
    )

def log_comparison_run(results_df: pd.DataFrame, feature_set_name=FEATURE_SET_NAME):
    """Create a separate run with aggregate comparisons (plots + CSV)."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    csv_path = OUTPUTS_DIR / "model_comparison.csv"
    results_df.to_csv(csv_path, index=True)
    
    with mlflow.start_run(run_name=f"{feature_set_name}__COMPARISON", nested=False):
        mlflow.log_param("feature_set", feature_set_name)
        mlflow.log_artifact(str(csv_path))
        
        for metric in ["test_MAE", "test_RMSE", "test_R2"]:
            if metric in results_df.columns:
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.bar(results_df.index, results_df[metric].values)
                ax.set_title(f"Model Comparison — {metric}")
                ax.set_ylabel(metric)
                ax.grid(axis="y", linestyle="--", alpha=0.4)
                fig.tight_layout()
                mlflow.log_figure(fig, f"compare_{metric}.png")
                plt.close(fig)
        
        if {"cv_MAE_mean", "cv_MAE_std"}.issubset(results_df.columns):
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.bar(
                results_df.index, 
                results_df["cv_MAE_mean"].values,
                yerr=results_df["cv_MAE_std"].values, 
                capsize=5
            )
            ax.set_title("Cross-Validation MAE (mean ± std)")
            ax.set_ylabel("MAE")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            fig.tight_layout()
            mlflow.log_figure(fig, "compare_cv_MAE.png")
            plt.close(fig)
