import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from src.config import (
    TEST_SIZE, FEATURE_SET_NAME, PRIMARY_METRIC, TARGET,
    RANDOM_STATE, EXPERIMENT_NAME, MLFLOW_TRACKING_URI
)
from src.data_loading import load_df, define_columns
from src.preprocessing import build_preprocessor, get_feature_names_from_preprocessor
from src.models import build_models
from src.evaluation import cv_scores, fit_and_test, compute_permutation_importance, plot_topk_pi
from src.mlflow_utils import (
    init_mlflow, log_core_params_for_compare, log_predictions_csv,
    log_permutation_plot, log_sklearn_model, log_comparison_run
)

def run_one_model(model_name, model_tuple, prep, X_train, y_train, X_test, y_test,
                 feature_cols, cat_cols, random_state=None):
    """Train and evaluate a single model with hyperparameter tuning."""
    base_model, param_grid = model_tuple
    pipe = Pipeline([("prep", prep), ("model", base_model)])
    
    with mlflow.start_run(run_name=f"{FEATURE_SET_NAME}__{model_name}", nested=True):
        log_core_params_for_compare(model_name, base_model, len(feature_cols), cat_cols)
        
        search = GridSearchCV(
            pipe, param_grid, cv=5, n_jobs=-1, 
            scoring='neg_mean_absolute_error',
            return_train_score=True
        )
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_
        
        mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})
        cv = cv_scores(best_pipe, X_train, y_train, random_state=random_state)
        mlflow.log_metrics(cv)
        test, preds_df, fig_avp, fig_res = fit_and_test(best_pipe, X_train, y_train, X_test, y_test, model_name=model_name)
        mlflow.log_metrics(test)
        
        # Log visual analytics
        mlflow.log_figure(fig_avp, f"actual_vs_predicted_{model_name}.png")
        mlflow.log_figure(fig_res, f"residuals_{model_name}.png")
        
        log_predictions_csv(preds_df, model_name)
        feat_names = get_feature_names_from_preprocessor(best_pipe.named_steps["prep"])
        pi = compute_permutation_importance(best_pipe, X_test, y_test, scoring=PRIMARY_METRIC, random_state=random_state)
        fig = plot_topk_pi(feat_names, pi, k=15, title=f"Top 15 Features — {model_name}")
        log_permutation_plot(fig, model_name)
        log_sklearn_model(best_pipe, model_name, X_train.iloc[:2])
        
        out = {"model": model_name}
        out.update(cv)
        out.update(test)
        out["best_pipe"] = best_pipe
        return out

def train_hybrid_models(base_models, prep, X_train, y_train, X_test, y_test, 
                       feature_cols, cat_cols, random_state=None):
    """Train and evaluate hybrid stacking models."""
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    base_preds_train = {}
    base_preds_test = {}
    
    for name, (model, _) in base_models.items():
        pipe = Pipeline([("prep", prep), ("model", model)])
        cv_preds = cross_val_predict(pipe, X_train, y_train, cv=5, n_jobs=-1)
        base_preds_train[name] = cv_preds
        pipe.fit(X_train, y_train)
        base_preds_test[name] = pipe.predict(X_test)
    
    hybrid_configs = [(['ET', 'GB'], 'HM1_ET_GB'), (['ET', 'XGB'], 'HM2_ET_XGB'), (['GB', 'XGB'], 'HM3_GB_XGB'), (['ET', 'GB', 'XGB'], 'HM4_ET_GB_XGB')]
    hybrid_results = []
    for base_names, model_name in hybrid_configs:
        with mlflow.start_run(run_name=f"{FEATURE_SET_NAME}__{model_name}", nested=True):
            X_meta_train = np.column_stack([base_preds_train[name] for name in base_names])
            X_meta_test = np.column_stack([base_preds_test[name] for name in base_names])
            meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
            meta_model.fit(X_meta_train, y_train)
            y_pred = meta_model.predict(X_meta_test)
            metrics = {'test_MAE': mean_absolute_error(y_test, y_pred), 'test_RMSE': np.sqrt(mean_squared_error(y_test, y_pred)), 'test_R2': r2_score(y_test, y_pred)}
            mlflow.log_params({"base_models": ", ".join(base_names), "meta_model": "RidgeCV", "best_alpha": float(meta_model.alpha_)})
            mlflow.log_metrics(metrics)
            log_predictions_csv(pd.DataFrame({'y_true': y_test, 'y_pred': y_pred, 'model': model_name}), model_name)
            res = {'model': model_name}
            res.update(metrics)
            hybrid_results.append(res)
    return hybrid_results

def main(random_state=None, selected_model_name=None, custom_params=None):
    if random_state is None: random_state = RANDOM_STATE
    init_mlflow()
    try:
        df = load_df()
        feature_cols, num_cols, cat_cols, target = define_columns(df)
        X = df[feature_cols]; y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=random_state)
        prep = build_preprocessor(num_cols, cat_cols)
        models = build_models(random_state=random_state)
        
        import joblib
        from src.config import PROJECT_ROOT
        _outputs_dir = PROJECT_ROOT / "outputs" if os.access(str(PROJECT_ROOT), os.W_OK) else Path("/tmp/outputs")
        os.makedirs(str(_outputs_dir), exist_ok=True)
        _model_out = str(_outputs_dir / "model.pkl")

        if selected_model_name and selected_model_name in models:
            summary = run_one_model(selected_model_name, (models[selected_model_name][0], custom_params or models[selected_model_name][1]), prep, X_train, y_train, X_test, y_test, feature_cols, cat_cols, random_state)
            joblib.dump(summary["best_pipe"], _model_out)
            return [summary]

        base_results = []
        for name, tup in models.items():
            res = run_one_model(name, tup, prep, X_train, y_train, X_test, y_test, feature_cols, cat_cols, random_state)
            base_results.append(res)
        
        hybrid_results = train_hybrid_models(models, prep, X_train, y_train, X_test, y_test, feature_cols, cat_cols, random_state)
        all_res = pd.DataFrame(base_results + hybrid_results)
        best_row = all_res.sort_values("test_R2", ascending=False).iloc[0]
        print(f"\n🏆 Best Model: {best_row['model']} (R2: {best_row['test_R2']:.4f})")
        
        best_model_obj = next((r["best_pipe"] for r in base_results if r["model"] == best_row["model"]), base_results[0]["best_pipe"])
        joblib.dump(best_model_obj, _model_out)
        
        log_comparison_run(all_res.drop(columns=["best_pipe"], errors="ignore").set_index("model"), feature_set_name=f"{FEATURE_SET_NAME}_rs{random_state}")
        return all_res
    except Exception as e:
        import traceback; print(f"Error: {e}"); print(traceback.format_exc()); raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_state", type=int, default=None)
    args = parser.parse_args()
    results = main(random_state=args.random_state)
