import argparse
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from .config import (
    TEST_SIZE, FEATURE_SET_NAME, PRIMARY_METRIC, TARGET,
    RANDOM_STATE, EXPERIMENT_NAME, MLFLOW_TRACKING_URI
)
from .data_loading import load_df, define_columns
from .preprocessing import build_preprocessor, get_feature_names_from_preprocessor
from .models import build_models
from .evaluation import cv_scores, fit_and_test, compute_permutation_importance, plot_topk_pi
from .mlflow_utils import (
    init_mlflow, log_core_params_for_compare, log_predictions_csv,
    log_permutation_plot, log_sklearn_model, log_comparison_run
)

def run_one_model(model_name, model_tuple, prep, X_train, y_train, X_test, y_test,
                 feature_cols, cat_cols, random_state=None):
    """Train and evaluate a single model with hyperparameter tuning."""
    # Unpack model and parameter grid
    base_model, param_grid = model_tuple
    
    # Create pipeline
    pipe = Pipeline([
        ("prep", prep),
        ("model", base_model)
    ])
    
    # Set up MLflow run
    with mlflow.start_run(run_name=f"{FEATURE_SET_NAME}__{model_name}", nested=True):
        # Log core parameters
        log_core_params_for_compare(model_name, base_model, len(feature_cols), cat_cols)
        
        # Hyperparameter tuning with cross-validation
        search = GridSearchCV(
            pipe, param_grid, cv=5, n_jobs=-1, 
            scoring='neg_mean_absolute_error',
            return_train_score=True
        )
        
        # Fit the model
        search.fit(X_train, y_train)
        
        # Get the best model
        best_pipe = search.best_estimator_
        
        # Log best parameters
        mlflow.log_params({
            f"best_{k}": v for k, v in search.best_params_.items()
        })
        
        # Cross-validation scores
        cv = cv_scores(best_pipe, X_train, y_train, random_state=random_state)
        mlflow.log_metrics(cv)
        
        # Test set evaluation
        test, preds_df = fit_and_test(best_pipe, X_train, y_train, X_test, y_test)
        mlflow.log_metrics(test)
        
        # Log predictions
        log_predictions_csv(preds_df, model_name)
        
        # Permutation importance
        feat_names = get_feature_names_from_preprocessor(best_pipe.named_steps["prep"])
        pi = compute_permutation_importance(
            best_pipe, X_test, y_test, 
            scoring=PRIMARY_METRIC, 
            random_state=random_state
        )
        
        # Plot and log permutation importance
        fig = plot_topk_pi(
            feat_names, pi, 
            k=15, 
            title=f"Top 15 Features — {model_name}"
        )
        log_permutation_plot(fig, model_name)
        
        # Log the model
        log_sklearn_model(best_pipe, model_name, X_train.iloc[:2])
        
        # Prepare output
        out = {"model": model_name}
        out.update(cv)
        out.update(test)
        
        return out

def train_hybrid_models(base_models, prep, X_train, y_train, X_test, y_test, 
                       feature_cols, cat_cols, random_state=None):
    """Train and evaluate hybrid stacking models."""
    from sklearn.linear_model import RidgeCV
    
    # Prepare base model predictions for stacking
    base_preds_train = {}
    base_preds_test = {}
    
    # Train base models and get predictions
    for name, (model, _) in base_models.items():
        pipe = Pipeline([("prep", prep), ("model", model)])
        
        # Get out-of-fold predictions for training the meta-model
        cv_preds = cross_val_predict(
            pipe, X_train, y_train, 
            cv=5, n_jobs=-1,
            method="predict"
        )
        base_preds_train[name] = cv_preds
        
        # Get test set predictions
        pipe.fit(X_train, y_train)
        test_preds = pipe.predict(X_test)
        base_preds_test[name] = test_preds
    
    # Define hybrid model configurations
    hybrid_configs = [
        (['ET', 'GB'], 'HM1_ET_GB'),
        (['ET', 'XGB'], 'HM2_ET_XGB'),
        (['GB', 'XGB'], 'HM3_GB_XGB'),
        (['ET', 'GB', 'XGB'], 'HM4_ET_GB_XGB')
    ]
    
    # Train and evaluate each hybrid model
    hybrid_results = []
    for base_names, model_name in hybrid_configs:
        with mlflow.start_run(run_name=f"{FEATURE_SET_NAME}__{model_name}", nested=True):
            # Prepare meta-features
            X_meta_train = np.column_stack([base_preds_train[name] for name in base_names])
            X_meta_test = np.column_stack([base_preds_test[name] for name in base_names])
            
            # Train meta-model (RidgeCV with built-in cross-validation)
            meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
            meta_model.fit(X_meta_train, y_train)
            
            # Evaluate on test set
            y_pred = meta_model.predict(X_meta_test)
            
            # Calculate metrics
            metrics = {
                'test_MAE': mean_absolute_error(y_test, y_pred),
                'test_RMSE': mean_squared_error(y_test, y_pred, squared=False),
                'test_R2': r2_score(y_test, y_pred)
            }
            
            # Log parameters and metrics
            mlflow.log_param("base_models", ", ".join(base_names))
            mlflow.log_param("meta_model", "RidgeCV")
            mlflow.log_param("best_alpha", float(meta_model.alpha_))
            mlflow.log_metrics(metrics)
            
            # Log predictions
            preds_df = pd.DataFrame({
                'y_true': y_test,
                'y_pred': y_pred,
                'model': model_name
            })
            log_predictions_csv(preds_df, model_name)
            
            # Store results
            result = {
                'model': model_name,
                'base_models': ", ".join(base_names),
                'meta_model': 'RidgeCV',
                'best_alpha': float(meta_model.alpha_)
            }
            result.update(metrics)
            hybrid_results.append(result)
    
    return hybrid_results

def main(random_state=None):
    """Main function to run the complete pipeline."""
    # Set random seed for reproducibility
    if random_state is None:
        random_state = RANDOM_STATE
    
    # Initialize MLflow
    init_mlflow()
    
    try:
        # Load and prepare data
        df = load_df()
        feature_cols, num_cols, cat_cols, target = define_columns(df)
        
        # Split data
        X = df[feature_cols]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=random_state
        )
        
        # Build preprocessing pipeline
        prep = build_preprocessor(num_cols, cat_cols)
        
        # Get models
        models = build_models(random_state=random_state)
        
        # Train and evaluate each base model
        base_results = []
        for model_name, model_tuple in models.items():
            with mlflow.start_run(run_name=f"{FEATURE_SET_NAME}__{model_name}", nested=True):
                summary = run_one_model(
                    model_name, model_tuple, prep,
                    X_train, y_train, X_test, y_test,
                    feature_cols, cat_cols, random_state
                )
                summary["type"] = "base"
                base_results.append(summary)
        
        # Train and evaluate hybrid models
        hybrid_results = train_hybrid_models(
            models, prep, X_train, y_train, X_test, y_test,
            feature_cols, cat_cols, random_state
        )
        
        # Log comparison of all models
        mlflow.end_run()  # End any active runs
        
        # Combine results
        all_results = pd.DataFrame(base_results + hybrid_results)
        
        # Log comparison
        log_comparison_run(
            all_results.set_index("model"), 
            feature_set_name=f"{FEATURE_SET_NAME}_rs{random_state}"
        )
        
        return all_results
        
    except Exception as e:
        import traceback
        print(f"Error in main execution: {str(e)}")
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_state", type=int, default=None,
                       help="Random seed for split, models, CV, and PI.")
    args = parser.parse_args()
    
    results = main(random_state=args.random_state)
    print("\nModel Comparison:")
    print(results[['model', 'test_MAE', 'test_RMSE', 'test_R2']].to_string(index=False))
