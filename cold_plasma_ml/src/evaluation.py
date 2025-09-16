import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from .preprocessing import get_feature_names_from_preprocessor

def cv_scores(pipe: Pipeline, X, y, n_splits=5, n_repeats=3, random_state=None):
    """Calculate cross-validation scores for a pipeline."""
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    
    cv_mae = -cross_val_score(pipe, X, y, cv=rkf, scoring="neg_mean_absolute_error", n_jobs=-1)
    cv_rmse = -cross_val_score(pipe, X, y, cv=rkf, scoring="neg_root_mean_squared_error", n_jobs=-1)
    cv_r2 = cross_val_score(pipe, X, y, cv=rkf, scoring="r2", n_jobs=-1)
    
    return {
        "cv_MAE_mean": float(cv_mae.mean()), "cv_MAE_std": float(cv_mae.std()),
        "cv_RMSE_mean": float(cv_rmse.mean()), "cv_RMSE_std": float(cv_rmse.std()),
        "cv_R2_mean": float(cv_r2.mean()), "cv_R2_std": float(cv_r2.std()),
    }

def fit_and_test(pipe: Pipeline, X_train, y_train, X_test, y_test):
    """Fit pipeline on training data and evaluate on test data."""
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    test = {
        "test_MAE": float(mean_absolute_error(y_test, y_pred)),
        "test_RMSE": float(mean_squared_error(y_test, y_pred, squared=False)),
        "test_R2": float(r2_score(y_test, y_pred)),
    }
    
    preds_df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})
    return test, preds_df

def compute_permutation_importance(fitted_pipe: Pipeline, X_test, y_test, scoring="r2", n_repeats=10, random_state=None):
    """Compute permutation importance for a fitted pipeline."""
    # Get the preprocessed test data
    Xt = fitted_pipe.named_steps["prep"].transform(X_test)
    if hasattr(Xt, "toarray"):  # handle sparse matrices
        Xt = Xt.toarray()
    
    # Get the fitted estimator
    est = fitted_pipe.named_steps["model"]
    
    # Compute permutation importance
    pi = permutation_importance(
        est, Xt, y_test, 
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring=scoring
    )
    return pi

def plot_topk_pi(feat_names, pi, k=15, title="Permutation importance"):
    """Plot top-k permutation importances."""
    # Sort features by importance
    order = np.argsort(pi.importances_mean)[-k:]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.barh(feat_names[order], pi.importances_mean[order])
    ax.set_title(title)
    ax.set_xlabel("Permutation importance (mean Δscore)")
    ax.set_ylabel("")
    fig.tight_layout()
    
    return fig
