from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

def build_models(random_state=None):
    """
    Return dict of name -> (estimator, param_grid).
    Random state is passed through for reproducibility.
    """
    return {
        "ET": (
            ExtraTreesRegressor(random_state=random_state, n_jobs=-1),
            {
                "model__n_estimators": [500],
                "model__max_depth": [8],
                "model__min_samples_leaf": [2],
                "model__min_samples_split": [4],
            }
        ),
        "GB": (
            GradientBoostingRegressor(random_state=random_state),
            {
                'model__n_estimators': [400],
                'model__learning_rate': [0.01],
                'model__max_depth': [2],
                'model__min_samples_split': [4],
                'model__min_samples_leaf': [4],
            }
        ),
        "XGB": (
            XGBRegressor(random_state=random_state, n_jobs=-1, tree_method='hist'),
            {
                "model__n_estimators": [300],
                "model__learning_rate": [0.05],
                "model__max_depth": [5],
                "model__subsample": [0.4],
                "model__colsample_bytree": [0.8],
                "model__min_child_weight": [4],
                "model__gamma": [10],
            }
        )
    }
