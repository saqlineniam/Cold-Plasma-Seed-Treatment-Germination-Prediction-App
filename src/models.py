from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor

def build_models(random_state=None):
    """
    Return dict of name -> (estimator, param_grid).
    Improved models and expanded search grids for better performance.
    """
    return {
        "RF": (
            RandomForestRegressor(random_state=random_state, n_jobs=-1),
            {
                "model__n_estimators": [300, 500],
                "model__max_depth": [10, 15, None],
                "model__min_samples_leaf": [1, 2],
            }
        ),
        "ET": (
            ExtraTreesRegressor(random_state=random_state, n_jobs=-1),
            {
                "model__n_estimators": [500, 1000],
                "model__max_depth": [10, 20, None],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": [1.0, "sqrt"],
            }
        ),
        "GB": (
            GradientBoostingRegressor(random_state=random_state),
            {
                'model__n_estimators': [500, 1000],
                'model__learning_rate': [0.01, 0.05],
                'model__max_depth': [3, 4, 5],
                'model__subsample': [0.8, 1.0],
            }
        ),
        "XGB": (
            XGBRegressor(random_state=random_state, n_jobs=-1, tree_method='hist'),
            {
                "model__n_estimators": [500, 800],
                "model__learning_rate": [0.01, 0.05],
                "model__max_depth": [5, 7, 9],
                "model__subsample": [0.6, 0.8],
                "model__colsample_bytree": [0.7, 0.9],
            }
        )
    }
