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
                "model__max_depth": [10, 20, None],
                "model__min_samples_leaf": [1, 2],
            }
        ),
        "ET": (
            ExtraTreesRegressor(random_state=random_state, n_jobs=-1),
            {
                "model__n_estimators": [500, 1000],
                "model__max_depth": [20, None],
                "model__min_samples_leaf": [1, 2],
            }
        ),
        "GB": (
            GradientBoostingRegressor(random_state=random_state),
            {
                'model__n_estimators': [500, 1000],
                'model__learning_rate': [0.01, 0.05],
                'model__max_depth': [3, 4, 5],
            }
        ),
        "XGB": (
            XGBRegressor(random_state=random_state, n_jobs=-1, tree_method='hist'),
            {
                "model__n_estimators": [500, 1000],
                "model__learning_rate": [0.01, 0.05],
                "model__max_depth": [5, 7, 9],
                "model__subsample": [0.6, 0.8],
            }
        )
    }
