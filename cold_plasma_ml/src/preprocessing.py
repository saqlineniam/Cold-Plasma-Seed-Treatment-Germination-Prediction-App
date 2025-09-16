from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

def build_preprocessor(num_cols, cat_cols):
    """
    Mean imputation + StandardScaler for numeric; most_frequent + OHE for categoricals.
    """
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    
    prep = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )
    return prep

def get_feature_names_from_preprocessor(prep) -> np.ndarray:
    """Robustly retrieve output feature names after preprocessing."""
    try:
        return prep.get_feature_names_out()
    except Exception:
        names = []
        for name, trans, cols in prep.transformers_:
            if name == "num":
                names.extend(list(cols))
            elif name == "cat":
                ohe = trans.named_steps.get("onehot")
                if ohe is not None:
                    try:
                        names.extend(ohe.get_feature_names_out(cols))
                    except Exception:
                        names.extend([f"{c}_cat" for c in cols])
        return np.array(names, dtype=object)
