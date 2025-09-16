from pathlib import Path
import pandas as pd
from .config import DATA_PATH, TARGET, SEED_COLS, PLASMA_COLS, GERMINATION_COLS, CATEGORICAL_COLS

def load_df(path=DATA_PATH):
    """Load dataset from Excel or CSV file."""
    path = Path(path)
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
    else:
        # Default to Excel for backward compatibility
        df = pd.read_excel(path)
    return df

def define_columns(df: pd.DataFrame):
    """
    Filter to columns that actually exist in df.
    Returns: feature_cols, num_cols, cat_cols, target
    """
    def keep_existing(cols):
        return [c for c in cols if c in df.columns]
    
    seed_cols = keep_existing(SEED_COLS)
    plasma_cols = keep_existing(PLASMA_COLS)
    germ_cols = keep_existing(GERMINATION_COLS)
    cat_cols = keep_existing(CATEGORICAL_COLS)

    feature_cols = seed_cols + plasma_cols + germ_cols + cat_cols
    num_cols = [c for c in feature_cols if c not in cat_cols]
    
    return feature_cols, num_cols, cat_cols, TARGET
