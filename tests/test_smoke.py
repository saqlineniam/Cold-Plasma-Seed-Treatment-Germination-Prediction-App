import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loading import load_df, define_columns
from src.preprocessing import build_preprocessor, get_feature_names_from_preprocessor
from src.models import build_models
from src.config import (
    DATA_PATH, TARGET, SEED_COLS, PLASMA_COLS, 
    GERMINATION_COLS, CATEGORICAL_COLS
)

def test_load_and_define_columns():
    """Test that data loading and column definition works."""
    # Check if data file exists
    assert DATA_PATH.exists(), f"Data file not found at {DATA_PATH}"
    
    # Load data
    df = load_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    
    # Define columns
    feature_cols, num_cols, cat_cols, target = define_columns(df)
    
    # Check target exists
    assert target == TARGET
    assert target in df.columns
    
    # Check feature columns
    assert len(feature_cols) > 0
    assert all(col in df.columns for col in feature_cols)
    
    # Check numeric/categorical split
    assert len(num_cols) + len(cat_cols) == len(feature_cols)
    assert all(col in feature_cols for col in num_cols + cat_cols)

def test_preprocessing():
    """Test that preprocessing pipeline can be built and transforms data."""
    # Load test data
    df = load_df()
    feature_cols, num_cols, cat_cols, _ = define_columns(df)
    
    # Build preprocessor
    prep = build_preprocessor(num_cols, cat_cols)
    
    # Test fit_transform
    X = df[feature_cols]
    X_transformed = prep.fit_transform(X)
    
    # Check output shape
    assert X_transformed.shape[0] == len(X)
    assert X_transformed.shape[1] >= len(num_cols)  # At least as many as numeric columns
    
    # Check feature names
    feat_names = get_feature_names_from_preprocessor(prep)
    assert len(feat_names) == X_transformed.shape[1]

def test_models():
    """Test that models can be built and have expected structure."""
    models = build_models(random_state=42)
    
    # Check model types
    assert isinstance(models, dict)
    assert len(models) > 0
    
    for name, (model, params) in models.items():
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert isinstance(params, dict)

def test_imports():
    """Test that all required modules can be imported."""
    import src.main
    import src.evaluation
    import src.mlflow_utils
    
    # If we get here, imports worked
    assert True
