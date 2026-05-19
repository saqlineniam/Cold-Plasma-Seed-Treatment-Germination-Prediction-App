import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = os.getenv("DATA_FILE", "train.xlsx")
DATA_PATH = PROJECT_ROOT / "data" / DATA_FILE
_default_mlruns = str(PROJECT_ROOT / "mlruns") if os.access(str(PROJECT_ROOT), os.W_OK) else "/tmp/mlruns"
MLFLOW_TRACKING_URI = "file:///tmp/mlruns" if os.environ.get("STREAMLIT_RUNTIME_ENV") else os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

# Experiment / Feature Set names
EXPERIMENT_NAME = "cold_plasma_seed_priming"
FEATURE_SET_NAME = "WITH_BASELINE"

# ML settings
PRIMARY_METRIC = "r2"  # for permutation importance scoring
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Columns (the code will auto-filter to existing columns)
TARGET = "germination rate"
SEED_COLS = [
    "size of each seed (mm)", "weight of each seed (gr)",
    "baseline SOD (u g-1)", "base germination rate",
    "base germination potential", "base germination index",
]
PLASMA_COLS = ["voltage (kV)", "power (w)", "plasma time"]
GERMINATION_COLS = ["germination days"]
CATEGORICAL_COLS = ["gas"]
