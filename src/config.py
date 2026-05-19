import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = os.getenv("DATA_FILE", "train.xlsx")
DATA_PATH = PROJECT_ROOT / "data" / DATA_FILE

# MLflow path fix for Streamlit Cloud
if os.environ.get("STREAMLIT_RUNTIME_ENV"):
    # Force fresh local storage in /tmp to avoid any carried-over path issues
    MLFLOW_TRACKING_URI = "file:///tmp/mlruns"
    OUTPUTS_DIR = Path("/tmp/outputs")
else:
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:{(PROJECT_ROOT / 'mlruns').as_posix()}")
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Experiment / Feature Set names
EXPERIMENT_NAME = "cold_plasma_seed_priming"
FEATURE_SET_NAME = "WITH_BASELINE"

# ML settings
PRIMARY_METRIC = "r2"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Columns
TARGET = "germination rate"
SEED_COLS = [
    "size of each seed (mm)", "weight of each seed (gr)",
    "baseline SOD (u g-1)", "base germination rate",
    "base germination potential", "base germination index",
]
PLASMA_COLS = ["voltage (kV)", "power (w)", "plasma time"]
GERMINATION_COLS = ["germination days"]
CATEGORICAL_COLS = ["gas"]
