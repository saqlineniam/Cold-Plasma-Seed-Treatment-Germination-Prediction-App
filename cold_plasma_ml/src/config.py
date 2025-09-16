from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "New_dataset.xlsx"
MLFLOW_TRACKING_URI = "file:./mlruns"  # local folder

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
