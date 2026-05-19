# cold_plasma_ml

Reproducible ML pipeline to predict **post-plasma germination rate** from baseline seed metrics and plasma settings, with **MLflow** tracking.

## Quickstart
```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows
pip install -r requirements.txt

# Put your dataset at data/New_dataset.xlsx (or edit src/config.py)
# Run:
python -m src.main --random_state 42

# Launch UI:
mlflow ui
```

Open http://127.0.0.1:5000 and explore the runs.
