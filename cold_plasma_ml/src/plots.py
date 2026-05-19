import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_actual_vs_predicted(y_true, y_pred, model_name="Model"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, color='#10b981')
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    ax.set_title(f"Actual vs Predicted - {model_name}")
    ax.set_xlabel("Actual Germination Rate")
    ax.set_ylabel("Predicted Germination Rate")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def plot_residuals(y_true, y_pred, model_name="Model"):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='#0ea5e9', ax=ax)
    
    ax.set_title(f"Residuals Distribution - {model_name}")
    ax.set_xlabel("Error (Actual - Predicted)")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
