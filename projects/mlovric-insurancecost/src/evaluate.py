"""Evaluation utilities and plots."""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir


def save_results_table(results: list, out_dir: Path) -> Path:
    """Save model results as a CSV."""
    ensure_dir(out_dir)
    df = pd.DataFrame([r.__dict__ for r in results])
    df = df.sort_values("rmse")
    out_path = out_dir / "model_comparison.csv"
    df.to_csv(out_path, index=False)
    return out_path


def plot_model_comparison(results: list, out_path: Path) -> None:
    """Simple bar chart of RMSE."""
    df = pd.DataFrame([r.__dict__ for r in results]).sort_values("rmse")

    plt.figure()
    plt.bar(df["name"], df["rmse"])
    plt.title("Model RMSE Comparison (lower is better)")
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def load_model(model_path: Path):
    return joblib.load(model_path)
