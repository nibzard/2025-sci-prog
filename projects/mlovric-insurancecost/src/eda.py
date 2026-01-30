"""Exploratory Data Analysis.

This file creates plots and saves them to reports/figures.
No notebook needed: you just run `python run.py eda`.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir


def plot_distribution(series: pd.Series, title: str, out_path: Path) -> None:
    plt.figure()
    plt.hist(series, bins=30)
    plt.title(title)
    plt.xlabel(series.name)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_box(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: Path) -> None:
    """A basic boxplot using matplotlib."""
    plt.figure()
    df.boxplot(column=y_col, by=x_col)
    plt.title(title)
    plt.suptitle("")  # removes the automatic pandas subtitle
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, title: str, out_path: Path) -> None:
    """Simple correlation heatmap for numeric columns."""
    corr = df.select_dtypes("number").corr()

    plt.figure()
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_eda(df: pd.DataFrame, figures_dir: Path) -> None:
    ensure_dir(figures_dir)

    plot_distribution(df["age"], "Age distribution", figures_dir / "age_distribution.png")
    plot_distribution(df["bmi"], "BMI distribution", figures_dir / "bmi_distribution.png")
    plot_distribution(df["charges"], "Charges distribution", figures_dir / "charges_distribution.png")

    plot_box(df, "smoker", "charges", "Charges by smoker", figures_dir / "charges_by_smoker.png")
    plot_box(df, "sex", "charges", "Charges by sex", figures_dir / "charges_by_sex.png")
    plot_box(df, "region", "charges", "Charges by region", figures_dir / "charges_by_region.png")

    plot_correlation_heatmap(df, "Correlation heatmap (numeric features)", figures_dir / "correlation_heatmap.png")
