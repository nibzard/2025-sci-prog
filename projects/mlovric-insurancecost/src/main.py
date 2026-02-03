"""Main entry points.

We keep this file small and call other modules for the real work.
"""

from pathlib import Path
import pandas as pd

from .config import Paths, RANDOM_SEED
from .data_loader import load_raw_data, quick_info
from .preprocessing import clean_data, save_clean_data
from .features import add_features
from .eda import run_eda
from .train import train_and_compare, save_best_model
from .evaluate import save_results_table, plot_model_comparison
from .interpretation import top_linear_coefficients, top_tree_importances


def run(command: str) -> None:
    paths = Paths()

    # Load + clean + feature engineering (shared for all commands)
    df = load_raw_data(paths)
    df = clean_data(df)
    df = add_features(df)

    save_clean_data(df, paths)

    print(quick_info(df))

    if command == "eda":
        run_eda(df, paths.reports_figures)
        print(f"Saved EDA plots to: {paths.reports_figures}")
        return

    if command == "train":
        results, best_pipeline = train_and_compare(df, RANDOM_SEED)

        # Save results + plot
        metrics_csv = save_results_table(results, paths.reports_metrics)
        plot_model_comparison(results, paths.reports_figures / "model_rmse_comparison.png")

        model_path = save_best_model(best_pipeline, paths.models_dir)

        print(f"Saved metrics to: {metrics_csv}")
        print(f"Saved best model to: {model_path}")

        # Quick interpretation printout
        lin = top_linear_coefficients(best_pipeline, top_n=10)
        tree = top_tree_importances(best_pipeline, top_n=10)

        if not lin.empty:
            print("\nTop linear coefficients (absolute):")
            print(lin.to_string(index=False))

        if not tree.empty:
            print("\nTop tree importances:")
            print(tree.to_string(index=False))

        return

    if command == "eval":
        # For this simple project, 'train' already makes metrics/plots.
        # Eval can just remind the user what to run.
        model_path = paths.models_dir / "best_model.joblib"
        if not model_path.exists():
            print("No saved model found. Run: python run.py train")
            return

        print(f"Found saved model at: {model_path}")
        print("Tip: run `python run.py train` again if you changed the code or features.")
        return

    print("Unknown command. Use: eda | train | eval")
