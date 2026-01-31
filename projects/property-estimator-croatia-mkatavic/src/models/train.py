"""
Skripta za treniranje LightGBM modela za predikciju cijena nekretnina.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path


# Putanje
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(dataset_type: str) -> pd.DataFrame:
    """Ucitava ociscene podatke."""
    if dataset_type == "houses":
        return pd.read_parquet(DATA_DIR / "houses_clean.parquet")
    elif dataset_type == "apartments":
        return pd.read_parquet(DATA_DIR / "apartments_clean.parquet")
    else:
        raise ValueError(f"Nepoznat dataset: {dataset_type}")


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Priprema features i target varijablu."""
    # Target
    y = df["cijena"]

    # Features - ukloni target i kategoricke stupce
    exclude_cols = ["cijena", "zupanija", "grad_opcina", "naselje"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()

    # Kategoricke varijable - label encoding
    for col in ["zupanija", "grad_opcina", "naselje"]:
        if col in df.columns:
            X[col] = df[col].astype("category").cat.codes

    return X, y


def train_model(X: pd.DataFrame, y: pd.Series, dataset_name: str) -> tuple:
    """Trenira LightGBM model."""

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTreniranje modela za {dataset_name}...")
    print(f"  Train set: {len(X_train)} uzoraka")
    print(f"  Test set: {len(X_test)} uzoraka")
    print(f"  Broj featurea: {len(X.columns)}")

    # LightGBM parametri
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_estimators": 1000,
        "early_stopping_rounds": 50,
    }

    # Treniraj model
    model = lgb.LGBMRegressor(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
    )

    # Predikcije
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrike
    metrics = {
        "train": {
            "rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "mae": mean_absolute_error(y_train, y_pred_train),
            "r2": r2_score(y_train, y_pred_train),
        },
        "test": {
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "mae": mean_absolute_error(y_test, y_pred_test),
            "r2": r2_score(y_test, y_pred_test),
        }
    }

    return model, metrics, X.columns.tolist()


def print_metrics(metrics: dict, dataset_name: str):
    """Ispisuje metrike modela."""
    print(f"\n{'='*50}")
    print(f"REZULTATI - {dataset_name.upper()}")
    print("="*50)

    print("\nTrain set:")
    print(f"  RMSE: {metrics['train']['rmse']:,.2f} EUR")
    print(f"  MAE:  {metrics['train']['mae']:,.2f} EUR")
    print(f"  R2:   {metrics['train']['r2']:.4f}")

    print("\nTest set:")
    print(f"  RMSE: {metrics['test']['rmse']:,.2f} EUR")
    print(f"  MAE:  {metrics['test']['mae']:,.2f} EUR")
    print(f"  R2:   {metrics['test']['r2']:.4f}")


def print_feature_importance(model, feature_names: list, top_n: int = 15):
    """Ispisuje najvaznije featuree."""
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print(f"\nTop {top_n} najvaznijih featurea:")
    print("-" * 40)
    for i, row in importance.head(top_n).iterrows():
        print(f"  {row['feature']}: {row['importance']}")


def save_model(model, feature_names: list, metrics: dict, dataset_name: str):
    """Sprema model i metapodatke."""
    model_path = MODELS_DIR / f"{dataset_name}_model.joblib"

    model_data = {
        "model": model,
        "feature_names": feature_names,
        "metrics": metrics,
    }

    joblib.dump(model_data, model_path)
    print(f"\nModel spremljen: {model_path}")


def main():
    print("="*50)
    print("TRENIRANJE MODELA ZA PREDIKCIJU CIJENA NEKRETNINA")
    print("="*50)

    # Treniraj model za kuce
    print("\n" + "="*50)
    print("1. KUCE")
    print("="*50)

    df_houses = load_data("houses")
    X_houses, y_houses = prepare_features(df_houses)
    model_houses, metrics_houses, features_houses = train_model(X_houses, y_houses, "kuce")

    print_metrics(metrics_houses, "kuce")
    print_feature_importance(model_houses, features_houses)
    save_model(model_houses, features_houses, metrics_houses, "houses")

    # Treniraj model za stanove
    print("\n" + "="*50)
    print("2. STANOVI")
    print("="*50)

    df_apartments = load_data("apartments")
    X_apartments, y_apartments = prepare_features(df_apartments)
    model_apartments, metrics_apartments, features_apartments = train_model(X_apartments, y_apartments, "stanovi")

    print_metrics(metrics_apartments, "stanovi")
    print_feature_importance(model_apartments, features_apartments)
    save_model(model_apartments, features_apartments, metrics_apartments, "apartments")

    # Sazetak
    print("\n" + "="*50)
    print("SAZETAK")
    print("="*50)
    print(f"\nKuce:")
    print(f"  Test R2: {metrics_houses['test']['r2']:.4f}")
    print(f"  Test MAE: {metrics_houses['test']['mae']:,.0f} EUR")

    print(f"\nStanovi:")
    print(f"  Test R2: {metrics_apartments['test']['r2']:.4f}")
    print(f"  Test MAE: {metrics_apartments['test']['mae']:,.0f} EUR")

    print("\nGotovo!")


if __name__ == "__main__":
    main()
