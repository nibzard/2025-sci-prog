"""
Optimizirana skripta za treniranje modela s:
- Outlier removal
- Feature engineering
- Hyperparameter tuning (Optuna)
- Cross-validation
- Usporedba LightGBM vs CatBoost
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cb
import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Putanje
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_YEAR = 2026


def load_data(dataset_type: str) -> pd.DataFrame:
    """Ucitava ociscene podatke."""
    if dataset_type == "houses":
        return pd.read_parquet(DATA_DIR / "houses_clean.parquet")
    elif dataset_type == "apartments":
        return pd.read_parquet(DATA_DIR / "apartments_clean.parquet")
    else:
        raise ValueError(f"Nepoznat dataset: {dataset_type}")


def remove_outliers(df: pd.DataFrame, column: str = "cijena", lower_pct: float = 1, upper_pct: float = 99) -> pd.DataFrame:
    """Uklanja outliere bazirano na percentilima."""
    lower = np.percentile(df[column], lower_pct)
    upper = np.percentile(df[column], upper_pct)

    mask = (df[column] >= lower) & (df[column] <= upper)
    removed = len(df) - mask.sum()

    print(f"  Uklonjeno {removed} outliera ({removed/len(df)*100:.1f}%)")
    return df[mask].copy()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje nove featuree."""
    df = df.copy()

    # Starost nekretnine
    if "godina_izgradnje" in df.columns:
        df["starost"] = CURRENT_YEAR - df["godina_izgradnje"]
        df["starost"] = df["starost"].clip(lower=0)

    # Godine od renovacije
    if "godina_renovacije" in df.columns:
        df["godine_od_renovacije"] = CURRENT_YEAR - df["godina_renovacije"]
        df["godine_od_renovacije"] = df["godine_od_renovacije"].clip(lower=0)

    # Povrsina transformacije
    if "stambena_povrsina" in df.columns:
        df["povrsina_log"] = np.log1p(df["stambena_povrsina"])

    # Ukupan broj kupaonica
    bathroom_cols = ["wc_broj", "kupaonica_s_wc_broj"]
    existing_cols = [c for c in bathroom_cols if c in df.columns]
    if existing_cols:
        df["ukupno_kupaonica"] = df[existing_cols].sum(axis=1, skipna=True)

    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Priprema features i target varijablu."""
    y = df["cijena"]

    # Features - ukloni target
    exclude_cols = ["cijena"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()

    # Kategoricke varijable - label encoding
    cat_cols = ["zupanija", "grad_opcina", "naselje"]
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype("category").cat.codes

    return X, y


def objective_lgb(trial, X, y):
    """Optuna objective za LightGBM."""
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "n_estimators": 500,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "early_stopping_rounds": 30,
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        y_pred = model.predict(X_val)
        scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))

    return np.mean(scores)


def objective_catboost(trial, X, y):
    """Optuna objective za CatBoost."""
    params = {
        "iterations": 500,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "verbose": False,
        "early_stopping_rounds": 30,
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = cb.CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))

        y_pred = model.predict(X_val)
        scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))

    return np.mean(scores)


def optimize_hyperparameters(X, y, model_type: str = "lgb", n_trials: int = 30):
    """Optimizira hyperparametre koristeci Optuna."""
    print(f"  Optimizacija {model_type.upper()} ({n_trials} triala)...")

    if model_type == "lgb":
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective_lgb(trial, X, y), n_trials=n_trials, show_progress_bar=True)
    else:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective_catboost(trial, X, y), n_trials=n_trials, show_progress_bar=True)

    print(f"  Najbolji CV RMSE: {study.best_value:,.0f} EUR")
    return study.best_params


def train_final_model(X, y, best_params: dict, model_type: str = "lgb"):
    """Trenira finalni model s najboljim parametrima."""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_type == "lgb":
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "n_estimators": 1000,
            "early_stopping_rounds": 50,
            **best_params
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    else:
        params = {
            "iterations": 1000,
            "verbose": False,
            "early_stopping_rounds": 50,
            **best_params
        }
        model = cb.CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_test, y_test))

    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluira model."""
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

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

    # MAPE - izbjegni dijeljenje s nulom
    mape_train = np.mean(np.abs((y_train - y_pred_train) / y_train.clip(lower=1))) * 100
    mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test.clip(lower=1))) * 100
    metrics["train"]["mape"] = mape_train
    metrics["test"]["mape"] = mape_test

    return metrics


def print_metrics(metrics: dict, dataset_name: str):
    """Ispisuje metrike modela."""
    print(f"\n{'='*50}")
    print(f"REZULTATI - {dataset_name.upper()}")
    print("="*50)

    print("\nTrain set:")
    print(f"  RMSE: {metrics['train']['rmse']:,.0f} EUR")
    print(f"  MAE:  {metrics['train']['mae']:,.0f} EUR")
    print(f"  R2:   {metrics['train']['r2']:.4f}")
    print(f"  MAPE: {metrics['train']['mape']:.1f}%")

    print("\nTest set:")
    print(f"  RMSE: {metrics['test']['rmse']:,.0f} EUR")
    print(f"  MAE:  {metrics['test']['mae']:,.0f} EUR")
    print(f"  R2:   {metrics['test']['r2']:.4f}")
    print(f"  MAPE: {metrics['test']['mape']:.1f}%")


def print_feature_importance(model, feature_names: list, top_n: int = 15, model_type: str = "lgb"):
    """Ispisuje najvaznije featuree."""
    if model_type == "lgb":
        importance = model.feature_importances_
    else:
        importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)

    print(f"\nTop {top_n} najvaznijih featurea:")
    print("-" * 40)
    for _, row in importance_df.head(top_n).iterrows():
        # Ukloni problematicne znakove iz imena
        feature_name = row['feature'].encode('ascii', 'replace').decode('ascii')
        print(f"  {feature_name}: {row['importance']:.0f}")


def save_model(model, feature_names: list, metrics: dict, best_params: dict,
               dataset_name: str, model_type: str):
    """Sprema model i metapodatke."""
    model_path = MODELS_DIR / f"{dataset_name}_model_optimized.joblib"

    model_data = {
        "model": model,
        "model_type": model_type,
        "feature_names": feature_names,
        "metrics": metrics,
        "best_params": best_params,
    }

    joblib.dump(model_data, model_path)
    print(f"\nModel spremljen: {model_path}")


def train_dataset(dataset_name: str, dataset_type: str, n_trials: int = 30):
    """Trenira i optimizira model za jedan dataset."""
    print(f"\n{'='*60}")
    print(f" {dataset_name.upper()}")
    print("="*60)

    # 1. Ucitaj podatke
    print("\n1. Ucitavanje podataka...")
    df = load_data(dataset_type)
    print(f"   Ucitano {len(df)} zapisa")

    # 2. Ukloni outliere
    print("\n2. Uklanjanje outliera...")
    df = remove_outliers(df, "cijena", lower_pct=1, upper_pct=99)

    # 3. Feature engineering
    print("\n3. Feature engineering...")
    df = add_features(df)
    new_features = [c for c in ['starost', 'godine_od_renovacije', 'povrsina_log', 'ukupno_kupaonica'] if c in df.columns]
    print(f"   Dodano {len(new_features)} novih featurea")

    # 4. Pripremi features
    print("\n4. Priprema featurea...")
    X, y = prepare_features(df)
    print(f"   {len(X.columns)} featurea, {len(X)} uzoraka")

    # 5. Optimizacija - samo LightGBM za brzinu
    print("\n5. Hyperparameter optimizacija...")
    best_params_lgb = optimize_hyperparameters(X, y, model_type="lgb", n_trials=n_trials)

    # 6. Treniraj finalni model
    print("\n6. Treniranje finalnog modela...")
    model, X_train, X_test, y_train, y_test = train_final_model(X, y, best_params_lgb, "lgb")
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

    # 7. Ispisi rezultate
    print_metrics(metrics, dataset_name)
    print_feature_importance(model, X.columns.tolist(), model_type="lgb")

    # 8. Spremi model
    save_model(model, X.columns.tolist(), metrics, best_params_lgb, dataset_type, "lgb")

    return metrics


def main():
    print("="*60)
    print(" OPTIMIZIRANO TRENIRANJE MODELA")
    print(" Croatian Property Price Estimator")
    print("="*60)

    # Treniraj oba dataseta
    metrics_houses = train_dataset("Kuce", "houses", n_trials=30)
    metrics_apartments = train_dataset("Stanovi", "apartments", n_trials=30)

    # Sazetak
    print("\n" + "="*60)
    print(" FINALNI SAZETAK")
    print("="*60)

    print(f"\nKuce:")
    print(f"  Test R2:   {metrics_houses['test']['r2']:.4f} ({metrics_houses['test']['r2']*100:.1f}%)")
    print(f"  Test MAE:  {metrics_houses['test']['mae']:,.0f} EUR")
    print(f"  Test MAPE: {metrics_houses['test']['mape']:.1f}%")

    print(f"\nStanovi:")
    print(f"  Test R2:   {metrics_apartments['test']['r2']:.4f} ({metrics_apartments['test']['r2']*100:.1f}%)")
    print(f"  Test MAE:  {metrics_apartments['test']['mae']:,.0f} EUR")
    print(f"  Test MAPE: {metrics_apartments['test']['mape']:.1f}%")

    print("\n" + "="*60)
    print(" GOTOVO!")
    print("="*60)


if __name__ == "__main__":
    main()
