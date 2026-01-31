"""
V3 Model - Poboljsana verzija s:
- log(cijena) kao target
- Target encoding za lokacije (K-fold bez leaka)
- Bez outlier removal
- Ispravljeno rukovanje renovacijom
- Interaction featurei za kuce
- Odvojen tuning za kuce/stanove
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
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


def remove_data_errors(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    """
    Uklanja OCITE GRESKE u podacima (ne legitimne outliere).

    Npr. nekretnina od 1 EUR je greska u unosu, ne pravi podatak.
    Ovo NIJE isto kao outlier removal koji uklanja skupe nekretnine.
    """
    df = df.copy()
    initial_count = len(df)

    # Minimalne realne cijene (ispod ovoga su greske)
    if dataset_type == "houses":
        min_price = 15000  # Kuca ispod 15k EUR je greska
        max_price = 10000000  # Iznad 10M EUR je vjerojatno greska
    else:
        min_price = 10000  # Stan ispod 10k EUR je greska
        max_price = 5000000  # Iznad 5M EUR je vjerojatno greska

    # Minimalna povrsina
    min_area = 10  # Ispod 10m2 je vjerojatno greska

    # Filtriranje
    mask = (
        (df["cijena"] >= min_price) &
        (df["cijena"] <= max_price) &
        (df["stambena_povrsina"] >= min_area)
    )

    df = df[mask].copy()
    removed = initial_count - len(df)

    if removed > 0:
        print(f"   Uklonjeno {removed} ocitih gresaka u podacima ({removed/initial_count*100:.1f}%)")

    return df


def fix_renovation_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ispravlja stupac godina_renovacije.
    NE postavljamo godinu izgradnje ako nema renovacije - to je laziranje podataka.
    Umjesto toga koristimo NaN + flag.
    """
    df = df.copy()

    # Ucitaj originalne podatke da vidimo prave vrijednosti renovacije
    # (clean_data.py je postavio godina_izgradnje gdje nije bilo renovacije)
    # Moramo to ispraviti - ako je godina_renovacije == godina_izgradnje,
    # vjerojatno nije bilo renovacije

    # Flag: ima li renovaciju (ako je godina_renovacije razlicita od godine_izgradnje)
    if "godina_izgradnje" in df.columns and "godina_renovacije" in df.columns:
        # Ako su jednake, vjerojatno nema prave renovacije
        df["ima_renovaciju"] = (
            df["godina_renovacije"].notna() &
            (df["godina_renovacije"] != df["godina_izgradnje"])
        ).astype(int)

        # Postavi NaN gdje nema prave renovacije
        mask_no_renovation = df["godina_renovacije"] == df["godina_izgradnje"]
        df.loc[mask_no_renovation, "godina_renovacije"] = np.nan
    else:
        df["ima_renovaciju"] = df["godina_renovacije"].notna().astype(int)

    return df


def add_features_v3(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    """Dodaje feature engineering za V3."""
    df = df.copy()

    # Starost nekretnine
    if "godina_izgradnje" in df.columns:
        df["starost"] = CURRENT_YEAR - df["godina_izgradnje"]
        df["starost"] = df["starost"].clip(lower=0)

    # Godine od renovacije (samo ako ima renovaciju)
    if "godina_renovacije" in df.columns:
        df["godine_od_renovacije"] = np.where(
            df["godina_renovacije"].notna(),
            CURRENT_YEAR - df["godina_renovacije"],
            np.nan
        )
        df["godine_od_renovacije"] = df["godine_od_renovacije"].clip(lower=0)

    # Log povrsine
    if "stambena_povrsina" in df.columns:
        df["povrsina_log"] = np.log1p(df["stambena_povrsina"])

    # Ukupan broj kupaonica
    bathroom_cols = ["wc_broj", "kupaonica_s_wc_broj"]
    existing_cols = [c for c in bathroom_cols if c in df.columns]
    if existing_cols:
        df["ukupno_kupaonica"] = df[existing_cols].sum(axis=1, skipna=True)

    # === INTERACTION FEATURES ZA KUCE ===
    if dataset_type == "houses":
        # Pogled na more je vrjedniji u nekim zupanijama
        if "pogled_more" in df.columns:
            df["pogled_more_primorje"] = df["pogled_more"] * df["zupanija"].isin([
                "Istarska", "Primorsko-goranska", "Zadarska",
                "Splitsko-dalmatinska", "Dubrovacko-neretvanska"
            ]).astype(int)

        # Log povrsine okucnice
        if "povrsina_okucnice" in df.columns:
            df["okucnica_log"] = np.log1p(df["povrsina_okucnice"].fillna(0))

        # Ima li bazen (vrijednije uz more)
        if "objekt_bazen" in df.columns and "pogled_more" in df.columns:
            df["bazen_uz_more"] = df["objekt_bazen"] * df["pogled_more"]

    # === INTERACTION FEATURES ZA STANOVE ===
    if dataset_type == "apartments":
        # Lift je bitniji na visim katovima
        if "podatak_lift" in df.columns and "kat" in df.columns:
            df["lift_visoki_kat"] = df["podatak_lift"] * (df["kat"] > 3).astype(int)

        # Novogradnja
        if "podatak_novogradnja" in df.columns and "stambena_povrsina" in df.columns:
            df["novogradnja_povrsina"] = df["podatak_novogradnja"] * df["povrsina_log"]

    return df


def target_encode_kfold(df: pd.DataFrame, column: str, target: pd.Series,
                         n_splits: int = 5, smoothing: float = 10.0) -> pd.Series:
    """
    Target encoding s K-fold cross-validation da se izbjegne data leakage.

    smoothing: koliko tezine dati globalnom prosjeku vs. lokalnom
    """
    encoded = pd.Series(index=df.index, dtype=float)
    global_mean = target.mean()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(df):
        # Izracunaj statistike samo na train foldu
        train_target = target.iloc[train_idx]
        train_col = df[column].iloc[train_idx]

        # Grupiranje po kategoriji
        stats = train_target.groupby(train_col).agg(['mean', 'count'])
        stats.columns = ['mean', 'count']

        # Smoothing formula: (count * mean + smoothing * global_mean) / (count + smoothing)
        stats['smoothed'] = (
            (stats['count'] * stats['mean'] + smoothing * global_mean) /
            (stats['count'] + smoothing)
        )

        # Primijeni na validation fold
        val_col = df[column].iloc[val_idx]
        encoded.iloc[val_idx] = val_col.map(stats['smoothed']).fillna(global_mean)

    return encoded


def prepare_features_v3(df: pd.DataFrame, y: pd.Series,
                        dataset_type: str, is_train: bool = True,
                        encodings: dict = None) -> tuple:
    """
    Priprema features s target encodingom za lokacije.

    Args:
        df: DataFrame s featureima
        y: target varijabla (log cijena)
        dataset_type: "houses" ili "apartments"
        is_train: True za trening, False za test/inference
        encodings: dict s encodinzima za test set (None za train)

    Returns:
        X: pripremljeni features
        encodings: dict s encodinzima (za kasnije koristenje na test setu)
    """
    X = df.copy()

    # Ukloni target ako postoji
    if "cijena" in X.columns:
        X = X.drop(columns=["cijena"])

    # === TARGET ENCODING ZA LOKACIJE ===
    location_cols = ["zupanija", "grad_opcina", "naselje"]

    if is_train:
        encodings = {}
        for col in location_cols:
            if col in X.columns:
                # K-fold target encoding
                X[f"{col}_encoded"] = target_encode_kfold(X, col, y, n_splits=5, smoothing=10)

                # Spremi mapping za test set (prosjek po kategoriji)
                col_mapping = y.groupby(X[col]).mean().to_dict()
                global_mean = y.mean()
                encodings[col] = {"mapping": col_mapping, "global_mean": global_mean}

                # Ukloni originalni stupac
                X = X.drop(columns=[col])
    else:
        # Za test set, koristi spremljene encodinge
        for col in location_cols:
            if col in X.columns and col in encodings:
                mapping = encodings[col]["mapping"]
                global_mean = encodings[col]["global_mean"]
                X[f"{col}_encoded"] = X[col].map(mapping).fillna(global_mean)
                X = X.drop(columns=[col])

    # Ukloni preostale string stupce
    string_cols = X.select_dtypes(include=['object']).columns
    X = X.drop(columns=string_cols, errors='ignore')

    return X, encodings


def objective_lgb_v3(trial, X, y, dataset_type: str):
    """Optuna objective za LightGBM V3."""

    # Razliciti rasponi za kuce vs stanove
    if dataset_type == "houses":
        # Kuce trebaju vise fleksibilnosti (manje regularizacije)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 150),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "early_stopping_rounds": 50,
        }
    else:
        # Stanovi - standardni rasponi
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "early_stopping_rounds": 50,
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


def optimize_hyperparameters_v3(X, y, dataset_type: str, n_trials: int = 50):
    """Optimizira hyperparametre za V3."""
    print(f"  Optimizacija za {dataset_type} ({n_trials} triala)...")

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective_lgb_v3(trial, X, y, dataset_type),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print(f"  Najbolji CV RMSE (log-space): {study.best_value:.4f}")
    return study.best_params


def train_final_model_v3(X_train, y_train, X_test, y_test,
                         best_params: dict, dataset_type: str):
    """Trenira finalni model s najboljim parametrima."""

    if dataset_type == "houses":
        base_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "n_estimators": 2000,  # Vise stabala za kuce
            "early_stopping_rounds": 100,
        }
    else:
        base_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "n_estimators": 1500,
            "early_stopping_rounds": 75,
        }

    params = {**base_params, **best_params}

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    return model


def evaluate_model_v3(model, X_train, X_test, y_train, y_test,
                      y_train_original, y_test_original):
    """
    Evaluira model.

    y_train, y_test: log-transformirane vrijednosti (za RMSE u log-space)
    y_train_original, y_test_original: originalne cijene (za MAE, MAPE u EUR)
    """
    # Predikcije u log-space
    y_pred_train_log = model.predict(X_train)
    y_pred_test_log = model.predict(X_test)

    # Predikcije u originalnom prostoru (EUR)
    y_pred_train = np.expm1(y_pred_train_log)
    y_pred_test = np.expm1(y_pred_test_log)

    # Osiguraj pozitivne predikcije
    y_pred_train = np.maximum(y_pred_train, 0)
    y_pred_test = np.maximum(y_pred_test, 0)

    metrics = {
        "train": {
            "rmse_log": np.sqrt(mean_squared_error(y_train, y_pred_train_log)),
            "rmse_eur": np.sqrt(mean_squared_error(y_train_original, y_pred_train)),
            "mae_eur": mean_absolute_error(y_train_original, y_pred_train),
            "r2": r2_score(y_train_original, y_pred_train),
        },
        "test": {
            "rmse_log": np.sqrt(mean_squared_error(y_test, y_pred_test_log)),
            "rmse_eur": np.sqrt(mean_squared_error(y_test_original, y_pred_test)),
            "mae_eur": mean_absolute_error(y_test_original, y_pred_test),
            "r2": r2_score(y_test_original, y_pred_test),
        }
    }

    # MAPE - izbjegni dijeljenje s nulom
    def safe_mape(y_true, y_pred):
        mask = y_true > 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    metrics["train"]["mape"] = safe_mape(y_train_original.values, y_pred_train)
    metrics["test"]["mape"] = safe_mape(y_test_original.values, y_pred_test)

    # MAPE po cjenovnim bucketima
    def mape_by_bucket(y_true, y_pred, buckets):
        results = {}
        for name, (low, high) in buckets.items():
            mask = (y_true >= low) & (y_true < high)
            if mask.sum() > 0:
                bucket_mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                results[name] = {"mape": bucket_mape, "count": mask.sum()}
        return results

    buckets = {
        "0-200k": (0, 200000),
        "200k-500k": (200000, 500000),
        "500k-1M": (500000, 1000000),
        "1M+": (1000000, float('inf'))
    }

    metrics["test"]["mape_buckets"] = mape_by_bucket(
        y_test_original.values, y_pred_test, buckets
    )

    return metrics


def print_metrics_v3(metrics: dict, dataset_name: str):
    """Ispisuje metrike modela V3."""
    print(f"\n{'='*60}")
    print(f"REZULTATI V3 - {dataset_name.upper()}")
    print("="*60)

    print("\nTrain set:")
    print(f"  RMSE (log-space): {metrics['train']['rmse_log']:.4f}")
    print(f"  RMSE (EUR):       {metrics['train']['rmse_eur']:,.0f} EUR")
    print(f"  MAE (EUR):        {metrics['train']['mae_eur']:,.0f} EUR")
    print(f"  R2:               {metrics['train']['r2']:.4f}")
    print(f"  MAPE:             {metrics['train']['mape']:.1f}%")

    print("\nTest set:")
    print(f"  RMSE (log-space): {metrics['test']['rmse_log']:.4f}")
    print(f"  RMSE (EUR):       {metrics['test']['rmse_eur']:,.0f} EUR")
    print(f"  MAE (EUR):        {metrics['test']['mae_eur']:,.0f} EUR")
    print(f"  R2:               {metrics['test']['r2']:.4f}")
    print(f"  MAPE:             {metrics['test']['mape']:.1f}%")

    print("\nMAPE po cjenovnim bucketima (test):")
    for bucket, data in metrics['test']['mape_buckets'].items():
        print(f"  {bucket}: {data['mape']:.1f}% (n={data['count']})")


def print_feature_importance(model, feature_names: list, top_n: int = 15):
    """Ispisuje najvaznije featuree."""
    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)

    print(f"\nTop {top_n} najvaznijih featurea:")
    print("-" * 50)
    for _, row in importance_df.head(top_n).iterrows():
        feature_name = row['feature'].encode('ascii', 'replace').decode('ascii')
        print(f"  {feature_name}: {row['importance']:.0f}")


def save_model_v3(model, feature_names: list, metrics: dict, best_params: dict,
                  encodings: dict, dataset_name: str):
    """Sprema model V3 i sve potrebne metapodatke."""
    model_path = MODELS_DIR / f"{dataset_name}_model_v3.joblib"

    model_data = {
        "model": model,
        "model_type": "lgb",
        "version": "v3",
        "feature_names": feature_names,
        "metrics": metrics,
        "best_params": best_params,
        "encodings": encodings,  # Potrebno za inference
        "log_transform": True,  # Flag da se zna da treba expm1 na predikcije
    }

    joblib.dump(model_data, model_path)
    print(f"\nModel V3 spremljen: {model_path}")


def train_dataset_v3(dataset_name: str, dataset_type: str, n_trials: int = 50):
    """Trenira V3 model za jedan dataset."""
    print(f"\n{'='*70}")
    print(f" V3 - {dataset_name.upper()}")
    print("="*70)

    # 1. Ucitaj podatke
    print("\n1. Ucitavanje podataka...")
    df = load_data(dataset_type)
    print(f"   Ucitano {len(df)} zapisa")

    # 2. Ukloni ocite greske (NE legitimne outliere!)
    print("\n2. Uklanjanje ocitih gresaka u podacima...")
    df = remove_data_errors(df, dataset_type)
    print(f"   Preostalo {len(df)} zapisa")

    # 3. Ispravi renovaciju
    print("\n3. Ispravljanje stupca godina_renovacije...")
    df = fix_renovation_column(df)
    renovated_count = df["ima_renovaciju"].sum()
    print(f"   Nekretnina s renovacijom: {renovated_count} ({renovated_count/len(df)*100:.1f}%)")

    # 4. Feature engineering
    print("\n4. Feature engineering...")
    df = add_features_v3(df, dataset_type)

    # 5. Log transformacija cijena
    print("\n5. Log transformacija cijena...")
    y_original = df["cijena"].copy()
    y_log = np.log1p(df["cijena"])
    print(f"   Cijena range: {y_original.min():,.0f} - {y_original.max():,.0f} EUR")
    print(f"   Log cijena range: {y_log.min():.2f} - {y_log.max():.2f}")

    # 6. Train/test split (PRIJE target encodinga!)
    print("\n6. Train/test split...")
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=42
    )

    df_train = df.loc[train_idx].copy()
    df_test = df.loc[test_idx].copy()
    y_train_log = y_log.loc[train_idx]
    y_test_log = y_log.loc[test_idx]
    y_train_original = y_original.loc[train_idx]
    y_test_original = y_original.loc[test_idx]

    print(f"   Train: {len(df_train)} uzoraka")
    print(f"   Test: {len(df_test)} uzoraka")

    # 7. Priprema featurea s target encodingom
    print("\n7. Target encoding lokacija...")
    X_train, encodings = prepare_features_v3(
        df_train, y_train_log, dataset_type, is_train=True
    )
    X_test, _ = prepare_features_v3(
        df_test, y_test_log, dataset_type, is_train=False, encodings=encodings
    )
    print(f"   Broj featurea: {len(X_train.columns)}")

    # 8. Hyperparameter optimizacija
    print("\n8. Hyperparameter optimizacija...")
    best_params = optimize_hyperparameters_v3(X_train, y_train_log, dataset_type, n_trials)

    # 9. Treniraj finalni model
    print("\n9. Treniranje finalnog modela...")
    model = train_final_model_v3(
        X_train, y_train_log, X_test, y_test_log,
        best_params, dataset_type
    )

    # 10. Evaluacija
    print("\n10. Evaluacija...")
    metrics = evaluate_model_v3(
        model, X_train, X_test,
        y_train_log, y_test_log,
        y_train_original, y_test_original
    )

    # 11. Ispisi rezultate
    print_metrics_v3(metrics, dataset_name)
    print_feature_importance(model, X_train.columns.tolist())

    # 12. Spremi model
    save_model_v3(
        model, X_train.columns.tolist(), metrics,
        best_params, encodings, dataset_type
    )

    return metrics


def main():
    print("="*70)
    print(" V3 MODEL - CROATIAN PROPERTY PRICE ESTIMATOR")
    print(" Poboljsanja: log(cijena), target encoding, interaction features")
    print("="*70)

    # Treniraj oba dataseta
    metrics_houses = train_dataset_v3("Kuce", "houses", n_trials=50)
    metrics_apartments = train_dataset_v3("Stanovi", "apartments", n_trials=50)

    # Finalni sazetak
    print("\n" + "="*70)
    print(" FINALNI SAZETAK - V3")
    print("="*70)

    print(f"\nKuce:")
    print(f"  Test R2:   {metrics_houses['test']['r2']:.4f} ({metrics_houses['test']['r2']*100:.1f}%)")
    print(f"  Test MAE:  {metrics_houses['test']['mae_eur']:,.0f} EUR")
    print(f"  Test MAPE: {metrics_houses['test']['mape']:.1f}%")

    print(f"\nStanovi:")
    print(f"  Test R2:   {metrics_apartments['test']['r2']:.4f} ({metrics_apartments['test']['r2']*100:.1f}%)")
    print(f"  Test MAE:  {metrics_apartments['test']['mae_eur']:,.0f} EUR")
    print(f"  Test MAPE: {metrics_apartments['test']['mape']:.1f}%")

    print("\n" + "="*70)
    print(" GOTOVO!")
    print("="*70)


if __name__ == "__main__":
    main()
