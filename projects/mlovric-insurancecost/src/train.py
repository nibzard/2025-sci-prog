"""Training and model comparison.

This file:
- splits the data (80/20)
- trains multiple models
- evaluates each model
- saves the best one (by RMSE) to models/best_model.joblib
"""

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .models import get_model_zoo
from .utils import ensure_dir


@dataclass
class TrainResult:
    name: str
    mae: float
    mse: float
    rmse: float
    r2: float


def build_preprocess_pipeline(df: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing: one-hot encode categorical features and scale numeric ones.

    Student note: scaling helps Ridge/Lasso behave better.
    """

    feature_cols = [c for c in df.columns if c != "charges"]

    categorical = [c for c in feature_cols if df[c].dtype == "object"]
    numeric = [c for c in feature_cols if df[c].dtype != "object"]

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", numeric_transformer, numeric),
        ]
    )
    return preprocess


def train_and_compare(df: pd.DataFrame, random_state: int) -> tuple[list[TrainResult], object]:
    """Train all models and return results + best fitted pipeline."""

    X = df.drop(columns=["charges"])
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    preprocess = build_preprocess_pipeline(df)
    model_zoo = get_model_zoo(random_state)

    results: list[TrainResult] = []
    best_rmse = float("inf")
    best_pipeline = None

    for name, model in model_zoo.items():
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", model),
            ]
        )

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mae = float(mean_absolute_error(y_test, preds))
        mse = float(mean_squared_error(y_test, preds))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, preds))

        results.append(TrainResult(name, mae, mse, rmse, r2))

        if rmse < best_rmse:
            best_rmse = rmse
            best_pipeline = pipe

    return results, best_pipeline


def save_best_model(best_pipeline: object, models_dir: Path) -> Path:
    ensure_dir(models_dir)
    model_path = models_dir / "best_model.joblib"
    joblib.dump(best_pipeline, model_path)
    return model_path
