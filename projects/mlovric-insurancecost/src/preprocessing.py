"""Data cleaning step.

This dataset is already pretty clean, but we still:
- check missing values
- fix column types if needed
- save a cleaned copy to data/processed/
"""

import pandas as pd
from .config import Paths
from .utils import ensure_dir


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning.

    Student note: Kaggle's insurance dataset usually has no missing values,
    but it's good practice to write the checks anyway.
    """
    df = df.copy()

    # Strip any accidental spaces in column names (happens sometimes)
    df.columns = [c.strip() for c in df.columns]

    # Drop rows with missing values (if any)
    df = df.dropna()

    return df


def save_clean_data(df: pd.DataFrame, paths: Paths) -> None:
    ensure_dir(paths.data_processed.parent)
    df.to_csv(paths.data_processed, index=False)
