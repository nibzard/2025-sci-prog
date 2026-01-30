"""Loading and basic inspection of the insurance dataset."""

import pandas as pd
from .config import Paths


def load_raw_data(paths: Paths) -> pd.DataFrame:
    """Read the raw CSV file.

    We keep this in a function so later we can swap in a different dataset easily.
    """
    df = pd.read_csv(paths.data_raw)
    return df


def quick_info(df: pd.DataFrame) -> str:
    """Return a small string summary (shape + missing values)."""
    missing = df.isna().sum().to_dict()
    return f"Shape: {df.shape}\nMissing values: {missing}"
