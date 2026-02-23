"""
Utility functions for the sales forecasting project.
Contains helper functions for file operations, plotting, and formatting.
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Dict, Any
from . import config


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists. Create it if it doesn't.

    Parameters:
    -----------
    path : str or Path
        Directory path to ensure exists

    Returns:
    --------
    Path
        Path object of the ensured directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, filename: str, dpi: int = None, tight_layout: bool = True) -> Path:
    """
    Save a matplotlib figure to the figures directory.

    Parameters:
    -----------
    fig : plt.Figure
        Matplotlib figure object to save
    filename : str
        Name of the file (with or without extension)
    dpi : int, optional
        DPI for saving (default: from config.DPI)
    tight_layout : bool, default=True
        Whether to apply tight_layout before saving

    Returns:
    --------
    Path
        Path where the figure was saved
    """
    if dpi is None:
        dpi = config.DPI

    # Ensure filename has extension
    if not filename.endswith(('.png', '.jpg', '.pdf', '.svg')):
        filename += '.png'

    # Ensure figures directory exists
    ensure_dir(config.FIGURES_PATH)

    # Full path
    filepath = config.FIGURES_PATH / filename

    # Apply tight layout if requested
    if tight_layout:
        fig.tight_layout()

    # Save figure
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to: {filepath}")

    return filepath


def format_metrics_table(results_dict: Dict[str, Dict[str, float]],
                         sort_by: str = 'MAE',
                         ascending: bool = True) -> pd.DataFrame:
    """
    Format model evaluation results into a clean DataFrame.

    Parameters:
    -----------
    results_dict : dict
        Dictionary with structure: {model_name: {'MAE': value, 'RMSE': value, ...}}
    sort_by : str, default='MAE'
        Metric to sort by
    ascending : bool, default=True
        Sort in ascending order (True = lower is better)

    Returns:
    --------
    pd.DataFrame
        Formatted results table

    Example:
    --------
    >>> results = {
    ...     'Naive': {'MAE': 100.5, 'RMSE': 150.2},
    ...     'SARIMA': {'MAE': 85.3, 'RMSE': 120.1}
    ... }
    >>> format_metrics_table(results)
    """
    df = pd.DataFrame(results_dict).T

    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)

    # Round to 2 decimal places
    df = df.round(2)

    return df


def load_and_prepare_data(filepath: Union[str, Path],
                         date_col: str = 'date',
                         parse_dates: bool = True) -> pd.DataFrame:
    """
    Load data from CSV and prepare datetime column.

    Parameters:
    -----------
    filepath : str or Path
        Path to CSV file
    date_col : str, default='date'
        Name of the date column
    parse_dates : bool, default=True
        Whether to parse dates

    Returns:
    --------
    pd.DataFrame
        Loaded data with parsed dates
    """
    if parse_dates:
        df = pd.read_csv(filepath, parse_dates=[date_col])
    else:
        df = pd.read_csv(filepath)

    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")

    return df


def print_section_header(title: str, char: str = "=", width: int = 80) -> None:
    """
    Print a formatted section header for better notebook readability.

    Parameters:
    -----------
    title : str
        Section title
    char : str, default='='
        Character to use for the border
    width : int, default=80
        Total width of the header
    """
    print("\n" + char * width)
    print(f" {title}")
    print(char * width + "\n")


def describe_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print comprehensive description of a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to describe
    name : str, default='DataFrame'
        Name to use in the output
    """
    print_section_header(f"{name} Summary")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn types:")
    print(df.dtypes)
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing'] > 0])
    if missing_df['Missing'].sum() == 0:
        print("No missing values found.")


def get_date_features(date_series: pd.Series) -> pd.DataFrame:
    """
    Extract date features from a datetime series.

    Parameters:
    -----------
    date_series : pd.Series
        Series of datetime objects

    Returns:
    --------
    pd.DataFrame
        DataFrame with extracted features
    """
    features = pd.DataFrame(index=date_series.index)
    features['day_of_week'] = date_series.dt.dayofweek
    features['day_of_month'] = date_series.dt.day
    features['month'] = date_series.dt.month
    features['year'] = date_series.dt.year
    features['is_weekend'] = (date_series.dt.dayofweek >= 5).astype(int)
    features['is_month_start'] = date_series.dt.is_month_start.astype(int)
    features['is_month_end'] = date_series.dt.is_month_end.astype(int)

    return features
