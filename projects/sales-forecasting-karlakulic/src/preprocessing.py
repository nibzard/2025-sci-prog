"""
Preprocessing and feature engineering functions for sales forecasting.
All functions ensure no data leakage with proper time-aware operations.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from . import config
from .utils import print_section_header


def split_time_series(df: pd.DataFrame,
                     date_col: str = 'date',
                     test_ratio: float = None,
                     verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Split time series data chronologically into train and test sets.

    CRITICAL: Uses chronological split (NO random shuffling) to prevent data leakage.

    Parameters:
    -----------
    df : pd.DataFrame
        Time series dataframe (must have date column)
    date_col : str
        Name of the date column
    test_ratio : float, optional
        Proportion of data for testing (default: from config.TEST_RATIO)
    verbose : bool
        Whether to print split information

    Returns:
    --------
    tuple
        (train_df, test_df, split_date)
    """
    if test_ratio is None:
        test_ratio = config.TEST_RATIO

    # Sort by date to ensure chronological order
    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)

    # Calculate split index
    n_total = len(df_sorted)
    n_train = int(n_total * (1 - test_ratio))

    # Split data
    train_df = df_sorted.iloc[:n_train].copy()
    test_df = df_sorted.iloc[n_train:].copy()

    # Get split date
    split_date = train_df[date_col].max()

    if verbose:
        print_section_header("Time Series Split")
        print(f"Total rows: {n_total}")
        print(f"Train rows: {n_train} ({100*(1-test_ratio):.1f}%)")
        print(f"Test rows: {len(test_df)} ({100*test_ratio:.1f}%)")
        print(f"\nTrain date range: {train_df[date_col].min()} to {train_df[date_col].max()}")
        print(f"Test date range: {test_df[date_col].min()} to {test_df[date_col].max()}")
        print(f"\nSplit date: {split_date}")
        print("\n✓ Chronological split - no data leakage")

    return train_df, test_df, split_date


def create_lag_features(df: pd.DataFrame,
                       target_col: str = 'sales',
                       lags: List[int] = None) -> pd.DataFrame:
    """
    Create lag features from the target variable.

    CRITICAL: Uses shift() which properly handles time series ordering.
    No future information leaks into past.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with target column
    target_col : str
        Column to create lags from
    lags : list of int
        List of lag periods (default: from config.LAG_FEATURES)

    Returns:
    --------
    pd.DataFrame
        DataFrame with lag features added
    """
    if lags is None:
        lags = config.LAG_FEATURES

    df_lagged = df.copy()

    for lag in lags:
        # shift(lag) moves values DOWN by 'lag' rows, so row t gets value from row t-lag
        df_lagged[f'{target_col}_lag_{lag}'] = df_lagged[target_col].shift(lag)

    return df_lagged


def create_rolling_features(df: pd.DataFrame,
                           target_col: str = 'sales',
                           windows: List[int] = None) -> pd.DataFrame:
    """
    Create rolling window statistics (mean and std).

    CRITICAL: Uses .rolling() with proper window and min_periods to prevent leakage.
    Each rolling statistic only uses past values.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with target column
    target_col : str
        Column to compute rolling stats for
    windows : list of int
        Window sizes (default: from config.ROLLING_WINDOWS)

    Returns:
    --------
    pd.DataFrame
        DataFrame with rolling features added
    """
    if windows is None:
        windows = config.ROLLING_WINDOWS

    df_rolling = df.copy()

    for window in windows:
        # Rolling mean - uses past 'window' values (including current)
        # Set min_periods=window to ensure we have full window
        df_rolling[f'rolling_mean_{window}'] = (
            df_rolling[target_col]
            .shift(1)  # Shift by 1 to exclude current day
            .rolling(window=window, min_periods=window)
            .mean()
        )

        # Rolling standard deviation
        df_rolling[f'rolling_std_{window}'] = (
            df_rolling[target_col]
            .shift(1)
            .rolling(window=window, min_periods=window)
            .std()
        )

    return df_rolling


def create_calendar_features(df: pd.DataFrame,
                            date_col: str = 'date') -> pd.DataFrame:
    """
    Create calendar-based features from date column.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with date column
    date_col : str
        Name of the date column

    Returns:
    --------
    pd.DataFrame
        DataFrame with calendar features added
    """
    df_cal = df.copy()

    # Extract date components
    df_cal['day_of_week'] = df_cal[date_col].dt.dayofweek  # 0 = Monday, 6 = Sunday
    df_cal['day_of_month'] = df_cal[date_col].dt.day
    df_cal['month'] = df_cal[date_col].dt.month
    df_cal['year'] = df_cal[date_col].dt.year
    df_cal['quarter'] = df_cal[date_col].dt.quarter
    df_cal['is_weekend'] = (df_cal[date_col].dt.dayofweek >= 5).astype(int)
    df_cal['is_month_start'] = df_cal[date_col].dt.is_month_start.astype(int)
    df_cal['is_month_end'] = df_cal[date_col].dt.is_month_end.astype(int)

    return df_cal


def create_promotion_features(df: pd.DataFrame,
                             promo_col: str = 'onpromotion',
                             windows: List[int] = None) -> pd.DataFrame:
    """
    Create promotion-related features.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with promotion column
    promo_col : str
        Name of promotion column
    windows : list of int, optional
        Windows for rolling promotion stats

    Returns:
    --------
    pd.DataFrame
        DataFrame with promotion features added
    """
    df_promo = df.copy()

    # Binary promotion indicator
    df_promo['has_promotion'] = (df_promo[promo_col] > 0).astype(int)

    # Rolling promotion count (how many items on promotion recently)
    if windows is None:
        windows = [7]  # Default to 7-day window

    for window in windows:
        df_promo[f'rolling_promo_{window}'] = (
            df_promo[promo_col]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )

    return df_promo


def build_features(df: pd.DataFrame,
                  target_col: str = 'sales',
                  date_col: str = 'date',
                  lags: List[int] = None,
                  roll_windows: List[int] = None,
                  include_promotions: bool = True,
                  verbose: bool = True) -> pd.DataFrame:
    """
    Build all features for the time series forecasting model.

    CRITICAL: All features use only past values - no data leakage.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with time series data
    target_col : str
        Target variable column name
    date_col : str
        Date column name
    lags : list of int, optional
        Lag periods to create
    roll_windows : list of int, optional
        Rolling window sizes
    include_promotions : bool
        Whether to include promotion features
    verbose : bool
        Print progress information

    Returns:
    --------
    pd.DataFrame
        DataFrame with all engineered features
    """
    if verbose:
        print_section_header("Feature Engineering")

    df_features = df.copy()

    # 1. Lag features
    if verbose:
        print("Creating lag features...")
    df_features = create_lag_features(df_features, target_col=target_col, lags=lags)

    # 2. Rolling features
    if verbose:
        print("Creating rolling window features...")
    df_features = create_rolling_features(df_features, target_col=target_col, windows=roll_windows)

    # 3. Calendar features
    if verbose:
        print("Creating calendar features...")
    df_features = create_calendar_features(df_features, date_col=date_col)

    # 4. Promotion features
    if include_promotions and 'onpromotion' in df_features.columns:
        if verbose:
            print("Creating promotion features...")
        df_features = create_promotion_features(df_features)

    if verbose:
        print(f"\n✓ Feature engineering complete")
        print(f"Original columns: {len(df.columns)}")
        print(f"Total columns after engineering: {len(df_features.columns)}")
        print(f"New features created: {len(df_features.columns) - len(df.columns)}")

    return df_features


def handle_missing_from_features(df: pd.DataFrame,
                                 strategy: str = 'drop',
                                 verbose: bool = True) -> pd.DataFrame:
    """
    Handle missing values created by lag and rolling window features.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features (potentially containing NaNs)
    strategy : str
        How to handle missing values ('drop' or 'fill')
    verbose : bool
        Print information about missing values

    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values handled
    """
    if verbose:
        print_section_header("Handling Missing Values")
        print(f"Rows before: {len(df)}")
        print(f"\nMissing values per column:")
        missing = df.isnull().sum()
        print(missing[missing > 0])

    if strategy == 'drop':
        df_clean = df.dropna().reset_index(drop=True)
        if verbose:
            print(f"\n✓ Dropped rows with missing values")
            print(f"Rows after: {len(df_clean)}")
            print(f"Rows dropped: {len(df) - len(df_clean)}")
    elif strategy == 'fill':
        df_clean = df.fillna(method='ffill').fillna(0)
        if verbose:
            print(f"\n✓ Forward filled and then zero-filled missing values")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return df_clean


def prepare_model_data(df: pd.DataFrame,
                      feature_cols: List[str],
                      target_col: str = 'sales',
                      date_col: str = 'date',
                      return_dates: bool = False) -> Tuple:
    """
    Prepare final X and y arrays for model training/testing.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with all features
    feature_cols : list of str
        List of feature column names to use
    target_col : str
        Target variable column name
    date_col : str
        Date column name (excluded from X)
    return_dates : bool
        Whether to also return dates array

    Returns:
    --------
    tuple
        (X, y) or (X, y, dates) if return_dates=True
    """
    # Ensure all feature columns exist
    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Extract features and target
    X = df[feature_cols].values
    y = df[target_col].values

    if return_dates:
        dates = df[date_col].values
        return X, y, dates
    else:
        return X, y


def get_feature_names(lags: List[int] = None,
                     roll_windows: List[int] = None,
                     include_promotions: bool = True) -> List[str]:
    """
    Get list of feature names that will be created by build_features().
    Useful for model training.

    Parameters:
    -----------
    lags : list of int, optional
        Lag periods
    roll_windows : list of int, optional
        Rolling window sizes
    include_promotions : bool
        Whether promotion features are included

    Returns:
    --------
    list of str
        List of feature column names
    """
    if lags is None:
        lags = config.LAG_FEATURES
    if roll_windows is None:
        roll_windows = config.ROLLING_WINDOWS

    features = []

    # Lag features
    for lag in lags:
        features.append(f'sales_lag_{lag}')

    # Rolling features
    for window in roll_windows:
        features.append(f'rolling_mean_{window}')
        features.append(f'rolling_std_{window}')

    # Calendar features
    calendar_features = [
        'day_of_week',
        'day_of_month',
        'month',
        'year',
        'quarter',
        'is_weekend',
        'is_month_start',
        'is_month_end'
    ]
    features.extend(calendar_features)

    # Promotion features
    if include_promotions:
        features.append('onpromotion')
        features.append('has_promotion')
        features.append('rolling_promo_7')

    return features


def create_train_test_features(train_df: pd.DataFrame,
                               test_df: pd.DataFrame,
                               target_col: str = 'sales',
                               date_col: str = 'date',
                               lags: List[int] = None,
                               roll_windows: List[int] = None,
                               verbose: bool = True) -> Tuple:
    """
    Create features for both train and test sets in a leakage-free manner.

    IMPORTANT: This function concatenates train and test before feature engineering,
    then splits them back. This ensures rolling/lag features in test set can use
    information from the end of the train set (which is correct - test comes after train).

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    target_col : str
        Target column
    date_col : str
        Date column
    lags : list of int, optional
        Lag periods
    roll_windows : list of int, optional
        Rolling window sizes
    verbose : bool
        Print information

    Returns:
    --------
    tuple
        (train_features, test_features, feature_names)
    """
    if verbose:
        print_section_header("Creating Train/Test Features")
        print(f"Train size: {len(train_df)}")
        print(f"Test size: {len(test_df)}")

    # Mark train/test with a flag
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['_is_train'] = 1
    test_df['_is_train'] = 0

    # Concatenate
    combined_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

    # Build features on combined data
    combined_features = build_features(
        combined_df,
        target_col=target_col,
        date_col=date_col,
        lags=lags,
        roll_windows=roll_windows,
        verbose=verbose
    )

    # Handle missing values
    combined_clean = handle_missing_from_features(combined_features, verbose=verbose)

    # Split back into train and test
    train_features = combined_clean[combined_clean['_is_train'] == 1].copy()
    test_features = combined_clean[combined_clean['_is_train'] == 0].copy()

    # Drop the flag column
    train_features = train_features.drop(columns=['_is_train'])
    test_features = test_features.drop(columns=['_is_train'])

    # Get feature names
    feature_names = get_feature_names(lags=lags, roll_windows=roll_windows)

    if verbose:
        print(f"\n✓ Train features shape: {train_features.shape}")
        print(f"✓ Test features shape: {test_features.shape}")
        print(f"✓ Number of features: {len(feature_names)}")

    return train_features, test_features, feature_names
