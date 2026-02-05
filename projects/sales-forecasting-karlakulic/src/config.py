"""
Central configuration file for the sales forecasting project.
Contains all hyperparameters, paths, and settings for reproducibility.
"""

import os
from pathlib import Path

# Random seed for reproducibility
RANDOM_SEED = 42

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
FIGURES_PATH = PROJECT_ROOT / "figures"
NOTEBOOKS_PATH = PROJECT_ROOT / "notebooks"

# Ensure directories exist
FIGURES_PATH.mkdir(exist_ok=True)
NOTEBOOKS_PATH.mkdir(exist_ok=True)

# Data files
TRAIN_FILE = DATA_PATH / "train.csv"
HOLIDAYS_FILE = DATA_PATH / "holidays_events.csv"
OIL_FILE = DATA_PATH / "oil.csv"
STORES_FILE = DATA_PATH / "stores.csv"
TRANSACTIONS_FILE = DATA_PATH / "transactions.csv"

# Scope selection (to be determined from EDA)
# These will be updated after exploratory analysis
SELECTED_STORE = None  # e.g., 44
SELECTED_FAMILY = None  # e.g., 'GROCERY I'

# Time series split parameters
TEST_RATIO = 0.15  # 15% of data for testing
VALIDATION_SPLITS = 5  # Number of TimeSeriesSplit folds for cross-validation

# Feature engineering parameters
LAG_FEATURES = [1, 7, 14, 30]  # Lag days to create
ROLLING_WINDOWS = [7, 14, 30]  # Rolling window sizes for mean and std

# Forecast horizons
FORECAST_HORIZONS = [14, 30]  # Days to forecast

# Model parameters
# SARIMA
DEFAULT_SEASONAL_PERIOD = 7  # Weekly seasonality

# RandomForest/GradientBoosting hyperparameter grid
RF_PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

GB_PARAM_GRID = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5]
}

# Business use case parameters
SAFETY_MARGIN = 0.10  # 10% safety margin for inventory planning

# Plotting parameters
FIGURE_SIZE = (12, 6)
LARGE_FIGURE_SIZE = (14, 8)
DPI = 100
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# Date column name
DATE_COL = 'date'
TARGET_COL = 'sales'
