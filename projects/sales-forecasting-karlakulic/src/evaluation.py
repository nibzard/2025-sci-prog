"""
Model evaluation functions for sales forecasting.
Comprehensive metrics, visualization, and error analysis tools.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from . import config


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive forecasting metrics.

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns:
    --------
    dict
        Dictionary with MAE, RMSE, and MAPE
    """
    # MAE: Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))

    # RMSE: Root Mean Squared Error
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # MAPE: Mean Absolute Percentage Error (only if no zeros)
    if np.all(y_true != 0):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    else:
        mape = np.nan

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }


def plot_predictions(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    dates: np.ndarray,
                    title: str = 'Actual vs Predicted Sales',
                    model_name: str = None,
                    figsize: Tuple[int, int] = None,
                    save_filename: str = None) -> plt.Figure:
    """
    Plot actual vs predicted values over time.

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    dates : np.ndarray
        Date array
    title : str
        Plot title
    model_name : str, optional
        Model name for display
    figsize : tuple, optional
        Figure size
    save_filename : str, optional
        If provided, save the figure

    Returns:
    --------
    plt.Figure
        The created figure
    """
    if figsize is None:
        figsize = config.LARGE_FIGURE_SIZE

    fig, ax = plt.subplots(figsize=figsize)

    dates_dt = pd.to_datetime(dates)

    ax.plot(dates_dt, y_true, label='Actual', linewidth=2, alpha=0.8, color='black')
    ax.plot(dates_dt, y_pred, label='Predicted', linewidth=1.5, alpha=0.8, color='red')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sales', fontsize=12)

    if model_name:
        title = f'{title} - {model_name}'

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    if save_filename:
        from .utils import save_figure
        save_figure(fig, save_filename)

    return fig


def plot_errors(y_true: np.ndarray,
               y_pred: np.ndarray,
               dates: np.ndarray,
               title: str = 'Prediction Errors Over Time',
               figsize: Tuple[int, int] = None,
               save_filename: str = None) -> plt.Figure:
    """
    Plot prediction errors over time.

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    dates : np.ndarray
        Date array
    title : str
        Plot title
    figsize : tuple, optional
        Figure size
    save_filename : str, optional
        If provided, save the figure

    Returns:
    --------
    plt.Figure
        The created figure
    """
    if figsize is None:
        figsize = config.LARGE_FIGURE_SIZE

    errors = y_true - y_pred
    dates_dt = pd.to_datetime(dates)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(dates_dt, errors, linewidth=1.5, alpha=0.7, color='purple')
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.fill_between(dates_dt, 0, errors, alpha=0.3, color='purple')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Error (Actual - Predicted)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    if save_filename:
        from .utils import save_figure
        save_figure(fig, save_filename)

    return fig


def plot_residuals(residuals: np.ndarray,
                  title: str = 'Residual Distribution',
                  figsize: Tuple[int, int] = None,
                  save_filename: str = None) -> plt.Figure:
    """
    Plot histogram and statistics of residuals.

    Parameters:
    -----------
    residuals : np.ndarray
        Prediction residuals (errors)
    title : str
        Plot title
    figsize : tuple, optional
        Figure size
    save_filename : str, optional
        If provided, save the figure

    Returns:
    --------
    plt.Figure
        The created figure
    """
    if figsize is None:
        figsize = (12, 5)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    axes[0].hist(residuals, bins=30, alpha=0.7, color='steelblue', edgecolor='black', density=True)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    axes[0].axvline(np.mean(residuals), color='orange', linestyle='--', linewidth=2, label=f'Mean={np.mean(residuals):.2f}')
    axes[0].set_xlabel('Residual', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Residual Histogram', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Q-Q plot (manual)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_filename:
        from .utils import save_figure
        save_figure(fig, save_filename)

    return fig


def create_comparison_table(results_dict: Dict[str, Dict[str, float]],
                           sort_by: str = 'MAE',
                           ascending: bool = True) -> pd.DataFrame:
    """
    Create a comparison table of model results.

    Parameters:
    -----------
    results_dict : dict
        Dictionary with structure: {model_name: {'MAE': value, 'RMSE': value, ...}}
    sort_by : str
        Metric to sort by
    ascending : bool
        Sort in ascending order (True = lower is better)

    Returns:
    --------
    pd.DataFrame
        Formatted results table
    """
    df = pd.DataFrame(results_dict).T

    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)

    # Round to 2 decimal places
    df = df.round(2)

    return df


def error_diagnostics(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     dates: np.ndarray,
                     threshold_percentile: float = 95) -> pd.DataFrame:
    """
    Identify and analyze prediction errors.

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    dates : np.ndarray
        Date array
    threshold_percentile : float
        Percentile threshold for identifying large errors

    Returns:
    --------
    pd.DataFrame
        DataFrame with error diagnostics
    """
    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    # Calculate threshold for large errors
    threshold = np.percentile(abs_errors, threshold_percentile)

    # Find large error indices
    large_error_idx = np.where(abs_errors >= threshold)[0]

    # Create diagnostics dataframe
    diagnostics = pd.DataFrame({
        'date': pd.to_datetime(dates[large_error_idx]),
        'actual': y_true[large_error_idx],
        'predicted': y_pred[large_error_idx],
        'error': errors[large_error_idx],
        'abs_error': abs_errors[large_error_idx],
        'pct_error': (errors[large_error_idx] / y_true[large_error_idx] * 100)
    })

    diagnostics = diagnostics.sort_values('abs_error', ascending=False)

    return diagnostics


def plot_error_by_day_of_week(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              dates: np.ndarray,
                              figsize: Tuple[int, int] = None,
                              save_filename: str = None) -> plt.Figure:
    """
    Analyze errors by day of week.

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    dates : np.ndarray
        Date array
    figsize : tuple, optional
        Figure size
    save_filename : str, optional
        If provided, save the figure

    Returns:
    --------
    plt.Figure
        The created figure
    """
    if figsize is None:
        figsize = (12, 6)

    dates_dt = pd.to_datetime(dates)
    errors = y_true - y_pred

    # Create dataframe
    df = pd.DataFrame({
        'day_of_week': dates_dt.dayofweek,
        'error': errors,
        'abs_error': np.abs(errors)
    })

    # Calculate statistics by day of week
    stats_by_day = df.groupby('day_of_week').agg({
        'error': ['mean', 'std'],
        'abs_error': 'mean'
    }).reset_index()

    stats_by_day.columns = ['day_of_week', 'mean_error', 'std_error', 'mae']

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Mean error by day
    axes[0].bar(range(7), stats_by_day['mean_error'], alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].errorbar(range(7), stats_by_day['mean_error'], yerr=stats_by_day['std_error'],
                    fmt='none', color='red', alpha=0.5, capsize=5)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[0].set_xticks(range(7))
    axes[0].set_xticklabels(days, rotation=45)
    axes[0].set_xlabel('Day of Week', fontsize=12)
    axes[0].set_ylabel('Mean Error', fontsize=12)
    axes[0].set_title('Mean Error by Day of Week', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # MAE by day
    axes[1].bar(range(7), stats_by_day['mae'], alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(days, rotation=45)
    axes[1].set_xlabel('Day of Week', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('MAE by Day of Week', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_filename:
        from .utils import save_figure
        save_figure(fig, save_filename)

    return fig


def plot_scatter_actual_vs_predicted(y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     title: str = 'Actual vs Predicted Scatter Plot',
                                     figsize: Tuple[int, int] = None,
                                     save_filename: str = None) -> plt.Figure:
    """
    Create scatter plot of actual vs predicted values.

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    title : str
        Plot title
    figsize : tuple, optional
        Figure size
    save_filename : str, optional
        If provided, save the figure

    Returns:
    --------
    plt.Figure
        The created figure
    """
    if figsize is None:
        figsize = (8, 8)

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolor='black', linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax.set_xlabel('Actual Sales', fontsize=12)
    ax.set_ylabel('Predicted Sales', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add R² score
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}',
           transform=ax.transAxes, fontsize=12,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_filename:
        from .utils import save_figure
        save_figure(fig, save_filename)

    return fig


def compare_multiple_models(y_true: np.ndarray,
                           predictions_dict: Dict[str, np.ndarray],
                           dates: np.ndarray,
                           figsize: Tuple[int, int] = None,
                           save_filename: str = None) -> plt.Figure:
    """
    Plot multiple model predictions on the same chart.

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    predictions_dict : dict
        Dictionary of {model_name: predictions}
    dates : np.ndarray
        Date array
    figsize : tuple, optional
        Figure size
    save_filename : str, optional
        If provided, save the figure

    Returns:
    --------
    plt.Figure
        The created figure
    """
    if figsize is None:
        figsize = (14, 8)

    dates_dt = pd.to_datetime(dates)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot actual
    ax.plot(dates_dt, y_true, label='Actual', linewidth=2.5, alpha=0.9, color='black')

    # Plot predictions
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_dict)))
    for i, (model_name, predictions) in enumerate(predictions_dict.items()):
        ax.plot(dates_dt, predictions, label=model_name, linewidth=1.5,
               alpha=0.7, color=colors[i])

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sales', fontsize=12)
    ax.set_title('Model Comparison: Actual vs Predictions', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_filename:
        from .utils import save_figure
        save_figure(fig, save_filename)

    return fig
