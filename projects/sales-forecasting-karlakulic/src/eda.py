"""
Exploratory Data Analysis (EDA) functions for sales forecasting project.
Contains functions for time series visualization, seasonality analysis, and pattern detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
from statsmodels.tsa.seasonal import seasonal_decompose
from . import config


def plot_time_series(df: pd.DataFrame,
                     date_col: str = 'date',
                     value_col: str = 'sales',
                     title: str = 'Time Series Plot',
                     figsize: Tuple[int, int] = None,
                     save_filename: str = None) -> plt.Figure:
    """
    Plot a time series with proper formatting.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the time series data
    date_col : str
        Name of the date column
    value_col : str
        Name of the value column to plot
    title : str
        Plot title
    figsize : tuple, optional
        Figure size (width, height)
    save_filename : str, optional
        If provided, save the figure with this filename

    Returns:
    --------
    plt.Figure
        The created figure
    """
    if figsize is None:
        figsize = config.FIGURE_SIZE

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(df[date_col], df[value_col], linewidth=1, alpha=0.8)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(value_col.title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    if save_filename:
        from .utils import save_figure
        save_figure(fig, save_filename)

    return fig


def plot_rolling_averages(df: pd.DataFrame,
                          date_col: str = 'date',
                          value_col: str = 'sales',
                          windows: List[int] = None,
                          title: str = 'Sales with Rolling Averages',
                          figsize: Tuple[int, int] = None,
                          save_filename: str = None) -> plt.Figure:
    """
    Plot time series with rolling averages overlay.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the time series data
    date_col : str
        Name of the date column
    value_col : str
        Name of the value column
    windows : list of int
        Window sizes for rolling averages (default: [7, 30])
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
    if windows is None:
        windows = [7, 30]

    if figsize is None:
        figsize = config.LARGE_FIGURE_SIZE

    fig, ax = plt.subplots(figsize=figsize)

    # Plot original series
    ax.plot(df[date_col], df[value_col], linewidth=0.8, alpha=0.5,
            label='Daily Sales', color='gray')

    # Plot rolling averages
    colors = ['blue', 'red', 'green', 'orange']
    for i, window in enumerate(windows):
        rolling_mean = df[value_col].rolling(window=window, center=False).mean()
        ax.plot(df[date_col], rolling_mean, linewidth=2,
                label=f'{window}-Day Moving Average',
                color=colors[i % len(colors)])

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(value_col.title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    if save_filename:
        from .utils import save_figure
        save_figure(fig, save_filename)

    return fig


def plot_seasonality(df: pd.DataFrame,
                     value_col: str = 'sales',
                     groupby: str = 'day_of_week',
                     title: str = None,
                     figsize: Tuple[int, int] = None,
                     save_filename: str = None) -> plt.Figure:
    """
    Plot seasonality patterns (e.g., average sales by day of week or month).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with time series data
    value_col : str
        Column to aggregate
    groupby : str
        Column to group by ('day_of_week' or 'month')
    title : str, optional
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
        figsize = (10, 6)

    if title is None:
        title = f'Average {value_col.title()} by {groupby.replace("_", " ").title()}'

    # Calculate mean and std
    grouped = df.groupby(groupby)[value_col].agg(['mean', 'std']).reset_index()

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(grouped[groupby], grouped['mean'], alpha=0.7, color='steelblue',
           edgecolor='black', label='Mean')
    ax.errorbar(grouped[groupby], grouped['mean'], yerr=grouped['std'],
                fmt='none', color='red', alpha=0.5, capsize=5, label='Std Dev')

    # Format x-axis labels
    if groupby == 'day_of_week':
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        ax.set_xticks(range(7))
        ax.set_xticklabels(days, rotation=45, ha='right')
        ax.set_xlabel('Day of Week', fontsize=12)
    elif groupby == 'month':
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(months)
        ax.set_xlabel('Month', fontsize=12)
    else:
        ax.set_xlabel(groupby.replace('_', ' ').title(), fontsize=12)

    ax.set_ylabel(f'Average {value_col.title()}', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    if save_filename:
        from .utils import save_figure
        save_figure(fig, save_filename)

    return fig


def plot_promotion_effect(df: pd.DataFrame,
                          date_col: str = 'date',
                          sales_col: str = 'sales',
                          promo_col: str = 'onpromotion',
                          figsize: Tuple[int, int] = None,
                          save_filename: str = None) -> plt.Figure:
    """
    Visualize the effect of promotions on sales.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sales and promotion data
    date_col : str
        Date column name
    sales_col : str
        Sales column name
    promo_col : str
        Promotion column name
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

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=False)

    # Top plot: Sales and promotions over time
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    ax1.plot(df[date_col], df[sales_col], linewidth=0.8, alpha=0.7,
             color='blue', label='Sales')
    ax1_twin.fill_between(df[date_col], df[promo_col], alpha=0.3,
                          color='orange', label='Promotions')

    ax1.set_ylabel('Sales', fontsize=12, color='blue')
    ax1_twin.set_ylabel('Promotion (Items)', fontsize=12, color='orange')
    ax1.set_title('Sales and Promotions Over Time', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1_twin.tick_params(axis='y', labelcolor='orange')
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Average sales with/without promotion
    ax2 = axes[1]
    promo_comparison = df.groupby(df[promo_col] > 0)[sales_col].agg(['mean', 'std']).reset_index()
    promo_comparison.columns = ['has_promo', 'mean', 'std']
    promo_comparison['promo_label'] = promo_comparison['has_promo'].map({False: 'No Promotion', True: 'With Promotion'})

    ax2.bar(promo_comparison['promo_label'], promo_comparison['mean'],
            alpha=0.7, color=['gray', 'orange'], edgecolor='black')
    ax2.errorbar(range(len(promo_comparison)), promo_comparison['mean'],
                yerr=promo_comparison['std'], fmt='none', color='red',
                alpha=0.5, capsize=5)

    ax2.set_ylabel('Average Sales', fontsize=12)
    ax2.set_xlabel('Promotion Status', fontsize=12)
    ax2.set_title('Sales Comparison: With vs Without Promotion', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_filename:
        from .utils import save_figure
        save_figure(fig, save_filename)

    return fig


def time_series_decomposition(series: pd.Series,
                              freq: int = 7,
                              model: str = 'additive',
                              figsize: Tuple[int, int] = None,
                              save_filename: str = None) -> Tuple[plt.Figure, object]:
    """
    Perform time series decomposition into trend, seasonal, and residual components.

    Parameters:
    -----------
    series : pd.Series
        Time series data (must have datetime index)
    freq : int
        Frequency of the seasonality (default: 7 for weekly)
    model : str
        'additive' or 'multiplicative'
    figsize : tuple, optional
        Figure size
    save_filename : str, optional
        If provided, save the figure

    Returns:
    --------
    tuple
        (Figure, decomposition_result)
    """
    if figsize is None:
        figsize = config.LARGE_FIGURE_SIZE

    # Perform decomposition
    decomposition = seasonal_decompose(series, model=model, period=freq, extrapolate_trend='freq')

    # Create plot
    fig, axes = plt.subplots(4, 1, figsize=figsize)

    # Original
    decomposition.observed.plot(ax=axes[0], legend=False)
    axes[0].set_ylabel('Observed', fontsize=11)
    axes[0].set_title('Time Series Decomposition', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Trend
    decomposition.trend.plot(ax=axes[1], legend=False, color='orange')
    axes[1].set_ylabel('Trend', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Seasonal
    decomposition.seasonal.plot(ax=axes[2], legend=False, color='green')
    axes[2].set_ylabel('Seasonal', fontsize=11)
    axes[2].grid(True, alpha=0.3)

    # Residual
    decomposition.resid.plot(ax=axes[3], legend=False, color='red')
    axes[3].set_ylabel('Residual', fontsize=11)
    axes[3].set_xlabel('Date', fontsize=11)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_filename:
        from .utils import save_figure
        save_figure(fig, save_filename)

    return fig, decomposition


def plot_sales_distribution(df: pd.DataFrame,
                            value_col: str = 'sales',
                            figsize: Tuple[int, int] = None,
                            save_filename: str = None) -> plt.Figure:
    """
    Plot the distribution of sales values.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sales data
    value_col : str
        Column to plot distribution for
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
    axes[0].hist(df[value_col], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].set_xlabel(value_col.title(), fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'Distribution of {value_col.title()}', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Box plot
    axes[1].boxplot(df[value_col], vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1].set_ylabel(value_col.title(), fontsize=12)
    axes[1].set_title(f'Box Plot of {value_col.title()}', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_filename:
        from .utils import save_figure
        save_figure(fig, save_filename)

    return fig


def top_stores_and_families(df: pd.DataFrame,
                            group_col: str = 'store_nbr',
                            value_col: str = 'sales',
                            top_n: int = 10,
                            figsize: Tuple[int, int] = None,
                            save_filename: str = None) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Analyze and plot top stores or product families by total sales.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sales data
    group_col : str
        Column to group by ('store_nbr' or 'family')
    value_col : str
        Value column to aggregate
    top_n : int
        Number of top items to display
    figsize : tuple, optional
        Figure size
    save_filename : str, optional
        If provided, save the figure

    Returns:
    --------
    tuple
        (Figure, DataFrame with top results)
    """
    if figsize is None:
        figsize = (12, 6)

    # Calculate total sales
    top_items = df.groupby(group_col)[value_col].sum().sort_values(ascending=False).head(top_n)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    top_items.plot(kind='barh', ax=ax, color='steelblue', alpha=0.8, edgecolor='black')
    ax.set_xlabel(f'Total {value_col.title()}', fontsize=12)
    ax.set_ylabel(group_col.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Top {top_n} {group_col.replace("_", " ").title()} by Total {value_col.title()}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    if save_filename:
        from .utils import save_figure
        save_figure(fig, save_filename)

    return fig, top_items.to_frame()


def correlation_analysis(df: pd.DataFrame,
                         columns: List[str] = None,
                         method: str = 'pearson',
                         figsize: Tuple[int, int] = None,
                         save_filename: str = None) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Compute and visualize correlation matrix.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with numeric columns
    columns : list of str, optional
        Columns to include (if None, use all numeric)
    method : str
        Correlation method ('pearson', 'spearman', 'kendall')
    figsize : tuple, optional
        Figure size
    save_filename : str, optional
        If provided, save the figure

    Returns:
    --------
    tuple
        (Figure, correlation matrix DataFrame)
    """
    if columns is not None:
        df_corr = df[columns]
    else:
        df_corr = df.select_dtypes(include=[np.number])

    if figsize is None:
        figsize = (10, 8)

    # Compute correlation
    corr_matrix = df_corr.corr(method=method)

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax)
    ax.set_title(f'Correlation Matrix ({method.title()})', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_filename:
        from .utils import save_figure
        save_figure(fig, save_filename)

    return fig, corr_matrix
