"""
Forecasting models for sales prediction.
Includes baseline models and advanced forecasting models.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')


class BaseForecaster:
    """Base class for all forecasters."""

    def __init__(self):
        self.is_fitted = False
        self.model_name = "BaseForecaster"

    def fit(self, y_train, X_train=None):
        """Fit the model."""
        raise NotImplementedError

    def predict(self, steps=1, X_test=None):
        """Make predictions."""
        raise NotImplementedError


class NaiveForecaster(BaseForecaster):
    """
    Naive forecasting: Next value = Last observed value.

    y_pred[t] = y_true[t-1]

    This is the simplest baseline - assumes tomorrow will be like today.
    """

    def __init__(self):
        super().__init__()
        self.model_name = "Naive"
        self.last_value = None

    def fit(self, y_train, X_train=None):
        """
        Fit by storing the last observed value.

        Parameters:
        -----------
        y_train : array-like
            Training target values
        X_train : ignored
        """
        self.last_value = y_train[-1]
        self.is_fitted = True
        return self

    def predict(self, steps=1, X_test=None):
        """
        Predict by repeating the last observed value.

        Parameters:
        -----------
        steps : int
            Number of steps to forecast
        X_test : ignored

        Returns:
        --------
        np.array
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return np.full(steps, self.last_value)


class SeasonalNaiveForecaster(BaseForecaster):
    """
    Seasonal Naive forecasting: Next value = Value from same season last period.

    y_pred[t] = y_true[t - seasonal_period]

    For daily data with weekly seasonality: tomorrow's sales = same day last week's sales.
    """

    def __init__(self, seasonal_period=7):
        """
        Parameters:
        -----------
        seasonal_period : int
            Length of the seasonal cycle (7 for weekly, 30 for monthly, etc.)
        """
        super().__init__()
        self.model_name = f"Seasonal Naive (period={seasonal_period})"
        self.seasonal_period = seasonal_period
        self.history = None

    def fit(self, y_train, X_train=None):
        """
        Fit by storing the training history.

        Parameters:
        -----------
        y_train : array-like
            Training target values
        X_train : ignored
        """
        self.history = np.array(y_train)
        self.is_fitted = True
        return self

    def predict(self, steps=1, X_test=None):
        """
        Predict using seasonal naive method.

        Parameters:
        -----------
        steps : int
            Number of steps to forecast
        X_test : ignored

        Returns:
        --------
        np.array
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        predictions = []
        for i in range(steps):
            # Get value from seasonal_period steps ago
            idx = len(self.history) - self.seasonal_period + i
            if idx >= 0 and idx < len(self.history):
                predictions.append(self.history[idx])
            else:
                # If we don't have enough history, use last value
                predictions.append(self.history[-1])

        return np.array(predictions)


class MovingAverageForecaster(BaseForecaster):
    """
    Moving Average forecasting: Next value = Average of last N values.

    y_pred[t] = mean(y_true[t-N:t])

    Smooths out noise by averaging recent observations.
    """

    def __init__(self, window=7):
        """
        Parameters:
        -----------
        window : int
            Number of recent values to average
        """
        super().__init__()
        self.model_name = f"Moving Average (window={window})"
        self.window = window
        self.history = None

    def fit(self, y_train, X_train=None):
        """
        Fit by storing the training history.

        Parameters:
        -----------
        y_train : array-like
            Training target values
        X_train : ignored
        """
        self.history = np.array(y_train)
        self.is_fitted = True
        return self

    def predict(self, steps=1, X_test=None):
        """
        Predict using moving average.

        Parameters:
        -----------
        steps : int
            Number of steps to forecast
        X_test : ignored

        Returns:
        --------
        np.array
            Predictions (all steps have same value = average of last window)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Calculate mean of last 'window' values
        last_window = self.history[-self.window:]
        forecast_value = np.mean(last_window)

        return np.full(steps, forecast_value)


class LinearRegressionForecaster(BaseForecaster):
    """
    Linear Regression with engineered features.

    Uses all provided features (lags, rolling stats, calendar, etc.) to predict sales.
    """

    def __init__(self):
        super().__init__()
        self.model_name = "Linear Regression"
        self.model = LinearRegression()

    def fit(self, y_train, X_train):
        """
        Fit linear regression model.

        Parameters:
        -----------
        y_train : array-like
            Training target values
        X_train : array-like
            Training features
        """
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        self.feature_importance = self.model.coef_
        return self

    def predict(self, steps=None, X_test=None):
        """
        Predict using linear regression.

        Parameters:
        -----------
        steps : ignored (uses length of X_test)
        X_test : array-like
            Test features

        Returns:
        --------
        np.array
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if X_test is None:
            raise ValueError("X_test required for LinearRegressionForecaster")

        return self.model.predict(X_test)

    def get_feature_importance(self, feature_names=None):
        """
        Get feature coefficients.

        Parameters:
        -----------
        feature_names : list of str, optional
            Names of features

        Returns:
        --------
        pd.DataFrame or np.array
            Feature importance (coefficients)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        if feature_names is not None:
            return pd.DataFrame({
                'feature': feature_names,
                'coefficient': self.feature_importance
            }).sort_values('coefficient', key=abs, ascending=False)
        else:
            return self.feature_importance


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA/SARIMA forecasting model.

    Auto-Regressive Integrated Moving Average model for time series forecasting.
    Supports seasonal components (SARIMA).
    """

    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
        """
        Parameters:
        -----------
        order : tuple (p, d, q)
            ARIMA order: (autoregressive, differencing, moving average)
        seasonal_order : tuple (P, D, Q, s)
            Seasonal order: (seasonal AR, seasonal diff, seasonal MA, seasonal period)
        """
        super().__init__()
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_name = f"SARIMA{order}×{seasonal_order}"
        self.model = None
        self.model_fit = None

    def fit(self, y_train, X_train=None):
        """
        Fit SARIMA model.

        Parameters:
        -----------
        y_train : array-like
            Training target values
        X_train : array-like, optional
            Exogenous variables
        """
        # Create and fit SARIMAX model
        self.model = SARIMAX(
            y_train,
            exog=X_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        self.model_fit = self.model.fit(disp=False, maxiter=200)
        self.is_fitted = True
        return self

    def predict(self, steps=1, X_test=None):
        """
        Forecast using SARIMA model.

        Parameters:
        -----------
        steps : int
            Number of steps to forecast
        X_test : array-like, optional
            Exogenous variables for forecast period

        Returns:
        --------
        np.array
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        forecast = self.model_fit.forecast(steps=steps, exog=X_test)
        return np.array(forecast)

    def get_model_summary(self):
        """Get statistical summary of fitted model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        return self.model_fit.summary()


class TreeBasedForecaster(BaseForecaster):
    """
    Tree-based forecasting (RandomForest or GradientBoosting).

    Uses ensemble of decision trees with engineered features.
    Supports TimeSeriesSplit cross-validation and hyperparameter tuning.
    """

    def __init__(self, model_type='random_forest', params=None, cv_splits=5, tune=False):
        """
        Parameters:
        -----------
        model_type : str
            'random_forest' or 'gradient_boosting'
        params : dict, optional
            Model hyperparameters
        cv_splits : int
            Number of TimeSeriesSplit folds for cross-validation
        tune : bool
            Whether to perform hyperparameter tuning
        """
        super().__init__()
        self.model_type = model_type
        self.params = params
        self.cv_splits = cv_splits
        self.tune = tune
        self.model_name = f"{model_type.replace('_', ' ').title()}"

        # Initialize model
        if model_type == 'random_forest':
            if params is None:
                params = {'n_estimators': 100, 'random_state': 42}
            self.model = RandomForestRegressor(**params)
        elif model_type == 'gradient_boosting':
            if params is None:
                params = {'n_estimators': 100, 'random_state': 42}
            self.model = GradientBoostingRegressor(**params)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.cv_scores = None
        self.best_params = None

    def fit(self, y_train, X_train, param_grid=None):
        """
        Fit tree-based model with optional hyperparameter tuning.

        Parameters:
        -----------
        y_train : array-like
            Training target values
        X_train : array-like
            Training features
        param_grid : dict, optional
            Parameter grid for GridSearchCV
        """
        if self.tune and param_grid is not None:
            # Perform time series cross-validation with grid search
            tscv = TimeSeriesSplit(n_splits=self.cv_splits)

            grid_search = GridSearchCV(
                self.model,
                param_grid,
                cv=tscv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(X_train, y_train)

            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.cv_scores = -grid_search.best_score_  # Convert back to positive MAE

            print(f"Best parameters: {self.best_params}")
            print(f"Best CV MAE: {self.cv_scores:.2f}")
        else:
            # Simple fit without tuning
            self.model.fit(X_train, y_train)

        self.is_fitted = True
        return self

    def predict(self, steps=None, X_test=None):
        """
        Predict using tree-based model.

        Parameters:
        -----------
        steps : ignored (uses length of X_test)
        X_test : array-like
            Test features

        Returns:
        --------
        np.array
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if X_test is None:
            raise ValueError("X_test required for TreeBasedForecaster")

        return self.model.predict(X_test)

    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance from tree-based model.

        Parameters:
        -----------
        feature_names : list of str, optional
            Names of features

        Returns:
        --------
        pd.DataFrame or np.array
            Feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        importance = self.model.feature_importances_

        if feature_names is not None:
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        else:
            return importance


def check_stationarity(series, significance_level=0.05):
    """
    Check if a time series is stationary using Augmented Dickey-Fuller test.

    Parameters:
    -----------
    series : array-like
        Time series data
    significance_level : float
        Significance level for the test (default: 0.05)

    Returns:
    --------
    dict
        Test results with p-value and stationarity decision
    """
    result = adfuller(series, autolag='AIC')

    output = {
        'test_statistic': result[0],
        'p_value': result[1],
        'n_lags': result[2],
        'n_obs': result[3],
        'critical_values': result[4],
        'is_stationary': result[1] < significance_level
    }

    return output


def print_stationarity_test(series, series_name="Series"):
    """
    Print formatted results of stationarity test.

    Parameters:
    -----------
    series : array-like
        Time series data
    series_name : str
        Name of the series for display
    """
    result = check_stationarity(series)

    print(f"\nAugmented Dickey-Fuller Test for {series_name}")
    print("=" * 60)
    print(f"Test Statistic: {result['test_statistic']:.4f}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Number of lags used: {result['n_lags']}")
    print(f"Number of observations: {result['n_obs']}")
    print("\nCritical Values:")
    for key, value in result['critical_values'].items():
        print(f"  {key}: {value:.3f}")

    if result['is_stationary']:
        print("\n✓ Result: Series is STATIONARY (p-value < 0.05)")
    else:
        print("\n✗ Result: Series is NON-STATIONARY (p-value >= 0.05)")
        print("  Consider differencing the series.")

    return result
