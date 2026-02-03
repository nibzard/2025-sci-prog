"""Model definitions.

We keep the models in one place so `train.py` stays clean.
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def get_model_zoo(random_state: int):
    """Return a dictionary of model name -> model object."""
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "Lasso": Lasso(alpha=0.01, random_state=random_state, max_iter=50000),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
    }
