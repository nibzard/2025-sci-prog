"""Interpretation helpers.

Not every model has easy interpretation, but we can still try:
- For linear models: show largest coefficients
- For tree models: show feature importances (if available)
"""

import numpy as np
import pandas as pd


def get_feature_names(fitted_pipeline) -> list[str]:
    """Try to recover feature names after one-hot encoding."""
    preprocess = fitted_pipeline.named_steps["preprocess"]
    cat_transformer = preprocess.named_transformers_["cat"]
    num_features = preprocess.transformers_[1][2]  # ("num", "passthrough", numeric_cols)

    cat_features = preprocess.transformers_[0][2]
    cat_names = list(cat_transformer.get_feature_names_out(cat_features))

    return cat_names + list(num_features)


def top_linear_coefficients(fitted_pipeline, top_n: int = 10) -> pd.DataFrame:
    model = fitted_pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        return pd.DataFrame()

    feature_names = get_feature_names(fitted_pipeline)
    coefs = model.coef_.ravel()

    df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    df["abs_coef"] = np.abs(df["coef"])
    df = df.sort_values("abs_coef", ascending=False).head(top_n)
    return df[["feature", "coef"]]


def top_tree_importances(fitted_pipeline, top_n: int = 10) -> pd.DataFrame:
    model = fitted_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()

    feature_names = get_feature_names(fitted_pipeline)
    importances = model.feature_importances_

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n)
    return df
