"""Very small tests just to show a real project structure."""

from src.features import add_features
import pandas as pd


def test_add_features_creates_columns():
    df = pd.DataFrame(
        {
            "age": [19],
            "sex": ["female"],
            "bmi": [27.9],
            "children": [0],
            "smoker": ["yes"],
            "region": ["southwest"],
            "charges": [16884.9],
        }
    )

    out = add_features(df)
    assert "bmi_category" in out.columns
    assert "is_smoker" in out.columns
    assert "smoker_x_bmi" in out.columns
    assert "age_x_bmi" in out.columns
