"""Feature engineering.

We create a few extra features to test the hypothesis that smoking, BMI, and age
are big drivers of charges.
"""

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # BMI category (simple bins)
    def bmi_category(bmi: float) -> str:
        if bmi < 18.5:
            return "underweight"
        if bmi < 25:
            return "normal"
        if bmi < 30:
            return "overweight"
        return "obese"

    df["bmi_category"] = df["bmi"].apply(bmi_category)

    # Smoker as 0/1
    df["is_smoker"] = (df["smoker"] == "yes").astype(int)

    # Interaction terms (sometimes models like tree-based ones can use these)
    df["smoker_x_bmi"] = df["is_smoker"] * df["bmi"]
    df["age_x_bmi"] = df["age"] * df["bmi"]

    return df
