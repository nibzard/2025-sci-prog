import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    return (
        LinearRegression,
        mean_squared_error,
        mo,
        pd,
        plt,
        r2_score,
        train_test_split,
    )


@app.cell
def _(pd):
    csv_path = "../data/28_sales_revenue.csv"
    df = pd.read_csv(csv_path)
    df.head()
    return (df,)


@app.cell
def _(df):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    lower_cols = [c.lower() for c in df.columns]
    if "revenue" in lower_cols:
        y_col = df.columns[lower_cols.index("revenue")]
    elif "sales" in lower_cols:
        y_col = df.columns[lower_cols.index("sales")]
    else:
        y_col = numeric_cols[-1]

    X_cols = [c for c in numeric_cols if c != y_col]

    X = df[X_cols]
    y = df[y_col]

    X.head(), y.head()
    return X, y


@app.cell
def _(
    LinearRegression,
    X,
    mean_squared_error,
    mo,
    r2_score,
    train_test_split,
    y,
):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    r2 = r2_score(y_test, y_pred)

    mo.md(f"**RMSE:** {rmse:.4f}  \n**R²:** {r2:.4f}")
    return r2, rmse, y_pred, y_test


@app.cell
def _(plt, r2, rmse, y_pred, y_test):
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Actual vs Predicted (R²={r2:.3f}, RMSE={rmse:.3f})")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.show()

    return


if __name__ == "__main__":
    app.run()
