# marimo app: pokreni s `marimo run sjadrijev_marimo.py`

import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from joblib import dump
    import marimo as mo
    return (
        LinearRegression,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        pd,
        r2_score,
        train_test_split,
    )


@app.cell
def _(pd):
    # Putanja do CSV-a (skripta je u students/03/exercise2/)
    CSV_PATH = '/workspaces/2025-sci-prog/labs/03/marketing.csv'

    df = pd.read_csv(CSV_PATH)
    df.head()
    return (df,)


@app.cell
def _(df):
    feature_cols = ["TV", "Radio", "Newspaper"]
    target_col = "Sales"
    X = df[feature_cols]
    y = df[target_col]
    return X, feature_cols, y


@app.cell
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train.shape, X_test.shape
    return X_test, X_train, y_test, y_train


@app.cell
def _(LinearRegression, X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    model
    return (model,)


@app.cell
def _(
    X_test,
    mean_absolute_error,
    mean_squared_error,
    mo,
    model,
    r2_score,
    y_test,
):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)

    mo.md(
        f"""
    ### Metrike na test skupu
    - **R²:** {r2:.4f}  
    - **MAE:** {mae:.4f}  
    - **RMSE:** {rmse:.4f}
    """
    )
    return (y_pred,)


@app.cell
def _(mo, model):
    intercept = model.intercept_
    coefs = model.coef_
    mo.md(
        f"""
    ### Model: `Sales ~ TV + Radio + Newspaper`

    - **Intercept:** {intercept:.4f}  
    - **Koeficijenti:**  
      - TV: {coefs[0]:.4f}  
      - Radio: {coefs[1]:.4f}  
      - Newspaper: {coefs[2]:.4f}
    """
    )
    return


@app.cell
def _(model):
    def predict_sales(tv: float, radio: float, newspaper: float) -> float:
        import pandas as pd
        X_new = pd.DataFrame(
            [[tv, radio, newspaper]],
            columns=["TV", "Radio", "Newspaper"]
        )
        return float(model.predict(X_new)[0])

    return


@app.cell
def _():
    # Jednostavna "ručna" predikcija – promijeni vrijednosti i ponovno pokreni ćeliju
    TV = 120.0     # <— promijeni po potrebi
    Radio = 25.0   # <— promijeni po potrebi
    Newspaper = 10.0  # <— promijeni po potrebi
    TV, Radio, Newspaper
    return


@app.cell
def _(y_pred, y_test):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Stvarne prodaje (y_test)")
    plt.ylabel("Predviđene prodaje (y_pred)")
    plt.title("Actual vs Predicted")

    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims)  # 45° linija
    plt.show()

    return (plt,)


@app.cell
def _(plt, y_pred, y_test):
    residuals = y_test - y_pred

    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predviđene")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")
    plt.show()

    plt.figure()
    plt.hist(residuals, bins=20)
    plt.title("Distribucija residuala")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.show()
    return


@app.cell
def _(feature_cols, model, plt):
    plt.figure()
    plt.bar(feature_cols, model.coef_)
    plt.title("Koeficijenti modela")
    plt.ylabel("Veličina koeficijenta")
    plt.show()

    return


@app.cell
def _(X, model, np, pd, plt, y):
    means = X.mean()

    def plot_partial(feature):
        xs = np.linspace(X[feature].min(), X[feature].max(), 100)
        X_plot = pd.DataFrame({
            "TV": means["TV"],
            "Radio": means["Radio"],
            "Newspaper": means["Newspaper"],
        }, index=range(len(xs)))
        X_plot[feature] = xs
        ys = model.predict(X_plot)

        plt.figure()
        plt.scatter(X[feature], y, alpha=0.4, label="podaci")
        plt.plot(xs, ys, label="model (ostalo = mean)")
        plt.xlabel(feature); plt.ylabel("Sales")
        plt.title(f"Sales vs {feature}")
        plt.legend()
        plt.show()

    plot_partial("TV")
    plot_partial("Radio")
    plot_partial("Newspaper")
    return


if __name__ == "__main__":
    app.run()
