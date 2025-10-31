import marimo

__generated_with = "0.17.2"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    import marimo as mo
    return (
        LinearRegression,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        pd,
        plt,
        r2_score,
        train_test_split,
    )


@app.cell
def _(pd):
    DATA_PATH = "/workspaces/2025-sci-prog/labs/03/marketing.csv"

    data = pd.read_csv(DATA_PATH)
    data.info()
    return (data,)


@app.cell
def _(data, mo):
    mo.md(f"""
    Dimenzije skupa: **{data.shape[0]} redova × {data.shape[1]} kolone**

    Prvih 5 redova:
    """)
    data.head()
    return


@app.cell
def _(data):
    features = ["TV", "Radio", "Newspaper"]
    target = "Sales"

    X = data[features]
    y = data[target]
    return X, features, y


@app.cell
def _(X, train_test_split, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=7)
    (X_tr.shape, X_te.shape)
    return X_te, X_tr, y_te, y_tr


@app.cell
def _(LinearRegression, X_tr, y_tr):
    reg = LinearRegression()
    reg.fit(X_tr, y_tr)
    return (reg,)


@app.cell
def _(X_te, mean_absolute_error, mean_squared_error, mo, r2_score, reg, y_te):
    preds = reg.predict(X_te)

    results = {
        "R2": r2_score(y_te, preds),
        "MAE": mean_absolute_error(y_te, preds),
        "RMSE": mean_squared_error(y_te, preds),
    }

    mo.md(f"""
    ### Rezultati modela
    - **R² (koeficijent determinacije):** {results['R2']:.3f}  
    - **MAE (prosječna aps. pogreška):** {results['MAE']:.3f}  
    - **RMSE (korijen MSE):** {results['RMSE']:.3f}
    """)
    return (preds,)


@app.cell
def _(mo, reg):
    coefs = reg.coef_
    intercept = reg.intercept_

    mo.md(f"""
    ### Parametri linearnog modela

    Formula:  
    **Sales = {intercept:.3f} + ({coefs[0]:.3f})·TV + ({coefs[1]:.3f})·Radio + ({coefs[2]:.3f})·Newspaper**
    """)
    return


@app.cell
def _(plt, preds, y_te):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_te, preds, alpha=0.7)
    lims = [min(y_te.min(), preds.min()), max(y_te.max(), preds.max())]
    plt.plot(lims, lims, 'r--')
    plt.xlabel("Stvarne vrijednosti")
    plt.ylabel("Predviđene vrijednosti")
    plt.title("Stvarne vs Predviđene prodaje")
    plt.show()
    return


@app.cell
def _(plt, preds, y_te):
    residuals = y_te - preds

    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=25)
    plt.title("Distribucija pogrešaka (residuala)")
    plt.xlabel("Residual")
    plt.ylabel("Broj pojava")
    plt.show()
    return (residuals,)


@app.cell
def _(plt, preds, residuals):
    plt.figure(figsize=(6, 4))
    plt.scatter(preds, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predviđene vrijednosti")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted")
    plt.show()
    return


@app.cell
def _(features, plt, reg):
    plt.figure(figsize=(5, 4))
    plt.bar(features, reg.coef_)
    plt.title("Utjecaj pojedinih varijabli na Sales")
    plt.ylabel("Koeficijent")
    plt.show()
    return


@app.cell
def _(pd, reg):
    def predict_sales(tv, radio, newspaper):
        X_new = pd.DataFrame([[tv, radio, newspaper]], columns=[
                             "TV", "Radio", "Newspaper"])
        return float(reg.predict(X_new)[0])
    return (predict_sales,)


@app.cell
def _(mo, predict_sales):
    mo.md("### Probna predikcija")
    tv_val, radio_val, news_val = 100.0, 30.0, 10.0
    mo.md(f"Za **TV={tv_val}**, **Radio={radio_val}**, **Newspaper={news_val}**, predviđena prodaja je: **{predict_sales(tv_val, radio_val, news_val):.2f}**")
    return


@app.cell
def _(X, np, pd, plt, reg, y):
    mean_values = X.mean()

    def partial_dependence_plot(feature):
        xs = np.linspace(X[feature].min(), X[feature].max(), 100)
        X_ref = pd.DataFrame(
            {col: mean_values[col] for col in X.columns}, index=range(100))
        X_ref[feature] = xs
        ys = reg.predict(X_ref)

        plt.figure(figsize=(6, 4))
        plt.scatter(X[feature], y, alpha=0.3, label="Podaci")
        plt.plot(xs, ys, color='orange', label="Model")
        plt.xlabel(feature)
        plt.ylabel("Sales")
        plt.title(f"Utjecaj {feature} na Sales (ostalo = sredina)")
        plt.legend()
        plt.show()

    for feat in X.columns:
        partial_dependence_plot(feat)
    return


if __name__ == "__main__":
    app.run()
