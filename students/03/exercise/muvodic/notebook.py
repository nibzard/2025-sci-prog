import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    import numpy as np
    return LinearRegression, mo, pd, plt


@app.cell
def _(pd):
    data = pd.read_csv(
        "/workspaces/2025-sci-prog/students/03/data/16_rent_price.csv")
    data.head()
    return (data,)


@app.cell
def _(LinearRegression, data):
    X = data[["distance_to_center"]].values
    y = data["rent_price"].values
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    return X, model


@app.cell
def _(X, data, mo, model, plt):
    X_data = data["distance_to_center"].values.reshape(-1, 1)
    y_pred = model.predict(X)

    fig, ax = plt.subplots()
    ax.scatter(data["distance_to_center"], data["rent_price"], label="Data")
    ax.plot(data["distance_to_center"], y_pred,
            color="red", label="Regression Line")
    ax.set_xlabel("Distance to City Center (km)")
    ax.set_ylabel("Rent Price ($)")
    ax.legend()
    mo.mpl.interactive(fig)
    return


if __name__ == "__main__":
    app.run()
