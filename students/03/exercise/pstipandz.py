import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    import numpy as np
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    return LinearRegression, mean_squared_error, np, pd, plt, r2_score, sns


@app.cell
def _(LinearRegression, mean_squared_error, np, pd, plt, r2_score, sns):
    df = pd.read_csv("/workspaces/2025-sci-prog/students/03/data/04_fuel_consumption.csv")

    X = df[['engine_size']]
    Y = df['fuel_consumption']

    model = LinearRegression()
    model.fit(X, Y)

    y_pred = model.predict(X)

    r2 = r2_score(Y, y_pred)
    mse = mean_squared_error(Y, y_pred)
    rmse = np.sqrt(mse)

    print(f"Intercept: {model.intercept_:.3f}")
    print(f"R-squared: {r2:.5f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MSE: {mse:.2f}")

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x='engine_size', y='fuel_consumption', data=df, s=80)
    plt.plot(X, y_pred, color='red', linewidth=1)
    plt.title('Veličina motora vs Potrošnja goriva')
    plt.xlabel('Veličina motora')
    plt.ylabel('Potrošnja goriva')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
