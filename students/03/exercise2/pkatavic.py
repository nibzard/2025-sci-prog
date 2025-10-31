import marimo

__generated_with = "0.17.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    import os
    return LinearRegression, metrics, os, pd, plt, sns, train_test_split


@app.cell
def _(os, pd):
    os.chdir("/workspaces/2025-sci-prog/labs/03/")
    data = pd.read_csv('marketing.csv')
    return (data,)


@app.cell
def _(data):
    print("Informacije o datasetu:\n")
    print(data.info())
    print("\nOpis statistike:\n")
    print(data.describe())
    return


@app.cell
def _(data, plt, sns):
    sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=1, kind='scatter')
    plt.suptitle("Odnosi ulaganja u marketing i prodaje", y=1.02)
    plt.show()
    return


@app.cell
def _(data, train_test_split):
    X = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, X_train, y_test, y_train


@app.cell
def _(LinearRegression, X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return (model,)


@app.cell
def _(X_test, metrics, model, y_test):
    y_pred = model.predict(X_test)

    r2 = metrics.r2_score(y_test, y_pred)
    print(f"R² vrijednost modela: {r2 * 100:.2f}%")
    return (y_pred,)


@app.cell
def _(pd, y_pred, y_test):
    results = pd.DataFrame({'Stvarna prodaja': y_test.values, 'Predviđena prodaja': y_pred})
    print(results)
    return


@app.cell
def _(plt, sns, y_pred, y_test):
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=y_test, y=y_pred, color='blue')
    plt.xlabel("Stvarna prodaja")
    plt.ylabel("Predviđena prodaja")
    plt.title("Stvarna vs. Predviđena prodaja")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
