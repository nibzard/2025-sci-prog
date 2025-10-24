import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    return LinearRegression, train_test_split


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import numpy as np
    return pd, plt, sns


@app.cell
def _(pd):
    data = pd.read_csv("/workspaces/2025-sci-prog/labs/03/marketing.csv")
    data.head(5)
    return (data,)


@app.cell
def _(data):
    X = data[['TV', 'Radio', 'Newspaper']]
    y = data[['Sales']]
    return X, y


@app.cell
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42)
    return X_test, X_train, y_train


@app.cell
def _(LinearRegression, X_train, y_train):
    regr = LinearRegression()
    regr.fit(X_train,y_train)
    return (regr,)


@app.cell
def _(X_test, regr):
    predictedSales = regr.predict(X_test)
    return


@app.cell
def _(data, sns):
    sns.pairplot(data)
    return


@app.cell
def _(data):
    data.isna().sum()
    return


@app.cell
def _(data):
    data.duplicated().sum()
    return


@app.cell
def _(data, sns):
    sns.boxplot(data=data)
    ## postoji nekoliko outliera
    return


@app.cell
def _(data, sns):
    sns.distplot(data)
    return


@app.cell
def _(data, plt, sns):
    plt.figure(figsize=(7,5))
    sns.heatmap(data.corr(numeric_only=True, method="pearson"), annot=True, cmap="rocket", fmt='.2f')
    plt.show()
    return


@app.cell
def _(data, sns):
    sns.pairplot(data)
    return


if __name__ == "__main__":
    app.run()
