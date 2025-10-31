import marimo

__generated_with = "0.16.5"
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
    return LinearRegression, metrics, np, pd, plt, sns


@app.cell
def _(pd):
    df = pd.read_csv('/workspaces/2025-sci-prog/labs/03/marketing.csv')
    return (df,)


@app.cell
def _(df):
    print(df.head())
    return


@app.cell
def _(df):
    print("Nedostaju li vrijednosti po stupcima:")
    print(df.isna().sum(), "\n")
    return


@app.cell
def _(df):
    print("Broj duplikata:", df.duplicated().sum(), "\n")
    return


@app.cell
def _(df, plt, sns):
    print("Korelacijska matrica:")
    corr = df.corr()
    print(corr, "\n")
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="Greens", fmt=".2f")
    plt.title("Korelacija između varijabli")
    plt.show()
    return


@app.cell
def _(df, plt, sns):
    sns.pairplot(df)
    plt.show()
    return


@app.cell
def _(df, plt):
    df.boxplot()
    plt.show()
    return


@app.cell
def _(df, plt):
    df.hist()
    plt.show()
    return


@app.cell
def _(df):
    X = df.drop('Sales', axis=1)
    y = df['Sales']
    return X, y


@app.cell
def _(X):
    print(X)
    return


@app.cell
def _(y):
    print(y)
    return


@app.cell
def _(LinearRegression, X, y):
    regr = LinearRegression()
    regr.fit(X, y)
    return (regr,)


@app.cell
def _(X, regr):
    predictions = regr.predict(X)
    return (predictions,)


@app.cell
def _(regr):
    print("Intercept:", regr.intercept_)
    return


@app.cell
def _(regr):
    print("Coefficients:", regr.coef_)
    return


@app.cell
def _(metrics, np, predictions, y):
    print("R² Score:", metrics.r2_score(y, predictions))
    print("Mean Absolute Error:", metrics.mean_absolute_error(y, predictions))
    print("Mean Squared Error:", metrics.mean_squared_error(y, predictions))
    print("Root Mean Squared Error:", np.sqrt(
        metrics.mean_squared_error(y, predictions)))
    return


if __name__ == "__main__":
    app.run()
