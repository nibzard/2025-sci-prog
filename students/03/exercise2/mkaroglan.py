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
    return LinearRegression, pd, sns


@app.cell
def _(pd):
    file_df = pd.read_csv('/workspaces/2025-sci-prog/labs/03/marketing.csv')
    return (file_df,)


@app.cell
def _(file_df):
    print(file_df.head())
    return


@app.cell
def _(file_df, sns):
    sns.pairplot(file_df)
    return


@app.cell
def _(file_df):
    file_df.boxplot()
    return


@app.cell
def _(file_df):
    file_df.hist()
    return


@app.cell
def _(file_df):
    X=file_df.drop('Sales', axis=1)
    y = file_df['Sales']
    return X, y


@app.cell
def _(LinearRegression, X, y):
    regr=LinearRegression()
    regr.fit(X,y)
    return (regr,)


@app.cell
def _(X, regr):
    predvidanje=regr.predict(X)
    return


@app.cell
def _(regr):
    regr.intercept_
    return


@app.cell
def _(impo):
    impo
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
