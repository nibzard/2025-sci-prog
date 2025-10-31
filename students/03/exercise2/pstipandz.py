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
    return LinearRegression, pd, plt, sns


@app.cell
def _(LinearRegression, pd):
    df = pd.read_csv('/workspaces/2025-sci-prog/labs/03/marketing.csv')
    print(df.head(5))

    X = df[['TV','Radio','Newspaper']]

    Y = df['Sales']

    regr = LinearRegression()
    regr.fit(X,Y)

    print(regr)

    predikcija = regr.predict(X)
    #print(predikcija)

    R2 = regr.score(X, Y)
    print(R2)
    return (df,)


@app.cell
def _(df):
    print("Null vrj.:",df.isna().sum())
    print(f"Broj dupliranih redaka: ",df.duplicated().sum())
    return


@app.cell
def _(df, sns):
    sns.pairplot(df)
    return


@app.cell
def _(df, plt):
    plt.figure(figsize=(10,5))
    plt.title("Boxplot - outliers")
    df.boxplot()
    plt.show()
    return


@app.cell
def _(df, plt, sns):
    plt.figure(figsize=(6,4))
    sns.histplot(df['Sales'], kde=True, bins=20)
    plt.title("Distribucija ciljne varijable")
    plt.xlabel("Prodaja")
    plt.ylabel("Frekvencija")
    plt.show()

    return


@app.cell
def _(df, plt, sns):
    plt.figure(figsize=(8,8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Korelacija između varijabli")
    plt.show()
    return


@app.cell
def _(df):
    correlation_matrix = df.corr(numeric_only=True)
    print("Korelacijska matrica")
    print(correlation_matrix)
    return


if __name__ == "__main__":
    app.run()
