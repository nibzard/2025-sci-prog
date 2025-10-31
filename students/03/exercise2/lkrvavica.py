import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    return pd, plt, sns


@app.cell
def _():
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    return LinearRegression, metrics, train_test_split


@app.cell
def _(pd):
    df = pd.read_csv("labs/03/marketing.csv")

    print(df.head())
    print(df.info())
    print(df.describe())
    return (df,)


@app.cell
def _(df):
    print("Nedostajuće vrijednosti:")
    print(df.isna().sum()) 

    print("Broj dupliciranih redova:", df.duplicated().sum())

    return


@app.cell
def _(df, plt, sns):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1)
    sns.boxplot(y=df['TV'])
    plt.title('TV')

    plt.subplot(1, 4, 2)
    sns.boxplot(y=df['Radio'])
    plt.title('Radio marketing')

    plt.subplot(1, 4, 3)
    sns.boxplot(y=df['Newspaper'])
    plt.title('Novine marketing')

    plt.subplot(1, 4, 4)
    sns.boxplot(y=df['Sales']) 
    plt.title('Prodaja')

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df, plt, sns):
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Sales'], kde=True)
    plt.title('Distribucija prodaje')
    plt.xlabel('Prodaja')
    plt.ylabel('Frekvencija')
    plt.show()
    return


@app.cell
def _(df, plt, sns):
    sns.pairplot(df)
    plt.show()

    plt.figure(figsize=(8, 6))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Korelacijska matrica')
    plt.show()
    return


@app.cell
def _(df):
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    return X, y


@app.cell
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, X_train, y_test, y_train


@app.cell
def _(LinearRegression, X_test, X_train, metrics, y_test, y_train):
    # Kreiranje i treniranje modela
    model_multiple = LinearRegression()
    model_multiple.fit(X_train, y_train)

    # Predikcija na testnom skupu
    y_pred_multiple = model_multiple.predict(X_test)

    # Evaluacija modela
    r2_multiple = metrics.r2_score(y_test, y_pred_multiple)
    mse_multiple = metrics.mean_squared_error(y_test, y_pred_multiple)
    mae_multiple = metrics.mean_absolute_error(y_test, y_pred_multiple)

    print("Višestruka linearna regresija - Rezultati:")
    print(f"R-squared: {r2_multiple:.4f}")
    print(f"Mean Squared Error: {mse_multiple:.4f}")
    print(f"Mean Absolute Error: {mae_multiple:.4f}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
