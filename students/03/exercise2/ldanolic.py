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
    from sklearn.preprocessing import StandardScaler
    from sklearn import metrics
    return (
        LinearRegression,
        StandardScaler,
        metrics,
        pd,
        plt,
        sns,
        train_test_split,
    )

@app.cell
def _():
    #labs -> podaci
    return

@app.cell
def _(pd):
    data = pd.read_csv("/workspaces/2025-sci-prog/labs/03/marketing.csv")
    data.head()
    return (data,)

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
    data_corr = data.corr()
    sns.heatmap(data=data_corr, annot=True)
    return

@app.cell
def _(StandardScaler, data, train_test_split):
    X_train, X_test, y_train, y_test = train_test_split(
        data[['TV', 'Radio', 'Newspaper']], data['Sales'],
        random_state=42, test_size=0.25
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    return X_test, X_train, y_test, y_train

@app.cell
def _(LinearRegression, X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return (model,)

@app.cell
def _(X_test, model):
    y_predict = model.predict(X_test)
    return (y_predict,)

@app.cell
def _(metrics, y_predict, y_test):
    metrics.mean_absolute_error(y_test, y_predict)
    return

@app.cell
def _(metrics, y_predict, y_test):
    metrics.mean_squared_error(y_test, y_predict)
    return

@app.cell
def _(metrics, y_predict, y_test):
    metrics.root_mean_squared_error(y_test, y_predict)
    return

@app.cell
def _(metrics, y_predict, y_test):
    metrics.r2_score(y_test, y_predict)
    return

@app.cell
def _(plt, y_predict, y_test):
    plt.scatter(y_test, y_predict, alpha=0.7, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Prodaja")
    plt.ylabel("Predvidanje")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return

if __name__ == "__main__":
    app.run()
