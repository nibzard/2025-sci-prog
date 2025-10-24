import marimo

__generated_with = "0.17.0"
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
    return LinearRegression, metrics, pd, plt, sns, train_test_split


@app.cell
def _(pd):
    data = pd.read_csv('/workspaces/2025-sci-prog/labs/03/marketing.csv')

    print(data.head(5))
    return (data,)


@app.cell
def _(LinearRegression, data, metrics, train_test_split):
    X = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = metrics.r2_score(y_test, y_pred)
    print(f"R-squared value of the model: {r2 * 100:.2f}")
    return y_pred, y_test


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
    sns.boxplot(data)
    return


@app.cell
def _(data, sns):
    sns.displot(data)
    return


@app.cell
def _(data, sns):
    sns.pairplot(data)
    return


@app.cell
def _(data, sns):
    sns.heatmap(data)
    return


@app.cell
def _(
    LinearRegression,
    data,
    metrics,
    plt,
    sns,
    train_test_split,
    y_pred,
    y_test,
):
    X_tv = data[['TV']]
    y_target = data['Sales']

    X_train_tv, X_test_tv, y_train_tv, y_test_tv = train_test_split(
        X_tv, y_target, test_size=0.2, random_state=42)
    model_tv = LinearRegression()
    model_tv.fit(X_train_tv, y_train_tv)
    y_pred_tv = model_tv.predict(X_test_tv)

    r2_tv = metrics.r2_score(y_test_tv, y_pred_tv)
    print(f"R^2 jednostavnog modela (samo za TV): {r2_tv * 100:.2f}%")

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred,
                    label='Višestruka regresija', color='blue')
    sns.scatterplot(x=y_test_tv, y=y_pred_tv,
                    label='Jednostavna regresija (TV)', color='red')
    plt.xlabel("Stvarne vrijednosti prodaje")
    plt.ylabel("Predviđene vrijednosti")
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
