import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    return LinearRegression, metrics, np, pd, plt, sns, train_test_split


@app.cell
def _(pd):
    # load data
    data=pd.read_csv("/workspaces/2025-sci-prog/labs/03/marketing.csv")
    data.head(3)
    return (data,)


@app.cell
def _(data):
    print(data.isnull().sum())
    print("\n Duplikati")
    print(data.duplicated().sum())
    return


@app.cell
def _(LinearRegression, data, metrics, np, plt, train_test_split):
    x_train, x_test, y_train, y_test = train_test_split(data[["TV","Radio","Newspaper"]], data["Sales"], test_size=0.2, random_state=42)
    model=LinearRegression()
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print("R2 Score:", metrics.r2_score(y_test, y_pred))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual sales vs Predicted sales")
    return


@app.cell
def _(data, plt, sns):
    # # korelacija
    correlation=data.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(correlation, annot=True, cmap="coolwarm")
    plt.title("Correlation")
    return


if __name__ == "__main__":
    app.run()
