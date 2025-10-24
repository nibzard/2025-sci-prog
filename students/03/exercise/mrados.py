import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    return (
        LinearRegression,
        mean_squared_error,
        pd,
        plt,
        r2_score,
        train_test_split,
    )


@app.cell
def _(pd):
    df = pd.read_csv("../data/05_house_price.csv")
    df.head()
    return (df,)


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    X = df.drop(columns=["price"])   # Features
    y = df["price"]                  # Target
    return X, y


@app.cell
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(LinearRegression, X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return (model,)


@app.cell
def _(X_test, mean_squared_error, model, r2_score, y_test):
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R² Score:", r2)
    return (y_pred,)


@app.cell
def _(X, model, pd):
    coefficients = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    })
    coefficients
    return (coefficients,)


@app.cell
def _(coefficients):
    print(coefficients)
    return


@app.cell
def _(plt, y_pred, y_test):
    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color="royalblue", edgecolor="k")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             "r--", lw=2, label="Perfect Fit")

    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted House Prices")
    plt.legend()
    plt.grid(True)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
