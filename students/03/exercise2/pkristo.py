import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return mo


@app.cell
def _():
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    return pd, LinearRegression, train_test_split


@app.cell
def _(pd):
    # Učitavanje podataka
    data = pd.read_csv("/workspaces/2025-sci-prog/labs/03/marketing.csv")
    data
    return data


@app.cell
def _(data):
    # Priprema X i y
    X = data[['TV', 'Radio', 'Newspaper']]
    y = data[['Sales']]
    return X, y


@app.cell
def _(X, y, train_test_split):
    # Podjela na train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )
    return X_train, X_test, y_train, y_test


@app.cell
def _(LinearRegression, X_train, y_train):
    # Treniranje modela
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


@app.cell
def _(model, X_test):
    # Predikcija
    y_pred = model.predict(X_test)
    return y_pred


@app.cell
def _(model, X_test, y_test):
    # R2 evaluacija
    r2_score = model.score(X_test, y_test)
    r2_score
    return r2_score


if __name__ == "__main__":
    app.run()
