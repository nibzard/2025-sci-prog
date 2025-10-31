import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    return LinearRegression, pd, train_test_split


@app.cell
def _(pd):
    # Učitavanje podataka
    df = pd.read_csv("/workspaces/2025-sci-prog/labs/03/marketing.csv")
    df
    return (df,)


@app.cell
def _(df):
    # Priprema X i y
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df[['Sales']]
    return X, y


@app.cell
def _(X, train_test_split, y):
    # Podjela na train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(LinearRegression, X_train, y_train):
    # Treniranje modela
    model = LinearRegression()
    model.fit(X_train, y_train)
    return (model,)


@app.cell
def _(X_test, model):
    # Predikcija
    y_pred = model.predict(X_test)
    return


@app.cell
def _(X_test, model, y_test):
    # R2 evaluacija
    r2_score = model.score(X_test, y_test)
    r2_score
    return


if __name__ == "__main__":
    app.run()
