import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics

    df = pd.read_csv('marketing.csv')
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r_squared = r2_score(y_test, y_pred)
    print(f"R-squared value: {r_squared}")


    return


if __name__ == "__main__":
    app.run()
