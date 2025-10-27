# Marimo aplikacija: linearna regresija - Education vs Salary

import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell
def import_libs():
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    return LinearRegression, pd, plt, r2_score


@app.cell
def dataset(pd):
    data = pd.DataFrame({
        "education_years": [
            12, 14, 16, 18, 20, 22,
            13, 15, 17, 19, 21, 23,
            12.5, 14.5, 16.5, 18.5, 20.5,
            11, 24, 13.5
        ],
        "salary": [
            35000, 42000, 50000, 58000, 66000, 74000,
            38500, 46000, 54000, 62000, 70000, 78000,
            36750, 44000, 52000, 60000, 68000,
            31500, 82000, 40250
        ]
    })
    data.head()
    return (data,)


@app.cell
def prepare_data(data):
    X = data[["education_years"]]
    y = data["salary"]
    return X, y


@app.cell
def train_model(LinearRegression, X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred


@app.cell
def model_results(model, r2_score, y, y_pred):
    print(
        f"Jednadžba regresije: salary = {model.coef_[0]:.2f} * education_years + {model.intercept_:.2f}")
    print(f"R² score: {r2_score(y, y_pred):.4f}")
    return


@app.cell
def plot_results(X, plt, y, y_pred):
    plt.scatter(X, y, color='blue', label='Stvarni podaci')
    plt.plot(X, y_pred, color='red', label='Linearni model')
    plt.xlabel('Education Years')
    plt.ylabel('Salary')
    plt.title('Linearna regresija: Education Years vs Salary')
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
