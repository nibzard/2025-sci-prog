import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas
    return (pandas,)


@app.cell
def _(pandas):
    df = pandas.read_csv("data.csv")
    return (df,)


@app.cell
def _(df):
    X = df[['Weight', 'Volume']]
    y = df['CO2']
    return X, y


@app.cell
def _():
    from sklearn import linear_model
    return (linear_model,)


@app.cell
def _(X, linear_model, y):
    regr = linear_model.LinearRegression()
    regr.fit(X,y)
    return (regr,)


@app.cell
def _(pandas, regr):
    # Redefining the prediction input with valid feature names

    prediction_input = pandas.DataFrame([[2300, 1300]], columns=['Weight', 'Volume'])
    predicted_CO2 = regr.predict(prediction_input)
    predicted_CO2
    return


@app.cell
def _():
    import seaborn as sns
    return (sns,)


@app.cell
def _(df, sns):
    sns.pairplot(df)
    return


@app.cell
def _(regr):
    ### ERROR
    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True

    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)

    x, y = np.meshgrid(x, y)
    eq = regr.coef_[0] * x + regr.coef_[1] * y + regr.intercept_

    fig = plt.figure()

    ax = fig.gca(projection='3d')

    ax.plot_surface(x, y, eq)

    plt.show()
    return (y,)


@app.cell
def _(regr):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True

    _x = np.linspace(-10, 10, 100)
    _y = np.linspace(-10, 10, 100)

    _x, _y = np.meshgrid(_x, _y)
    _eq = regr.coef_[0] * _x + regr.coef_[1] * _y + regr.intercept_

    fig = plt.figure()

    _ax = fig.add_subplot(111, projection='3d')

    _ax.plot_surface(_x, _y, _eq)

    plt.gca()
    return


@app.cell
def _(X, y):
    import statsmodels.api as sm

    model = sm.OLS(y, X).fit()
    print(model.summary())
    return


if __name__ == "__main__":
    app.run()
