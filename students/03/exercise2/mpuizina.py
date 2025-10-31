import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    return LinearRegression, metrics, np, pd, plt, sns, train_test_split


@app.cell
def _(pd):
    marketing_df = pd.read_csv(
        "/workspaces/2025-sci-prog/labs/03/marketing.csv")
    print(marketing_df)
    return (marketing_df,)


@app.cell
def _(marketing_df):
    print("Broj nedostajućih vrijednosti po stupcima:")
    print(marketing_df.isnull().sum())
    return


@app.cell
def _(marketing_df):
    print("Broj dupliciranih redaka:", marketing_df.duplicated().sum())
    return


@app.cell
def _(marketing_df, plt, sns):
    plt.figure(figsize=(8, 6))
    sns.heatmap(marketing_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Korelacijska matrica")
    plt.show()
    return


@app.cell
def _(marketing_df, plt, sns):
    pair_plot = sns.pairplot(marketing_df, x_vars=[
                             'TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=0.7, kind='reg')
    pair_plot.fig.suptitle(
        'Veza između kanala za reklamiranje i ostvarenih prodaja', y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    return


@app.cell
def _(LinearRegression, marketing_df, pd, train_test_split):
    X = marketing_df[['TV', 'Radio', 'Newspaper']]
    y = marketing_df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    coeff_df = pd.DataFrame(
        model.coef_, ['TV', 'Radio', 'Newspaper'], columns=['Coefficient'])
    print(coeff_df)
    print(f"\nIntercept: {model.intercept_:.4f}")
    return X_test, model, y_test


@app.cell
def _(X_test, metrics, model, np, y_test):
    y_pred = model.predict(X_test)

    r2 = metrics.r2_score(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    print(f'R-squared (R2): {r2:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    print(
        f'Mean Absolute Error (MAE): {metrics.mean_absolute_error(y_test, y_pred):.4f}')
    print(
        f'Mean Squared Error (MSE): {metrics.mean_squared_error(y_test, y_pred):.4f}')
    return (y_pred,)


@app.cell
def _(plt, y_pred, y_test):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k',
                label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(),
             y_test.max()], 'r--', lw=2, label='Ideal Fit (y=x)')
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs. Predicted Sales (Test Set)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
