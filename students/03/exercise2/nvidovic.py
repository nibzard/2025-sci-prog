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
    from sklearn import linear_model
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics

    from sklearn.preprocessing import StandardScaler
    return (
        StandardScaler,
        linear_model,
        metrics,
        np,
        pd,
        plt,
        sns,
        train_test_split,
    )


@app.cell
def _(pd):
    marketing_df = pd.read_csv("/workspaces/2025-sci-prog/labs/03/marketing.csv")
    marketing_df
    return (marketing_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""###Skup podataka se sastoji od 4 stupca numeričkih podataka. Logički zaključak je da pokušamo napraviti model za predviđanje `'Sales'` stupca za neke proizvoljne vrijednosti ostalih stupaca.""")
    return


@app.cell
def _(marketing_df):
    marketing_df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""###Vidimo da je srednja vrijednost stupca 'TV' dosta veća od ostalih, to bi nam moglo stvarati preveliku ovisnost modela o tome stupcu.""")
    return


@app.cell
def _(marketing_df):
    marketing_df.isna().value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Nema nedostajućih vrijednosti.""")
    return


@app.cell
def _(marketing_df):
    marketing_df.duplicated().value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Nema ni duplikata.""")
    return


@app.cell
def _(marketing_df):
    marketing_df.boxplot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### U stupcu `'Newspaper'` imamo dva outliera.""")
    return


@app.cell
def _(marketing_df):
    marketing_df[marketing_df['Newspaper'] > 100]
    return


@app.cell
def _(marketing_df, sns):
    marketing_df_corr = marketing_df.corr()
    sns.heatmap(data=marketing_df_corr, annot=True)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Iz matrice korelacije se da zaključiti da `'Sales'` najviše ovisi o `'TV'`.""")
    return


@app.cell
def _(StandardScaler, marketing_df, train_test_split):
    X_train, X_test, y_train, y_test = train_test_split(marketing_df[['TV', 'Radio', 'Newspaper']], marketing_df['Sales'], random_state=7,test_size=0.2)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_test, X_train, linear_model, metrics, np, pd, plt, y_test, y_train):
    model = linear_model.LinearRegression()
    model.fit(X=X_train, y=y_train)

    y_pred = model.predict(X_test)
    df_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    df_results.sort_values(by='Actual').plot(kind='bar', figsize=(10,6))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_test, y_pred)
    print('Mean Absolute Error (MAE):', mae)
    print('Mean Squared Error (MSE):', mse)
    print('Root Mean Squared Error (RMSE):', rmse)
    print('R² Score:', r2)

    return (df_results,)


@app.cell
def _(df_results, plt, sns):
    df_results['Error'] = df_results['Actual'] - df_results['Predicted']
    plt.figure(figsize=(10,6))
    sns.histplot(df_results['Error'], bins=10, kde=True)
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
