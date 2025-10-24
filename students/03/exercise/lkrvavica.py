import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    return LinearRegression, mo, pd, plt


@app.cell
def _(pd):
    # Učitavanje podataka
    df = pd.read_csv(
        '/workspaces/2025-sci-prog/students/03/data/17_student_gpa.csv'
    )
    df.head()
    return (df,)


@app.cell
def _(LinearRegression, df):
    # Priprema i treniranje modela
    X = df[['attendance_rate']].values
    y = df['gpa'].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    return X, y, y_pred, model


@app.cell
def _(X, y, y_pred, mo, plt):
    # Crtanje rezultata
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', label='Stvarni GPA')
    ax.plot(X, y_pred, color='red', linewidth=2, label='Regresijska linija')
    ax.set_title('GPA vs. Stopa dolaznosti')
    ax.set_xlabel('Stopa dolaznosti (%)')
    ax.set_ylabel('GPA')
    ax.legend()
    ax.grid(True)

    mo.md("# Model linearne regresije: GPA vs. Dolaznost")
    mo.mpl.interactive(fig)
    return


if __name__ == "__main__":
    app.run()
