import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    # === 1. Uvoz biblioteka ===
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    import os

    # === 2. Učitavanje podataka ===
    os.chdir("/workspaces/2025-sci-prog/labs/03/")
    df = pd.read_csv('marketing.csv')

    # === 3. Osnovni prikaz ===
    print("\nPrvih 5 redaka:\n", df.head())

    # === 4. Provjera nedostajućih vrijednosti ===
    print("\nNedostajuće vrijednosti (isna):\n", df.isna().sum())

    # === 5. Provjera duplikata ===
    print("\nBroj duplikata:", df.duplicated().sum())

    # === 6. Boxplot za detekciju stršećih vrijednosti (outlieri) ===
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df)
    plt.title("Boxplot – provjera stršećih vrijednosti (outlieri)")
    plt.show()

    # === 7. Distribucija ciljne varijable (Sales) ===
    plt.figure(figsize=(8,5))
    sns.histplot(df['Sales'], kde=True)
    plt.title("Distribucija ciljne varijable (Sales)")
    plt.xlabel("Sales")
    plt.ylabel("Frekvencija")
    plt.show()

    # === 8. Korelacija između varijabli ===
    plt.figure(figsize=(6,4))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Korelacijska matrica")
    plt.show()

    # === 9. Pairplot (odnosi među varijablama) ===
    sns.pairplot(df)
    plt.suptitle("Parni odnosi između varijabli", y=1.02)
    plt.show()

    # === 10. Višestruka linearna regresija ===
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']

    model = LinearRegression()
    model.fit(X, y)

    r_squared = model.score(X, y)

    print("\n=== REZULTATI MODELA ===")
    print(f"R² vrijednost modela: {r_squared * 100:.2f}%")
    print(f"Intercept (β₀): {model.intercept_:.3f}")
    print(f"Koeficijenti (β): {model.coef_}")

    # Jednadžba modela
    print(f"\nJednadžba modela:")
    print(f"Sales = {model.intercept_:.3f} + "
          f"{model.coef_[0]:.3f}*TV + "
          f"{model.coef_[1]:.3f}*Radio + "
          f"{model.coef_[2]:.3f}*Newspaper")

    return


if __name__ == "__main__":
    app.run()
