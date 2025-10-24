import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    # sales_prediction.py

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    return LinearRegression, metrics, pd, plt, sns, train_test_split


@app.cell
def _(pd):
    # ----------------------------
    # 1. Load the dataset
    # ----------------------------
    data = pd.read_csv("marketing.csv")

    # Display first few rows
    print("Dataset preview:")
    print(data.head(), "\n")
    return (data,)


@app.cell
def _(data):
    # ----------------------------
    # 2. Define features (X) and target (y)
    # ----------------------------
    X = data[['TV', 'Radio', 'Newspaper']]  # independent variables
    y = data['Sales']                       # dependent variable
    return X, y


@app.cell
def _(X, train_test_split, y):
    # ----------------------------
    # 3. Split the dataset into training and testing sets
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, X_train, y_test, y_train


@app.cell
def _(LinearRegression, X_train, y_train):
    # ----------------------------
    # 4. Create and train the model
    # ----------------------------
    model = LinearRegression()
    model.fit(X_train, y_train)
    return (model,)


@app.cell
def _(X_test, model):
    # ----------------------------
    # 5. Make predictions
    # ----------------------------
    y_pred = model.predict(X_test)
    return (y_pred,)


@app.cell
def _(metrics, model, y_pred, y_test):
    # ----------------------------
    # 6. Evaluate the model
    # ----------------------------
    r2 = metrics.r2_score(y_test, y_pred)
    print(f"R-squared value of the model: {r2 * 100:.2f}")

    # Optional: print coefficients
    print("\nModel Coefficients:")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"TV Coefficient: {model.coef_[0]:.4f}")
    print(f"Radio Coefficient: {model.coef_[1]:.4f}")
    print(f"Newspaper Coefficient: {model.coef_[2]:.4f}")
    return


@app.cell
def _(plt, sns, y_pred, y_test):
    # ----------------------------
    # 7. (Optional) Visualization
    # ----------------------------
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales")
    plt.grid(True)
    plt.show()
    return


@app.function
def analyze_and_compare_models(data, y_test, y_pred):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics

    # --- Analiza podataka ---
    print("🔹 Nedostajuće vrijednosti po stupcima:")
    print(data.isna().sum(), "\n")

    print(f"🔹 Broj duplikata: {data.duplicated().sum()}\n")

    plt.figure(figsize=(12, 4))
    for i, col in enumerate(['TV', 'Radio', 'Newspaper', 'Sales']):
        plt.subplot(1, 4, i + 1)
        sns.boxplot(y=data[col], color='lightblue')
        plt.title(col)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.histplot(data['Sales'], kde=True, bins=20, color='lightgreen')
    plt.title("Distribucija prodaje (Sales)")
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Korelacijska matrica")
    plt.show()

    sns.pairplot(data)
    plt.show()

    # --- Usporedba modela ---
    r2_multi = metrics.r2_score(y_test, y_pred)
    print(f"📊 R² višestrukog modela (TV + Radio + Newspaper): {r2_multi * 100:.2f}%")

    X_tv = data[['TV']]
    y_target = data['Sales']

    X_train_tv, X_test_tv, y_train_tv, y_test_tv = train_test_split(X_tv, y_target, test_size=0.2, random_state=42)
    model_tv = LinearRegression()
    model_tv.fit(X_train_tv, y_train_tv)
    y_pred_tv = model_tv.predict(X_test_tv)

    r2_tv = metrics.r2_score(y_test_tv, y_pred_tv)
    print(f"📉 R² jednostavnog modela (samo TV): {r2_tv * 100:.2f}%")

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred, label='Višestruka regresija', color='blue')
    sns.scatterplot(x=y_test_tv, y=y_pred_tv, label='Jednostavna regresija (TV)', color='red')
    plt.xlabel("Stvarne vrijednosti prodaje")
    plt.ylabel("Predviđene vrijednosti")
    plt.title("Usporedba višestrukog i jednostavnog modela")
    plt.legend()
    plt.show()


@app.cell
def _(data, y_pred, y_test):
    analyze_and_compare_models(data, y_test, y_pred)

    return


if __name__ == "__main__":
    app.run()
