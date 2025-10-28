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
    from sklearn.ensemble import RandomForestRegressor
    from sklearn import metrics
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    # Load data
    data = pd.read_csv('/workspaces/2025-sci-prog/labs/03/marketing.csv')
    data.fillna(data.mean(), inplace=True)
    data.drop_duplicates(inplace=True)

    # RandomForestRegressor
    X = data.drop('Sales', axis=1)
    y = data['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_test, y_pred)
    print(f'MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2:{r2}')
    print(f'Feature Importances: {model.feature_importances_}')
    
    # Actual vs Predicted plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.show()

    # Linear vs Polynomial plot
    X_tv = data[['TV']]
    y_tv = data['Sales']
    linear_model = LinearRegression()
    linear_model.fit(X_tv, y_tv)
    y_linear_pred = linear_model.predict(X_tv)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X_tv)
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y_tv)
    X_fit = np.arange(X_tv.min().iloc[0], X_tv.max().iloc[0], 1)[:, np.newaxis]
    y_poly_pred = poly_model.predict(poly.transform(X_fit))
    plt.figure(figsize=(10, 6))
    plt.scatter(X_tv, y_tv, label='Actual Data')
    plt.plot(X_tv, y_linear_pred, color='red', label='Linear Regression')
    plt.plot(X_fit, y_poly_pred, color='green', label='Polynomial Regression (d=2)')
    plt.xlabel('TV Advertising')
    plt.ylabel('Sales')
    plt.title('Linear vs Polynomial Regression')
    plt.legend()
    plt.show()

    # Metrics comparison plot
    metrics_data = {
        'Step': ['Baseline', 'Fillna/DropDup', 'Standardize', 'Polynomial', 'RandomForest'],
        'MAE': [1.2748, 1.2748, 1.2748, 0.9034, 0.9180],
        'MSE': [2.9077, 2.9077, 2.9077, 1.4425, 1.4374],
        'RMSE': [1.7052, 1.7052, 1.7052, 1.2011, 1.1989],
        'R2': [0.9059, 0.9059, 0.9059, 0.9533, 0.9535]
    }
    df_metrics = pd.DataFrame(metrics_data)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Metrics Comparison')
    sns.barplot(x='Step', y='MAE', data=df_metrics, ax=axes[0, 0])
    axes[0, 0].set_title('Mean Absolute Error (MAE)')
    sns.barplot(x='Step', y='MSE', data=df_metrics, ax=axes[0, 1])
    axes[0, 1].set_title('Mean Squared Error (MSE)')
    sns.barplot(x='Step', y='RMSE', data=df_metrics, ax=axes[1, 0])
    axes[1, 0].set_title('Root Mean Squared Error (RMSE)')
    sns.barplot(x='Step', y='R2', data=df_metrics, ax=axes[1, 1])
    axes[1, 1].set_title('R-squared (R2)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return


if __name__ == "__main__":
    app.run()
