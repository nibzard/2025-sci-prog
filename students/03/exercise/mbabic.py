import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import io

# 1. Load the dataset
try:
    df = pd.read_csv('students/03/data/06_crop_yield.csv')

    # 2. Define the input (X) and output (y)
    X = df[['fertilizer_amount']]  # Needs to be 2D
    y = df['crop_yield']           # This is 1D

    # 3. Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # 4. Make predictions (for plotting or evaluation)
    df['predicted_yield'] = model.predict(X)

    # 5. Calculate evaluation metrics
    mse = mean_squared_error(y, df['predicted_yield'])
    rmse = np.sqrt(mse)
    r2 = r2_score(y, df['predicted_yield'])

    # 6. Print model parameters and evaluation metrics
    print(f"\nLinear Regression Results:")
    print(f"Coefficient (slope): {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"R-squared: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")

    # 7. Make predictions for the existing fertilizer amounts
    existing_predictions = model.predict(X)

    # 8. Interpretation
    print(f"\nInterpretation:")
    print(
        f"- For every 1 unit increase in fertilizer amount, crop yield increases by {model.coef_[0]:.2f}")
    print(f"- The model explains {r2*100:.1f}% of the variance in crop yield")
    print(f"- Root Mean Square Error: {rmse:.2f}")

    # 9. Plot actual vs predicted
    plt.scatter(df['fertilizer_amount'], df['crop_yield'],
                color='blue', label='Actual')
    plt.plot(df['fertilizer_amount'], df['predicted_yield'],
             color='red', label='Regression Line')
    plt.xlabel('Fertilizer Amount')
    plt.ylabel('Crop Yield')
    plt.title('Linear Regression: Crop Yield vs Fertilizer Amount')
    plt.legend()
    plt.show()
except FileNotFoundError:
    print("Error: '06_crop_yield.csv' not found. Please make sure the file is in the correct directory.")
