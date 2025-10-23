# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load the ice cream sales data
data = pd.read_csv('../data/01_ice_cream_sales.csv')

# Extract features (temperature) and target (sales)
X = data[['temperature']]
y = data['sales']

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate metrics
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

# Print results
print("Linear Regression Results")
print("=" * 40)
print(f"Slope (coefficient): {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"R-squared Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"\nEquation: sales = {model.coef_[0]:.4f} * temperature + {model.intercept_:.4f}")

# Create visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.6, label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Temperature (degrees C)')
plt.ylabel('Ice Cream Sales')
plt.title('Ice Cream Sales vs Temperature - Linear Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Make example predictions
print("\nExample Predictions:")
print("-" * 40)
example_temps = [20, 25, 30]
for temp in example_temps:
    predicted_sales = model.predict([[temp]])[0]
    print(f"Temperature {temp} degrees C -> Predicted Sales: {predicted_sales:.2f}")
