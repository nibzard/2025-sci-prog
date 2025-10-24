import marimo as mo
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load the data
data = pd.read_csv('students/03/data/03_advertising_sales.csv')
data.dropna()

print("Dataset shape:", data.shape)
print("\nFirst few rows:")
print(data.head())

print("\nDataset statistics:")
print(data.describe())

#Prepare the data
x = data[['advertising_spending']].values  # Features (Advertising Spending)
y = data['total_sales'].values   # Target (Total sales)

model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(x)

# Calculate metrics
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print(f"\nLinear Regression Results:")
print(f"Coefficient (slope): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"R-squared: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")

# Create visualization
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.7, label='Actual Data')
plt.plot(x, y_pred, color='red', linewidth=2, label='Linear Regression')
plt.xlabel('Advertising Spending ($)')
plt.ylabel('Total Sales')
plt.title('Advertising Spending vs Total Sales')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
