import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load the data
data = pd.read_csv('students/03/data/08_electricity_cost.csv')

# Remove any rows with missing values
data = data.dropna()

print("Dataset shape:", data.shape)
print("\nFirst few rows:")
print(data.head())

print("\nDataset statistics:")
print(data.describe())

# Prepare the data
X = data[['avg_temperature']].values  # Features (temperature)
y = data['electricity_cost'].values   # Target (electricity cost)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

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
plt.scatter(X, y, alpha=0.7, label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Linear Regression')
plt.xlabel('Average Temperature (°C)')
plt.ylabel('Electricity Cost ($)')
plt.title('Electricity Cost vs Average Temperature')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Make some predictions for new temperatures
new_temperatures = np.array([[0], [10], [20], [30], [40]])
predictions = model.predict(new_temperatures)

print(f"\nPredictions for new temperatures:")
for temp, cost in zip(new_temperatures.flatten(), predictions):
    print(f"Temperature: {temp}°C -> Predicted Cost: ${cost:.2f}")

# Interpretation
print(f"\nInterpretation:")
print(f"- For every 1°C increase in temperature, electricity cost decreases by ${abs(model.coef_[0]):.2f}")
print(f"- The model explains {r2*100:.1f}% of the variance in electricity costs")
print(f"- Root Mean Square Error: ${rmse:.2f}")
