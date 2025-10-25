import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load the data
data = pd.read_csv('../../data/15_fuel_efficiency.csv')

# Remove any rows with missing values
data = data.dropna()

print("Dataset shape:", data.shape)
print("\nFirst few rows:")
print(data.head())
print("\nDataset statistics:")
print(data.describe())

# Prepare the data
# Feature: vehicle weight (kg)
# Target: fuel efficiency (e.g., km/l or mpg)
X = data[['vehicle_weight']].values
y = data['fuel_efficiency'].values

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions on the training data
y_pred = model.predict(X)

# Calculate performance metrics
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print(f"\nLinear Regression Results:")
print(
    f"Coefficient (slope): {model.coef_[0]:.4f}  (change in efficiency per 1 kg)")
print(f"Intercept: {model.intercept_:.4f}")
print(f"R-squared: {r2:.4f}")
print(f"RMSE: {rmse:.4f}  (same units as 'fuel_efficiency')")

# Visualization
# Sort values for a smooth regression line
order = np.argsort(X.flatten())
X_sorted = X[order]
y_pred_sorted = y_pred[order]

plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7, label='Actual Data')
plt.plot(X_sorted, y_pred_sorted, color='red',
         linewidth=2, label='Linear Regression')
plt.xlabel('Vehicle Weight (kg)')
plt.ylabel('Fuel Efficiency')
plt.title('Fuel Efficiency vs Vehicle Weight')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Make predictions for new weights
new_weights = np.array([[1000], [1250], [1500], [1750], [2000]])
new_predictions = model.predict(new_weights)

print(f"\nPredictions for new vehicle weights:")
for w, fe in zip(new_weights.flatten(), new_predictions):
    print(f"Weight: {w} kg -> Predicted Fuel Efficiency: {fe:.2f}")

# Interpretation
print("\nInterpretation:")
print(
    f"- For each +1 kg increase in vehicle weight, fuel efficiency changes by {model.coef_[0]:.4f} (usually negative).")
print(
    f"- The model explains about {r2*100:.1f}% of the variance in fuel efficiency.")
print(
    f"- RMSE of {rmse:.2f} indicates the average prediction error in the same units as the target.")
