import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

data = pd.read_csv("students/03/data/27_recovery_time.csv")

X = data[["medication_dosage"]]
y = data["recovery_time"]

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

model = LinearRegression()
model.fit(X, y)

print("\nModel trained successfully.")
print(f"\nCoefficient (slope): {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

y_pred = model.predict(X)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print(f"\nR² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(X, y_pred, color="red", label="Linear regression")
plt.xlabel("Medication Dosage")
plt.ylabel("Recovery Time")
plt.title("Medication Dosage vs Recovery Time - Linear Regression")
plt.legend()
plt.grid(True)
plt.show()

print("\n" + "=" * 60)
print("SAMPLE PREDICTION")
print("=" * 60)

sample_dosage = 5.0
sample_prediction = model.predict(np.array([[sample_dosage]]))
print(f"\nFor a medication dosage of {sample_dosage}:")
print(f"Predicted recovery time: {sample_prediction[0]:.2f}")
