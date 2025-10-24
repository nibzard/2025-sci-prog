from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('students/03/data/11_plant_growth.csv')
print(df)

X = df['days_since_planting'].values.reshape(-1, 1)
y = df['height'].values

model = LinearRegression()
model.fit(X, y)

print(f"\nIntercept: {model.intercept_:.2f}")
print(f"Coefficient: {model.coef_[0]:.2f}")

y_pred = model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Linear regression')
plt.xlabel('Days since planting')
plt.ylabel('Height (cm)')
plt.title('Plant Growth: Linear Regression')
plt.legend()
plt.grid(True)
plt.show()

r_squared = model.score(X, y)
print(f"R-squared: {r_squared:.4f}")

new_days = np.array([50, 60, 70]).reshape(-1, 1)
predicted_heights = model.predict(new_days)
print(f"\nPredicted heights for days 50, 60, 70:")
for days, height in zip(new_days.flatten(), predicted_heights):
    print(f"  Day {days}: {height:.2f} cm")
