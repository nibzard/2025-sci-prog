from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('students/03/data/12_stopping_distance.csv')
print(df)

X = df['speed'].values.reshape(-1, 1)
y = df['stopping_distance'].values

model = LinearRegression()
model.fit(X, y)

print(f"\nIntercept: {model.intercept_:.2f}")
print(f"Coefficient: {model.coef_[0]:.2f}")

y_pred = model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Stvarni podaci')
plt.plot(X, y_pred, color='red', label='Linear regression')
plt.xlabel('Speed (km/h)')
plt.ylabel('Stopping Distance (m)')
plt.title('Stopping Distance vs Speed: Linear Regression')
plt.legend()
plt.grid(True)
plt.show()

r_squared = model.score(X, y)
print(f"R-squared: {r_squared:.4f}")

new_speeds = np.array([40, 60, 80, 100]).reshape(-1, 1)
predicted_distances = model.predict(new_speeds)
print(f"\nPredicted stopping distances for speeds 40, 60, 80, 100 km/h:")
for speed, dist in zip(new_speeds.flatten(), predicted_distances):
    print(f"  Speed {speed} km/h: {dist:.2f} m")
