import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

file_path = '/workspaces/2025-sci-prog/students/03/data/10_maintenance_cost.csv'

data = pd.read_csv(file_path)

X = data['vehicle_age'].values.reshape(-1, 1)
y = data['maintenance_cost'].values

model = LinearRegression()

model.fit(X, y)

y_pred = model.predict(X)


plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', label='Linear regression line')
plt.xlabel('Vehicle Age')
plt.ylabel('Maintenance Cost')
plt.title('Vehicle Age vs. Maintenance Cost')
plt.legend()
plt.show()

print(f'Intercept: {model.intercept_}')
print(f'Coefficient (Slope): {model.coef_}')
