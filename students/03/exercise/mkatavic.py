import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('students/03/data/18_monthly_sales.csv')

# Separate features (X) and target (y)
X = data[['customer_visits']]  # independent variable
y = data['monthly_sales']      # dependent variable

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Get slope and intercept
slope = model.coef_[0]
intercept = model.intercept_

# Predict sales for the given X
y_pred = model.predict(X)

# Plot results
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('Customer Visits')
plt.ylabel('Monthly Sales')
plt.title('Customer Visits vs Monthly Sales')
plt.legend()
plt.grid(True)
plt.show()
