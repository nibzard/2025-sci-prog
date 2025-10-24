import pandas as pd
from sklearn.linear_model import LinearRegression

# Correct relative path from solution.py
df = pd.read_csv("../../data/15_fuel_efficiency.csv")

X = df[["vehicle_weight"]]
y = df["fuel_efficiency"]

model = LinearRegression().fit(X, y)

weight = 1500
pred = model.predict([[weight]])[0]

print(f"Predicted fuel efficiency for {weight} kg: {pred:.2f}")
