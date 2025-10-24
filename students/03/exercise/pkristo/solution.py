import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# 1️⃣ Učitavanje podataka
df = pd.read_csv("../../data/30_battery_life.csv")

# 2️⃣ Priprema varijabli
X = df[['usage_time_per_day']]  # neovisna varijabla
y = df['battery_life']          # ovisna varijabla

# 3️⃣ Treniranje linearnog modela
model = LinearRegression()
model.fit(X, y)

# 4️⃣ Ispis rezultata modela
print("Koeficijent (slope):", model.coef_[0])
print("Presjek (intercept):", model.intercept_)

# 5️⃣ Vizualizacija
plt.scatter(X, y, color='blue', label='Podaci')
plt.plot(X, model.predict(X), color='red', label='Linearni model')
plt.xlabel("Usage time per day (h)")
plt.ylabel("Battery life (h)")
plt.title("Linearna regresija – Battery Life")
plt.legend()
plt.show()


hours = 5
hours_array = np.array([[float(hours)]])
pred = model.predict(hours_array)

print(
    f"Predviđeni vijek baterije za {hours}h korištenja dnevno: {pred[0]:.2f}")
