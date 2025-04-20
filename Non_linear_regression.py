import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv('china_gdp.csv')
print(df.head())




x_data = df["Year"].values
y_data = df["Value"].values




x = x_data / max(x_data)
y = y_data / max(y_data)


def sigmoid(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))




def gradient_descent(x, y, L, k, x0, alpha, iterations):
    m = len(x)
    cost_history = []
    for i in range(iterations):
        y_pred = sigmoid(x, L, k, x0)
        error = y_pred - y
        cost = np.sum(error ** 2) / (2 * m)
        cost_history.append(cost)




        dL = np.sum(error * (1 / (1 + np.exp(-k * (x - x0))))) / m
        dk = np.sum(error * L * (x - x0) * np.exp(-k * (x - x0)) / ((1 + np.exp(-k * (x - x0))) ** 2)) / m
        dx0 = np.sum(error * L * k * np.exp(-k * (x - x0)) / ((1 + np.exp(-k * (x - x0))) ** 2)) / m



        L -= alpha * dL
        k -= alpha * dk
        x0 -= alpha * dx0



        if i > 1 and abs(cost_history[-1] - cost_history[-2]) < 1e-6:
            break
    return L, k, x0, cost_history



L = 1
k = 1
x0 = 0.5
alpha = 0.1
iterations = 10000
L_final, k_final, x0_final, cost_history = gradient_descent(x, y, L, k, x0, alpha, iterations)
print(f"Optimal parameters: L={L_final}, k={k_final}, x0={x0_final}")



x_range = np.linspace(min(x), max(x), 100)
y_pred = sigmoid(x_range, L_final, k_final, x0_final)



plt.scatter(x_data, y_data, label='Original Data')
plt.plot(x_range * max(x_data), y_pred * max(y_data), color='red', label='Fitted Logistic Curve')
plt.xlabel("Year")
plt.ylabel("GDP")
plt.legend()
plt.title("China GDP Non-linear Regression")
plt.grid(True)
plt.show()











from sklearn.metrics import mean_squared_error, r2_score
y_fit = sigmoid(x, L_final, k_final, x0_final)
mse = mean_squared_error(y, y_fit)
r2 = r2_score(y, y_fit)
print(f"MSE: {mse:.6f}")
print(f"R² Score: {r2:.6f}")
