
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('FuelConsumption.csv')
print(df.head())



def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        predictions = X.dot(theta)
        error = predictions - y
        gradient = (1/m) * X.T.dot(error)
        theta -= alpha * gradient
        cost = (1/(2*m)) * np.sum(error ** 2)
        cost_history.append(cost)
        if len(cost_history) > 1 and abs(cost_history[-1] - cost_history[-2]) < 1e-6:
            break
    return theta, cost_history
def evaluate(X, y, theta):
    predictions = X.dot(theta)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    return mse, r2




print("\n--- Univariate Linear Regression ---")
X_uni = df[['ENGINESIZE']].values
y_uni = df[['CO2EMISSIONS']].values
X_uni = np.c_[np.ones(X_uni.shape[0]), X_uni]
X_train_uni, X_test_uni, y_train_uni, y_test_uni = train_test_split(X_uni, y_uni, test_size=0.2, random_state=4)
theta_uni = np.zeros((X_uni.shape[1], 1))
alpha = 0.01
iterations = 1000
theta_uni_final, cost_history_uni = gradient_descent(X_train_uni, y_train_uni, theta_uni, alpha, iterations)



print(f"Hypothesis: h(x) = {theta_uni_final[0][0]:.2f} + {theta_uni_final[1][0]:.2f} * EngineSize")



mse_uni, r2_uni = evaluate(X_test_uni, y_test_uni, theta_uni_final)
print(f"MSE: {mse_uni:.2f}, R²: {r2_uni:.2f}")




plt.plot(cost_history_uni)
plt.title("Univariate Cost Convergence")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.grid(True)
plt.show()













# ========== STEP 4: MULTIVARIATE LINEAR REGRESSION ==========
print("\n--- Multivariate Linear Regression ---")
X_multi = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']].values
y_multi = df[['CO2EMISSIONS']].values
X_multi = np.c_[np.ones(X_multi.shape[0]), X_multi]
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
theta_multi = np.zeros((X_multi.shape[1], 1))
theta_multi_final, cost_history_multi = gradient_descent(X_train_multi, y_train_multi, theta_multi, alpha, iterations)

# Print hypothesis
print("Hypothesis: h(x) = {:.2f} + {:.2f}*ENGINESIZE + {:.2f}*CYLINDERS + {:.2f}*FUELCONSUMPTION_COMB"
      .format(*theta_multi_final.flatten()))
# Accuracy
mse_multi, r2_multi = evaluate(X_test_multi, y_test_multi, theta_multi_final)
print(f"MSE: {mse_multi:.2f}, R²: {r2_multi:.2f}")
# Plot cost
plt.plot(cost_history_multi)
plt.title("Multivariate Cost Convergence")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.grid(True)
plt.show()










