import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math

# Provided data
x_data = np.array([16,16,18,28,18])
x_uncertainty = np.array([.5,.5,.5,.5,.5])
y_data = np.array([5600,5000,6800,8200,5700])
y_uncertainty = np.array([141,141,141,141,141])

for i in range(len(x_uncertainty)):
    x_uncertainty[i] = x_uncertainty[i]*4

for i in range(len(y_uncertainty)):
    y_uncertainty[i] = 700

# Define the linear model function
def linear_model(params, x):
    slope, intercept = params
    return slope * x + intercept

# Define the weighted least squares error function
def wls_error(params, x, y, x_uncertainty, y_uncertainty):
    predicted_y = linear_model(params, x)
    weights = 1 / (x_uncertainty**2 + y_uncertainty**2)
    error = np.sum(weights * (y - predicted_y)**2)
    return error

# Initial guess for the parameters
initial_guess = [1.0, 0.0]

# Minimize the error to find the best-fit parameters
result = minimize(wls_error, initial_guess, args=(x_data, y_data, x_uncertainty, y_uncertainty))

# Extract the best-fit parameters
best_fit_slope, best_fit_intercept = result.x

# Plot the data and the best-fit line
plt.errorbar(x_data, y_data, xerr=x_uncertainty, yerr=y_uncertainty, fmt='o', label='Data with Uncertainty')
plt.plot(x_data, linear_model([best_fit_slope, best_fit_intercept], x_data), label='Best Fit Line', color='green')

plt.xlabel('Number of Fringes')
plt.ylabel('Distance Moved (nanometers)')
plt.legend()
plt.title('Linear Regression with Uncertainty')
plt.grid(True)
plt.show()

# Print the best-fit parameters
print(f'Best-fit slope: {best_fit_slope}')
print(f'Best-fit intercept: {best_fit_intercept}')

ycomposite = []
for xuncertain, yuncertain in zip(x_uncertainty, y_uncertainty):
    ycomposite.append(math.sqrt(((xuncertain * best_fit_slope)**2) + (yuncertain**2)))

print("Composite uncertainties")
print(ycomposite)

fit_points = []
for xpoint in x_data:
    fit_points.append((best_fit_slope*xpoint) + best_fit_intercept)

print("y points on the line of best fit corresponding with xpoints")
print(fit_points)

chi_squared = 0
for uncertain_point, fitted_point, ypoint in zip(ycomposite, fit_points, y_data):
    chi_squared += ((fitted_point - xpoint)**2)/(uncertain_point**2)

print("chi squared")
print(chi_squared)