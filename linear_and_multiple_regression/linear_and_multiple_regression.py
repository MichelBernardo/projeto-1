import numpy as np
import matplotlib.pyplot as plt


'''
Linear and multiple regression based on the least squares method
'''

data = np.loadtxt(r'C:\Users\miche\Documents\Repositories\python-machine_learning\simple_and_multiple_regression\aerogerador.dat')  # loads the database

X = data[:,0]  # Speed vector
Y = data[:,1]  # Power vector
X_mean = np.mean(X)  # Speed vector mean
Y_mean = np.mean(Y)  # Power vector mean

# Plots the database
plt.plot(X, Y, marker='*',linestyle='None')
plt.xlabel('Speed')
plt.ylabel('Power')
plt.title('Wind turbine')
plt.grid(True)
plt.show()

# Simple Regression
Beta_1 = (np.sum(X*Y) - Y_mean*np.sum(X))/(np.sum(X**2) - X_mean*np.sum(X))  # Slope
Beta_0 = Y_mean - Beta_1*X_mean  # Linear coefficient

Y_simple = Beta_0 + Beta_1*X  # model output (prediction)

# Plotting the prediction
plt.plot(X, Y_simple, marker='None',linestyle='-', color='r')
plt.plot(X, Y, marker='*',linestyle='None')
plt.xlabel('Speed')
plt.ylabel('Power')
plt.title('Wind turbine - Simple Regression')
plt.grid(True)
plt.show()

# Multiple Regression
X = np.array(X)
degrees_of_the_polynomial = 5  # Degrees of the regression polynomial

polynomial_matrix = np.column_stack([X ** i for i in range(1, degrees_of_the_polynomial)])
XX = np.hstack((np.ones((len(X), 1)), polynomial_matrix))

Beta = np.linalg.inv(XX.transpose()@XX)@XX.transpose()@Y  # Calculation of regression coefficients

Y_multiple = Beta[0] + np.dot(XX[:, 1:], Beta[1:])  # model output (prediction)

# Plotting the prediction
plt.plot(X, Y_multiple, marker='None',linestyle='-', color='r')
plt.plot(X, Y, marker='*',linestyle='None')
plt.xlabel('Speed')
plt.ylabel('Power')
plt.title('Wind turbine - Multiple Regression')
plt.grid(True)
plt.show()