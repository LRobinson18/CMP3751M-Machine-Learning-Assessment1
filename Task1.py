# Import all relevant libraries and modules
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data from the csv file
data = pd.read_csv('Task1 - dataset - pol_regression.csv')

# Split data into x and y
x_data = data['x'].sort_values()
y_data = data['y'].sort_values()

# Split the dataset 70% train, 30% test
x_train = x_data[0:math.ceil(len(x_data) * 0.7)] 
y_train = y_data[0:math.ceil(len(x_data) * 0.7)] 

x_test = x_data[0:math.floor(len(x_data) * 0.3)] 
y_test = y_data[0:math.floor(len(x_data) * 0.3)] 

# Calculate feature expansion up to a certain degree for a given data set x
def getPolynomialDataMatrix(x, degree):
    # Initialise matrix using the shape of the data given
    X = np.ones(x.shape)

    # For specified dth order
    for i in range(1,degree + 1):
        # Construct the matrix
        X = np.column_stack((X, x ** i))

    return X

# Calculate parameters given the data x/y and degree
def pol_regression(features_train,  y_train, degree):
    # Use previous definition to get a matrix from the data
    X = getPolynomialDataMatrix(features_train, degree)

    # Transpose (inverse) the matrix
    XX = X.transpose().dot(X)

    # Solve simultaneous matrixes, returns matrix of x coefficients 
    parameters = np.linalg.solve(XX, X.transpose().dot(y_train))

    return parameters

# Evaluate the polynomial using RMSE
def eval_pol_regression(parameters, x_data, y_data, degree):
    # Squared Error (SE)
    se = 0
    # Mean Squared Error (MSE)
    mse = 0

    # For each item in the x/y datasets
    for x, y in zip(x_data, y_data):
        y2 = 0

        # For each degree ran until 0
        for i in range(degree, -1, -1):

            # Add the previous x parameters to get the predicted y value
            y2 = y2 + (parameters[i] * (x ** i))

        # Subtract the y data from the predicted y data then square
        se = se + ((y2 - y) ** 2)

    # Calculate MSE by dividing total squared error by number of data points
    mse = se / len(x_data)

    # Square root to get Root Mean Squared Error (RMSE)
    rmse = math.sqrt(mse)
    return rmse     

# Clear the current plot
plt.clf()
# Create new matplotlib frame and subplots
plt.subplot(121)
# Plot the data points
plt.plot(x_data,y_data, 'ko')

# Plot polynomial with degree 0 (x^0)
plt.axhline(y=np.mean(y_data), color='r', linestyle='-')

degrees = [1, 2, 3, 6, 10]
colours = ['b','g','#00eaff','#fc00ff','#fff600','#9600ff','#0090ff']
rmse_train = []
rmse_test = []

# Regress polynomials with degrees in list
for d in degrees:
    # Run polynomial regression with training data
    parameters = pol_regression(x_data, y_data, d)

    xtestD = getPolynomialDataMatrix(x_data, d)
    ytestD = xtestD.dot(parameters)

    # Plot the specific polynomial graph
    plt.subplot(121)
    plt.plot(x_data, ytestD, colours[degrees.index(d)])

    # Calculate training data RMSE
    rmse = eval_pol_regression(parameters, x_train, y_train, d)
    rmse_train.append(rmse)
    
    # Calculate test data RMSE
    rmse = eval_pol_regression(parameters, x_test, y_test, d)
    rmse_test.append(rmse)

print(rmse_train,"\n",rmse_test)

plt.subplot(122)
# Plot training RMSE in blue
plt.plot(degrees, rmse_train, 'b')
# Plot test RMSE in red
plt.plot(degrees, rmse_test, 'r')
plt.legend(('Train RMSE', 'Test RMSE'), loc = 'upper right')
plt.xlabel('Degree of Polynomial') 
plt.ylabel('RMSE')

plt.subplot(121)
plt.legend(('training points', '$x^0$', '$x^1$', '$x^2$', '$x^3$', '$x^6$', '$x^{10}$'), loc = 'lower right')
plt.savefig('polynomial_regression.png')
plt.show()