import os
import scipy.io
import argparse
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-file', default='ex5data1.mat', type=str,  help='File name.')
parser.add_argument('-alpha', default=568, type=float, help='Regularization parameter.')
parser.add_argument('-degree', default=3, type=int, help='Polynomial degree.')
args = parser.parse_args()


os.chdir('C:/Users/hecto/OneDrive/Documents/Coursera Machine Learning Assignments/machine-learning-ex5/ex5')
# Import and extract the data
data = scipy.io.loadmat(args.file)

X, Xtest, Xval = data['X'], data['Xtest'], data['Xval']
y, ytest, yval = data['y'], data['ytest'], data['yval']


def error_function(parameter, p_min, p_max):
    """
    Gets the error values for changes in the specified parameter.
    :param parameter: string options are: 'm' for length of test set;
    'alpha' for regularization parameter; 'degree' for polynomial degree.
    :param p_min: Parameter start point.
    :param p_max: Parameter end point.
    :return: Arrays of the parameter values, their respective errors, and the parameter label for future plotting.
    """
    curve, curve_val = [], []  # Lists will hold the arrays.
    p = {'m': len(X), 'alpha': args.alpha, 'degree': args.degree}  # Dictionary to hold parameters.

    for p[parameter] in range(p_min, p_max):
        # Solver 'svd' makes feature normalization unnecessary.
        # Fit using mapped features with polynomial value 'degree.'
        reg = linear_model.Ridge(solver='svd', alpha=p['alpha'])
        reg.fit(PolynomialFeatures(p['degree']).fit_transform(X[:p['m']]), y[:p['m']])

        y_pred = reg.predict(PolynomialFeatures(p['degree']).fit_transform(X[:p['m']]))
        yval_pred = reg.predict(PolynomialFeatures(p['degree']).fit_transform(Xval))

        err = metrics.mean_squared_error(y[:p['m']], y_pred) / (2 * p['m'])  # Training set cost.
        err_val = metrics.mean_squared_error(yval, yval_pred) / (2 * len(yval))  # Cross-validation set cost.

        curve.append([p[parameter], err])
        curve_val.append([p[parameter], err_val])

    return {'vectors': [np.array(curve).T, np.array(curve_val).T], 'parameter': parameter}


def parameter_plot(vectors):
    """
    Plots the error as a function of the specified parameter.
    :param vectors: Dictionary of arrays and the parameter label.
    :return: Plot of error vs parameter.
    """
    curve, curve_val = vectors['vectors']
    plt.plot(curve[0], curve[1], label='Training Set')
    plt.plot(curve_val[0], curve_val[1], label='Cross-Validataion Set')
    plt.legend()
    plt.ylabel('Cost Error')
    plt.xlabel('Parameter: %s' % vectors['parameter'])
    plt.show()


def fit_plot(x_set, y_set, alpha, degree):
    """
    Plots fit for any x and y set, with any regularization parameter and polynomial degree values.
    :param x_set: The features.
    :param y_set: The targets.
    :param alpha: The regularization parameter.
    :param degree: Polynomial degree.
    :return: Plot of fit and data points.
    """
    reg = linear_model.Ridge(solver='svd', alpha=alpha)
    reg.fit(PolynomialFeatures(degree).fit_transform(x_set), y_set)

    xmin, xmax = np.min(X), np.max(X)
    x_rng = np.linspace(xmin, xmax, num=20)
    plt.scatter(x_set, y_set)
    plt.plot(x_rng, reg.predict(PolynomialFeatures(degree).fit_transform(x_rng.reshape(-1, 1))).ravel())
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def main():

    # The changing parameter is: test size.
    error_m = error_function('m', 1, len(X))  # Curve data for training and validation sets.
    parameter_plot(error_m)

    # The changing parameter is: regularization parameter.
    error_alpha = error_function('alpha', 0, args.alpha + 10)  # Curve data for training and validation sets.
    parameter_plot(error_alpha)

    # The changing parameter is: polynomial degree.
    error_degree = error_function('degree', 0, args.degree + 5)  # Curve data for training and validation sets.
    parameter_plot(error_degree)

    # Plot of test data fit using default parameters.
    fit_plot(X, y, args.alpha, args.degree)


if __name__ == '__main__':
    main()
