import argparse
import random
from scipy.io import loadmat
import math
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


parse = argparse.ArgumentParser()
parse.add_argument('-file', default='ex8_movies.mat', type=str, help='File name.')
parse.add_argument('-n', default=100, type=int, help='Length of features and weights vectors.')
parse.add_argument('-iter', default=200, type=int, help='Number of gradient descent iterations.')


def extract_xy(file):
    """
    Extracts data from file. Can extract training data or debugging data.
    :param file: Data or training .mat file.
    :return: Parameters from the file.
    """
    if file == 'ex8_movieParams.mat':  # The debug data.
        data = loadmat('ex8_movieParams.mat')
        X, Theta = data['X'], data['Theta']
        num_users, num_movies, num_features = data['num_users'], data['num_movies'], data['num_features']

        return X, Theta, num_users, num_movies, num_features

    else:
        data = loadmat(file)  # This is the data.
        Y, R = data['Y'], data['R']

        return Y, R


def train(Y, R, n=100, iter=200):
    """
    Finds X and Theta from Y and R.
    :param Y: Input data.
    :param R: Rated/Unrated index.
    :param n: Number of features.
    :param iter: Gradient descent iterations.
    :return: X, T, and cost function J.
    """
    dim_X, dim_T = np.shape(Y)  # Dimensions of X and Theta.

    # Initialize random Theta and X. Generate n*dim random numbers.
    X = np.array(random.sample(list(np.linspace(-1, 1, n*dim_X)*np.sqrt(5/n)), n*dim_X)).reshape(dim_X, n)
    Theta = np.array(random.sample(list(np.linspace(-1, 1, n*dim_T)*np.sqrt(5/n)), n*dim_T)).reshape(dim_T, n)

    # Gradient Descent
    a, l = 0.001, 0.1  # Learning rate and regularization parameter.

    J = []  # Array to hold the costs over all iterations.

    # Find X, Theta.
    for i in range(iter):

        XT = np.matmul(X, Theta.T)
        XT[Y == 0] = 0  # There will be no contribution from unrated movies.

        J_temp = np.sum((XT - Y)**2) + l*(np.sum(X**2) + np.sum(Theta**2))
        J.append(J_temp)
        print(J_temp)

        X_temp = X - a*(np.matmul(XT - Y, Theta) + l*X)  # No filter? Yelem=0 when R=0, so no contribution?
        T_temp = Theta - a*(np.matmul((XT - Y).T, X) + l*Theta)

        X = X_temp
        Theta = T_temp

    # Find and print the accuracy. Y elements with no rating (R=0) are removed.
    Y_pred = np.round(np.matmul(X, Theta.T)).astype(int)  # The Y prediction vector.

    acc_temp = []  # Holds predictions for each movie.
    for i in range(len(Y)):

        r, y, y_pred = R[i], Y[i], Y_pred[i]

        y_temp, y_pred_temp = y[r == 1], y_pred[r == 1]

        a = y_temp == y_pred_temp
        b = len(a[a == True])
        acc_temp.append(b / np.size(y_temp))

    print('Accuracy:', np.mean(acc_temp))  # Average accuracy is same as total accuracy.

    return X, Theta, J


def plot(J):
    """
    Plots the cost function.
    :param J: List of cost function values over the iterations.
    :return: Plot of iteration vs J.
    """
    plt.plot(J)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()


def main():
    args = parse.parse_args()

    print('Using data Y, R to find a features vector X, and a ratings (weights) vector \u03B8:')

    y, r = extract_xy(args.file)

    x, theta, j = train(y, r, n=args.n, iter=args.iter)

    plot(j)


if __name__ == '__main__':
    main()
