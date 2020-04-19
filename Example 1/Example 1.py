"""
This module uses linear regression to fit the data in '.txt' files.
"""

import argparse
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics


parser = argparse.ArgumentParser()
parser.add_argument('-file', default='ex1data1.txt', type=str,  help='file name')


def extract_xy(file):
    """
    Extracts x, y from the .txt file and re-shapes the arrays for the scikit-learn fit function.
    The raw data files are formatted in a specific way: features are in columns, and the last column is 'y'.
    :param file: The txt file with x and y.
    :return: x and y.
    """

    data = np.loadtxt(file, delimiter=',')
    n = np.size(data, 1) - 1  # Number of features. The first n columns are features, and the last column is 'y'.
    x = data[:, :n].reshape(-1, n)
    y = data[:, n].reshape(-1, 1)

    return x, y


def train(x, y):
    """
    Uses linear regression to fit the data.
    :param x: The features data.
    :param y: The targets.
    :return: Weight(s) and bias term.
    """
    lin_reg = LinearRegression()
    lin_reg.fit(x, y)  # Training with linear regression.

    # The trained weight(s) and bias term.
    b = lin_reg.intercept_[0]  # The bias term.
    w = lin_reg.coef_[0]  # Weight(s).

    print('Weight(s):', w)
    print('Bias term:', b)

    return w, b


def performance(x, y, w, b):

    # Produce the y prediction set using the data x and the trained weight(s) and bias.
    y_pred = np.matmul(x, w) + b

    print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))


def main():
    args = parser.parse_args()

    x, y = extract_xy(args.file)

    print('Fitting data: %s' % args.file)

    w, b = train(x, y)

    performance(x, y, w, b)


if __name__ == '__main__':
    main()
