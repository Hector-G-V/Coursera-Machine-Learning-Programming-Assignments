import argparse
from scipy.io import loadmat
import math
import numpy as np
from sklearn import metrics


parser = argparse.ArgumentParser()
parser.add_argument('-file', default='ex8data1.mat', type=str, help='File name.')
parser.add_argument('-e_min', default=10**-6, type=float, help='Minimum epsilon value in search range.')
parser.add_argument('-e_max', default=10**-4, type=float, help='Maximum epsilon value in search range.')

def extract_xy(file):
    """
    Extracts data from file.
    :param file: The file with the X data.
    :return: Array with the X data.
    """
    data = loadmat(file)
    X, Xval, yval = data['X'], data['Xval'], data['yval']

    return X, Xval, yval


def train(X, Xval):
    """
    Trains the model using X, and finds p using Xval.
    :param X: The training data X.
    :param Xval: The validation data.
    :return: p for all Xval.
    """

    m, n = np.shape(X)

    mu = np.mean(X, axis=0)

    s = 1 / m * np.matmul((X - mu).T, X - mu)  # Variance, symbol 'sigma.'

    s_inv = np.linalg.inv(s)  # Inverse of sigma.

    s_cons = np.sqrt(np.linalg.det(s))  # Constant in normalization parameter, square root of the determinant.
    N = (2 * math.pi) ** (n / 2) * s_cons  # p normalization parameter.

    # Implement vectorization.
    p_matrix = 1 / N * np.exp(
        -1 / 2 * np.matmul(Xval - mu, np.matmul(s_inv, (Xval - mu).T)))  # This operation produces cross-terms.
    p_vector = p_matrix.diagonal()  # Diagonals equal to p(x) for each data point x.

    return p_vector


def prediction(yval, p_vector, e_min, e_max):
    """
    Prints the F1 scores in epsilon range e_min, e_max.
    :param yval: Targets in validation set.
    :param p_vector: p values for validation set on trained model.
    :param e_min: Minimum epsilon value in search range.
    :param e_max: Maximum epsilon value in search range.
    :return: Prints epsilon, F1 score, and detected anomalies in the specified range.
    """

    print("Epsilon: \t F1 Score: \t Anomalies:")

    for eps in np.linspace(e_min, e_max, 10):
        y_pred = np.zeros(len(p_vector))
        y_pred[p_vector < eps] = 1
        y_pred = y_pred.reshape(-1, 1)  # Makes dimensions consistent with validation set.

        print('%.2e \t %.2e \t %d' % (eps, metrics.f1_score(yval, y_pred), len(y_pred[y_pred == 1])))


def main():

    args = parser.parse_args()

    print('Anomaly detection on file: %s' % args.file)

    x, x_val, y_val = extract_xy(args.file)

    p_x = train(x, x_val)

    prediction(y_val, p_x, args.e_min, args.e_max)


if __name__ == '__main__':
    main()
