import argparse
import scipy.io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-file', default='ex6data1.mat', type=str,  help='File name.')
parser.add_argument('-X_key', default='X', type=str, help='Dictionary key for X in data.')
parser.add_argument('-y_key', default='y', type=str, help='Dictionary key for y in data.')
parser.add_argument('-C', default=1.0, type=float, help='Inverse of regularization strength.')
parser.add_argument('-kernel', default='rbf', type=str, help='SVM Kernel.')
parser.add_argument('-test_size', default='0.1', type=float, help='Test data fraction of total data.')
parser.add_argument('-rand', default=1, type=int, help='random_state input for the sklearn train_test_split function.')
parser.add_argument('-step', default=0.01, type=float, help='Mesh step size for plot.')


def extract_xy(file, x_key='X', y_key='y'):
    """
    Extracts x, y from the file. Will work with pre-processed .mat files,
    and with .csv files that require pre-processing.
    :param file: The tile with x and y.
    :param x_key: Dictionary key for the features data.
    :param y_key: Dictionary key for the targets.
    :return: x and y.
    """

    if '.mat' in file:
        data = scipy.io.loadmat(file)
        x, y = data[x_key], data[y_key][:, 0]
        return x, y

    if '.csv' in file:
        data = pd.read_csv(file)
        x, y = data[x_key], data[y_key]
        x = CountVectorizer().fit_transform(x)  # pre-processing: converts string email into features vector.
        return x, y
    else:
        print('Cannot use this file.')
        return None, None


def plot(x, y, func, step=0.01):  # pulled directly from Example 2
    """
    Plots the classified data points and the decision boundary.
    :param x: The features data.
    :param y: The targets.
    :param func: The trained model.
    :param step: Mesh step size
    :return: Plot
    """

    # Confirm that the features data is 2D.
    if np.ndim(x) != 2 or np.shape(x)[1] != 2:
        return print('Cannot plot this data.')

    # Max and min points in the mesh.
    x1_min, x1_max = x[:, 0].min() - .01, x[:, 0].max() + .01  # Utils parameter.
    x2_min, x2_max = x[:, 1].min() - .01, x[:, 1].max() + .01

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step))
    Z = func.predict(np.c_[xx1.ravel(), xx2.ravel()])

    # Color plot for mesh.
    Z = Z.reshape(xx1.shape)
    plt.pcolormesh(xx1, xx2, Z, cmap='Pastel2')

    # Separate the features, and separate by class.
    class0_x1, class0_x2 = np.array(x[y == 0]).T
    class1_x1, class1_x2 = np.array(x[y == 1]).T

    # Plot the data points.
    plt.scatter(class0_x1, class0_x2, label='class 0', color='y', edgecolors='black')
    plt.scatter(class1_x1, class1_x2, label='class 1', color='k', marker='P')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()


def train(x, y, C=1.0, kernel='rbf', test_size=0.1, rand=1):
    """
    Trains the model with an SVM algorithm.
    :param x: The features data.
    :param y: The targets.
    :param C: Inverse of regularization strength.
    :param kernel: SVM kernel.
    :param test_size: Test data fraction of total data.
    :param rand: random_state input for the sklearn train_test_split function.
    :return: Trained model for classification.
    """
    # Split data into training and test sets.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=rand)  # 10% training and 90% test

    # Train the model.
    clf = svm.SVC(gamma='scale', kernel=kernel, C=C)  # SVM classifier.  Gaussian Kernel: kernel='rbf', C=1.0
    clf.fit(x_train, y_train)  # Train the model.

    # Evaluation.
    print('Training Score:', clf.score(x_train, y_train))  # Prediction accuracy using the training data.
    print('Testing Score:', clf.score(x_test, y_test))  # Prediction accuracy using the testing data.

    return clf


def main():

    args = parser.parse_args()

    print('Training the model with an SVM algorithm and data: %s' % args.file)

    X, Y = extract_xy(args.file, x_key=args.X_key, y_key=args.y_key)

    classifier = train(X, Y, C=args.C, kernel=args.kernel, test_size=args.test_size, rand=args.rand)

    plot(X, Y, classifier, step=args.step)


if __name__ == '__main__':
    main()
