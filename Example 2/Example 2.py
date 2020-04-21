import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('-file', default='ex2data1.txt', type=str,  help='File name.')
parser.add_argument('-step', default=0.5, type=float, help='Mesh step size for plot.')
parser.add_argument('-C', default=1.0, type=float, help='Inverse of regularization strength.')
parser.add_argument('-d', default=1, type=int, help='Mapped features polynomial degree.')
parser.add_argument('-rand', default=1, type=int, help='random_state input for the sklearn train_test_split function.')
args = parser.parse_args()


def extract_xy(file):
    """
    Extracts x, y from the .txt file.
    The raw data files are formatted in a specific way: features are in columns, and the last column is 'y'.
    :param file: The txt file with x and y.
    :return: x and y.
    """
    data = np.loadtxt(file, delimiter=',')
    x, y = data[:, :2], data[:, 2]

    return x, y


def map_feature(x, degree=args.d):
    """
    Maps the features to all polynomial values up to the specified degree.
    :param x:
    :param degree: The max polynomial degree.
    :return: Mapped features in columns.
    """

    mapped = [np.ones(len(x))]

    for i in range(1, degree + 1):
        for j in range(i + 1):
            mapped.append((x[:, 0] ** (i - j)) * (x[:, 1] ** j))

    return np.array(mapped).T


def train(x, y, C=1.0, rand=1):
    """
    Trains the model with a logistic regression algorithm.
    :param x: The features data.
    :param y: The targets.
    :param C: Inverse of regularization strength.
    :param rand: random_state input for the sklearn train_test_split function.
    :return: Trained model for classification.
    """

    # Split data into training and test sets.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=rand)  # 30% training and 70% test

    log_reg = LogisticRegression(solver='lbfgs', C=C, max_iter=10**4)
    log_reg.fit(map_feature(x_train), y_train)

    score_train = log_reg.score(map_feature(x_train), y_train)
    score_test = log_reg.score(map_feature(x_test), y_test)
    print('Training Score:', score_train)  # Prediction accuracy using the training data.
    print('Testing Score:', score_test)  # Prediction accuracy using the testing data.

    return log_reg


def plot(x, y, log_reg, step=0.5):  # (i) Look for ways to shrink these lines. (ii) Change the variables.
    """
    Plots the classified data points and the decision boundary.
    :param x: The features data.
    :param y: The target.
    :param log_reg: The trained model.
    :param step: Mesh step size
    :return: Plot
    """

    # Max and min points in the mesh.
    x1_min, x1_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    x2_min, x2_max = x[:, 1].min() - .5, x[:, 1].max() + .5

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step))
    Z = log_reg.predict(map_feature(np.c_[xx1.ravel(), xx2.ravel()]))

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


def main():

    print('Training the model with a logistic regression algorithm and data: %s' % args.file)

    X, Y = extract_xy(args.file)

    logistic_regression = train(X, Y, C=args.C, rand=args.rand)

    plot(X, Y, logistic_regression, step=args.step)


if __name__ == '__main__':
    main()
