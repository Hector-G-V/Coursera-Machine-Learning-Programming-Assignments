import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

parser = argparse.ArgumentParser()
parser.add_argument('-file', default='ex3data1.mat', type=str,  help='File name.')
parser.add_argument('-C', default=1.0, type=float, help='Inverse of regularization strength.')
parser.add_argument('-rand', default=1, type=int, help='random_state input for the sklearn train_test_split function.')


def extract_xy(file):
    """
    Extracts x, y from the .mat file.
    The x raw data files format: Pixels for one image are in a single row.
    :param file: The .mat file with x and y.
    :return: x and y.
    """

    data = loadmat(file)
    x, y = data['X'], data['y'].ravel()
    y[y == 10] = 0

    return x, y


def train(x, y, C=1.0, rand=1):
    """
    Trains the model with a one-vs-rest logistic regression algorithm.
    :param x: The images data.
    :param y: The targets.
    :param C: Inverse of regularization strength.
    :param rand: random_state input for the sklearn train_test_split function.
    :return: Trained model for classification.
    """
    # Split data into training and test sets.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=rand)  # 10% training and 90% test

    log_reg = LogisticRegression(solver='lbfgs', C=C, max_iter=10**4, multi_class='ovr')  # ovr: one-vs-rest
    log_reg.fit(x_train, y_train)

    score_train = log_reg.score(x_train, y_train)
    score_test = log_reg.score(x_test, y_test)

    print('Training Score:', score_train)  # Prediction accuracy using the training data.
    print('Testing Score:', score_test)  # Prediction accuracy using the testing data.

    return log_reg


def main():
    args = parser.parse_args()

    print('Training the model with a one-vs-rest logistic regression algorithm and data: %s' % args.file)

    X, Y = extract_xy(args.file)

    logistic_regression = train(X, Y, C=args.C, rand=args.rand)


if __name__ == '__main__':
    main()
