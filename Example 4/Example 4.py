import argparse
from scipy.io import loadmat
import tensorflow as tf
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-file', default='ex4data1.mat', type=str,  help='File name.')
parser.add_argument('-n', default=25, type=int, help='Number of nodes in the hidden layer')
parser.add_argument('-epochs', default=100, type=int, help='Number of epochs.')


def extract_xy(file):

    data = loadmat(file)
    x_data, y_data = data["X"], data["y"].flatten()  # x_data: pixel values
    y_data[y_data == 10] = 0  # Matlab artifact; change all 10's to zeros.

    y_temp = np.zeros((len(y_data), 10))
    for i in range(len(y_data)):
        y_temp[i][y_data[i]] = 1
    y_data = y_temp

    return x_data, y_data


def neural_network_model(x, n_nodes_hl=25):
    """
    The neural network model with one hidden layer.
    :param x: Input layer data.
    :param n_nodes_hl: Number of nodes in the hidden layer.
    :return: Output layer, hidden layer weights & biases, output layer weights & biases.
    """

    # Hidden layer matrices
    hl_matrices = {"weights": tf.Variable(tf.random_normal([400, n_nodes_hl])),
                   "biases": tf.Variable(tf.random_normal([n_nodes_hl]))}
    # Output layer matrices
    ol_matrices = {"weights": tf.Variable(tf.random_normal([n_nodes_hl, 10])),
                   "biases": tf.Variable(tf.random_normal([10]))}

    hidden_layer = tf.matmul(x, hl_matrices['weights']) + hl_matrices['biases']
    hidden_layer = tf.nn.relu(hidden_layer)  # Activation: Rectified Linear Unit

    output_layer = tf.matmul(hidden_layer, ol_matrices['weights']) + ol_matrices['biases']

    return output_layer, hl_matrices, ol_matrices


def train_neural_network(x, y, x_data, y_data, n_nodes_hl=25, n_epochs=100):
    """
    Trains the neural network.
    :param x: Input layer data.
    :param y: Test labels.
    :param x_data: Image data.
    :param y_data: Target data.
    :param n_nodes_hl: Number of nodes in the hidden layer.
    :param n_epochs: Number of epochs.
    :return: Trained hidden and output layer matrices.
    """

    # Tensorflow optimizer and cost functions.
    prediction = neural_network_model(x, n_nodes_hl)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction[0], labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            epoch_loss = 0

            _, c = sess.run([optimizer, cost], feed_dict={x: x_data, y: y_data})
            epoch_loss += c

        correct = tf.equal(tf.argmax(prediction[0], 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: x_data, y: y_data}))

        # The trained matrices.
        trained_hidden_layer, trained_output_layer = sess.run(prediction[1]), sess.run(prediction[2])

        return trained_hidden_layer, trained_output_layer


def main():

    args = parser.parse_args()

    print('Feed-Forward Neural Network with one hidden layer is '
          'trained with TensorFlow tools using data: %s' % args.file)

    X_data, Y_data = extract_xy(args.file)

    X = tf.placeholder("float", [None, 400])
    Y = tf.placeholder("float")

    # Collect the trained matrices.
    matrices = train_neural_network(X, Y, X_data, Y_data, args.n, args.epochs)


if __name__ == '__main__':
    main()
