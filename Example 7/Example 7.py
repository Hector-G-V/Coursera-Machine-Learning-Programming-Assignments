import argparse
from scipy.io import loadmat
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-file', default='bird_small.mat', type=str, help='File name.')
parser.add_argument('-model', default='kmeans', type=str, help='Model selection: kmeans or pca.')
parser.add_argument('-n', default=16, type=int, help='Number of clusters(K-means) or components (PCA).')
parser.add_argument('-pca_img', default=0, type=int, help='Image number to plot in PCA session.')


def extract_xy(file):
    """
    Extracts data from file. In the original Example 7 assignment data, all keys are either 'A' or 'X.'
    :param file: The file with the X data.
    :return: Array with the X data.
    """
    data = loadmat(file)

    if 'A' in data.keys():
        X = data['A']
        return X

    elif 'X' in data.keys():
        X = data['X']
        return X

    else:
        print('The file data dictionary key does not match with the assignment keys.')
        return None


def train(X, n, model):
    """
    Trains the model using either the K-Means or PCA algorithm.
    :param X: The data.
    :param n: The number of centroids (K-Means) or components (PCA) to keep.
    :param model: Either 'kmeans' for K-Means, or 'pca' for pca.
    :return: Processed X data for plotting.
    """
    if model == 'kmeans':

        if np.shape(X)[1] > 2 and np.ndim(X) == 3:

            h, w, r = np.shape(X)  # Pixel height(rows), width(columns), and representation.
            X_preprocessed = X.reshape(h*w, r)  # X dim needs to be 2.

            kmeans = KMeans(n_clusters=n, random_state=0).fit(X_preprocessed)

            labels = kmeans.labels_
            centers = np.round(kmeans.cluster_centers_).astype(int)  # Converts centers to int.

            X_rebuild = centers[labels]  # Map labels to array of centers.
            X_rebuild = X_rebuild.reshape(h, w, r)  # Reshape to image dimensions.

            return X_rebuild

        else:

            kmeans = KMeans(n_clusters=n, random_state=0).fit(X)

            labels = kmeans.labels_
            centers = kmeans.cluster_centers_

            clustered = []  # Will hold clustered data points.
            for i in range(np.max(labels) + 1):
                xtemp = X[labels == i]
                clustered.append(xtemp)

            return np.array(clustered), centers

    elif model == 'pca':

        X_norm = scale(X)  # Normalize.

        pca = PCA(n_components=n)  # Parameters are reduced here.
        pca.fit(X_norm)

        components = pca.components_

        z = np.matmul(components, X_norm.T)  # Reduced vector.

        X_approx = np.matmul(components.T, z)  # Recovered vector.

        return X_approx

    else:
        print('Neither model was matched with the input model parameter.')
        return None


def plot(X, X_returned, model, pca_img=0):
    """
    Plots the data.
    :param X: The data.
    :param X_returned: The processed X data.
    :param model: Either 'kmeans' for K-Means, or 'pca' for pca.
    :param pca_img: The image number in the X data, if running PCA.
    :return: Plot.
    """
    if model == 'kmeans':
        if np.shape(X)[1] > 2 and np.ndim(X) == 3:

            plt.figure('X')
            plt.imshow(X)

            plt.figure('X_rebuild')
            plt.imshow(X_returned)

            plt.show()

            return None

        else:

            plt.figure('Clustered Data')

            for i in X_returned[0]:  # Plots the clustered data.
                plt.scatter(i.T[0], i.T[1], edgecolors='black')

            plt.scatter(X_returned[1].T[0], X_returned[1].T[1], label='Centers', color='y',
                        marker='P', edgecolors='k', s=100)

            plt.show()

            return None

    if model == 'pca':

        if np.shape(X)[1] == 2 and np.ndim(X) == 2:  # Plot data.
            plt.figure('X')
            plt.scatter(X.T[0], X.T[1], label='X')
            plt.legend()

            plt.figure('Scaled and Recovered Vectors')
            plt.scatter(scale(X).T[0], scale(X).T[1], label='Scaled')
            plt.scatter(X_returned[0], X_returned[1], label='Recovered')
            plt.legend()
            plt.show()

        elif np.shape(X)[1] > 2 and np.ndim(X) == 2:  # Plot images.
            h, w = int(np.sqrt(len(X[0]))), int(np.sqrt(len(X[0])))  # Image height(rows), width(columns).

            plt.figure('X')
            plt.imshow(X[pca_img].reshape(h, w, order='F'), cmap='gray')

            plt.figure('X Recovered')
            plt.imshow(X_returned.T[pca_img].reshape(h, w, order='F'), cmap='gray')

            plt.show()

        else:
            print('Cannot plot this file.')
            return None


def main():

    args = parser.parse_args()

    print('%s on file: %s' % (args.model, args.file))

    x = extract_xy(args.file)

    x_returned = train(x, args.n, args.model)

    plot(x, x_returned, args.model, pca_img=args.pca_img)


if __name__ == '__main__':
    main()
