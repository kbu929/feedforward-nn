import numpy as np
import h5py
import scipy
import sklearn
import sklearn.datasets
import matplotlib
import matplotlib.pyplot as plt
import math


def version_check():
    print("numpy = ", np.__version__)
    print("scipy = ", scipy.__version__)
    print("sklearn = ", sklearn.__version__)
    print("matplotlib = ", matplotlib.__version__)


def load_dataset_cat(verbose=False):
    filename = 'datasets/train_catvnoncat.h5'
    with h5py.File(filename, 'r') as hf:
        if verbose:
            print("{}: {}".format(filename, list(hf.keys())))
        train_X = np.array(hf['train_set_x'][:])
        train_Y = np.array(hf['train_set_y'][:])

    filename = 'datasets/test_catvnoncat.h5'
    with h5py.File(filename, 'r') as hf:
        if verbose:
            print("{}: {}".format(filename, list(hf.keys())))
        test_X = np.array(hf['test_set_x'][:])
        test_Y = np.array(hf['test_set_y'][:])
        classes = np.array(hf['list_classes'][:])

    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_Y = test_Y.reshape((1, test_Y.shape[0]))

    if verbose:
        print("{}, {}, {}, {}, {}".format(train_X.shape, train_Y.shape, test_Y.shape, test_Y.shape, classes))

    return train_X, train_Y, test_X, test_Y, classes


def load_dataset_sklearn_moon(verbose=False):
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2)  # 300 #0.2

    # Visualize the data
    if verbose:
        plt.figure()
        plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y.ravel().tolist(), s=40, cmap=plt.cm.Spectral);
        plt.show()

    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))

    if verbose:
        print("{} {}".format(train_X.shape, train_Y.shape))

    return train_X, train_Y


def load_dataset_sklearn_circle(verbose=False):
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)

    # Visualize the data
    if verbose:
        plt.figure()
        plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
        plt.show()

    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))

    if verbose:
        print("Train set: {} {}".format(train_X.shape, train_Y.shape))
        print("Test set: {} {}".format(test_X.shape, test_Y.shape))

    return train_X, train_Y, test_X, test_Y


def load_dataset_planar(verbose=False):
    np.random.seed(1)
    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


def load_dataset_2D(verbose=False):
    data = scipy.io.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    if verbose:
        # plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral);
        plt.scatter(train_X[0, :], train_X[1, :], c=train_Y.ravel().tolist(), s=40, cmap=plt.cm.Spectral)
        plt.show()

        print("Train set: {} {}".format(train_X.shape, train_Y.shape))
        print("Test set: {} {}".format(test_X.shape, test_Y.shape))

    return train_X, train_Y, test_X, test_Y


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    # plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.scatter(X[0, :], X[1, :], c=y.ravel().tolist(), cmap=plt.cm.Spectral)
    plt.show()


def print_mislabeled_images(classes, X, Y, Y_pred):
    a = Y_pred + Y
    mislabeled_indices = np.asarray(np.where(a == 1))
    backup = plt.rcParams['figure.figsize']
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title(
            "Prediction: " + classes[int(Y_pred[0, index])].decode("utf-8") + " \n Class: " + classes[
                Y[0, index]].decode(
                "utf-8"))
    plt.show()
    plt.rcParams['figure.figsize'] = backup  # restore the previous setting


def sigmoid(Z):
    return 1. / (1. + np.exp(-Z))


def d_sigmoid(Z):
    A = sigmoid(Z)
    return np.multiply(A, 1 - A)  # A * (1 - A)


def tanh(Z):
    return np.tanh(Z)


def d_tanh(Z):
    A = tanh(Z)
    return 1 - np.power(A, 2)


def relu(Z):
    return np.maximum(0, Z)


def d_relu(Z):
    # dZ = np.ones((Z.shape[0], Z.shape[1]), dtype=int)
    dZ = np.ones((Z.shape[0], Z.shape[1]))
    dZ[Z <= 0] = 0
    return dZ


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)  # To make your "random" minibatches the same as ours
    m = X.shape[1]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


if __name__ == '__main__':
    pass
