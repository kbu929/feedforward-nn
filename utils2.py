from sklearn.datasets import load_iris
import numpy as np
import math
import sys
import sklearn
import matplotlib


def version_check():
    print("python = ", sys.version)
    print("numpy = ", np.__version__)
    print("sklearn = ", sklearn.__version__)
    print("matplotlib = ", matplotlib.__version__)


def load_data_iris(seed=3, verbose=False):

    # load data from sklearn.datasets
    iris_dataset = load_iris()

    # get the data and target(labels)
    data = iris_dataset['data']
    target = iris_dataset['target']

    # transform data/target suitable for our neural network
    # transpose data: (training examples, features) to (features, training examples)
    # reshape target (labels, ) to (1, labels) [better playing with the explicit shape]
    data = data.T
    target = target.reshape((1, target.shape[-1]))

    # split the data into train and test
    # so use 0.75:0.25 ratio for training and testing data as in sklearn
    train_X, train_Y, test_X, test_Y = train_test_split(data, target, seed=seed)
    return train_X, train_Y, test_X, test_Y, iris_dataset['target_names']


def train_test_split(X, Y, seed=0):
    np.random.seed(seed)
    m = X.shape[1]

    # Important!!! the original data should be permuted before splitting
    # since the original target is ordered in their label like [0, 0, 0, ...1, 1, 1, ... 2, 2, 2]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    # Split
    # use 0.75: 0.25 ratio as in sklearn
    train_length = int(m * 0.75)
    train_X = shuffled_X[:, :train_length]
    train_Y = shuffled_Y[:, :train_length]
    test_X = shuffled_X[:, train_length:]
    test_Y = shuffled_Y[:, train_length:]

    assert train_X.shape[1] + test_X.shape[1] == m
    assert train_Y.shape[1] + test_Y.shape[1] == m

    return train_X, train_Y, test_X, test_Y


def k_fold(kfold=5, random_seed=3):
    np.random.seed(random_seed)

    # load data from sklearn.datasets
    iris_dataset = load_iris()

    # get the data and target(labels)
    data = iris_dataset['data']
    target = iris_dataset['target']

    # transform data/target suitable for our neural network
    # transpose data: (training examples, features) to (features, training examples)
    # reshape target (labels, ) to (1, labels) [better playing with the explicit shape]
    data = data.T
    target = target.reshape((1, target.shape[-1]))

    m = data.shape[1]

    permutation = list(np.random.permutation(m))
    shuffled_X = data[:, permutation]
    shuffled_Y = target[:, permutation]

    # split examples into k-fold
    fold_length = math.ceil(m / kfold)
    fold_list = []
    for k in range(kfold-1):
        fold_X = shuffled_X[:, fold_length * k:fold_length * (k + 1)]
        fold_Y = shuffled_Y[:, fold_length * k:fold_length * (k + 1)]
        fold_list.append((fold_X, fold_Y))

    fold_X = shuffled_X[:, fold_length * (kfold-1):]
    fold_Y = shuffled_Y[:, fold_length * (kfold-1):]
    fold_list.append((fold_X, fold_Y))
    assert len(fold_list) == kfold

    folded_data = dict()
    # split train and test examples (j: test example position)
    for j in range(kfold):
        (test_X, test_Y) = fold_list[j]
        very_first = True
        for i in range(kfold):
            if i != j:
                if very_first:
                    (train_X, train_Y) = fold_list[i]
                    very_first = False
                else:
                    (part_X, part_Y) = fold_list[i]
                    train_X = np.concatenate((train_X, part_X), axis=1)
                    train_Y = np.concatenate((train_Y, part_Y), axis=1)
        folded_data["train_X" + str(j)] = train_X
        folded_data["train_Y" + str(j)] = train_Y
        folded_data["test_X" + str(j)] = test_X
        folded_data["test_Y" + str(j)] = test_Y

    return folded_data


def one_hot(targets, nb_classes):
    encoded = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return encoded.T


def sigmoid(Z):
    return 1. / (1. + np.exp(-Z))


def d_sigmoid(Z):
    A = sigmoid(Z)
    return np.multiply(A, 1 - A)


# def softmax(Z):
#     return np.divide(np.exp(Z), np.sum(np.exp(Z), axis=0, keepdims=True))

# numerically stable
def softmax(Z):
    exps = np.exp(Z - np.max(Z, axis=0))
    return np.divide(exps, np.sum(exps, axis=0))


def relu(Z):
    return np.maximum(0, Z)


def d_relu(Z):
    # dZ = np.ones((Z.shape[0], Z.shape[1]), dtype=int)
    dZ = np.ones((Z.shape[0], Z.shape[1]))
    dZ[Z <= 0] = 0
    return dZ


def tanh(Z):
    return np.tanh(Z)


def d_tanh(Z):
    A = tanh(Z)
    return 1 - np.power(A, 2)


def cross_entropy(A, Y):
    assert A.shape == Y.shape

    m = A.shape[1]

    logprobs = np.multiply(Y, np.log(A))
    cost = -np.nansum(logprobs) / float(m)

    assert cost.shape == ()

    return cost
