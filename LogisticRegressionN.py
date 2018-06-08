import numpy as np
from utils import sigmoid, relu, d_relu
from utils import load_dataset_cat, print_mislabeled_images
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, layer_dims, lr=0.01, n_epochs=10000, step_size=100):
        self.layer_dims = layer_dims
        self.lr = lr
        self.n_epochs = n_epochs
        self.step_size = step_size

        # intermediate data
        self.params = dict()
        self.grads = dict()
        self.caches = dict()

        # initialize parameters (W, b)
        self.initialize_params()

        # log
        self.costs = list()

    def fit(self, X, Y):

        for epoch in range(self.n_epochs):

            # Step 1) forward propagation
            AL = self.forward(X)

            # Step 2) compute cost
            cost = self.compute_cost(AL, Y)

            # Step 3) backward
            self.backward(AL, Y)

            # Step 4) update parameters
            self.update_parameters()

            # log
            if epoch % self.step_size == 0:
                print("{}, cost = {:.6f}".format(epoch, cost))
                self.costs.append(cost)

    def plot_cost(self):
        # plot the cost
        plt.plot(self.costs)
        plt.ylabel('cost')
        plt.xlabel('epoch (per {}s)'.format(self.step_size))
        plt.title("Learning rate = " + str(self.lr))
        plt.show()

    def predict(self, X):
        AL = self.forward(X)
        Y_pred = np.zeros((AL.shape[0], AL.shape[1]))

        for j in range(AL.shape[1]):
            if AL[0][j] > 0.5:
                Y_pred[0][j] = 1

        return Y_pred

    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        accuracy = np.mean(np.equal(Y, Y_pred))
        print("Accuracy = {:.2f}%".format(accuracy * 100))

    def forward(self, X):
        L = len(self.params) // 2
        self.caches['A0'] = X

        for l in range(1, L + 1):  # [1, 2, .. L]
            W = self.params['W' + str(l)]
            A = self.caches['A' + str(l - 1)]
            b = self.params['b' + str(l)]
            self.caches['Z' + str(l)] = np.dot(W, A) + b
            if l == L:  # sigmoid for the last (output) layer
                self.caches['A' + str(l)] = sigmoid(self.caches['Z' + str(l)])
            else:  # relu for hidden layers
                self.caches['A' + str(l)] = relu(self.caches['Z' + str(l)])

        return self.caches['A' + str(L)]

    def compute_cost(self, A, Y):
        assert A.shape == Y.shape

        m = A.shape[1]
        logprobs = np.multiply(Y, np.log(A)) + np.multiply(1 - Y, np.log(1 - A))
        cost = -np.sum(logprobs) / float(m)

        return cost

    def backward(self, A, Y):
        assert A.shape == Y.shape

        m = A.shape[1]
        L = len(self.params) // 2
        # shortcut for cross-entropy
        dZ = A - Y
        for l in range(L, 0, -1):  # [L, L-1, ...2, 1]
            # from dZ[l] we compute dW[l] and db[l]
            A = self.caches['A' + str(l - 1)]
            self.grads['dW' + str(l)] = np.dot(dZ, A.T) / float(m)
            self.grads['db' + str(l)] = np.sum(dZ, axis=1, keepdims=True) / float(m)

            # stop at [1] since we don't need to compute dW[0] and db[0] and same for dZ[0] (and dA[0])
            if l == 1:
                break

            # compute dZ[l-1] (implicitly dA[l-1] also) for the next iteration
            W = self.params['W' + str(l)]
            Z = self.caches['Z' + str(l - 1)]
            dZ = np.multiply(np.dot(W.T, dZ), d_relu(Z))  # dA = np.dot(W.T, dZ)

    def initialize_params(self):
        np.random.seed(1)
        L = len(self.layer_dims) - 1  # exclude the first (input) layer

        for l in range(1, L + 1):  # [1, 2, .. L]
            self.params['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) / np.sqrt(
                self.layer_dims[l - 1])
            self.params['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

    def update_parameters(self):
        L = len(self.params) // 2

        for l in range(1, L + 1):  # [1, 2, .., L]
            self.params["W" + str(l)] = \
                self.params["W" + str(l)] - self.lr * self.grads["dW" + str(l)]
            self.params["b" + str(l)] = \
                self.params["b" + str(l)] - self.lr * self.grads["db" + str(l)]


if __name__ == '__main__':

    np.random.seed(1)

    train_orig_X, train_Y, test_orig_X, test_Y, classes = load_dataset_cat()

    index = 10
    plt.imshow(train_orig_X[index])
    print("y = " + str(train_Y[0, index]) + ". It's a " + classes[train_Y[0, index]].decode("utf-8") + " picture.")
    plt.show()

    train_X_flatten = train_orig_X.reshape(train_orig_X.shape[0], -1).T
    test_X_flatten = test_orig_X.reshape(test_orig_X.shape[0], -1).T

    train_X = train_X_flatten / 255.
    test_X = test_X_flatten / 255.

    n_x = train_X.shape[0]
    layer_dims = [train_X.shape[0], 20, 7, 5, 1]
    clf = LogisticRegression(layer_dims, lr=0.0075, n_epochs=2500)

    clf.fit(train_X, train_Y)
    clf.plot_cost()

    #clf.evaluate(train_X, train_Y)
    #clf.evaluate(test_X, test_Y)

    Y_pred = clf.predict(test_X)

    print_mislabeled_images(classes, test_X, test_Y, Y_pred)
