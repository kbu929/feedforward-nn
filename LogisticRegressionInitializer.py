import numpy as np
from utils import sigmoid, relu, d_relu
from utils import load_dataset_sklearn_circle, plot_decision_boundary, load_dataset_2D
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, layer_dims, lr=0.01, n_epochs=10000, step_size=100, initializer='he', init_scale=1):
        self.layer_dims = layer_dims
        self.lr = lr
        self.n_epochs = n_epochs
        self.step_size = step_size
        self.initializer = initializer

        # intermediate data
        self.params = dict()
        self.grads = dict()
        self.caches = dict()

        # initialize parameters
        if initializer == 'zeros':
            self.initialize_params_zeros()
        elif initializer == 'random':
            self.initialize_params_random(init_scale)  # 10 for dramatic illustration
        elif initializer == 'he':
            self.initialize_params_he()

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

    def initialize_params_zeros(self):
        L = len(self.layer_dims) - 1  # exclude the first (input) layer

        for l in range(1, L + 1):  # [1, 2, .. L]
            self.params['W' + str(l)] = np.zeros((self.layer_dims[l], self.layer_dims[l - 1]))
            self.params['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

    def initialize_params_random(self, scale):
        np.random.seed(3)
        L = len(self.layer_dims) - 1  # exclude the first (input) layer

        for l in range(1, L + 1):  # [1, 2, .. L]
            self.params['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) * scale
            self.params['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

    def initialize_params_he(self):
        np.random.seed(3)
        L = len(self.layer_dims) - 1  # exclude the first (input) layer

        for l in range(1, L + 1):  # [1, 2, .. L]
            self.params['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) * np.sqrt(
                2 / self.layer_dims[l - 1])
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

    train_X, train_Y, test_X, test_Y = load_dataset_sklearn_circle()

    n_x = train_X.shape[0]
    layer_dims = [train_X.shape[0], 10, 5, 1]

    clf = LogisticRegression(layer_dims, lr=0.01, n_epochs=15000, step_size=1000, initializer='zeros')

    clf.fit(train_X, train_Y)
    clf.plot_cost()

    clf.evaluate(train_X, train_Y)
    clf.evaluate(test_X, test_Y)

    plt.title("Model with Zeros initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: clf.predict(x.T), train_X, train_Y)
    plt.show()

    clf = LogisticRegression(layer_dims, lr=0.01, n_epochs=15000, step_size=1000, initializer='random', init_scale=10)
    clf.fit(train_X, train_Y)
    clf.plot_cost()

    clf.evaluate(train_X, train_Y)
    clf.evaluate(test_X, test_Y)

    plt.title("Model with Random initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: clf.predict(x.T), train_X, train_Y)
    plt.show()

    clf = LogisticRegression(layer_dims, lr=0.01, n_epochs=15000, step_size=1000, initializer='he')
    clf.fit(train_X, train_Y)
    clf.plot_cost()

    clf.evaluate(train_X, train_Y)
    clf.evaluate(test_X, test_Y)

    plt.title("Model with Random initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: clf.predict(x.T), train_X, train_Y)
    plt.show()