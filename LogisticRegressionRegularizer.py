import numpy as np
from utils import sigmoid, relu, d_relu
from utils import load_dataset_sklearn_circle, plot_decision_boundary, load_dataset_2D
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, layer_dims, lr=0.01, n_epochs=10000, step_size=100,
                 initializer='he', init_scale=1,
                 lambd=0, keep_prob=1):
        self.layer_dims = layer_dims
        self.lr = lr
        self.n_epochs = n_epochs
        self.step_size = step_size
        self.initializer = initializer
        self.lambd = lambd
        self.keep_prob = keep_prob

        # only allow one regularization at a time
        # assert self.lambd == 0 or self.keep_prob == 1
        if self.lambd != 0 and self.keep_prob != 1:
            self.keep_prob = 1

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
        elif initializer == 'xavier':
            self.initialize_params_xavier()

        # log
        self.costs = list()

    def fit(self, X, Y):

        for epoch in range(self.n_epochs):

            # Step 1) forward propagation
            if self.keep_prob == 1:
                AL = self.forward(X)
            elif self.keep_prob < 1:
                AL = self.forward_with_dropout(X, self.keep_prob)

            # Step 2) compute cost
            if self.lambd == 0:
                cost = self.compute_cost(AL, Y)
            else:
                cost = self.compute_cost_with_regularization(AL, Y, self.lambd)

            # Step 3) backward
            if self.lambd == 0 and self.keep_prob == 1:
                self.backward(AL, Y)
            elif self.lambd != 0:
                self.backward_with_regularization(AL, Y, self.lambd)
            elif self.keep_prob < 1:
                self.backward_with_dropout(AL, Y, self.keep_prob)

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

    def forward_with_dropout(self, X, keep_prob):
        np.random.seed(1)

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
                A = relu(self.caches['Z' + str(l)])
                D = np.random.rand(A.shape[0], A.shape[1])
                D = (D < keep_prob)
                A = A * D
                self.caches['D' + str(l)] = D
                self.caches['A' + str(l)] = A / keep_prob

        return self.caches['A' + str(L)]

    def compute_cost(self, A, Y):
        assert A.shape == Y.shape

        m = A.shape[1]
        logprobs = np.multiply(Y, np.log(A)) + np.multiply(1 - Y, np.log(1 - A))
        cost = -np.nansum(logprobs) / float(m)

        return cost

    def compute_cost_with_regularization(self, A, Y, lambd):
        assert A.shape == Y.shape
        m = A.shape[1]

        cross_entropy = self.compute_cost(A, Y)

        L = len(self.params) // 2

        L2_cost = 0
        for l in range(1, L + 1):  # [1, 2, .. L]
            L2_cost += np.sum(np.square(self.params['W' + str(l)]))

        L2_cost = (lambd / (2. * float(m))) * L2_cost

        return cross_entropy + L2_cost

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

    def backward_with_regularization(self, A, Y, lambd):
        assert A.shape == Y.shape

        m = A.shape[1]
        L = len(self.params) // 2
        # shortcut for cross-entropy
        dZ = A - Y
        for l in range(L, 0, -1):  # [L, L-1, ...2, 1]
            # from dZ[l] we compute dW[l] and db[l]
            A = self.caches['A' + str(l - 1)]
            self.grads['dW' + str(l)] = np.dot(dZ, A.T) / float(m) + (lambd / float(m)) * self.params['W' + str(l)]
            self.grads['db' + str(l)] = np.sum(dZ, axis=1, keepdims=True) / float(m)

            # stop at [1] since we don't need to compute dW[0] and db[0] and same for dZ[0] (and dA[0])
            if l == 1:
                break

            # compute dZ[l-1] (implicitly dA[l-1] also) for the next iteration
            W = self.params['W' + str(l)]
            Z = self.caches['Z' + str(l - 1)]
            dZ = np.multiply(np.dot(W.T, dZ), d_relu(Z))  # dA = np.dot(W.T, dZ)

    def backward_with_dropout(self, A, Y, keep_prob):
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
            D = self.caches['D' + str(l - 1)]
            dA = np.dot(W.T, dZ)
            dA = dA * D
            dA = dA / keep_prob
            self.grads['dA' + str(l - 1)] = dA
            dZ = np.multiply(dA, d_relu(Z))

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

    def initialize_params_xavier(self):
        np.random.seed(3)
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
    train_X, train_Y, test_X, test_Y = load_dataset_2D(True)
    layer_dims = [train_X.shape[0], 20, 3, 1]

    clf = LogisticRegression(layer_dims, lr=0.3, n_epochs=30000, step_size=1000, initializer='xavier',
                             lambd=0.7)

    clf.fit(train_X, train_Y)
    clf.plot_cost()

    clf.evaluate(train_X, train_Y)
    clf.evaluate(test_X, test_Y)

    plt.title("Model with L2-regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: clf.predict(x.T), train_X, train_Y)
    plt.show()

    clf = LogisticRegression(layer_dims, lr=0.3, n_epochs=30000, step_size=1000, initializer='xavier',
                             keep_prob=0.86)

    clf.fit(train_X, train_Y)
    clf.plot_cost()

    clf.evaluate(train_X, train_Y)
    clf.evaluate(test_X, test_Y)

    plt.title("Model with Dropout({})".format(clf.keep_prob))
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: clf.predict(x.T), train_X, train_Y)
    plt.show()

