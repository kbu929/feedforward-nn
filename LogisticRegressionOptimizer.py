import numpy as np
from utils import sigmoid, relu, d_relu
from utils import load_dataset_sklearn_moon, plot_decision_boundary
from utils import random_mini_batches
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, layer_dims, lr=0.01, n_epochs=10000, step_size=100,
                 initializer='he', init_scale=1, lambd=0, keep_prob=1,
                 mini_batch_size=64, optimizer='adam', beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
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

        # optimizer
        self.optimizer = optimizer
        self.mini_batch_size = mini_batch_size
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # intermediate data
        self.params = dict()
        self.grads = dict()
        self.caches = dict()

        # optimization
        self.v = dict()
        self.s = dict()

        # initialize parameters
        if initializer == 'zeros':
            self.initialize_params_zeros()
        elif initializer == 'random':
            self.initialize_params_random(init_scale)  # 10 for dramatic illustration
        elif initializer == 'he':
            self.initialize_params_he()
        elif initializer == 'xavier':
            self.initialize_params_xavier()

        if optimizer == 'momentum':
            self.initialize_velocity()
        elif optimizer == 'adam':
            self.initialize_adam()

        # log
        self.costs = list()

    def fit(self, X, Y):
        seed = 10
        t = 0
        self.costs = []

        for epoch in range(self.n_epochs):

            seed = seed + 1
            minibatches = random_mini_batches(X, Y, self.mini_batch_size, seed)

            for minibatch in minibatches:

                minibatch_X, minibatch_Y = minibatch

                # Step 1) forward propagation
                if self.keep_prob == 1:
                    AL = self.forward(minibatch_X)
                elif self.keep_prob < 1:
                    AL = self.forward_with_dropout(minibatch_X, self.keep_prob)

                # Step 2) compute cost
                if self.lambd == 0:
                    cost = self.compute_cost(AL, minibatch_Y)
                else:
                    cost = self.compute_cost_with_regularization(AL, minibatch_Y, self.lambd)

                # Step 3) backward
                if self.lambd == 0 and self.keep_prob == 1:
                    self.backward(AL, minibatch_Y)
                elif self.lambd != 0:
                    self.backward_with_regularization(AL, minibatch_Y, self.lambd)
                elif self.keep_prob < 1:
                    self.backward_with_dropout(AL, minibatch_Y, self.keep_prob)

                # Step 4) update parameters
                if self.optimizer == 'gd':
                    self.update_parameters_gd()
                elif self.optimizer == 'momentum':
                    self.update_parameters_with_momentum()
                elif self.optimizer == 'adam':
                    t = t + 1
                    self.update_parameters_with_adam(t)

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

        L = len(self.params) // 2
        m = Y.shape[1]

        # shotcut for cross-entropy
        dZ = A - Y
        for l in range(L, 0, -1):  # [L, L-2, 2, 1]
            # from dZ[l] we compute dW[l] and db[l]
            A = self.caches['A' + str(l - 1)]
            self.grads['dW' + str(l)] = np.dot(dZ, A.T) / float(m)
            self.grads['db' + str(l)] = np.sum(dZ, axis=1, keepdims=True) / float(m)

            # stop at [1] since we don't need to compute dW[0] and db[0], so same for dZ[0] (and dA[0])
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

    def update_parameters_gd(self):
        L = len(self.params) // 2

        for l in range(1, L + 1):  # [1, 2, .., L]
            self.params["W" + str(l)] = \
                self.params["W" + str(l)] - self.lr * self.grads["dW" + str(l)]
            self.params["b" + str(l)] = \
                self.params["b" + str(l)] - self.lr * self.grads["db" + str(l)]

    def initialize_velocity(self):
        L = len(self.params) // 2

        for l in range(1, L + 1):  # [1, 2, .., L]
            self.v["dW" + str(l)] = \
                np.zeros((self.params["W" + str(l)].shape[0], self.params["W" + str(l)].shape[1]))
            self.v["db" + str(l)] = \
                np.zeros((self.params["b" + str(l)].shape[0], self.params["b" + str(l)].shape[1]))

    def update_parameters_with_momentum(self):
        L = len(self.params) // 2

        # Momentum update for each parameter
        for l in range(1, L + 1):  # [1, 2, .., L]
            # compute velocities
            self.v["dW" + str(l)] = \
                self.beta * self.v["dW" + str(l)] + (1 - self.beta) * self.grads["dW" + str(l)]
            self.v["db" + str(l)] = \
                self.beta * self.v["db" + str(l)] + (1 - self.beta) * self.grads["db" + str(l)]
            # update parameters
            self.params["W" + str(l)] = \
                self.params["W" + str(l)] - self.lr * self.v["dW" + str(l)]
            self.params["b" + str(l)] = \
                self.params["b" + str(l)] - self.lr * self.v["db" + str(l)]

    def initialize_adam(self):
        L = len(self.params) // 2

        # Initialize v, s. Input: "parameters". Outputs: "v, s".
        for l in range(1, L + 1):  # [1, 2, .., L]
            self.v["dW" + str(l)] = \
                np.zeros((self.params["W" + str(l)].shape[0], self.params["W" + str(l)].shape[1]))
            self.v["db" + str(l)] = \
                np.zeros((self.params["b" + str(l)].shape[0], self.params["b" + str(l)].shape[1]))
            self.s["dW" + str(l)] = \
                np.zeros((self.params["W" + str(l)].shape[0], self.params["W" + str(l)].shape[1]))
            self.s["db" + str(l)] = \
                np.zeros((self.params["b" + str(l)].shape[0], self.params["b" + str(l)].shape[1]))

    def update_parameters_with_adam(self, t):
        L = len(self.params) // 2
        v_corrected = dict()
        s_corrected = dict()

        # Perform Adam update on all parameters
        for l in range(1, L + 1):
            # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
            self.v["dW" + str(l)] = \
                self.beta1 * self.v["dW" + str(l)] + (1 - self.beta1) * self.grads["dW" + str(l)]
            self.v["db" + str(l)] = \
                self.beta1 * self.v["db" + str(l)] + (1 - self.beta1) * self.grads["db" + str(l)]

            # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
            v_corrected["dW" + str(l)] = self.v["dW" + str(l)] / (1 - self.beta1 ** t)
            v_corrected["db" + str(l)] = self.v["db" + str(l)] / (1 - self.beta1 ** t)

            # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
            self.s["dW" + str(l)] = self.beta2 * self.s["dW" + str(l)] + (1 - self.beta2) * (
                    self.grads["dW" + str(l)] ** 2)
            self.s["db" + str(l)] = self.beta2 * self.s["db" + str(l)] + (1 - self.beta2) * (
                    self.grads["db" + str(l)] ** 2)

            # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
            s_corrected["dW" + str(l)] = self.s["dW" + str(l)] / (1 - self.beta2 ** t)
            s_corrected["db" + str(l)] = self.s["db" + str(l)] / (1 - self.beta2 ** t)

            # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon".
            # Output: "parameters".
            self.params["W" + str(l)] = self.params["W" + str(l)] \
                                        - self.lr * np.divide(v_corrected["dW" + str(l)],
                                                              np.sqrt(s_corrected["dW" + str(l)]) + self.epsilon)
            self.params["b" + str(l)] = self.params["b" + str(l)] \
                                        - self.lr * np.divide(v_corrected["db" + str(l)],
                                                              np.sqrt(s_corrected["db" + str(l)]) + self.epsilon)


if __name__ == '__main__':
    train_X, train_Y = load_dataset_sklearn_moon(True)

    # layer_dims = [train_X.shape[0], 5, 2, 1]
    # clf = LogisticRegression(layer_dims, optimizer='gd', lr=0.0007)
    # clf.fit(train_X, train_Y)
    # clf.evaluate(train_X, train_Y)
    # plot_decision_boundary(lambda x: clf.predict(x.T), train_X, train_Y)
    #
    # layer_dims = [train_X.shape[0], 5, 2, 1]
    # clf = LogisticRegression(layer_dims, optimizer='momentum', lr=0.0007, beta=0.9)
    # clf.fit(train_X, train_Y)
    # clf.evaluate(train_X, train_Y)
    # plot_decision_boundary(lambda x: clf.predict(x.T), train_X, train_Y)

    layer_dims = [train_X.shape[0], 5, 2, 1]
    # clf = LogisticRegression(layer_dims, optimizer='adam', lr=0.0007)
    # clf.fit(train_X, train_Y)
    # clf.plot_cost()
    # clf.evaluate(train_X, train_Y)
    # plot_decision_boundary(lambda x: clf.predict(x.T), train_X, train_Y)

    # clf = LogisticRegression(layer_dims, optimizer='adam', lr=0.0007, lambd=0.5)
    # clf.fit(train_X, train_Y)
    # clf.plot_cost()
    # clf.evaluate(train_X, train_Y)
    # plot_decision_boundary(lambda x: clf.predict(x.T), train_X, train_Y)

    clf = LogisticRegression(layer_dims, optimizer='adam', lr=0.0007, keep_prob=0.9)
    clf.fit(train_X, train_Y)
    clf.plot_cost()
    clf.evaluate(train_X, train_Y)
    plot_decision_boundary(lambda x: clf.predict(x.T), train_X, train_Y)
