"""
Vanilla implementation of Neural Network

- support multi-class classification using Softmax
- support any length of layer
- initialization: random, xavier, he
- activation function: sigmoid, relu, tanh
- output and cost function: softmax / cross entropy
- regularization: L2, dropout
- optimization: batch gradient descent, momentum, adam

- test data: Iris (from sklearn.datasets)
"""
import numpy as np
import matplotlib.pyplot as plt
from utils2 import load_data_iris, one_hot, k_fold, version_check
from utils2 import sigmoid, d_sigmoid, softmax, relu, d_relu, tanh, d_tanh, cross_entropy


class SoftmaxRegression:
    def __init__(self, layer_dims, lr=0.01, n_epochs=3000,
                 initializer='he',
                 act_func='sigmoid',
                 lambd=0, keep_prob=1,
                 optimizer='gd', beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        initialize (hyper)parameters for Neural Network

        layer_dims: list of each layer's dimension: [n_x, n_h, ..., n_y]
        lr: learning rate
        n_epochs: num of iteration for training
        initializer: ['random', 'xavier', 'he'(*)]
        act_func: ['sigmoid'(*), 'relu', 'tanh']
        lambd: parameter for L2-regularization
        keep_prob: parameter for Drop out
        optimizer: ['gd'(*), 'momentum', 'adam']
        beta: parameter for momentum (default value is enough)
        beta1: parameter for adam (default value is enough)
        beta2: parameter for adam (default value is enough)
        epsilon: parameter for adam (default value is enough)

        (*): default
        """
        # helpers
        self.random_seed = 1  # for reproducible experiments
        self.step_size = n_epochs // 20  # 20 intervals for cost plot

        # hyper-parameters
        self.layer_dims = layer_dims  # [n_x, n_h, ..., n_y]
        self.lr = lr  # learning rate
        self.n_epochs = n_epochs  # number of iterations for training
        self.initializer = initializer  # ['he'(*), 'xavier', 'random']
        self.lambd = lambd  # L2 regularization
        self.keep_prob = keep_prob  # drop out

        # optimizer
        self.optimizer = optimizer  # optimization ['gd'(*), 'momentum', 'adam']
        self.beta = beta  # 0.9(*) for momentum
        self.beta1 = beta1  # 0.9(*) for adam
        self.beta2 = beta2  # 0.999(*) for adam
        self.epsilon = epsilon  # 1e-8(*) for adam

        # placeholders and variables for training/optimization
        self.params = dict()  # W1, b1, W2, b2, ... WL, bL for L-layered neural network
        self.grads = dict()  # dW1, db1, dW2, db2, ..., dWL, dbL for L-layered neural network
        # Z: linear transformed, A: nonlinear transformed, D: shutdown mask for dropout
        self.caches = dict()  # A0, Z1, A1, D1, ...., AL, ZL, AL for L-layered neural network
        # first and second momentum for optimization
        self.v = dict()
        self.s = dict()

        # log
        self.costs = list()

        # set the activation function (default: sigmoid, this is a requirement from the assignment)
        if act_func == 'sigmoid':
            self.act_func = sigmoid
            self.d_act_func = d_sigmoid
        elif act_func == 'relu':
            self.act_func = relu
            self.d_act_func = d_relu
        elif act_func == 'tanh':
            self.act_func = tanh
            self.d_act_func = d_tanh
        else:
            self.act_func = sigmoid
            self.d_act_func = d_sigmoid

        # initialize weights and bias
        self.initialize_params(initializer)
        # initialize parameters for optimization
        self.initialize_params_for_optimization(optimizer)

    def fit(self, X, Y, verbose=False):
        """
        train the neural network from given (X, Y) pairs

        X: (n_x, m) = (feature dimensions, num of training examples)
        Y: (n_y, m) = (classes, num of training examples)
                        Y comes with hot-encoding
        """
        self.costs = []
        t = 0   # time step for adam optimization

        # loop over iterations
        for epoch in range(self.n_epochs):

            # Step 1) forward propagation (including Dropout)
            A = self.forward_propagation(X)

            # Step 2) compute cost (cross entropy + L2 regularization)
            cost = self.compute_cost(A, Y)

            # Step 3) backward propagation (including Dropout and L2 regularization)
            self.backward_propagation(A, Y)

            # Step 4) update weights and bias: [gd(*), momentum, adam]
            if self.optimizer == 'gd':
                self.update_params()
            elif self.optimizer == 'momentum':
                self.update_params_with_momentum()
            elif self.optimizer == 'adam':
                t += 1
                self.update_params_with_adam(t)
            else:
                self.update_params()

            if epoch % self.step_size == 0:
                if verbose:
                    print("{}, cost = {:.6f}".format(epoch, cost))
                self.costs.append(cost)

    def plot_cost(self, savefig=False):
        """
        plot the cost over iterations
        """
        plt.plot(self.costs)
        plt.ylabel('cost')
        plt.xlabel('epoch (per {}s)'.format(self.step_size))
        plt.title("Learning rate = " + str(self.lr))

        if savefig:
            plt.savefig("cost.png")
        else:
            plt.show()

    def predict(self, X):
        """
        For a given X, predict Y (one-hot encoded)
        """
        A = self.forward_propagation(X)
        Y_pred = np.zeros((A.shape[0], A.shape[1]))

        # loop along with columns (training examples)
        # find the index with the largest value
        for j in range(A.shape[1]):
            idx = np.argmax(A[:, j])
            Y_pred[idx][j] = 1

        return Y_pred

    def initialize_params(self, initializer):
        """
        initialize weights(W) and biases(b)

        input: [random, xavier, he(*)]
        """
        # for reproducibility
        np.random.seed(self.random_seed)

        # get the length of layer (exclude the input layer by definition)
        L = len(self.layer_dims) - 1

        layer_dims = self.layer_dims  # for shortcut coding
        random_init_scale = 0.01  # help random values closer to zero (tunable?)

        # loop through each layer [1, 2, ... L]
        for l in range(1, L + 1):

            # weights initialization with [random, xavier, he(*)]
            if initializer == 'random':
                W = np.random.randn(layer_dims[l], layer_dims[l - 1]) * random_init_scale
            elif initializer == 'xavier':
                W = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(1. / layer_dims[l - 1])
            elif initializer == 'he':
                W = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2. / layer_dims[l - 1])
            else:
                W = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2. / layer_dims[l - 1])

            # bias initialization with zeros
            b = np.zeros((self.layer_dims[l], 1))

            # set the initial parameters
            self.params['W' + str(l)] = W
            self.params['b' + str(l)] = b

    def initialize_params_for_optimization(self, optimizer):
        """
        initialize first and second parameters for optimization

        input: [gd(*), momentum]
        """
        # get the length of layer (exclude the input layer by definition)
        L = len(self.params) // 2

        # first momentum for momentum and adam
        if optimizer == 'momentum' or optimizer == 'adam':
            for l in range(1, L + 1):  # [1, 2, .., L]
                self.v["dW" + str(l)] = \
                    np.zeros((self.params["W" + str(l)].shape[0], self.params["W" + str(l)].shape[1]))
                self.v["db" + str(l)] = \
                    np.zeros((self.params["b" + str(l)].shape[0], self.params["b" + str(l)].shape[1]))

        # second momentum for adam
        if optimizer == 'adam':
            for l in range(1, L + 1):  # [1, 2, .., L]
                self.s["dW" + str(l)] = np.zeros((self.params["W" + str(l)].shape[0], self.params["W" + str(l)].shape[1]))
                self.s["db" + str(l)] = np.zeros((self.params["b" + str(l)].shape[0], self.params["b" + str(l)].shape[1]))

    def evaluate(self, X, Y):
        """
        report the accuracy between ground truth(Y) and prediction(Y_pred)
        """
        # get the prediction
        Y_pred = self.predict(X)

        Y_pred_idx = np.argmax(Y_pred, axis=0)
        Y_idx = np.argmax(Y, axis=0)
        accuracy = np.mean(np.equal(Y_pred_idx, Y_idx))

        # print("Accuracy = {:.2f}%".format(accuracy * 100))
        return accuracy * 100

    def forward_propagation(self, X):
        """
        For a given X, do a forward propagation

        Basic process for intermediate layers
            Z[l] = W[l] A[l-1] + b[l], where A[0] = X
            A[l] = act_func(Z[l])
            # Dropout
            D[l] = (randomly generate D[l] < keep_prob)
            A[l] = A[l] * D[l] # shutdown some neurons


        Output layer:
            A[L] = softmax(Z[L])
        """
        # for reproducibility
        np.random.seed(self.random_seed)

        # get the length of layers
        L = len(self.params) // 2

        # A0: an alias for X (input to a Neural Network)
        self.caches['A0'] = X

        # loop through each layer [1, 2, ..., L]
        for l in range(1, L + 1):

            # shortcuts for parameters at the current layer and activation value from the previous layer
            W = self.params['W' + str(l)]
            b = self.params['b' + str(l)]
            A = self.caches['A' + str(l - 1)]

            # affine transformation: Z^{l} = W^{l} A^{l-1} + b^{l}
            Z = np.dot(W, A) + b

            # softmax for the last (output) layer (multi-class classifier)
            if l == L:
                self.caches['A' + str(l)] = softmax(Z)
            # nonlinear activation function for intermediate (hidden) layers
            else:
                A = self.act_func(Z)
                # Dropout: shutdown some neurons
                D = np.random.rand(A.shape[0], A.shape[1])
                D = (D < self.keep_prob)
                A = A * D
                A = A / self.keep_prob  # inverted dropout

                # keep track of intermediate data for back propagation
                self.caches['Z' + str(l)] = Z
                self.caches['A' + str(l)] = A
                self.caches['D' + str(l)] = D

        return self.caches['A' + str(L)]

    def compute_cost(self, A, Y):
        """
        compute cross_entropy + L2-regularization (if lambd > 0)

        A: forwarded computation
        Y: ground truth
        """
        assert A.shape == Y.shape
        m = A.shape[1]

        # compute the cross entropy
        cross_entropy_cost = cross_entropy(A, Y)

        # L2-regularization term (if any, lambd > 0)
        l2_cost = 0
        if self.lambd == 0:
            pass
        else:
            l2_cost = 0
            L = len(self.params) // 2
            for l in range(1, L + 1):  # [1, 2, .. L]
                l2_cost += np.sum(np.square(self.params['W' + str(l)]))
            l2_cost = (self.lambd / (2. * float(m))) * l2_cost

        return cross_entropy_cost + l2_cost

    def backward_propagation(self, A, Y):
        """
        A: forwarded computation
        Y: ground truth

        Output layer:
            dZ[L]: A - Y (derivative of L(loss function) respective to Z[L])

        Basic process for intermediate layers
            l2_grads = (lamb/m) * W[l]
            dW[l] = (1/m) * np.dot(dZ[l],  A[l-1].T) + l2_grads
            db[l] = (1/m) * np.sum(dZ[l])

            dA[l-1] = np.dot(W[l].T, dZ[l])
            dA[l-1] = dA[l-1] * D[l-1] # Dropout
            dA[l-1] = dA[l-1] / keep_prob # inverted
            dZ[l-1] = dA[l-1] * g'(Z[l-1]) # g' : derivative of activation function
        """
        assert A.shape == Y.shape
        m = A.shape[1]

        L = len(self.params) // 2

        dZ = A - Y  # shortcut for cross-entropy (dL/dA * dA/dZ)
        self.grads['dZ' + str(L)] = dZ

        # loop through layers in the reversed order (since this is BACK propagation!!!) [L, L-1, ..., 2, 1]
        for l in range(L, 0, -1):
            # derive dW[l] and db[l] from dZ[l]
            A = self.caches['A' + str(l - 1)]
            l2_grads = (self.lambd / float(m)) * self.params['W' + str(l)]
            dW = np.dot(dZ, A.T) / float(m) + l2_grads
            db = np.sum(dZ, axis=1, keepdims=True) / float(m)

            self.grads['dW' + str(l)] = dW
            self.grads['db' + str(l)] = db

            # break out the rest part at the first hidden layer
            # since we don't need to compute dZ[0] (and dA[0])
            if l == 1:
                break

            # before moving to the previous layer, compute dZ[l-1] (also dA[l-1] implicitly)
            W = self.params['W' + str(l)]
            Z = self.caches['Z' + str(l - 1)]
            D = self.caches['D' + str(l - 1)]
            dA = np.dot(W.T, dZ)
            dA = dA * D
            dA = dA / self.keep_prob
            dZ = np.multiply(dA, self.d_act_func(Z))
            self.grads['dZ' + str(l-1)] = dZ
            self.grads['dA' + str(l-1)] = dA

    def update_params(self):
        """
        Gradient Descent Optimization
        """
        L = len(self.params) // 2

        for l in range(1, L + 1):  # [1, 2, .., L]
            self.params["W" + str(l)] = self.params["W" + str(l)] - self.lr * self.grads["dW" + str(l)]
            self.params["b" + str(l)] = self.params["b" + str(l)] - self.lr * self.grads["db" + str(l)]

    def update_params_with_momentum(self):
        """
        Momentum Optimization
        - use velocity
        """
        L = len(self.params) // 2

        for l in range(1, L + 1):  # [1, 2, .., L]
            # compute first momentum (velocity)
            self.v["dW" + str(l)] = self.beta * self.v["dW" + str(l)] + (1 - self.beta) * self.grads["dW" + str(l)]
            self.v["db" + str(l)] = self.beta * self.v["db" + str(l)] + (1 - self.beta) * self.grads["db" + str(l)]

            # update parameters
            self.params["W" + str(l)] = self.params["W" + str(l)] - self.lr * self.v["dW" + str(l)]
            self.params["b" + str(l)] = self.params["b" + str(l)] - self.lr * self.v["db" + str(l)]

    def update_params_with_adam(self, t):
        """
        Adam Optimization
        - use the first momentum (velocity)
        - use the second momentum (squared gradients)
        - use bias correction for the first few iterations
        """
        L = len(self.params) // 2

        v_corrected = dict()
        s_corrected = dict()

        # loop through layers
        for l in range(1, L + 1):
            # Moving average of the gradients
            self.v["dW" + str(l)] = self.beta1 * self.v["dW" + str(l)] + (1 - self.beta1) * self.grads["dW" + str(l)]
            self.v["db" + str(l)] = self.beta1 * self.v["db" + str(l)] + (1 - self.beta1) * self.grads["db" + str(l)]

            # Compute bias-corrected first moment estimate
            v_corrected["dW" + str(l)] = self.v["dW" + str(l)] / (1 - self.beta1 ** t)
            v_corrected["db" + str(l)] = self.v["db" + str(l)] / (1 - self.beta1 ** t)

            # Moving average of the squared gradients
            self.s["dW" + str(l)] = self.beta2 * self.s["dW" + str(l)] + (1 - self.beta2) * (
                    self.grads["dW" + str(l)] ** 2)
            self.s["db" + str(l)] = self.beta2 * self.s["db" + str(l)] + (1 - self.beta2) * (
                    self.grads["db" + str(l)] ** 2)

            # Compute bias-corrected second raw moment estimate
            s_corrected["dW" + str(l)] = self.s["dW" + str(l)] / (1 - self.beta2 ** t)
            s_corrected["db" + str(l)] = self.s["db" + str(l)] / (1 - self.beta2 ** t)

            # Update parameters
            self.params["W" + str(l)] = self.params["W" + str(l)] - self.lr * np.divide(
                v_corrected["dW" + str(l)], np.sqrt(s_corrected["dW" + str(l)]) + self.epsilon)
            self.params["b" + str(l)] = self.params["b" + str(l)] - self.lr * np.divide(
                v_corrected["db" + str(l)], np.sqrt(s_corrected["db" + str(l)]) + self.epsilon)


def cross_validation(kfold):

    folded_data = k_fold(kfold)
    #print(len(folded_data))

    train_accs = []
    test_accs = []
    for k in range(kfold):
        train_X = folded_data['train_X' + str(k)]
        train_label = folded_data['train_Y' + str(k)]
        test_X = folded_data['test_X' + str(k)]
        test_label = folded_data['test_Y' + str(k)]
        #print(train_X.shape, train_label.shape, test_X.shape, test_label.shape)

        train_Y = one_hot(train_label, 3)
        test_Y = one_hot(test_label, 3)
        #print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)

        print("{}-th validation".format(k+1))

        layer_dims = [train_X.shape[0], 8, 5, train_Y.shape[0]]
        clf = SoftmaxRegression(layer_dims, lr=0.01, n_epochs=3000, optimizer='adam', act_func='sigmoid')

        clf.fit(train_X, train_Y)
        clf.plot_cost()
        train_acc = clf.evaluate(train_X, train_Y)
        test_acc = clf.evaluate(test_X, test_Y)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print("  training accuracy = {:.2f}%".format(train_acc))
        print("  testing accuracy = {:.2f}%".format(test_acc))
    print("-" * 10)
    print("training accuracy on average = {:.2f}%".format(np.mean(train_accs)))
    print("testing accuracy on average = {:.2f}%".format(np.mean(test_accs)))


if __name__ == '__main__':

    version_check()

    ####################
    # example of single evaluation
    ####################
    print()
    print("#" * 30)
    print("# Example of single evaluation")
    print("#" * 30)
    # load a dataset
    train_X, train_label, test_X, test_label, classes = load_data_iris(5)
    train_Y = one_hot(train_label, 3)
    test_Y = one_hot(test_label, 3)

    # layer configuration [input, hidden, ..., output]
    layer_dims = [train_X.shape[0], 8, 5, train_Y.shape[0]]

    # create a classifier from Neural Network
    clf = SoftmaxRegression(layer_dims, lr=0.01, n_epochs=3000, optimizer='adam', act_func='sigmoid')

    # training
    clf.fit(train_X, train_Y, verbose=True)

    # plot cost
    clf.plot_cost()

    # evaluation
    train_acc = clf.evaluate(train_X, train_Y)
    test_acc = clf.evaluate(test_X, test_Y)
    print("training accuracy = {:.2f}%".format(train_acc))
    print("testing accuracy = {:.2f}%".format(test_acc))

    ####################
    # example of cross validation
    ####################
    print()
    print("#" * 30)
    print("# Example of cross validation")
    print("#" * 30)
    cross_validation(5)

