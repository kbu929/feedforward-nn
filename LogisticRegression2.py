import numpy as np
from utils import sigmoid, d_sigmoid, tanh, d_tanh
from utils import load_dataset_planar, plot_decision_boundary
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, n_x, n_h, n_y, n_epochs, lr):
        assert n_y == 1  # binary classification
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        self.n_epochs = n_epochs
        self.lr = lr
        # parameters, gradients, caches for optimization
        self.params = dict()
        self.caches = dict()
        self.grads = dict()
        # log
        self.costs = list()
        self.step_size = n_epochs / 20
        # initialize parameters
        self._initialize_params(n_x, n_h, n_y)

    def fit(self, X, Y):
        for epoch in range(self.n_epochs):
            # Step 1) forward propagation
            AL = self._forward_propagation(X)

            # Step 2) compute cost
            cost = self._cost_function(AL, Y)

            # Step 3) backward propagation
            self._backward_propagation(X, Y)

            # Step 4) update parameters
            self._update_parameters()

            if epoch % self.step_size == 0:
                print("{}, cost = {:.6f}".format(epoch, cost))
                self.costs.append(cost)

    def predict(self, X):
        AL = self._forward_propagation(X)
        Y_pred = np.zeros((AL.shape[0], AL.shape[1]))
        for i in range(AL.shape[1]):
            Y_pred[0][i] = 1 if AL[0][i] >= 0.5 else 0
        return Y_pred

    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        accuracy = np.mean(np.equal(Y_pred, Y)) * 100
        print("Accuracy = {:.2f}%".format(accuracy))

    def _initialize_params(self, n_x, n_h, n_y):
        np.random.seed(2)

        self.params['W1'] = np.random.randn(n_h, n_x) * 0.01
        self.params['b1'] = np.zeros((n_h, 1))
        self.params['W2'] = np.random.randn(n_y, n_h) * 0.01
        self.params['b2'] = np.zeros((n_y, 1))

    def _forward_propagation(self, X):
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        Z1 = np.dot(W1, X) + b1
        A1 = tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)

        self.caches['Z1'] = Z1
        self.caches['A1'] = A1
        self.caches['Z2'] = Z2
        self.caches['A2'] = A2

        return A2

    def _cost_function(self, A, Y):
        assert A.shape == Y.shape

        m = A.shape[1]

        # cross-entropy cost
        logprobs = np.multiply(Y, np.log(A)) + np.multiply((1-Y), np.log(1-A))
        cost = -np.sum(logprobs) / float(m)

        cost = np.squeeze(cost)
        assert cost.shape == ()  # scalar(float)

        return cost

    def _backward_propagation(self, X, Y):
        m = X.shape[1]

        W1 = self.params['W1']
        W2 = self.params['W2']
        Z1 = self.caches['Z1']
        A1 = self.caches['A1']
        Z2 = self.caches['Z2']
        A2 = self.caches['A2']

        # dZ2 = dA2 * dA2/dZ2
        # A2-Y = ( -Y/A2 + (1-Y)/(1-A2) ) * ( A2*(1-A2) )
        shortcut = True
        if shortcut:
            dZ2 = A2 - Y
        else:
            dA2 = -np.divide(Y, A2) + np.divide(1 - Y, 1 - A2) # -(Y/A2) + (1-Y) / (1-A2)
            dZ2 = np.multiply(dA2, d_sigmoid(Z2))  # dA2 * g'(Z2)
        dW2 = np.dot(dZ2, A1.T) / float(m)
        db2 = np.sum(dZ2, axis=1, keepdims=True) / float(m)
        #dZ1 = np.dot(W2.T, dZ2) * (1-np.power(A1, 2))
        dZ1 = np.dot(W2.T, dZ2) * d_tanh(Z1)  # dZ1 = dZA1 * dA1/dZ1 (=g'(Z1))
        dW1 = np.dot(dZ1, X.T) / float(m)
        db1 = np.sum(dZ1, axis=1, keepdims=True) / float(m)

        self.grads['dW2'] = dW2
        self.grads['db2'] = db2
        self.grads['dW1'] = dW1
        self.grads['db1'] = db1

    def _update_parameters(self):
        self.params['W1'] -= self.lr * self.grads['dW1']
        self.params['b1'] -= self.lr * self.grads['db1']
        self.params['W2'] -= self.lr * self.grads['dW2']
        self.params['b2'] -= self.lr * self.grads['db2']

    def report_cost(self):
        plt.plot(self.costs)
        plt.title('cost over iteration')
        plt.ylabel('cost')
        plt.xlabel('iterations (1 iter = {:d} epochs)'.format(int(self.step_size)))
        plt.show()


if __name__ == '__main__':

    train_X, train_Y = load_dataset_planar()
    n_x = train_X.shape[0]
    n_h = 4
    n_y = train_Y.shape[0]

    clf = LogisticRegression(n_x, n_h, n_y, n_epochs=10000, lr=1.2)

    clf.fit(train_X, train_Y)

    clf.evaluate(train_X, train_Y)

    plt.plot(clf.costs)
    plt.title('cost over iterations')
    plt.ylabel('cost')
    plt.xlabel('iterations (1 iter = {:d} epochs)'.format(int(clf.step_size)))

