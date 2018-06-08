import numpy as np
from utils import sigmoid


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = np.zeros((1, 1))
    return w, b


class LogisticRegression:
    def __init__(self, n_x, n_epochs=3000, lr=0.001):
        """
        n_x: num of features
        n_epochs: num of iterations for optimization
        lr: learning rate for updating parameters
        """
        self.n_x = n_x
        self.n_epochs = n_epochs
        self.lr = lr

        # initialize parameters
        self.w, self.b = initialize_with_zeros(n_x)
        assert self.w.shape == (self.n_x, 1)
        assert self.b.shape == (1, 1)

    def fit(self, X, Y):
        """
        X: (n_x, m)
        Y: (1, m)
        """
        assert X.shape[0] == self.n_x and Y.shape[0] == 1
        assert X.shape[1] == Y.shape[1]

        m = X.shape[1]

        for epoch in range(self.n_epochs):
            # Step 1: forward propagation
            Z = np.dot(self.w.T, X) + self.b
            A = sigmoid(Z)

            # Step 2: loss (cross entropy)
            cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A), axis=1, keepdims=True) / float(m)
            cost = np.squeeze(cost)

            # Step 3: backward propagation
            dZ = A - Y
            dw = np.dot(X, dZ.T) / float(m)
            db = np.sum(dZ, axis=1, keepdims=True) / float(m)

            assert dw.shape == self.w.shape
            assert db.shape == self.b.shape

            # Step 4: update parameters (gradient descent)
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db

            # log
            if epoch % 100 == 0:
                print("{}, cost = {:.7f}".format(epoch, cost))

    def predict(self, X):
        Z = np.dot(self.w.T, X) + self.b
        A = sigmoid(Z)

        m = X.shape[1]
        Y_pred = np.zeros((1, m))
        for i in range(A.shape[1]):
            if A[0, i] >= 0.5:
                Y_pred[0, i] = 1
            else:
                Y_pred[0, i] = 0

        return Y_pred

    def evaluate(self, X, Y):
        assert X.shape[1] == Y.shape[1]

        m = X.shape[1]

        Z = np.dot(self.w.T, X) + self.b
        A = sigmoid(Z)

        Y_pred = np.zeros((1, m))
        for i in range(A.shape[1]):
            if A[0, i] >= 0.5:
                Y_pred[0, i] = 1
            else:
                Y_pred[0, i] = 0

        accuracy = (100 - np.mean(np.abs(Y_pred - Y)) * 100)
        print("Accuracy: {:.7f}%".format(accuracy))

    def forward(self, X):
        Z = np.dot(self.w.T, X) + self.b
        A = sigmoid(Z)
        return A

    def loss_function(self, A, Y):
        assert A.shape == Y.shape
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A), axis=1, keepdims=True) / float(m)
        return np.squeeze(cost)  # [[cost]] --> cost


if __name__ == '__main__':
    from utils import load_dataset_cat

    train_orig_X, train_Y, test_orig_X, test_Y, classes = load_dataset_cat()

    train_X_flatten = train_orig_X.reshape(train_orig_X.shape[0], -1).T
    test_X_flatten = test_orig_X.reshape(test_orig_X.shape[0], -1).T

    train_X = train_X_flatten / 255.
    test_X = test_X_flatten / 255.

    n_x = train_X.shape[0]

    clf = LogisticRegression(
        n_x,
        n_epochs=2000,
        lr=0.005)

    clf.fit(train_X, train_Y)

    clf.evaluate(train_X, train_Y)
    clf.evaluate(test_X, test_Y)

    test_image_idx = 0
    test_image = test_X[:, test_image_idx].reshape(test_X.shape[0], 1)  # (n_x, m=1)
    test_label = clf.predict(test_image)  # (1, m=1)
    print(np.squeeze(test_label))
