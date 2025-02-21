# -*- coding = utf-8 -*-
# @Time: 2025/2/21 15:51
# @Author: Zhihang Yi
# @File: LinearClassifier.py
# @Software: PyCharm

import torch

class LinearClassifier:
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, regularization=1e-5, epochs=1000, batch=128):
        """
        steps to train the model

        :param X: training features
        :param y: training labels
        :param learning_rate: number of learning rate
        :param regularization: number of regularization
        :param epochs: number of epochs
        :param batch: batch size
        :return: loss_history
        """
        from SVMClassifier import SVMClassifier
        from SoftmaxClassifier import SoftmaxClassifier

        m, n = X.shape
        c = y.max() + 1

        loss_history = []

        if self.W is None:
            self.W = torch.rand(n, c)

        for i in range(epochs):
            X_batch, y_batch = self.sample_data(X, y, batch)
            if isinstance(self, SVMClassifier):
                loss, dW = self.loss(X_batch, y_batch, delta, regularization)
            elif isinstance(self, SoftmaxClassifier):
                loss, dW = self.loss(X_batch, y_batch, regularization)

            self.W -= learning_rate * dW
            loss_history.append(loss)

        return loss_history

    def predict(self, X):
        """
        predict labels based on the trained model

        :param X: testing features, shape: (m, n)
        :return: predicted labels, shape: (m,)
        """
        scores = X @ self.W
        y_pred = scores.argmax(dim=1)
        return y_pred

    def accuracy(self, X, y):
        """
        compute the accuracy of predicted labels based on trained W

        :param X: validation features
        :param y: validation labels
        :return: accuracy
        """
        y_pred = self.predict(X)

        m = X.shape[0]
        correct = y_pred[y_pred == y].shape[0]

        return correct / m

    def sample_data(self, X, y, batch):
        """
        return batch number of X and y

        :param X: training features
        :param y: training labels
        :param batch: batch size
        :return: X and y in batch
        """
        m, n = X.shape
        c = y.max() + 1

        indices = torch.randint(0, m, (batch,))

        return X[indices, :], y[indices]
