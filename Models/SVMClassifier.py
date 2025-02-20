# -*- coding = utf-8 -*-
# @Time: 2025/2/20 11:45
# @Author: Zhihang Yi
# @File: SVMClassifier.py
# @Software: PyCharm

import torch

class SVMClassifier:
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
        m, n = X.shape
        c = y.shape[1]

        loss_history = []

        if self.W is None:
            self.W = torch.rand(n, c)

        for i in range(epochs):
            X_batch, y_batch = self.sample_data(X, y, batch)
            loss, dW = self.svm_loss(X_batch, y_batch, delta, regularization)

            self.W -= learning_rate * dW
            loss_history.append(loss)

        return loss_history

    def svm_loss(self, X, y, delta=1, regularization=1e-5):
        """
        compute the total SVM loss and gradient of weight matrix W

        :param X: training features
        :param y: training labels
        :param delta:
        :param regularization:
        :return: total SVM loss and gradient of W
        """
        m, n = X.shape
        c = y.shape[1]

        scores = X @ self.W  # (m, c)
        correct_class_scores = scores[range(m), y]  # (m,)
        margins = scores - correct_class_scores.view(-1, 1) + delta

        loss = margins[margins > 0].sum() / m  # scalar
        loss += regularization * (self.W * self.W).sum()  # add regularization term

        binary = (margins > 0).to(X.dtype)  # (m, c)
        row_sum = binary.sum(dim=1)  # (m,)
        binary[range(m), y] -= row_sum  # (m, c)

        dW = (X.T @ binary) / m
        dW += 2 * self.W * regularization  # add regularization term

        return loss, dW

    def sample_data(self, X, y, batch):
        """
        return batch number of X and y

        :param X: training features
        :param y: training labels
        :param batch: batch size
        :return: X and y in batch
        """
        m, n = X.shape
        c = y.shape[1]

        indices = torch.randint(0, m, (batch,))

        return X[indices, :], y[indices]

    def predict(self, X):
        """
        predict the labels of X based on trained W

        :param X: testing features
        :return: predicted labels
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
        m = X.shape[0]
        correct = 0

        y_pred = self.predict(X)

        for i in range(m):
            if y_pred[i] == y[i]:
                correct += 1

        return correct / m
