# -*- coding = utf-8 -*-
# @Time: 2025/2/27 14:46
# @Author: Zhihang Yi
# @File: TwoLayerNet.py
# @Software: PyCharm

import torch

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        """
        initialize weight matrices W1 and W2 and biases b1 and b2

        :param input_size: the dimension of input training features
        :param hidden_size:the number of neurons in hidden layer
        :param output_size: the number of classes for classification
        """
        self.W1 = torch.randn(input_size, hidden_size)
        self.W2 = torch.randn(hidden_size, output_size)
        self.b1 = torch.randn(hidden_size)
        self.b2 = torch.randn(output_size)

    def train(self, X, y, lr=1e-3, reg=1, epochs=100, batch=128):
        """
        training process of two-layer neural network

        :param X:
        :param y:
        :param lr:
        :param reg:
        :param epochs:
        :param batch:
        :return: loss history of the model
        """
        loss_history = []

        for i in range(epochs):
            X_batch, y_batch = self.sample_data(X, y, batch)
            loss, gradients = loss(X_batch, y_batch, reg)

            self.W1 -= lr * gradients['dW1']
            self.W2 -= lr * gradients['dW2']
            self.b1 -= lr * gradients['db1']
            self.b2 -= lr * gradients['db2']

            loss_history.append(loss)

        return loss_history

    def loss(self, X, y, reg=1):
        """
        compute the loss and gradients of W1, W2, b1 and b2

        :param X:
        :param y:
        :param reg:
        :return: total loss over the batch and grads
        """
        N = X.shape[0]

        # compute the loss

        feature_map = X @ self.W1 + self.b1.view(-1, 1)  # (N, H)
        feature_map[feature_map < 0] = 0  # ReLU

        scores = feature_map @ self.W2 + self.b2.view(-1, 1)  # (N, C)
        socres -= scores.max(dim=1).values.view(-1, 1)  # avoid overflow

        softmax_scores = scores.exp() / scores.exp().sum(dim=1).view(-1, 1)
        correct_softmax_scores = softmax_scores[range(N), y]

        loss = (-1 * correct_softmax_scores.log()).sum() / N  # scalar
        loss += reg * ((self.W1 ** 2).sum() + (self.W2 ** 2).sum())  # leave alone the bias regularization

        # compute the gradients

        softmax_scores[range(N), y] -= 1
        dscores = softmax_scores / N

        dW2 = feature_map.T @ dscores + reg * 2 * self.W2
        db2 = dscores.sum(dim=0)

        dhiddenlayer = dscores @ self.W2.T
        dhiddenlayer[dhiddenlayer <= 0] = 0

        dW1 = X.T @ dhiddenlayer + reg * 2 * self.W1
        db1 = dhiddenlayer.sum(dim=0)

        gradients = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

        return loss, gradients

    def sample_data(self, X, y, batch=128):
        """
        return batch number of X and y

        :param X: training features
        :param y: training labels
        :param batch: batch size
        :return: X and y in batch
        """
        m = X.shape[0]
        indices = torch.randint(0, m, (batch,))
        return X[indices, :], y[indices]

    def predict(self, X):
        feature_map = X @ self.W1 + self.b1.view(-1, 1)  # (N, H)
        feature_map[feature_map < 0] = 0  # ReLU
        scores = feature_map @ self.W2 + self.b2.view(-1, 1)  # (N, C)

        return scores.argmax(dim=1)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        N = X.shape[0]
        correct = y[y == y_pred].shape[0]
        return correct / N
