# -*- coding = utf-8 -*-
# @Time: 2025/2/21 15:35
# @Author: Zhihang Yi
# @File: SoftmaxClassifier.py
# @Software: PyCharm

import torch
from LinearClassifier import LinearClassifier

class SoftmaxClassifier(LinearClassifier):
    def loss(self, X, y, regularization):
        """
        compute the total cross-entropy loss and gradient of weight matrix W

        :param X: training features, shape: (m, n)
        :param y: training labels
        :param regularization:
        :return: total cross-entropy loss and gradient of W
        """
        m, n = X.shape
        c = y.max() + 1

        scores = X @ self.W
        scores -= scores.max(dim=1).values.view(-1, 1)  # 减去每一行最大的值，防止exp后溢出
        softmax_scores = scores.exp() / scores.exp().sum(dim=1).view(-1, 1)  # (m, c)

        correct_softmax_scores = softmax_scores[range(m), y]

        loss = (-1 * correct_softmax_scores.sum()) / m
        loss += regularization * (self.W * self.W).sum()

        softmax_scores[range(m), y] -= 1

        dW = X.T @ softmax_scores
        dW += regularization * 2 * self.W

        return loss, dW
