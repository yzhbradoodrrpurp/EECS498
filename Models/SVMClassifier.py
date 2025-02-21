# -*- coding = utf-8 -*-
# @Time: 2025/2/20 11:45
# @Author: Zhihang Yi
# @File: SVMClassifier.py
# @Software: PyCharm

import torch
from LinearClassifier import LinearClassifier

class SVMClassifier(LinearClassifier):
    def loss(self, X, y, delta=1, regularization=1e-5):
        """
        compute the total SVM loss and gradient of weight matrix W

        :param X: training features
        :param y: training labels
        :param delta:
        :param regularization:
        :return: total SVM loss and gradient of W
        """
        m, n = X.shape
        c = y.max() + 1

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
