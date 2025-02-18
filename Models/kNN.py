# -*- coding = utf-8 -*-
# @Time: 2025/2/18 09:45
# @Author: Zhihang Yi
# @File: kNN.py
# @Software: PyCharm

import torch

class kNN:
    def __init__(self, train_features, train_labels):
        """
        simply remember the training features and labels

        :param train_features: tensor with a size of m * n
        :param train_labels: label of each row of train_features
        """
        self.features = train_features
        self.labels = train_labels

    def predict(self, test_features, k=1):
        """
        predict the labels of the given test_features,
        using the Euclidean distance to measure the nearest neighbors

        :param test_features: tensor with a size of x * n
        :param k: the number of nearest neighbors
        :return: the labels of the given test_features with a size of x * 1
        """
        m, n, x = self.features.shape[0], self.features.shape[1], test_features.shape[0]

        # a tensor with a size of m * x, where each cell is
        # the distance bwtween training sample and test sample
        dists = (self.features ** 2).sum(dim=1).reshape(1, -1) + (test_features ** 2).sum(dim=1) - 2 * (test_features @ self.features.T)

        predicted_labels = []

        for col in range(m):
            distances, indices = torch.topk(dists[:, col], k=k, largest=False)
            target_labels = self.labels[indices]
            label, frequency = target_labels.mode()
            predicted_labels.append(label)

        predicted_labels = torch.tensor(predicted_labels)

        return predicted_labels

    def check_accuracy(self, test_features, test_labels, k=1):
        """
        check the accuracy of the model

        :param test_features: tensor with a size of m * n
        :param test_labels: tensor with a size of m * 1
        :param k: k nearest neighbors
        :return: the accuracy of the model
        """
        predicted_labels = self.predict(test_features, k)

        m = test_labels.shape[0]
        correct = 0

        for i in range(m):
            if predicted_labels[i] == test_labels[i]:
                correct += 1

        accuracy = correct / m

        return accuracy

    def cross_validation(self, folds, k_choices):
        """
        cross validation on the model

        :param folds: the number of folds on the whole dataset
        :param k_choices: choices of the hyperparameter k.
        :return: dictionary of k to the accuracy list on different folds
        """
        features_folds = self.features.chunk(folds)
        labels_folds = self.labels.chunk(folds)

        k_to_accuracies = {}

        for k in k_choices:
            accuracies = []

            for i in range(folds):
                train_features = torch.cat([features_folds[j] for j in range(folds) if i != j], dim=0)
                train_labels = torch.cat([labels_folds[j] for j in range(folds) if i != j], dim=0)

                validation_features = features_folds[i]
                validation_labels = labels_folds[i]

                net = kNN(train_features, train_labels)
                accuracy = net.check_accuracy(validation_features, validation_labels, k)

                accuracies.append(accuracy)

            k_to_accuracies[k] = accuracies

        return k_to_accuracies
