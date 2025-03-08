# -*- coding = utf-8 -*-
# @Time: 2025/3/7 19:07
# @Author: Zhihang Yi
# @File: DeepConvNet.py
# @Software: PyCharm

import torch
from Layers import ReLU, SoftmaxLoss, Convolution, BatchNorm, MaxPooling, Linear

class DeepConvNet:
    def __init__(
            self, dimensions=(3, 32, 32), filters=(8, 8, 8, 8, 8, 8), poolings=(0, 2, 4),
            batchnorm=False, classes=10, epochs=20, batch=128, dtype=torch.float, device='cpu'
    ):
        """

        [conv -> (batchnorm) -> ReLU -> (pooling)] * N -> [linear -> (batchnorm) -> ReLU] * 3 -> softmax

        kernel_size is set to 3
        stride is set to 1 and padding is  set to 1 to preserve the shape of hidden layer
        max pooling height, width and stride are all set to 2
        hidden size of fully-connected layers are all set to 100

        :param dimensions: the dimensions of the input images
        :param filters: numbers of filters in different convolutional layers
        :param poolings: layer indices to apply max pooling index 0
        :param batchnorm: whether to use batch normalization
        :param classes: number of classes to be divided into
        :param dtype: data type
        :param device: device to run on
        """
        self.dimensions = dimensions
        self.num_layers = len(filters) + 3
        self.poolings = poolings
        self.batchnorm = batchnorm
        self.epochs = epochs
        self.batch = batch

        self.sequence = []

        C, H, W = dimensions

        # initialize the weights and biases for convolutional layers
        for layer in range(self.num_layers - 3):
            if layer == 0:
                convolutional_layer = Convolution(filters[layer], C, 3, 3, stride=1, padding=1, dtype=dtype, device=device)
                self.sequence.append(convolutional_layer)

                if batchnorm:
                    batch_normalization = BatchNorm(C, mode='train')
                    self.sequence.append(batch_normalization)
            else:
                convolutional_layer = Convolution(filters[layer], filters[layer - 1], 3, 3, stride=1, padding=1, dtype=dtype, device=device)
                self.sequence.append(convolutional_layer)

                if batchnorm:
                    batch_normalization = BatchNorm(filters[layer - 1], mode='train')
                    self.sequence.append(batch_normalization)

        H_fc, W_fc = H // (2 ** len(poolings)), W // (2 ** len(poolings))
        neurons = H_fc * W_fc * filters[-1]  # the total neuron numbers when the image is flattened

        # initialize the weights and biases for fully-connected layers
        self.sequence.append(Linear(neurons, 100))
        if batchnorm:
            self.sequence.append(BatchNorm(100, mode='train'))
        self.sequence.append(Linear(100, 100))
        if batchnorm:
            self.sequence.append(BatchNorm(100, mode='train'))
        self.sequence.append(Linear(100, classes))
        if batchnorm:
            self.sequence.append(BatchNorm(classes, mode='train'))

    def train(self, X, y):
        """

        :param X: Input images (N, C, H, W)
        :param y: the labels of input iamges (N,)
        :return: the loss history of the network
        """
        loss_history = []

        # forward pass
        for epoch in range(self.epochs):
            X_batch, y_batch = self.sample_data(X, y)
            scores = self.forward(X_batch, 'train')
            loss = SoftmaxLoss.forward(scores, y_batch)

            loss_history.append(loss.item())

            loss.backward()

            self.update()  # SGD update

        return loss_history

    def predict(self, X):
        scores = self.forward(X, 'test')
        y_pred = scores.argmax(dim=1)
        return y_pred

    def forward(self, X, mode='train'):
        hidden_layer = X.clone()

        for layer in self.sequence:
            if isinstance(layer, BatchNorm):
                layer.mode = mode
            hidden_layer = layer.forward(hidden_layer)

        scores = hidden_layer.clone()

        return scores

    def sample_data(self, X, y):
        N = X.shape[0]
        indices = torch.randint(0, N, (self.batch,))
        return X[indices], y[indices]

    def update(self):
        with torch.no_grad():
            for layer in self.sequence:
                if layer.is_parametric:
                    layer.update()

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        total = len(y)
        correct = len(y[y == y_pred])
        return correct / total
