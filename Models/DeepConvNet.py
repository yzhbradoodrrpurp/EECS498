# -*- coding = utf-8 -*-
# @Time: 2025/3/7 19:07
# @Author: Zhihang Yi
# @File: DeepConvNet.py
# @Software: PyCharm

import torch

class DeepConvNet:
    def __init__(
            self, dimensions=(3, 32, 32), filters=(8, 8, 8, 8, 8, 8), poolings=(0, 2, 4),
            batchnorm=False, classes=10, dtype=torch.float, device='cpu'
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
        self.layers = len(filters) + 3
        self.poolings = poolings
        self.batchnorm = batchnorm

        self.params = {}

        C, H, W = dimensions

        # initialize the weights and biases for convolutional layers
        for layer in range(self.layers - 3):
            if layer == 0:
                self.params[f'W{layer + 1}'] = torch.randn((C, 3, 3)) * math.sqrt((2 / C * 3 * 3), dtype=dtype, device=device)
                self.params[f'b{layer + 1}'] = torch.zeros(C, dtype=dtype, device=device)

                if batchnorm:
                    self.params[f'gamma{layer + 1}'] = torch.ones(C, dtype=dtype, device=device)
                    self.params[f'beta{layer + 1}'] = torch.zeros(C, dtype=dtype, device=device)
            else:
                self.params[f'W{layer + 1}'] = torch.randn(filters[layer - 1], 3, 3) * math.sqrt((2 / C * 3 * 3), dtype=dtype, device=device)
                self.params[f'b{layer + 1}'] = torch.zeros(filters[layer - 1], dtype=dtype, device=device)

                if batchnorm:
                    self.params[f'gamma{layer + 1}'] = torch.ones(filters[layer - 1], dtype=dtype, device=device)
                    self.params[f'beta{layer + 1}'] = torch.zeros(filters[layer - 1], dtype=dtype, device=device)

        H_fc, W_fc = H // (2 ** len(poolings)), W // (2 ** len(poolings))
        neurons = H_fc * W_fc * filters[-1]  # the total neuron numbers when the image is flattened

        # initialize the weights and biases for fully-connected layers
        self.params[f'W{self.layers - 2}'] = torch.randn(neurons, 100, dtype=dtype, device=device) * math.sqrt(2 / neurons)
        self.params[f'b{self.layers - 2}'] = torch.randn(100, dtype=dtype, device=device)

        self.params[f'W{self.layers - 1}'] = torch.randn(100, 100, dtype=dtype, device=device) * math.sqrt(2 / 100)
        self.params[f'b{self.layers - 1}'] = torch.randn(100, dtype=dtype, device=device)

        self.params[f'W{self.layers}'] = torch.randn(100, classes, dtype=dtype, device=device) * math.sqrt(2 / 100)
        self.params[f'b{self.layers}'] = torch.randn(classes, dtype=dtype, device=device)

        if batchnorm:
            self.params[f'gamma{self.layers - 2}'] = torch.ones(100, dtype=dtype, device=device)
            self.params[f'beta{self.layers - 2}'] = torch.zeros(100, dtype=dtype, device=device)

            self.params[f'gamma{self.layers - 1}'] = torch.ones(100, dtype=dtype, device=device)
            self.params[f'beta{self.layers - 1}'] = torch.zeros(100, dtype=dtype, device=device)

            self.params[f'gamma{self.layers}'] = torch.ones(classes, dtype=dtype, device=device)
            self.params[f'beta{self.layers}'] = torch.zeros(classes, dtype=dtype, device=device)
