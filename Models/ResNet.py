# -*- coding = utf-8 -*-
# @Time: 2025/3/11 19:07
# @Author: Zhihang Yi
# @File: ResNet.py
# @Software: PyCharm

import torch

# NOTE: 没有 shortcut 的普通传播块
class PlainBlock(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=8):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

    def forward(self, X):
        scores = self.model(X)
        return scores

# NOTE: 加上 shortcut 后的 Residual 传播块
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=8):
        super().__init__()

        self.block = PlainBlock(in_channels, out_channels)

        if in_channels == out_channels:
            # NOTE: `torch.nn.Identity` 对接受的变量不做任何改变
            self.shortcut = torch.nn.Identity()
        else:
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(1, 1))

    def forward(self, X):
        scores = self.block(X) + self.shortcut(X)
        return scores

# NOTE: 全连接层
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self, num_dimensions, hidden_size, num_classes):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_dimensions, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_classes),
        )

    def forward(self, X):
        scores = self.model(X)
        return scores

# NOTE: 由多个 Residual Blocks, 一个 Average Pooling layer, 一个 Flatten Layer 和全连接层组成的 ResNet
class ResNet(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=64, num_blocks=4):
        super().__init__()

        residual_blocks = []

        for block in range(num_blocks):
            if block == 0:
                residual_blocks.append(ResidualBlock(in_channels, out_channels))
            else:
                residual_blocks.append(ResidualBlock(out_channels, out_channels))

        self.residual_blocks = torch.nn.Sequential(*residual_blocks)
        self.fully_connected = FullyConnectedLayer(out_channels, 100, 10)

    def forward(self, X):
        N, C, H, W = X.shape
        # residual blocks forward pass
        hidden_layer = self.residual_blocks(X)
        # pooling layer to scale H and W to 1
        hidden_layer = torch.nn.AvgPool2d(kernel_size=(H, W))(hidden_layer)
        # flat
        hidden_layer = torch.nn.Flatten()(hidden_layer)
        # fully-connected layer
        scores = self.fully_connected(hidden_layer)

        return scores


