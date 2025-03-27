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
        super().__init__()

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
    def __init__(self, channels=(3, 8, 16, 32, 64, 128, 256), num_classes=10):
        super().__init__()

        # stack up the residual blocks
        residual_blocks = []
        num_blocks = len(channels)

        for block in range(num_blocks - 1):
            residual_blocks.append(ResidualBlock(channels[block], channels[block + 1]))

        self.residual_blocks = torch.nn.Sequential(*residual_blocks)

        # add fully-connected layers to the end
        self.fully_connected = FullyConnectedLayer(channels[-1], (channels[-1] + num_classes) // 2, num_classes)

    def forward(self, X):
        N, C, H, W = X.shape
        # residual blocks forward pass
        feature_map = self.residual_blocks(X)
        # pooling layer to scale H and W to 1
        feature_map = torch.nn.AvgPool2d(kernel_size=(H, W))(feature_map)
        # flat
        feature_map = torch.nn.Flatten()(feature_map)
        # fully-connected layer
        scores = self.fully_connected(feature_map)

        return scores


if __name__ == '__main__':
    # generate random data for testing
    X = torch.randint(0, 256, (64, 3, 32, 32), dtype=torch.float)
    y = torch.randint(0, 10, (64,))

    model = ResNet(channels=[3, 8, 16, 32, 64, 128, 256], num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    epochs = range(10)

    for epoch in epochs:
        scores = model(X)
        loss = torch.nn.functional.cross_entropy(scores, y)

        print(f"Epoch: {epoch}, Loss: {loss:.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
