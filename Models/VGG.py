# -*- coding = utf-8 -*-
# @Time: 2025/3/27 23:24
# @Author: Zhihang Yi
# @File: VGG.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from ResNet import ResNet, FullyConnectedLayer


class VGGBlock(nn.Module):

    def __init__(self, in_channels=3, out_channels=8):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(0.2),
        )

    def forward(self, X):
        scores = self.model(X)
        return scores


class VGG(nn.Module):

    def __init__(self, channels=(3, 8, 16, 32, 64, 128, 256), num_classes=10):
        super().__init__()

        vgg_blocks = []
        num_blocks = len(channels)

        for block in range(num_blocks - 1):
            vgg_blocks.append(VGGBlock(channels[block], channels[block + 1]))

        self.vgg_blocks = nn.Sequential(*vgg_blocks)
        self.fc = FullyConnectedLayer(channels[-1], (channels[-1] + num_classes) // 2, num_classes)

    def forward(self, X):
        N, C, H, W = X.shape

        feature_map = self.vgg_blocks(X)
        feature_map = torch.nn.AvgPool2d(kernel_size=(H, W))(feature_map)
        feature_map = torch.nn.Flatten()(feature_map)
        scores = self.fc(feature_map)

        return scores


if __name__ == '__main__':
    # generate random data for testing
    X = torch.randint(0, 256, (64, 3, 32, 32), dtype=torch.float)
    y = torch.randint(0, 10, (64,))

    model = VGG(channels=[3, 8, 16, 32, 64, 128, 256], num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    epochs = range(10)

    for epoch in epochs:
        scores = model(X)
        loss = torch.nn.functional.cross_entropy(scores, y)

        print(f"Epoch: {epoch}, Loss: {loss:.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
