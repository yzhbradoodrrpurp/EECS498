# -*- coding = utf-8 -*-
# @Time: 2025/3/27 23:09
# @Author: Zhihang Yi
# @File: GAN.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

from ResNet import ResNet
from VGG import VGG


class Generator(nn.Module):

    def __init__(self, channels=(3, 8, 16, 32, 64, 128, 256), num_classes=10):
        super().__init__()
        self.generator = ResNet(channels=channels, num_classes=num_classes)

    def forward(self, X):
        scores = self.generator(X)
        return scores


class Discriminator(nn.Module):

    def __init__(self, channels=(3, 8, 16, 32, 64, 128, 256), num_classes=10):
        super().__init__()
        self.discriminator = VGG(channels=channels, num_classes=num_classes)

    def forward(self, X):
        scores = self.discriminator(X)
        return scores


class GAN(nn.Module):

    def __init__(self, channels=(3, 8, 16, 32, 64, 128, 256), num_classes=10):
        super().__init__()

        self.generator = Generator(channels=channels, num_classes=num_classes)
        self.discriminator = Discriminator(channels=channels, num_classes=num_classes)

    def forward(self, X=None):

        if X is None:
            mode = 'test'
        else:
            mode = 'train'
            noise = torch.randn_like(X)

        if mode == 'train':
            scores = self.discriminator(X)
            scores_generated = self.generator(noise)
            return scores, scores_generated
        else:
            scores_generated = self.discriminator(noise)
            return scores_generated


if __name__ == '__main__':
    X = torch.randint(0, 256, (64, 3, 32, 32), dtype=torch.float)

    model = GAN(channels=(3, 8, 16, 32, 64, 128), num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = range(20)

    for epoch in epochs:
        scores, scores_generated = model(X)
        loss = criterion(scores, torch.ones(64, dtype=torch.long))
        loss += criterion(scores_generated, torch.zeros(64, dtype=torch.long))

        print(f'Epoch {epoch} Loss {loss.item():.4f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

