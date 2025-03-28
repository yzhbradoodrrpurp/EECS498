# -*- coding = utf-8 -*-
# @Time: 2025/3/27 23:09
# @Author: Zhihang Yi
# @File: GAN.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

from ResNet import ResidualBlock
from VGG import VGG


class Generator(nn.Module):

    def __init__(self, channels=(3, 8, 16, 32, 64, 128, 256), num_classes=10):
        super().__init__()

        # stack up the residual blocks
        residual_blocks = []
        num_blocks = len(channels)

        for block in range(num_blocks - 1):
            residual_blocks.append(ResidualBlock(channels[block], channels[block + 1]))
        residual_blocks.append(ResidualBlock(channels[num_blocks - 1], channels[0]))

        self.generator = torch.nn.Sequential(*residual_blocks)

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
            noise = torch.randn(64, 3, 32, 32)
        else:
            mode = 'train'
            noise = torch.randn_like(X)

        if mode == 'train':
            scores = self.discriminator(X)
            X_generated = self.generator(noise)
            scores_generated = self.discriminator(X_generated)
            return scores, scores_generated
        else:
            X_generated = self.generator(noise)
            return X_generated


if __name__ == '__main__':
    X = torch.randint(0, 256, (64, 3, 32, 32), dtype=torch.float)

    model = GAN(channels=(3, 8, 16, 32, 64, 128), num_classes=2)
    optimizer_G = optim.Adam(model.generator.parameters(), lr=1e-5, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    epochs = range(20)

    for epoch in epochs:
        scores, scores_generated = model(X)

        generator_loss = (scores_generated[:, 0].sigmoid().log() * -1).mean()
        discriminator_loss = (-scores[:, 1].sigmoid().log() - (1 - scores_generated[:, 0].sigmoid()).log()).mean()

        print(f"Epoch: {epoch}, Generator Loss: {generator_loss.item():.4f}, Discriminator Loss: {discriminator_loss.item():.4f}")

        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        generator_loss.backward(retain_graph=True)
        discriminator_loss.backward()

        optimizer_G.step()
        optimizer_D.step()
