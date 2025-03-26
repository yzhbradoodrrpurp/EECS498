# -*- coding = utf-8 -*-
# @Time: 2025/3/26 23:05
# @Author: Zhihang Yi
# @File: VAE.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init


class VAE(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=(2, 2), stride=2),  # (16, 32, 32)
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(8, 16, kernel_size=(2, 2), stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16)  # (16, 4, 4)
        )

        self.fc_mean = nn.Linear(16 * 4 * 4, 16 * 4 * 4)
        self.fc_std = nn.Linear(16 * 4 * 4, 16 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=(1, 1), stride=5),  # (8, 16, 16)
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, 3, kernel_size=(2, 2), stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(3)  # (3, 32, 32)
        )

    def reparameterize(self, latent_space):
        flatten = nn.Flatten()

        latent_space_flat = flatten(latent_space)
        mean = self.fc_mean(latent_space_flat)

        log_var = self.fc_std(latent_space_flat)
        std = (log_var * 0.5).exp()
        epsilon = torch.randn_like(std)

        return mean + std * epsilon, mean, log_var, epsilon  # (16 * 4 * 4,)

    def forward(self, X):
        latent_space = self.encoder(X)  # (N, 16, 4, 4)
        z, mean, log_var, epsilon = self.reparameterize(latent_space)  # (N, 16 * 4 * 4)

        z = z.view(-1, 16, 4, 4)

        return self.decoder(z), mean, log_var, epsilon


if __name__ == '__main__':
    X = torch.randn(64, 3, 32, 32)
    y = torch.randn(64, 3, 32, 32)

    model = VAE()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    epochs = range(20)

    for epoch in epochs:
        y_pred, mean, log_var, epsilon = model(X)
        loss = criterion(y_pred, y)
        loss += (1 + log_var - mean ** 2 - log_var.exp()).sum() * (-0.5)

        print(f"Epoch {epoch} Loss {loss.item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
