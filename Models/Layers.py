# -*- coding = utf-8 -*-
# @Time: 2025/3/8 12:31
# @Author: Zhihang Yi
# @File: Layers.py
# @Software: PyCharm

import torch

# NOTE: Acivation Function: ReLU
class ReLU:
    is_parametric = False

    def forward(self, X):
        X[X < 0] = 0
        return X

# NOTE: Loss Function: Softmax
class SoftmaxLoss:
    is_parametric = False

    def forward(self, scores, labels):
        N = scores.shape[0]

        # avoid overflow
        scores -= scores.max(dim=1).values.view(N, -1)

        softmax_scores = scores.exp() / scores.exp().sum(dim=1).view(N, -1)

        correct_softmax_scores = scores[range(N), labels]

        loss = (-1 * correct_softmax_scores.log()).sum() / N

        return loss

# NOTE: Convolutional Layer
class Convolution:
    is_parametric = True

    def __init__(self, filter=8, channels=3, kernel_size=3, stride=1, padding=1, dtype=torch.float, device='cpu'):
        self.filter = filter
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dtype = dtype
        self.device = device

        # use kaiming initialization
        self.W = torch.randn(filter, channels, kernel_size, kernel_size, dtype=dtype, device=device, requires_grad=True) * math.sqrt(2 / (channels * kernel_size * kernel_size))
        self.b = torch.zeros(filter, dtype=dtype, device=device, requires_grad=True)

    def forward(self, X):
        N, C, H, W = X.shape
        H_out = (H - self.kernel_size + 2 * self.padding) // self.stride + 1
        W_out = (W - self.kernel_size + 2 * self.padding) // self.stride + 1

        result = torch.zeros(N, self.filter, H_out, W_out, dtype=self.dtype, device=self.device)

        import torch.nn.functional as f
        X_padded = f.pad(X, (self.padding, self.padding, self.padding, self.padding))

        for i in range(H_out):
            for j in range(W_out):
                window = X_padded[:, :, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size]  # (1, C, kernel_size, kernel_size)
                # reshaping for broadcasting
                window_reshaped = window.view(N, 1, C, self.kernel_size, self.kernel_size)
                W_reshaped = self.W.view(1, self.filter, self.channels, self.kernel_size, self.kernel_size)
                result[:, :, i, j] = (window * self.W).sum(dim=(2, 3, 4))

        result += self.b.view(1, self.filter, 1, 1)

        return result

    def update(self):
        self.W -= self.W.grad
        self.b -= self.b.grad

        self.W.grad.zero_()
        self.b.grad.zero_()

# NOTE: Batch Normalization Layer
class BatchNorm:
    is_parametric = True

    def __init__(self, dimension, mode='train', dtype=torch.float, device='cpu'):
        self.dimension = dimension
        self.mode = mode

        self.gamma = torch.ones(dimension, dtype=dtype, device=device, requires_grad=True)
        self.beta = torch.zeros(dimension, dtype=dtype, device=device, requires_grad=True)

        self.running_mean = torch.zeros(dimension, dtype=dtype, device=device)
        self.running_var = torch.ones(dimension, dtype=dtype, device=device)

        self.eps = 1e-5
        self.momentum = 0.9

    def forward(self, X):
        if X.dim() == 2:
            N, D = X.shape

            if self.mode == 'train':
                mean = X.mean(dim=0)
                var = X.var(dim=0, unbiased=False)
                center = (X - mean) / (var + self.eps).sqrt()

                result = self.gamma * center + self.beta

                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            elif self.mode == 'test':
                center = (X - self.running_mean) / (self.running_var + self.eps).sqrt()
                result = self.gamma * center + self.beta
            else:
                raise NotImplementedError('Only train and test modes are supported.')
        elif X.dim() == 4:
            # X: (N, C, H, W)
            N, C, H, W = X.shape

            # (N, C, H, W) -> (N, H, W, C) -> (N * H * W, C)
            X_reshaped = X.permute(0, 2, 3, 1).reshape(-1, C)

            if self.mode == 'train':
                mean = X.mean(dim=0)
                var = X.var(dim=0, unbiased=False)
                center = (X - mean) / (var + self.eps).sqrt()

                result = self.gamma * center + self.beta
                # (N * H * W, C) -> (N, H, W, C) -> (N, C, H, W)
                result = result.reshape(N, H, W, C).permute(0, 3, 1, 2).contiguous()

                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            elif self.mode == 'test':
                center = (X_reshaped - self.running_mean) / (self.running_var + self.eps).sqrt()

                result = self.gamma * center + self.beta
                # (N * H * W, C) -> (N, H, W, C) -> (N, C, H, W)
                result = result.reshape(N, H, W, C).permute(0, 3, 1, 2).contiguous()
            else:
                raise NotImplementedError('Only train and test modes are supported.')

        return result

    def update(self):
        self.gamma -= self.gamma.grad
        self.beta -= self.beta.grad

        self.gamma.grad.zero_()
        self.beta.grad.zero_()

# NOTE: Max Pooling Layer
class MaxPooling:
    is_parametric = False

    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, X):
        N, C, H, W = X.shape

        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1

        result = torch.zeros(N, C, H_out, W_out, dtype=X.dtype, device=X.device)

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        result[n, c, h, w] = X[n, c, h * self.stride:h * self.stride + self.kernel_size, w * self.stride:w * self.stride + self.kernel_size].max()

        return result

# NOTE: Linear Layer
class Linear:
    is_parametric = True

    def __init__(self, Din, Dout, dtype=torch.float, device='cpu'):
        self.Din = Din
        self.Dout = Dout
        self.W = torch.randn(Din, Dout, dtype=dtype, device=device) * math.sqrt(2 / Din)
        self.b = torch.zeros(Dout, dtype=torch.float, device=device)

    def forward(self, X):
        return X @ self.W + self.b

    def update(self):
        self.W -= self.W.grad
        self.b -= self.b.grad
        self.W.grad.zero_()
        self.b.grad.zero_()
