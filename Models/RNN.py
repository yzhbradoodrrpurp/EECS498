# -*- coding = utf-8 -*-
# @Time: 2025/3/13 15:49
# @Author: Zhihang Yi
# @File: RNN.py
# @Software: PyCharm

import torch

class RNN(torch.nn.Module):
    def __init__(self, input_dimensions, hidden_dimensions, output_dimensions, max_length, dtype=torch.float, device='cpu'):
        super().__init__()

        self.max_length = max_length

        # for hidden state part
        self.Wx = torch.nn.Parameter(torch.randn(input_dimensions, hidden_dimensions, dtype=dtype, device=device))
        self.Wh = torch.nn.Parameter(torch.randn(hidden_dimensions, hidden_dimensions, dtype=dtype, device=device))
        self.b = torch.nn.Parameter(torch.zeros(hidden_dimensions, dtype=dtype, device=device))

        torch.nn.init.kaiming_normal_(self.Wx)
        torch.nn.init.kaiming_normal_(self.Wh)
        torch.nn.init.zeros_(self.b)

        # convert hidden state into output
        self.fc = torch.nn.Linear(hidden_dimensions, output_dimensions, dtype=dtype, device=device)

    def train_forward(self, X, h0=None):
        """

        :param X: (N, T, D), D dimensions, each sample with T sequences and N samples
        :param h0: (N, H), initial hidden state
        :return: outputs
        """
        N, T, D = X.shape
        H = self.Wh.shape[0]

        # initialize h0 if it's not initialized
        if h0 is None:
            h0 = torch.zeros(N, H, dtype=X.dtype, device=X.device)

        h_next = h0
        outputs = []

        # forward pass
        for t in range(T):
            h_next = X[:, t, :] @ self.Wx + h_next @ self.Wh + self.b
            h_next = h_next.tanh()

            out = self.fc(h_next)
            outputs.append(out)

        # convert Python list to PyTorch tensor
        outputs = torch.stack(outputs)

        return outputs

    def test_forward(self, h0):
        """

        :param h0: (N, H), initial hidden state
        :return: output tokens
        """
        N = h0.shape[0]
        D = self.Wx.shape[0]

        start_token = torch.ones(N, D, dtype=h0.dtype, device=h0.device)

        tokens = []

        for i in range(self.max_length):
            if i == 0:
                h_next = start_token @ self.Wx + h0 @ self.Wh + self.b
            else:
                h_next = next_token @ self.Wx + h_next @ self.Wh + self.b

            h_next = h_next.tanh()

            next_token = self.fc(h_next)
            tokens.append(next_token)

        # convert Python list to PyTorch tensor
        tokens = torch.stack(tokens)

        return tokens
