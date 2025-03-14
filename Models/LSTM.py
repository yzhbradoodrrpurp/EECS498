# -*- coding = utf-8 -*-
# @Time: 2025/3/14 12:13
# @Author: Zhihang Yi
# @File: LSTM.py
# @Software: PyCharm

import torch

class LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, max_length, dtype=torch.float, device='cpu'):
        super().__init__()

        self.max_length = max_length

        self.Wx = torch.nn.Parameter(torch.empty(input_dim, 4 * hidden_dim, dtype=dtype, device=device))  # (D, 4H)
        self.Wh = torch.nn.Parameter(torch.empty(hidden_dim, 4 * hidden_dim, dtype=dtype, device=device))  # (H, 4H)
        self.b = torch.nn.Parameter(torch.empty(4 * hidden_dim, dtype=dtype, device=device))  # (4H,)

        torch.nn.init.kaiming_normal_(self.Wx)
        torch.nn.init.kaiming_normal_(self.Wh)
        torch.nn.init.zeros_(self.b)

        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def train_forward(self, X, h0=None, c0=None):
        """

        :param X: (N, T, D)
        :param h0: (N, H), initial hidden state
        :param c0: (N, H)
        :return: outputs: (N, T, out_dim)
        """
        N, T, D = X.shape
        H = self.Wh.shape[0]

        if h0 is None:
            h0 = torch.zeros(N, H)

        if c0 is None:
            c0 = torch.zeros(N, H)

        h_next = h0
        c_next = c0
        outputs = []

        for t in range(T):
            feature_map = X[:, t, :] @ self.Wx + h_next @ self.Wh + self.b

            i = feature_map[:, 0:H]  # (N, H)
            f = feature_map[:, H:2*H]  # (N, H)
            o = feature_map[:, 2*H:3*H]  # (N, H)
            g = feature_map[:, 3*H:4*H]  # (N, H)

            c_next = c_next * f + i * g
            h_next = o * c_next.tanh()

            output = self.fc(h_next)  # (N, out_dim)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)  # (N, T, out_dim)

        return outputs

    def test_forward(self, h0, c0):
        """

        :param h0:
        :param c0:
        :return: (N, max_length, out_dim), output tokens
        """
        N = self.h0.shape[0]
        D = self.Wx.sjape[1]

        h_next = h0
        c_next = c0

        start_token = torch.ones(N, D)
        next_token = start_token
        tokens = []

        for length in range(self.max_length):
            feature_map = next_token @ self.Wx + h_next @ self.Wh + self.b

            i = feature_map[:, 0:H]  # (N, H)
            f = feature_map[:, H:2 * H]  # (N, H)
            o = feature_map[:, 2 * H:3 * H]  # (N, H)
            g = feature_map[:, 3 * H:4 * H]  # (N, H)

            c_next = c_next * f + i * g
            h_next = o * c_next.tanh()

            next_token = self.fc(h_next)  # (N, out_dim)
            tokens.append(next_token)

        tokens = torch.stack(tokens, dim=1)  # (N, max_length, out_dim)

        return tokens

