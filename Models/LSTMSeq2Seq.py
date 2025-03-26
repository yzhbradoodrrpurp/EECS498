# -*- coding = utf-8 -*-
# @Time: 2025/3/26 21:23
# @Author: Zhihang Yi
# @File: LSTM.py
# @Software: PyCharm

import torch


class LSTMSeq2Seq(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, max_seq_length, dtype=torch.float, device='cpu'):
        super().__init__()

        self.Wx = torch.nn.Parameter(torch.empty(input_dim, 4 * hidden_dim, dtype=dtype, device=device))  # (D, 4H)
        self.Wh = torch.nn.Parameter(torch.empty(hidden_dim, 4 * hidden_dim, dtype=dtype, device=device))  # (H, 4H)
        self.b = torch.nn.Parameter(torch.empty(4 * hidden_dim, dtype=dtype, device=device))  # (4H,)

        torch.nn.init.kaiming_normal_(self.Wx)
        torch.nn.init.kaiming_normal_(self.Wh)
        torch.nn.init.zeros_(self.b)

        self.max_seq_length = max_seq_length

        # convert hidden state into final outputs
        self.fully_connected = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, X, h0=None, c0=None):
        """

        :param X: (N, T, D)
        :param h0: (N, H)
        :param c0: (N, H)
        :return:
        """
        H = self.Wh.shape[0]
        N, T, D = X.shape

        if h0 is None:
            h0 = torch.zeros(N, H, dtype=X.dtype, device=X.device)

        if c0 is None:
            c0 = torch.zeros(N, H, dtype=X.dtype, device=X.device)

        h_next = h0  # (N, H)
        c_next = c0  # (N, H)

        # Encoder
        for t in range(T):
            feature_map = X[:, t, :] @ self.Wx + h_next @ self.Wh + self.b  # (N, 4H)

            i = feature_map[:, 0:H].sigmoid()  # (N, H)
            f = feature_map[:, H:2 * H].sigmoid()  # (N, H)
            o = feature_map[:, 2 * H:3 * H].sigmoid()  # (N, H)
            g = feature_map[:, 3 * H:4 * H].tanh()  # (N, H)

            c_next = c_next * f + i * g  # (N, H)
            h_next = o * c_next.tanh()  # (N, H)

        start_token = torch.zeros(N, D, dtype=X.dtype, device=X.device)
        outputs = []

        # Decoder
        for t in range(self.max_seq_length):
            feature_map = start_token @ self.Wx + h_next @ self.Wh + self.b

            i = feature_map[:, 0:H]  # (N, H)
            f = feature_map[:, H:2 * H]  # (N, H)
            o = feature_map[:, 2 * H:3 * H]  # (N, H)
            g = feature_map[:, 3 * H:4 * H]  # (N, H)

            c_next = c_next * f + i * g  # (N, H)
            h_next = o * c_next.tanh()  # (N, H)

            output = self.fully_connected(h_next)  # (N, O)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)  # (N, max_seq_length, O)

        return outputs


if __name__ == '__main__':
    N, T, D = 32, 15, 16
    H, max_seq_length = 16, 16
    O = 16

    X = torch.randn(N, T, D)
    y = torch.randn(N, max_seq_length, O)

    model = LSTMSeq2Seq(D, H, O, max_seq_length, dtype=torch.float, device='cpu')
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)

    epochs = range(20)

    for epoch in epochs:
        y_pred = model(X)
        loss = criterion(y_pred, y)

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

