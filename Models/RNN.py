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
        self.Wx = torch.nn.Parameter(torch.empty(input_dimensions, hidden_dimensions, dtype=dtype, device=device))
        self.Wh = torch.nn.Parameter(torch.empty(hidden_dimensions, hidden_dimensions, dtype=dtype, device=device))
        self.b = torch.nn.Parameter(torch.zeros(hidden_dimensions, dtype=dtype, device=device))

        torch.nn.init.kaiming_normal_(self.Wx, nonlinearity='tanh')
        torch.nn.init.kaiming_normal_(self.Wh, nonlinearity='tanh')
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

            out = self.fc(h_next)  # (N, output_dim)
            outputs.append(out)

        # convert Python list to PyTorch tensor
        outputs = torch.stack(outputs, dim=1)  # (N, T, output_dim)

        return outputs

    def test_forward(self, h0):
        """

        :param h0: (N, H), initial hidden state
        :return: output tokens
        """
        N = h0.shape[0]
        D = self.Wx.shape[0]

        start_token = torch.ones(N, D, dtype=h0.dtype, device=h0.device)
        next_token = start_token
        h_next = h0

        tokens = []

        for i in range(self.max_length):
            h_next = next_token @ self.Wx + h_next @ self.Wh + self.b
            h_next = h_next.tanh()

            next_token = self.fc(h_next)  # (N, out_dim)
            tokens.append(next_token)

        # convert Python list to PyTorch tensor
        tokens = torch.stack(tokens, dim=1)  # (N, max_length, out_dim)

        return tokens


if __name__ == '__main__':
    X = torch.randint(0, 100, (32, 15, 1), dtype=torch.float)
    y = torch.randint(0, 100, (32, 15, 1), dtype=torch.float)

    model = RNN(input_dimensions=1, hidden_dimensions=10, output_dimensions=1, max_length=20)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    epochs = range(10)

    for epoch in epochs:
        y_pred = model.train_forward(X)
        loss = criterion(y_pred, y)

        print(f"Epoch: {epoch}, Loss: {loss:.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

