# -*- coding = utf-8 -*-
# @Time: 2025/3/13 15:49
# @Author: Zhihang Yi
# @File: RNN.py
# @Software: PyCharm

import torch

class RNN(torch.nn.Module):
    def __init__(self, input_dimensions, hidden_dimensions):
        self.Wx = torch.nn.Parameter(torch.randn(input_dimensions, hidden_dimensions))
        self.Wh = torch.nn.Parameter(torch.randn(hidden_dimensions, hidden_dimensions))
        self.b = torch.nn.Parameter(torch.zeros(hidden_dimensions,))

    def forward(self, X, h0=None):

        # initialize h0 if it's not initialized
        if h0 is None:
            h0 = torch.zeros(self.Wh.shape)

