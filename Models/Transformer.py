# -*- coding = utf-8 -*-
# @Time: 2025/3/16 09:45
# @Author: Zhihang Yi
# @File: Transformer.py
# @Software: PyCharm

import torch

class SelfAttention(torch.nn.Module):
    def __init__(self, input_dim, query_dim, value_dim, dtype=torch.float, device='cpu'):
        """

        :param input_dim: dimension of input X
        :param query_dim: dimension of query and key
        :param value_dim: dimension of value
        :param dtype: data type
        :param device: device to run on
        """
        super().__init__()

        # query 和 key 的维度相同
        self.query_linear = torch.nn.Linear(input_dim, query_dim, dtype=dtype, device=device)
        self.key_linear = torch.nn.Linear(input_dim, query_dim, dtype=dtype, device=device)
        self.value_linear = torch.nn.Linear(input_dim, value_dim, dtype=dtype, device=device)

    def forward(self, X):
        """

        :param X: (N, T, D)
        :return:
        """
        N, T, D = X.shape

        query = self.query_linear(X)  # (N, T, Q)
        key = self.key_linear(X)  # (N, T, Q)
        value = self.value_linear(X)  # (N, T, V)

        alignment_scores = query @ key.transpose(1, 2) / math.sqrt(D)  # (N, T, T)
        attention_weights = torch.nn.functional.softmax(alignment_scores, dim=2)  # (N, T, T)

        scores = attention_weights @ value  # (N, T, V)

        return scores

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, X):
        scores = self.model(X)
        return scores

class TransformerBlock(torch.nn.Module):
    def __init__(self, input_dim, query_dim, value_dim, dtype=torch.float, device='cpu'):
        """

        :param input_dim: dimension of input X
        :param query_dim: dimension of query and key
        :param value_dim: dimension of value
        :param dtype: data type
        :param device: device to run on
        NOTE: 一般将 query, key, value 的维度都设置为和输入的维度相同，便于进行残差连接
        """
        super().__init__()

        self.self_attention = SelfAttention(input_dim, query_dim, value_dim, dtype=dtype, device=device)
        # `torch.nn.Identity` 不改变输入
        self.residual_shortcut1 = torch.nn.Identity()
        # Transformer 中使用 Layer Normalization
        self.batch_norm = torch.nn.LayerNorm(input_dim)
        # MLP 层分别处理每一个输入数据
        self.fully_connected = MLP(input_dim, input_dim, input_dim)
        self.residual_shortcut2 = torch.nn.Identity()

    def forward(self, X):
        """

        :param X: (N, T, D)
        :return:
        """
        N, T, D = X.shape

        feature_map = self.self_attention(X) + self.residual_shortcut1(X)  # (N, T, D)
        feature_map = self.batch_norm(feature_map)  # (N, T, D)
        scores = self.fully_connected(feature_map) + self.residual_shortcut2(feature_map)  # (N, T, D)

        return scores

class Transformer(torch.nn.Module):
    def __init__(self, input_dim, num_blocks, dtype=torch.float, device='cpu'):
        super().__init__()

        self.transformer_blocks = [TransformerBlock(input_dim, input_dim, input_dim) for _ in range(num_blocks)]
        self.model = torch.nn.Sequential(*self.transformer_blocks)

    def forward(self, X):
        scores = self.model(X)
        return scores


if __name__ == '__main__':

    # TODO: Positional Embedding and Word Embedding

    # 生成 0～500 范围内的整数，通过 Word Encoding 转换为输入 X
    # 通过 Positional Encoding 加上顺序
    prompts = torch.randint(0, 500, (8, 32, 20))
    answers = torch.randint(0, 500, (8, 32, 20))
    N, T, D = prompts.shape

    model = Transformer(D, 4)
    epochs = range(10)

    for epoch in epochs:
        pass

