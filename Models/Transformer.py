# -*- coding = utf-8 -*-
# @Time: 2025/3/16 09:45
# @Author: Zhihang Yi
# @File: Transformer.py
# @Software: PyCharm

import torch
import math

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
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, X):
        """

        :param X: (N, T, D)
        :return:
        """
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
        self.layer_norm1 = torch.nn.LayerNorm(input_dim)
        # MLP 层分别处理每一个输入数据
        self.fully_connected = MLP(input_dim, input_dim, input_dim)
        self.residual_shortcut2 = torch.nn.Identity()
        self.layer_norm2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        """

        :param X: (N, T, D)
        :return:
        """
        N, T, D = X.shape

        feature_map = self.self_attention(X) + self.residual_shortcut1(X)  # (N, T, D)
        feature_map = self.layer_norm1(feature_map)  # (N, T, D)
        feature_map = self.fully_connected(feature_map) + self.residual_shortcut2(feature_map)  # (N, T, D)
        Y = self.layer_norm2(feature_map)

        return Y

class WordEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, dtype=torch.float, device='cpu'):
        """

        :param vocab_size: size of vocabulary
        :param embedding_dim: dimension of word vector
        :param dtype: data type
        :param device: device to run on
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, dtype=dtype, device=device)

    def forward(self, num):
        word_vec = self.embedding(num)
        return word_vec

class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_seq_len, embedding_dim, dtype=torch.float, device='cpu'):
        super().__init__()
        self.positional_encoding = torch.nn.Embedding(max_seq_len, embedding_dim, dtype=dtype, device=device)

    def forward(self, X):
        """

        :param X: (N, T, D) word vector
        :return:
        """
        N, T, D = X.shape

        position_ids = torch.arange(T, device=X.device)  # (T,)
        positions = self.positional_encoding(position_ids)  # (T, D)

        return X + positions  # (N, T, D)

class Transformer(torch.nn.Module):
    def __init__(self, input_dim, num_blocks, max_seq_len, vocab_size=500, dtype=torch.float, device='cpu'):
        super().__init__()

        # 将 id 转换为词向量
        self.word_embedding = WordEmbedding(vocab_size, input_dim)
        # 给词向量进行位置编码
        self.positional_encoding = PositionalEncoding(max_seq_len, input_dim)
        self.transformer_blocks = [TransformerBlock(input_dim, input_dim, input_dim) for _ in range(num_blocks)]
        self.model = torch.nn.Sequential(*self.transformer_blocks)
        # 将结果词向量转换为在整个字典中得分的分布
        self.linear = torch.nn.Linear(input_dim, vocab_size)

    def forward(self, num):
        """

        :param num: (N, T)
        :return:
        """
        X = self.word_embedding(num)  # (N, T, D)
        X = self.positional_encoding(X)
        Y = self.model(X)  # (N, T, D)
        scores = self.linear(Y)  # (N, T, vocab_size)
        return scores


if __name__ == '__main__':

    # TODO: Positional Embedding and Word Embedding

    # NOTE: 生成 0～500 范围内的整数，有 8 组 prompt，每组 prompt 的长度为 20 个 token
    prompts_id = torch.randint(0, 500, (8, 20))
    answers_id = torch.randint(0, 500, (8, 20))
    N, T = prompts_id.shape

    # NOTE: 每个词向量的维度为 32，Transformer 模型由 4 个 Transformer Block 组成
    # NOTE: 一组 prompt 中最多有 20 个 token，Word Embedding 中字典的大小为 500
    model = Transformer(32, 4, 20, vocab_size=500, dtype=torch.float, device='cpu')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = range(20)

    for epoch in epochs:
        scores = model(prompts_id)  # (N, T, 500)

        scores_flat = scores.view(N * T, -1)
        answers_id_flat = answers_id.view(N * T,)

        loss = criterion(scores_flat, answers_id_flat)

        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
