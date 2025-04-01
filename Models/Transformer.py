# -*- coding = utf-8 -*-
# @Time: 2025/3/16 09:45
# @Author: Zhihang Yi
# @File: Transformer.py
# @Software: PyCharm

import torch
import math

class SelfAttention(torch.nn.Module):
    def __init__(self, input_dim, num_heads=1, dtype=torch.float, device='cpu'):
        """
        NOTE: single-head attention layer

        :param input_dim: dimension of input X
        :param QKV_dim: dimension of QKV
        :param num_heads: number of heads
        :param dtype: data type
        :param device: device to run on
        """
        super().__init__()

        self.num_heads = num_heads  # H
        self.input_dim = input_dim  # D

        assert input_dim % num_heads == 0, 'input_dim must be divisible by num_heads.'

        # 将 QKV 和 input dimension 全部统一
        self.query_linear = torch.nn.Linear(input_dim, input_dim, dtype=dtype, device=device)
        self.key_linear = torch.nn.Linear(input_dim, input_dim, dtype=dtype, device=device)
        self.value_linear = torch.nn.Linear(input_dim, input_dim, dtype=dtype, device=device)

    def forward(self, X):
        """

        :param X: (N, T, D)
        :return:
        """
        N, T, D = X.shape
        H = self.num_heads
        head_dim = D // H

        # NOTE: 分别得到 QKV 后再拆分为多头
        query = self.query_linear(X)  # (N, T, D)
        query = query.view(N, T, H, D // H).permute(0, 2, 1, 3).contiguous()  # (N, H, T, head_dim)

        key = self.key_linear(X)  # (N, T, D)
        key = key.view(N, T, H, D // H).permute(0, 2, 1, 3).contiguous()  # (N, H, T, head_dim)

        value = self.value_linear(X)  # (N, T, D)
        value = value.view(N, T, H, D // H).permute(0, 2, 1, 3).contiguous()  # (N, H, T, head_dim)

        alignment_scores = query @ key.transpose(2, 3) / math.sqrt(head_dim)  # (N, H, T, T)
        attention_weights = torch.nn.functional.softmax(alignment_scores, dim=-1)  # (N, H, T, T)

        scores = attention_weights @ value  # (N, H, T, head_dim)

        # 将多头转换为原来的样子
        scores = scores.permute(0, 2, 1, 3).contiguous().view(N, T, -1)

        return scores

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 4 * hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * hidden_dim, hidden_dim),
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
    def __init__(self, input_dim, num_heads=1, dtype=torch.float, device='cpu'):
        """

        :param input_dim: dimension of input X
        :param QKV_dim: dimension of query, key and value
        :param dtype: data type
        :param device: device to run on
        NOTE: 一般将 query, key, value 的维度都设置为和输入的维度相同，便于进行残差连接
        """
        super().__init__()

        self.self_attention = SelfAttention(input_dim, num_heads, dtype=dtype, device=device)
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

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, num_heads=1, num_blocks=8, max_seq_len=32, vocab_size=500, dtype=torch.float, device='cpu'):
        super().__init__()

        # 将 id 转换为词向量
        self.word_embedding = WordEmbedding(vocab_size, input_dim)
        # 给词向量进行位置编码
        self.positional_encoding = PositionalEncoding(max_seq_len, input_dim)
        self.transformer_blocks = [TransformerBlock(input_dim, num_heads) for _ in range(num_blocks)]
        self.model = torch.nn.Sequential(*self.transformer_blocks)
        # # 将结果词向量转换为在整个字典中得分的分布
        # self.linear = torch.nn.Linear(input_dim, vocab_size)

    def forward(self, num):
        """

        :param num: (N, T)
        :return:
        """
        X = self.word_embedding(num)  # (N, T, D)
        X = self.positional_encoding(X)
        Y = self.model(X)  # (N, T, D)
        # scores = self.linear(Y)  # (N, T, vocab_size)
        return Y

class Decoder(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, X):
        pass

class Transformer(torch.nn.Module):
    def __init__(self, input_dim, num_heads=1, num_blocks=8, max_seq_len=32, vocab_size=500, dtype=torch.float,
                 device='cpu'):
        super().__init__()

        self.encoder = Encoder(input_dim, num_heads, num_blocks, max_seq_len, vocab_size, dtype, device)
        self.decoder = Decoder()

    def forward(self, X):
        encoded_output = self.encoder(X)


if __name__ == '__main__':

    # TODO: Positional Embedding and Word Embedding

    # NOTE: 生成 0～500 范围内的整数，有 8 组 prompt，每组 prompt 的长度为 20 个 token
    prompts_id = torch.randint(0, 500, (8, 20))
    answers_id = torch.randint(0, 500, (8, 20))
    N, T = prompts_id.shape

    # NOTE: 每个词向量的维度为 32，Transformer 模型由 4 个 Transformer Block 组成
    # NOTE: 一组 prompt 中最多有 20 个 token，Word Embedding 中字典的大小为 500
    model = Encoder(32, 8, 25, vocab_size=500, dtype=torch.float, device='cpu')
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
