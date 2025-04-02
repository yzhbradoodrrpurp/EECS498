# -*- coding = utf-8 -*-
# @Time: 2025/3/16 09:45
# @Author: Zhihang Yi
# @File: Transformer.py
# @Software: PyCharm

import torch
import math

class SelfAttention(torch.nn.Module):
    def __init__(self, input_dim, num_heads=1, masked=False, dtype=torch.float, device='cpu'):
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
        self.masked = masked

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

        if self.masked:
            mask = torch.triu(torch.ones(T, T, dtype=X.dtype, device=X.device), diagonal=1)  # 将 T*T 的 1 矩阵的对角线及以下的部分替换成0
            mask[mask == 1] = float('-inf')
            alignment_scores += mask

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

class EncoderBlock(torch.nn.Module):
    def __init__(self, input_dim, num_heads=1, dtype=torch.float, device='cpu'):
        """

        :param input_dim: dimension of input X
        :param QKV_dim: dimension of query, key and value
        :param dtype: data type
        :param device: device to run on
        NOTE: 一般将 query, key, value 的维度都设置为和输入的维度相同，便于进行残差连接
        """
        super().__init__()

        self.self_attention = SelfAttention(input_dim, num_heads, masked=False, dtype=dtype, device=device)
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
        self.encoder_blocks = [EncoderBlock(input_dim, num_heads) for _ in range(num_blocks)]
        self.model = torch.nn.Sequential(*self.encoder_blocks)

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

class CrossAttention(torch.nn.Module):
    def __init__(self, input_dim, num_heads, dtype=torch.float, device='cpu'):
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads

        assert input_dim % num_heads == 0, 'input_dim must be divisible by num_heads.'

        self.query_linear = torch.nn.Linear(input_dim, input_dim, dtype=dtype, device=device)
        self.key_linear = torch.nn.Linear(input_dim, input_dim, dtype=dtype, device=device)
        self.value_linear = torch.nn.Linear(input_dim, input_dim, dtype=dtype, device=device)

    def forward(self, previous_output, context):
        """

        :param previous_output: (N, X, D)
        :param context: (N, T, D)
        :return: (N, T, D)
        """
        N, T, D = context.shape
        X = previous_output.shape[1]
        H = self.num_heads
        head_dim = D // H

        query = self.query_linear(previous_output)  # (N, X, D)
        query = query.view(N, X, H, D // H).permute(0, 2, 1, 3).contiguous()  # (N, H, X, head_dim)

        key = self.key_linear(context)  # (N, T, D)
        key = key.view(N, T, H, D // H).permute(0, 2, 1, 3).contiguous()  # (N, H, T, head_dim)

        value = self.value_linear(context)  # (N, T, D)
        value = value.view(N, T, H, D // H).permute(0, 2, 1, 3).contiguous()  # (N, H , T, head_dim)

        alignment_scores = query @ key.transpose(2, 3) / math.sqrt(head_dim)  # (N, H, X, T)
        attention_weights = torch.nn.functional.softmax(alignment_scores, dim=-1)  # (N, H, X, T)

        scores = attention_weights @ value  # (N, H, X, head_dim)
        scores = scores.permute(0, 2, 1, 3).contiguous().view(N, X, -1)  # (N, X, D)

        return scores

class DecoderBlock(torch.nn.Module):
    def __init__(self, input_dim, num_heads=1, dtype=torch.float, device='cpu'):
        super().__init__()

        self.masked_self_attention = SelfAttention(input_dim, num_heads, masked=True, dtype=dtype, device=device)
        self.residual_shortcut1 = torch.nn.Identity()
        self.layer_norm1 = torch.nn.LayerNorm(input_dim)

        self.cross_attention = CrossAttention(input_dim, num_heads, dtype=dtype, device=device)
        self.residual_shortcut2 = torch.nn.Identity()
        self.layer_norm2 = torch.nn.LayerNorm(input_dim)

        self.fully_connected = MLP(input_dim, input_dim, input_dim)
        self.layer_norm3 = torch.nn.LayerNorm(input_dim)
        self.residual_shortcut3 = torch.nn.Identity()

    def forward(self, previous_outputs):
        """

        :param previous_outputs: (N, X, D) NOTE: X is the length of previous outputs
        :return:
        """
        feature_map = self.masked_self_attention(previous_outputs) + self.residual_shortcut1(previous_outputs)  # (N, X, D)
        feature_map = self.layer_norm1(feature_map)  # (N, X, D)

        feature_map = self.cross_attention(previous_outputs, self.context)  # (N, X, D)
        feature_map = self.layer_norm2(feature_map)  # (N, X, D)

        feature_map = self.fully_connected(feature_map) + self.residual_shortcut3(feature_map)  # (N, X, D)
        feature_map = self.layer_norm3(feature_map)  # (N, X, D)

        return feature_map  # (N, X, D)

class Decoder(torch.nn.Module):
    def __init__(self, input_dim, num_heads=1, num_blocks=8, max_seq_length=32, vocab_size=500, dtype=torch.float, device='cpu'):
        super().__init__()

        self.max_seq_len = max_seq_length

        self.word_embedding = WordEmbedding(vocab_size, input_dim)
        self.decoder_blocks = [DecoderBlock(input_dim, num_heads, dtype=dtype, device=device) for _ in range(num_blocks)]
        self.model = torch.nn.Sequential(*self.decoder_blocks)
        # 将结果词向量转换为在整个字典中得分的分布
        self.linear = torch.nn.Linear(input_dim, vocab_size)

    def forward(self, context):
        """

        :param context: (N, T, D)
        :return:
        """
        N = context.shape[0]
        D = context.shape[-1]

        DecoderBlock.context = context

        start_token = self.word_embedding(torch.zeros(N, 1, dtype=torch.long))  # start token id is 0
        answers = [start_token]  # every element shape: (N, D)

        for i in range(self.max_seq_len):
            previous_outputs = torch.cat(answers, dim=1)  # (N, X, D)

            next_output_distribution = self.model(previous_outputs)  # (N, X, D)
            word_distribution = self.linear(next_output_distribution)  # (N, X, vocab_size)
            word_distribution = torch.nn.functional.softmax(word_distribution, dim=-1)  # (N, X, vocab_size)

            next_word = word_distribution.argmax(dim=-1).to(torch.float).multinomial(num_samples=1)  # (N, 1) 从 X 个中随机采样出 1 个
            next_token = self.word_embedding(next_word)  # (N, D)

            answers.append(next_token)

        answers = torch.cat(answers, dim=1)  # (N, max_seq_length, D)
        answer_scores = self.linear(answers)  # (N, max_seq_length, 500)

        return answer_scores

class Transformer(torch.nn.Module):
    def __init__(self, input_dim, num_heads=1, num_blocks=8, max_seq_len=32, vocab_size=500, dtype=torch.float,
                 device='cpu'):
        super().__init__()

        self.encoder = Encoder(input_dim, num_heads, num_blocks, max_seq_len, vocab_size, dtype, device)
        self.decoder = Decoder(input_dim, num_heads, num_blocks, max_seq_len, vocab_size, dtype, device)

    def forward(self, X):
        context = self.encoder(X)  # (N, T, D)
        answer_scores = self.decoder(context)  # (N, max_seq_length, D)
        return answer_scores


if __name__ == '__main__':

    # NOTE: Word Embedding 中字典的大小为 500，有 8 组 prompt，每组 prompt 的长度为 20 个 token
    # NOTE: 每个词向量的维度为 32，Transformer 模型由 4 个 Transformer Block 组成，
    # NOTE: 多头注意力的头数为 4
    N, T, D = 8, 20, 32
    max_seq_length = 30
    vocab_size = 500
    H = 4

    prompts_id = torch.randint(0, vocab_size, (N, T))
    answers_id = torch.randint(0, vocab_size, (N, max_seq_length))

    model = Transformer(D, H, 4, max_seq_length, vocab_size, dtype=torch.float, device='cpu')
    optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = range(20)

    for epoch in epochs:
        answer_scores = model(prompts_id)  # (N, max_seq_length + 1, D)
        answer_scores = answer_scores[:, 1:, :]  # (N, max_seq_length, D)

        answer_scores_ = answer_scores.reshape(N * max_seq_length, -1)
        answers_id_ = answers_id.reshape(N * max_seq_length)
        loss = criterion(answer_scores_, answers_id_)

        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
