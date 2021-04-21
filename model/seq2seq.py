# -*- coding: utf-8 -*-
# @Time    : 2021/4/20 下午8:24

"""
Model: seq2seq
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # input_size是词典大小, hidden_size是词向量维度
        self.embedding = nn.Embedding(input_size, hidden_size)
        # 这里nn.GRU(x, h)两个参数指明输入x和隐藏层状态的维度, 这里都用hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        """
        :param input: 这里的input是每次一个词, 具体形式为: [word_idx]
        :param hidden:
        :return:
        """
        # input: [1]
        # embedding(input): [1, emb_dim]
        # embedded: [1, 1, 1 * emb_dim]
        embedded = self.embedding(input).view(1, 1, -1)

        # 关于gru的输入输出参数
        # [seq_len, batch_size, feture_size]
        # output: [1, 1, 1 * emb_dim]
        output = embedded
        # hidden: [1, 1, hidden_size]
        # 这里hidden_size == emb_dim
        output, hidden = self.gru(output, hidden)
        # output: [seq_len, batch, num_directions * hidden_size]
        # output: [1, 1, hidden_size]
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # embedding层的结构，1. 有多少个词，2. 每个词多少维
        self.embedding = nn.Embedding(output_size, hidden_size)
        # GRU的参数: 1. 输入x的维度, 2. 隐藏层状态的维度; 这里都用了hidden_size
        # emb_dim == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)
        # [batch_size, hidden_size] -> [batch_size, output_size]
        # 这里output_size就是目标语言字典的大小V
        self.out = nn.Linear(hidden_size, output_size)
        # softmax层, 求每一个单词的概率
        self.softmax = nn.LogSoftmax(dim=1)  # ?

    def forward(self, input, hidden):
        # input: [1], 一个单词的下标
        # hidden: [1, 1, hidden_size]
        # embedding(input): [emb_dim]
        output = self.embedding(input).view(1, 1, -1)  # 展开
        # output: [1, 1, emb_dim]
        output = F.relu(output)
        # output: [1, 1, emb_dim]

        # 关于gru的输入输出参数
        # [seq_len, batch_size, input_size],  [num_layers * num_directions, batch_size, hidden_size]
        # output: [1, 1, emb_dim], hidden: [1, 1, hidden_size]
        output, hidden = self.gru(output, hidden)
        # output: [1, 1, hidden_size] # [seq_len, batch, num_directions * hidden_size] # 这里hidden_size == emb_dim
        # output[0]: [1, emb_dim]
        # self.out(output[0]): [1, V]
        # output: [1, V] 值为每个单词的概率
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttentionDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        # embedding层的结构，1. 有多少个词，2. 每个词多少维
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # * 2 = cat(embeding, hidden)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        # GRU的参数: 1. 输入x的维度, 2. 隐藏层状态的维度; 这里都用了hidden_size
        # emb_dim == hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        # [batch_size, hidden_size] -> [batch_size, output_size]
        # 这里output_size就是目标语言字典的大小V
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # input: [1], 一个单词的下标
        # hidden: [1, 1, hidden_size]
        # embedding(input): [emb_dim]
        embedded = self.embedding(input).view(1, 1, -1)  # 展开
        embedded = self.dropout(embedded)
        # embedded: [1, 1, emb_dim]

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)),
            dim=1
        )
        # attn_weights: [1, MAX_LENGTH]
        # encoder_outputs: [max_length, encoder.hidden_size]
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        # attn_applied: [1, 1, hidden_size]
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        # output: [1, hidden_size * 2]
        output = self.attn_combine(output).unsqueeze(0)
        # output: [1, 1, hidden_size]

        output = F.relu(output)
        # output: [1, 1, hidden_size]

        # 关于gru的输入输出参数
        # [seq_len, batch_size, input_size],  [num_layers * num_directions, batch_size, hidden_size]
        # emb_dim == hidden_size
        # output: [1, 1, emb_dim], hidden: [1, 1, hidden_size]
        output, hidden = self.gru(output, hidden)
        # output: [1, 1, hidden_size] # [seq_len, batch, num_directions * hidden_size] # 这里hidden_size == emb_dim
        # output[0]: [1, emb_dim]
        # self.out(output[0]): [1, V]
        output = F.log_softmax(self.out(output[0]), dim=1)
        # output[1, V], 值为每个单词的概率
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

