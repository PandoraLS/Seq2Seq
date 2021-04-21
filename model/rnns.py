# -*- coding: utf-8 -*-
# @Time    : 2021/4/21 下午8:16


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary

class Net(nn.Module):
    def __init__(self,
                 vocab_size=20, embed_dim=300,
                 hidden_dim=512, num_layers=2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim,
                               num_layers=num_layers)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embed = self.embedding(x)
        out, hidden = self.encoder(embed)
        out = self.decoder(out)
        out = out.view(-1, out.size(2))
        return out, hidden
inputs = torch.zeros((100, 1), dtype=torch.long) # [length, batch_size]
summary(Net(), inputs)