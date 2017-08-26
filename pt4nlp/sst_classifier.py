#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/8/26
import torch.nn as nn
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.input_size = 300
        self.hidden_size = 168
        self.num_layers = 1
        self.dropout = 0.2
        self.bidirectional = True
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                           num_layers=self.num_layers, dropout=self.dropout,
                           bidirectional=self.bidirectional, batch_first=True)

    def forward(self, inputs):
        batch_size = inputs.size()[0]
        n_cells = self.num_layers * 2 if self.bidirectional else self.num_layers
        state_shape = n_cells, batch_size, self.hidden_size
        h0 = c0 = Variable(inputs.data.new(*state_shape).zero_())
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        return ht[-1] if not self.bidirectional else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)


class SSTClassifier(nn.Module):
    def __init__(self, word_num):
        super(SSTClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=word_num,
                                      embedding_dim=300,
                                      padding_idx=None, )
        self.encoder = Encoder()
        self.out = nn.Sequential(nn.ReLU(),
                                 nn.Linear(168 * 2, 5))

    def forward(self, batch):
        words_embeddings = self.embedding(batch.text)
        sentence_embedding = self.encoder(words_embeddings)
        scores = self.out(sentence_embedding)
        return scores
