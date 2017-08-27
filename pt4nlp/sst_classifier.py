#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/8/26
from __future__ import absolute_import
import torch.nn as nn
from rnn_encoder import RNNEncoder
from embedding import Embeddings


class SSTClassifier(nn.Module):
    def __init__(self, dicts, word_vec_dim=300):
        super(SSTClassifier, self).__init__()
        self.embedding = Embeddings(word_vec_size=word_vec_dim,
                                    dicts=dicts)
        self.encoder = RNNEncoder(input_size=self.embedding.output_size)
        self.out = nn.Sequential(nn.ReLU(),
                                 nn.Linear(self.encoder.output_size, 5),)
        self.init_model()

    def init_model(self):
        for weight in self.out.parameters():
            if weight.data.dim() == 2:
                nn.init.xavier_normal(weight)

    def forward(self, batch):
        words_embeddings = self.embedding(batch.text)
        sentence_embedding = self.encoder(words_embeddings)
        scores = self.out(sentence_embedding)
        return scores
