#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/8/26
from __future__ import absolute_import
import torch.nn as nn
from rnn_encoder import RNNEncoder
from embedding import Embeddings


class SSTClassifier(nn.Module):
    def __init__(self, dicts, opt, label_num):
        super(SSTClassifier, self).__init__()
        self.embedding = Embeddings(word_vec_size=opt.word_vec_size,
                                    dicts=dicts)
        self.encoder = RNNEncoder(input_size=self.embedding.output_size,
                                  hidden_size=opt.hidden_size,
                                  num_layers=opt.num_layers,
                                  dropout=opt.dropout,
                                  brnn=opt.brnn,
                                  rnn_type=opt.rnn_type)
        self.out = nn.Sequential(nn.Dropout(opt.dropout),
                                 nn.Linear(self.encoder.output_size, label_num),)
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
