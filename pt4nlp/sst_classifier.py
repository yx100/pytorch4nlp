#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/8/26
from __future__ import absolute_import
import torch.nn as nn
from rnn_encoder import RNNEncoder
from embedding import Embeddings

from cbow import CBOW


class SSTClassifier(nn.Module):
    def __init__(self, dicts, opt, label_num):
        super(SSTClassifier, self).__init__()
        self.embedding = Embeddings(word_vec_size=opt.word_vec_size,
                                    dicts=dicts)
        if opt.encoder == "rnn":
            self.encoder = RNNEncoder(input_size=self.embedding.output_size,
                                      hidden_size=opt.hidden_size,
                                      num_layers=opt.num_layers,
                                      dropout=opt.dropout,
                                      brnn=opt.brnn,
                                      rnn_type=opt.rnn_type)
        elif opt.encoder == "cbow":
            self.encoder = CBOW(self.embedding.output_size)
        else:
            raise NotImplementedError
        self.out = nn.Sequential(nn.Dropout(opt.dropout),
                                 nn.Linear(self.encoder.output_size, label_num), )
        self.init_model()

    def init_model(self):
        for name, param in self.out.named_parameters():
            if param.data.dim() == 2:
                print("Init %s with %s" % (name, "xavier_uniform"))
                nn.init.xavier_uniform(param)

    def forward(self, batch):
        words_embeddings = self.embedding(batch.text)
        sentence_embedding = self.encoder(words_embeddings, batch.lengths)
        scores = self.out(sentence_embedding)
        return scores
