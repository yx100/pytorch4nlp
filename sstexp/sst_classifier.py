#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/8/26
from __future__ import absolute_import
import common
from collections import OrderedDict
import torch.nn as nn
from pt4nlp import RNNEncoder, Embeddings, CBOW, MultiSizeCNNEncoder


class SSTClassifier(nn.Module):
    def __init__(self, dicts, opt, label_num):
        super(SSTClassifier, self).__init__()
        self.embedding = Embeddings(word_vec_size=opt.word_vec_size,
                                    dicts=dicts)
        if opt.encoder == "rnn":
            self.encoder = RNNEncoder(input_size=self.embedding.output_size,
                                      hidden_size=opt.hidden_size,
                                      num_layers=opt.num_layers,
                                      dropout=opt.encoder_dropout,
                                      brnn=opt.brnn,
                                      rnn_type=opt.rnn_type)
        elif opt.encoder == "cbow":
            self.encoder = CBOW(self.embedding.output_size)
        elif opt.encoder == "cnn":
            self.encoder = MultiSizeCNNEncoder(self.embedding.output_size,
                                               hidden_size=opt.hidden_size,
                                               window_size=[int(ws) for ws in opt.cnn_size],
                                               pooling_type=opt.cnn_pooling,
                                               padding=True,
                                               dropout=opt.encoder_dropout,
                                               bias=True)
        else:
            raise NotImplementedError
        out_component = OrderedDict()
        if opt.bn:
            out_component['bn'] = nn.BatchNorm1d(self.encoder.output_size)
        out_component['dropout'] = nn.Dropout(opt.dropout)
        out_component['linear'] = nn.Linear(self.encoder.output_size, label_num)
        self.out = nn.Sequential(out_component)
        self.init_model()

    def init_model(self):
        for name, param in self.out.named_parameters():
            if param.data.dim() == 2:
                print("Init %s with %s" % (name, "xavier_uniform"))
                nn.init.xavier_uniform(param)

    def forward(self, batch):
        words_embeddings = self.embedding.forward(batch.text)
        sentence_embedding = self.encoder.forward(words_embeddings, batch.lengths)
        scores = self.out(sentence_embedding)
        return scores
