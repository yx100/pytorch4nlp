#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/9/14
from collections import OrderedDict

import torch.nn as nn
from pt4nlp import Embeddings, MultiPoolingCNNEncoder


class DynamicMultiPoolingCNN(nn.Module):
    def __init__(self, dicts, opt, label_num, position_dict, lexi_window=1):
        super(DynamicMultiPoolingCNN, self).__init__()
        self.lexi_window = lexi_window
        self.embedding = Embeddings(word_vec_size=opt.word_vec_size,
                                    dicts=dicts,
                                    feature_dicts=[position_dict],
                                    feature_dims=[5],
                                    )
        self.encoder = MultiPoolingCNNEncoder(self.embedding.output_size,
                                              hidden_size=opt.hidden_size,
                                              window_size=int(opt.cnn_size[0]),
                                              pooling_type=opt.cnn_pooling,
                                              padding=True,
                                              dropout=opt.encoder_dropout,
                                              bias=True,
                                              split_point_number=1)
        if lexi_window > 0:
            encoder_output_size = self.encoder.output_size + (2 * lexi_window + 1) * self.embedding.output_size
        else:
            encoder_output_size = self.encoder.output_size
        out_component = OrderedDict()
        if opt.bn:
            out_component['bn'] = nn.BatchNorm1d(encoder_output_size)
        out_component['dropout'] = nn.Dropout(opt.dropout)
        out_component['linear'] = nn.Linear(encoder_output_size, label_num)
        self.out = nn.Sequential(out_component)
        self.init_model()

    def init_model(self):
        for name, param in self.out.named_parameters():
            if param.data.dim() == 2:
                print("Init %s with %s" % (name, "xavier_uniform"))
                nn.init.xavier_uniform(param)

    def forward(self, batch):
        if self.lexi_window > 0:
            lexi_embeddings = self.embedding(batch.lexi)
        words_embeddings = self.embedding(batch.text)
        sentence_embedding = self.encoder(words_embeddings, position=batch.position, lengths=batch.lengths)
        scores = self.out(sentence_embedding)
        return scores
