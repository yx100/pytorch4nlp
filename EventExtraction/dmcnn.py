#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/9/14
from collections import OrderedDict

import torch
import torch.nn as nn
from pt4nlp import Embeddings, MultiPoolingCNNEncoder


class DynamicMultiPoolingCNN(nn.Module):
    def __init__(self, dicts, opt, label_num, position_dict, lexi_window=1):
        super(DynamicMultiPoolingCNN, self).__init__()
        self.word_vec_size = opt.word_vec_size
        self.posi_vec_size = opt.posi_vec_size
        self.lexi_window = lexi_window
        if opt.posi_vec_size > 0:
            self.embedding = Embeddings(word_vec_size=opt.word_vec_size,
                                        dicts=dicts,
                                        feature_dicts=[position_dict],
                                        feature_dims=[opt.posi_vec_size],
                                        )
        else:
            self.embedding = Embeddings(word_vec_size=opt.word_vec_size,
                                        dicts=dicts,
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
            encoder_output_size = self.encoder.output_size + (2 * lexi_window + 1) * self.word_vec_size
        else:
            encoder_output_size = self.encoder.output_size
        out_component = OrderedDict()
        if opt.bn:
            out_component['bn'] = nn.BatchNorm1d(encoder_output_size)
        out_component['act'] = getattr(nn, opt.act)()
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
            lexi_feature = self.embedding(batch.lexi)
            lexi_feature = lexi_feature.resize(lexi_feature.size()[0], (2 * self.lexi_window + 1) * self.word_vec_size)
        if self.posi_vec_size > 0:
            words_embeddings = self.embedding(batch.text)
        else:
            words_embeddings = self.embedding(batch.text[:, 0])
        sentence_embedding = self.encoder(words_embeddings, position=batch.position, lengths=batch.lengths)
        sentence_feature = torch.cat([sentence_embedding, lexi_feature], dim=1)
        scores = self.out(sentence_feature)
        return scores
