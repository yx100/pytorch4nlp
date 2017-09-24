#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/9/14
from collections import OrderedDict

import torch
import torch.nn as nn
from pt4nlp import Embeddings, MultiSizeMultiPoolingCNNEncoder


class DynamicMultiPoolingCNN(nn.Module):
    def __init__(self, dicts, opt, label_num, position_dict):
        super(DynamicMultiPoolingCNN, self).__init__()
        self.word_vec_size = opt.word_vec_size
        self.posi_vec_size = opt.posi_vec_size
        self.lexi_window = opt.lexi_window

        feature_dicts = list()
        feature_dims = list()
        if opt.posi_vec_size > 0:
            feature_dicts += [position_dict]
            feature_dims += [opt.posi_vec_size]

        self.embedding = Embeddings(word_vec_size=opt.word_vec_size,
                                    dicts=dicts,
                                    feature_dicts=feature_dicts,
                                    feature_dims=feature_dims,
                                    )

        self.encoder = MultiSizeMultiPoolingCNNEncoder(self.embedding.output_size,
                                                       hidden_size=opt.hidden_size,
                                                       window_size=[int(ws) for ws in opt.cnn_size],
                                                       pooling_type=opt.cnn_pooling,
                                                       dropout=opt.encoder_dropout,
                                                       bias=True,
                                                       split_point_number=1)

        encoder_output_size = self.encoder.output_size
        self.act_function = getattr(nn, opt.act)()
        if self.lexi_window >= 0:
            encoder_output_size += (2 * self.lexi_window + 1) * self.word_vec_size
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
        if self.lexi_window >= 0:
            lexi_feature = self.embedding.forward(batch.lexi)
            lexi_feature = lexi_feature.resize(lexi_feature.size()[0], (2 * self.lexi_window + 1) * self.word_vec_size)

        if self.posi_vec_size > 0:
            words_embeddings = self.embedding.forward(batch.text)
        else:
            # ignore position
            words_embeddings = self.embedding.forward(batch.text[:, :, 0])

        sentence_embedding = self.encoder.forward(words_embeddings, position=batch.position, lengths=batch.lengths)
        sentence_embedding = self.act_function(sentence_embedding)
        if self.lexi_window >= 0:
            sentence_feature = torch.cat([sentence_embedding, lexi_feature], dim=1)
        else:
            sentence_feature = sentence_embedding

        scores = self.out(sentence_feature)
        return scores
