#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/10/8
from collections import OrderedDict
import torch
import torch.nn as nn
from pt4nlp import Embeddings
from pt4nlp.pooling import get_pooling


class ANNEventExtractor(nn.Module):
    """
    ANN Event Extractor from Liu Shulin
    """
    def __init__(self, dicts, opt, label_num):
        super(ANNEventExtractor, self).__init__()
        self.word_vec_size = opt.word_vec_size
        self.hidden_size = opt.hidden_size

        self.embedding = Embeddings(word_vec_size=opt.word_vec_size,
                                    dicts=dicts,
                                    )

        encoder_output_size = self.word_vec_size

        if self.lexi_window >= 0:
            encoder_output_size += (2 * self.lexi_window + 1) * self.word_vec_size
        out_component = OrderedDict()
        if opt.bn:
            out_component['bn'] = nn.BatchNorm1d(encoder_output_size)
        out_component['dropout1'] = nn.Dropout(opt.dropout)
        out_component['linear1'] = nn.Linear(encoder_output_size, self.hidden_siz)
        out_component['act'] = getattr(nn, opt.act)()
        out_component['dropout2'] = nn.Dropout(opt.dropout)
        out_component['linear2'] = nn.Linear(self.hidden_siz, label_num)

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

        words_embeddings = self.embedding.forward(batch.text[:, :, 0])

        sentence_embedding = get_pooling(words_embeddings, pooling_type='mean', lengths=batch.lengths)

        if self.lexi_window >= 0:
            sentence_feature = torch.cat([sentence_embedding, lexi_feature], dim=1)
        else:
            sentence_feature = sentence_embedding

        scores = self.out(sentence_feature)
        return scores
