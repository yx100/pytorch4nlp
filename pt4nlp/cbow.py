#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/8/27
import torch
import torch.nn as nn


class CBOW(nn.Module):
    def __init__(self,
                 input_size,
                 pooling_type="mean"):
        super(CBOW, self).__init__()
        self.pooling_type = pooling_type
        self.output_size = input_size

        self.init_model()

    def init_model(self): pass

    def forward(self, inputs, lengths=None):
        """
        :param inputs:   batch x len x dim
        :param lengths:  batch
        :return:
        """
        if self.pooling_type == 'mean':
            hidden = torch.sum(inputs, 1)
            return hidden / lengths[:, None].float()
        return
