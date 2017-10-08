#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/9/19
import torch
import torch.nn as nn
from torch.nn import Parameter
from pt4nlp.utils import aeq


class Matcher(nn.Module):

    def __init__(self):
        super(Matcher, self).__init__()

    def score(self, x1, x2): raise NotImplementedError

    def forward(self, x1, x2):
        self.score(x1, x2)


class DotMatcher(Matcher):

    def __init__(self):
        super(DotMatcher, self).__init__()

    def score(self, x1, x2):
        """
        Score with Dot
        :param x1: (batch, dim)
        :param x2: (batch, dim)
        :return: (batch, )
        """
        aeq([x1.size()[0], x2.size()[0]])
        aeq([x1.size()[1], x2.size()[1]])

        return torch.sum(x1 * x2, 1)


class MLPMatcher(Matcher):

    def __init__(self, input_dim1, input_dim2, output_dim, dropout=0):
        super(MLPMatcher, self).__init__()
        out_component = OrderedDict()
        self.linear_layer = nn.Linear(input_dim * 2, 1)

    def score(self, x1, x2):
        """
        Score with Dot
        :param x1: (batch, dim)
        :param x2: (batch, dim)
        :return: (batch, )
        """
        hidden = torch.cat([x1, x2], 1)
        return self.linear_layer(hidden)


class BilinearMatcher(Matcher):

    def __init__(self, input_dim1, input_dim2, output_dim):
        super(BilinearMatcher, self).__init__()
        self.bilinear = nn.Bilinear(input_dim1, input_dim2, output_dim)

    def score(self, x1, x2):
        return self.bilinear.forward(x1, x2)
