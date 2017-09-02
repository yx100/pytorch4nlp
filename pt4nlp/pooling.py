#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/9/2
import torch
from torch.autograd import Variable


def lengths2mask(lengths, max_length):
    batch_size = lengths.size(0)
    # print max_length
    # print torch.max(lengths)[0]
    # assert max_length == torch.max(lengths)[0]
    range_i = torch.arange(0, max_length).expand(batch_size, max_length)
    range_i = Variable(range_i)
    return torch.le(range_i, lengths.float()[:, None]).float()


def mask_mean_pooling(inputs, mask):
    return torch.sum(inputs, 1) / torch.sum(mask, 1)[:, None].float()


def mask_sum_pooling(inputs, mask):
    return torch.sum(inputs * mask[:, :, None], 1)


def mask_max_pooling(inputs, mask):
    return torch.max(inputs * mask[:, :, None], 1)[0]


def mask_min_pooling(inputs, mask):
    return torch.min(inputs * mask[:, :, None], 1)[0]


def mean_pooling(inputs):
    return torch.sum(inputs, 1)


def sum_pooling(inputs):
    return torch.sum(inputs, 1)


def max_pooling(inputs):
    return torch.max(inputs, 1)[0]


def min_pooling(inputs):
    return torch.min(inputs, 1)[0]


def get_pooling(inputs, pooling_type='mean', lengths=None):
    if lengths is not None:
        mask = lengths2mask(lengths, inputs.size()[1])
        if pooling_type == 'mean':
            return mask_mean_pooling(inputs, mask)
        elif pooling_type == 'max':
            return mask_max_pooling(inputs, mask)
        elif pooling_type == 'min':
            return mask_min_pooling(inputs, mask)
        elif pooling_type == 'sum':
            return mask_sum_pooling(inputs, mask)
        else:
            raise NotImplementedError
    else:
        if pooling_type == 'mean':
            return mean_pooling(inputs)
        elif pooling_type == 'max':
            return max_pooling(inputs)
        elif pooling_type == 'min':
            return min_pooling(inputs)
        elif pooling_type == 'sum':
            return sum_pooling(inputs)
        else:
            raise NotImplementedError
