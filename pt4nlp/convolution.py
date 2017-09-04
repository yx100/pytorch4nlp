#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/9/4
import torch
import torch.nn as nn

from pooling import get_pooling


class CNNEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=168,
                 window_size=3,
                 pooling_type='max',
                 padding=True,
                 dropout=0.5,
                 bias=True):
        super(CNNEncoder, self).__init__()
        # Define Parameter
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.pooling_type = pooling_type
        self.padding = padding
        self.dropout = dropout
        self.bias = bias

        # Define Layer
        # (N, Cin, Hin, Win)
        # In NLP, Hin is length, Win is Word Embedding Size
        self.conv_layer = nn.Conv2d(in_channels=1,
                                    out_channels=hidden_size,
                                    kernel_size=(self.window_size, self.input_size),
                                    # window_size-1 padding for length
                                    # zero padding for word dim
                                    padding=(self.window_size - 1, 0) if padding else 0,
                                    bias=bias)
        self.dropout_layer = nn.Dropout(self.dropout)

        self.init_model()

        self.output_size = self.hidden_size

    def init_model(self):
        pass

    def forward_conv(self, inputs):
        """
        :param inputs: batch x len x input_size
        :return:
                if padding is False:
                    batch x len - window_size + 1 x hidden_size
                if padding is True
                    batch x len + window_size - 1 x hidden_size
        """
        # (batch x len x input_size) -> (batch x 1 x len x input_size)
        inputs = torch.unsqueeze(inputs, 1)
        # (batch x 1 x len x input_size) -> (batch x hidden_size x new_len x 1)
        _temp = self.conv_layer(inputs)
        # (batch x hidden_size x new_len x 1)
        # -> (batch x hidden_size x new_len)
        # -> (batch x new_len x hidden_size)
        _temp.squeeze_(3)
        return torch.transpose(_temp, 1, 2)

    def forward(self, inputs, lengths=None):
        """
        :param inputs: batch x len x input_size
        :param lengths: batch
        :return: batch x hidden_size
        """
        dp_input = self.dropout_layer(inputs)
        conv_result = self.forward_conv(dp_input)
        if lengths is not None:
            if self.padding:
                lengths = lengths + (self.window_size - 1)
            else:
                lengths = lengths - (self.window_size + 1)
        pooling_result = get_pooling(conv_result, pooling_type=self.pooling_type, lengths=lengths)
        return pooling_result
