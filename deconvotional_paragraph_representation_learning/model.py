#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/9/7
import torch
import torch.nn as nn


class Encoder(nn.ModuleList):
    def __init__(self,
                 input_size=300,
                 hidden_size=(300, 600, 500),
                 window_size=(5, 5, 12),
                 stride_size=(2, 2, 1)
                 ):
        super(Encoder, self).__init__()
        self.en_layer1 = nn.Conv2d(in_channels=1, out_channels=hidden_size[0], kernel_size=(window_size[0], input_size),
                                   stride=stride_size[0])
        self.en_layer2 = nn.Conv2d(in_channels=1, out_channels=hidden_size[1],
                                   kernel_size=(window_size[1], hidden_size[0]), stride=stride_size[1])
        self.fclayer = nn.Conv2d(in_channels=1, out_channels=hidden_size[2],
                                 kernel_size=(window_size[2], hidden_size[1]), stride=stride_size[2])

    def forward(self, inputs):
        """
        :param inputs: (batch, len, dim)
        :return:
        """
        # (batch, len, dim) -> (batch, 1, len, dim)
        x = torch.unsqueeze(inputs, 1)
        # (batch, 1, len, dim) -> (batch, hidden, len_c1, 1)
        h1 = self.en_layer1(x)
        # (batch, hidden, len_c1, 1) -> (batch, 1, len_c1, hidden)
        x1 = torch.transpose(h1, 1, 3)
        # (batch, 1, len_c1, hidden) -> (batch, hidden, len_c2, 1)
        h2 = self.en_layer2(x1)
        # (batch, hidden, len_c2, 1) -> (batch, 1, len_c2, hidden)
        x2 = torch.transpose(h2, 1, 3)
        # (batch, 1, len_c2, hidden) -> (batch, 1, 1, hidden)
        h3 = self.fclayer(x2)
        # (batch, 1, 1, hidden) -> (batch, hidden)
        return h3.squeeze(2).squeeze(2)


class Decoder(nn.ModuleList):
    def __init__(self,
                 output_size=300,
                 hidden_size=(300, 600, 500),
                 window_size=(5, 5, 12),
                 stride_size=(2, 2, 1)
                 ):
        super(Decoder, self).__init__()
        self.size_de_fc = 12
        self.size_de_layer2 = 28
        self.size_de_layer1 = 60

        self.de_layer1 = nn.ConvTranspose2d(in_channels=hidden_size[0], out_channels=1,
                                            kernel_size=(window_size[0], output_size),
                                            stride=stride_size[0])
        self.de_layer2 = nn.ConvTranspose2d(in_channels=hidden_size[1], out_channels=1,
                                            kernel_size=(window_size[1], hidden_size[0]),
                                            stride=stride_size[1])
        self.de_fclayer = nn.ConvTranspose2d(in_channels=hidden_size[2], out_channels=1,
                                             kernel_size=(window_size[2], hidden_size[1]), stride=stride_size[2])

    def forward(self, inputs):
        """
        :param inputs: (batch, dim)
        :return:
        """
        # (batch, dim) -> (batch, dim, 1, 1)
        x = torch.unsqueeze(torch.unsqueeze(inputs, 2), 2)
        # (batch, 1, 1, dim) -> (batch, 1, len_c2, dim)
        h1 = self.de_fclayer(x)

        expand_size = self.size_de_fc - h1.size(2)
        if expand_size > 0:
            h1 = torch.cat([h1, h1[:, :, -expand_size:, :]], 2)

        # (batch, 1, len_c2, hidden) -> (batch, hidden, len_c2, 1)
        x1 = torch.transpose(h1, 1, 3)
        # (batch, hidden, len_c2, 1) -> (batch, 1, len_c1, hidden)
        h2 = self.de_layer2(x1)

        expand_size = self.size_de_layer2 - h2.size(2)
        if expand_size > 0:
            h2 = torch.cat([h2, h2[:, :, -expand_size:, :]], 2)

        # (batch, 1, len_c1, hidden) -> (batch, hidden, len_c1, 1)
        x2 = torch.transpose(h2, 1, 3)
        # (batch, hidden, len_c1, 1) -> (batch, 1, len, hidden)
        h3 = self.de_layer1(x2)
        expand_size = self.size_de_layer1 - h3.size(2)
        if expand_size > 0:
            h3 = torch.cat([h3, h3[:, :, -expand_size:, :]], 2)

        # (batch, 1, len, hidden) -> (batch, len, hidden)
        return h3.squeeze(1)
