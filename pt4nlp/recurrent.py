#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/8/27
import torch.nn as nn
from torch.autograd import Variable


class RNNEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=168,
                 num_layers=1,
                 dropout=0.2,
                 brnn=True,
                 rnn_type="LSTM"):
        super(RNNEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = brnn
        self.rnn_type = rnn_type.lower()
        self.output_size = 2 * self.hidden_size if self.bidirectional else self.hidden_size

        if self.rnn_type == "lstm":
            rnn = nn.LSTM
        elif self.rnn_type == "gru":
            rnn = nn.GRU
        elif self.rnn_type == "rnn":
            rnn = nn.RNN
        else:
            raise NotImplementedError("RNN Type: LSTM GRU RNN")
        self.rnn = rnn(input_size=self.input_size, hidden_size=self.hidden_size,
                       num_layers=self.num_layers, dropout=self.dropout,
                       bidirectional=self.bidirectional, batch_first=True)
        self.init_model()

    def init_model(self):
        for weight in self.rnn.parameters():
            if weight.data.dim() == 2:
                nn.init.xavier_normal(weight)

    def forward(self, *args, **kwargs):
        """
        :param args:
        :param kwargs:
        :return:
            outputs:     (batch, num_directions * hidden_size)
            last_hidden: (batch, batch, hidden_size * num_directions)
        """
        inputs = args[0]
        batch_size = inputs.size()[0]

        # (num_layers * num_directions, batch, hidden_size)
        n_cells = self.num_layers * 2 if self.bidirectional else self.num_layers
        state_shape = n_cells, batch_size, self.hidden_size

        # h0 for GRU RNN, (h0, c0) for LSTM
        if self.rnn_type == "lstm":
            h0 = c0 = Variable(inputs.data.new(*state_shape).zero_())
            h0_state = (h0, c0)
        else:
            h0_state = Variable(inputs.data.new(*state_shape).zero_())

        # # ht for GRU RNN, (ht, ct) for LSTM
        outputs, ht_state = self.rnn(inputs, h0_state)

        if self.rnn_type == "lstm":
            # (num_layers * num_directions, batch, hidden_size), (num_layers * num_directions, batch, hidden_size)
            ht = ht_state[0]
        else:
            # (num_layers * num_directions, batch, hidden_size)
            ht = ht_state

        # (num_layers * num_directions, batch, hidden_size)
        # -> Last Layer (batch, num_directions * hidden_size)
        last_hidden = ht[-1] if not self.bidirectional else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)

        return outputs, last_hidden
