#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/9/7
import torch
import torch.nn as nn

from pt4nlp import Embeddings


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

        # (batch, 1, len_c2, hidden) -> (batch, hidden, len_c2, 1)
        x1 = torch.transpose(h1, 1, 3)
        # (batch, hidden, len_c2, 1) -> (batch, 1, len_c1, hidden)
        h2 = self.de_layer2(x1)

        # (batch, 1, len_c1, hidden) -> (batch, hidden, len_c1, 1)
        x2 = torch.transpose(h2, 1, 3)
        # (batch, hidden, len_c1, 1) -> (batch, 1, len, hidden)
        h3 = self.de_layer1(x2)

        # (batch, 1, len, hidden) -> (batch, len, hidden)
        return h3.squeeze(1)


class DeconvolutionAutoEncoer(nn.Module):
    def __init__(self,
                 input_size=300,
                 hidden_size=(300, 600, 500),
                 window_size=(5, 5, 12),
                 stride_size=(2, 2, 1)
                 ):
        super(DeconvolutionAutoEncoer, self).__init__()
        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size,
                               window_size=window_size, stride_size=stride_size)
        self.decoder = Decoder(output_size=input_size, hidden_size=hidden_size,
                               window_size=window_size, stride_size=stride_size)

    def forward(self, x):
        hidden = self.encoder.forward(x)
        return self.decoder.forward(hidden)


class TextDeconvolutionAutoEncoer(nn.Module):
    def __init__(self,
                 dicts,
                 word_vec_size=300,
                 hidden_size=(300, 600, 500),
                 window_size=(5, 5, 12),
                 stride_size=(2, 2, 1)
                 ):
        super(TextDeconvolutionAutoEncoer, self).__init__()
        self.embedding = Embeddings(word_vec_size=word_vec_size,
                                    dicts=dicts,
                                    )
        self.auto_encoder = DeconvolutionAutoEncoer(input_size=word_vec_size,
                                                    hidden_size=hidden_size,
                                                    window_size=window_size,
                                                    stride_size=stride_size,
                                                    )

    def forward(self, batch):
        # (batch, len) -> (batch, len, dim)
        word_embeddings = self.embedding.forward(batch.text)
        # (batch, len, dim) -> (batch, len, dim)
        de_word_embeddings = self.auto_encoder.forward(word_embeddings)
        # (batch, len, dim) dot (dim, dictionary) -> (batch, len, dictionary)
        to_decode = de_word_embeddings.view(de_word_embeddings.size(0) * de_word_embeddings.size(1),
                                            de_word_embeddings.size(2))
        decoded = torch.mm(to_decode,
                           self.embedding.word_lookup_table.weight.t())
        return decoded.view(de_word_embeddings.size(0) * de_word_embeddings.size(1), decoded.size(1))


def test():
    from torch.autograd import Variable

    x = torch.FloatTensor(20, 59, 300)
    x.normal_()
    x = Variable(x)
    encoder = Encoder()
    decoder = Decoder()
    loss_function = torch.nn.MSELoss()
    lr = 0.1
    for i in xrange(500):
        hidden = encoder.forward(x)
        y = decoder.forward(hidden)
        loss = loss_function.forward(y, x)
        print(i, loss)
        loss.backward()
        for para in encoder.parameters():
            para.data.add_(-lr, para.grad.data)
        for para in decoder.parameters():
            para.data.add_(-lr, para.grad.data)


if __name__ == "__main__":
    test()
