# -*- coding:utf-8 -*- 
# Author: Roger
# Created by Roger on 2017/10/23
import abc
from collections import OrderedDict
import torch
import torch.nn as nn
from mask_util import lengths2mask


class WordSeqAttentionModel(nn.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self, input_size, seq_size):
        super(WordSeqAttentionModel, self).__init__()
        self.input_size = input_size
        self.seq_size = seq_size

    @abc.abstractmethod
    def score(self, x, seq): pass

    def attention(self, x, seq, lengths=None):
        """
        :param x: (batch, dim, )
        :param seq: (batch, length, dim, )
        :param lengths: (batch, )
        :return: weight: (batch, length)
        """
        # Check Size
        batch_size, input_size = x.size()
        seq_batch_size, max_len, seq_size = seq.size()
        assert batch_size == seq_batch_size

        score = self.score(x, seq, lengths)
        exp_score = torch.exp(score)
        if lengths is not None:
            mask = lengths2mask(lengths, max_len)
            exp_score = mask * exp_score
        sum_exp_score = torch.sum(exp_score, 1)
        weight = exp_score / sum_exp_score[:, None]
        return weight

    def forward(self, x, seq, lengths):
        """
        :param x: (batch, dim, )
        :param seq: (batch, length, dim, )
        :param lengths: (batch, )
        :return: weight: (batch, length)
        """
        weight = self.attention(x, seq, lengths)
        weight_sum = weight[:, :, None] * seq
        return torch.mean(weight_sum, 1)

    def check_size(self, x, seq):
        batch_size, input_size = x.size()
        seq_batch_size, max_len, seq_size = seq.size()
        assert batch_size == seq_batch_size
        assert input_size == self.input_size
        assert seq_size == self.seq_size

    @staticmethod
    def mask_score(score, lengths=None):
        if lengths is None:
            return score
        else:
            mask = lengths2mask(lengths, score.size(1))
            return score * mask

    @staticmethod
    def expand_x(x, max_len):
        """
        :param x: (batch, input_size)
        :param max_len: scalar
        :return:  (batch * max_len, input_size)
        """
        batch_size, input_size = x.size()
        return torch.unsqueeze(x, 1).expand(batch_size, max_len, input_size).contiguous().view(batch_size * max_len, -1)

    @staticmethod
    def pack_seq(seq):
        """
        :param seq: (batch_size, max_len, seq_size)
        :return: (batch_size * max_len, seq_size)
        """
        return seq.view(seq.size(0) * seq.size(1), -1)


class DotWordSeqAttetnion(WordSeqAttentionModel):
    """
    Effective Approaches to Attention-based Neural Machine Translation
    Minh-Thang Luong, Hieu Pham, and Christopher D. Manning
    In Proceedings of EMNLP 2015
    http://aclweb.org/anthology/D/D15/D15-1166.pdf
    """
    def __init__(self, input_size, seq_size):
        super(DotWordSeqAttetnion, self).__init__(input_size=input_size, seq_size=seq_size)
        assert input_size == seq_size

    def score(self, x, seq, lengths=None):
        """
        :param x: (batch, dim)
        :param seq: (batch, length, dim)
        :param lengths: (batch, )
        :return: weight: (batch, length)
        """
        self.check_size(x, seq)

        # (batch, dim) -> (batch, dim, 1)
        _x = torch.unsqueeze(x, -1)

        # (batch, length, dim) dot (batch, dim, 1) -> (batch, length, 1)
        score = torch.bmm(seq, _x)

        # (batch, length, 1) -> (batch, length)
        score = torch.squeeze(score, -1)

        return self.mask_score(score, lengths)


class BilinearWordSeqAttention(WordSeqAttentionModel):
    """
    Effective Approaches to Attention-based Neural Machine Translation
    Minh-Thang Luong, Hieu Pham, and Christopher D. Manning
    In Proceedings of EMNLP 2015
    http://aclweb.org/anthology/D/D15/D15-1166.pdf
    """
    def __init__(self, input_size, seq_size):
        super(BilinearWordSeqAttention, self).__init__(input_size=input_size, seq_size=seq_size)
        # (word_dim, seq_dim)
        self.bilinear = nn.Bilinear(in1_features=input_size, in2_features=seq_size, out_features=1, bias=False)

    def score(self, x, seq, lengths=None):
        """
        :param x: (batch, word_dim)
        :param seq: (batch, length, seq_dim)
        :param lengths: (batch, )
        :return: score: (batch, length, )
        """
        self.check_size(x, seq)

        # (batch, word_dim) -> (batch * max_len, word_dim)
        _x = self.expand_x(x, max_len=seq.size(1))

        # (batch, max_len, seq_dim) -> (batch * max_len, seq_dim)
        _seq = self.pack_seq(seq)

        # (batch * max_len, word_dim) bilinear (batch * max_len, seq_dim)
        # -> (batch * max_len, 1)
        score = self.bilinear.forward(_x, _seq)

        # (batch * max_len, 1) -> (batch * max_len) -> (batch, max_len)
        score = torch.squeeze(score, -1).view(x.size(0), -1)

        return self.mask_score(score, lengths)


class ConcatWordSeqAttention(WordSeqAttentionModel):
    """
    Effective Approaches to Attention-based Neural Machine Translation
    Minh-Thang Luong, Hieu Pham, and Christopher D. Manning
    In Proceedings of EMNLP 2015
    http://aclweb.org/anthology/D/D15/D15-1166.pdf
    """
    def __init__(self, input_size, seq_size):
        super(ConcatWordSeqAttention, self).__init__(input_size=input_size, seq_size=seq_size)
        # (word_dim + seq_dim)
        self.layer = nn.Linear(input_size + seq_size, 1, bias=False)

    def score(self, x, seq, lengths=None):
        """
        :param x: (batch, word_dim)
        :param seq: (batch, length, seq_dim)
        :param lengths: (batch, )
        :return: score: (batch, length, )
        """
        self.check_size(x, seq)

        # (batch, word_dim) -> (batch * max_len, word_dim)
        _x = self.expand_x(x, max_len=seq.size(1))

        # (batch, max_len, seq_dim) -> (batch * max_len, seq_dim)
        _seq = self.pack_seq(seq)

        # (batch * max_len, word_dim) (batch * max_len, seq_dim) -> (batch * max_len, word_dim + seq_dim)
        to_input = torch.cat([_x, _seq], 1)

        # (batch * max_len, word_dim + seq_dim) -> (batch * max_len, 1) -> (batch * max_len, ) -> (batch, max_len)
        score = self.layer.forward(to_input).squeeze(-1).view(seq.size(0), seq.size(1))

        return self.mask_score(score, lengths)


class MLPWordSeqAttention(WordSeqAttentionModel):
    """
    Neural Machine Translation By Jointly Learning To Align and Translate
    Dzmitry Bahdanau, KyungHyun Cho, and Yoshua Bengio
    In Proceedings of ICLR 2015
    http://arxiv.org/abs/1409.0473v3
    """
    def __init__(self, input_size, seq_size, hidden_size, activation="Tanh", bias=False):
        super(MLPWordSeqAttention, self).__init__(input_size=input_size, seq_size=seq_size)
        self.bias = bias
        self.hidden_size = hidden_size
        component = OrderedDict()
        component['layer1'] = nn.Linear(input_size + seq_size, hidden_size, bias=bias)
        component['act'] = getattr(nn, activation)()
        component['layer2'] = nn.Linear(hidden_size, 1, bias=bias)
        self.layer = nn.Sequential(component)

    def score(self, x, seq, lengths=None):
        """
        :param x: (batch, word_dim)
        :param seq: (batch, length, seq_dim)
        :param lengths: (batch, )
        :return: score: (batch, length, )
        """
        self.check_size(x, seq)

        # (batch, word_dim) -> (batch * max_len, word_dim)
        _x = self.expand_x(x, max_len=seq.size(1))

        # (batch, max_len, seq_dim) -> (batch * max_len, seq_dim)
        _seq = self.pack_seq(seq)

        # (batch * max_len, word_dim) (batch * max_len, seq_dim) -> (batch * max_len, word_dim + seq_dim)
        to_input = torch.cat([_x, _seq], 1)

        # (batch * max_len, word_dim + seq_dim)
        #   -> (batch * max_len, 1)
        #       -> (batch * max_len, )
        #           -> (batch, max_len)
        score = self.layer.forward(to_input).squeeze(-1).view(seq.size(0), seq.size(1))

        return self.mask_score(score, lengths)


class DotMLPWordSeqAttention(WordSeqAttentionModel):
    """
    WebQA: A Chinese Open-Domain Factoid Question Answering Dataset
    Peng Li, Wei Li, Zhengyan He, Xuguang Wang, Ying Cao, Jie Zhou, and Wei Xu
    http://arxiv.org/abs/1607.06275
    """
    def __init__(self, input_size, seq_size, activation="Tanh", bias=False):
        super(DotMLPWordSeqAttention, self).__init__(input_size=input_size, seq_size=seq_size)
        self.bias = bias
        component = OrderedDict()
        component['layer1'] = nn.Linear(seq_size, input_size, bias=bias)
        component['act'] = getattr(nn, activation)()
        self.layer = nn.Sequential(component)

    def score(self, x, seq, lengths=None):
        """
        :param x: (batch, word_dim)
        :param seq: (batch, length, seq_dim)
        :param lengths: (batch, )
        :return: score: (batch, length, )
        """
        self.check_size(x, seq)

        # (batch, word_dim) -> (batch * max_len, word_dim)
        _x = self.expand_x(x, max_len=seq.size(1))

        # (batch, max_len, seq_dim) -> (batch * max_len, seq_dim)
        _seq = self.pack_seq(seq)

        # (batch * max_len, seq_dim) -> (batch * max_len, word_dim)
        _seq_output = self.layer(_seq)

        # (batch * max_len, word_dim) * (batch * max_len, word_dim)
        #   -> (batch * max_len, word_dim)
        #       -> (batch * max_len, 1)
        #           -> (batch, max_len)
        score = torch.sum(_x * _seq_output, 1).view(x.size(0), -1)

        return self.mask_score(score, lengths)


def get_test_val(batch_size=3, input_size=4, seq_size=4, max_len=5):
    import random
    from torch.autograd import Variable
    x = torch.FloatTensor(batch_size, input_size)
    x.normal_()
    seq = torch.FloatTensor(batch_size, max_len, seq_size)
    lens = [random.randint(1, max_len-1) for i in range(batch_size)]
    lengths = Variable(torch.LongTensor(lens))
    seq.normal_()
    x = Variable(x)
    seq = Variable(seq)
    return x, seq, lengths


def test_dot_attention(batch_size=3, input_size=4, seq_size=4, max_len=5):
    print("Dot Attention")
    x, seq, lengths = get_test_val(batch_size, input_size, seq_size, max_len)
    attention = DotWordSeqAttetnion(input_size, seq_size)
    print(x)
    print(lengths)
    print(attention.score(x, seq, lengths))
    print(attention.forward(x, seq, lengths))


def test_bilinear_attention(batch_size=3, input_size=4, seq_size=3, max_len=5):
    print("Bilinear Attention")
    x, seq, lengths = get_test_val(batch_size, input_size, seq_size, max_len)
    attention = BilinearWordSeqAttention(input_size, seq_size)
    print(x)
    print(lengths)
    print(attention.score(x, seq, lengths))
    print(attention.forward(x, seq, lengths))


def test_concat_attention(batch_size=3, input_size=4, seq_size=3, max_len=5):
    print("Concat Attention")
    x, seq, lengths = get_test_val(batch_size, input_size, seq_size, max_len)
    attention = ConcatWordSeqAttention(input_size, seq_size)
    print(x)
    print(lengths)
    print(attention.score(x, seq, lengths))
    print(attention.forward(x, seq, lengths))


def test_mlp_attention(batch_size=3, input_size=4, seq_size=3, max_len=5, hidden=6):
    print("MLP Attention")
    x, seq, lengths = get_test_val(batch_size, input_size, seq_size, max_len)
    attention = MLPWordSeqAttention(input_size, seq_size, hidden)
    print(x)
    print(lengths)
    print(attention.score(x, seq, lengths))
    print(attention.forward(x, seq, lengths))


def test_mlp_dot_attention(batch_size=3, input_size=4, seq_size=3, max_len=5):
    print("DotMLP Attention")
    x, seq, lengths = get_test_val(batch_size, input_size, seq_size, max_len)
    attention = DotMLPWordSeqAttention(input_size, seq_size)
    print(x)
    print(lengths)
    print(attention.score(x, seq, lengths))
    print(attention.forward(x, seq, lengths))


def test_all():
    test_dot_attention()
    test_bilinear_attention()
    test_concat_attention()
    test_mlp_attention()
    test_mlp_dot_attention()


if __name__ == "__main__":
    test_all()
