#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/9/28
import codecs
import common
import math
import torch
from torch.autograd import Variable
from pt4nlp import Constants, Dictionary


class Corpus():
    def __init__(self,
                 filename,
                 word_dictionary,
                 volatile=False,
                 batch_size=64,
                 max_length=300,
                 device=-1, ):

        self.word_dictionary = word_dictionary

        self.volatile = volatile
        self.cuda = device >= 0
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        self.data = self.load_data_file(filename,
                                        word_dict=word_dictionary,
                                        max_length=self.max_length, )

    @property
    def data_size(self):
        return len(self.data)

    @staticmethod
    def convert2longtensor(x):
        return torch.LongTensor(x)

    @staticmethod
    def _batchify(data):
        _, lengths = zip(*data)

        max_length = max(lengths)

        # (batch, max_len, feature size)
        text = data[0][0].new(len(data), max_length).fill_(Constants.PAD)

        for i in range(len(data)):
            length = data[i][0].size(0)
            text[i, :].narrow(0, 0, length).copy_(data[i][0])

        text = torch.stack(text, 0)
        lengths = Corpus.convert2longtensor(lengths)

        return text, lengths

    def next_batch(self):
        data = [self.data[index] for index in torch.randperm(len(self.data))]

        num_batch = int(math.ceil(len(data) / float(self.batch_size)))

        random_indexs = torch.randperm(num_batch)

        def convert2variable(x):
            if self.cuda:
                x = x.cuda(self.device)
            return Variable(x, volatile=self.volatile)

        for index, i in enumerate(random_indexs):
            start, end = i * self.batch_size, (i + 1) * self.batch_size
            _batch_size = len(data[start:end])
            text, lengths = self._batchify(data[start:end])

            text = convert2variable(text)
            lengths = convert2variable(lengths)

            yield Batch(text, _batch_size, lengths)

    @staticmethod
    def load_data_file(filename, word_dict, min_length=1, max_length=300, ):
        data = list()
        escape_count = 0
        sentence_count = 0

        with codecs.open(filename, 'r', 'utf8') as fin:
            for line in fin:
                words = line.strip().split()
                if len(words) < min_length:
                    escape_count += 1
                    continue
                if len(words) > max_length:
                    words = words[:max_length]
                else:
                    words += [Constants.PAD_WORD] * (max_length - len(words))

                sentence_count += 1
                sentence_length = len(words)

                token_ids = word_dict.convert_to_index(words, unk_word=Constants.UNK_WORD)
                _data = [Corpus.convert2longtensor(token_ids),
                         sentence_length]

                data.append(_data)

        print("Sentence: %d, Load Sentence: %s, Escape: %d." % (len(data), sentence_count, escape_count))

        return data

    @staticmethod
    def get_word_dictionary_from_file(filename, word_dict=None, lower=True):
        if word_dict is None:
            word_dict = Dictionary(lower)
            word_dict.add_specials([Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD],
                                   [Constants.PAD, Constants.UNK, Constants.BOS, Constants.EOS])

        with codecs.open(filename, 'r', 'utf8') as fin:
            for line in fin:
                words = line.strip().split()
                for token in words:
                    word_dict.add(token)

        return word_dict


class Batch(object):
    def __init__(self, text, batch_size, lengths):
        self.text = text
        self.batch_size = batch_size
        self.lengths = lengths


if __name__ == "__main__":
    word_d = Corpus.get_word_dictionary_from_file("trigger_ace_data/train/train.ids.dat")
    print(len(word_d))
