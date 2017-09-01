#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/8/26
from __future__ import absolute_import
import codecs
import math
import torch
from torch.autograd import Variable
import Constants


class SSTCorpus():
    def __init__(self,
                 data_path,
                 dictionary,
                 volatile=False,
                 batch_size=64,
                 max_length=200,
                 cuda=False):
        self.dictionary = dictionary
        self.volatile = volatile
        self.cuda = cuda
        self.data = self.load_data_file(data_path=data_path, dictionary=self.dictionary)
        self.batch_size = batch_size
        self.max_length = max_length
        self.sort()

    @staticmethod
    def load_data_file(data_path, dictionary, split_symbol='\t', max_length=200):
        data = list()
        with codecs.open(data_path, 'r', 'utf8') as fin:
            for line in fin:
                label, _, text = line.strip().partition(split_symbol)
                text = dictionary.convert_to_index(text.split(), unk_word=Constants.UNK_WORD)
                if len(text) > max_length:
                    text = text[:max_length]
                data.append((torch.LongTensor(text), int(label), len(text)))
        return data

    @staticmethod
    def add_word_to_dictionary(data_path, dictionary, split_symbol='\t', label_dictionary=None):
        with codecs.open(data_path, 'r', 'utf8') as fin:
            for line in fin:
                label, _, text = line.strip().partition(split_symbol)
                if label_dictionary is not None:
                    label_dictionary.add(label)
                for word in text.split():
                    dictionary.add(word)

    def __len__(self):
        return len(self.data)

    def __sizeof__(self):
        return len(self.data)

    def sort(self):
        lengths = [(length, index) for index, (_, _, length) in enumerate(self.data)]
        lengths.sort()
        _, indexs = zip(*lengths)
        self.data = [self.data[i] for i in indexs]

    @staticmethod
    def _batchify(data):
        text, label, lengths = zip(*data)
        max_length = max(lengths)
        text = data[0][0].new(len(data), max_length).fill_(Constants.PAD)
        label = torch.LongTensor(label)

        for i in range(len(data)):
            length = data[i][0].size(0)
            text[i].narrow(0, 0, length).copy_(data[i][0])

        return torch.stack(text, 0), label, torch.LongTensor(lengths)

    def next_batch(self):
        num_batch = int(math.ceil(len(self.data) / float(self.batch_size)))
        random_index = torch.randperm(num_batch)
        for index, i in enumerate(random_index):
            start, end = i * self.batch_size, (i + 1) * self.batch_size
            _batch_size = len(self.data[start:end])
            text, label, lengths = self._batchify(self.data[start:end])

            if self.cuda:
                text = text.cuda()
                label = label.cuda()
                lengths = lengths.cuda()

            text = Variable(text, volatile=self.volatile)
            label = Variable(label, volatile=self.volatile)
            lengths = Variable(lengths, volatile=self.volatile)

            yield Batch(text, label, _batch_size, lengths)


class Batch(object):
    def __init__(self, text, label, batch_size, lengths):
        self.text = text
        self.label = label
        self.batch_size = batch_size
        self.lengths = lengths
