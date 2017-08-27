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
                 cuda=False):
        self.dictionary = dictionary
        self.volatile = volatile
        self.cuda = cuda
        self.data = self.load_data_file(data_path=data_path, dictionary=self.dictionary)
        self.sort()

    @staticmethod
    def load_data_file(data_path, dictionary, split_symbol='\t'):
        data = list()
        with codecs.open(data_path, 'r', 'utf8') as fin:
            for line in fin:
                label, _, text = line.strip().partition(split_symbol)
                text = dictionary.convert_to_index(text.split(), unk_word=Constants.UNK_WORD)
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
        text, label, length = zip(*data)
        max_length = max(length)
        text = data[0][0].new(len(data), max_length).fill_(Constants.PAD)
        label = torch.LongTensor(label)

        for i in range(len(data)):
            length = data[i][0].size(0)
            text[i].narrow(0, 0, length).copy_(data[i][0])

        return torch.stack(text, 0), label

    def next_batch(self, batch_size):
        num_batch = int(math.ceil(len(self.data) / float(batch_size)))
        random_index = torch.randperm(num_batch)
        for index, i in enumerate(random_index):
            start, end = i * batch_size, (i + 1) * batch_size
            batch_size = len(self.data[start:end])
            text, label = self._batchify(self.data[start:end])

            if self.cuda:
                text = text.cuda()
                label = label.cuda()

            text = Variable(text, volatile=self.volatile)
            label = Variable(label, volatile=self.volatile)

            yield Batch(text, label, batch_size)


class Batch(object):
    def __init__(self, text, label, batch_size):
        self.text = text
        self.label = label
        self.batch_size = batch_size
