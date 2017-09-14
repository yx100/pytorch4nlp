#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/9/14
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/8/26
from future.utils import viewitems
import common
import codecs
import math
import torch
from torch.autograd import Variable
from pt4nlp import Constants, Dictionary


class EECorpus():
    def __init__(self,
                 gold_file,
                 ids_file,
                 sents_file,
                 word_dictionary,
                 pos_dictionary,
                 label_dictionary,
                 volatile=False,
                 batch_size=64,
                 max_length=200,
                 device=-1,
                 lexi_window=1,
                 random=True):
        self.word_dictionary = word_dictionary
        self.pos_dictionary = pos_dictionary
        self.label_dictionary = label_dictionary
        self.volatile = volatile
        self.cuda = device >= 0
        self.device = device
        self.lexi_win = lexi_window
        self.event_data, self.non_event_data = self.load_data_file(gold_file, ids_file, sents_file,
                                                                   label_dict=label_dictionary,
                                                                   word_dict=word_dictionary, pos_dict=pos_dictionary,
                                                                   lexi_window=self.lexi_win)
        self.batch_size = batch_size
        self.max_length = max_length
        self.random = random
        self.neg_ratio = 14

    @property
    def event_data_size(self):
        return len(self.event_data)

    @property
    def nonevent_data_size(self):
        return len(self.non_event_data)

    @staticmethod
    def _batchify(data):
        _, label, lengths, lexi, position = zip(*data)
        max_length = max(lengths)
        text = data[0][0].new(len(data), max_length, data[0][0].size(1)).fill_(Constants.PAD)
        label = torch.LongTensor(label)

        for i in range(len(data)):
            length = data[i][0].size(0)
            text[i, :].narrow(0, 0, length).copy_(data[i][0])

        return torch.stack(text, 0), label, torch.LongTensor(lengths), torch.stack(lexi, 0), torch.LongTensor(position)

    def next_batch(self):
        neg_index = torch.randperm(self.nonevent_data_size)[:int(self.event_data_size) * self.neg_ratio]
        neg_data = [self.non_event_data[index] for index in neg_index]
        num_batch = int(math.ceil((self.event_data_size + len(neg_data)) / float(self.batch_size)))

        data = self.event_data + neg_data
        data = [data[index] for index in torch.randperm(len(data))]

        random_indexs = torch.randperm(num_batch)

        for index, i in enumerate(random_indexs):
            start, end = i * self.batch_size, (i + 1) * self.batch_size
            _batch_size = len(data[start:end])
            text, label, lengths, lexi, position = self._batchify(data[start:end])

            if self.cuda:
                text = text.cuda(self.device)
                label = label.cuda(self.device)
                lengths = lengths.cuda(self.device)
                lexi = lexi.cuda(self.device)
                position = position.cuda(self.device)

            text = Variable(text, volatile=self.volatile)
            label = Variable(label, volatile=self.volatile)
            lengths = Variable(lengths, volatile=self.volatile)
            lexi = Variable(lexi, volatile=self.volatile)
            position = Variable(position, volatile=self.volatile)

            yield Batch(text, label, _batch_size, lengths, lexi, position)

    @staticmethod
    def load_data_file(gold_file, ids_file, sents_file, label_dict, word_dict, pos_dict, max_length=200, lexi_window=1):
        pos_data = list()
        neg_data = list()
        ids_data = EECorpus.load_ids_file(ids_file)
        gold_data = EECorpus.load_gold_file(gold_file)
        sents_data = EECorpus.load_sents_file(sents_file)
        for (key, sentence) in viewitems(sents_data):
            docid, sentid = key
            sentence_length = len(sentence)
            sent_token_ids = word_dict.convert_to_index(sentence, unk_word=Constants.UNK_WORD)
            for tokenid, token in enumerate(sentence):

                lexi = (2 * lexi_window + 1) * [Constants.BOS_WORD]
                for i in range(-lexi_window, lexi_window + 1):
                    if tokenid + i < 0 or tokenid + i >= sentence_length:
                        continue
                    lexi[i] = sentence[tokenid + i]
                lexi_ids = word_dict.convert_to_index(lexi, unk_word=Constants.UNK_WORD)

                label = label_dict.lookup(ids_data[docid, sentid, tokenid]['type'])
                relative_position = [pos_dict.convert_to_index([d], unk_word=Constants.UNK_WORD)[0]
                                     for d in range(-tokenid, sentence_length - tokenid)]

                assert token == ids_data[docid, sentid, tokenid]['token']
                # lexi feature, token + relative position, label, length, position to split
                _data = [torch.LongTensor([sent_token_ids, relative_position]).t(),
                         label,
                         sentence_length,
                         torch.LongTensor(lexi_ids),
                         tokenid]
                if ids_data[docid, sentid, tokenid]['type'] != "other":
                    pos_data.append(_data)
                else:
                    neg_data.append(_data)

        return pos_data, neg_data

    @staticmethod
    def load_ids_file(filename):
        """
        :param filename: ids file name
        :return: (docid, senid, tokenid) -> start, length, type, token
        """
        gold_data = dict()
        with codecs.open(filename, 'r', 'utf8') as fin:
            for line in fin:
                att = line.strip().split('\t')
                key = (att[0], int(att[1]), int(att[2]))
                gold_data[key] = {
                    'start': att[3],
                    'length': att[4],
                    'type': att[5],
                    'token': att[6],
                }
        return gold_data

    @staticmethod
    def load_gold_file(filename):
        """
        :param filename: gold file name
        :return: docid, start, length, token, type
        """
        ids_data = list()
        with codecs.open(filename, 'r', 'utf8') as fin:
            for line in fin:
                att = line.strip().split('\t')
                ids_data.append({'docid': att[0],
                                 'start': att[1],
                                 'length': att[2],
                                 'token': att[3],
                                 'type': att[4],
                                 })
        return ids_data

    @staticmethod
    def load_sents_file(filename):
        """
        :param filename: sents file name
        :return: (docid, sentid) -> tokens
        """
        sents_data = dict()
        with codecs.open(filename, 'r', 'utf8') as fin:
            for line in fin:
                att = line.strip().split('\t')
                sents_data[(att[0], int(att[1]))] = att[2].split()
        return sents_data

    @staticmethod
    def load_label_dictionary(label2id_file, label_dictionary=None):
        if label_dictionary is None:
            label_dictionary = Dictionary()
        with codecs.open(label2id_file, 'r', 'utf8') as fin:
            for line in fin:
                label, index = line.strip().split('\t')
                label_dictionary.add_special(key=label, idx=int(index))
        return label_dictionary

    @staticmethod
    def get_position_dictionary(max_length=200, position_dictionary=None):
        if position_dictionary is None:
            position_dictionary = Dictionary()
            position_dictionary.add_specials([Constants.UNK_WORD], [Constants.UNK])
        for number in range(-max_length, max_length + 1):
            position_dictionary.add_special(number)
        return position_dictionary

    @staticmethod
    def get_word_dictionary_from_ids_file(ids_file, word_dictionary=None):
        if word_dictionary is None:
            word_dictionary = Dictionary()
            word_dictionary.add_specials([Constants.PAD_WORD, Constants.UNK_WORD,
                                          Constants.BOS_WORD, Constants.EOS_WORD],
                                         [Constants.PAD, Constants.UNK, Constants.BOS, Constants.EOS])
        with codecs.open(ids_file, 'r', 'utf8') as fin:
            for line in fin:
                token = line.strip().split('\t')[-1]
                word_dictionary.add(token)
        return word_dictionary


class Batch(object):
    def __init__(self, text, label, batch_size, lengths, lexi, position):
        self.text = text
        self.label = label
        self.batch_size = batch_size
        self.lengths = lengths
        self.lexi = lexi
        self.position = position


if __name__ == "__main__":
    label_d = EECorpus.load_label_dictionary("trigger_ace_data/label2id.dat")
    print(len(label_d))
    posit_d = EECorpus.get_position_dictionary(200)
    print(len(posit_d))
    word_d = EECorpus.get_word_dictionary_from_ids_file("trigger_ace_data/train/train.ids.dat")
    print(len(word_d))
