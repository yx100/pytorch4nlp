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
                 gold_file, ids_file, sents_file,
                 word_dictionary, pos_dictionary, label_dictionary,
                 volatile=False,
                 batch_size=64,
                 max_length=200,
                 device=-1,
                 lexi_window=1,
                 random=True,
                 neg_ratio=14,
                 fix_neg=False,
                 train=True):

        self.word_dictionary = word_dictionary
        self.pos_dictionary = pos_dictionary
        self.label_dictionary = label_dictionary

        self.volatile = volatile
        self.cuda = device >= 0
        self.device = device
        self.lexi_win = max(lexi_window, 0)
        self.max_length = max_length
        self.train = train
        self.batch_size = batch_size
        self.random = random
        self.neg_ratio = neg_ratio
        self.fix_neg = fix_neg

        data = self.load_data_file(gold_file, ids_file,
                                   sents_file,
                                   label_dict=label_dictionary,
                                   word_dict=word_dictionary,
                                   pos_dict=pos_dictionary,
                                   lexi_window=self.lexi_win,
                                   max_length=self.max_length,
                                   train=self.train)

        self.event_data, self.non_event_data, self.ids_data, self.gold_data = data
        self.data = self.sample_data()

    @property
    def event_data_size(self):
        return len(self.event_data)

    @property
    def nonevent_data_size(self):
        return len(self.non_event_data)

    @staticmethod
    def convert2longtensor(x):
        return torch.LongTensor(x)

    @staticmethod
    def _batchify(data):
        _, label, lengths, lexi, position, ident = zip(*data)

        max_length = max(lengths)
        # (batch, max_len, feature size)
        text = data[0][0].new(len(data), max_length, data[0][0].size(1)).fill_(Constants.PAD)
        # (batch)
        label = EECorpus.convert2longtensor(label)

        for i in range(len(data)):
            length = data[i][0].size(0)
            text[i, :].narrow(0, 0, length).copy_(data[i][0])

        text = torch.stack(text, 0)
        lengths = EECorpus.convert2longtensor(lengths)
        lexi = torch.stack(lexi, 0)
        position = EECorpus.convert2longtensor(position)

        return text, label, lengths, lexi, position, ident

    def sample_data(self):
        if self.neg_ratio > 0:
            neg_index = torch.randperm(self.nonevent_data_size)[:int(self.event_data_size * self.neg_ratio)]
            neg_data = [self.non_event_data[index] for index in neg_index]
            return self.event_data + neg_data
        else:
            return self.event_data + self.non_event_data

    def next_batch(self):
        if self.neg_ratio == 0:
            # evaluate
            data = self.event_data + self.non_event_data
            num_batch = int(math.ceil((len(data) / float(self.batch_size))))
            random_indexs = range(num_batch)
        else:
            if not self.fix_neg:
                self.data = self.sample_data()
            num_batch = int(math.ceil(len(self.data) / float(self.batch_size)))

            data = [self.data[index] for index in torch.randperm(len(self.data))]

            random_indexs = torch.randperm(num_batch)

        def convert2variable(x):
            if self.cuda:
                x = x.cuda(self.device)
            return Variable(x, volatile=self.volatile)

        for index, i in enumerate(random_indexs):

            start, end = i * self.batch_size, (i + 1) * self.batch_size
            _batch_size = len(data[start:end])
            text, label, lengths, lexi, position, ident = self._batchify(data[start:end])

            text = convert2variable(text)
            label = convert2variable(label)
            lengths = convert2variable(lengths)
            lexi = convert2variable(lexi)
            position = convert2variable(position)

            yield Batch(text, label, _batch_size, lengths, lexi, position, ident)

    @staticmethod
    def load_data_file(gold_file, ids_file, sents_file,
                       label_dict, word_dict, pos_dict,
                       min_length=2, max_length=200, lexi_window=1, train=False):
        pos_data = list()
        neg_data = list()

        ids_data, posi_sent_set = EECorpus.load_ids_file(ids_file)
        gold_data = EECorpus.load_gold_file(gold_file)
        sents_data = EECorpus.load_sents_file(sents_file)

        for (key, sentence) in viewitems(sents_data):

            docid, sentid = key
            sentence_length = len(sentence)

            # Train: Only keep the sentence which has trigger
            # Train: Only keep the sentence which smaller than max length
            # Train: Only keep the sentence which bigger than min length
            if train:
                if key not in posi_sent_set:
                    continue
                if sentence_length > max_length:
                    continue
                if sentence_length < min_length:
                    continue

            sent_token_ids = word_dict.convert_to_index(sentence, unk_word=Constants.UNK_WORD)
            for tokenid, token in enumerate(sentence):

                type_name = ids_data[docid, sentid, tokenid]['type']
                token_from_ids = ids_data[docid, sentid, tokenid]['token']

                # Lexi Info
                lexi = (2 * lexi_window + 1) * [Constants.BOS_WORD]
                for i in range(-lexi_window, lexi_window + 1):
                    if tokenid + i < 0 or tokenid + i >= sentence_length:
                        continue
                    lexi[i] = sentence[tokenid + i]
                lexi_ids = word_dict.convert_to_index(lexi, unk_word=Constants.UNK_WORD)

                # Label Info
                label = label_dict.lookup(type_name, default=label_dict.lookup(common.OTHER_NAME))

                # Position Info
                relative_position = [pos_dict.convert_to_index([d], unk_word=Constants.UNK_WORD)[0]
                                     for d in range(-tokenid, sentence_length - tokenid)]

                # Check Whether same token sents and ids
                if token != token_from_ids:
                    print("[WARNING]")
                    print(token, token_from_ids)
                    print(docid, sentid, tokenid)
                    break
                assert token == token_from_ids

                # token + relative position, label, length,  lexi feature, position to split, ident info
                _data = [EECorpus.convert2longtensor([sent_token_ids, relative_position]).t(),
                         label,
                         sentence_length,
                         EECorpus.convert2longtensor(lexi_ids),
                         tokenid,
                         (docid, sentid, tokenid)]

                if type_name != common.OTHER_NAME:
                    pos_data.append(_data)
                else:
                    neg_data.append(_data)

        return pos_data, neg_data, ids_data, gold_data

    @staticmethod
    def load_ids_file(filename):
        """
        :param filename: ids file name
        :return: (docid, senid, tokenid) -> start, length, type, token
        """
        ids_data = dict()
        pos_sent_set = set()
        with codecs.open(filename, 'r', 'utf8') as fin:
            for line in fin:
                att = line.strip().split('\t')
                key = (att[0], int(att[1]), int(att[2]))
                ids_data[key] = {
                    'start': int(att[3]),
                    'length': int(att[4]),
                    'type': att[5].split(';')[0],
                    'token': att[6],
                }
                if att[5].split(';')[0] != common.OTHER_NAME:
                    pos_sent_set.add((att[0], int(att[1])))
        return ids_data, pos_sent_set

    @staticmethod
    def load_gold_file(filename):
        """
        :param filename: gold file name
        :return: docid, start, length, type
        """
        gold_data = list()
        with codecs.open(filename, 'r', 'utf8') as fin:
            for line in fin:
                att = line.strip().split('\t')
                if ';' in att[4]:
                    for label in att[4].split(';'):
                        gold_data += [(att[0], int(att[1]), int(att[2]), label)]
                else:
                    gold_data += [(att[0], int(att[1]), int(att[2]), att[4])]
        return gold_data

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
                sents_data[(att[0], int(att[1]))] = att[2].split(' ')
        return sents_data

    @staticmethod
    def load_label_dictionary(label2id_file, label_dict=None):
        if label_dict is None:
            label_dict = Dictionary(lower=False)

        with codecs.open(label2id_file, 'r', 'utf8') as fin:
            for line in fin:
                label, index = line.strip().split()
                label_dict.add_special(key=label, idx=int(index))

        return label_dict

    @staticmethod
    def get_position_dictionary(max_length=200, position_dict=None):
        if position_dict is None:
            position_dict = Dictionary()
            position_dict.add_specials([Constants.UNK_WORD], [Constants.UNK])

        for number in range(-max_length, max_length + 1):
            position_dict.add_special(number)

        return position_dict

    @staticmethod
    def get_word_dictionary_from_ids_file(ids_file, word_dict=None):
        if word_dict is None:
            word_dict = Dictionary()
            word_dict.add_specials([Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD],
                                   [Constants.PAD, Constants.UNK, Constants.BOS, Constants.EOS])

        _, pos_set = EECorpus.load_ids_file(ids_file)

        with codecs.open(ids_file, 'r', 'utf8') as fin:
            for line in fin:
                docid, senid, tokenid = line.split('\t')[:3]
                if (docid, int(senid), int(tokenid)) not in pos_set:
                    continue
                token = line.strip().split('\t')[6]
                word_dict.add(token)

        return word_dict

    def batch2pred(self, batch):
        pred_list = list()
        assert batch.pred is not None
        for ident, pred in zip(batch.ident, batch.pred):
            docid, sentid, tokenid = ident
            ids_dict = self.ids_data[ident]
            pred_list += [(docid, ids_dict['start'], ids_dict['length'],
                           self.label_dictionary.index2word[pred]
                           )]
        return pred_list


class Batch(object):
    def __init__(self, text, label, batch_size, lengths, lexi, position, ident):
        self.text = text
        self.label = label
        self.batch_size = batch_size
        self.lengths = lengths
        self.lexi = lexi
        self.position = position
        self.ident = ident
        self.pred = None


if __name__ == "__main__":
    label_d = EECorpus.load_label_dictionary("trigger_ace_data/label2id.dat")
    print(len(label_d))
    posit_d = EECorpus.get_position_dictionary(200)
    print(len(posit_d))
    word_d = EECorpus.get_word_dictionary_from_ids_file("trigger_ace_data/train/train.ids.dat")
    print(len(word_d))
