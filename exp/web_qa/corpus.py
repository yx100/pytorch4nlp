# -*- coding:utf-8 -*-
# Author: Roger
# Created by Roger on 2017/10/24
import codecs
import math

import numpy
import simplejson as json
import torch
from torch.autograd import Variable
import common
from pt4nlp import Dictionary, Constants


class WebQACorpus(object):

    def __init__(self, filename, batch_size=64, device=-1, volatile=False):
        self.word_d = self.load_word_dictionary(filename)
        self.label_d = self.load_label_dictionary()
        self.data = self.load_data_file(filename, self.word_d, self.label_d)
        self.batch_size = batch_size
        self.device = device
        self.cuda = self.device >= 0
        self.volatile = volatile

    @staticmethod
    def convert2longtensor(x):
        return torch.LongTensor(x)

    @staticmethod
    def _batchify(data):
        q_lens, e_lens, q_text_index, e_text_index, label_index, qe_feature_index, ee_feature_index = zip(*data)

        max_q_length = max(q_lens)
        # (batch, max_len, feature size)
        q_text = q_text_index[0].new(len(data), max_q_length).fill_(Constants.PAD)

        max_e_length = max(e_lens)
        e_text = e_text_index[0].new(len(data), max_e_length).fill_(Constants.PAD)
        qe_feature = qe_feature_index[0].new(len(data), max_e_length).fill_(Constants.PAD)
        ee_feature = ee_feature_index[0].new(len(data), max_e_length).fill_(Constants.PAD)
        label = label_index[0].new(len(data), max_e_length).fill_(5)

        for i in range(len(data)):
            length = q_text_index[i].size(0)
            q_text[i, :].narrow(0, 0, length).copy_(q_text_index[i])

            length = e_text_index[i].size(0)
            e_text[i, :].narrow(0, 0, length).copy_(e_text_index[i])
            qe_feature[i, :].narrow(0, 0, length).copy_(qe_feature_index[i])
            ee_feature[i, :].narrow(0, 0, length).copy_(ee_feature_index[i])
            label[i, :].narrow(0, 0, length).copy_(label_index[i])

        q_lens = WebQACorpus.convert2longtensor(q_lens)
        e_lens = WebQACorpus.convert2longtensor(e_lens)

        return q_text, e_text, label, q_lens, e_lens, qe_feature, ee_feature

    def next_batch(self):
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
            q_text, e_text, label, q_lens, e_lens, qe_feature, ee_feature = self._batchify(data[start:end])

            q_text = convert2variable(q_text)
            e_text = convert2variable(e_text)
            label = convert2variable(label)
            q_lens = convert2variable(q_lens)
            e_lens = convert2variable(e_lens)
            qe_feature = convert2variable(qe_feature)
            ee_feature = convert2variable(ee_feature)

            yield Batch(q_text, e_text, label, q_lens, e_lens, qe_feature, ee_feature, _batch_size)

    @staticmethod
    def load_data_file(filename, word_dict, label_dict):
        data_list = list()
        with codecs.open(filename, 'r') as fin:
            for line in fin:
                data = json.loads(line)
                q_text = data["question_tokens"]
                q_text_index = WebQACorpus.convert2longtensor(word_dict.convert_to_index(q_text, Constants.UNK_WORD))
                for evidence in data["evidences"]:
                    e_text = evidence["evidence_tokens"]
                    e_text_index = WebQACorpus.convert2longtensor(word_dict.convert_to_index(e_text,
                                                                                             Constants.UNK_WORD))
                    golden = evidence["golden_labels"]
                    label = WebQACorpus.convert2longtensor(label_dict.convert_to_index(golden, "*END"))
                    qe_feature = WebQACorpus.convert2longtensor(evidence["q-e.comm_features"])
                    ee_feature_list = evidence["eecom_features_list"]
                    ee_feature = list()
                    for i in range(len(e_text)):
                        ee = sum([ee_feature_list[j]['e-e.comm_features'][i] for j in range(len(ee_feature_list))])
                        ee_feature.append(ee)
                    ee_feature = WebQACorpus.convert2longtensor(ee_feature)
                    d = [len(q_text), len(e_text), q_text_index, e_text_index, label, qe_feature, ee_feature]
                    data_list.append(d)
                    if len(data_list) % 5000 == 0:
                        print(len(data_list))
        return data_list

    @staticmethod
    def load_word_dictionary(filename, word_dict=None):
        if word_dict is None:
            word_dict = Dictionary()
            word_dict.add_specials([Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD],
                                   [Constants.PAD, Constants.UNK, Constants.BOS, Constants.EOS])
        with codecs.open(filename, 'r') as fin:
            for line in fin:
                data = json.loads(line)
                q_text = data["question_tokens"]
                for token in q_text:
                    word_dict.add(token)
                for evidence in data["evidences"]:
                    e_text = evidence["evidence_tokens"]
                    for token in e_text:
                        word_dict.add(token)
        return word_dict

    @staticmethod
    def load_label_dictionary():
        label_dict = Dictionary()
        label_dict.add_specials(["*START", "o1", "o2", "b", "i", "*END"],
                                [0, 1, 2, 3, 4, 5])
        return label_dict


class Batch(object):
    def __init__(self, q_text, e_text, label, q_lens, e_lens, qe_feature, ee_feature, batch_size):
        self.q_text = q_text
        self.e_text = e_text
        self.label = label
        self.q_lens = q_lens
        self.e_lens = e_lens
        self.qe_feature = qe_feature
        self.ee_feature = ee_feature
        self.batch_size = batch_size
        self.pred = None


if __name__ == "__main__":
    corpus = WebQACorpus("training.json.head")
    for data in corpus.next_batch():
        print(data.label)
