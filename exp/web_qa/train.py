# -*- coding:utf-8 -*- 
# Author: Roger
# Created by Roger on 2017/10/24
import common
from model import SeqLabelModel
from corpus import WebQACorpus
from argparse import Namespace

corpus = WebQACorpus("training.json.head")
args = Namespace(word_vec_size=5, hidden_size=5, num_layers=1, encoder_dropout=0, brnn=False, rnn_type='rnn')
model = SeqLabelModel(corpus.word_d, args, corpus.label_d.size())

for i in range(100):
    for batch in corpus.next_batch():
        loss = model.loss(batch)
        loss.backward()

        for _, param in model.named_parameters():
            param.data -= 0.01 * param.grad.data
