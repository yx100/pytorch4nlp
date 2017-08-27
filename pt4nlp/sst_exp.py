#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/8/26
from __future__ import absolute_import
import torch
import torch.optim as O
import torch.nn as nn
from builtins import range
from sst_classifier import SSTClassifier
from dictionary import Dictionary
import Constants
from sst_corpus import SSTCorpus
from argparse import ArgumentParser


parser = ArgumentParser(description='SST Text Classifier')
# Train Option
parser.add_argument('-epoch', type=int, dest="epoch", default=50)
parser.add_argument('-batch', type=int, dest="batch", default=128)
parser.add_argument('-device', type=int, dest="device", default=0)

# Model Option
parser.add_argument('-word-vec-size', type=int, dest="word_vec_size", default=300)
parser.add_argument('-hidden-size', type=int, dest="hidden_size",default=168)
parser.add_argument('-no-bidirection', action='store_false', dest='brnn')
parser.add_argument('-word-vectors', type=str, default='glove.42B')
parser.add_argument('-rnn-type', type=str, dest='rnn_type', default='LSTM')
parser.add_argument('-dropout', type=float, dest='dropout', default=0.2)

# Optimizer Option
parser.add_argument('-optimizer', type=str, dest="optimizer", default="Adadelta")
parser.add_argument('-lr', type=float, dest="lr", default=1)

args = parser.parse_args()

usecuda = False
batch_size = args.batch

if args.device > 0:
    usecuda = True

dictionary = Dictionary()
dictionary.add_specials([Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD],
                        [Constants.PAD, Constants.UNK, Constants.BOS, Constants.EOS])
SSTCorpus.add_word_to_dictionary("en_emotion_data/sst5_train_phrases.csv", dictionary)
train_data = SSTCorpus("en_emotion_data/sst5_train_phrases.csv", dictionary, cuda=usecuda)
dev_data = SSTCorpus("en_emotion_data/sst5_dev.csv", dictionary, cuda=usecuda, volatile=True)
test_data = SSTCorpus("en_emotion_data/sst5_test.csv", dictionary, cuda=usecuda, volatile=True)

model = SSTClassifier(dictionary)
model.embedding.load_pretrained_vectors("en.emotion.glove.emb.bin")
criterion = nn.CrossEntropyLoss()
opt = getattr(torch.optim, args.optimizer)(model.parameters, lr=args.lr)

if args.device > 0:
    model.cuda()

def eval_epoch(data):
    n_correct, n_total = 0, 0
    for batch in data.next_batch(batch_size):
        model.eval()

        pred = model(batch)

        n_correct += (torch.max(pred, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        n_total += batch.batch_size

    return 100. * n_correct/n_total


def train_epoch(epoch_index):
    n_correct, n_total = 0, 0

    for batch in train_data.next_batch(batch_size):
        model.train(); opt.zero_grad()

        pred = model(batch)

        n_correct += (torch.max(pred, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        n_total += batch.batch_size

        # calculate loss of the network output with respect to training labels
        loss = criterion(pred, batch.label)

        # backpropagate and update optimizer learning rate
        loss.backward()

        opt.step()

    return 100. * n_correct/n_total


for i in range(50):
    train_acc = train_epoch(i)
    dev_acc = eval_epoch(dev_data)
    test_acc = eval_epoch(test_data)
    print("iter %2d | %6.2f | %6.2f | %6.2f |" % (i, train_acc, dev_acc, test_acc))
