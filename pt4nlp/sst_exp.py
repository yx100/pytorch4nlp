#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/8/26
from __future__ import absolute_import
import time
import torch
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
parser.add_argument('-seed', type=int, dest="seed", default=1993)

# Model Option
parser.add_argument('-word-vec-size', type=int, dest="word_vec_size", default=300)
parser.add_argument('-hidden-size', type=int, dest="hidden_size",default=168)
parser.add_argument('-num-layers', type=int, dest='num_layers', default=1)
parser.add_argument('-dropout', type=float, dest='dropout', default=0.5)
parser.add_argument('-no-bidirection', action='store_false', dest='brnn')
parser.add_argument('-word-vectors', type=str, dest="word_vectors", default='en.emotion.glove.emb.bin')
parser.add_argument('-rnn-type', type=str, dest='rnn_type', default='LSTM')

# Optimizer Option
parser.add_argument('-optimizer', type=str, dest="optimizer", default="Adadelta")
parser.add_argument('-lr', type=float, dest="lr", default=1.0)
parser.add_argument('-word-optimizer', type=str, dest="word_optimizer", default="SGD")
parser.add_argument('-word-lr', type=float, dest="word_lr", default=0.1)
parser.add_argument('-clip', type=float, default=9.0, dest="clip", help='clip grad by norm')

args = parser.parse_args()
torch.manual_seed(args.seed)

usecuda = False
batch_size = args.batch

if args.device >= 0:
    usecuda = True

label_dictionary = Dictionary()
dictionary = Dictionary()
dictionary.add_specials([Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD],
                        [Constants.PAD, Constants.UNK, Constants.BOS, Constants.EOS])
SSTCorpus.add_word_to_dictionary("en_emotion_data/sst2_train_sentence.csv", dictionary,
                                 label_dictionary=label_dictionary)
train_data = SSTCorpus("en_emotion_data/sst2_train_sentence.csv", dictionary, cuda=usecuda)
dev_data = SSTCorpus("en_emotion_data/sst2_dev.csv", dictionary, cuda=usecuda, volatile=True)
test_data = SSTCorpus("en_emotion_data/sst2_test.csv", dictionary, cuda=usecuda, volatile=True)

model = SSTClassifier(dictionary, opt=args, label_num=label_dictionary.size())
model.embedding.load_pretrained_vectors(args.word_vectors)
criterion = nn.CrossEntropyLoss()


param_wo_embedding = []
param_embedding = []

for name, param in model.named_parameters():
    if "emb_luts" in name:
        print("%s(%s)\t%s with %s" % (name, param.size(), args.word_optimizer, args.word_lr))
        param_embedding.append(param)
    else:
        print("%s(%s)\t%s with %s" % (name, param.size(), args.optimizer, args.lr))
        param_wo_embedding.append(param)

wo_word_opt = getattr(torch.optim, args.optimizer)(param_wo_embedding, lr=args.lr, weight_decay=10e-4)
word_opt = getattr(torch.optim, args.word_optimizer)(param_embedding, lr=args.word_lr, weight_decay=10e-4)

for name, param in wo_word_opt.param_groups:
    print("%s(%s)\t%s with %s" % (name, param.size(), args.optimizer, args.lr))
for name, param in word_opt.param_groups:
    print("%s(%s)\t%s with %s" % (name, param.size(), args.word_optimizer, args.word_lr))

if args.device >= 0:
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
        model.train(); wo_word_opt.zero_grad(); word_opt.zero_grad()

        pred = model(batch)

        n_correct += (torch.max(pred, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        n_total += batch.batch_size

        # calculate loss of the network output with respect to training labels
        loss = criterion(pred, batch.label)

        # backpropagate and update optimizer learning rate
        loss.backward()

        if args.clip > 0:
            nn.utils.clip_grad_norm(model.parameters(), args.clip)

        wo_word_opt.step()
        word_opt.step()

    return 100. * n_correct/n_total


for i in range(50):
    start = time.time()
    train_acc = train_epoch(i)
    end = time.time()
    dev_acc = eval_epoch(dev_data)
    test_acc = eval_epoch(test_data)
    print("iter %2d | %6.2f | %6.2f | %6.2f | %6.2f |" % (i, end - start, train_acc, dev_acc, test_acc))
