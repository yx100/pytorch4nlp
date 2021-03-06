#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/8/26
from __future__ import absolute_import
import common
import time
import torch
import torch.nn as nn
from builtins import range
from pt4nlp import Constants, Dictionary
from sst_classifier import SSTClassifier
from sst_corpus import SSTCorpus
from argparse import ArgumentParser
import numpy


def get_filename(expname):
    if expname == 'sst2':
        dev_filename = "en_emotion_data/sst2_dev.csv"
        test_filename = "en_emotion_data/sst2_test.csv"
        train_filename = "en_emotion_data/sst2_train_sentence.csv"
    elif expname == 'sst5':
        dev_filename = "en_emotion_data/sst5_dev.csv"
        test_filename = "en_emotion_data/sst5_test.csv"
        train_filename = "en_emotion_data/sst5_train_sentence.csv"
    elif expname == 'sst2subtree':
        dev_filename = "en_emotion_data/sst2_dev.csv"
        test_filename = "en_emotion_data/sst2_test.csv"
        train_filename = "en_emotion_data/sst2_train_phrases.csv"
    elif expname == 'sst5subtree':
        dev_filename = "en_emotion_data/sst5_dev.csv"
        test_filename = "en_emotion_data/sst5_test.csv"
        train_filename = "en_emotion_data/sst5_train_phrases.csv"
    else:
        raise NotImplementedError
    return train_filename, dev_filename, test_filename


parser = ArgumentParser(description='SST Text Classifier')
# Train Option
parser.add_argument('-epoch', type=int, dest="epoch", default=50)
parser.add_argument('-batch', type=int, dest="batch", default=128)
parser.add_argument('-device', type=int, dest="device", default=0)
parser.add_argument('-seed', type=int, dest="seed", default=1993)
parser.add_argument('-exp', type=str, dest="exp_name", default="sst2",
                    choices=["sst2", "sst5", "sst2subtree", "sst5subtree", "imdb"])
parser.add_argument('-train-file', type=str, dest="train_file", default=None)
parser.add_argument('-dev-file', type=str, dest="dev_file", default=None)
parser.add_argument('-test-file', type=str, dest="test_file", default=None)


# Model Option
parser.add_argument('-encoder', type=str, dest="encoder", default="rnn", choices=["rnn", "cbow", "cnn"])
parser.add_argument('-word-vec-size', type=int, dest="word_vec_size", default=300)
parser.add_argument('-hidden-size', type=int, dest="hidden_size", default=168)
parser.add_argument('-num-layers', type=int, dest='num_layers', default=1)
parser.add_argument('-encoder-dropout', type=float, dest='encoder_dropout', default=0)
parser.add_argument('-dropout', type=float, dest='dropout', default=0.5)
parser.add_argument('-brnn', action='store_true', dest='brnn')
parser.add_argument('-bn', action='store_true', dest='bn')
parser.add_argument('-word-vectors', type=str, dest="word_vectors", default='en.emotion.glove.emb.bin')
parser.add_argument('-rnn-type', type=str, dest='rnn_type', default='LSTM')
parser.add_argument('-cnn-size', nargs='+', dest='cnn_size', default=[3])
parser.add_argument('-cnn-pooling', type=str, dest='cnn_pooling', default="max", choices=["max", "sum", "mean"])


# Optimizer Option
parser.add_argument('-word-normalize', action='store_true', dest="word_normalize")
parser.add_argument('-optimizer', type=str, dest="optimizer", default="Adadelta")
parser.add_argument('-lr', type=float, dest="lr", default=0.05)
parser.add_argument('-word-optimizer', type=str, dest="word_optimizer", default="SGD")
parser.add_argument('-word-lr', type=float, dest="word_lr", default=0.1)
parser.add_argument('-clip', type=float, default=9.0, dest="clip", help='clip grad by norm')
parser.add_argument('-regular', type=float, default=10e-4, dest="regular_weight", help='regular weight')

args = parser.parse_args()
if args.seed < 0:
    seed = time.time() % 10000
else:
    seed = args.seed
print("Random Seed: %d" % seed)
torch.manual_seed(int(seed))

if args.train_file is None and args.dev_file is None and args.test_file is None:
    train_file, dev_file, test_file = get_filename(args.exp_name)
else:
    train_file = args.train_file
    dev_file = args.dev_file
    test_file = args.test_file

usecuda = False
batch_size = args.batch

if args.device >= 0:
    usecuda = True

label_dictionary = Dictionary()
dictionary = Dictionary()
dictionary.add_specials([Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD],
                        [Constants.PAD, Constants.UNK, Constants.BOS, Constants.EOS])

SSTCorpus.add_word_to_dictionary(train_file, dictionary, label_dictionary=label_dictionary)
SSTCorpus.add_word_to_dictionary(dev_file, dictionary, label_dictionary=label_dictionary)
SSTCorpus.add_word_to_dictionary(test_file, dictionary, label_dictionary=label_dictionary)

train_data = SSTCorpus(train_file, dictionary, device=args.device,
                       batch_size=batch_size, label_dictionary=label_dictionary)
dev_data = SSTCorpus(dev_file, dictionary, device=args.device, volatile=True,
                     batch_size=batch_size, label_dictionary=label_dictionary, random=False)
test_data = SSTCorpus(test_file, dictionary, device=args.device, volatile=True,
                      batch_size=batch_size, label_dictionary=label_dictionary, random=False)

print("Train Size: %s" % len(train_data))
print("Dev   Size: %s" % len(dev_data))
print("Test  Size: %s" % len(test_data))

model = SSTClassifier(dictionary, opt=args, label_num=label_dictionary.size())
if args.word_vectors != "random":
    model.embedding.load_pretrained_vectors(args.word_vectors, normalize=args.word_normalize)
criterion = nn.CrossEntropyLoss()

if args.device >= 0:
    model.cuda(args.device)

param_wo_embedding = []
param_embedding = []

for name, param in model.named_parameters():
    if "emb_luts" in name:
        print("%s(%s)\t%s with %s" % (name, param.size(), args.word_optimizer, args.word_lr))
        param_embedding.append(param)
    else:
        print("%s(%s)\t%s with %s" % (name, param.size(), args.optimizer, args.lr))
        param_wo_embedding.append(param)

wo_word_opt = getattr(torch.optim, args.optimizer)(param_wo_embedding, lr=args.lr, weight_decay=args.regular_weight)
word_opt = getattr(torch.optim, args.word_optimizer)(param_embedding, lr=args.word_lr, weight_decay=args.regular_weight)


def eval_epoch(data):

    n_correct, n_total = 0, 0
    model.eval()
    for batch in data.next_batch():

        pred = model(batch)

        n_correct += (torch.max(pred, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        n_total += batch.batch_size

    return 100. * n_correct/n_total


def train_epoch(epoch_index):

    n_correct, n_total = 0, 0

    model.train()
    for batch in train_data.next_batch():
        wo_word_opt.zero_grad()
        word_opt.zero_grad()

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


result = list()
for i in range(args.epoch):
    start = time.time()
    train_acc = train_epoch(i)
    end = time.time()
    dev_acc = eval_epoch(dev_data)
    test_acc = eval_epoch(test_data)
    result.append((dev_acc, test_acc))
    print("iter %2d | %6.2f | %6.2f | %6.2f | %6.2f |" % (i, end - start, train_acc, dev_acc, test_acc))

result = torch.from_numpy(numpy.array(result))
max_dev_acc, max_index = torch.max(result[:, 0], 0)
print("Best Iter %d, Dev Acc: %s, Test Acc: %s" % (max_index[0], result[max_index[0], 0], result[max_index[0], 1]))
