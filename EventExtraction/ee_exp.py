#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/9/14
from __future__ import absolute_import
import common
import time
import torch
import torch.nn as nn
from builtins import range

from dmcnn import DynamicMultiPoolingCNN
from evaluate import evalute
from event_corpus import EECorpus
from argparse import ArgumentParser
import numpy

parser = ArgumentParser(description='DMCNN Text Classifier')
# Train Option
parser.add_argument('-epoch', type=int, dest="epoch", default=50)
parser.add_argument('-batch', type=int, dest="batch", default=128)
parser.add_argument('-device', type=int, dest="device", default=0)
parser.add_argument('-seed', type=int, dest="seed", default=-1)
parser.add_argument('-exp', type=str, dest="exp_name", default="sst2",
                    choices=["sst2", "sst5", "sst2subtree", "sst5subtree", "imdb"])
parser.add_argument('-train-file', type=str, dest="train_file", default=None)
parser.add_argument('-dev-file', type=str, dest="dev_file", default=None)
parser.add_argument('-test-file', type=str, dest="test_file", default=None)
parser.add_argument('-neg-ratio', type=float, dest="neg_ratio", default=14.)
parser.add_argument('-fix-neg', action='store_true', dest='fix_neg')

# Model Option
parser.add_argument('-encoder', type=str, dest="encoder", default="rnn", choices=["rnn", "cbow", "cnn"])
parser.add_argument('-word-vec-size', type=int, dest="word_vec_size", default=300)
parser.add_argument('-posi-vec-size', type=int, dest="posi_vec_size", default=5)
parser.add_argument('-hidden-size', type=int, dest="hidden_size", default=168)
parser.add_argument('-encoder-dropout', type=float, dest='encoder_dropout', default=0)
parser.add_argument('-dropout', type=float, dest='dropout', default=0.5)
parser.add_argument('-bn', action='store_true', dest='bn')
parser.add_argument('-word-vectors', type=str, dest="word_vectors", default='word_word2vec.bin')
parser.add_argument('-cnn-size', nargs='+', dest='cnn_size', default=[3])
parser.add_argument('-cnn-pooling', type=str, dest='cnn_pooling', default="max", choices=["max", "sum", "mean"])
parser.add_argument('-lexi-window', type=int, dest='lexi_window', default=1)

# Optimizer Option
parser.add_argument('-word-normalize', action='store_true', dest="word_normalize")
parser.add_argument('-optimizer', type=str, dest="optimizer", default="Adadelta")
parser.add_argument('-lr', type=float, dest="lr", default=0.05)
parser.add_argument('-word-optimizer', type=str, dest="word_optimizer", default="SGD")
parser.add_argument('-word-lr', type=float, dest="word_lr", default=0.1)
parser.add_argument('-clip', type=float, default=9.0, dest="clip", help='clip grad by norm')
parser.add_argument('-regular', type=float, default=0, dest="regular_weight", help='regular weight')

args = parser.parse_args()
if args.seed < 0:
    seed = time.time() % 10000
else:
    seed = args.seed
print("Random Seed: %d" % seed)
torch.manual_seed(int(seed))

usecuda = False
batch_size = args.batch

if args.device >= 0:
    usecuda = True

label_d = EECorpus.load_label_dictionary("trigger_ace_data/label2id.dat")
print("Label Size: %s" % len(label_d))
posit_d = EECorpus.get_position_dictionary(200)
print("Position Vocab Size: %s" % len(posit_d))
word_d = EECorpus.get_word_dictionary_from_ids_file("trigger_ace_data/train/train.ids.dat")
word_d.cut_by_count(2)

train_data = EECorpus("trigger_ace_data/train/train.golden.dat",
                      "trigger_ace_data/train/train.ids.dat",
                      "trigger_ace_data/train/train.sents.dat",
                      word_d, posit_d, label_d, lexi_window=1,
                      device=args.device, neg_ratio=args.neg_ratio, fix_neg=args.fix_neg)
train_eval_data = EECorpus("trigger_ace_data/train/train.golden.dat",
                           "trigger_ace_data/train/train.ids.dat",
                           "trigger_ace_data/train/train.sents.dat",
                           word_d, posit_d, label_d, lexi_window=1,
                           device=args.device, neg_ratio=0)
dev_data = EECorpus("trigger_ace_data/dev/dev.golden.dat",
                    "trigger_ace_data/dev/dev.ids.dat",
                    "trigger_ace_data/dev/dev.sents.dat",
                    word_d, posit_d, label_d, lexi_window=1,
                    device=args.device, neg_ratio=0)
test_data = EECorpus("trigger_ace_data/test/test.golden.dat",
                     "trigger_ace_data/test/test.ids.dat",
                     "trigger_ace_data/test/test.sents.dat",
                     word_d, posit_d, label_d, lexi_window=1,
                     device=args.device, neg_ratio=0)

model = DynamicMultiPoolingCNN(word_d, opt=args, label_num=label_d.size(), position_dict=posit_d, lexi_window=1)
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
    pred_results = list()
    model.eval()
    for batch in data.next_batch():
        pred = model(batch)
        pred_label = torch.max(pred, 1)[1].data
        batch.pred = pred_label
        batch_pred = data.batch2pred(batch)
        pred_results += batch_pred
    p, r, f = evalute(data.gold_data, pred_results)
    return f


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

    return 100. * n_correct / n_total


result = list()
for i in range(args.epoch):
    start = time.time()
    train_acc = train_epoch(i)
    end = time.time()
    train_f1 = eval_epoch(train_eval_data)
    dev_f1 = eval_epoch(dev_data)
    test_f1 = eval_epoch(test_data)
    result.append((dev_f1, test_f1))
    print("iter %2d | %6.2f | %6.2f | %6.2f | %6.2f | %6.2f |" % (i, end - start, train_acc, train_f1, dev_f1, test_f1))

result = torch.from_numpy(numpy.array(result))
max_dev_acc, max_index = torch.max(result[:, 0], 0)
print("Best Iter %d, Dev Acc: %s, Test Acc: %s" % (max_index[0], result[max_index[0], 0], result[max_index[0], 1]))
