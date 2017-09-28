#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/9/28
from __future__ import absolute_import
from builtins import range
import common
import time
import torch
import torch.nn as nn
import pt4nlp
from argparse import ArgumentParser
from deconv_corpus import Corpus
from model import TextDeconvolutionAutoEncoer


parser = ArgumentParser(description='Deconvolution Autoencoder')
# Train Option
parser.add_argument('-epoch', type=int, dest="epoch", default=100)
parser.add_argument('-batch', type=int, dest="batch", default=64)
parser.add_argument('-device', type=int, dest="device", default=0)
parser.add_argument('-seed', type=int, dest="seed", default=-1)
parser.add_argument('-train-data', type=str, dest="train_data", default="train.data")
parser.add_argument('-dev-data', type=str, dest="dev_data", default="dev.data")
parser.add_argument('-word-cut', type=int, dest="word_cut", default=2)
parser.add_argument('-no-lower', action='store_false', dest='lower')

# Model Option
parser.add_argument('-word-vectors', type=str, dest="word_vectors", default='random')

# Optimizer Option
parser.add_argument('-word-normalize', action='store_true', dest="word_normalize")
parser.add_argument('-optimizer', type=str, dest="optimizer", default="Adadelta")
parser.add_argument('-lr', type=float, dest="lr", default=0.95)
parser.add_argument('-word-optimizer', type=str, dest="word_optimizer", default="Adadelta")
parser.add_argument('-word-lr', type=float, dest="word_lr", default=0.95)
parser.add_argument('-grad-clip', type=float, default=9.0, dest="grad_clip", help='clip grad by norm')
parser.add_argument('-weight-clip', type=float, default=9.0, dest="weight_clip", help='clip weight by norm')
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

word_d = Corpus.get_word_dictionary_from_file(args.train_data, lower=args.lower)
word_d.cut_by_count(args.word_cut)
n_token = len(word_d)

train_data = Corpus(args.train_data, word_d,
                    batch_size=args.batch, device=args.device,
                    max_length=57,
                    )

model = TextDeconvolutionAutoEncoer(word_d)

if args.word_vectors != "random":
    model.embedding.load_pretrained_vectors(args.word_vectors, normalize=args.word_normalize)


weight = torch.ones(word_d.size())
weight[pt4nlp.PAD] = 0


if args.device >= 0:
    model.cuda(args.device)
    weight = weight.cuda(args.device)


criterion = nn.CrossEntropyLoss(weight=weight)
param_wo_embedding = []
param_embedding = []

for name, param in model.named_parameters():
    if "emb_luts" in name:
        print("%s(%s)\t%s with %s" % (name, param.size(), args.word_optimizer, args.word_lr))
        param_embedding.append(param)
    else:
        print("%s(%s)\t%s with %s" % (name, param.size(), args.optimizer, args.lr))
        param_wo_embedding.append(param)

if args.optimizer == 'Adadelta':
    wo_word_opt = getattr(torch.optim, args.optimizer)(param_wo_embedding, rho=args.lr,
                                                       weight_decay=args.regular_weight)
    word_opt = getattr(torch.optim, args.word_optimizer)(param_embedding, rho=args.word_lr,
                                                         weight_decay=args.regular_weight)
else:
    wo_word_opt = getattr(torch.optim, args.optimizer)(param_wo_embedding, lr=args.lr, weight_decay=args.regular_weight)
    word_opt = getattr(torch.optim, args.word_optimizer)(param_embedding, lr=args.word_lr,
                                                         weight_decay=args.regular_weight)


def train_epoch():
    epoch_loss = 0.
    model.train()

    for batch in train_data.next_batch():
        wo_word_opt.zero_grad()
        word_opt.zero_grad()

        pred = model(batch)

        output = pred.view(-1, n_token)
        target = batch.text.view(-1)

        # calculate loss of the network output with respect to training labels
        loss = criterion(output, target)

        # backpropagate and update optimizer learning rate
        loss.backward()

        if args.grad_clip > 0:
            nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)

        wo_word_opt.step()
        word_opt.step()

        epoch_loss += loss.data[0]

        if args.weight_clip > 0:
            pt4nlp.clip_weight_norm(model, args.weight_clip, except_params=['emb_luts.0'])

    return epoch_loss


for i in range(args.epoch):
    start = time.time()
    train_loss = train_epoch()
    end = time.time()
    torch.save(model, open('epoch_%s_%s' % (i, train_loss)))
    print("| %s | %s |" % (end - start, train_loss))
