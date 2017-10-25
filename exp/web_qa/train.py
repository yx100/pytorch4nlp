# -*- coding:utf-8 -*- 
# Author: Roger
# Created by Roger on 2017/10/24
import time
import torch
import torch.nn as nn
import common
from model import SeqLabelModel
from corpus import WebQACorpus
from argparse import ArgumentParser


parser = ArgumentParser(description='Web QA Reader')


# Train Option
parser.add_argument('-epoch', type=int, dest="epoch", default=50)
parser.add_argument('-batch', type=int, dest="batch", default=128)
parser.add_argument('-device', type=int, dest="device", default=-1)
parser.add_argument('-seed', type=int, dest="seed", default=1993)
parser.add_argument('-train-file', type=str, dest="train_file", default="train.json.head")
parser.add_argument('-dev-file', type=str, dest="dev_file", default=None)
parser.add_argument('-test-file', type=str, dest="test_file", default=None)


# Model Option
parser.add_argument('-word-vec-size', type=int, dest="word_vec_size", default=5)
parser.add_argument('-hidden-size', type=int, dest="hidden_size", default=5)
parser.add_argument('-num-layers', type=int, dest='num_layers', default=1)
parser.add_argument('-encoder-dropout', type=float, dest='encoder_dropout', default=0)
parser.add_argument('-dropout', type=float, dest='dropout', default=0.5)
parser.add_argument('-brnn', action='store_true', dest='brnn')
parser.add_argument('-word-vectors', type=str, dest="word_vectors", default='random')
parser.add_argument('-rnn-type', type=str, dest='rnn_type', default='LSTM', choices=["RNN", "GRU", "LSTM"])


# Optimizer Option
parser.add_argument('-word-normalize', action='store_true', dest="word_normalize")
parser.add_argument('-optimizer', type=str, dest="optimizer", default="Adadelta")
parser.add_argument('-lr', type=float, dest="lr", default=0.95)
parser.add_argument('-clip', type=float, default=9.0, dest="clip", help='clip grad by norm')
parser.add_argument('-regular', type=float, default=10e-4, dest="regular_weight", help='regular weight')


loss_acc = torch.Tensor([0])
args = parser.parse_args()

corpus = WebQACorpus("training.json.head", batch_size=args.batch, device=args.device)

model = SeqLabelModel(corpus.word_d, args, corpus.label_d.size())

if args.seed < 0:
    seed = time.time() % 10000
else:
    seed = args.seed
print("Random Seed: %d" % seed)
torch.manual_seed(int(seed))

params = list()
for name, param in model.named_parameters():
    print(name, param.size())
    params.append(param)

opt = getattr(torch.optim, args.optimizer)(params, lr=args.lr, weight_decay=args.regular_weight)


for i in range(100):
    loss_acc = 0
    num_batch = len(corpus) / 64
    for batch in corpus.next_batch():
        opt.zero_grad()

        loss = model.loss(batch)
        loss.backward()
        loss_acc += loss.data

        if args.clip > 0:
            nn.utils.clip_grad_norm(model.parameters(), args.clip)

        opt.step()

    print(loss_acc / num_batch)
    torch.save([corpus.word_d, model], "model.epoch.%s.loss.%.4f.pkl" % (i, (loss_acc / num_batch)[0]))
