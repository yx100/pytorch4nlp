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
import evaluate
from event_corpus import EECorpus
from argparse import ArgumentParser
import numpy
import pt4nlp

parser = ArgumentParser(description='DMCNN Event Detector')
# Train Option
parser.add_argument('-epoch', type=int, dest="epoch", default=100)
parser.add_argument('-batch', type=int, dest="batch", default=170)
parser.add_argument('-device', type=int, dest="device", default=0)
parser.add_argument('-seed', type=int, dest="seed", default=-1)
parser.add_argument('-data-folder', type=str, dest="data_folder", default="trigger_ace_data")
parser.add_argument('-neg-ratio', type=float, dest="neg_ratio", default=14.)
parser.add_argument('-fix-neg', action='store_true', dest='fix_neg')
parser.add_argument('-word-cut', type=int, dest="word_cut", default=1)
parser.add_argument('-no-lower', action='store_false', dest='lower')
parser.add_argument('-dev-test-pre', action='store_true', dest="dev_test_pre")
parser.add_argument('-just-pos-sent-word', action='store_true', dest="just_pos_sent_word")
parser.add_argument('-set-eval', action='store_true', dest="set_eval")
parser.add_argument('-neg-from-global', action='store_true', dest="neg_from_global")
parser.add_argument('-log-file', type=str, dest="err_instance_file_name", default=None)

# Model Option
parser.add_argument('-word-vec-size', type=int, dest="word_vec_size", default=100)
parser.add_argument('-posi-vec-size', type=int, dest="posi_vec_size", default=5)
parser.add_argument('-hidden-size', type=int, dest="hidden_size", default=200)
parser.add_argument('-encoder-dropout', type=float, dest='encoder_dropout', default=0)
parser.add_argument('-dropout', type=float, dest='dropout', default=0.5)
parser.add_argument('-bn', action='store_true', dest='bn')
parser.add_argument('-act', type=str, dest='act', default='Tanh')
parser.add_argument('-word-vectors', type=str, dest="word_vectors", default='word_word2vec.bin')
parser.add_argument('-cnn-size', nargs='+', dest='cnn_size', default=[3])
parser.add_argument('-cnn-pooling', type=str, dest='cnn_pooling', default="max", choices=["max", "sum", "mean"])
parser.add_argument('-lexi-window', type=int, dest='lexi_window', default=1,
                    help='-1 is no lexi feature, 0 is just centre word')

# Optimizer Option
parser.add_argument('-word-normalize', action='store_true', dest="word_normalize")
parser.add_argument('-optimizer', type=str, dest="optimizer", default="Adadelta")
parser.add_argument('-lr', type=float, dest="lr", default=0.05)
parser.add_argument('-word-optimizer', type=str, dest="word_optimizer", default="SGD")
parser.add_argument('-word-lr', type=float, dest="word_lr", default=0.1)
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

if args.set_eval:
    evalute = evaluate.evalute_set
else:
    evalute = evaluate.evalute_hongyu

usecuda = False
batch_size = args.batch

if args.device >= 0:
    usecuda = True


def get_data_file_names(file_type='train'):
    return (args.data_folder + "/%s/%s.golden.dat" % (file_type, file_type),
            args.data_folder + "/%s/%s.ids.dat" % (file_type, file_type),
            args.data_folder + "/%s/%s.sents.dat" % (file_type, file_type))


label_d = EECorpus.load_label_dictionary(args.data_folder + "/label2id.dat")
print("Label Size: %s" % len(label_d))
posit_d = EECorpus.get_position_dictionary(50)
print("Position Vocab Size: %s" % len(posit_d))
word_d = EECorpus.get_word_dictionary_from_ids_file(get_data_file_names('train')[1],
                                                    just_pos_sent=args.just_pos_sent_word,
                                                    lower=args.lower)
if args.dev_test_pre:
    word_d = EECorpus.get_word_dictionary_from_ids_file(get_data_file_names('dev')[1], word_d)
    word_d = EECorpus.get_word_dictionary_from_ids_file(get_data_file_names('test')[1], word_d)
word_d.cut_by_count(args.word_cut)

train_data = EECorpus(get_data_file_names('train')[0],
                      get_data_file_names('train')[1],
                      get_data_file_names('train')[2],
                      word_d, posit_d, label_d, lexi_window=args.lexi_window,
                      batch_size=args.batch, device=args.device, neg_ratio=args.neg_ratio, fix_neg=args.fix_neg,
                      train=True, neg_from_global=args.neg_from_global)
train_eval_data = EECorpus(get_data_file_names('train')[0],
                           get_data_file_names('train')[1],
                           get_data_file_names('train')[2],
                           word_d, posit_d, label_d,
                           lexi_window=args.lexi_window, batch_size=1000,
                           device=args.device, neg_ratio=0, random=False)
dev_data = EECorpus(get_data_file_names('dev')[0],
                    get_data_file_names('dev')[1],
                    get_data_file_names('dev')[2],
                    word_d, posit_d, label_d, lexi_window=args.lexi_window, batch_size=1000,
                    device=args.device, neg_ratio=0, random=False)
test_data = EECorpus(get_data_file_names('test')[0],
                     get_data_file_names('test')[1],
                     get_data_file_names('test')[2],
                     word_d, posit_d, label_d, lexi_window=args.lexi_window, batch_size=1000,
                     device=args.device, neg_ratio=0, random=False)

model = DynamicMultiPoolingCNN(word_d, opt=args, label_num=label_d.size(), position_dict=posit_d)
if args.word_vectors != "random":
    model.embedding.load_pretrained_vectors(args.word_vectors, normalize=args.word_normalize)

if len(model.embedding.emb_luts) > 1:
    torch.nn.init.uniform(model.embedding.emb_luts[-1].weight, -1, 1)

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

if args.optimizer == 'Adadelta':
    wo_word_opt = getattr(torch.optim, args.optimizer)(param_wo_embedding, rho=args.lr,
                                                       weight_decay=args.regular_weight)
    word_opt = getattr(torch.optim, args.word_optimizer)(param_embedding, rho=args.word_lr,
                                                         weight_decay=args.regular_weight)
else:
    wo_word_opt = getattr(torch.optim, args.optimizer)(param_wo_embedding, lr=args.lr, weight_decay=args.regular_weight)
    word_opt = getattr(torch.optim, args.word_optimizer)(param_embedding, lr=args.word_lr,
                                                         weight_decay=args.regular_weight)


def eval_epoch(data, log_out=None):
    pred_results = list()
    model.eval()
    for batch in data.next_batch():
        pred = model(batch)
        pred_label = torch.max(pred, 1)[1].data
        batch.pred = pred_label
        batch_pred = data.batch2pred(batch)
        if log_out is not None:
            data.batch2log(batch)
        pred_results += batch_pred
    type_p, type_r, type_f = evalute(data.gold_data, pred_results)
    untype_p, untype_r, untype_f = evalute(data.gold_data, pred_results, trigger_type=False)
    return type_p, type_r, type_f, untype_p, untype_r, untype_f


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

        if args.grad_clip > 0:
            nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)

        wo_word_opt.step()
        word_opt.step()

        if args.weight_clip > 0:
            pt4nlp.clip_weight_norm(model, args.weight_clip, except_params=['emb_luts.0'])

    return 100. * n_correct / n_total


result = list()
for i in range(args.epoch):
    start = time.time()
    train_acc = train_epoch(i)
    end = time.time()

    if args.err_instance_file_name is not None:
        err_output = open(args.err_instance_file_name + ".%s" % i + '.log', 'w')
    else:
        err_output = None

    train_type_p, train_type_r, train_type_f1, train_span_p, train_span_r, train_span_f1 = eval_epoch(train_eval_data)
    dev_type_p, dev_type_r, dev_type_f1, dev_span_p, dev_span_r, dev_span_f1 = eval_epoch(dev_data)
    test_type_p, test_type_r, test_type_f1, test_span_p, test_span_r, test_span_f1 = eval_epoch(test_data, True)

    result.append((dev_span_f1, test_span_f1, dev_type_f1, test_type_f1))
    print("SPAN iter %2d | %6.2f | %6.2f || %6.2f | %6.2f | %6.2f || %6.2f | %6.2f | %6.2f || %6.2f | %6.2f | %6.2f ||"
          % (i, end - start, train_acc,
             train_span_p, train_span_r, train_span_f1,
             dev_span_p, dev_span_r, dev_span_f1,
             test_span_p, test_span_r, test_span_f1,))
    print("TYPE iter %2d | %6.2f | %6.2f || %6.2f | %6.2f | %6.2f || %6.2f | %6.2f | %6.2f || %6.2f | %6.2f | %6.2f ||"
          % (i, end - start, train_acc,
             train_type_p, train_type_r, train_type_f1,
             dev_type_p, dev_type_r, dev_type_f1,
             test_type_p, test_type_r, test_type_f1,))
    print
    if err_output is not None:
        err_output.close()


result = torch.from_numpy(numpy.array(result))
_, max_index = torch.max(result[:, 0], 0)
print("Best Untype Iter %d, Dev F1: %s, Test F1: %s" % (max_index[0], result[max_index[0], 0], result[max_index[0], 1]))
_, max_index = torch.max(result[:, 2], 0)
print("Best Typed  Iter %d, Dev F1: %s, Test F1: %s" % (max_index[0], result[max_index[0], 2], result[max_index[0], 3]))
