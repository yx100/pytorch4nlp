# -*- coding:utf-8 -*-
# Author: Roger
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
parser.add_argument('-batch', type=int, dest="batch", default=128)
parser.add_argument('-device', type=int, dest="device", default=-1)
parser.add_argument('-model', type=str, dest="model_file", default=None)
parser.add_argument('-test-file', type=str, dest="test_file", default=None)
parser.add_argument('-out-file', type=str, dest="out_file", default=None)

args = parser.parse_args()
word_d, model = torch.load(args.model_file)
if args.device == -1:
    model = model.cpu()
else:
    model = model.cuda(args.device)

corpus = WebQACorpus(args.test_file, batch_size=args.batch, device=args.device)

with open(args.out_file, 'w') as output:
    for batch in corpus.next_batch():

        def to_int_str(x):
            return str(int(x))

        scores, paths = model.predict(batch)
        for i in range(paths.size(0)):
            path = paths[i][:batch.e_lens[i].data[0]].squeeze(-1).data
            path_str = " ".join(map(to_int_str, path))
            output.write(path_str + "\n")
