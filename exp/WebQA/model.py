# -*- coding:utf-8 -*- 
# Author: Roger
# Created by Roger on 2017/10/20
import torch
import torch.nn as nn
from pt4nlp import RNNEncoder, Embeddings
from pt4nlp import CRFClassifier


class SeqLabelModel(nn.Module):
    def __init__(self, dicts, opt, label_num):

        super(SeqLabelModel, self).__init__()

        self.embedding = Embeddings(word_vec_size=opt.word_vec_size,
                                    dicts=dicts)
        self.question_encoder = RNNEncoder(input_size=self.embedding.output_size,
                                           hidden_size=opt.hidden_size,
                                           num_layers=opt.num_layers,
                                           dropout=opt.encoder_dropout,
                                           brnn=opt.brnn,
                                           rnn_type=opt.rnn_type)

        self.evidence_encoder = RNNEncoder(input_size=self.embedding.output_size,
                                           hidden_size=opt.hidden_size + opt.hidden_size,
                                           num_layers=opt.num_layers,
                                           dropout=opt.encoder_dropout,
                                           brnn=True,
                                           rnn_type=opt.rnn_type)

        self.classifier = CRFClassifier(label_num + 2)

    def forward(self, batch):
        q_word_emb = self.embedding.forward(batch.question)
        q_hidden_emb = self.question_encoder.forward(q_word_emb)

        e_word_emb = self.emb
        # change q_hidden_emb size
        evidence_input_emb = torch.cat([e_word_emb, q_hidden_emb, batch.qe_comm, batch.ee_comm])

        evidence_hidden = self.evidence_encoder.forward(evidence_input_emb)

        pred = self.classifier.forward(evidence_hidden)

        return pred
