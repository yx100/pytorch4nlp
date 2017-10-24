# -*- coding:utf-8 -*- 
# Author: Roger
# Created by Roger on 2017/10/20
import torch
import torch.nn as nn
from pt4nlp import RNNEncoder, Embeddings
from pt4nlp import CRFClassifier, DotMLPWordSeqAttention


class SeqLabelModel(nn.Module):

    def __init__(self, dicts, opt, label_num):

        super(SeqLabelModel, self).__init__()

        self.question_attention_p = nn.Parameter(torch.Tensor(opt.word_vec_size))

        self.embedding = Embeddings(word_vec_size=opt.word_vec_size,
                                    dicts=dicts)
        self.question_encoder = RNNEncoder(input_size=self.embedding.output_size,
                                           hidden_size=opt.hidden_size,
                                           num_layers=opt.num_layers,
                                           dropout=opt.encoder_dropout,
                                           brnn=opt.brnn,
                                           rnn_type=opt.rnn_type)

        self.question_attention = DotMLPWordSeqAttention(input_size=opt.word_vec_size, seq_size=opt.hidden_size)

        self.evidence_encoder = RNNEncoder(input_size=self.embedding.output_size,
                                           hidden_size=opt.hidden_size + opt.hidden_size,
                                           num_layers=opt.num_layers,
                                           dropout=opt.encoder_dropout,
                                           brnn=True,
                                           rnn_type=opt.rnn_type)

        self.classifier = CRFClassifier(label_num + 2)

    def get_feature(self, batch):
        question_attention_p = self.question_attention_p.unsqueeze(0).expand(batch.batch_size, -1)

        q_word_emb = self.embedding.forward(batch.question)
        q_hidden_embs, _ = self.question_encoder.forward(q_word_emb)

        q_hidden_emb = self.question_attention.forward(question_attention_p, q_hidden_embs, lengths=batch.lengths)

        e_word_emb = self.embedding.forward(batch.evidence)

        evidence_input_emb = torch.cat([e_word_emb, q_hidden_emb, batch.qe_comm, batch.ee_comm], 1)

        evidence_hidden = self.evidence_encoder.forward(evidence_input_emb)

        return evidence_hidden

    def loss(self, batch):
        feature = self.get_feature(batch)

        loss = self.classifier.neg_log_loss(feature, batch.label, lengths=batch.lengths)

        return loss

    def predict(self, batch):
        feature = self.get_feature(batch)

        scores, paths = self.classifier.viterbi_decode(feature, lengths=batch.lengths)

        return scores, paths

    def forward(self, batch):
        return self.loss(batch)
