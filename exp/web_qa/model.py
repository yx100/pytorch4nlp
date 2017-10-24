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

        # TODO
        self.ee_embedding = Embeddings(word_vec_size=opt.word_vec_size,
                                       dicts=dicts)
        self.qe_embedding = Embeddings(word_vec_size=opt.word_vec_size,
                                       dicts=dicts)
        self.question_encoder = RNNEncoder(input_size=self.embedding.output_size,
                                           hidden_size=opt.hidden_size,
                                           num_layers=opt.num_layers,
                                           dropout=opt.encoder_dropout,
                                           brnn=opt.brnn,
                                           rnn_type=opt.rnn_type)

        self.question_attention = DotMLPWordSeqAttention(input_size=opt.word_vec_size, seq_size=opt.hidden_size)

        self.evidence_encoder = RNNEncoder(
            input_size=self.embedding.output_size * 3 + self.question_encoder.output_size,
            hidden_size=opt.hidden_size,
            num_layers=opt.num_layers,
            dropout=opt.encoder_dropout,
            brnn=opt.brnn,
            rnn_type=opt.rnn_type)

        self.tag_encoder = RNNEncoder(input_size=self.evidence_encoder.output_size,
                                      hidden_size=label_num,
                                      num_layers=opt.num_layers,
                                      dropout=opt.encoder_dropout,
                                      brnn=False,
                                      rnn_type=opt.rnn_type)

        self.classifier = CRFClassifier(label_num)

    def get_feature(self, batch):
        question_attention_p = self.question_attention_p.unsqueeze(0).expand(batch.batch_size,
                                                                             self.question_attention_p.size(0))
        q_word_emb = self.embedding.forward(batch.q_text)
        q_hidden_embs, _ = self.question_encoder.forward(q_word_emb)
        q_hidden_embs = q_hidden_embs.contiguous()

        q_hidden_emb = self.question_attention.forward(question_attention_p, q_hidden_embs, lengths=batch.q_lens)

        e_word_emb = self.embedding.forward(batch.e_text)

        eq_embedding = self.qe_embedding.forward(batch.qe_feature)
        ee_embedding = self.ee_embedding.forward(batch.ee_feature)

        q_hidden_emb = q_hidden_emb.unsqueeze(1).expand_as(e_word_emb)

        evidence_input_emb = torch.cat([e_word_emb, q_hidden_emb, eq_embedding, ee_embedding], 2)

        evidence_hidden, _ = self.evidence_encoder.forward(evidence_input_emb)

        tag_feature, _ = self.tag_encoder.forward(evidence_hidden)

        return tag_feature

    def loss(self, batch):
        feature = self.get_feature(batch)

        loss = self.classifier.neg_log_loss(feature, batch.label, lengths=batch.q_lens)

        return loss

    def predict(self, batch):
        feature = self.get_feature(batch)

        scores, paths = self.classifier.viterbi_decode(feature, lengths=batch.q_lens)

        return scores, paths

    def forward(self, batch):
        return self.loss(batch)
