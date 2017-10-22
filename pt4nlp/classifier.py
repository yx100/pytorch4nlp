# -*- coding:utf-8 -*- 
# Author: Roger
import torch
import torch.nn as nn


class SoftmaxClassifier(nn.Module):
    """
    Applies the Softmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range (0,1) and sum to 1

    Softmax is defined as
    :math:`f_i(x) = exp(x_i) / sum_j exp(x_j)`

    Shape:
        - Input: :math:`(N, L)`
        - Output: :math:`(N, L)`
    """

    def __init__(self):
        super(SoftmaxClassifier, self).__init__()

        self.classifier = nn.Softmax()

    def forward(self, x):
        """
        :param x: `(N, L)`
        :return: `(N, L)`
        """
        assert x.dim() == 2, 'Softmax requires a 2D tensor as input'
        return self.classifier.forward(x)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'

    def predict_prob(self, x):
        """
        :param x: `(N, L)`
        :return: `(N, L)`
        """
        return self.forward(x)

    def predict_label(self, x):
        """
        :param x: `(N, L)`
        :return: `(N, L)`
        """
        prob = self.predict_prob(x)
        _, pred_label = torch.max(prob, 1)
        return pred_label

    def predict(self, x, prob=True):
        return self.predict_prob(x) if prob else self.predict_label(x)


class CRFClassifier(nn.Module):
    pass
