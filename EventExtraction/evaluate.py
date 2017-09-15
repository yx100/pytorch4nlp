#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/9/15
def evalute(golden_list, pred_list):
    """
    :param golden_list: [(docid, start, length, type), ...]
    :param pred_list: [(docid, start, length, type), ...]
    :return:
    """
    tp, fp, fn = 0., 0., 0.

    gold_set = set(golden_list)
    pred_set = set(pred_list)

    tp += len(gold_set & pred_set)
    fp += len(pred_set - gold_set)
    fn += len(gold_set - pred_set)

    prec = tp / (tp + fp)
    reca = tp / (tp + fn)
    f1 =  2 * tp / (2 * tp + fp + fn)
    return prec, reca, f1
