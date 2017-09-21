#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/9/15
def evalute(golden_list, pred_list, trigger_type=True):
    """
    :param golden_list: [(docid, start, length, type), ...]
    :param pred_list: [(docid, start, length, type), ...]
    :return:
    """
    if trigger_type:
        gold_set = set(golden_list)
        pred_set = set()
        for pred in pred_list:
            if pred[3] != u'other':
                pred_set.add(pred)
    else:
        gold_set = set([d[:3] for d in golden_list])
        pred_set = set()
        for pred in pred_list:
            if pred[3] != u'other':
                pred_set.add(pred[:3])

    if len(pred_set) == 0:
        return 0, 0, 0

    tp, fp, fn = 0., 0., 0.

    tp += len(gold_set & pred_set)
    fp += len(pred_set - gold_set)
    fn += len(gold_set - pred_set)
    prec = tp / (tp + fp)
    reca = tp / (tp + fn)
    f1 =  2 * tp / (2 * tp + fp + fn)

    print("||tp: %6d | fp: %6d | fn: %6d || %6.2f | %6.2f | %6.2f ||"
          % (tp, fp, fn, prec * 100., reca * 100., f1 * 100.))

    return prec * 100., reca * 100., f1 * 100.
