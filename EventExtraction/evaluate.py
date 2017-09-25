#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/9/15
def evalute_set(golden_list, pred_list, trigger_type=True):
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

    # print("||tp: %6d | fp: %6d | fn: %6d || %6.2f | %6.2f | %6.2f ||"
    #       % (tp, fp, fn, prec * 100., reca * 100., f1 * 100.))

    print("Pred Right: %s, Pred Num %s, Total Right: %s." % (tp, len(pred_set), len(gold_set)))

    return prec * 100., reca * 100., f1 * 100.


def evalute(golden_list, pred_list, trigger_type=True):
    return evalute_hongyu(golden_list, pred_list, trigger_type)


def evalute_hongyu(golden_list, pred_list, trigger_type=True):
    """
    :param golden_list: [(docid, start, length, type), ...]
    :param pred_list: [(docid, start, length, type), ...]
    :return:
    """
    pred_sum = 0.
    span_tp = 0.
    type_tp = 0.
    span_set = set([(docid, start, length) for docid, start, length, _ in golden_list])
    type_set = set([(docid, start, length, typename) for docid, start, length, typename in golden_list])

    for docid, start, length, typename in pred_list:
        if typename == 'other':
            continue
        pred_sum += 1
        if (docid, start, length) in span_set:
            span_tp += 1
            if (docid, start, length, typename) in type_set:
                type_tp += 1

    if span_tp < 1 or type_tp < 1:
        return 0, 0, 0

    if trigger_type:
        tp = type_tp
    else:
        tp = span_tp

    prec = tp / pred_sum
    reca = tp / len(golden_list)
    f1 =  2 * prec * reca / (prec + reca)

    # print("||tp: %6d | fp: %6d | fn: %6d || %6.2f | %6.2f | %6.2f ||"
    #       % (tp, fp, fn, prec * 100., reca * 100., f1 * 100.))

    print("Pred Right: %s, Pred Num %s, Total Right: %s." % (tp, pred_sum, len(golden_list)))

    return prec * 100., reca * 100., f1 * 100.
