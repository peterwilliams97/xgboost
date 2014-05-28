#!/usr/bin/python
# -*- coding: utf-8 -*-
# this is the example script to use xgboost to train 
from __future__ import division, print_function
import os
import sys
import math
from pprint import pprint
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.cross_validation import KFold
# add path of xgboost python module
sys.path.append('../../python/')
import xgboost as xgb

test_size = 550000
N_ROUNDS = 150 # 500
DEFAULT_POINTS = 10
DEFAULT_FOLDS = 5
DEFAULT_RUNS = 1

RANDOM_STATE = 111


def AMS(s, b):
    """ Approximate Median Significance defined as:
        AMS = sqrt(2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s})
        where b_r = 10, b = background, s = signal, log is natural logarithm

        bReg is like a fixed addition to background. Noise from unmodelled part of detector
        pipeline that we need to get above?
    """
    assert s >= 0
    assert b >= 0
    bReg = 10.0
    return np.sqrt(2.0 * ((s + b + bReg) * np.log(1 + s / (b + bReg)) - s))


def get_y_pred_rank(y_out, threshold_ratio):
    """Return binary classification y_pred for xgboost output y_out and
        Entries with top threshold_ratio  * n_samples y_out values are
        1, others are 0"""
    n_samples = y_out.shape[0]

    # indexes sorted by y_out in decreasing order
    indexes_by_y_out = y_out.argsort()
    # indexes to be set to 1
    positive_indexes = indexes_by_y_out[-int(threshold_ratio * n_samples):]

    y_pred = np.zeros(n_samples, dtype=int)
    y_pred[positive_indexes] = 1

    # Rank from lowest to highest, starting at 1
    rank = indexes_by_y_out.argsort()[::1] + 1

    return y_pred, rank


def get_score(w_test, y_test, y_out, threshold_ratio):
    y_pred, _ = get_y_pred_rank(y_out, threshold_ratio)

    positives = y_pred != 0
    true_positives = positives & (y_test != 0)
    false_positives = positives & (y_test == 0)
    signal = w_test[true_positives].sum()
    background = w_test[false_positives].sum()
    return AMS(signal, background)


def fit_predict(X_train, w_train, y_train, X_test, params):
    #n_samples, n_features = X.shape
    params = params.copy()
    # rescale weight to make it same as test set, Why? !@#$
    #weights = dtrain[:, 31] * test_size / n_samples

    # !@#$ One param set here
    sum_wpos = w_train[y_train == 1.0].sum()
    sum_wneg = w_train[y_train == 0.0].sum()
    # scale weight of positive examples
    params['scale_pos_weight'] = sum_wneg / sum_wpos
    params['silent'] =  1
    params['nthread'] = 4

    # boost N_ROUNDS (was 120) trees

    plst = list(params.items())

    xgmat_train = xgb.DMatrix(X_train, label=y_train, missing=-999.0, weight=w_train)
    #print('loading data end, start to boost trees')
    watchlist = [] # [(xgmat_train, 'train')]
    bst = xgb.train(plst, xgmat_train, N_ROUNDS, watchlist)

    xgmat_test = xgb.DMatrix(X_test, missing=-999.0)
    print('training data end, start to predict')
    y_out = bst.predict(xgmat_test)
    return y_out


def eval_params(X, weight, y, params, n_points=DEFAULT_POINTS, n_folds=DEFAULT_FOLDS,
    n_runs=DEFAULT_RUNS):

    cutoffs = np.linspace(0.10, 0.25, n_points)
    all_scores = np.empty((n_points, n_runs, n_folds))

    n_samples = y.shape[0]
    for run in xrange(n_runs):
        fold_list = KFold(n_samples, n_folds=n_folds, indices=False, shuffle=True,
                        random_state=RANDOM_STATE + 10 + run)
        for fold, (train, test) in enumerate(fold_list):
            X_train, X_test = X[train], X[test]
            w_train, w_test = weight[train], weight[test]
            y_train, y_test = y[train], y[test]

            w_train *= (sum(weight) / sum(w_train))
            w_test *= (sum(weight) / sum(w_test))

            y_out = fit_predict(X_train, w_train, y_train, X_test, params)
            for ic, threshold_ratio in enumerate(cutoffs):
                score = get_score(w_test, y_test, y_out, threshold_ratio)
                all_scores[ic, run, fold] = score
                print('eval_params: fold=%d,run=%d,score=%.3f' % (fold, run, score))
                sys.stdout.flush()

    score_summaries = {}
    for ic in xrange(all_scores.shape[0]):
        scores = all_scores[ic, :, :]
        threshold_ratio = cutoffs[ic]
        score_summaries[threshold_ratio] = np.mean(scores), np.std(scores, dtype=np.float64)
        print('***eval_params: tr=%.3f, score=%.3f+-%.3f' % (
            cutoffs[ic], score_summaries[threshold_ratio][0], score_summaries[threshold_ratio][1]))
    print('^' * 80)
    print(repr(score_summaries))
    print('-' * 80)
    sys.stdout.flush()

    return score_summaries


def load_training_data():

    # path to where the data lies
    dpath = 'data'
    train_path = os.path.join(dpath, 'training.csv')

    # load in training data, directly use numpy
    dtrain = np.loadtxt(train_path, delimiter=',', skiprows=1,
        converters={32: lambda x: int(x=='s'.encode('utf-8')) } )
    print ('finished loading from "%s",dtrain=%s' % (train_path, list(dtrain.shape)))

    label = dtrain[:, 32]  # Why float ? !@#$
    data = dtrain[:, 1:30]

    n_samples, n_features = data.shape
    # rescale weight to make it same as test set
    weight = dtrain[:, 31] # * float(test_size) / n_samples

    sum_wpos = weight[label == 1.0].sum()
    sum_wneg = weight[label == 0.0].sum()

    # print weight statistics 
    print('weight statistics: wpos=%.2f, wneg=%.2f, ratio=%.2f'
            % (sum_wpos, sum_wneg, sum_wneg / sum_wpos))
    print('-' * 80)
    sys.stdout.flush()

    X, y = data, label
    return X, weight, y


def load_test_data():

    # path to where the data lies
    dpath = 'data'
    test_path = os.path.join(dpath, 'test.csv')

    # load in training data, directly use numpy
    dtrain = np.loadtxt(test_path, delimiter=',', skiprows=1)
    print ('finished loading from "%s",dtrain=%s' % (test_path, list(dtrain.shape)))

    indexes = dtrain[:, 0].astype(int)
    data = dtrain[:, 1:30]

    n_samples, n_features = data.shape

    X = data
    return indexes, X


test_params = {
    # setup parameters for xgboost
    2: {
        # use logistic regression loss, use raw prediction before logistic transformation
        # since we only need the rank
        'objective': 'binary:logitraw',
        'bst:eta': 0.1,
        'bst:max_depth': 6,
        'eval_metric': 'ams@0.15',
    },

    1: {
        'objective': 'binary:logitraw',
        'bst:eta': 0.1,
        'bst:max_depth': 6,
        'eval_metric': 'auc',
    },

    3: {
        'objective': 'binary:logitraw',
        'bst:eta': 0.5,
        'bst:max_depth': 6,
        'eval_metric': 'auc',
    },

    4: {
        'objective': 'binary:logitraw',
        'bst:eta': 0.1,
        'bst:max_depth': 2,
        'eval_metric': 'auc',
    },

    5: {
        'objective': 'binary:logitraw',
        'bst:eta': 0.1,
        'bst:max_depth': 10,
        'eval_metric': 'auc',
    },

    6: {
        'objective': 'binary:logitraw',
        'bst:eta': 0.05,
        'bst:max_depth': 6,
        'eval_metric': 'auc',
    },

    7: {
        'objective': 'binary:logitraw',
        'bst:eta': 0.05,
        'bst:max_depth': 10,
        'eval_metric': 'auc',
    },
}


def make_submission(path, params, threshold_ratio):

    X_train, w_train, y_train = load_training_data()
    indexes_test, X_test = load_test_data()
    y_out = fit_predict(X_train, w_train, y_train, X_test, params)
    y_pred, rank = get_y_pred_rank(y_out, threshold_ratio)

    submission = DataFrame({'EventId': indexes_test, 'RankOrder': rank, 'Class': y_pred},
        columns=['EventId', 'RankOrder', 'Class'])
    submission['Class'] = submission['Class'].apply(lambda x: 's' if x else 'b')

    submission.to_csv(path, index=False)
    print('--------------------- Submission')
    print(submission.head())
    print(path)
    return submission


#
# Execution starts here
#
if __name__ == '__main__':

    np.random.seed(RANDOM_STATE)

    import optparse

    parser = optparse.OptionParser('python %s [options]' % sys.argv[0])
    parser.add_option('-a', '--do-all', action='store_true', dest='do_all', default=False,
            help='bootstrap samples')
    parser.add_option('-s', '--submit', dest='submission', default=None,
            help='make submission with this name')
    parser.add_option('-c', '--number-rounds', dest='n_rounds', default=N_ROUNDS, type=int,
            help='number of rounds')
    parser.add_option('-t', '--test-number', dest='test_number', default=-1, type=int,
            help='test number to run')
    parser.add_option('-r', '--threshold-ratio', dest='threshold_ratio', default=-1.0, type=float,
            help='threshold ration for signal')
    options, args = parser.parse_args()

    print(__file__)

    N_ROUNDS = options.n_rounds
    print('N_ROUNDS=%d' % N_ROUNDS)

    if options.submission:
        assert options.threshold_ratio > 0.0, 'Need to specify threshold_ratio'
        make_submission(options.submission, test_params[options.test_number], options.threshold_ratio)
        exit(0)

    test_keys = set()
    if options.do_all:
        test_keys = test_keys.union(test_params.keys())
    if options.test_number >= 0:
        test_keys.add(options.test_number)

    if not test_keys:
        print('no tests specified')
        exit(0)

    print('N_ROUNDS=%d,test_keys=%s' % (N_ROUNDS, sorted(test_keys)))

    X, weight, y = load_training_data()
    test_scores = {}
    best_test = -1
    best_threshold = -1.0
    best_score = 0.0

    for test in sorted(test_keys):
        score_summaries = eval_params(X, weight, y, test_params[test])
        test_scores[test] = score_summaries
        for threshold_ratio, (score, score_sd) in score_summaries.items():
           if score > best_score:
                best_test, best_threshold, best_score = test, threshold_ratio, score
        print('-' * 80)
        print('%d tests: best_test=%d,best_threshold=%.3f: best_score=%.3f' % (len(test_scores),
            best_test, best_threshold, best_score))
        print('best_test = %s' % repr(test_params[best_test]))
        print()
        print(repr(test_scores))
        print('=' * 80)

