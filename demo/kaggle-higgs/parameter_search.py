#!/usr/bin/python
# -*- coding: utf-8 -*-
# this is the example script to use xgboost to train 
from __future__ import division, print_function
import os
import sys
import math
from pprint import pprint
import numpy as np
from sklearn.cross_validation import KFold
# add path of xgboost python module
sys.path.append('../../python/')
import xgboost as xgb

test_size = 550000
DEFAULT_POINTS = 8
DEFAULT_FOLDS = 5
DEFAULT_RUNS = 2

RANDOM_STATE = 111


def AMS(s, b):
    """ Approximate Median Significance defined as:
        AMS = sqrt(2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s})
        where b_r = 10, b = background, s = signal, log is natural logarithm
    """
    assert s >= 0
    assert b >= 0
    bReg = 10.0
    return np.sqrt(2.0 * ((s + b + bReg) * np.log(1 + s / (b + bReg)) - s))



def get_y_pred(y_out, threshold_ratio):
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
    return y_pred


def get_score(w_test, y_test, y_out, threshold_ratio):
    y_pred = get_y_pred(y_out, threshold_ratio)

    positives = y_pred != 0
    true_positives = positives & (y_test != 0)
    false_positives = positives & (y_test == 0)
    signal = w_test[true_positives].sum()
    background = w_test[false_positives].sum()
    return AMS(signal, background)


def fit_predict(X_train, w_train, y_train, X_test, params):
    n_samples, n_features = X.shape
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

    # boost 120 trees
    num_round = 120
    plst = list(params.items())

    xgmat_train = xgb.DMatrix(X_train, label=y_train, missing=-999.0, weight=w_train)
    #print('loading data end, start to boost trees')
    watchlist = [] # [(xgmat_train, 'train')]
    bst = xgb.train(plst, xgmat_train, num_round, watchlist)

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

    score_summaries = np.empty((n_points, 2))
    for ic in xrange(all_scores.shape[0]):
        scores = all_scores[ic, :, :]
        score_summaries[ic, 0] = np.mean(scores)
        score_summaries[ic, 1] = np.std(scores, dtype=np.float64)
        print('***eval_params: tr=%.3f, score=%.3f+-%.3f' % (
            cutoffs[ic], score_summaries[ic, 0], score_summaries[ic, 1]))
    print('-' * 80)
    sys.stdout.flush()

    return score_summaries


def load_data():

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
}


if __name__ == '__main__':
    X, weight, y = load_data()
    test_scores = {test: eval_params(X, weight, y, test_params[test])
        for test in sorted(test_params.keys())}
    print('=' * 80)
    pprint(test_scores)

