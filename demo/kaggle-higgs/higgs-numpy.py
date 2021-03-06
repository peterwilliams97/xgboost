#!/usr/bin/python
# this is the example script to use xgboost to train 
from __future__ import division, print_function
import os
import sys
import numpy as np
# add path of xgboost python module
sys.path.append('../../python/')
import xgboost as xgb

test_size = 550000

# path to where the data lies
dpath = 'data'

# load in training data, directly use numpy
dtrain = np.loadtxt( dpath+'/training.csv', delimiter=',', skiprows=1,
    converters={32: lambda x: int(x=='s'.encode('utf-8')) } )
print ('finished loading from csv: dtrain=%s' % list(dtrain.shape))

label = dtrain[:, 32]  # Why float ? !@#$
data  = dtrain[:, 1:31]
# rescale weight to make it same as test set
weight = dtrain[:, 31] * float(test_size) / len(label)

sum_wpos = weight[label == 1.0].sum()
sum_wneg = weight[label == 0.0].sum()

# print weight statistics 
print('weight statistics: wpos=%.2f, wneg=%.2f, ratio=%.2f'
        % (sum_wpos, sum_wneg, sum_wneg / sum_wpos))

# construct xgboost.DMatrix from numpy array, treat -999.0 as missing value
xgmat = xgb.DMatrix(data, label=label, missing=-999.0, weight=weight)

# setup parameters for xgboost
param = {}
# use logistic regression loss, use raw prediction before logistic transformation
# since we only need the rank
param['objective'] = 'binary:logitraw'
# scale weight of positive examples
param['scale_pos_weight'] = sum_wneg / sum_wpos
param['bst:eta'] = 0.1
param['bst:max_depth'] = 6
param['eval_metric'] = 'auc'
param['silent'] = 1
param['nthread'] = 16

# you can directly throw param in, though we want to watch multiple metrics here 
plst = list(param.items()) + [('eval_metric', 'ams@0.15')]

watchlist = [ (xgmat,'train') ]
# boost 120 trees
num_round = 120
print ('loading data end, start to boost trees')
bst = xgb.train( plst, xgmat, num_round, watchlist )
# save out model
bst.save_model('higgs.model')

print ('finished training')
