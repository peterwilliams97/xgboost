#!/usr/bin/python
# make prediction 
from __future__ import division, print_function
import sys, os
import numpy as np
# add path of xgboost python module
sys.path.append('../../python/')
import xgboost as xgb

# path to where the data lies
dpath = 'data'

modelfile = 'higgs.model'
outfile = 'higgs.pred.csv'
# mark top 15% as positive
threshold_ratio = 0.15

# load in training data, directly use numpy
test_path = os.path.join(dpath, 'test.csv')
dtest = np.loadtxt(test_path, delimiter=',', skiprows=1)
data = dtest[:, 1:31]
indexes = dtest[:, 0].astype(int)
ntot = len(indexes)

print ('finished loading from "%s", %d samples' % (test_path, ntot))
xgmat = xgb.DMatrix(data, missing=-999.0)
bst = xgb.Booster({'nthread': 16})
bst.load_model( modelfile )
ypred = bst.predict( xgmat )

# indexes sorted by ypred in decreasing order
indexes_by_ypred = [idx for idx, _ in sorted(zip(indexes, ypred), key=lambda x: -x[1])]

# rorder[idx] = rank by ypred, starting at 1
rorder = {idx: i + 1 for i, idx in enumerate(indexes_by_ypred)}

# label top threshold_ratio elements in terms of ypred as signal 
ntop = threshold_ratio * ntot
labels = {idx: 's' if rorder[idx] <= ntop else 'b' for idx in indexes}

# write out predictions
with open(outfile, 'w') as fo:
    fo.write('EventId,RankOrder,Class\n')
    for idx in indexes:
        fo.write('%s,%d,%s\n' % (idx, rorder[idx], labels[idx]))

print('finished writing into prediction file')

nhit = sum(lbl == 's' for lbl in labels.values())
print('ntot=%d,nhit=%d,ratio=%.3f' % (ntot, nhit, nhit / ntot))

