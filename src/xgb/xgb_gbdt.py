# _*_ coding: utf-8 _*_

import zipfile
import pandas as pd
import xgboost as xgb
import gc
import numpy as np
from sklearn_pandas import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

path = '../../data/'
out_path = '../../output/xgb/'

drop_list = ['usertag']

train = pd.read_csv(path+'train.csv', dtype='object')
label_train = train['click']
train_data = train.drop(['click', 'usertag'], axis=1)
# test = pd.read_csv(path+'test.csv')
# y_test = test['click']
# X_test = test.drop(['click'], axis=1)

X_train, X_val, y_train, y_val = train_test_split(train_data, label_train, test_size=0.3, random_state=2017)
del train_data, label_train
gc.collect()
# print(X_train.dtypes)
xgb_val = xgb.DMatrix(X_val, label=y_val)
xgb_train = xgb.DMatrix(X_train.values, label=y_train.values)
evallist = [(xgb_train, 'train'), (xgb_val, 'eval')]

del X_train, y_train, X_val, y_val
gc.collect()

params = {'booster': 'gbtree', 'learning_rate': 0.1, 'n_estimators': 500, 'bst:max_depth': 4,
          'bst:min_child_weight': 1, 'bst:eta': 0.05, 'silent': 1, 'objective': 'binary:logistic',
          'gamma': 0.1, 'subsample': 0.8, 'scale_pos_weight': 1, 'colsample_bytree': 0.8,
          'eval_metric': 'logloss', 'nthread': 4, 'sample_type': 'uniform',
          'normalize_type': 'forest', 'tree_method': 'approx'}

num_round = 1000

bst = xgb.train(params, xgb_train, num_round, evals=evallist, early_stopping_rounds=5)
del xgb_train, xgb_val
gc.collect()

#
# bst = xgb.cv(params=params, dtrain=xgb_train, nfold=5, metrics='logloss', verbose_eval=2, early_stopping_rounds=5)

bst.save_model(out_path + 'xgb.model')

print(bst.get_fscore())

test_data = pd.read_csv(path + 'test.csv')

test_data = test_data.drop(['click', 'usertag'], axis=1)

xgb_test = xgb.DMatrix(test_data)

del test_data
gc.collect()

y_pred = bst.predict(xgb_test)

output = open(out_path + 'submission.csv', 'w')
output.write('id,prob\n')
for t, p in enumerate(y_pred, start=1):
    output.write('{0},{1}\n'.format(t, p))
output.close()
