import json
from csv import DictReader

import lightgbm as lgb
import scipy as sp
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn_pandas import GridSearchCV


'''
输入数据为libsvm格式
'''

data_path = '../../output'
path = '../../output'

y_true = []
fi = open('{0}/test.txt'.format(path), 'r')
for line in fi:
    y_true.append(line.split(' ')[0])
fi.close()


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll


print('Load data...')
train = '{0}/train.txt'.format(data_path)
test = '{0}/test.txt'.format(data_path)
lgb_train = lgb.Dataset(train)
lgb_eval = lgb.Dataset(test, reference=lgb_train)

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'binary_logloss', 'auc'},
    'metric_freq': 1,
    'is_training_metric': 'false',
    'max_bin': 255,
    'num_leaves': 31,
    'learning_rate': 0.2,
    'tree_learner': 'serial',
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 20,
    'min_sum_hessian_in_leaf': 1e-3,
    'max_depth': 20

}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=150,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

# print('Save model...')
# # save model to file
# gbm.save_model('{0}/model.txt'.format(path))

# print('Start predicting...')
# predict
y_pred = gbm.predict('{0}/test.txt'.format(data_path), num_iteration=gbm.best_iteration)
# pd.DataFrame(y_pred).to_csv('{0}/pred_lgb.csv'.format(path), index=False)
# eval
print('The auc of prediction is:', roc_auc_score(y_true, y_pred))
print('The logloss of prediction is:', logloss(y_true, y_pred))

# print('Dump model to JSON...')
# dump model to json (and save to file)
# model_json = gbm.dump_model()

# with open('{0}/model.json'.format(path), 'w+') as f:
#     json.dump(model_json, f, indent=4)

# print('Feature names:', gbm.feature_name())
#
# print('Calculate feature importances...')
# # feature importances
# print('Feature importances:', list(gbm.feature_importance()))

# other scikit-learn modules
# estimator = lgb.LGBMRegressor(num_leaves=31)
#
# param_grid = {
#     'learning_rate': [0.01, 0.1, 1],
#     'n_estimators': [50, 100, 150]
# }
#
# gbm = GridSearchCV(estimator, param_grid)
#
# gbm.fit(train)

# print('Best parameters found by grid search are:', gbm.best_params_)
