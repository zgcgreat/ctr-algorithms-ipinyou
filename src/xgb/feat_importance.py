import numpy as np
import pandas as pd
import xgboost as xgb
import gc
import operator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def ceate_feature_map(features):
    outfile = open('../../output/xgb/xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


def hour(series):
    return int(str(series)[2:4])


if __name__ == '__main__':
    train = pd.read_csv("../../data/train_data/train.csv")

    features = [x for x in train.columns if x not in ['label']]
    print(features)
    ceate_feature_map(features)

    train = train.sample(frac=0.8, random_state=1)


    params = {'booster': 'gbtree', 'learning_rate': 0.1, 'n_estimators': 100, 'bst:max_depth': 4,
              'bst:min_child_weight': 1, 'bst:eta': 0.05,
              'silent': 1, 'objective': 'reg:logistic', 'gamma': 0.1, 'subsample': 0.8, 'scale_pos_weight': 1,
              'colsample_bytree': 0.8, 'eval_metric': 'logloss', 'nthread': 4, 'sample_type': 'uniform',
              'normalize_type': 'forest'}
    rounds = 200

    y_train = train['label']
    X_train = train.drop(['label'], axis=1)
    del train
    gc.collect()
    # y_test = test['label']
    # X_test = test.drop(['label'], axis=1)
    #
    # xgtrain = xgb.DMatrix(X_train, label=y_train)

    X_train, val_X, y_train, val_y = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

    xgb_val = xgb.DMatrix(val_X, label=val_y)
    xgb_train = xgb.DMatrix(X_train, label=y_train)

    del X_train, y_train, val_X, val_y
    gc.collect()

    evallist = [(xgb_train, 'train'), (xgb_val, 'eval')]

    bst = xgb.train(params, xgb_train, evals=evallist, num_boost_round=rounds, early_stopping_rounds=5)

    importance = bst.get_fscore(fmap='../../output/xgb/xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    df.to_csv("../../output/xgb/feat_importance.csv", index=False)

    plt.figure()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.savefig('../../output/xgb/feats_importance.png', dpi=100)
    # plt.show()
