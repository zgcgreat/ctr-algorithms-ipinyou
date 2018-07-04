# _*_ coding: utf-8 _*_
import math
import numpy as np
import scipy as sp
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

'''
功能：逻辑回归的简单实现
日期； 2017.7.16
'''


def sigmoid(p):
    return 1.0 / (1.0 + math.exp(-p))


def pred_lr(x):
    p = w_0
    for (feat, val) in x:
        p += w[feat] * float(val)
    p = sigmoid(p)
    return p


# 更新权重
def update_w(y, p, x):
    global w_0, w
    d = y - p
    w_0 = w_0 + learning_rate * d
    for (feat, value) in x:
        w[feat] = w[feat] + learning_rate * d * float(value)


def one_data_y_x(line):
    s = line.strip().replace(':', ' ').split(' ')
    y = int(s[0])
    x = []
    for i in range(1, len(s), 2):
        val = 1
        if not True:
            val = float(s[i + 1])
        x.append((int(s[i]), val))
    return (y, x)


# 训练
def train(f1):
    fi = open(f1, 'r')
    next(fi)
    y_true = []
    y_pred = []
    for t, line in enumerate(fi):
        data = one_data_y_x(line)
        y = data[0]
        x = data[1]
        p = pred_lr(x)
        update_w(y, p, x)
        y_true.append(y)
        y_pred.append(p)
    return y_true, y_pred


# 预测
def eval(f2):
    fi = open(f2, 'r')
    next(fi)
    y_true = []
    y_pred = []
    for t, line in enumerate(fi):
        data = one_data_y_x(line)
        y = data[0]
        x = data[1]
        p = pred_lr(x)
        y_true.append(y)
        y_pred.append(p)
    return y_true, y_pred


def get_cnt(f):
    features = set()
    fi = open(f, 'r')
    for line in fi:
        features.add(line.replace('\n', '').split('\t')[1])
    fi.close()
    feature_cnt = len(features)
    return feature_cnt


if __name__=='__main__':
    f1 = '../../output/fm/train.txt'
    f2 = '../../output/fm/test.txt'
    f3 = '../../output/fm/featindex.txt'

    feature_num = get_cnt(f3)
    print('feature number:', feature_num)

    w = np.zeros(feature_num)  # 初始化权重
    w_0 = 0  # 初始化w_0
    learning_rate = 0.1  # 学习率
    train_rounds = 15  # 训练轮数

    # 训练
    print('training...')
    for round in range(1, train_rounds+1):
        y_true, y_pred = train(f1)
        # auc = roc_auc_score(y_true, y_pred)
        # logloss = log_loss(y_true, y_pred)
        # print('round:{0}\tauc:{1}\tlogloss:{2}'.format(round, auc, logloss))

        # 验证
        # print('testing...')
        y_true, y_pred = eval(f2)
        auc = roc_auc_score(y_true, y_pred)
        logloss = log_loss(y_true, y_pred)
        print('round:{0}\tauc:{1}\tlogloss:{2}'.format(round, auc, logloss))

