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
    for feat, val in enumerate(x):
        p += w[feat] * float(val)
    p = sigmoid(p)
    return p


# 更新权重
def update_w(y, p, x):
    global w_0, w
    d = y - p
    w_0 = w_0 + learning_rate * d
    for feat_index, value in enumerate(x):
        if value != 0:  # 值为0的地方没必要进行更新，因为权重不会变，节省时间
            w[feat_index] = w[feat_index] + learning_rate * d * float(value)


# 训练
def train():
    fi = open('../data/train.csv', 'r')
    next(fi)
    y_true = []
    y_pred = []
    for t, line in enumerate(fi):
        x = line.replace('\n', '').split(',')[2:]
        y = int(line.replace('\n', '').split(',')[1])
        p = pred_lr(x)
        update_w(y, p, x)
        y_true.append(y)
        y_pred.append(p)
    fi.close()
    return y_true, y_pred


# 预测
def eval():
    fi = open('../data/test.csv', 'r')
    next(fi)
    y_true = []
    y_pred = []
    for t, line in enumerate(fi):
        x = line.replace('\n', '').split(',')[2:]
        y = int(line.replace('\n', '').split(',')[1])
        p = pred_lr(x)
        y_true.append(y)
        y_pred.append(p)
    fi.close()
    return y_true, y_pred


if __name__=='__main__':
    feature_num = 9943  # one-hot编码后的特征数量
    w = np.zeros(feature_num)  # 初始化权重
    w_0 = 0  # 初始化w_0
    learning_rate = 0.1  # 学习率
    train_rounds = 10  # 训练轮数

    # 训练
    print('training...')
    for round in range(1, train_rounds+1):
        y_true, y_pred = train()
        auc = roc_auc_score(y_true, y_pred)
        logloss = log_loss(y_true, y_pred)
        print('round:{0}\tauc:{1}\tlogloss:{2}'.format(round, auc, logloss))


    # 验证
    print('testing...')
    y_true, y_pred = eval()
    auc = roc_auc_score(y_true, y_pred)
    logloss = log_loss(y_true, y_pred)
    print('auc:{0}\tlogloss:{1}'.format(auc, logloss))

