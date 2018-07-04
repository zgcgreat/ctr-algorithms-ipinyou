# _*_ coding: utf-8 _*_
import time
import math
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss


def sigmoid(p):
    return 1.0 / (1.0 + math.exp(-p))


def pred_lr(x):
    p = w_0
    for (feat, val) in x:
        p += w[feat] * val
    p = sigmoid(p)
    return p


def pred(x):
    p = w_0
    sum_1 = 0
    sum_2 = 0
    for (feat, val) in x:
        tmp = v[feat] * val
        sum_1 += tmp
        sum_2 += tmp * tmp
    p = np.sum(sum_1 * sum_1 - sum_2) / 2.0 + w_0
    for (feat, val) in x:
        p += w[feat] * val
    p = sigmoid(p)
    return (p, sum_1)


def update_w(y, p, x, vsum):
    global w_0
    d = y - p
    w_0 = w_0 * (1 - weight_decay) + learning_rate * d
    for (feat, val) in x:
        w[feat] = w[feat] * (1 - weight_decay) + learning_rate * d * val
    for (feat, val) in x:
        v[feat] = v[feat] * (1 - v_weight_decay) + learning_rate * d * (val * vsum - v[feat] * val * val)


def one_data_y_x(line):
    s = line.strip().replace(':', ' ').split(' ')
    y = int(s[0])
    x = []
    for i in range(1, len(s), 2):
        val = 1
        if not one_value:
            val = float(s[i + 1])
        x.append((int(s[i]), val))
    return (y, x)


def output_model(model_file):
    print('output model to ' + model_file)
    foo = open(model_file, 'w')
    foo.write('%.5f %d %d\n' % (w_0, feature_num, k))
    for i in range(feature_num):
        foo.write('%d %.5f' % (i, w[i]))
        for j in range(k):
            foo.write(' %.5f' % v[i][j])
        foo.write(' %s\n' % index_feature[str(i)])
    foo.close()


def load_model(model_file):
    global feature_num, k, w_0, w, v, index_feature, feature_index
    print('loading model from ' + model_file)
    fi = open(model_file, 'r')
    line_num = 0
    for line in fi:
        line_num += 1
        s = line.strip().split()
        if line_num == 1:
            w_0 = float(s[0])
            feature_num = int(s[1])
            k = int(s[2])
            v = np.zeros((feature_num, k))
            w = np.zeros(feature_num)
            index_feature = {}
            feature_index = {}
        else:
            i = int(s[0])
            w[i] = float(s[1])
            for j in range(2, 2 + k):
                v[i][j] = float(s[j])
            feature = s[2 + k]
            index_feature[i] = feature
            feature_index[feature] = i
    fi.close()


# 将预测结果写入文件
def pred_to_sub(y_pred):
    with open('../../output/fm/pred.csv', 'w') as fo:
        fo.write('id,prob\n')
        for t, prob in enumerate(y_pred, start=1):
            fo.write('{0},{1}\n'.format(t, prob))


# start here

data_path = '../../output/pnn/'
out_path = '../../output/pnn/'

f1 = '{0}/train.txt'.format(data_path)
f2 = '{0}/test.txt'.format(data_path)
f3 = '{0}/featindex.txt'.format(data_path)
f4 = '{0}/model_file.txt'.format(data_path)

# global setting
np.random.seed(10)
one_value = True
k = 10  # 隐含因子个数
learning_rate = 0.01  # 学习率
weight_decay = 1E-6
v_weight_decay = 1E-6
train_rounds = 20  # 训练轮数
buffer_num = 100000

# initialise
feature_index = {}
index_feature = {}
max_feature_index = 0


def get_cnt(f):
    features = set()
    fi = open(f, 'r')
    for line in fi:
        s = line.replace('\n', '').split('\t')
        features.add(s[1])
        index = s[1]
        index_feature[index] = s[0]
    fi.close()
    feature_cnt = len(features)
    return feature_cnt


# 读取特征个数，用于初始化
feature_num = get_cnt(f3)
print('feature number: ' + str(feature_num))

print('initialising')
init_weight = 0.05
v = (np.random.rand(feature_num, k) - 0.5) * init_weight
w = np.zeros(feature_num)
w_0 = 0

# train
best_auc = 0.
overfitting = False
print('training:')
print('round\tauc\t\tlogloss\t\ttime')
for round in range(1, train_rounds + 1):
    start_time = time.time()
    fi = open(f1, 'r')
    line_num = 0
    train_data = []
    while True:
        line = fi.readline().strip()
        if len(line) > 0:
            line_num = (line_num + 1) % buffer_num
            train_data.append(one_data_y_x(line))
        if line_num == 0 or len(line) == 0:
            for data in train_data:
                y = data[0]
                x = data[1]
                # train one data
                (p, vsum) = pred(x)
                update_w(y, p, x, vsum)  # 更新权值
            train_data = []
        if len(line) == 0:
            break
    fi.close()
    train_time = time.time() - start_time
    train_min = int(train_time / 60)
    train_sec = int(train_time % 60)

    # test for this round
    y_true = []
    y_pred = []
    fi = open(f2, 'r')
    for line in fi:
        data = one_data_y_x(line)
        clk = data[0]
        pclk = pred(data[1])[0]
        y_true.append(clk)
        y_pred.append(pclk)
    fi.close()
    auc = roc_auc_score(y_true, y_pred)
    logloss = log_loss(y_true, y_pred)
    print('%d\t%.8f\t%.8f\t%dm%ds' % (round, auc, logloss, train_min, train_sec))
    pred_to_sub(y_pred)

    if overfitting and auc < best_auc:
        output_model(f4)
        # pred_to_sub(y_pred)
        break  # stop training when overfitting two rounds already
    if auc > best_auc:
        best_auc = auc
        overfitting = False
    else:
        overfitting = True


        # print('v:', v)
        # print('w:', w)
        # print('w_0', w_0)
