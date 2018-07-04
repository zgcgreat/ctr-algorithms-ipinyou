# _*_ coding: utf-8 _*_

import collections
from csv import DictReader
from datetime import datetime

train_path = '../../data/train.csv'
test_path = '../../data/test.csv'
train_fm = '../../output/cnn-keras/train.cnn'
test_fm = '../../output/cnn-keras/test.cnn'
vali_path = '../../output/cnn-keras/validation.csv'
feature_index = '../../output/cnn-keras/feature_index.txt'

field = ['hour', 'useragent', 'IP', 'region', 'city', 'adexchange', 'domain', 'slotid', 'slotwidth',
         'slotheight', 'slotvisibility', 'slotformat', 'creative', 'keypage', 'usertag']


table = collections.defaultdict(lambda: 0)


# 读取频繁特征
def read_frequent_feats(threshold):
    # frequent_feats = collections.defaultdict(lambda: [0, 0, 0])
    frequent_feats = set()
    # fc.trav.t10.txt为出现频率超过10的表
    for row in DictReader(open('../../output/fc.csv')):
        if int(row['Total']) < threshold:
            continue
        # frequent_feats[row['Field'] + '-' + row['Value']][0] = row['Ratio']
        frequent_feats.add(row['Field'] + '-' + row['Value'])
        # frequent_feats[row['Field'] + '-' + row['Value']][2] = row['Pos']
    return frequent_feats


threshold = 10

frequent_feats = read_frequent_feats(threshold)
print(len(frequent_feats))


def getIndices(key):
    indices = table.get(key)
    if indices is None:
        indices = len(table)
        table[key] = indices
    return indices

feature_indices = set()
with open(train_fm, 'w') as outfile, open(feature_index, 'w') as fo:
    for e, row in enumerate(DictReader(open(train_path)), start=1):
        features = []
        for k, v in row.items():
            if k in field:
                if len(v) > 0:
                    kv = k + '-' + v
                    if kv not in frequent_feats:
                        kv = k + '-other'

                    features.append('{0}'.format(getIndices(kv)))
                    # if kv + '\t' + str(getIndices(kv)) not in feature_indices:
                    #     fo.write(kv + '\t' + str(getIndices(kv)) + '\n')
                    feature_indices.add(kv + '\t' + str(getIndices(kv)))

        if e % 100000 == 0:
            print(datetime.now(), 'creating train.ffm...', e)
            # break
        outfile.write('{0} {1}\n'.format(row['click'], ' '.join('{0}'.format(val) for val in features)))

with open(test_fm, 'w') as f1, open(vali_path, 'w') as f2:
    f2.write('id,label'+'\n')
    for t, row in enumerate(DictReader(open(test_path)), start=1):
        features = []
        for k, v in row.items():
            if k in field:
                if len(v) > 0:
                    kv = k + '-' + v
                    if kv not in frequent_feats:
                        kv = k + '-other'
                    # if kv + '\t' + str(getIndices(kv)) in feature_indices:
                    features.append('{0}'.format(getIndices(kv)))
        if t % 100000 == 0:
            print(datetime.now(), 'creating validation data and test.ffm...', t)
            # break
        f1.write('{0} {1}\n'.format(row['click'], ' '.join('{0}'.format(val) for val in features)))
        f2.write(str(t) + ',' + row['click'] + '\n')

fo = open(feature_index, 'w')
fo.write(str(len(table)))
fo.close()
