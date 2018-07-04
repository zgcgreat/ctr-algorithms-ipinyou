# _*_ coding: utf-8 _*_

import collections
import operator
from csv import DictReader
from datetime import datetime

train_path = '../../data/train.csv'
test_path = '../../data/test.csv'
train_fm = '../../output/fm/train.fm'
test_fm = '../../output/fm/test.fm'
vali_path = '../../output/fm/validation.csv'
feature_index = '../../output/fm/feat_index.txt'

field = ['hour', 'useragent', 'IP', 'region', 'city', 'adexchange', 'domain', 'slotid', 'slotwidth',
         'slotheight', 'slotvisibility', 'slotformat', 'creative', 'keypage', 'usertag']


table = collections.defaultdict(lambda: 0)


# 为特征名建立编号, filed
def field_index(x):
    index = field.index(x)
    return index


def getIndices(key):
    indices = table.get(key)
    if indices is None:
        indices = len(table)
        table[key] = indices
    return indices

for f in field:
    getIndices(str(field_index(f))+':other')


with open(train_fm, 'w') as outfile:
    for e, row in enumerate(DictReader(open(train_path)), start=1):
        features = []
        for k, v in row.items():
            if k in field:
                if len(v) > 0:
                    kv = str(field_index(k)) + ':' + v
                    features.append('{0}:1'.format(getIndices(kv)))

        if e % 100000 == 0:
            print(datetime.now(), 'creating train.fm...', e)
            # break
        outfile.write('{0} {1}\n'.format(row['click'], ' '.join('{0}'.format(val) for val in features)))

with open(test_fm, 'w') as f1, open(vali_path, 'w') as f2:
    f2.write('id,label'+'\n')
    for t, row in enumerate(DictReader(open(test_path)), start=1):
        features = []
        for k, v in row.items():
            if k in field:
                if len(v) > 0:
                    kv = str(field_index(k)) + ':' + v
                    if kv in table.keys():
                        features.append('{0}:1'.format(getIndices(kv)))
                    else:
                        kv = str(field_index(k)) + ':' + 'other'
                        features.append('{0}:1'.format(getIndices(kv)))

        if t % 100000 == 0:
            print(datetime.now(), 'creating validation data and test.fm...', t)
            # break
        f1.write('{0} {1}\n'.format(row['click'], ' '.join('{0}'.format(val) for val in features)))
        f2.write(str(t) + ',' + row['click'] + '\n')

featvalue = sorted(table.items(), key=operator.itemgetter(1))
fo = open(feature_index, 'w')
for fv in featvalue:
    fo.write(fv[0] + '\t' + str(fv[1]) + '\n')
fo.close()
