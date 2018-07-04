# _*_ coding: utf-8 _*_

import collections
from csv import DictReader
from datetime import datetime

train_path = '../../data/train.csv'
test_path = '../../data/test.csv'
train_fm = '../../output/fm/train.fm'
test_fm = '../../output/fm/test.fm'
vali_path = '../../output/alphaFM/validation.csv'


field = ['hour', 'useragent', 'IP', 'region', 'city', 'adexchange', 'domain', 'slotid', 'slotwidth',
         'slotheight', 'slotvisibility', 'slotformat', 'creative', 'keypage', 'advertiser', 'usertag']

table = collections.defaultdict(lambda: 0)


def getIndices(key):
    indices = table.get(key)
    if indices is None:
        indices = len(table)
        table[key] = indices
    return indices


feature_indices = set()
with open(train_fm, 'w') as outfile:
    for e, row in enumerate(DictReader(open(train_path)), start=1):
        features = []
        for k, v in row.items():
            if k in field:
                if len(v) > 0:
                    kv = k + '_' + v
                    features.append('{0}:1'.format(getIndices(kv)))
                    feature_indices.add(kv + '\t' + str(getIndices(kv)))

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
                    kv = k + '_' + v
                    # if kv + '\t' + str(getIndices(kv)) in feature_indices:
                    features.append('{0}:1'.format(getIndices(kv)))
                    feature_indices.add(kv + '\t' + str(getIndices(kv)))
        if t % 100000 == 0:
            print(datetime.now(), 'creating validation data and test.fm...', t)
            # break
        f1.write('{0} {1}\n'.format(row['click'], ' '.join('{0}'.format(val) for val in features)))
        f2.write(str(t) + ',' + row['click'] + '\n')

