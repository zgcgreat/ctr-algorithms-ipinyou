import random

random.seed(888)

path = '../../data/'

# 负采样后达到的点击率
CLICK_RATE = 0.001  # 0.256


def update_pred(p):
    sample_rate = 0.001
    q = p / (p + (1 - p) / sample_rate)
    return q


def getSampleRate():
    click = 2098  # 原始数据中的点击
    total = 2635563 # 原始数据中的曝光总数
    # click *= 0.93
    rate = click / CLICK_RATE / (total - click)
    # 原始数据中的点击和曝光总数
    print('clicks: {0} impressions: {1}\n'.format(click, total))
    # 一个负例被选中的概率，每多少个负例被选中一次
    # print('sample rate: {0} sample num: {1}'.format(rate, 1 / rate))
    return rate


# sample = getSampleRate()
# print(sample)

sample_rate = getSampleRate()
with open(path + 'train_sample.csv', 'w') as fo:
    fi = open(path + 'train.csv')
    header = next(fi)
    fo.write(header)
    p = 0
    n = 0
    nn = 0
    c = 0
    for t, line in enumerate(fi, start=1):
        c += 1
        label = line.split(',')[0]
        if int(label) == 0:
            n += 1
            if random.randint(0, 10000) <= 10000 * sample_rate:  # down sample
                # if random.randint(1, sample) == random.randint(1, sample):
                fo.write(line)
                nn += 1
        else:
            p += 1
            fo.write(line)

        if t % 1000000 == 0:
            print(t)
    fi.close()

print(c, n, p + nn, p, nn, (p + nn) / c, nn / n, p / nn)
