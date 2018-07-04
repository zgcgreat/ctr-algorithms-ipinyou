
path = '../../output/pnn/'

p = 0
n = 0

# fo = open(path + 'tr_up.txt', 'w')
#
# fi = open(path + 'train.txt', 'r')
# for line in fi:
#     fo.write(line)
#     clk = line.split(' ')[0]
#     if clk == '1':
#         for i in range(4):
#             fo.write(line)
#             p += 1
#     if clk == '0':
#         n += 1
# fi.close()
# fo.close()
# print(p, n, p / n)

import random

samplerate = 0.8
l = 1000
c = 0
for i in range(l):
    if random.randint(0, l) < l * samplerate:
        c += 1
print(c)