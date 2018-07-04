# _*_ coding: utf-8 _*_

import math
import sys
import subprocess

data_path = '../../output/fm/'
result_path = '../../output/fm/'

cmd = './libFM -task r -train {train} -test {test} -out {out} -method mcmc -learn_rate 0.1 -dim \'1,1,8\' -iter 50 '\
      '-validation {test}'.format(train=result_path + 'train.txt', test=result_path + 'test.txt', out=result_path + 'preds.txt')
subprocess.call(cmd, shell=True, stdout=sys.stdout)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


with open(result_path + 'submission.csv', 'w') as outfile:
    outfile.write('id,prob\n')
    for t, line in enumerate(open(result_path + 'preds.txt'), start=1):
        # outfile.write('{0},{1}\n'.format(t, sigmoid(float(line.rstrip()))))
        outfile.write('{0},{1}\n'.format(t, float(line.rstrip())))

# cmd = 'rm {0}preds.txt'.format(result_path)
# subprocess.call(cmd, shell=True)


