import subprocess

NR_THREAD = 8

path = '../../output/ffm/'


# 训练
cmd = './ffm-train -l 0.00002 -k 10 -t 50 -r 0.1 -s {nr_thread} -p {save}test.ffm {save}train.ffm ' \
      '{save}model'.format(nr_thread=NR_THREAD, save=path)
subprocess.call(cmd, shell=True)
# 预测
cmd = './ffm-predict {save}test.ffm {save}model {save}test.out'.format(save=path)
subprocess.call(cmd, shell=True)

with open(path + 'submission.csv', 'w') as f:
    f.write('id,prob\n')
    for i, row in enumerate(open(path + 'test.out')):
        f.write('{0},{1}'.format(i, row))


# 删除中间文件
# cmd = 'rm {path}model {path}test.out'.format(path=path)
# subprocess.call(cmd, shell=True)

