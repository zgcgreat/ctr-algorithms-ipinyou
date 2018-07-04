import subprocess

data_path = '../../output/fm'
out_path = '../../output/alphaPLM'

# 训练
print('training...')
cmd = 'cat {data_path}/train.fm | ./plm_train -m {out_path}/model.txt -u_bias 1 -w_bias 1 -u_l1 0.001 -u_l2 0.1 -w_l1 0.001 ' \
      '-w_l2 0.1 -core 8 -piece_num 4 -u_stdev 1 -w_stdev 1 -u_alpha 10 -w_alpha 10'.format(data_path=data_path, out_path=out_path)
subprocess.call(cmd, shell=True)

# 预测
print('predicting...')
cmd = 'cat {data_path}/test.fm | ./plm_predict -core 8 -piece_num 4 -m {out_path}/model.txt -out {out_path}/plm_pre.txt'\
    .format(data_path=data_path, out_path=out_path)
subprocess.call(cmd, shell=True)


with open(out_path + '/submission.csv', 'w') as fo:
    fo.write('id,prob\n')
    for t, line in enumerate(open(out_path + '/plm_pre.txt'), start=1):

        fo.write('{0},{1}\n'.format(t, float(line.replace('\n', '').split(' ')[1])))