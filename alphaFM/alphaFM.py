import subprocess


'''
fm_train的参数：

-m <model_path>: 设置模型文件的输出路径。
-dim <k0,k1,k2>: k0为1表示使用偏置w0参数，0表示不使用；k1为1表示使用w参数，为0表示不使用；k2为v的维度，可以是0。	default:1,1,8
-init_stdev <stdev>: v的初始化使用均值为0的高斯分布，stdev为标准差。	default:0.1
-w_alpha <w_alpha>: w0和w的FTRL超参数alpha。	default:0.05
-w_beta <w_beta>: w0和w的FTRL超参数beta。	default:1.0
-w_l1 <w_L1_reg>: w0和w的L1正则。	default:0.1
-w_l2 <w_L2_reg>: w0和w的L2正则。	default:5.0
-v_alpha <v_alpha>: v的FTRL超参数alpha。	default:0.05
-v_beta <v_beta>: v的FTRL超参数beta。	default:1.0
-v_l1 <v_L1_reg>: v的L1正则。	default:0.1
-v_l2 <v_L2_reg>: v的L2正则。	default:5.0
-core <threads_num>: 计算线程数。	default:1
-im <initial_model_path>: 上次模型的路径，用于初始化模型参数。如果是第一次训练则不用设置此参数。
-fvs <force_v_sparse>: 为了获得更好的稀疏解。当fvs值为1, 则训练中每当wi = 0，即令vi = 0；当fvs为0时关闭此功能。 default:0

fm_predict的参数：

-m <model_path>: 模型文件路径。
-dim <factor_num>: v的维度。	default:8
-core <threads_num>: 计算线程数。	default:1
-out <predict_path>: 输出文件路径。
'''

data_path = '../../output/fm'
out_path = '../../output/alphaFM'

# 训练
print('training...')
cmd = 'cat {data_path}/train.fm | ./fm_train -core 8 -dim 1,1,8 -m {out_path}/fm_model.txt'.format(data_path=data_path, out_path=out_path)
subprocess.call(cmd, shell=True)

# 预测
print('predicting...')
cmd = 'cat {data_path}/test.fm | ./fm_predict -core 8 -dim 8 -m {out_path}/fm_model.txt -out {out_path}/fm_pre.txt'\
    .format(data_path=data_path, out_path=out_path)
subprocess.call(cmd, shell=True)


with open(out_path + '/submission.csv', 'w') as fo:
    fo.write('id,prob\n')
    for t, line in enumerate(open(out_path + '/fm_pre.txt'), start=1):

        fo.write('{0},{1}\n'.format(t, float(line.replace('\n', '').split(' ')[1])))