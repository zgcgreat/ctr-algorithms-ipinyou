
import os
import sys
import math
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from time import time
import argparse
import LoadData as DATA
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


# tf.random_seed = 2017
# np.random_seed = 2017


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepCross.")
    parser.add_argument('--path', nargs='?', default='../../output/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='pnn',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='Pre-train flag. 0: train from scratch; 1: load from pretrain file')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=10,
                        help='Number of hidden factors.')
    parser.add_argument('--deep_layers', nargs='?', default='[1024, 1024]',
                        help="Size of each layer.")
    parser.add_argument('--cross_layer_num', nargs='?', default='6',
                        help="number of cross layers.")
    parser.add_argument('--keep_prob', nargs='?', default='[0.5, 0.5, 1, 1, 1]',
                        help='Keep probability (i.e., 1-dropout_ratio) for each deep layer and the Bi-Interaction layer. 1: no dropout. Note that the last index is for the Bi-Interaction layer.')
    parser.add_argument('--lamda', type=float, default=0.00,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=1,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='log_loss',
                        help='Specify a loss type (square_loss or log_loss).')
    parser.add_argument('--optimizer', nargs='?', default='gd',
                        help='Specify an optimizer type (adam, adagrad, gd, ftrl, adaldeta, padagrad, rmsprop, pgd, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=0,
                        help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--activation', nargs='?', default='relu',
                        help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity, elu')
    parser.add_argument('--early_stop', type=int, default=1,
                        help='Whether to perform early stop (0 or 1)')
    return parser.parse_args()


class NeuralFM(BaseEstimator, TransformerMixin):
    def __init__(self, field_cnt, features_M, hidden_factor, deep_layers, cross_layer_num, loss_type, pretrain_flag, epoch, batch_size,
                 learning_rate,
                 lamda_bilinear,
                 keep_prob, optimizer_type, batch_norm, activation_function, verbose, early_stop, random_seed=2017):
        # bind params to class
        self.batch_size = batch_size
        self.hidden_factor = hidden_factor
        self.deep_layers = deep_layers
        self.cross_layer_num = cross_layer_num
        self.loss_type = loss_type
        self.pretrain_flag = pretrain_flag
        self.field_cnt = field_cnt
        self.features_M = features_M
        self.lamda_bilinear = lamda_bilinear
        self.epoch = epoch
        self.random_seed = random_seed
        self.keep_prob = np.array(keep_prob)
        self.no_dropout = np.array([1 for i in range(len(keep_prob))])
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.verbose = verbose
        self.activation_function = activation_function
        self.early_stop = early_stop
        # performance of each epoch
        self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []

        self.nn_input_dim = self.field_cnt * self.hidden_factor
        self.cross_dim = self.field_cnt*self.hidden_factor

        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.train_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1
            self.dropout_keep = tf.placeholder(tf.float32, shape=[None])
            self.train_phase = tf.placeholder(tf.bool)

            # Variables.
            self.weights = self._initialize_weights()

            # Model.

            # get the summed up embeddings of features.
            nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features)

            w_fm_nn_input = tf.reshape(nonzero_embeddings, [-1, self.field_cnt * self.hidden_factor])

            self.FM = w_fm_nn_input

            print('w_fm_nn_input.shape:', w_fm_nn_input.shape)

            # ________ Deep Layers __________
            for i in range(0, len(self.deep_layers)):
                self.FM = tf.add(tf.matmul(self.FM, self.weights['deep_layer_%d' % i]),
                                 self.weights['deep_bias_%d' % i])  # None * layer[i] * 1
                if self.batch_norm:
                    self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase,
                                                    scope_bn='deep_bn_%d' % i)  # None * layer[i] * 1
                self.FM = self.activation_function(self.FM)
                self.FM = tf.nn.dropout(self.FM, self.dropout_keep[i])  # dropout at each Deep layer
            self.deep_net_out = self.FM

            # ________ cross Layers __________
            x_0 = tf.reshape(nonzero_embeddings, [-1, self.field_cnt*self.hidden_factor, 1])
            x_1 = x_0
            for i in range(0, self.cross_layer_num):
                dot = tf.matmul(x_0, x_1, transpose_b=True)
                dot = tf.tensordot(dot, self.weights['cross_layer_%d' % i], 1)
                x_1 = tf.add(dot, self.weights['cross_bias_%d' % i]) + x_1

            self.cross_net_out = tf.reshape(x_1, [-1, self.field_cnt*self.hidden_factor])

            x_stack = tf.concat([self.deep_net_out, self.cross_net_out], 1)

            self.out = tf.matmul(x_stack, self.weights['prediction'])  # None * 1
            print(self.out)

            self.prob = tf.sigmoid(self.out)

            # Compute the loss.
            if self.loss_type == 'square_loss':
                if self.lamda_bilinear > 0:
                    self.loss = tf.nn.l2_loss(
                        tf.subtract(self.train_labels, self.out)) + tf.contrib.layers.l2_regularizer(
                        self.lamda_bilinear)(self.weights['feature_embeddings'])  # regulizer
                else:
                    self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out))
            elif self.loss_type == 'log_loss':
                # self.out = tf.sigmoid(self.out)
                if self.lamda_bilinear > 0:
                    # self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, epsilon=1e-07,
                    #                                        scope=None) + tf.contrib.layers.l2_regularizer(
                    #     self.lamda_bilinear)(self.weights['feature_embeddings'])  # regulizer
                    self.loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.train_labels) \
                        + tf.contrib.layers.l2_regularizer(
                            self.lamda_bilinear)(self.weights['feature_embeddings']))  # regulizer

                    # self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.train_labels) \
                    #            + tf.contrib.layers.l2_regularizer(
                    #    self.lamda_bilinear)(self.weights['feature_embeddings']) # regulizer

                else:
                    # self.loss = tf.contrib.losses.log_loss(self.prob, self.train_labels, epsilon=1e-07, scope=None)
                    self.loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.train_labels))

            # Optimizer.
            if self.optimizer_type == 'adam':
                self.optimizer = tf.train.AdamOptimizer(beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'gd':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'ftrl':
                self.optimizer = tf.train.FtrlOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'adaldeta':
                self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)
            elif self.optimizer_type == 'padagrad':
                self.optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'pgd':
                self.optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                    self.loss)
            elif self.optimizer_type == 'rmsprop':
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            # elif self.optimizer_type == 'nadam':
            #     self.optimizer = tf.train.Na(learning_rate=self.learning_rate).minimize(self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()  # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)

    def _initialize_weights(self):
        all_weights = dict()
        if self.pretrain_flag > 0:  # with pretrain
            pretrain_file = '../pretrain/%s_%d/%s_%d' % (
                args.dataset, args.hidden_factor, args.dataset, args.hidden_factor)
            weight_saver = tf.train.import_meta_graph(pretrain_file + '.meta')
            pretrain_graph = tf.get_default_graph()
            feature_embeddings = pretrain_graph.get_tensor_by_name('feature_embeddings:0')
            feature_bias = pretrain_graph.get_tensor_by_name('feature_bias:0')
            bias = pretrain_graph.get_tensor_by_name('bias:0')
            with tf.Session() as sess:
                weight_saver.restore(sess, pretrain_file)
                fe, fb, b = sess.run([feature_embeddings, feature_bias, bias])
            all_weights['feature_embeddings'] = tf.Variable(fe, dtype=tf.float32)
            all_weights['feature_bias'] = tf.Variable(fb, dtype=tf.float32)
            all_weights['bias'] = tf.Variable(b, dtype=tf.float32)
        else:  # without pretrain
            all_weights['linear_weights'] = tf.Variable(
                tf.random_normal([self.features_M, 1], 0.0, 0.001),
                name='linear_weights')  # features_M * K
            #
            all_weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([self.features_M, self.hidden_factor], 0.0, 0.001),
                name='feature_embeddings')  # features_M * K
            all_weights['bias_0'] = tf.Variable(tf.random_normal([1, self.deep_layers[0]], 0.0, 0.001))

            # maxval = np.sqrt(6. / np.sum([self.nn_input_dim, self.deep_layers[0]]))
            # minval = -maxval
            # all_weights['feature_embeddings'] = tf.Variable(
            #     tf.random_uniform([self.features_M, self.hidden_factor], minval=minval,
            #                       maxval=maxval, dtype=tf.float32), name='feature_embeddings', dtype=tf.float32)
            # all_weights['bias_0'] = tf.Variable(tf.random_uniform([1, self.deep_layers[0]], minval=minval,
            #                                                       maxval=maxval, dtype=tf.float32),
            #                                     dtype=tf.float32)  # 1 * layers[0]

            # all_weights['feature_bias'] = tf.Variable(tf.random_uniform([self.features_M, 1], 0.0, 0.0),
            #                                           name='feature_bias')  # features_M * 1
            all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1


        # deep layers
        deep_layer_num = len(self.deep_layers)
        if deep_layer_num > 0:
            maxval = np.sqrt(6. / np.sum([self.nn_input_dim, self.deep_layers[0]]))
            minval = -maxval
            all_weights['deep_layer_0'] = tf.Variable(
                tf.random_uniform([self.nn_input_dim, self.deep_layers[0]], minval=minval,
                                  maxval=maxval, dtype=tf.float32), dtype=tf.float32)
            all_weights['deep_bias_0'] = tf.Variable(tf.random_uniform([1, self.deep_layers[0]], minval=minval,
                                                                       maxval=maxval, dtype=tf.float32),
                                                     dtype=tf.float32)  # 1 * layers[0]

            for i in range(1, deep_layer_num):
                glorot = np.sqrt(6.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
                all_weights['deep_layer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                all_weights['deep_bias_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])), dtype=np.float32)  # 1 * layer[i]

        if self.cross_layer_num > 0:
            maxval = np.sqrt(6. / np.sum([self.nn_input_dim, self.deep_layers[0]]))
            minval = -maxval
            all_weights['cross_layer_0'] = tf.Variable(
                tf.random_uniform([self.cross_dim, 1], minval=minval,
                                  maxval=maxval, dtype=tf.float32), dtype=tf.float32)
            all_weights['cross_bias_0'] = tf.Variable(tf.random_uniform([self.cross_dim, 1], minval=minval,
                                                                        maxval=maxval, dtype=tf.float32),
                                                      dtype=tf.float32)

            for i in range(1, self.cross_layer_num):
                glorot = np.sqrt(6.0 / (2 * self.cross_dim))

                all_weights['cross_layer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.cross_dim, 1)),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                all_weights['cross_bias_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.cross_dim, 1)),
                    dtype=np.float32)  # 1 * layer[i]

        if deep_layer_num > 0 and self.cross_layer_num > 0:
            # prediction layer
            glorot = np.sqrt(6.0 / (self.deep_layers[-1] + 1))
            all_weights['prediction'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot,
                                 size=(self.deep_layers[-1] + self.cross_dim, 1)),
                dtype=np.float32)  # layers[-1] * 1
        else:
            all_weights['prediction'] = tf.Variable(
                np.ones((self.hidden_factor, 1), dtype=np.float32))  # hidden_factor * 1
        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.train_features: data['X'], self.train_labels: data['Y'], self.dropout_keep: self.keep_prob,
                     self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, data, batch_size):  # generate a random block of training data
        start_index = np.random.randint(0, len(data['Y']) - batch_size)
        X, Y = [], []
        # forward get sample
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i + 1
            else:
                break
        # backward get sample
        i = start_index
        while len(X) < batch_size and i >= 0:
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i - 1
            else:
                break
        return {'X': X, 'Y': Y}

    def get_next_block_from_data(self, data, start_index=0, batch_size=-1):  # generate a random block of training data
        # start_index = np.random.randint(0, len(data['Y']) - batch_size)
        X, Y = [], []
        # forward get sample
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i + 1
            else:
                break

        return {'X': X, 'Y': Y}

    def shuffle_in_unison_scary(self, a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def train(self, Train_data, Validation_data):  # fit a dataset
        # Check Init performance
        if self.verbose > 0:
            t2 = time()
            init_train, train_auc = self.evaluate(Train_data)
            init_valid, valid_auc = self.evaluate(Validation_data)

            print("Init: \t train_loss=%.8f\tvalid_loss=%.8f\ttrain_auc=%.8f\tvalid_auc=%.8f\t[%.1f s]" % (
                init_train, init_valid, train_auc, valid_auc, time() - t2))

        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Train_data['X'], Train_data['Y'])
            total_batch = int(len(Train_data['Y']) / self.batch_size)
            for i in range(total_batch):
                # generate a batch
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size)
                # batch_xs = self.get_next_block_from_data(Train_data, self.batch_size * i, self.batch_size)
                # Fit training
                self.partial_fit(batch_xs)
            t2 = time()

            # output validation
            train_result, train_auc = self.evaluate(Train_data)
            valid_result, valid_auc = self.evaluate(Validation_data)

            self.train_rmse.append(train_result)
            self.valid_rmse.append(valid_result)

            if self.verbose > 0 and epoch % self.verbose == 0:
                print("Epoch %d [%.1f s]\ttrain_loss=%.8f\tvalid_loss=%.8f\ttrain_auc=%.8f\tvalid_auc=%.8f\t[%.1f s]"
                      % (epoch + 1, t2 - t1, train_result, valid_result, train_auc, valid_auc, time() - t2))

            if self.early_stop > 0 and self.eva_termination(self.valid_rmse):
                # print "Early stop at %d based on validation result." %(epoch+1)
                break

    def eva_termination(self, valid):
        if self.loss_type == 'square_loss':
            if len(valid) > 5:
                if valid[-1] > valid[-2] and valid[-2] > valid[-3] and valid[-3] > valid[-4] and valid[-4] > valid[-5]:
                    return True
        else:
            if len(valid) > 5:
                if valid[1] < valid[2] and valid[2] < valid[3] and valid[3] < valid[-4] and valid[-4] < valid[-5]:
                    return True
        return False

    def evaluate(self, data):  # evaluate the results for an input set
        num_example = len(data['Y'])
        size = 10000
        total_batch = int(num_example / size)
        preds = []
        for i in range(total_batch + 1):
            batch_xs = self.get_next_block_from_data(data, size * i, size)
            feed_dict = {self.train_features: batch_xs['X'], self.train_labels: [y for y in batch_xs['Y']],
                         self.dropout_keep: self.no_dropout, self.train_phase: False}
            predictions = self.sess.run((self.prob), feed_dict=feed_dict)
            preds.extend(list(predictions))
        # print(preds[:5])
        y_pred = np.reshape(preds, (num_example,))
        y_true = np.reshape(data['Y'], (num_example,))
        auc = roc_auc_score(y_true, y_pred)
        if self.loss_type == 'square_loss':
            predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
            predictions_bounded = np.minimum(predictions_bounded,
                                             np.ones(num_example) * max(y_true))  # bound the higher values
            RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
            return RMSE, auc
        elif self.loss_type == 'log_loss':
            logloss = log_loss(y_true, y_pred)
            return logloss, auc

if __name__ == '__main__':
    # Data loading
    args = parse_args()
    data = DATA.LoadData(args.path, args.dataset, args.loss_type)

    if args.verbose > 0:
        print(
            "DeepCross: dataset=%s, hidden_factor=%d, dropout_keep=%s, layers=%s, loss_type=%s, pretrain=%d, #epoch=%d, batch=%d, lr=%.4f, lambda=%.4f, optimizer=%s, batch_norm=%d, activation=%s, early_stop=%d"
            % (args.dataset, args.hidden_factor, args.keep_prob, args.deep_layers, args.loss_type, args.pretrain, args.epoch,
               args.batch_size, args.lr, args.lamda, args.optimizer, args.batch_norm, args.activation, args.early_stop))
    activation_function = tf.nn.relu
    if args.activation == 'sigmoid':
        activation_function = tf.nn.sigmoid
    elif args.activation == 'tanh':
        activation_function = tf.nn.tanh
    elif args.activation == 'identity':
        activation_function = tf.identity
    elif activation_function == 'elu':
        activation_function = tf.nn.elu

    # Training
    t1 = time()
    model = NeuralFM(data.field_cnt, data.features_M, args.hidden_factor, eval(args.deep_layers), eval(args.cross_layer_num), args.loss_type,
                     args.pretrain, args.epoch,
                     args.batch_size, args.lr, args.lamda, eval(args.keep_prob), args.optimizer, args.batch_norm,
                     activation_function, args.verbose, args.early_stop)
    model.train(data.Train_data, data.Validation_data)

    # model.predict(data.Test_data)

    # Find the best validation result across iterations
    best_valid_score = 0
    if args.loss_type == 'square_loss':
        best_valid_score = min(model.valid_rmse)
    elif args.loss_type == 'log_loss':
        best_valid_score = min(model.valid_rmse)
    best_epoch = model.valid_rmse.index(best_valid_score)
    print("Best Iter(validation)= %d\t train = %.8f, valid = %.8f [%.1f s]"
          % (best_epoch + 1, model.train_rmse[best_epoch], model.valid_rmse[best_epoch], time() - t1))
