import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys
import utils


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x
                      )


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


# for test
def sample_y(m, n, ind):
    y = np.zeros([m, n])
    for i in range(m):
        y[i, ind] = 1
    return y


def concat(z, y):
    return tf.concat([z, y], 1)


###############################################  mlp #############################################
class G_mlp(object):
    def __init__(self):
        self.name = 'G_mlp'

    def __call__(self, z):
        with tf.variable_scope(self.name) as scope:
            g = tcl.fully_connected(z, 4 * 4 * 512, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            g = tcl.fully_connected(z, 64, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            g = tcl.fully_connected(z, 64, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            g = tcl.fully_connected(z, 28 * 28 * 1, activation_fn=tf.nn.tanh, normalizer_fn=tcl.batch_norm)
            g = tf.reshape(g, tf.stack([tf.shape(z)[0], 28, 28, 1]))
            return g

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class D_mlp(object):
    def __init__(self):
        self.name = "D_mlp"

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            size = 64
            x = tcl.fully_connected(tf.contrib.layers.flatten(x), 64, activation_fn=tf.nn.relu)
            x = tcl.fully_connected(x, 64,
                                    activation_fn=tf.nn.relu)
            x = tcl.fully_connected(x, 64,
                                    activation_fn=tf.nn.relu)
            logit = tcl.fully_connected(x, 1, activation_fn=None)

        return logit

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class G_mlp_mnist(object):
    def __init__(self):
        self.name = "G_mlp_mnist"
        self.X_dim = 20711

    def __call__(self, z):
        with tf.variable_scope(self.name) as vs:
            g = tcl.fully_connected(z, 128, activation_fn=tf.nn.relu,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.fully_connected(g, self.X_dim, activation_fn=tf.nn.sigmoid,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))
        return g

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class D_mlp_mnist():
    def __init__(self):
        self.name = "D_mlp_mnist"

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            shared = tcl.fully_connected(x, 128, activation_fn=tf.nn.relu,
                                         weights_initializer=tf.random_normal_initializer(0, 0.02))
            d = tcl.fully_connected(shared, 1, activation_fn=None,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))

            q = tcl.fully_connected(shared, 2, activation_fn=None,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))  # 10 classes

        return d, q

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class CGAN():
    def __init__(self, generator, discriminator, data):
        self.generator = generator
        self.discriminator = discriminator
        self.data = data

        # data
        self.z_dim = 100
        self.y_dim = 2  # condition
        self.X_dim = 20711
        print(self.X_dim)

        # self.X = tf.sparse_placeholder(tf.float32, shape=[None, self.X_dim])
        self.X = tf.sparse_placeholder(tf.float32)
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])
        print(self.X)
        print(self.z)
        print(self.y)

        self.x = tf.sparse_tensor_to_dense(self.X)
        print(self.x)
        self.x = tf.reshape(self.x, [-1, self.X_dim])
        # self.X = tf.transpose(tf.reshape(self.X, [-1, 1, self.X_dim]), [0, 2, 1])
        print(self.x)

        # nets
        self.G_sample = self.generator(concat(self.z, self.y))

        self.D_real, _ = self.discriminator(concat(self.x, self.y))
        self.D_fake, _ = self.discriminator(concat(self.G_sample, self.y), reuse=True)

        # loss
        self.D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(
            self.D_real))) + tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))

        # solver
        self.D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.discriminator.vars)
        self.G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.generator.vars)

        # for var in self.discriminator.vars:
        #     print(var.name)

        self.saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, sample_dir, ckpt_dir='ckpt', training_epoches=1000000, batch_size=512):
        fig_count = 0
        self.sess.run(tf.global_variables_initializer())
        train_size = train_data[0].shape[0]
        # test_size = test_data[0].shape[0]

        for epoch in range(training_epoches):
            # update D
            for j in range(int(train_size / batch_size + 1)):
                X_b, y_b = utils.slice(train_data, j * batch_size, batch_size)
                from keras.utils import to_categorical
                y_b = to_categorical(y_b, 2)

                # X_b, y_b = self.data(batch_size)

                self.sess.run(
                    self.D_solver,
                    feed_dict={self.X: X_b, self.y: y_b, self.z: sample_z(len(y_b), self.z_dim)}

                )
                # update G
                k = 1
                for _ in range(k):
                    self.sess.run(
                        self.G_solver,
                        feed_dict={self.y: y_b, self.z: sample_z(len(y_b), self.z_dim)}
                    )

            # save img, model. print loss
            if epoch % 100 == 0 or epoch < 100:
                D_loss_curr = self.sess.run(
                    self.D_loss,
                    feed_dict={self.X: X_b, self.y: y_b, self.z: sample_z(len(y_b), self.z_dim)})
                G_loss_curr = self.sess.run(
                    self.G_loss,
                    feed_dict={self.y: y_b, self.z: sample_z(len(y_b), self.z_dim)})
                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(epoch, D_loss_curr, G_loss_curr))

                if epoch % 1000 == 0:
                    y_s = sample_y(16, self.y_dim, fig_count % 10)
                    samples = self.sess.run(self.G_sample, feed_dict={self.y: y_s, self.z: sample_z(16, self.z_dim)})
                    print(y_s.shape)
                    print(samples.shape)
                    # fig = self.data.data2fig(samples)
                    # plt.savefig('{}/{}_{}.png'.format(sample_dir, str(fig_count).zfill(3), str(fig_count % 10)),
                    #             bbox_inches='tight')
                    # fig_count += 1
                    # plt.close(fig)

                    # if epoch % 2000 == 0:
                    #	self.saver.save(self.sess, os.path.join(ckpt_dir, "cgan.ckpt"))


if __name__ == '__main__':
    print(10000 % 512)
    train_file = '../../output/fm/train.txt'
    test_file = '../../output/fm/test.txt'

    input_dim = utils.INPUT_DIM

    train_data = utils.read_data(train_file)
    # print(train_data[0])

    # save generated images
    sample_dir = 'Samples/mnist_cgan_mlp'
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # param
    generator = G_mlp_mnist()
    discriminator = D_mlp_mnist()

    # run
    cgan = CGAN(generator, discriminator, train_data)
    cgan.train(sample_dir)
