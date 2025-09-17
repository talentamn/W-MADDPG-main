import tensorflow as tf
import numpy as np


class MemoryDNN:
    def __init__(self, net, net_num, learning_rate=0.05, training_interval=10, batch_size=128, memory_size=1024,
                 output_graph=False):
        self.net = net
        self.net_num = net_num
        self.lr = learning_rate
        self.training_interval = training_interval
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory_counter = 1
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        tf.compat.v1.reset_default_graph()
        self._build_net()
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def _build_net(self):
        def build_layers(h, c_names, net, w_initializer, b_initializer):
            with tf.compat.v1.variable_scope('l1'):
                w1 = tf.compat.v1.get_variable('w1', [net[0], net[1]], initializer=w_initializer, collections=c_names)
                b1 = tf.compat.v1.get_variable('b1', [1, net[1]], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(h, w1) + b1)
            with tf.compat.v1.variable_scope('l2'):
                w2 = tf.compat.v1.get_variable('w2', [net[1], net[2]], initializer=w_initializer, collections=c_names)
                b2 = tf.compat.v1.get_variable('b2', [1, net[2]], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
            with tf.compat.v1.variable_scope('M'):
                w3 = tf.compat.v1.get_variable('w3', [net[2], net[3]], initializer=w_initializer, collections=c_names)
                b3 = tf.compat.v1.get_variable('b3', [1, net[3]], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l2, w3) + b3
            return out

        self.h = tf.compat.v1.placeholder(tf.float32, [None, self.net[0]], name='h')
        self.m = tf.compat.v1.placeholder(tf.float32, [None, self.net[-1]], name='mode')
        self.is_train = tf.compat.v1.placeholder("bool")
        self.m_pred = []
        self.loss = []
        self.train_op = []

        for i in range(self.net_num):
            with tf.compat.v1.variable_scope(f'memory{i}_net'):
                w_initializer, b_initializer = tf.random_normal_initializer(0.,
                                                                            1 / self.net[0]), tf.constant_initializer(0)
                self.m_pred.append(
                    build_layers(self.h, ['memory_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES], self.net,
                                 w_initializer, b_initializer))
            with tf.compat.v1.variable_scope(f'loss{i}'):
                self.loss.append(
                    tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.m, logits=self.m_pred[i])))
            with tf.compat.v1.variable_scope(f'train{i}'):
                self.train_op.append(tf.compat.v1.train.AdamOptimizer(self.lr, 0.09).minimize(self.loss[i]))

    def remember(self, h, m):
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))
        self.memory_counter += 1

    def encode(self, h, m):
        self.remember(h, m)
        if self.memory_counter >= 512 and self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        sample_index = []
        batch_memory = []
        h_train = []
        m_train = []

        if self.memory_counter > self.memory_size:
            for j in range(self.net_num):
                sample_index.append(np.random.choice(self.memory_size, size=self.batch_size))
        else:
            for j in range(self.net_num):
                sample_index.append(np.random.choice(self.memory_counter, size=self.batch_size))

        for j in range(self.net_num):
            batch_memory.append(self.memory[sample_index[j], :])
            h_train.append(batch_memory[j][:, 0: self.net[0]])
            m_train.append(batch_memory[j][:, self.net[0]:])
            _, cost = self.sess.run([self.train_op[j], self.loss[j]],
                                    feed_dict={self.h: h_train[j], self.m: m_train[j]})

    def decode(self, h):
        m_list = []
        h = h[np.newaxis, :]
        for k in range(self.net_num):
            m_pred = self.sess.run(self.m_pred[k], feed_dict={self.h: h})
            m_list.append(1 * (m_pred[0] > 0))
        return m_list