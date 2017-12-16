import tensorflow as tf
import itertools as it
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt


class MnistData():
    """
    Arrange MNIST data set to suit siamese networking training
    """
    def __init__(self):
        try:
            with open('mnist.pkl', 'rb') as file:
                mnist = pkl.load(file)
        except IOError:
            from tensorflow.examples.tutorials.mnist import input_data
            mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
            with open('mnist.pkl', 'wb') as file:
                pkl.dump(mnist, file)

        self.train_images = mnist.train.images.reshape((-1, 28, 28, 1))
        self.train_labels = mnist.train.labels
        self.test_images = mnist.test.images.reshape((-1, 28, 28, 1))
        self.test_labels = mnist.test.labels

        self.train_amt = self.train_images.shape[0]
        self.test_amt = self.test_images.shape[0]

        self.images_in_label = [[] for i in range(10)]

        for i in range(self.train_amt):
            self.images_in_label[self.train_labels[i]].append(i)


    def get_siamese_train_batch(self):
        left_images = []
        right_images = []
        labels = []
        for i in range(10):
            n = 45
            l = np.random.choice(self.images_in_label[i], n * 2, replace=False).tolist()
            for j in range(n):
                left_images.append(self.train_images[l.pop(), :, :, :])
                right_images.append(self.train_images[l.pop(), :, :, :])
                labels.append([1])

        # impostor
        for i, j in it.combinations(range(10), 2):
            l = [np.random.choice(self.images_in_label[i]), np.random.choice(self.images_in_label[j])]
            left_images.append(self.train_images[l.pop(), :, :, :])
            right_images.append(self.train_images[l.pop(), :, :, :])
            labels.append([0])
        return left_images, right_images, np.array(labels)

    def get_test(self):
        return self.test_images, self.test_labels

class Network:
    """
    Build a convolutional network
    """

    def __init__(self, width, height, nch, target = 2):
        """

        :param width: width of input images
        :param height: height of input images
        :param nch: number of channels of input images
        :param target: target dimensionality
        """
        self.target = target
        self.width = self.cur_width = width
        self.height = self.cur_height = height
        self.nch = self.cur_nch = nch
        self.locked = False
        self.layers = []

    # Some utilities
    def weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool(self, x, pool_width, pool_height):
        return tf.nn.max_pool(x, ksize=[1, pool_height, pool_width, 1], strides=[1, pool_height, pool_width, 1], padding='VALID')

    def add_conv(self, conv_width, conv_height, pool_width, pool_height, next_nch):
        """
        Add a convolutional layer to the network
        :param conv_width: width of conv kernel
        :param conv_height: height of conv kernel
        :param pool_width: width of max pooling
        :param pool_height: height of max pooling
        :param next_nch: number of channels of output
        """
        if self.locked:
            raise AssertionError('The network is connected. No more layer can be added.')

        if len(self.layers) > 0 and self.layers[-1]['type'] == 'fc':
            raise AssertionError('The data is flattened. No more conv layer can be added.')

        name = str(len(self.layers)) + '_conv_'
        W = self.weight_variable([conv_height, conv_width, self.cur_nch, next_nch], name + 'W')
        b = self.bias_variable([next_nch], name + 'b');

        self.layers.append({'W': W, 'b': b, 'p': (pool_width, pool_height), 'type': 'conv'})

        self.cur_height //= pool_height
        self.cur_width //= pool_width
        self.cur_nch = next_nch


    def add_fc(self, next_width):
        """
        Add a fully connected layer to the network
        :param next_width: dimensionality of the output
        """
        if self.locked:
            raise AssertionError('The network is connected. No more layer can be added.')

        if self.layers[-1]['type'] == 'conv':
            name = str(len(self.layers)) + '_ffc_'
            type = 'ffc'
            self.cur_width = self.cur_height * self.cur_width * self.cur_nch
            self.cur_height = self.cur_nch = 1
        else:
            name = str(len(self.layers)) + '_fc_'
            type = 'fc'
        W = self.weight_variable([self.cur_width, next_width], name + 'W')
        b = self.bias_variable([next_width], name + 'b')

        self.layers.append({'W': W, 'b': b, 'type': type})

        self.cur_width = next_width

    def connect(self, feed):
        """
        Connect the layers and return the output
        :param feed: input
        :return: output (tensorflow tensor)
        """
        act = tf.nn.tanh

        if not self.locked:
            name = str(len(self.layers)) + '_linear_'
            W = self.weight_variable([self.cur_width, self.target], name + 'W')
            b = self.bias_variable([self.target], name + 'b')
            self.layers.append({'W': W, 'b': b, 'type': 'linear'})
            del self.cur_width, self.cur_height, self.cur_nch

        for i in self.layers:
            if i['type'] == 'conv':
                print(i['W'], i['b'])
                feed = self.max_pool(act(self.conv2d(feed, i['W']) + i['b']), i['p'][0], i['p'][1])
            elif i['type'] == 'ffc':
                feed = act(tf.reshape(feed, [-1, int(i['W'].shape[0])]) @ i['W'] + i['b'])
            elif i['type'] == 'fc':
                feed = act(feed @ i['W'] + i['b'])
            elif i['type'] == 'linear':
                feed = feed @ i['W'] + i['b']
            else:
                raise AssertionError('Unknown layer type.')

        self.locked = True
        return feed


if __name__ == '__main__':
    data = MnistData() # Dataset
    net = Network(28, 28, 1) # Network

    # Add layers to the network
    net.add_conv(5, 5, 2, 2, 32)
    net.add_conv(5, 5, 2, 2, 64)
    net.add_fc(1024) # the network will be flattened automatically
    net.add_fc(256)

    # Build Siamese structure
    left_input = tf.placeholder(tf.float32, [None, 28, 28, 1], name='left_input')
    right_input = tf.placeholder(tf.float32, [None, 28, 28, 1], name='right_input')
    label_input = tf.placeholder(tf.float32, [None, 1], name='label_input')

    left_output = net.connect(left_input)
    right_output = net.connect(right_input)

    # Use CPU do run test (in case graphic memory is insufficient)
    with tf.device('/cpu:0'):
        test_input = tf.placeholder(tf.float32, [None, 28, 28, 1], name='test_input')
        test_output = net.connect(test_input)

    def calc_loss(model1, model2, y, margin):
        with tf.name_scope("contrastive-loss"):
            d = tf.sqrt(tf.reduce_sum(tf.pow(model1 - model2, 2), 1, keep_dims=True))
            tmp = y * tf.square(d)
            tmp2 = (1 - y) * tf.square(tf.maximum((margin - d), 0))
            return tf.reduce_mean(tmp + tmp2) / 2

    total_loss = calc_loss(left_output, right_output, label_input, 0.2)

    optimizer = tf.train.MomentumOptimizer(0.01, 0.99, use_nesterov=True).minimize(total_loss)

    def plot(output, labels, num):
        for j in range(10):
            plt.scatter(output[labels == j, 0], output[labels == j, 1], 5)
        plt.legend([str(i) for i in range(10)])
        plt.savefig('fig' + str(num) + '.png')
        plt.close()

    rec = [];

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        images, labels = data.get_test()
        output = sess.run(test_output, feed_dict={test_input: images})
        plot(output, labels, 0)

        for i in range(10000):
            left_images, right_images, labels = data.get_siamese_train_batch()
            if (i + 1) % 100 == 0:
                _, loss = sess.run([optimizer, total_loss],
                                   feed_dict={left_input: left_images, right_input: right_images,
                                              label_input: labels})
                print(i + 1, ': ', loss)
            else:
                sess.run(optimizer,
                         feed_dict={left_input: left_images, right_input: right_images, label_input: labels})

            if (i + 1) % 200 == 0:
                images, labels = data.get_test()
                output = sess.run(test_output, feed_dict={test_input: images})
                plot(output, labels, i + 1)
                rec.append(output)

        with open('triplet_output.pkl') as file:
            pkl.dump([labels, rec], file);