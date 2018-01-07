import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import itertools as it
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
#'aarch64'==platform.machine()
max_step = 10000
test_itv = 200
plot_itv = 500
test_batch_size = 10
class MnistData:
    def __init__(self):
        try:
            with open('mnist.pkl', 'rb') as file:
                mnist = pkl.load(file)
        except IOError:
            mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
            with open('mnist.pkl', 'wb') as file:
                pkl.dump(mnist, file)
        ## 55000 training images, each image is flatten as a 784-d vector
        self.train_images = mnist.train.images.reshape((-1, 28, 28, 1)) # np.reshape train_images to (55000, 28, 28, 1)
        self.train_labels = mnist.train.labels
        ## 10000 testing images, each image is flatten as a 784-d vector
        self.test_images = mnist.test.images.reshape((-1, 28, 28, 1))   # np.reshape test_images to (10000, 28, 28, 1) 
        self.test_labels = mnist.test.labels

        self.train_amt = self.train_images.shape[0]
        self.test_amt = self.test_images.shape[0]

        self.images_in_label = [[] for i in range(10)]

        for i in range(self.train_amt):
            self.images_in_label[self.train_labels[i]].append(i)


    def get_train_batch(self):
        left = []
        right = []
        similarity = []
        for i in range(10):
            n = 45
            l = np.random.choice(self.images_in_label[i], n * 2, replace=False).tolist()
            left.append(self.train_images[l.pop(), :, :, :])
            right.append(self.train_images[l.pop(), :, :, :])
            similarity.append([1])

        # impostor
        for i, j in it.combinations(range(10), 2):
            l = [np.random.choice(self.images_in_label[i]), np.random.choice(self.images_in_label[j])]
            left.append(self.train_images[l.pop(), :, :, :])
            right.append(self.train_images[l.pop(), :, :, :])
            similarity.append([0])
        return np.array(left), np.array(right), np.array(similarity)

    def get_test(self):
        return np.array(self.test_images), np.array(self.test_labels)


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# shared parameters
W_conv1 = weight_variable([5, 5, 1, 32], 'Wc1')  # height, width, input channel, output channel
b_conv1 = bias_variable([32], 'bc1')  # one bias for a convolutional branch, why?
W_conv2 = weight_variable([5, 5, 32, 64], 'Wc2')  # height, width, input channel, output channel
b_conv2 = bias_variable([64], 'bc2')  # one bias for a convolutional branch
W_fc1 = weight_variable([7 * 7 * 64, 1024], 'Wf1') # 28 / 2 / 2 = 7, 64 channels, 1024
b_fc1 = bias_variable([1024], 'bf1')               # each entry has its bias
W_fc2 = weight_variable([1024, 256], 'Wf2')
b_fc2 = bias_variable([256], 'bf2')
W_fc3 = weight_variable([256, 2], 'Wf3')
b_fc3 = bias_variable([2], 'bf3')

def build_network(x_image):
    # 1st convolutional layer
    h_conv1 = tf.nn.tanh(conv2d(x_image, W_conv1) + b_conv1) # How does dimension of b_conv1 extended?
    h_pool1 = max_pool_2x2(h_conv1)

    # 2nd convolutional layer
    h_conv2 = tf.nn.tanh(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # densely connected layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64]) # keep first dimension unchanged, other dimensions reshape into one
    h_fc1 = tf.nn.tanh(h_pool2_flat @ W_fc1 + b_fc1)

    h_fc2 = tf.nn.tanh(h_fc1 @ W_fc2 + b_fc2)

    h_fc3 = h_fc2 @ W_fc3 + b_fc3

    # Readout
    return h_fc3

left_input = tf.placeholder(tf.float32, [None, 28, 28, 1], name='left_input')
right_input = tf.placeholder(tf.float32, [None, 28, 28, 1], name='right_input')
label_input = tf.placeholder(tf.float32, [None, 1], name='label_input')

left_output = build_network(left_input)
right_output = build_network(right_input)


test_input = tf.placeholder(tf.float32, [None, 28, 28, 1], name='test_input')
test_output = build_network(test_input)

def calc_loss(model1, model2, y, margin):
    with tf.name_scope("contrastive-loss"):
        d = tf.sqrt(tf.reduce_sum(tf.pow(model1-model2, 2), 1, keep_dims=True))
        tmp= y * tf.square(d)
        tmp2 = (1 - y) * tf.square(tf.maximum((margin - d),0))
        return tf.reduce_mean(tmp + tmp2) /2

margin = 0.2
total_loss = calc_loss(left_output, right_output, label_input, margin)

optimizer = tf.train.AdamOptimizer(0.001).minimize(total_loss)

data = MnistData()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    images, labels = data.get_test()
    chunks_img = [images[x:x+test_batch_size] for x in range(0, len(images), test_batch_size)]
    chunks_lbl = [labels[x:x+test_batch_size] for x in range(0, len(labels), test_batch_size)]
#    for group in range(len(chunks_img)):    
    for group in range(1):
        output = sess.run(test_output, feed_dict={test_input: chunks_img[group]})
        for j in range(10):
            plt.scatter(output[chunks_lbl[group] == j, 0], output[chunks_lbl[group] == j, 1], 5)
    plt.legend([str(i) for i in range(10)])
    plt.savefig('fig0.png')
    plt.close()


    for i in range(max_step):
        left_images, right_images, labels = data.get_train_batch()
        if (i + 1) % test_itv == 0:
            _, loss = sess.run([optimizer, total_loss], feed_dict={left_input: left_images, right_input: right_images, label_input: labels})
            print(i+1, ': ', loss)
        else:
            sess.run(optimizer, feed_dict={left_input: left_images, right_input: right_images, label_input: labels})

        if (i + 1) % plot_itv == 0:
            for group in range(len(chunks_img)):
                output = sess.run(test_output, feed_dict={test_input: chunks_img[group]})
                for j in range(10):
                    plt.scatter(output[chunks_lbl[group] == j, 0], output[chunks_lbl[group] == j, 1], 5)
            plt.legend([str(idx) for idx in range(10)])
            plt.savefig('fig' + str(i + 1) + '.png')
            plt.close()

