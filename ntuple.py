import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import itertools as it
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
import math
num_label = 10
max_step =1000
div = 2
d = 10
number_group = math.ceil(float(500)/d)
class MnistData():
    """
    Arrange MNIST data set to suit siamese networking training
    """
    def __init__(self):
        try:
            with open('mnist.pkl', 'rb') as file:
                mnist = pkl.load(file)
        except IOError:
            
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
            left_images.append(self.train_images[l.pop(), :, :, :])
            right_images.append(self.train_images[l.pop(), :, :, :])
            labels.append([1])

        # impostor
        for i, j in it.combinations(range(10), 2):
            l = [np.random.choice(self.images_in_label[i]), np.random.choice(self.images_in_label[j])]
            left_images.append(self.train_images[l.pop(), :, :, :])
            right_images.append(self.train_images[l.pop(), :, :, :])
            labels.append([0])
        return left_images, right_images, labels
    
    def return_train_images(self, i):
        return self.train_images[i, :, :, :]

    def get_triplet_train_batch(self):
        '''
            Generate 450 pairs of triplets
        '''
        mid_images = []
        pos_images = []
        neg_images = []
        n = 5
        for i in range(10):
            for j in range(10):
                if j != i:
                    l_yes = np.random.choice(self.images_in_label[i], n * 2, replace=False).tolist() # 10 sample indices
                    # print(len(l_yes), 'len l_yes')
                    l_no = np.random.choice(self.images_in_label[j], n, replace=False).tolist() # 5 samples indices
                    # print(l_no, 'len l_no')

                    for k in range(n):
                        mid_images.append(self.train_images[l_yes.pop(), :, :, :])
                        pos_images.append(self.train_images[l_yes.pop(), :, :, :])
                        neg_images.append(self.train_images[l_no.pop(), :, :, :])
        return mid_images, pos_images, neg_images
    '''
    def get_tuple_train_batch(self, NUM=d):

        samples= []
        data = []
        assert NUM < num_label, 'NUM has to be smaller than number of labels'
        
        # number_group = math.ceil(float(500)/(NUM+1))
        # number_group = 3 #DEBUG
        for group in range(number_group):
            label = np.random.permutation(range(num_label))[0:NUM]
            
            samples = [np.random.choice(self.images_in_label[label[i]], 2, replace=False).tolist() for i in range(NUM)]
            #  print(samples, 'samples')
            new_data = [[samples[i][np.random.choice([0,1])] for i in range(NUM)] for j in range(NUM)]
            for i in range(NUM) : 
                del new_data[i][i]
                new_data[i] = samples[i]+new_data[i]
   
            data = data+list(map(self.return_train_images, new_data))

        return np.array(data)
    '''
    def get_pair_train_batch(self, NUM=d):
        '''
            Generate  pairs of NUM+1 tuplets
        '''
        samples= []
        data = []
        assert NUM < num_label, 'NUM has to be smaller than number of labels'
        

        # number_group = 3 #DEBUG
        piv = []
        pos = [] 
        for group in range(number_group):
            labels = np.random.permutation(range(num_label))[0:NUM]
            # piv.append([]); pos.append([])
            for lbl in labels:
                imgs = np.random.choice(self.images_in_label[lbl], 2, replace=False).tolist()
                piv.append(self.train_images[imgs[0],:,:,:])
                pos.append(self.train_images[imgs[1],:,:,:])

        return piv, pos, d

        
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
    # data.get_pair_train_batch()
    # Add layers to the network
    net.add_conv(5, 5, 2, 2, 32)
    net.add_conv(5, 5, 2, 2, 64)
    net.add_fc(1024) # the network will be flattened automatically
    net.add_fc(256)

    # cut distance for negative pairs
    cut = tf.Variable(tf.constant(2.0), name= 'cut')
    # Build Siamese structure, will use GPU to calculate when possible
    piv_input = tf.placeholder(tf.float32, [None, 28, 28, 1], name='piv_input')
    pos_input = tf.placeholder(tf.float32, [None, 28, 28, 1], name='pos_input')
    # neg_input = tf.placeholder(tf.float32, [None, 28, 28, 1], name='neg_input')

    piv_output = net.connect(piv_input)
    pos_output = net.connect(pos_input)

    # neg_output = net.connect(neg_input)
    with tf.device('/cpu:0'):
        test_input = tf.placeholder(tf.float32, [None, 28, 28, 1], name='test_input')
        test_output = net.connect(test_input)  
    # print(tf.shape(piv_output)[0])
    loss = []
    
    for idxGp in range(number_group):
        for idx in range(d):
            i = d*idxGp+idx
            penalty = piv_output[i]*pos_output[i]-tf.exp(cut)
            # print(penalty)
            exponents =[piv_output[i]*pos_output[j]-penalty for j in range(d)]
            tempLog =tf.log(tf.reduce_sum(tf.exp(exponents)))
            loss.append(tempLog)
    NORM = tf.norm(piv_output,axis=1, keep_dims=True) + tf.norm(pos_output,axis=1, keep_dims=True)
    total_loss = tf.reduce_mean(loss)+tf.reduce_mean(NORM)**2*0.3
    #total_loss = tf.reduce_mean(loss)+tf.reduce_mean(tf.exp(NORM-1))**2*0.01

    #optimizer = tf.train.MomentumOptimizer(0.01, 0.99, use_nesterov=True).minimize(total_loss)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(total_loss)

    def plot(output, labels, num):
        for j in range(10):
            plt.scatter(output[labels == j, 0], output[labels == j, 1], 5)
        plt.legend([str(i) for i in range(10)])
        plt.savefig('fig' + str(num) + '.png')
        plt.close()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        images, labels = data.get_test()
        output = sess.run(test_output, feed_dict={test_input: images})
        plot(output, labels, 0)

        for i in range(max_step):
            # mid_images, pos_images, neg_images = data.get_triplet_train_batch()
            # test_batch = data.get_tuple_train_batch(NUM=2)
            piv_images, pos_images, _ = data.get_pair_train_batch()

            ## DEBUG
            '''
            print(type(mid_images))
            print(np.array(mid_images).shape) 
            print(np.array(test_batch).shape)           
            '''
            ## DEBUG
            if (i + 1) % div == 0:
                #W0, b0 = sess.run([net.layers[0]['W'], net.layers[0]['b']])
                #W1, b1 = sess.run([net.layers[1]['W'], net.layers[1]['b']])
                #W2, b2 = sess.run([net.layers[2]['W'], net.layers[2]['b']])
                #W2, b2 = sess.run([net.layers[3]['W'], net.layers[3]['b']])
                # _, loss = sess.run([optimizer, total_loss],
                #                    feed_dict={mid_input: mid_images, pos_input: pos_images, neg_input: neg_images})#if np.isnan(loss):
                _, loss = sess.run([optimizer, total_loss],
                                    feed_dict={piv_input: piv_images, pos_input: pos_images})  

                #    print('!')
                print(i + 1, ': ', loss)
            else:
                sess.run(optimizer,
                         feed_dict={piv_input: piv_images, pos_input: pos_images})

            if (i + 1) % 30 == 0:
                images, labels = data.get_test()
                output = sess.run(test_output, feed_dict={test_input: images})
                plot(output, labels, i + 1)
