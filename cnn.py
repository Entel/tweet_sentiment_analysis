import os
import random
import tensorflow as tf
import pickle
import csv
import numpy as np
import os.path
import sys
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

reload(sys)
sys.setdefaultencoding('latin-1')

with open('lexcion.pickle', 'rb') as f:
    lex = pickle.load(f)
print lex
lex_length = len(lex)
append_length = 40000 - len(lex)

def get_n_random_lines(afile, n):
    with open(afile) as f:
        flines = f.readlines()
        lines = []
        for i in range(0, n):
            lines.append(random.choice(flines))
    return lines

def get_test_data(tfile):
    with open(tfile, 'rb') as f:
        test_x = []
        test_y = []
        lemmatizer = WordNetLemmatizer()
        reader = csv.reader(f)
        for row in reader:
            words = word_tokenize(row[-1].lower())
            words = [lemmatizer.lemmatize(word) for word in words]
            features = np.zeros(len(lex))
            for word in words:
                if word in lex:
                    features[lex.index(word)] = 1
            np.append(features, np.zeros(append_length))
            test_x.append(list(features))
            if row[0] == '0':
                test_y.append([0, 0, 1])
            elif row[0] == '2':
                test_y.append([0, 1, 0])
            elif row[0] == '4':
                test_y.append([1, 0, 0])
    return test_x, test_y

#test_x, test_y = get_test_data('testdata.manual.2009.06.14.csv')
#test_x, test_y = [], []
#print('Loaded test file.')

n_output_layer = 3

def convolution_neural_network(X):
    input_layer = tf.reshape(X, [-1, 200, 200, 1])
    
    '''
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    '''

    weight = {'w_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            'w_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            'w_fc': tf.Variable(tf.random_normal([50*50*64, 1024])),
            'out': tf.Variable(tf.random_normal([1024, n_output_layer]))}
    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
            'b_conv2': tf.Variable(tf.random_normal([64])),
            'b_fc': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([n_output_layer]))}
    
    conv1 = tf.nn.relu(tf.add(tf.nn.conv2d(input_layer, weight['w_conv1'], strides=[1, 1, 1, 1], padding='SAME'), biases['b_conv1']))
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.relu(tf.add(tf.nn.conv2d(pool1, weight['w_conv2'], strides=[1, 1, 1, 1], padding='SAME'), biases['b_conv2']))
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    fc = tf.reshape(pool2, [-1, 50*50*64])
    fc = tf.nn.relu(tf.add(tf.matmul(fc, weight['w_fc']), biases['b_fc']))

    output = tf.add(tf.matmul(fc, weight['out']), biases['out'])
    return output

X = tf.placeholder('float', [None, 40000])
Y = tf.placeholder('float')
batch_size = 100

def train_neural_network(X, Y):
    predict = convolution_neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)
    
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        lemmatizer = WordNetLemmatizer()
        saver = tf.train.Saver()
        i = 0
        pre_accuracy = 0
        while True:
            batch_x = []
            batch_y = []

            try:
                lines = get_n_random_lines('training.csv', batch_size)
                for line in lines:
                    label, tweet = line.split('|-|')
                    words = word_tokenize(tweet.lower())
                    words = [lemmatizer.lemmatize(word) for word in words]
                    
                    features = np.zeros(lex_length)
                    for word in words:
                        if word in lex:
                            features[lex.index(word)] = 1
                    features = np.append(features, np.zeros(append_length))
                    #print(len(np.zeros(append_length)), len(features))
                    
                    batch_x.append(list(features))
                    batch_y.append(eval(label))

                session.run([optimizer, cost_func], feed_dict={X: batch_x, Y: batch_y})

            except Exception as e:
                print(e)

            correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            accuracy = accuracy.eval({X:test_x, Y:test_y})
            if accuracy > pre_accuracy:
                pre_accuracy = accuracy
                saver.save(session, 'cnn.ckpt')
            print('Time: ', i, 'Accuracy: ', accuracy)
            i = i + 1

#train_neural_network(X, Y)
