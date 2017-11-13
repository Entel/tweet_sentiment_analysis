import os
import csv
import random
import tensorflow as tf
import pickle
import numpy as np
import os.path
import sys
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

reload(sys)
sys.setdefaultencoding('latin-1')

f = open('lexcion.pickle', 'rb')
lex = pickle.load(f)
f.close()

n_input_leyer = len(lex)
n_layer_1 = 2000
n_layer_2 = 2000
n_output_layer = 3

def get_random_line(afile):
    return random.choice(open(afile).readlines())

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
            test_x.append(list(features))
            if row[0] == '0':
                test_y.append([0, 0, 1])
            elif row[0] == '2':
                test_y.append([0, 1, 0])
            elif row[0] == '4':
                test_y.append([1, 0, 0])
    return test_x, test_y

test_x, test_y = get_test_data('testdata.manual.2009.06.14.csv')
#test_x, test_y = [], []
print('Loaded test file.')

'''
print get_n_random_lines('training.csv', 10)
l = get_random_line('training.csv')
print l
a,b = l.split('",')
a = a.replace('"', "")
print a, b
'''

def neural_network(data):
    layer_1_wb = {'w_':tf.Variable(tf.random_normal([n_input_leyer, n_layer_1])), 'b_':tf.Variable(tf.random_normal([n_layer_1]))}
    layer_2_wb = {'w_':tf.Variable(tf.random_normal([n_layer_1, n_layer_2])), 'b_':tf.Variable(tf.random_normal([n_layer_2]))}
    layer_output_wb = {'w_':tf.Variable(tf.random_normal([n_layer_2, n_output_layer])), 'b_':tf.Variable(tf.random_normal([n_output_layer]))}

    layer_1 = tf.add(tf.matmul(data, layer_1_wb['w_']), layer_1_wb['b_'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, layer_2_wb['w_']), layer_2_wb['b_'])
    layer_2 = tf.nn.relu(layer_2)
    layer_output = tf.add(tf.matmul(layer_2, layer_output_wb['w_']), layer_output_wb['b_'])

    return layer_output

X = tf.placeholder('float')
Y = tf.placeholder('float')
batch_size = 100

def train_neural_network(X, Y):
    predict = neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
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

            #if os.path.isfile('model.ckpt'):
            #    saver.restored(session, 'model.ckpt')

            try:
                lines = get_n_random_lines('training.csv', batch_size) 
                for line in lines:
                    label, tweet = line.split('|-|')
                    words = word_tokenize(tweet.lower())
                    words = [lemmatizer.lemmatize(word) for word in words]

                    features = np.zeros(len(lex))
                    for word in words:
                        if word in lex:
                            features[lex.index(word)] = 1

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
                saver.save(session, './feedforward_model.ckpt')
            print('Time: ', i, ', Accuracy: ', accuracy, ' - ', pre_accuracy)
            i = i + 1

train_neural_network(X, Y)
            

