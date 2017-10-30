import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pickle
import csv
import sys
import numpy as np
import pandas as pd
from collections import OrderedDict

reload(sys)
sys.setdefaultencoding('latin-1')

org_train_file = 'training.1600000.processed.noemoticon.csv'
org_test_file = 'testdata.manual.2009.06.14.csv'

def usefull_filed(org_file, output_file):
    output = open(output_file, 'w')
    with open(org_file, 'rb', buffering=1000) as f:
        reader = csv.reader(f)
        for row in reader:
            #print row
            clf = row[0]
            if clf == '0':
                clf = [0, 0, 1]
            elif clf == '2':
                clf = [0, 1, 0]
            elif clf == '4':
                clf = [1, 0, 0]

            outputline = str(clf) + '|-|' + row[-1] + '\n'
            output.write(outputline)
    output.close()
            
usefull_filed(org_train_file, 'training.csv')
print('Training data extracted!')
usefull_filed(org_test_file, 'test.csv')
print('Test data extracted!')

def create_lexicon(train_file):
    lex = []
    lemmatizer = WordNetLemmatizer()
    with open(train_file, buffering=10000) as f:
        try:
            reader = csv.reader(f)
            count_word = {}
            for row in reader:
                row[-1] = row[-1].decode('latin-1')
                #print row
                words = word_tokenize(row[-1].lower())
                for word in words:
                    word = lemmatizer.lemmatize(word)
                    if word not in count_word:
                        count_word[word] = 1
                    else:
                        count_word[word] += 1
        
            count_word = OrderedDict(sorted(count_word.items(), key=lambda t: t[1]))
            for word in count_word:
                if count_word[word] <800000 and count_word[word] > 10:
                    lex.append(word)
        except Exception as e:
            print(e)
    return lex

lex = create_lexicon('training.1600000.processed.noemoticon.csv')
print('Lexcion data extracted!')

with open('lexcion.pickle', 'wb') as f:
    pickle.dump(lex, f)

#f = open('lexcion.pickle', 'rb')
#lex = pickle.load(f)
#f.close()

def data_processing(inputfile, outputfile):
    output = open(outputfile, 'w')
    print len(lex)
    with open(inputfile, 'rb', buffering=1000) as f:
        lemmatizer = WordNetLemmatizer()
        i = 0
        for line in f.readlines():
            label, tweet = line.split('|-|')
            words = word_tokenize(tweet.lower())
            words = [lemmatizer.lemmatize(word) for word in words]

            features = np.zeros(len(lex))
            for word in words:
                if word in lex:
                    features[lex.index(word)] = 1

            #print(features)
            outputline = label + '|||' + str(list(features)) + '\n'
            output.write(outputline)
            if i % 1000 == 0:
                print('Data processing: ', i/1600000)
            i = i + 1
    output.close

#data_processing('training.csv', 'train_data.csv')
'''
#test output csv
f = open('training.csv', 'rb')
reader = csv.reader(f)
i = 0
for row in reader:
    if i<30:
        print row[0] + '\n' + row[1]
    i = i + 1
f.close
'''    
