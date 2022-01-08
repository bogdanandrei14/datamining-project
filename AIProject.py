import numpy as np
import re
from sklearn import preprocessing
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import csv


def tokenize(sentences):
    words = []
    wi = word_extraction(sentences)
    words.extend(wi)
    words = sorted(list(set(words)))
    return words


def word_extraction(sentence):
    words = re.sub("[^\w]", " ",  sentence).split()
    cleaned_text = []
    for w in words:
            if (w.isnumeric()) == False:
                if len(w) > 1:
                    if w.isalpha():
                        cleaned_text.append(w.lower())

    return cleaned_text




vocab = []
number_sentences = 0

def numbe (file):
    number_sentences = 0
    text = []
    fil = open(file, "r",  encoding="utf-8")
    for sentences in fil:
        number_sentences += 1
        text.append(sentences)

    return number_sentences, text
file = open('train_samples.txt', "r")
file2 = open('test_samples.txt', "r")
file3 = open('train_labels.txt', "r")
number_sentences, training_data = numbe('train_samples.txt')
number_sentences2, testing_data = numbe('test_samples.txt')

def test(file):
    number_sentences = 0
    test = []
    index1=[]

    with open(file, 'r+') as file:
       for sentences in file:
            i=sentences.split(None, 1)
            idx = i[0]

            index1.append(idx)
            index = int(i[1])
            number_sentences += 1
            test.append(index)
    return number_sentences, index1, test


number_sentences_test, index, training_lables = test('train_labels.txt')
number_sentences_test, index_test, test_labls = test('validation_target_labels.txt')


class BagOfWords:

    def __init__(self):
        self.vocab = {}
        self.words = []

    def build_vocabulary(self, train_data):
        for sentence in train_data:
            for word in sentence:
                if word not in self.vocab:
                    self.vocab[word] = len(self.words)
                    self.words.append(word)
        return len(self.words)

    def get_features(self, data):
        result = np.zeros((number_sentences, len(self.words)))
        for idx, sentence in enumerate(data):
            for word in sentence:
                if word in self.vocab:
                    result[idx, self.vocab[word]] += 1
        return result

bow_model = BagOfWords()
bow_model.build_vocabulary(training_data)
training_features = bow_model.get_features(training_data)
testing_features = bow_model.get_features(testing_data)

print(training_features[0])


def normalize_data(train_data, test_data, type=None):
    scaler = None
    if type == 'standard':
        scaler = preprocessing.StandardScaler()

    elif type == 'min_max':
        scaler = preprocessing.MinMaxScaler()

    elif type == 'l1' or type == 'l2':
        scaler = preprocessing.Normalizer(norm=type)

    if scaler is not None:
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        scaled_test_data = scaler.transform(test_data)
        return scaled_train_data, scaled_test_data
    else:
        return train_data, test_data

norm_train, norm_test = normalize_data(training_features, testing_features, type='l2')
'''
model = svm.SVC(C=1.0, kernel='linear')

model.fit(norm_train, training_lables)
test_preds = model.predict(norm_test)
'''


nb = MultinomialNB()
nb.fit(norm_train, training_lables)
test_preds = nb.predict(norm_test)

row_list = []
number = 0

with open('sample_submission.csv', mode='w') as employee_file:
    #capul de tabel
    row_list.append("id")
    row_list.append("label")
    employee_writer = csv.writer(employee_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(row_list)
    row_list.clear()

    for i in range(len(test_labls)):
        row_list.clear()
        row_list.append(index_test[i])
        row_list.append(test_preds[i])
        employee_writer = csv.writer(employee_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        employee_writer.writerow(row_list)

confusion = confusion_matrix(training_lables, test_preds)
print("Confusion matrix:\n{}".format(confusion))
precision = precision_score(training_lables, test_preds)
recall = recall_score(training_lables, test_preds)
f1 = 0.0
f1 = f1_score(training_lables, test_preds)
print("F1: {}".format(f1))
print('Precision: ', precision)
print('Recall: ', recall)
print(training_lables[0])




