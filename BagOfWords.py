import numpy as np

from Utils import numbe, test, tokenize
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

class BagOfWords:

    def __init__(self):
        self.vocab = {}
        self.words = []

    def build_vocabulary(self, train_data):
        print("traindata: ", train_data)
        for paragraph in train_data:
            print("paragraph : ", paragraph)
            for word in tokenize(paragraph):
                print("word: ", word)
                if word not in self.vocab:
                    self.vocab[word] = len(self.words)
                    self.words.append(word)
        return len(self.words)

    def get_features(self, data, number_sentences):
        result = np.zeros((number_sentences, len(self.words)))
        for idx, sentence in enumerate(data):
            print("idx: ", idx)
            print("sentence: ", sentence)
            for word in tokenize(sentence):
                print("word2: ", word)
                if word in self.vocab:
                    result[idx, self.vocab[word]] += 1
        return result

number_sentences, training_data = numbe('train_samples2.txt')
number_sentences2, testing_data = numbe('test_samples2.txt')

number_sentences_test, index, training_lables = test('train_labels2.txt')
# number_sentences_test, index_test, test_labls = test('validation_target_labels.txt')
print(tokenize(training_data[0]))

count_vectorizer = CountVectorizer()
X = count_vectorizer.fit_transform(training_data)
df_bow_sklearn = pd.DataFrame(X.toarray(), columns=count_vectorizer.get_feature_names())
df_bow_sklearn.head()
# print("vectorizer: ", count_vectorizer.get_feature_names())
print(df_bow_sklearn)

bow_model = BagOfWords()
bow_model.build_vocabulary(training_data)
print("bow1: ", bow_model.build_vocabulary(training_data))
print("bow vocab: ", bow_model.vocab)
print("bow words: ", bow_model.words)
training_features = bow_model.get_features(training_data, number_sentences)
final_tokenized = ""
for ceva in training_data:
    final_tokenized += ceva
df_bow_model = pd.DataFrame(training_features,columns=tokenize(final_tokenized))
df_bow_model.head()
print("tf: ", training_features)
# print(len(training_features[0]))
# print(len(training_features))
# print("traingin features: ", training_features)
# testing_features = bow_model.get_features(testing_data, number_sentences)
# print("testing features", testing_features)
#
# print(training_features[0])