import re

def word_extraction(sentence):
    """
        Function extracts words of a text and removes special characters and numbers.
        Return a list of extracted words.
        :param sentence: string
        :return: cleaned_text - List<string>
    """
    words = re.sub("[^\w]", " ",  sentence).split()
    cleaned_text = []
    for w in words:
            if (w.isnumeric()) == False:
                if len(w) > 1:
                    if w.isalpha():
                        cleaned_text.append(w.lower())

    return cleaned_text

def tokenize(sentences):
    """
        Function extracts words of a text and removes duplicates and one letter words.
        Return a list of words alphabetically sorted.
        :param sentences: string
        :return: words - List<string>
    """
    words = []
    wi = word_extraction(sentences)
    words.extend(wi)
    words = sorted(list(set(words)))
    return words


str = "marian george nu prea stie stie python. Www  weee w w w www  wd ew  w wwfdwdfww."
str2 =  "marian stie george 112 nu prea  stie @ //  $#python ## %%."
wordss = tokenize(str)
print("tokenize: ", wordss)
cleaned_text = word_extraction(str2)
print("word_extaction: ", cleaned_text)

def numbe (file):
    """
        Function reads a file and count number of sentences.
        Return a list of sentences and number of sentences.
        :param file: string - File name
        :return: number_sentences - Integer
                 text - List<string>
    """
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
number_sentences, training_data = numbe('train_samples2.txt')
number_sentences2, testing_data = numbe('test_samples.txt')
print("numbe1 sent: ", number_sentences)
print("numbe1 trainingdata: ", training_data)
print("numbe2 sent: ", number_sentences2)
print("numbe2 testinggdata: ", testing_data)


def test(file):
    """
        Function reads a file, count number of sentences and extract ids of paragraphs and language codes.
        Return number of sentences, ids and codes extracted.
        :param file: string - File name
        :return: number_sentences - Integer
                 ids - List<string> - ids of paragraphs
                 language_codes - List<string>
    """
    number_sentences = 0
    language_codes = []
    ids = []

    with open(file, 'r+') as file:
       for sentences in file:
            i=sentences.split(None, 1)
            idx = i[0]
            ids.append(idx)
            index = int(i[1])
            number_sentences += 1
            language_codes.append(index)
    return number_sentences, ids, language_codes


number_sentences_test, index, training_lables = test('train_labels.txt')
print("numsent: ", number_sentences_test)
print("index: ", index)
print("traininglabels: ", training_lables)
number_sentences_test, index_test, test_labls = test('validation_target_labels.txt')

